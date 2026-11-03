//! `tritify-convert <model_dir>` — load a Hugging Face model, detect its family, plan the op graph, build the
//! exact-integer model, verify it against the float reference, print the argmax/mask agreement + reproducing hash,
//! and write a self-contained **KPI HTML card** (`tritify_kpi.html`).
//!
//! Works on decoder-only LMs (Llama/Qwen/Mistral-style), Mixture-of-Experts LMs (OLMoE/Mixtral/Qwen-MoE) and on
//! SAM 2.1. The calibration/probe tokens in this CLI are illustrative ids; a real run feeds tokenized calibration
//! text.

mod kpi;

use kpi::Kpi;
use kt_convert::spec::{convert_spec, ModelSpec, SpecRegistry};
use kt_convert::{convert, ConvertReport, HfModelSource};
use std::fs;
use std::path::Path;
use std::time::Instant;

const USAGE: &str =
    "usage: tritify-convert <model_dir>        convert + verify a checkpoint (dir with config.json + .safetensors)
       tritify-convert --detect <model_dir>  print the checkpoint's model spec (JSON)
       tritify-convert --spec <spec.json>    convert + verify through a model spec
       tritify-convert --list <specs dir>    list the specs in a directory, group → models
       --min-fidelity <v>  the binding decision-fidelity threshold of a conversion (default 0.99)";

/// The default binding threshold on the conversion's decision fidelity.
const MIN_FIDELITY: f64 = 0.99;

fn fail(what: &str, e: String) -> ! {
    eprintln!("{what}: {e}");
    std::process::exit(1)
}

/// A `--min-fidelity` value: a finite number in `[0, 1]`. Anything else is refused with the reason, so a NaN or an
/// infinite threshold can never disable the gate.
fn parse_min_fidelity(s: &str) -> Result<f64, String> {
    let v: f64 = s.trim().parse().map_err(|_| format!("--min-fidelity {s:?} is not a number"))?;
    if !v.is_finite() {
        return Err(format!("--min-fidelity {s:?} is not finite"));
    }
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("--min-fidelity {s:?} is outside [0, 1]"));
    }
    Ok(v)
}

/// The fidelity gate: a measurement that is not finite fails first, then one below the threshold.
fn fidelity_verdict(fidelity: f64, min: f64) -> Result<(), String> {
    if !fidelity.is_finite() {
        return Err(format!("the decision fidelity {fidelity} is not a finite measurement"));
    }
    if fidelity < min {
        return Err(format!("decision fidelity {fidelity:.4} is below the threshold {min}"));
    }
    Ok(())
}

fn main() {
    let mut a: Vec<String> = std::env::args().collect();
    let mut min_fidelity = MIN_FIDELITY;
    if let Some(i) = a.iter().position(|s| s == "--min-fidelity") {
        min_fidelity = a
            .get(i + 1)
            .ok_or_else(|| "--min-fidelity needs a value".to_string())
            .and_then(|s| parse_min_fidelity(s))
            .unwrap_or_else(|e| {
                eprintln!("{e}\n{USAGE}");
                std::process::exit(2)
            });
        a.drain(i..i + 2);
    }
    let arg = |i: usize| -> String {
        a.get(i).cloned().unwrap_or_else(|| {
            eprintln!("{USAGE}");
            std::process::exit(2)
        })
    };
    let t0 = Instant::now();
    let (dir, r): (String, ConvertReport) = match arg(1).as_str() {
        "--detect" => {
            println!("{}", ModelSpec::detect(&arg(2)).unwrap_or_else(|e| fail("detect error", e)).to_json());
            return;
        }
        "--list" => {
            let reg = SpecRegistry::load_dir(arg(2)).unwrap_or_else(|e| fail("spec error", e));
            for (group, ids) in reg.groups() {
                println!("{group:<8} {}", if ids.is_empty() { "—".to_string() } else { ids.join(", ") });
            }
            return;
        }
        "--spec" => {
            let spec = ModelSpec::load(arg(2)).unwrap_or_else(|e| fail("spec error", e));
            println!("tritify-convert");
            println!(
                "  spec       : {} (group {}, {} rename rules, {} config entries)",
                spec.id,
                spec.group,
                spec.weights.len(),
                spec.config.len()
            );
            println!("  model dir  : {}", spec.source.path);
            let r = convert_spec(&spec).unwrap_or_else(|e| fail("convert error", e));
            (spec.source.path, r)
        }
        d if d.starts_with("--") => {
            eprintln!("{USAGE}");
            std::process::exit(2)
        }
        d => {
            let src = HfModelSource::from_dir(d).unwrap_or_else(|e| fail("load error", e));
            println!("tritify-convert");
            println!("  model dir  : {d}");
            println!("  tensors    : {}", src.tensor_names().len());
            (d.to_string(), convert(&src).unwrap_or_else(|e| fail("convert error", e)))
        }
    };
    let convert_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let c = &r.check;
    println!("  family     : {}", r.family);
    println!("  plan       : {}", r.plan_summary);
    println!("  task       : {}", c.task);
    println!(
        "  verify     : {}={:.4}  max_abs_err={:.3e}  deterministic={}",
        c.metric_name, c.metric, c.max_abs_err, c.deterministic
    );
    println!("  detail     : {}", c.detail);
    println!("  repro hash : {}", c.hash);
    if !c.deterministic {
        eprintln!("FAIL: non-deterministic integer path");
        std::process::exit(1);
    }
    let (fidelity, how) = match &r.fidelity {
        Some((v, detail)) => (*v, detail.clone()),
        None => (c.metric, format!("{} (the self-check's)", c.metric_name)),
    };
    println!("  fidelity   : {fidelity:.4}  ({how}; binding threshold {min_fidelity})");
    if let Err(e) = fidelity_verdict(fidelity, min_fidelity) {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }

    // ---- KPI card (measured here so anyone can reproduce it for their own model) ----
    let (n_params, orig_bytes, orig_dtype) = kpi::model_stats(&dir);
    let model = Path::new(&dir).file_name().and_then(|s| s.to_str()).unwrap_or(&dir).to_string();
    let k = Kpi {
        model,
        family: r.family.clone(),
        task: c.task.clone(),
        plan: r.plan_summary.clone(),
        n_params,
        orig_bytes,
        orig_dtype,
        convert_ms,
        peak_rss: kpi::peak_rss_bytes(),
        metric_name: c.metric_name.clone(),
        metric: c.metric,
        max_abs_err: c.max_abs_err,
        deterministic: c.deterministic,
        hash: c.hash.clone(),
    };
    print!("{}", kpi::render_text(&k));
    let out = "tritify_kpi.html";
    match fs::write(out, kpi::render_html(&k)) {
        Ok(_) => println!("  KPI page   : {out}  (self-contained HTML — open it in a browser)"),
        Err(e) => eprintln!("  (could not write {out}: {e})"),
    }
    println!("  → re-run on any machine: the hash is identical. No float. No trick.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_measurement_that_is_not_finite_or_is_below_the_threshold_fails() {
        for f in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let e = fidelity_verdict(f, 0.99).unwrap_err();
            assert!(e.contains("not a finite measurement"), "{f}: {e}");
        }
        assert!(fidelity_verdict(0.98, 0.99).unwrap_err().contains("below the threshold"));
        assert_eq!(fidelity_verdict(1.0, 0.99), Ok(()));
        assert_eq!(fidelity_verdict(0.99, 0.99), Ok(()));
    }

    #[test]
    fn a_threshold_that_is_not_a_finite_number_in_0_1_is_refused() {
        for (s, why) in [
            ("NaN", "not finite"),
            ("nan", "not finite"),
            ("inf", "not finite"),
            ("-inf", "not finite"),
            ("1.5", "outside [0, 1]"),
            ("-0.1", "outside [0, 1]"),
            ("abc", "not a number"),
        ] {
            let e = parse_min_fidelity(s).unwrap_err();
            assert!(e.contains(why), "{s}: {e}");
        }
        assert_eq!(parse_min_fidelity("0"), Ok(0.0));
        assert_eq!(parse_min_fidelity("0.99"), Ok(0.99));
        assert_eq!(parse_min_fidelity("1"), Ok(1.0));
    }
}
