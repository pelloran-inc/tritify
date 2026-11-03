//! Validate the MoE layer on a REAL OLMoE layer: build `MoeMlp` from the checkpoint's router + all E
//! experts (lazy-loaded), run the verify gate on a probe, and report the int-vs-float gap, the top-k routing match,
//! and the reproducing integer hash.
//!   cargo run --release -p kt-convert --example olmoe_moe_check -- <olmoe_dir> [layer=0]

use kt_convert::HfModelSource;
use kt_core::layers::{MoeMlp, QuantLinear, SwiGluMlp, ACT_LEVELS};
use kt_core::plugin::ModelSource;
use kt_core::{verify, IntTensor, Module};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: olmoe_moe_check <olmoe_dir> [layer=0]");
        std::process::exit(2);
    }
    let dir = &a[1];
    let layer: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let src = HfModelSource::from_dir(dir).unwrap_or_else(|e| {
        eprintln!("load: {e}");
        std::process::exit(1);
    });
    let dim = src.config_usize("hidden_size").expect("hidden_size");
    let hidden = src.config_usize("intermediate_size").expect("intermediate_size");
    let n_exp = src.config_usize("num_experts").expect("num_experts");
    let top_k = src.config_usize("num_experts_per_tok").expect("num_experts_per_tok");
    let pfx = format!("model.layers.{layer}");

    let get = |name: String, rows: usize, cols: usize| -> Vec<f32> {
        let (t, shape) = src.tensor(&name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(shape, vec![rows, cols], "{name} shape");
        t
    };

    eprintln!("[olmoe] building MoE layer {layer}: {n_exp} experts (lazy-loaded) …");
    let rw = get(format!("{pfx}.mlp.gate.weight"), n_exp, dim);
    let mut experts = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let g = get(format!("{pfx}.mlp.experts.{e}.gate_proj.weight"), hidden, dim);
        let u = get(format!("{pfx}.mlp.experts.{e}.up_proj.weight"), hidden, dim);
        let d = get(format!("{pfx}.mlp.experts.{e}.down_proj.weight"), dim, hidden);
        experts.push(SwiGluMlp::new(
            format!("e{e}"),
            QuantLinear::from_weights("gate", &g, hidden, dim),
            QuantLinear::from_weights("up", &u, hidden, dim),
            QuantLinear::from_weights("down", &d, dim, hidden),
        ));
    }
    let moe = MoeMlp::new(
        format!("olmoe.l{layer}.moe"),
        QuantLinear::from_weights("router", &rw, n_exp, dim),
        experts,
        top_k,
    );

    // a deterministic probe activation [dim]
    let mut s = 0x5151u64;
    let lcg = |s: &mut u64| {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    };
    let real: Vec<f64> = (0..dim).map(|_| (lcg(&mut s) >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0).collect();
    let probe = IntTensor::from_reals(&real, vec![dim], ACT_LEVELS);

    // integer vs float router top-k (routing-flip check)
    let li = moe.router.forward_int(&probe).data;
    let lf = moe.router.forward_ref(&probe.to_f32());
    let tk = |v: &[f32]| {
        let mut i: Vec<usize> = (0..v.len()).collect();
        i.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
        i.truncate(top_k);
        i
    };
    let ti = tk(&li.iter().map(|&x| x as f32).collect::<Vec<_>>());
    let tf = tk(&lf);

    let rep = verify(&moe, &probe);
    let r = moe.forward_ref(&probe.to_f32());
    let peak = r.iter().fold(0f64, |a, &x| a.max((x as f64).abs()));
    println!("OLMoE REAL layer {layer}: dim={dim} hidden={hidden} E={n_exp} top_k={top_k}");
    println!(
        "  max_abs_err={:.3e}  peak_ref={:.3e}  rel={:.2e}  det={}",
        rep.max_abs_err,
        peak,
        rep.max_abs_err / peak.max(1e-30),
        rep.deterministic
    );
    println!("  router top-k  int={ti:?}");
    println!("  router top-k  float={tf:?}   no_flip={}", ti == tf);
    println!("  reproducing integer output hash = {}", rep.output_hash);
    let tol = peak * 3e-2;
    let ok = rep.deterministic && ti == tf && rep.max_abs_err <= tol;
    println!(
        "  -> {}",
        if ok {
            "MoE layer is correct on the REAL OLMoE weights (deterministic, no flip, tracks float)."
        } else {
            "CHECK"
        }
    );
    if !ok {
        std::process::exit(1);
    }
}
