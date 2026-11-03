//! KPI / model card for a transformed model. Measured at convert time (size, peak RAM, latency, fidelity,
//! determinism) and written as a **self-contained HTML page** (no network, no external assets) plus a terminal
//! summary — so anyone can reproduce the numbers for their own model with `tritify-convert`.
//!
//! Honest by design: tritify trades size/memory/speed for exactness, determinism, verifiability, zero retraining,
//! and no CUDA. The card states that plainly; it is not a compression or throughput claim.

use serde_json::Value;
use std::fs;
use std::path::Path;

/// b=12 balanced-ternary weight level — mirrors `kt_core::layers::WLEVELS`.
const WLEVELS: i64 = (3i64.pow(12) - 1) / 2;

pub struct Kpi {
    pub model: String,
    pub family: String,
    pub task: String,
    pub plan: String,
    pub n_params: u64,
    pub orig_bytes: u64,
    pub orig_dtype: String,
    pub convert_ms: f64,
    pub peak_rss: Option<u64>,
    pub metric_name: String,
    pub metric: f64,
    pub max_abs_err: f64,
    pub deterministic: bool,
    pub hash: String,
}

impl Kpi {
    /// Bits needed to store one b=12 weight code: ceil(log2(2·WLEVELS+1)) = ceil(log2(3^12)).
    pub fn bits_per_weight() -> u32 {
        let levels = (2 * WLEVELS + 1) as u64;
        64 - (levels - 1).leading_zeros()
    }
    pub fn transformed_packed(&self) -> u64 {
        (self.n_params * Self::bits_per_weight() as u64).div_ceil(8)
    }
    pub fn transformed_i32(&self) -> u64 {
        self.n_params * 4
    }
    fn ratio(&self, transformed: u64, bytes_per_param: f64) -> f64 {
        if self.n_params == 0 || bytes_per_param == 0.0 {
            return 0.0;
        }
        transformed as f64 / (self.n_params as f64 * bytes_per_param)
    }
}

pub fn human(b: u64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let (mut v, mut i) = (b as f64, 0usize);
    while v >= 1024.0 && i < units.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.2} {}", units[i])
    }
}

/// Parameter count + on-disk bytes + original dtype, from the `.safetensors` file(s) in `dir` (header only, no
/// data load): 8-byte LE header length, then a JSON map of `name → {dtype, shape, data_offsets}`.
pub fn model_stats(dir: &str) -> (u64, u64, String) {
    let (mut n_params, mut bytes, mut dtype) = (0u64, 0u64, String::from("?"));
    let entries = match fs::read_dir(Path::new(dir)) {
        Ok(e) => e,
        Err(_) => return (0, 0, dtype),
    };
    for ent in entries.flatten() {
        let p = ent.path();
        if p.extension().is_some_and(|x| x == "safetensors") {
            if let Ok(meta) = fs::metadata(&p) {
                bytes += meta.len();
            }
            if let Ok(buf) = fs::read(&p) {
                if buf.len() < 8 {
                    continue;
                }
                let hlen = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
                if 8 + hlen > buf.len() {
                    continue;
                }
                if let Ok(Value::Object(hdr)) = serde_json::from_slice::<Value>(&buf[8..8 + hlen]) {
                    for (name, v) in &hdr {
                        if name == "__metadata__" {
                            continue;
                        }
                        if let Some(shape) = v.get("shape").and_then(|s| s.as_array()) {
                            let n: u64 = shape.iter().filter_map(|x| x.as_u64()).product();
                            n_params += n;
                            if dtype == "?" {
                                if let Some(dt) = v.get("dtype").and_then(|x| x.as_str()) {
                                    dtype = dt.to_string();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (n_params, bytes, dtype)
}

/// Peak resident set size (high-water mark) of this process, via `getrusage(RUSAGE_SELF)`. `ru_maxrss` sits at the
/// same byte offset on Linux and macOS (two 16-byte `timeval`s precede it); the unit differs (KiB on Linux, bytes
/// on macOS). Returns `None` if the call fails.
pub fn peak_rss_bytes() -> Option<u64> {
    #[repr(C)]
    struct Rusage {
        _pre: [i64; 4],
        ru_maxrss: i64,
        _post: [i64; 16],
    }
    extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut u: Rusage = unsafe { std::mem::zeroed() };
    if unsafe { getrusage(0, &mut u) } != 0 {
        return None;
    }
    let m = u.ru_maxrss.max(0) as u64;
    Some(if cfg!(target_os = "macos") { m } else { m * 1024 })
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// A compact terminal summary.
pub fn render_text(k: &Kpi) -> String {
    let rss = k.peak_rss.map(human).unwrap_or_else(|| "n/a".into());
    format!(
        "  ── KPI (transformed model) ──────────────────────────\n\
         \x20   params        : {}\n\
         \x20   original      : {} ({})\n\
         \x20   transformed   : ~{} packed (b=12, {} bit/weight) · {} as i32\n\
         \x20   peak RAM      : {}\n\
         \x20   convert+verify: {:.0} ms\n\
         \x20   fidelity      : {}={:.4}  max_abs_err={:.3e}\n\
         \x20   determinism   : {}\n\
         \x20   no retraining : 0 fine-tuning / 0 calibration / 0 QAT  ·  no CUDA/GPU required\n",
        k.n_params,
        human(k.orig_bytes),
        k.orig_dtype,
        human(k.transformed_packed()),
        Kpi::bits_per_weight(),
        human(k.transformed_i32()),
        rss,
        k.convert_ms,
        k.metric_name,
        k.metric,
        k.max_abs_err,
        k.deterministic,
    )
}

/// A self-contained HTML KPI page (inline CSS, no network, no external assets).
pub fn render_html(k: &Kpi) -> String {
    let rss = k.peak_rss.map(human).unwrap_or_else(|| "n/a".into());
    let bpp = if k.n_params > 0 { k.orig_bytes as f64 / k.n_params as f64 } else { 0.0 };
    let packed = k.transformed_packed();
    let r_fp16 = k.ratio(packed, 2.0);
    let r_int8 = k.ratio(packed, 1.0);
    let r_orig = k.ratio(packed, bpp);
    let det = if k.deterministic { "yes" } else { "NO" };

    TEMPLATE
        .replace("{MODEL}", &esc(&k.model))
        .replace("{FAMILY}", &esc(&k.family))
        .replace("{TASK}", &esc(&k.task))
        .replace("{PLAN}", &esc(&k.plan))
        .replace("{PARAMS}", &fmt_int(k.n_params))
        .replace("{METRIC_NAME}", &esc(&k.metric_name))
        .replace("{METRIC}", &format!("{:.4}", k.metric))
        .replace("{MAE}", &format!("{:.3e}", k.max_abs_err))
        .replace("{DET}", det)
        .replace("{HASH}", &esc(&k.hash))
        .replace("{ORIG}", &format!("{} ({})", human(k.orig_bytes), esc(&k.orig_dtype)))
        .replace("{TPACK}", &human(packed))
        .replace("{BITS}", &Kpi::bits_per_weight().to_string())
        .replace("{TI32}", &human(k.transformed_i32()))
        .replace("{RFP16}", &format!("{r_fp16:.2}"))
        .replace("{RINT8}", &format!("{r_int8:.2}"))
        .replace("{RORIG}", &format!("{r_orig:.2}"))
        .replace("{RSS}", &rss)
        .replace("{MS}", &format!("{:.0}", k.convert_ms))
}

fn fmt_int(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

const TEMPLATE: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>tritify KPI — {MODEL}</title>
<style>
  :root { color-scheme: light dark; --bg:#f7f7f8; --card:#fff; --ink:#1a1a1c; --muted:#6b6b72; --line:#e6e6ea;
          --good:#11813a; --accent:#2b5cff; --mono:ui-monospace,SFMono-Regular,Menlo,monospace; }
  @media (prefers-color-scheme: dark) { :root { --bg:#111214; --card:#1a1b1e; --ink:#eaeaec; --muted:#9a9aa2;
          --line:#2a2b2f; --good:#4ade80; --accent:#7aa2ff; } }
  * { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--ink);
      font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; padding:24px; }
  .wrap { max-width:720px; margin:0 auto; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:22px 24px; margin-bottom:16px; }
  h1 { font-size:19px; margin:0 0 2px; } .sub { color:var(--muted); font-size:13px; margin:0 0 4px; }
  .facts { display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 2px; }
  .fact { background:color-mix(in srgb, var(--good) 12%, transparent); color:var(--good); border:1px solid
          color-mix(in srgb, var(--good) 30%, transparent); border-radius:999px; padding:5px 12px; font-size:12.5px; font-weight:600; }
  table { width:100%; border-collapse:collapse; margin-top:8px; }
  td { padding:9px 2px; border-bottom:1px solid var(--line); vertical-align:top; }
  td.k { color:var(--muted); width:38%; } td.v { font-weight:600; }
  .mono { font-family:var(--mono); font-size:12px; word-break:break-all; font-weight:400; }
  .note { color:var(--muted); font-size:12px; margin-top:12px; }
  .big { color:var(--good); } .hdr { font-size:12px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); margin:18px 0 4px; }
</style></head>
<body><div class="wrap">
  <div class="card">
    <h1>{MODEL}</h1>
    <p class="sub">family <b>{FAMILY}</b> · task <b>{TASK}</b> · transformed by tritify</p>
    <div class="facts">
      <span class="fact">0 retraining</span><span class="fact">0 calibration</span>
      <span class="fact">0 QAT</span><span class="fact">no CUDA / GPU required</span>
      <span class="fact">deterministic</span>
    </div>
    <p class="note">tritify converts a trained model to an exact-integer, deterministic, verifiable form —
    post-hoc, no fine-tuning. The value is exactness + reproducibility + portability, <b>not</b> smaller or faster
    than fp16/int8.</p>

    <div class="hdr">Fidelity &amp; determinism</div>
    <table>
      <tr><td class="k">decision vs float ({METRIC_NAME})</td><td class="v big">{METRIC}</td></tr>
      <tr><td class="k">max abs error (raw)</td><td class="v">{MAE}</td></tr>
      <tr><td class="k">deterministic</td><td class="v">{DET}</td></tr>
      <tr><td class="k">reproducing SHA-256</td><td class="v mono">{HASH}</td></tr>
    </table>

    <div class="hdr">Footprint</div>
    <table>
      <tr><td class="k">parameters</td><td class="v">{PARAMS}</td></tr>
      <tr><td class="k">original on disk</td><td class="v">{ORIG}</td></tr>
      <tr><td class="k">transformed (b=12, {BITS} bit/weight, packed)</td><td class="v">~{TPACK}</td></tr>
      <tr><td class="k">&nbsp;&nbsp;ratio vs original / fp16 / int8</td><td class="v">{RORIG}× / {RFP16}× / {RINT8}×</td></tr>
      <tr><td class="k">transformed (i32, unpacked)</td><td class="v">{TI32}</td></tr>
      <tr><td class="k">peak RAM (convert + self-verify)</td><td class="v">{RSS}</td></tr>
      <tr><td class="k">convert + self-verify time</td><td class="v">{MS} ms</td></tr>
    </table>
    <p class="note">Transformed size is estimated from the parameter count at the b=12 weight grid (the engine
    builds in memory; it does not yet serialize a packed model). Latency includes the float reference pass of the
    self-check, not inference alone. Reproduce: <span class="mono">tritify-convert &lt;model_dir&gt;</span>.</p>
  </div>
  <p class="note" style="text-align:center">op graph: {PLAN}</p>
</div></body></html>"##;
