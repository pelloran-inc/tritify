//! Foundation tests for the dynamic quantization scheme: each op tracks its float reference and is
//! deterministic; the plugin contract builds a CausalLM; a full tiny decoder forward matches float.

use crate::attention::Attention;
use crate::causal::CausalLM;
use crate::check::{CheckReport, VerifiableModel};
use crate::layers::{Embedding, LayerNorm, MoeMlp, QuantLinear, RmsNorm, SwiGluMlp, ACT_LEVELS};
use crate::model::{DecoderConfig, DecoderModel, DecoderWeights, LayerWeights};
use crate::module::{Module, Sequential};
use crate::plugin::*;
use crate::tensor::IntTensor;
use crate::verify::verify;
use kt_int::Scale;

pub(crate) fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s
}
pub(crate) fn unif(s: &mut u64) -> f32 {
    ((lcg(s) >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32
}
pub(crate) fn vecf(s: &mut u64, n: usize) -> Vec<f32> {
    (0..n).map(|_| unif(s)).collect()
}
fn peak(v: &[f32]) -> f64 {
    v.iter().fold(0.0f64, |a, &x| a.max((x as f64).abs()))
}
/// A probe activation tensor (dynamic per-tensor, ACT precision) from random reals.
pub(crate) fn probe(s: &mut u64, rows: usize, cols: usize) -> IntTensor {
    let real: Vec<f64> = (0..rows * cols).map(|_| unif(s) as f64).collect();
    IntTensor::from_reals(&real, vec![rows, cols], ACT_LEVELS)
}

#[test]
fn quantlinear_tracks_float_and_is_deterministic() {
    let (out, inp) = (64usize, 128usize);
    let mut s = 0xBEEFu64;
    let w = vecf(&mut s, out * inp);
    let p = probe(&mut s, 1, inp);
    let layer = QuantLinear::from_weights("lin", &w, out, inp);
    let rep = verify(&layer, &p);
    let r = layer.forward_ref(&p.to_f32());
    println!("QuantLinear: max_abs_err={:.3e} (peak_ref={:.3e}) det={}", rep.max_abs_err, peak(&r), rep.deterministic);
    assert!(rep.deterministic);
    assert!(rep.max_abs_err <= peak(&r) * 1e-2, "quantlinear drifted from float");
}

#[test]
fn sequential_composes_and_is_deterministic() {
    let (d0, d1, d2) = (128usize, 96usize, 64usize);
    let mut s = 7u64;
    let w1 = vecf(&mut s, d1 * d0);
    let w2 = vecf(&mut s, d2 * d1);
    let p = probe(&mut s, 1, d0);
    let model = Sequential::new("two")
        .push(Box::new(QuantLinear::from_weights("l1", &w1, d1, d0)))
        .push(Box::new(QuantLinear::from_weights("l2", &w2, d2, d1)));
    let out = model.forward_int(&p);
    assert_eq!(out.len(), d2);
    assert_eq!(out.data, model.forward_int(&p).data);
}

#[test]
fn rmsnorm_tracks_float_and_is_deterministic() {
    let dim = 128usize;
    let mut s = 0x5151u64;
    let w = vecf(&mut s, dim);
    let p = probe(&mut s, 1, dim);
    let layer = RmsNorm::new("norm", &w, 1e-6);
    let rep = verify(&layer, &p);
    let r = layer.forward_ref(&p.to_f32());
    println!("RmsNorm: max_abs_err={:.3e} (peak_ref={:.3e}) det={}", rep.max_abs_err, peak(&r), rep.deterministic);
    assert!(rep.deterministic);
    assert!(rep.max_abs_err <= peak(&r) * 1e-2, "rmsnorm drifted from float");
}

#[test]
fn layernorm_tracks_float_and_is_deterministic() {
    let dim = 128usize;
    let mut s = 0x1A4Eu64;
    let w = vecf(&mut s, dim);
    let b = vecf(&mut s, dim);
    // multi-row, so the common-scale path across rows is exercised.
    let p = probe(&mut s, 3, dim);
    let layer = LayerNorm::new("ln", &w, &b, 1e-5);
    let rep = verify(&layer, &p);
    let r = layer.forward_ref(&p.to_f32());
    println!("LayerNorm: max_abs_err={:.3e} (peak_ref={:.3e}) det={}", rep.max_abs_err, peak(&r), rep.deterministic);
    assert!(rep.deterministic);
    assert!(rep.max_abs_err <= peak(&r) * 2e-2, "layernorm drifted from float");
}

#[test]
fn swiglu_tracks_float_and_is_deterministic() {
    let (dim, hidden) = (64usize, 128usize);
    let mut s = 0x9999u64;
    let (wg, wu, wd) = (vecf(&mut s, hidden * dim), vecf(&mut s, hidden * dim), vecf(&mut s, dim * hidden));
    let p = probe(&mut s, 1, dim);
    let mlp = SwiGluMlp::new(
        "mlp",
        QuantLinear::from_weights("gate", &wg, hidden, dim),
        QuantLinear::from_weights("up", &wu, hidden, dim),
        QuantLinear::from_weights("down", &wd, dim, hidden),
    );
    let rep = verify(&mlp, &p);
    let r = mlp.forward_ref(&p.to_f32());
    println!("SwiGLU: max_abs_err={:.3e} (peak_ref={:.3e}) det={}", rep.max_abs_err, peak(&r), rep.deterministic);
    assert!(rep.deterministic);
    assert!(rep.max_abs_err <= peak(&r) * 2e-2, "swiglu drifted from float");
}

#[test]
fn moe_mlp_tracks_float_sparse_and_deterministic() {
    // The MoE layer mechanism: integer router top-k + sparse experts + softmax-weighted sum.
    let (dim, hidden, experts, top_k) = (64usize, 96usize, 8usize, 2usize);
    let mut s = 0x30E1u64;
    let router_w = vecf(&mut s, experts * dim); // [E, dim]
    let mut expert_mlps = Vec::new();
    for e in 0..experts {
        let (wg, wu, wd) = (vecf(&mut s, hidden * dim), vecf(&mut s, hidden * dim), vecf(&mut s, dim * hidden));
        expert_mlps.push(SwiGluMlp::new(
            format!("expert{e}"),
            QuantLinear::from_weights("gate", &wg, hidden, dim),
            QuantLinear::from_weights("up", &wu, hidden, dim),
            QuantLinear::from_weights("down", &wd, dim, hidden),
        ));
    }
    let moe = MoeMlp::new("moe", QuantLinear::from_weights("router", &router_w, experts, dim), expert_mlps, top_k);
    let p = probe(&mut s, 1, dim);

    // the integer and float router must pick the SAME top-k experts (no routing flip)
    let li = moe.router.forward_int(&p).data;
    let lf = moe.router.forward_ref(&p.to_f32());
    let tk = |v: &[f32]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(top_k);
        idx
    };
    let ti = tk(&li.iter().map(|&x| x as f32).collect::<Vec<_>>());
    let tf = tk(&lf);
    let no_flip = ti == tf;

    let rep = verify(&moe, &p);
    let r = moe.forward_ref(&p.to_f32());
    println!(
        "MoE({experts}e/top{top_k}): max_abs_err={:.3e} (peak_ref={:.3e}) det={} top-k int={ti:?} float={tf:?} no_flip={no_flip}",
        rep.max_abs_err, peak(&r), rep.deterministic
    );
    assert!(rep.deterministic, "MoE integer path not deterministic");
    assert!(no_flip, "router chose different top-k experts (int vs float) — a routing flip");
    assert!(rep.max_abs_err <= peak(&r) * 3e-2, "MoE drifted from float");
}

#[test]
fn moe_forward_int_with_experts_matches_forward_int() {
    // The distributed-expert seam is a numeric NO-OP: computing the selected experts through the callback
    // (here backed by the SAME local experts) must be byte-identical to forward_int, for a one-token AND a sequence
    // input (covering both the single-row and the row-merge paths). A local check: no network, no transport.
    let (dim, hidden, experts, top_k) = (64usize, 96usize, 8usize, 2usize);
    let mut s = 0x5117u64;
    let router_w = vecf(&mut s, experts * dim);
    let mut expert_mlps = Vec::new();
    for e in 0..experts {
        let (wg, wu, wd) = (vecf(&mut s, hidden * dim), vecf(&mut s, hidden * dim), vecf(&mut s, dim * hidden));
        expert_mlps.push(SwiGluMlp::new(
            format!("expert{e}"),
            QuantLinear::from_weights("gate", &wg, hidden, dim),
            QuantLinear::from_weights("up", &wu, hidden, dim),
            QuantLinear::from_weights("down", &wd, dim, hidden),
        ));
    }
    let moe = MoeMlp::new("moe", QuantLinear::from_weights("router", &router_w, experts, dim), expert_mlps, top_k);
    for rows in [1usize, 5] {
        let x = probe(&mut s, rows, dim);
        let base = moe.forward_int(&x);
        let via =
            moe.forward_int_with_experts(&x, |ids, row| ids.iter().map(|&e| moe.experts[e].forward_int(row)).collect());
        assert_eq!(base.data, via.data, "rows={rows}: int data differs — the seam is not a no-op");
        assert_eq!(base.to_f32(), via.to_f32(), "rows={rows}: f32 differs — the seam is not a no-op");
    }
}

#[test]
fn attention_gqa_causal_tracks_float_and_is_deterministic() {
    let (dim, nh, nkv, hd, seq) = (32usize, 4usize, 2usize, 8usize, 5usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let mut s = 0x4242u64;
    let (wq, wk, wv, wo) =
        (vecf(&mut s, qd * dim), vecf(&mut s, kvd * dim), vecf(&mut s, kvd * dim), vecf(&mut s, dim * qd));
    let p = probe(&mut s, seq, dim);
    let attn = Attention::from_weights("attn", dim, nh, nkv, hd, &wq, &wk, &wv, &wo, (None, None, None), 16, 10000.0);
    let rep = verify(&attn, &p);
    let r = attn.forward_ref(&p.to_f32());
    println!("Attention: max_abs_err={:.3e} (peak_ref={:.3e}) det={}", rep.max_abs_err, peak(&r), rep.deterministic);
    assert!(rep.deterministic);
    assert!(rep.max_abs_err <= peak(&r) * 3e-2, "attention drifted from float");
}

#[test]
fn attention_with_qk_norm_tracks_float_and_is_deterministic() {
    // Optional QK-norm (OLMoE/Qwen3): RMSNorm on the q/k projection before RoPE.
    let (dim, nh, nkv, hd, seq) = (32usize, 4usize, 2usize, 8usize, 5usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let mut s = 0x4243u64;
    let (wq, wk, wv, wo) =
        (vecf(&mut s, qd * dim), vecf(&mut s, kvd * dim), vecf(&mut s, kvd * dim), vecf(&mut s, dim * qd));
    let (qn, kn) = (vecf(&mut s, qd), vecf(&mut s, kvd));
    let p = probe(&mut s, seq, dim);
    let attn = Attention::from_weights("attn", dim, nh, nkv, hd, &wq, &wk, &wv, &wo, (None, None, None), 16, 10000.0)
        .with_qk_norm(Some(RmsNorm::new("q_norm", &qn, 1e-6)), Some(RmsNorm::new("k_norm", &kn, 1e-6)));
    let rep = verify(&attn, &p);
    let r = attn.forward_ref(&p.to_f32());
    println!(
        "Attention+QKnorm: max_abs_err={:.3e} (peak_ref={:.3e}) det={}",
        rep.max_abs_err,
        peak(&r),
        rep.deterministic
    );
    assert!(rep.deterministic);
    assert!(rep.max_abs_err <= peak(&r) * 3e-2, "qk-norm attention drifted from float");
}

#[test]
fn attention_with_per_head_qk_norm_tracks_float() {
    // Qwen3-style PER-HEAD QK-norm: the q_norm/k_norm weights are head_dim-length, so RMSNorm (row-wise over its own
    // weight length) normalizes EACH head independently — same attention code, no branch. Proves the generic/adaptive
    // QK-norm handles the per-head granularity, not just the full-projection (OLMoE) one.
    let (dim, nh, nkv, hd, seq) = (32usize, 4usize, 2usize, 8usize, 5usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let mut s = 0x9aa1u64;
    let (wq, wk, wv, wo) =
        (vecf(&mut s, qd * dim), vecf(&mut s, kvd * dim), vecf(&mut s, kvd * dim), vecf(&mut s, dim * qd));
    let (qn, kn) = (vecf(&mut s, hd), vecf(&mut s, hd)); // head_dim-length → per-head
    let p = probe(&mut s, seq, dim);
    let attn = Attention::from_weights("attn", dim, nh, nkv, hd, &wq, &wk, &wv, &wo, (None, None, None), 16, 10000.0)
        .with_qk_norm(Some(RmsNorm::new("q_norm", &qn, 1e-6)), Some(RmsNorm::new("k_norm", &kn, 1e-6)));
    let rep = verify(&attn, &p);
    let r = attn.forward_ref(&p.to_f32());
    println!(
        "Attention+per-head-QKnorm: max_abs_err={:.3e} (peak_ref={:.3e}) det={}",
        rep.max_abs_err,
        peak(&r),
        rep.deterministic
    );
    assert!(rep.deterministic);
    assert!(rep.max_abs_err <= peak(&r) * 3e-2, "per-head qk-norm attention drifted from float");
}

#[test]
fn embedding_lookup_is_deterministic() {
    let (vocab, dim) = (100usize, 48usize);
    let mut s = 0x7777u64;
    let w = vecf(&mut s, vocab * dim);
    let emb = Embedding::from_weights("embed", &w, vocab, dim);
    let t = emb.lookup(42);
    assert_eq!(t.len(), dim);
    assert_eq!(t.data, emb.lookup(42).data);
    assert_eq!(emb.lookup_seq(&[1, 2, 3]).shape, vec![3, dim]);
}

#[test]
fn full_decoder_forward_tracks_float_and_is_deterministic() {
    let cfg = DecoderConfig {
        dim: 16,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        hidden: 32,
        vocab: 20,
        eps: 1e-6,
        rope_theta: 10000.0,
        max_seq: 8,
    };
    let (dim, qd, kvd, hid, voc) =
        (cfg.dim, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim, cfg.hidden, cfg.vocab);
    let mut s = 0xD00Du64;
    let mut layers = Vec::new();
    for _ in 0..cfg.n_layers {
        layers.push(LayerWeights {
            input_norm: vecf(&mut s, dim),
            wq: vecf(&mut s, qd * dim),
            wk: vecf(&mut s, kvd * dim),
            wv: vecf(&mut s, kvd * dim),
            wo: vecf(&mut s, dim * qd),
            wq_b: None,
            wk_b: None,
            wv_b: None,
            post_norm: vecf(&mut s, dim),
            gate: vecf(&mut s, hid * dim),
            up: vecf(&mut s, hid * dim),
            down: vecf(&mut s, dim * hid),
        });
    }
    let w = DecoderWeights {
        embed: vecf(&mut s, voc * dim),
        layers,
        final_norm: vecf(&mut s, dim),
        lm_head: vecf(&mut s, voc * dim),
        cfg,
    };
    let test = [5usize, 6, 7, 8];
    let model = DecoderModel::build(&w);

    let li = model.forward_int(&test).to_f32();
    let lr = model.forward_ref(&test);
    let mae = li.iter().zip(&lr).fold(0.0f64, |a, (&x, &y)| a.max((x as f64 - y as f64).abs()));
    let pr = peak(&lr);
    let (ai, ar) = (model.argmax_int(&test), model.argmax_ref(&test));
    let agree = ai.iter().zip(&ar).filter(|(a, b)| a == b).count();
    println!(
        "DecoderModel: mae={:.3e} peak_ref={:.3e} argmax_agree={}/{} hash={}",
        mae,
        pr,
        agree,
        test.len(),
        &model.logits_hash(&test)[..16]
    );
    assert_eq!(model.logits_hash(&test), model.logits_hash(&test)); // deterministic
    assert!(mae <= pr * 0.1, "decoder logits drifted: mae={mae:.3e} peak={pr:.3e}");
    assert_eq!(ai, ar, "integer argmax token stream must match float");
}

// ── the plugin contract, proven with a synthetic source + a trivial CausalLM ──
struct MockSource {
    arch: String,
    vocab: usize,
}
impl ModelSource for MockSource {
    fn hf_arch_ids(&self) -> Vec<String> {
        vec![self.arch.clone()]
    }
    fn config_usize(&self, key: &str) -> Option<usize> {
        match key {
            "hidden_size" => Some(8),
            "vocab_size" => Some(self.vocab),
            _ => None,
        }
    }
    fn config_f64(&self, _k: &str) -> Option<f64> {
        None
    }
    fn tensor(&self, _n: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        None
    }
}

struct TinyLM {
    vocab: usize,
}
impl CausalLM for TinyLM {
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn forward_int(&self, tokens: &[usize]) -> IntTensor {
        IntTensor::new(vec![0i32; tokens.len() * self.vocab], vec![tokens.len(), self.vocab], Scale::from_f64(1.0))
    }
    fn forward_ref(&self, tokens: &[usize]) -> Vec<f32> {
        vec![0.0; tokens.len() * self.vocab]
    }
}
impl VerifiableModel for TinyLM {
    fn task(&self) -> &str {
        "mock"
    }
    fn self_check(&self) -> CheckReport {
        let probe = [0usize, 1, 2];
        let hash = crate::verify::hash_int(&self.forward_int(&probe));
        CheckReport {
            task: "mock".into(),
            metric_name: "argmax_agreement".into(),
            metric: 1.0,
            max_abs_err: 0.0,
            deterministic: hash == crate::verify::hash_int(&self.forward_int(&probe)),
            hash,
            detail: "mock".into(),
        }
    }
}

struct MockArch;
impl ModelArch for MockArch {
    fn meta(&self) -> ArchMeta {
        ArchMeta { family: "mock".into(), hf_arch_ids: vec!["MockForCausalLM".into()], summary: "demo".into() }
    }
    fn plan(&self, src: &dyn ModelSource) -> Result<ModelPlan, String> {
        let vocab = src.config_usize("vocab_size").ok_or("no vocab")?;
        let dim = src.config_usize("hidden_size").ok_or("no dim")?;
        Ok(ModelPlan {
            name: "mock".into(),
            family: self.meta().family,
            layers: vec![
                LayerSpec::Embedding { vocab, dim },
                LayerSpec::Linear { in_features: dim, out_features: vocab },
            ],
        })
    }
    fn build(&self, src: &dyn ModelSource) -> Result<Box<dyn VerifiableModel>, String> {
        let vocab = src.config_usize("vocab_size").ok_or("no vocab")?;
        Ok(Box::new(TinyLM { vocab }))
    }
}

#[test]
fn plugin_contract_detect_plan_build() {
    let mut reg = Registry::new();
    reg.register(Box::new(MockArch));
    let src = MockSource { arch: "MockForCausalLM".into(), vocab: 32 };

    let plugin = reg.select(&src).expect("a plugin must match");
    assert_eq!(plugin.meta().family, "mock");
    let plan = plugin.plan(&src).expect("plan");
    println!("plan: {}", plan.summary());
    assert_eq!(plan.layers.len(), 2);

    let model = plugin.build(&src).expect("build");
    let report = model.self_check();
    assert_eq!(report.task, "mock");
    assert!(report.deterministic);

    let other = MockSource { arch: "SomethingElse".into(), vocab: 32 };
    assert!(reg.select(&other).is_none());
}
