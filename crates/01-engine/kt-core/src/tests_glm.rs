//! GLM-4.5's architecture pieces against float, without a checkpoint: partial RoPE, the sigmoid router
//! with its selection bias, and the shared expert. Each new path is additive: with its knob unset the engine is the one
//! without them, which the model anchors check; here `rot_dim = head_dim` is checked against the attention without it.

use crate::attention::Attention;
use crate::layers::{MoeMlp, QuantLinear, SwiGluMlp, ROUTER_CODE_BITS};
use crate::module::Module;
use crate::router::{SigmoidScore, SIGMOID_IDX_BITS, SIGMOID_RANGE};
use crate::tensor::IntTensor;
use crate::tests::{probe, vecf};
use crate::verify::verify;
use kt_int::Scale;

const DIM: usize = 32;

fn peak(v: &[f32]) -> f64 {
    v.iter().fold(0.0f64, |a, &x| a.max((x as f64).abs()))
}

fn mlp(s: &mut u64, name: &str, hidden: usize) -> SwiGluMlp {
    let (wg, wu, wd) = (vecf(s, hidden * DIM), vecf(s, hidden * DIM), vecf(s, DIM * hidden));
    SwiGluMlp::new(
        name,
        QuantLinear::from_weights("gate", &wg, hidden, DIM),
        QuantLinear::from_weights("up", &wu, hidden, DIM),
        QuantLinear::from_weights("down", &wd, DIM, hidden),
    )
}

/// A small MoE with 16 experts, top-4, the same weights for every seed.
fn moe(seed: u64) -> MoeMlp {
    let (experts, top_k) = (16usize, 4usize);
    let mut s = seed;
    let router_w = vecf(&mut s, experts * DIM);
    let mlps = (0..experts).map(|e| mlp(&mut s, &format!("expert{e}"), 24)).collect();
    MoeMlp::new("moe", QuantLinear::from_weights("router", &router_w, experts, DIM), mlps, top_k)
        .with_norm_topk_prob(true)
}

fn bias(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    vecf(&mut s, n).iter().map(|b| b * 0.05).collect()
}

fn rows(x: &IntTensor) -> Vec<IntTensor> {
    (0..x.len() / DIM).map(|t| IntTensor::new(x.data[t * DIM..(t + 1) * DIM].to_vec(), vec![DIM], x.scale)).collect()
}

// ---- partial RoPE ------------------------------------------------------------------------------

/// `apply_rotary_pos_emb` of transformers' `modeling_glm4_moe.py` (f90e93c4), transcribed for one head at position
/// `p`: `inv_freq = 1/θ^(arange(0, rd, 2)/rd)`, `emb = cat(freqs, freqs)`, `x_rot·cos + rotate_half(x_rot)·sin`, the
/// dims past `rd` passed through.
fn hf_rope(x: &[f64], p: usize, rd: usize, theta: f64) -> Vec<f64> {
    let h = rd / 2;
    let freq = |i: usize| p as f64 * (1.0 / theta.powf((2 * (i % h)) as f64 / rd as f64));
    let rot = |i: usize| if i < h { -x[i + h] } else { x[i - h] };
    (0..x.len()).map(|i| if i < rd { x[i] * freq(i).cos() + rot(i) * freq(i).sin() } else { x[i] }).collect()
}

fn attention(rd: Option<usize>) -> Attention {
    let (nh, nkv, hd) = (4usize, 2usize, 16usize);
    let mut s = 0x0039u64;
    let (wq, wk, wv, wo) = (
        vecf(&mut s, nh * hd * DIM),
        vecf(&mut s, nkv * hd * DIM),
        vecf(&mut s, nkv * hd * DIM),
        vecf(&mut s, DIM * nh * hd),
    );
    let a = Attention::from_weights("attn", DIM, nh, nkv, hd, &wq, &wk, &wv, &wo, (None, None, None), 32, 1e6);
    match rd {
        Some(rd) => a.with_rotary_dim(rd),
        None => a,
    }
}

#[test]
fn partial_rope_is_the_reference_rotation_and_leaves_the_tail() {
    let (hd, rd, theta) = (16usize, 8usize, 1e6);
    let a = attention(Some(rd));
    assert_eq!(a.rotary_dim(), rd);
    let mut s = 0x7071u64;
    for p in [0usize, 1, 5, 31] {
        // the float reference is transformers' formula
        let x: Vec<f64> = vecf(&mut s, hd).iter().map(|&v| v as f64).collect();
        let mut r = x.clone();
        a.rope_ref(&mut r, p);
        let hf = hf_rope(&x, p, rd, theta);
        assert!(r.iter().zip(&hf).all(|(a, b)| (a - b).abs() <= 1e-12), "p={p}: rope_ref {r:?} != transformers {hf:?}");
        assert_eq!(r[rd..], x[rd..], "p={p}: the dims past rot_dim pass through");
        // the integer rotation is that rotation of its codes, within one rounding per product
        let codes: Vec<i32> = vecf(&mut s, hd).iter().map(|&v| (v as f64 * (1 << 28) as f64) as i32).collect();
        let mut ci = codes.clone();
        a.rope_int(&mut ci, p);
        let exact = hf_rope(&codes.iter().map(|&c| c as f64).collect::<Vec<_>>(), p, rd, theta);
        let worst = ci.iter().zip(&exact).map(|(&c, &e)| (c as f64 - e).abs()).fold(0.0, f64::max);
        assert!(worst <= 2.0, "p={p}: integer RoPE is {worst} codes off the exact rotation");
        assert_eq!(ci[rd..], codes[rd..], "p={p}: the integer tail passes through");
    }
}

#[test]
fn a_rotary_dim_of_head_dim_is_the_rope_without_it() {
    let (full, same) = (attention(None), attention(Some(16)));
    let mut s = 0x7072u64;
    for p in 0..32 {
        let v: Vec<i32> = vecf(&mut s, 16).iter().map(|&x| (x as f64 * (1 << 28) as f64) as i32).collect();
        let (mut a, mut b) = (v.clone(), v);
        full.rope_int(&mut a, p);
        same.rope_int(&mut b, p);
        assert_eq!(a, b, "p={p}");
    }
    let x = probe(&mut s, 5, DIM);
    let (a, b) = (full.forward_int(&x), same.forward_int(&x));
    assert_eq!((a.data, a.scale), (b.data, b.scale));
}

#[test]
fn partial_rope_attention_tracks_float_and_is_deterministic() {
    let a = attention(Some(8));
    let mut s = 0x7073u64;
    let p = probe(&mut s, 6, DIM);
    let rep = verify(&a, &p);
    let r = a.forward_ref(&p.to_f32());
    assert!(rep.deterministic);
    assert!(rep.max_abs_err <= peak(&r) * 3e-2, "partial-RoPE attention drifted from float: {}", rep.max_abs_err);
    assert_ne!(attention(None).forward_int(&p).data, a.forward_int(&p).data, "partial RoPE must change the output");
}

#[test]
#[should_panic(expected = "must be even")]
fn an_odd_rotary_dim_fails_closed() {
    let _ = attention(Some(7));
}

// ---- the sigmoid router ------------------------------------------------------------------------

#[test]
fn sigmoid_codes_are_within_one_table_step_and_clamp_at_the_ends() {
    let sig = SigmoidScore::new(&[0.0; 4]);
    let k = (1u64 << ROUTER_CODE_BITS) as f64;
    let step = 2.0 * SIGMOID_RANGE / ((1u64 << SIGMOID_IDX_BITS) - 1) as f64;
    let scale = Scale::from_f64(1.0 / 4096.0);
    for l in (-80_000i32..=80_000).step_by(997) {
        let codes = sig.codes(&IntTensor::new(vec![l, -l, 0, l / 3], vec![4], scale));
        for (c, v) in codes.iter().zip([l, -l, 0, l / 3]) {
            let real = v as f64 / 4096.0;
            let want = k / (1.0 + (-real).exp());
            // σ' ≤ 1/4, so one table step moves the code at most step/4 · 2^20; the index rounds once
            assert!((*c as f64 - want).abs() <= step / 4.0 * k + 1.0, "logit {real}: code {c}, σ·2^20 = {want}");
            assert!((0..=1 << ROUTER_CODE_BITS).contains(c));
        }
    }
    let ends = sig.codes(&IntTensor::new(vec![-1 << 20, 1 << 20, 0, 0], vec![4], Scale::from_f64(1.0)));
    assert_eq!((ends[0], ends[1]), (0, 1 << ROUTER_CODE_BITS), "past the table the code is 0 or 2^ROUTER_CODE_BITS");
}

#[test]
fn a_zero_bias_selects_what_the_softmax_router_selects() {
    let (soft, sig) = (moe(0x51), moe(0x51).with_sigmoid_router(&[0.0; 16]));
    let mut s = 0x52u64;
    let (mut compared, mut total) = (0, 0);
    for row in rows(&probe(&mut s, 64, DIM)) {
        total += 1;
        // the two agree wherever σ's codes separate the k-th and the (k+1)-th logit; a code tie may reorder them
        let logits = sig.router.forward_int(&row);
        let codes = SigmoidScore::new(&[0.0; 16]).codes(&logits);
        let mut sorted = codes.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        if sorted[3] == sorted[4] {
            continue;
        }
        compared += 1;
        let (mut a, mut b) = (soft.route(&row).experts, sig.route(&row).experts);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "a zero bias must select the softmax router's experts");
    }
    assert!(compared * 10 >= total * 9, "only {compared} of {total} rows were free of a code tie at the cut");
}

#[test]
fn the_bias_moves_the_choice_but_not_the_weights() {
    let base = moe(0x53);
    let mut s = 0x54u64;
    let row = &rows(&probe(&mut s, 1, DIM))[0];
    let plain = moe(0x53).with_sigmoid_router(&[0.0; 16]).route(row);
    let outsider = (0..16).find(|e| !plain.experts.contains(e)).unwrap();
    let mut b = [0.0f32; 16];
    b[outsider] = 1.5; // more than any σ difference: the outsider must be chosen
    let biased = moe(0x53).with_sigmoid_router(&b).route(row);
    assert!(biased.experts.contains(&outsider), "the bias must move the choice");
    let codes = SigmoidScore::new(&[0.0; 16]).codes(&base.router.forward_int(row));
    for (e, c) in biased.experts.iter().zip(&biased.ec) {
        assert_eq!(*c, codes[*e], "expert {e}: its weight is its σ code, without the bias");
    }
    assert_eq!(biased.sumc, biased.ec.iter().sum::<i64>(), "norm_topk_prob: the weights sum to 1");
    let raw = moe(0x53).with_norm_topk_prob(false).with_sigmoid_router(&b).route(row);
    assert_eq!(raw.sumc, 1 << ROUTER_CODE_BITS, "without norm_topk_prob the weights are the raw scores");
}

#[test]
fn the_sigmoid_moe_tracks_float_with_no_routing_flip() {
    let b = bias(0x55, 16);
    let m = moe(0x56).with_sigmoid_router(&b);
    let sig = SigmoidScore::new(&b);
    let mut s = 0x57u64;
    let x = probe(&mut s, 8, DIM);
    for row in rows(&x) {
        let (_, choice) = sig.reference(&m.router.forward_ref(&row.to_f32()));
        let mut want: Vec<usize> = (0..16).collect();
        want.sort_by(|&a, &b| choice[b].partial_cmp(&choice[a]).unwrap().then(a.cmp(&b)));
        want.truncate(4);
        assert_eq!(m.route(&row).experts, want, "the integer choice must be the float choice");
    }
    let rep = verify(&m, &x);
    let r = m.forward_ref(&x.to_f32());
    assert!(rep.deterministic);
    assert!(rep.max_abs_err <= peak(&r) * 3e-2, "the sigmoid MoE drifted from float: {}", rep.max_abs_err);
}

// ---- the shared expert -------------------------------------------------------------------------

fn glm_like(seed: u64) -> MoeMlp {
    let mut s = seed ^ 0xA5;
    moe(seed).with_sigmoid_router(&bias(seed, 16)).with_shared_expert(mlp(&mut s, "shared", 40))
}

#[test]
fn the_seam_is_bit_identical_with_a_shared_expert() {
    let m = glm_like(0x5A);
    let mut s = 0x5Bu64;
    for n in [1usize, 5] {
        let x = probe(&mut s, n, DIM);
        let (base, via) = (
            m.forward_int(&x),
            m.forward_int_with_experts(&x, |ids, row| ids.iter().map(|&e| m.experts[e].forward_int(row)).collect()),
        );
        assert_eq!((base.data, base.scale), (via.data, via.scale), "rows={n}");
    }
}

#[test]
fn the_shared_expert_is_added_unweighted_and_tracks_float() {
    let (with, without) = (glm_like(0x5C), moe(0x5C).with_sigmoid_router(&bias(0x5C, 16)));
    let mut s = 0x5Du64;
    let x = probe(&mut s, 6, DIM);
    // the float twin adds shared(x) to the routed sum, unweighted
    let (rw, rwo) = (with.forward_ref(&x.to_f32()), without.forward_ref(&x.to_f32()));
    let sh = with.shared_expert().unwrap().forward_ref(&x.to_f32());
    assert!(rw.iter().zip(&rwo).zip(&sh).all(|((a, b), c)| (a - (b + c)).abs() <= 1e-5 * (1.0 + a.abs())));
    let rep = verify(&with, &x);
    assert!(rep.deterministic);
    assert!(
        rep.max_abs_err <= peak(&rw) * 3e-2,
        "the MoE with a shared expert drifted from float: {}",
        rep.max_abs_err
    );
    assert_ne!(with.forward_int(&x).data, without.forward_int(&x).data, "the shared expert must change the output");
}
