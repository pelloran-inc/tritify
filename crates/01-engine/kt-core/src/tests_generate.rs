//! Greedy generation and embeddings on a tiny synthetic decoder: the canonical path, the stop rule, determinism.

use crate::generate::{argmax_codes, embed_mean, generate_greedy, Finish};
use crate::model::{DecoderConfig, DecoderModel, DecoderWeights, LayerWeights};
use crate::tests::vecf;

fn tiny() -> DecoderModel {
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
        max_seq: 32,
    };
    let (dim, qd, kvd, hid, voc) =
        (cfg.dim, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim, cfg.hidden, cfg.vocab);
    let mut s = 0x6E6Eu64;
    let layers = (0..cfg.n_layers)
        .map(|_| LayerWeights {
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
        })
        .collect();
    let w = DecoderWeights {
        embed: vecf(&mut s, voc * dim),
        layers,
        final_norm: vecf(&mut s, dim),
        lm_head: vecf(&mut s, voc * dim),
        cfg,
    };
    DecoderModel::build(&w)
}

#[test]
fn argmax_takes_the_lowest_index_on_a_tie() {
    assert_eq!(argmax_codes(&[3, 7, 7, 1]), 1);
    assert_eq!(argmax_codes(&[-5, -2, -9]), 1);
    assert_eq!(argmax_codes(&[4]), 0);
}

#[test]
fn greedy_follows_the_cached_decode_path_exactly() {
    let m = tiny();
    let prompt = [1u32, 2, 3];
    let mut seen = Vec::new();
    let g = generate_greedy(&m, &prompt, 6, &[], |t| seen.push(t));
    assert_eq!(g.finish, Finish::Length);
    assert_eq!(g.tokens.len(), 6);
    assert_eq!(seen, g.tokens, "the callback sees every token, in order");
    // the same chain by hand: prompt through decode_step, then argmax and feed back
    let mut cache = m.new_cache();
    let mut l = None;
    for (p, &t) in prompt.iter().enumerate() {
        l = Some(m.decode_step(t as usize, &mut cache, p));
    }
    let mut manual = Vec::new();
    for p in prompt.len()..prompt.len() + 6 {
        let next = argmax_codes(&l.take().unwrap().data) as u32;
        manual.push(next);
        l = Some(m.decode_step(next as usize, &mut cache, p));
    }
    assert_eq!(g.tokens, manual);
    assert_eq!(g, generate_greedy(&m, &prompt, 6, &[], |_| {}), "deterministic, hash included");
}

#[test]
fn a_stop_id_ends_generation_and_is_not_emitted() {
    let m = tiny();
    let first = generate_greedy(&m, &[4, 5], 1, &[], |_| {}).tokens[0];
    let g = generate_greedy(&m, &[4, 5], 10, &[first], |_| {});
    assert_eq!((g.finish, g.tokens.len()), (Finish::Stop, 0));
    let shorter = generate_greedy(&m, &[4, 5], 3, &[], |_| {});
    let longer = generate_greedy(&m, &[4, 5], 5, &[], |_| {});
    assert_eq!(longer.tokens[..3], shorter.tokens[..], "a longer run extends a shorter one");
    assert_ne!(longer.logits_hash, shorter.logits_hash, "the hash covers every step");
}

#[test]
fn the_embedding_is_the_rounded_mean_of_the_hidden_state() {
    let m = tiny();
    let toks = [3usize, 1, 4, 1, 5];
    let e = embed_mean(&m, &toks);
    let h = m.hidden_int(&toks);
    assert_eq!(e.shape, vec![m.dim()]);
    assert_eq!(e.scale, h.scale);
    for c in 0..m.dim() {
        let sum: i64 = (0..toks.len()).map(|r| h.data[r * m.dim() + c] as i64).sum();
        assert!((e.data[c] as i64 * 5 - sum).abs() <= 2, "column {c}: |mean·n − sum| ≤ n/2");
    }
    assert_eq!(e.data, embed_mean(&m, &toks).data);
}
