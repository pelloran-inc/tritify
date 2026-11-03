//! The distributed decode step is the local one: `decode_step_remote_moe` with the model's own experts gives
//! `decode_step`'s logits bit for bit, step after step, for a softmax MoE (OLMoE-shaped) and for GLM-4.5's layout (a
//! sigmoid router, a shared expert, a dense first layer); a whole generation through `generate_greedy_by` is
//! `generate_greedy`'s; and the shared expert reports exactly one execution per MoE layer per step, never without one.

use super::*;
use crate::tests::MockMoe;
use crate::tests_glm::MockGlm;
use kt_core::generate::{generate_greedy, generate_greedy_by};
use kt_core::IntTensor;

/// The model's own experts, as a worker would compute them.
fn local(model: &DecoderModel) -> impl FnMut(usize, &[usize], &IntTensor) -> Vec<IntTensor> + '_ {
    |layer, ids, row| ids.iter().map(|&e| model.moe(layer).expect("a MoE layer").experts[e].forward_int(row)).collect()
}

fn check_steps(model: &DecoderModel, moe_layers_with_shared: usize) {
    let (mut a, mut b) = (model.new_cache(), model.new_cache());
    for (pos, token) in [3usize, 7, 1, 12, 5, 9].into_iter().enumerate() {
        let want = model.decode_step(token, &mut a, pos);
        let mut shared = Vec::new();
        let got = model.decode_step_remote_moe(
            token,
            &mut b,
            pos,
            local(model),
            |layer, rows, _| shared.push((layer, rows)),
            |_, _| {},
        );
        assert_eq!((want.data, want.scale), (got.data, got.scale), "step at pos {pos}");
        assert_eq!(shared.len(), moe_layers_with_shared, "one shared execution per MoE layer with a shared expert");
        assert!(shared.iter().all(|&(l, rows)| rows == 1 && model.moe(l).is_some_and(|m| m.shared_expert().is_some())));
    }
}

#[test]
fn the_remote_decode_step_is_the_local_one_for_a_softmax_moe() {
    let model = MoeArch::build_decoder_moe(&MockMoe::tiny("OlmoeForCausalLM")).expect("build");
    check_steps(&model, 0);
}

#[test]
fn the_remote_decode_step_is_the_local_one_for_glm_with_a_dense_layer_and_a_shared_expert() {
    let model = MoeArch::build_decoder_moe(&MockGlm::glm()).expect("build");
    assert!(model.moe(0).is_none(), "layer 0 is dense");
    check_steps(&model, 2);
}

#[test]
fn a_distributed_generation_is_the_local_generation() {
    let model = MoeArch::build_decoder_moe(&MockGlm::glm()).expect("build");
    let prompt = [2u32, 5, 11, 4];
    let want = generate_greedy(&model, &prompt, 8, &[], |_| {});
    let mut cache = model.new_cache();
    let mut experts = local(&model);
    let got = generate_greedy_by(
        &prompt,
        8,
        &[],
        |_| {},
        |t, pos| model.decode_step_remote_moe(t, &mut cache, pos, &mut experts, |_, _, _| {}, |_, _| {}),
    );
    assert_eq!(want, got);
    assert_eq!(want.tokens.len(), 8);
}

/// The router-only build holds every tensor but the routed experts; with the experts computed elsewhere it is
/// the whole model, token for token and hash for hash.
#[test]
fn the_router_only_model_with_the_experts_elsewhere_is_the_whole_model() {
    let full = MoeArch::build_decoder_moe(&MockGlm::glm()).expect("build");
    let ro = MoeArch::build_decoder_moe_router_only(&MockGlm::glm()).expect("router-only build");
    assert!(ro.moe(0).is_none() && ro.moe(1).is_some_and(|m| m.is_remote() && m.shared_expert().is_some()));
    let prompt = [2u32, 5, 11, 4];
    let want = generate_greedy(&full, &prompt, 8, &[], |_| {});
    let mut cache = ro.new_cache();
    let mut experts = local(&full);
    let got = generate_greedy_by(
        &prompt,
        8,
        &[],
        |_| {},
        |t, pos| ro.decode_step_remote_moe(t, &mut cache, pos, &mut experts, |_, _, _| {}, |_, _| {}),
    );
    assert_eq!(want, got);
}

#[test]
#[should_panic(expected = "the routed experts are remote")]
fn a_router_only_model_does_not_run_alone() {
    let ro = MoeArch::build_decoder_moe_router_only(&MockGlm::glm()).expect("router-only build");
    let _ = ro.decode_step(1, &mut ro.new_cache(), 0);
}
