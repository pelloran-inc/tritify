//! The decoder-only model assembly (dynamic quantization, no calibration).
//!
//! A plugin hands the engine a [`DecoderWeights`] (neutral float weights + [`DecoderConfig`], no HF naming).
//! [`DecoderModel::build`] quantizes the weights once (b=12) and assembles the ops — **no calibration pass**:
//! every activation is quantized dynamically per-tensor at run time. The model implements
//! [`CausalLM`]: `forward_int` runs the integer stack, `forward_ref` the float twin. The residual stream is
//! added in integer at a common scale and re-quantized dynamically — there is no single global residual scale.

use crate::attention::{Attention, LayerKv};
use crate::causal::CausalLM;
use crate::check::{CheckReport, VerifiableModel};
use crate::layers::{Embedding, MoeRoute, QuantLinear, RmsNorm, SwiGluMlp};
use crate::module::Module;
use crate::tensor::IntTensor;

/// Shape + hyperparameters of a decoder-only transformer.
#[derive(Clone, Debug)]
pub struct DecoderConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f64,
    pub rope_theta: f64,
    pub max_seq: usize,
}

/// Float weights of one transformer layer (row-major, Hugging Face conventions). `wq_b`/`wk_b`/`wv_b` are the
/// optional attention projection biases (present in Qwen2-style models, absent in Llama).
pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub wq_b: Option<Vec<f32>>,
    pub wk_b: Option<Vec<f32>>,
    pub wv_b: Option<Vec<f32>>,
    pub post_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

/// All float weights of a decoder-only model.
pub struct DecoderWeights {
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub cfg: DecoderConfig,
}

struct Block {
    input_norm: RmsNorm,
    attn: Attention,
    post_norm: RmsNorm,
    /// The feed-forward is **pluggable**: dense `SwiGluMlp`, a `MoeMlp` (Mixture-of-Experts), or any future FFN — the
    /// block body never knows which. `Send + Sync` so a built model can be shared read-only across pipeline threads.
    ffn: Box<dyn Module + Send + Sync>,
}

/// One pre-built decoder layer, handed to [`DecoderModel::assemble`] by a generic architecture plugin. This is the seam
/// that lets a family plugin build each layer however it needs — dense `SwiGluMlp` or a `MoeMlp` FFN, attention with or
/// without QK-norm — while the engine's forward never changes. The dense `DecoderModel::build` is itself just one
/// producer of these.
pub struct DecoderLayer {
    pub input_norm: RmsNorm,
    pub attn: Attention,
    pub post_norm: RmsNorm,
    /// The layer's feed-forward (dense or MoE); `Send + Sync` for read-only sharing across pipeline threads.
    pub ffn: Box<dyn Module + Send + Sync>,
}

/// A built exact-integer decoder-only model (weights quantized; activations quantized dynamically at runtime).
pub struct DecoderModel {
    emb: Embedding,
    blocks: Vec<Block>,
    final_norm: RmsNorm,
    lm_head: QuantLinear,
    dim: usize,
    vocab: usize,
}

/// Residual add — integer, bring both to a common `Scale`, add, dynamic-quant (the shared [`crate::layers::add`]).
use crate::layers::add as residual_add;

impl DecoderModel {
    /// Quantize the dense weights and assemble. No calibration: activation scales are dynamic per-tensor at runtime.
    /// This is the dense (Llama/Qwen/Mistral) producer; it builds each layer's parts and hands them to [`assemble`],
    /// so dense and MoE share the one engine path. Byte-identical to the previous direct construction.
    ///
    /// [`assemble`]: DecoderModel::assemble
    pub fn build(w: &DecoderWeights) -> DecoderModel {
        let cfg = &w.cfg;
        let emb = Embedding::from_weights("embed", &w.embed, cfg.vocab, cfg.dim);
        let layers: Vec<DecoderLayer> = w
            .layers
            .iter()
            .map(|lw| DecoderLayer {
                input_norm: RmsNorm::new("in_norm", &lw.input_norm, cfg.eps),
                attn: Attention::from_weights(
                    "attn",
                    cfg.dim,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    &lw.wq,
                    &lw.wk,
                    &lw.wv,
                    &lw.wo,
                    (lw.wq_b.as_deref(), lw.wk_b.as_deref(), lw.wv_b.as_deref()),
                    cfg.max_seq,
                    cfg.rope_theta,
                ),
                post_norm: RmsNorm::new("post_norm", &lw.post_norm, cfg.eps),
                ffn: Box::new(SwiGluMlp::new(
                    "mlp",
                    QuantLinear::from_weights("gate", &lw.gate, cfg.hidden, cfg.dim),
                    QuantLinear::from_weights("up", &lw.up, cfg.hidden, cfg.dim),
                    QuantLinear::from_weights("down", &lw.down, cfg.dim, cfg.hidden),
                )),
            })
            .collect();
        let final_norm = RmsNorm::new("final_norm", &w.final_norm, cfg.eps);
        let lm_head = QuantLinear::from_weights("lm_head", &w.lm_head, cfg.vocab, cfg.dim);
        DecoderModel::assemble(emb, layers, final_norm, lm_head, cfg.dim, cfg.vocab)
    }

    /// **The generic assembly seam.** Build a decoder from pre-constructed per-layer parts ([`DecoderLayer`]) plus the
    /// embedding, final norm and LM head. A family plugin (dense or MoE) uses this to assemble a model whose layers it
    /// built however the family requires — the engine's integer/float forwards are identical regardless. The resulting
    /// model is indistinguishable from one `build` produced for the same ops (same `Block`s), so dense runs stay
    /// byte-identical.
    pub fn assemble(
        emb: Embedding,
        layers: Vec<DecoderLayer>,
        final_norm: RmsNorm,
        lm_head: QuantLinear,
        dim: usize,
        vocab: usize,
    ) -> DecoderModel {
        let blocks = layers
            .into_iter()
            .map(|l| Block { input_norm: l.input_norm, attn: l.attn, post_norm: l.post_norm, ffn: l.ffn })
            .collect();
        DecoderModel { emb, blocks, final_norm, lm_head, dim, vocab }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of transformer blocks — the split points for a layer-wise distributed run are `0..=n_layers`.
    /// The longest sequence (prompt plus generated tokens) the model's position tables cover.
    pub fn max_positions(&self) -> usize {
        self.blocks.first().map_or(0, |b| b.attn.max_seq())
    }

    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    /// **Stage 1 of a split:** embed the tokens into the integer residual stream (the input to block 0).
    pub fn embed_int(&self, tokens: &[usize]) -> IntTensor {
        self.emb.lookup_seq(tokens)
    }

    /// **A worker's share:** run blocks `start..end` of the integer forward on an incoming residual-stream tensor
    /// `x` and return the residual stream after block `end-1`. Each block keeps the *whole* tensor, so splitting by
    /// layers changes nothing numerically — the output is bit-identical to running all blocks on one machine.
    pub fn blocks_int(&self, mut x: IntTensor, start: usize, end: usize) -> IntTensor {
        for b in &self.blocks[start..end] {
            let a = b.attn.forward_int(&b.input_norm.forward_int(&x));
            x = residual_add(&x, &a);
            let m = b.ffn.forward_int(&b.post_norm.forward_int(&x));
            x = residual_add(&x, &m);
        }
        x
    }

    /// **Final stage of a split:** the final norm and the LM head, turning the last residual stream into logits.
    pub fn head_int(&self, x: &IntTensor) -> IntTensor {
        self.lm_head.forward_int(&self.final_norm.forward_int(x))
    }

    /// The final-normed hidden state of a whole-sequence forward, `[seq, dim]`: what the LM head reads. Embeddings
    /// pool it ([`crate::generate::embed_mean`]).
    pub fn hidden_int(&self, tokens: &[usize]) -> IntTensor {
        self.final_norm.forward_int(&self.blocks_int(self.embed_int(tokens), 0, self.blocks.len()))
    }

    /// **Full-model expert-parallel forward.** Run the whole integer forward, but compute every MoE layer's selected
    /// experts through `experts_fn(layer, &expert_ids, &row_activation)` — returning the per-expert outputs in `ids`
    /// order — instead of the local experts (e.g. backed by remote workers holding expert shards). Dense layers run
    /// locally and unchanged. With the local backend this is byte-identical to `forward_int`, so a distributed run
    /// reproduces the single-host logits bit-for-bit (the MoE `combine` gather is placement-independent). Returns logits.
    pub fn forward_int_remote_moe(
        &self,
        tokens: &[usize],
        experts_fn: impl FnMut(usize, &[usize], &IntTensor) -> Vec<IntTensor>,
    ) -> IntTensor {
        self.forward_int_remote_moe_observed(tokens, experts_fn, |_, _| {})
    }

    /// [`DecoderModel::forward_int_remote_moe`] that also hands `on_layer(i, &x)` the residual stream after block `i`
    /// (for per-layer hashes of a live view). The observer only reads, so the logits are the same integers;
    /// `blocks_int(embed_int(tokens), 0, i + 1)` is the single-host stream at the same point.
    pub fn forward_int_remote_moe_observed(
        &self,
        tokens: &[usize],
        mut experts_fn: impl FnMut(usize, &[usize], &IntTensor) -> Vec<IntTensor>,
        mut on_layer: impl FnMut(usize, &IntTensor),
    ) -> IntTensor {
        let mut x = self.embed_int(tokens);
        for (i, b) in self.blocks.iter().enumerate() {
            let a = b.attn.forward_int(&b.input_norm.forward_int(&x));
            x = residual_add(&x, &a);
            let h = b.post_norm.forward_int(&x);
            let m = match b.ffn.as_moe() {
                Some(moe) => moe.forward_int_with_experts(&h, |ids, row| experts_fn(i, ids, row)),
                None => b.ffn.forward_int(&h),
            };
            x = residual_add(&x, &m);
            on_layer(i, &x);
        }
        self.head_int(&x)
    }

    /// Run the integer **prefill** forward and record, for every layer whose FFN routes (MoE), the selected experts
    /// **per token** — `trace[i]` is layer `i`'s `Vec<per-token Vec<expert-id>>` (dense layers contribute nothing).
    /// The activations entering each router are the real forward activations (this mirrors `blocks_int(0..n)` exactly,
    /// adding only the read-only route hook). For routing-dynamics / expert-cache analysis, not the run-time path.
    pub fn forward_int_traced(&self, tokens: &[usize]) -> Vec<Vec<Vec<usize>>> {
        let mut x = self.embed_int(tokens);
        let mut trace = Vec::new();
        for b in &self.blocks {
            let a = b.attn.forward_int(&b.input_norm.forward_int(&x));
            x = residual_add(&x, &a);
            let h = b.post_norm.forward_int(&x);
            if let Some(r) = b.ffn.route_trace(&h) {
                trace.push(r);
            }
            let m = b.ffn.forward_int(&h);
            x = residual_add(&x, &m);
        }
        trace
    }

    /// [`DecoderModel::forward_int_traced`] with each token's whole routing decision — its experts, their numerators and
    /// the normalizer (the routing weights a run used, for a non-degeneracy check). `routes[i]` is the `i`-th
    /// routed layer's decisions, one per token. Mirrors `blocks_int` and reads only.
    pub fn forward_int_routes(&self, tokens: &[usize]) -> Vec<Vec<MoeRoute>> {
        let dim = self.dim();
        let mut x = self.embed_int(tokens);
        let mut out = Vec::new();
        for b in &self.blocks {
            let a = b.attn.forward_int(&b.input_norm.forward_int(&x));
            x = residual_add(&x, &a);
            let h = b.post_norm.forward_int(&x);
            if let Some(moe) = b.ffn.as_moe() {
                out.push(
                    (0..h.len() / dim)
                        .map(|t| {
                            moe.route(&IntTensor::new(h.data[t * dim..(t + 1) * dim].to_vec(), vec![dim], h.scale))
                        })
                        .collect(),
                );
            }
            let m = b.ffn.forward_int(&h);
            x = residual_add(&x, &m);
        }
        out
    }

    /// A fresh per-layer KV cache for incremental decode (one [`LayerKv`] per block).
    pub fn new_cache(&self) -> Vec<LayerKv> {
        let kstride = self.blocks.first().map_or(0, |b| b.attn.n_kv_heads * b.attn.head_dim);
        (0..self.blocks.len()).map(|_| LayerKv::new(kstride)).collect()
    }

    /// **Incremental decode of ONE token** at absolute position `pos` against the KV `cache`: embed the token, run
    /// every block with the cached attention (`Attention::decode_step`), and return the logits for this position.
    /// O(cache) per step, not O(n²). Deterministic and split-reproducible (the attention is an exact integer
    /// function of the cache); it is *not* required to equal a whole-sequence prefill of the same tokens (that is
    /// the dynamic-quant prefill/decode non-invariance — decode has its own internally consistent semantics).
    pub fn decode_step(&self, token: usize, cache: &mut [LayerKv], pos: usize) -> IntTensor {
        let x = self.embed_int(&[token]);
        let x = self.decode_blocks(x, cache, 0, self.blocks.len(), pos);
        self.head_int(&x)
    }

    /// [`DecoderModel::decode_step`] with every MoE layer's routed experts computed by `experts_fn(layer, ids, row)`
    /// (distributed generation). Dense layers and each MoE layer's shared expert run here; each shared-expert
    /// execution is reported to `on_shared(layer, rows, elapsed)`, and `on_layer(i, &x)` sees the residual stream after
    /// block `i`. With a local `experts_fn` the logits are `decode_step`'s, bit for bit, for the reason
    /// `forward_int_with_experts` is `forward_int`; the observers only read.
    pub fn decode_step_remote_moe(
        &self,
        token: usize,
        cache: &mut [LayerKv],
        pos: usize,
        mut experts_fn: impl FnMut(usize, &[usize], &IntTensor) -> Vec<IntTensor>,
        mut on_shared: impl FnMut(usize, usize, std::time::Duration),
        mut on_layer: impl FnMut(usize, &IntTensor),
    ) -> IntTensor {
        let mut x = self.embed_int(&[token]);
        for (i, (b, lkv)) in self.blocks.iter().zip(cache.iter_mut()).enumerate() {
            let a = b.attn.decode_step(&b.input_norm.forward_int(&x), lkv, pos);
            x = residual_add(&x, &a);
            let h = b.post_norm.forward_int(&x);
            let m = match b.ffn.as_moe() {
                Some(moe) => moe.forward_int_with_experts_observed(
                    &h,
                    |ids, row| experts_fn(i, ids, row),
                    |rows, d| on_shared(i, rows, d),
                ),
                None => b.ffn.forward_int(&h),
            };
            x = residual_add(&x, &m);
            on_layer(i, &x);
        }
        self.head_int(&x)
    }

    /// Layer `layer`'s MoE feed-forward, or `None` for a dense layer: what a distributed run routes and combines.
    pub fn moe(&self, layer: usize) -> Option<&crate::layers::MoeMlp> {
        self.blocks.get(layer).and_then(|b| b.ffn.as_moe())
    }

    /// **A worker's share of a decode step:** run blocks `start..end` with their KV caches (`cache` is aligned to
    /// those blocks, i.e. `cache.len() == end-start`) on the incoming one-token residual stream `x`, at absolute
    /// position `pos`. Splitting the decode by layers is bit-identical to one machine — each block keeps the whole
    /// (one-token) tensor and its own cache.
    pub fn decode_blocks(
        &self,
        mut x: IntTensor,
        cache: &mut [LayerKv],
        start: usize,
        end: usize,
        pos: usize,
    ) -> IntTensor {
        for (b, lkv) in self.blocks[start..end].iter().zip(cache.iter_mut()) {
            let a = b.attn.decode_step(&b.input_norm.forward_int(&x), lkv, pos);
            x = residual_add(&x, &a);
            let m = b.ffn.forward_int(&b.post_norm.forward_int(&x));
            x = residual_add(&x, &m);
        }
        x
    }
}

impl CausalLM for DecoderModel {
    fn vocab(&self) -> usize {
        self.vocab
    }

    fn forward_int(&self, tokens: &[usize]) -> IntTensor {
        // The single-machine path IS the all-blocks split (embed -> blocks[0..n] -> head), so a layer-wise
        // distributed run through the same three methods is bit-identical by construction.
        let x = self.embed_int(tokens);
        let x = self.blocks_int(x, 0, self.blocks.len());
        self.head_int(&x)
    }

    fn forward_ref(&self, tokens: &[usize]) -> Vec<f32> {
        let mut x: Vec<f32> = self.emb.lookup_seq(tokens).to_f32();
        for b in &self.blocks {
            let h = b.input_norm.forward_ref(&x);
            let a = b.attn.forward_ref(&h);
            for (xi, ai) in x.iter_mut().zip(&a) {
                *xi += *ai;
            }
            let h2 = b.post_norm.forward_ref(&x);
            let m = b.ffn.forward_ref(&h2);
            for (xi, mi) in x.iter_mut().zip(&m) {
                *xi += *mi;
            }
        }
        self.lm_head.forward_ref(&self.final_norm.forward_ref(&x))
    }
}

/// A decoder is self-verifying: run a fixed probe token sequence through both the integer and float forwards,
/// compare the argmax streams, and hash the integer logits.
impl VerifiableModel for DecoderModel {
    fn task(&self) -> &str {
        "causal-lm"
    }
    fn self_check(&self) -> CheckReport {
        let v = self.vocab();
        let probe: Vec<usize> = (0..8).map(|i| i % v.max(1)).collect();
        let li = self.forward_int(&probe).to_f32();
        let lr = self.forward_ref(&probe);
        let max_abs_err = li.iter().zip(&lr).fold(0.0f64, |a, (&x, &y)| a.max((x as f64 - y as f64).abs()));
        let (ai, ar) = (self.argmax_int(&probe), self.argmax_ref(&probe));
        let agree = ai.iter().zip(&ar).filter(|(a, b)| a == b).count();
        let hash = self.logits_hash(&probe);
        let deterministic = hash == self.logits_hash(&probe);
        CheckReport {
            task: self.task().into(),
            metric_name: "argmax_agreement".into(),
            metric: agree as f64 / probe.len() as f64,
            max_abs_err,
            deterministic,
            hash,
            detail: format!("{}/{} argmax positions == float", agree, probe.len()),
        }
    }
    fn fidelity(&self, positions: usize) -> Option<(f64, String)> {
        let v = self.vocab();
        let probe: Vec<usize> = (0..positions).map(|i| i % v.max(1)).collect();
        let (ai, ar) = (self.argmax_int(&probe), self.argmax_ref(&probe));
        let agree = ai.iter().zip(&ar).filter(|(a, b)| a == b).count();
        Some((agree as f64 / positions as f64, format!("{agree}/{positions} argmax positions == float")))
    }
}
