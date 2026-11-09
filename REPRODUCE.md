# Reproduce

This file lists every hash that this repository's README and the tritify whitepaper cite. For each one it gives the
full SHA-256, the model and its pinned revision, and the command that prints it. Run the commands from the repository
root. `rust-toolchain.toml` pins Rust 1.90.0.

Where a command runs every compute backend your host has, every line it prints must carry the same digest. The times
are for one Apple M3 Max, after the first build.

## Without a model

| Hash | Full SHA-256 | What it is | Command |
|---|---|---|---|
| `de67ba75…` | `de67ba758bd0d1fb574fa64d7ff326e9d17f536d7725659228bebddcc6478bfa` | A ternary GEMM, 64 × 768 × 768, on the GPU against the CPU | `cargo run --release -p kt-demo` |
| `781285bb…` | `781285bb99bcd9cb5233f708466676db41e717b6971ce70508b57ed291478352` | The backend probe's fixed corpus, on every CPU variant and GPU | `cargo run --release -p kt-runtime --example runtime_probe` |
| `8109ebad…` | `8109ebad0a2883838a9d133351ce421f676e8d66dbcaffc5cde000272b748d03` | A small seeded decoder, 32 greedy tokens, every position on every backend | `cargo run --release -p kt-convert --example verify_repro -- --synthetic` |

## Public checkpoints, pinned

Download every file of the pinned revision into one directory, for example:

`huggingface-cli download <repository> --revision <revision> --local-dir <dir>`

The README shows the same download with `curl`.

| Model | Hugging Face repository | Revision |
|---|---|---|
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| OLMoE-1B-7B-0924 | `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` |
| SAM 2.1 Hiera base+ | `facebook/sam2.1-hiera-base-plus` | `b7320756a13354e7530a63935656d35b2f91a290` |

| Hash | Full SHA-256 | What it is | Command (`<qwen>`, `<olmoe>`, `<sam>`: the directories) | Time |
|---|---|---|---|---|
| `d0f296c4…` | `d0f296c46277f1d078562135fe80847feb8497e33cc05be13fee1466d73e5c65` | Qwen: one fixed chat request, the logits hash, on every backend | `cargo run --release -p kt-convert --example generate_backends -- <qwen> 32` | 18 s |
| `d0f296c4…` | the same | The same request, every position's logits on every backend, against the shipped reference (written on ARM, checked on x86) | `cargo run --release -p kt-convert --example verify_repro -- <qwen> --check crates/04-ingest/kt-convert/references/qwen2.5-0.5b-instruct.repro.json` | 15 s |
| `98166930…` | `98166930a8af7026667db5d4ada8d3aae05527f4a25949c8dc27bbb26346558d` | The same command's fixed embedding input, its hash | the same command | — |
| `6c139feb…` | `6c139feb57ed595215cf1b1ef72f760d89789a5041efd946de0f550d6f891d4b` | Qwen end to end over 512 tokens: one thread and many give the same bytes | `cargo run --release -p kt-convert --example verify_lm_e2e -- <qwen> 512` | 5 min |
| `69b33985…` | `69b33985232a40d733cce739e78e61834c68726785d0edefde8a7520f1d8e334` | Qwen, converted and checked against its float reference | `cargo run --release -p kt-convert -- <qwen>` | 17 s |
| `505cb108…` | `505cb1081db7c1ab3970af8fd93a6aa70392e972674c6848b778dc2f781f03b9` | OLMoE, its first 2 layers, on every backend | `cargo run --release -p kt-convert --example anchor_backends -- <olmoe> 2` | 30 s |
| `74005166…` | `740051663af23d81a08366cfd1a657388b39c1b1d7d87b802d618e365a69d804` | OLMoE, all 16 layers, on every backend | `cargo run --release -p kt-convert --example anchor_backends -- <olmoe> all` | 4 min |
| `6ae7e544…` | `6ae7e5447694deae9968e3445591ab7ba5aa68954957d1e5cf51982b3dc3fe60` | As `505cb108…`, with the RMSNorm epsilon at `1e-6`, the engine's earlier fallback (see below) | `… anchor_backends -- <olmoe> 2 --rms-norm-eps 1e-6` | 30 s |
| `3dadb4f1…` | `3dadb4f14a593213ab238dfb1dd8f168a29d55c7455e4913a29115409b86983f` | As `74005166…`, with the epsilon at `1e-6` | `… anchor_backends -- <olmoe> all --rms-norm-eps 1e-6` | 3 min |
| `7177a5f5…` | `7177a5f5239b39aeaeaf69d5af4270ce322693c4547a0e4637d298536171dbb6` | SAM 2.1, converted and checked against its float reference (the mask) | `cargo run --release -p kt-convert -- <sam>` | 6 min |

**Why there are two OLMoE epsilons.** OLMoE's `config.json` sets no `rms_norm_eps`, so transformers uses `1e-5`, and
so does the engine. An earlier engine fell back to `1e-6`. The hashes `6ae7e544…` and `3dadb4f1…` are that model's,
and the overlay reproduces them exactly. Bit-identity holds for both.

## Not reproducible from this repository alone

| Hash | Full SHA-256 | What it is |
|---|---|---|
| `564f495c…` | `564f495cea735728c220a3c2a6bd64ae4937fb105e2625fb155970876b47f741` | The GLM-4.5-Air flagship run: the hash of its generation, the same in the distributed run and in the single-host streamed reference |
| `67bd34b3…` | `67bd34b30d2fae8198897a04449d14cf5b1b2dda8bfa78e3efe2524cb9f2b7bc` | The same run's output, distributed against single host |

The run spread GLM-4.5-Air over three machines. That is 106 B parameters, 427 GB of integer codes, more than any one
of them holds in RAM. The machines were coordinated by the orchestration that Pelloran offers separately.

This repository holds:
- the engine
- the expert seam the run computed through (`forward_int_with_experts`)
- the single-host path

It does not hold the orchestration.

The run's provenance is its sealed telemetry file, sha256
`e85f4f36031055fa2ee288c14bf46a1cedd142276c5c2f839a7efeb06fba6e66`, which Pelloran keeps with the run's record.

**What you can check here: [`FLAGSHIP_RUN.md`](FLAGSHIP_RUN.md).** It is a comparison report published by
Pelloran, with matching hashes. Independent verification requires recomputation or additional accessible
artifacts. The report covers:
- all 46 layers, the distributed run's hash beside the single-host reference's, equal at every layer
- the two final hashes above
- the machines, by public name
- the run's totals

[`flagship_run.json`](flagship_run.json) holds the same chain in machine-readable form.
