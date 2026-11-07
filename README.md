<div align="center">

# tritify

### The same bytes on every machine — real models, run in exact integer arithmetic.

<p>
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0">
  <img src="https://img.shields.io/badge/rust-1.90.0%20(pinned)-orange.svg" alt="Rust 1.90.0, pinned">
  <img src="https://img.shields.io/badge/integer%20path-bit--identical-brightgreen.svg" alt="Integer path: bit-identical">
  <img src="https://img.shields.io/badge/vs%20float-within%20tolerance-blue.svg" alt="Against float: within a declared tolerance">
</p>

**Clone it, run it, verify the hash.**

</div>

---

<div align="center">
  <img src="assets/flagship-demo.gif" width="860" alt="GLM-4.5-Air generated bit-identically across a fleet">
  <br>
  <sub><b>At scale:</b> GLM-4.5-Air — 106 billion parameters, 427 GB of integer codes — generated its answer across <b>three machines, none of which can hold it in RAM</b>, bit-for-bit identical to a single machine. Same SHA-256 on every node. <i>(A replay. This run used the orchestration Pelloran offers separately and 427 GB of codes, so it cannot be re-run from this repository alone: <a href="FLAGSHIP_RUN.md">FLAGSHIP_RUN.md</a> is Pelloran's comparison report for it, with matching hashes at every layer, and <a href="REPRODUCE.md">REPRODUCE.md</a> lists everything you can re-run here.)</i></sub>
</div>

---

`tritify` runs the whole forward pass of real models — decoder-only language models,
Mixture-of-Experts decoders, and the Segment Anything 2.1 image/video stack — in **exact integer
arithmetic**. The forward pass of the language models, the Mixture-of-Experts models and SAM 2.1's
image path computes on integers only.

Floating point remains in:
- the float reference the gate checks against
- the offline quantization of the weights
- read-only diagnostics
- **two named places on a model's path.** A float input (an image's normalized pixels, a prompt point)
  is quantized once at the entry. SAM 2.1's video tracker keeps its object pointers and their temporal
  encoding in float between frames.

The output is **bit-identical run to run and machine to machine**, carried by a reproducing **SHA-256**
you can check yourself.

## Two claims, kept apart

| | What it says | How you check it |
|---|---|---|
| **1 · Bit-identity** | The integer path's output **bytes** are identical run to run and machine to machine: across CPU instruction sets, thread counts, and CPU against GPU, at a fixed configuration. | Run the same command anywhere and compare one SHA-256. |
| **2 · Fidelity** | The integer path makes the float model's **decisions** (tokens, masks); its numbers are close to float, **not equal** to it. Each op ships an integer path and a float reference, and a gate refuses an op whose gap exceeds a declared tolerance. | `VerifyReport::passes(tol)`: the largest absolute difference is at most `tol`, and a repeat gives the same bytes. The ops' tests use 1% of the output's peak (`1e-2`). |

Claim 1 does not rest on claim 2: the bytes are exact and reproducible whatever their distance to float.

Optimized floating-point inference is usually *not* reproducible: the same model on different
hardware, compilers, or thread counts can produce different bits, because IEEE-754 addition is not
associative and platform math libraries round transcendental functions differently. `tritify`
removes that nondeterminism from the inference path — so a result can be **regenerated and checked**,
not merely trusted.

## Quick start

```sh
git clone https://github.com/pelloran-inc/tritify
cd tritify
cargo test --workspace            # every op: the integer path, checked against its float reference
cargo run --release -p kt-demo    # a zero-float ternary GEMM; GPU == CPU; prints a reproducing SHA-256
```

```text
  gpu        : Metal / Apple M3 Max
    tiled i32  : BIT-IDENTICAL to CPU ✓
    emul. i64  : BIT-IDENTICAL to CPU ✓  (lossless, any device)
  reproducing hash (sha256 of the output): de67ba758bd0d1fb574fa64d7ff326e9d17f536d7725659228bebddcc6478bfa
  → run again, or on another computer: the hash is identical. No float. No trick.
```

Run it again, or on another machine — **the digest is the same.**

GPU tests need a GPU adapter: on a host without one they say so and check nothing, so the CPU's
bit-identity is the gate. `bash ci/gates.sh` runs the repository's full CI gates on any host — format,
lints, build, tests, the backend probe, API docs and a secret scan — exactly as
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) does.

## Verify a real model in a minute

A public model, pinned to one Hugging Face revision, one command, one SHA-256. The download (about
1 GB) and the first build take longer; the run takes seconds.

```sh
REV=7ae557604adf67be50417f59c2c2f167def9a775   # Qwen/Qwen2.5-0.5B-Instruct
mkdir -p qwen
for f in config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json; do
  curl -sSfL -o "qwen/$f" "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/$REV/$f"
done
shasum -a 256 qwen/model.safetensors   # fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
cargo run --release -p kt-convert --example generate_backends -- qwen 32
```

It answers one fixed chat request on every compute backend your machine has, and compares them. Every
line must carry this logits hash. The program ends with `VERDICT: all <n> backends identical`. Any
difference ends it with `VERDICT: MISMATCH` and exit code 1.

```text
d0f296c46277f1d078562135fe80847feb8497e33cc05be13fee1466d73e5c65
```

[`REPRODUCE.md`](REPRODUCE.md) lists every hash this README and the whitepaper cite, in full, with its
model, revision and command.

## What we claim — and what you can check

Three precise claims, each verified; none conflated.

### 1 · Reproducible across hardware
The integer path produces the **same output bits** run to run, across CPU instruction-set
architectures, and across CPU and GPU, at a fixed numeric configuration and execution schedule —
carried by a reproducing SHA-256.

### 2 · Faithful to the float reference — with no cliff at depth
On the models and inputs tested, the integer path reproduces the floating-point model's *decision*;
the numeric exceptions are **measured, not assumed away**. Every op ships both an integer path and a
float reference, and a gate refuses anything the two disagree on beyond a declared tolerance. Measured
through sixteen layers of two models, the per-layer agreement never collapses:

<div align="center">
  <img src="assets/fidelity.png" width="680" alt="Per-layer fidelity: no cliff">
</div>

### 3 · Independent of where it runs
Because the output is a deterministic function of (weights, inputs, schedule) and is bit-identical
wherever each operation runs, a model's computation can be **split across processors and machines
without changing a single output bit**.

## <a name="verifiable-distributed-inference"></a>Verifiable distributed inference

A model larger than any one machine can run across a fleet, each node holding only a shard of the
experts, and reproduce the single-machine answer **exactly**. For the flagship run above, that is a
comparison report published by Pelloran with matching hashes ([`FLAGSHIP_RUN.md`](FLAGSHIP_RUN.md));
independent verification requires recomputation or additional accessible artifacts. A fast,
untrusted fleet can run the model while an open reference recomputes the exact bytes and checks them.

<div align="center">
  <img src="assets/capacity.png" width="720" alt="No single node holds the model">
  <br>
  <sub>GLM-4.5-Air's 427 GB of codes exceed every node's RAM — yet the fleet runs it, bit-identically.</sub>
</div>

The coordinator routes each token locally and sends only the token's **activation** to the worker
holding each selected expert, which returns the expert's **output vector**; expert *weights* stay
resident and never cross the wire.

<div align="center">
  <img src="assets/routing_illus.png" width="720" alt="The distributed seam">
</div>

## Deterministic routing introspection

Because routing is exact and reproducible, you can read off **which experts fire** for any input —
exactly, on any machine (optimized float implementations can route differently near ties). A bounded workload reuses a small set of
experts; a diverse one spreads across most of them:

<div align="center">
  <img src="assets/heatmap.png" width="760" alt="Which experts fire: bounded vs diverse workload">
</div>

## What we do *not* claim

- **Not faster** on a single device — the integer path is not a throughput win over optimized float;
  the win is determinism, portability, and capacity.
- **Not confidential** — verifiable is not private.
- **Not a quality improvement** — agreement with the float model on the tested models and inputs; no
  general quality guarantee.

The value is **exactness, determinism, portability, and, above all, verifiability.**

## Layout

A layered workspace — each layer depends only on lower-numbered ones:

| Layer | Crates | What |
|---|---|---|
| `00-kernels` | `kt-int`, `kt-r16` | exact-integer kernels; the R16 lossless balanced-ternary container |
| `01-engine` | `kt-core` | the op contract (integer path + float reference), the verify gate, the model/plugin seams, the read-only routing trace |
| `02-runtime` | `kt-runtime`, `kt-gpu`, `kt-cuda` | CPU partitioning and the portable GPU backend |
| `03-models` | `kt-model-decoder`, `kt-model-moe`, `kt-model-vision` | the model-family plugins |
| `04-ingest` | `kt-convert` | checkpoint conversion and the self-check harnesses |
| `06-server` | `kt-server` | an OpenAI-compatible inference server |
| `apps` | `kt-demo` | the 60-second determinism demonstration |

## Running your own model

```sh
# Convert a Hugging Face checkpoint and run the integer forward, checked against the float reference:
cargo run --release -p kt-convert -- <path/to/checkpoint>
# Serve it on an OpenAI-compatible endpoint:
cargo run --release -p kt-server -- --model <path/to/checkpoint>
```

### Serving / security

- **The defaults:** `kt-server` binds the loopback address, `127.0.0.1:8080`, and has **no authentication**.
  - As it is, it is for one machine.
  - On any other `--listen` address it prints a warning.
- **Before exposing it, put an authenticating reverse proxy in front of it.** For example nginx or Caddy, with
  authentication and TLS, forwarding to the loopback port.
- **The runtime switch is process-global, and off by default.**
  - `POST /kt/v1/runtime` installs the compute backend and sets the thread cap for the whole process, so for every
    client.
  - A server answers it with **403** (`runtime_switch_disabled`) unless it was started with
    `--allow-runtime-switch`.
  - Enable it only where every client is trusted. Keep it off on any exposed server.
  - A switch never changes an answer's bits: every backend gives the engine's integers. But it changes speed,
    resources and availability for everyone.
- `GET /kt/v1/runtime` only reads, and reports `switch_enabled`.

Decoder-only LMs, Mixture-of-Experts decoders, and SAM 2.1 are supported out of the box.

## Scope

`tritify` is a self-contained inference engine. Pelloran Inc. separately offers the system built on
it — orchestration for distributed and heterogeneous deployment, and verifiable-inference services —
and the independent **Generation 5 Embedding** retrieval product. Those are separate; the engine and
its reproducibility, fidelity, and bit-identity properties stand on their own and depend on none of
them.

## License

Apache-2.0 © Pelloran Inc. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
