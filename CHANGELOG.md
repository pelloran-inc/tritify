# Changelog

All notable changes to tritify are listed here, newest first. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A release's version is its git tag. How a release is
made: [RELEASING.md](RELEASING.md).

## [Unreleased]

## [0.0.2] - 2026-11-09

### Added

- **Check bit-identity yourself, position by position.** `verify_repro`, an example of `kt-convert`:
  - It runs one request on every compute backend your machine has, compares the logits at every position, and
    names the first position where anything differs.
  - A difference ends with `VERDICT: MISMATCH` and exit code 1.
- **A shipped reference** for Qwen2.5-0.5B-Instruct:
  `crates/04-ingest/kt-convert/references/qwen2.5-0.5b-instruct.repro.json`. It was written on an ARM Mac and checked
  on x86.
  - `verify_repro <qwen> --check <reference>` checks your machine against it.
  - A reference is bound to the checkpoint's files and the input's tokens. A reference for another checkpoint or
    another input is refused (exit 2).
- **Your own input:**
  - `--prompt FILE --reference-out FILE` writes a reference for your own text on one machine.
  - `--prompt FILE --check FILE` checks it on another machine.
- **No download needed:** `verify_repro --synthetic` runs a small seeded model.
- **Checks of the checker:**
  - `--perturb POS` flips one bit at position POS, which must be reported there.
  - `--plant` adds deliberately wrong CPU partitions, which must be found.
- `kt_convert::repro`:
  - the reference format `tritify-repro/1`
  - per-position logits hashes over the integer scale and every integer code
  - the comparison
- **README, "Check it position by position", with what was measured.** On Qwen2.5-0.5B-Instruct, every backend
  gave the same logits at every position, equal to the shipped reference:
  - on an Apple M3 Max: the portable CPU path, NEON, and the whole model on the GPU (Metal)
  - on an AMD EPYC: the portable CPU path, AVX2 and AVX-512
- REPRODUCE.md: `8109ebad…` (the seeded model) and the `d0f296c4…` reference check.
- `kt-server`: `ServeOptions`, with `router_with` and `serve_with`, to configure a server from the library.

### Changed

- **Safer serving defaults in `kt-server`,** following a community review:
  - **The runtime switch is off by default.** `POST /kt/v1/runtime` sets the compute backend and the thread cap for
    the whole process, so for every client.
    - A server now answers it with 403 (`runtime_switch_disabled`) unless it was started with
      `--allow-runtime-switch`.
    - `GET /kt/v1/runtime` reports `switch_enabled`.
  - A warning when `--listen` is not a loopback address.
  - **README, "Serving / security":**
    - The server binds loopback, with no authentication, by default.
    - Put an authenticating reverse proxy with TLS in front of it before exposing it.
  - **Upgrading:** if you use the runtime switch, start `kt-server` with `--allow-runtime-switch`, and only where
    every client is trusted.

## [0.0.1] - 2026-11-07

### Added

- **The first public release.** tritify runs the forward pass of decoder-only language models, Mixture-of-Experts
  decoders and the Segment Anything 2.1 image/video stack in exact integer arithmetic. The output is bit-identical
  run to run and machine to machine, carried by a reproducing SHA-256.
- **The workspace:**
  - `kt-int`, `kt-r16`: exact-integer kernels
  - `kt-core`: the op contract (an integer path and a float reference) and the verify gate
  - `kt-runtime`, `kt-gpu`, `kt-cuda`: CPU partitioning and the GPU backends
  - `kt-model-decoder`, `kt-model-moe`, `kt-model-vision`: the model-family plugins
  - `kt-convert`: checkpoint conversion and the self-check harnesses
  - `kt-server`: an OpenAI-compatible inference server
  - `kt-demo`: the 60-second determinism demonstration
- **REPRODUCE.md:** every hash the README and the whitepaper cite, with its model, revision and command.
- **FLAGSHIP_RUN.md and `flagship_run.json`:** the comparison report for the GLM-4.5-Air run.
- **`ci/gates.sh`:** the repository's CI gates, runnable on any host.

[Unreleased]: https://github.com/pelloran-inc/tritify/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/pelloran-inc/tritify/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/pelloran-inc/tritify/releases/tag/v0.0.1
