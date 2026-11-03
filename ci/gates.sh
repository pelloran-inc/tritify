#!/usr/bin/env bash
# The CI gates without GitHub Actions: runs exactly `.github/workflows/ci.yml`'s two jobs, `rust` and
# `secret-scan`, on this host. From a full clone:
#
#     bash ci/gates.sh [all|rust|secret-scan]        (default: all)
#
# It installs nothing. Every missing prerequisite is named, and the run stops before any gate:
#   - rustup with toolchain 1.90.0, clippy and rustfmt:  rustup toolchain install 1.90.0 --component clippy,rustfmt
#     (the toolchain's own bin directory goes first on PATH, so another Rust on PATH cannot shadow the pin)
#   - a C compiler as `cc` (rustc links through it) and python3 (the drift check)
#   - for secret-scan: gitleaks 8.30.1 on PATH, and a full (non-shallow) clone. ci.yml's checksum-verified fetch:
#       curl -sSfL -o gitleaks.tar.gz \
#         https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz
#       echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  gitleaks.tar.gz" | sha256sum -c -
#   - none of RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS, CARGO_BUILD_RUSTFLAGS, CARGO_BUILD_TARGET, RUSTDOCFLAGS set: ci.yml
#     runs without them. CARGO_TARGET_DIR may point anywhere.
# On a cold cache cargo fetches crates from crates.io.
#
# Exit: 0 every gate passed · 1 a gate failed · 2 a prerequisite is missing, or the usage is wrong.
# As in ci.yml, a job stops at its first failing step and the other job still runs. The drift check runs first: it
# fails the run if this file and ci.yml no longer run the same steps with the same pins (`ci/check_ci_drift.py`).
set -uo pipefail

TOOLCHAIN=1.90.0
COMPONENTS="clippy rustfmt"
GITLEAKS_VERSION=8.30.1
GITLEAKS_LINUX_X64_SHA256=551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb
JOBS="rust secret-scan"
FULL_HISTORY_JOBS="secret-scan"

# The jobs, step for step as in ci.yml. The drift check reads these lines: one `step '…'` per line, nothing else.
rust_job() {
  step 'cargo fmt --all --check' &&
  step 'cargo clippy --workspace --all-targets -- -D warnings' &&
  step 'cargo build --release --workspace' &&
  step 'cargo test --workspace --release' &&
  step 'cargo run --release -p kt-runtime --example runtime_probe' &&
  step 'RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps'
}

secret_scan_job() {
  step 'gitleaks git --redact --no-banner --exit-code 1 .'
}

step() {
  echo "## $1"
  eval "$1"
  local rc=$?
  echo "exit=$rc"
  return $rc
}

cd "$(dirname "$0")/.." || exit 2

selected=${1:-all}
case "$selected" in
  all) selected=$JOBS ;;
  rust | secret-scan) ;;
  *) echo "usage: bash ci/gates.sh [all|rust|secret-scan]" >&2; exit 2 ;;
esac
[ $# -le 1 ] || { echo "usage: bash ci/gates.sh [all|rust|secret-scan]" >&2; exit 2; }

# Prerequisites: collect every missing one, then stop before any gate. Nothing here installs anything.
missing=""
need() { missing="$missing
  - $1"; }
export RUSTUP_TOOLCHAIN=$TOOLCHAIN RUSTUP_AUTO_INSTALL=0

git rev-parse --is-inside-work-tree > /dev/null 2>&1 || need "a git clone (git, run from inside the repository)"
command -v python3 > /dev/null 2>&1 || need "python3 (the drift check)"
command -v cc > /dev/null 2>&1 || need "a C compiler as \`cc\` (rustc links through it)"
for v in RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS CARGO_BUILD_TARGET RUSTDOCFLAGS; do
  [ -z "${!v+x}" ] || need "$v unset (ci.yml runs without it; it is set here)"
done
toolchain_ok=1
if command -v rustup > /dev/null 2>&1; then
  if grep -q "^$TOOLCHAIN-" <<< "$(rustup toolchain list 2> /dev/null)"; then
    installed=$(rustup component list --toolchain "$TOOLCHAIN" --installed 2> /dev/null)
    for c in $COMPONENTS; do
      grep -q "^$c-" <<< "$installed" || { need "the $c component of toolchain $TOOLCHAIN"; toolchain_ok=0; }
    done
  else
    need "Rust toolchain $TOOLCHAIN (rustup toolchain install $TOOLCHAIN --component clippy,rustfmt)"
    toolchain_ok=0
  fi
else
  need "rustup with Rust toolchain $TOOLCHAIN"
  toolchain_ok=0
fi
if [ $toolchain_ok = 1 ]; then
  # The toolchain's own bin first on PATH, as the toolchain action makes it the default: another Rust on PATH (a
  # distribution's or Homebrew's cargo) would otherwise run instead and ignore the pin.
  tc_cargo=$(rustup which --toolchain "$TOOLCHAIN" cargo 2> /dev/null)
  if [ -x "$tc_cargo" ]; then
    tc_bin=$(dirname "$tc_cargo")
    export PATH="$tc_bin:$PATH"
    for t in rustc cargo; do
      grep -q "^$t $TOOLCHAIN " <<< "$($t --version 2> /dev/null)" || need "$t resolving to toolchain $TOOLCHAIN"
    done
  else
    need "rustup which --toolchain $TOOLCHAIN cargo (a rustup that can name the toolchain's own cargo)"
  fi
fi
for job in $selected; do
  case " $FULL_HISTORY_JOBS " in
    *" $job "*)
      [ "$(git rev-parse --is-shallow-repository 2> /dev/null)" = false ] ||
        need "a full (non-shallow) clone ($job scans every commit)"
      ;;
  esac
  if [ "$job" = secret-scan ]; then
    if command -v gitleaks > /dev/null 2>&1; then
      have=$(gitleaks version 2> /dev/null)
      [ "${have#v}" = "$GITLEAKS_VERSION" ] || need "gitleaks $GITLEAKS_VERSION on PATH (found ${have:-no version})"
    else
      need "gitleaks $GITLEAKS_VERSION on PATH"
    fi
  fi
done

dirty=""
[ -z "$(git status --porcelain 2> /dev/null)" ] || dirty=" + uncommitted changes"
echo "# ci/gates.sh $selected — $(date -u +%Y-%m-%dT%H:%M:%SZ) — $(git rev-parse --short HEAD 2> /dev/null)$dirty —" \
  "$(uname -sm)"
if [ -n "$missing" ]; then
  echo "MISSING PREREQUISITES (no gate ran, nothing was installed):$missing"
  exit 2
fi
echo "# $(rustc --version) · $(cargo --version) · $tc_bin"

failed=""
step 'python3 -I ci/check_ci_drift.py' || {
  echo "FAIL: drift (ci/gates.sh no longer runs ci.yml's steps; no job ran)"
  exit 1
}
for job in $selected; do
  echo "# job $job"
  if "${job//-/_}_job"; then
    echo "# job $job: PASS"
  else
    echo "# job $job: FAIL"
    failed="$failed $job"
  fi
done
if [ -n "$failed" ]; then
  echo "FAIL:$failed"
  exit 1
fi
echo "ALL PASS"
