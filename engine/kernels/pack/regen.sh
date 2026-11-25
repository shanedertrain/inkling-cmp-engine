#!/usr/bin/env bash
# regen.sh [OUT_DIR]: rebuild the sm_80 kernel pack locally and regenerate ink-kernels bindings.
# Requires nvcc (CUDA 13.x) and cargo on PATH, or run inside the root Dockerfile image.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
out=${1:-"$root/build/kpack"}
mkdir -p "$out"
"$here/build.sh" "$out"
cd "$root/engine/rust"
cargo run --release --locked -p xtask -- gen-kernels "$out" "$root/engine/rust/crates/ink-kernels"
python3 -I "$here/check_fresh.py"
echo "ok: regenerated ink-kernels from $out"
