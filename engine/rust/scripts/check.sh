#!/usr/bin/env bash
# check.sh: verify kernel pack freshness, build, clippy, rustfmt, and tests.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
python3 -I "$root/engine/kernels/pack/check_fresh.py"
cd "$here"
cargo build --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all --check
cargo test --workspace --locked
echo "ok: workspace builds, clippy and fmt are clean, tests pass"
