#!/usr/bin/env bash
# Gate E0 (docs/RUST-PLAN.md 7.3, package R2; post-cutover): per kernel, the sha256 of its SASS (.text section) in the
# kernel pack equals the committed reference (e0-ref.jsonl, recorded from the final C++ oracle pack at Step 2), matched
# by name (anonymous-namespace id normalised, e0_compare.py). No GPU; needs python3.
#   e0.sh REF_JSONL PACK_DIR OUT_DIR  -> OUT_DIR/{e0-pack.jsonl, e0.json}; exit 1 unless E0 passes
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ref=${1:?e0-ref.jsonl}; pack=${2:?pack dir}; out=${3:?out dir}
mkdir -p "$out"
python3 -I "$here/e0_sass.py" "$pack" "$out/e0-pack.jsonl"
python3 -I "$here/e0_compare.py" "$ref" "$out/e0-pack.jsonl" "$out/e0.json"
