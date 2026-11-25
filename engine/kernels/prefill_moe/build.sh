#!/usr/bin/env bash
# Build the S4 prefill MoE bench for sm_80. Usage: build.sh OUTDIR. Image: ink-ninfer-toolchain:cu1331 (nvcc 13.3).
# Writes the binary, ptxas.log (registers / spills per kernel), res-usage.txt, bench.sass, local-mem.txt.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-build}"
mkdir -p "$OUT"
nvcc --version | tail -1 > "$OUT/nvcc-version.txt"
nvcc -std=c++20 -O3 -lineinfo -gencode arch=compute_80,code=sm_80 -Xptxas -v ${EXTRA_NVCC:-} \
  -o "$OUT/bench" "$SRC/bench.cu" -lcuda -ldl -lpthread 2> "$OUT/ptxas.log"
cuobjdump -res-usage "$OUT/bench" 2>/dev/null | grep -A1 'moe_' | paste - - | \
  sed -E 's/.*Function ([^ :]+).*REG:([0-9]+) STACK:([0-9]+) SHARED:([0-9]+) LOCAL:([0-9]+).*/\1 REG=\2 STACK=\3 SHARED=\4 LOCAL=\5/' \
  > "$OUT/res-usage.txt" || true
cuobjdump -sass "$OUT/bench" > "$OUT/bench.sass" 2>/dev/null || true
grep -c -E '\b(LDL|STL)\b' "$OUT/bench.sass" > "$OUT/local-mem.txt" || true
cat "$OUT/res-usage.txt"
