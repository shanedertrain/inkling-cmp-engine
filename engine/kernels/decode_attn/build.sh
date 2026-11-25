#!/usr/bin/env bash
# Build the S2b bench for sm_80. Usage: ./build.sh [outdir]. Pinned image ink-ninfer-toolchain:cu1331 (nvcc 13.3).
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-build}"; mkdir -p "$OUT"
NVCC="${NVCC:-nvcc}"
"$NVCC" --version | tail -1 > "$OUT/nvcc-version.txt"
"$NVCC" -std=c++20 -O3 -lineinfo -gencode arch=compute_80,code=sm_80 -Xptxas -v -Xcompiler -Wall \
  ${DA_FLAGS:-} -o "$OUT/bench" bench/bench.cu -ldl -lpthread 2> "$OUT/ptxas.log"
grep -A2 'decode_attn_kernel\|combine_kernel' "$OUT/ptxas.log" | grep -E 'Compiling|registers|spill' > "$OUT/regs.txt" || true
if command -v cuobjdump > /dev/null; then
  cuobjdump -sass "$OUT/bench" > "$OUT/bench.sass"
  echo "FP8 hardware-convert lines (expect 0): $(grep -c -E 'F2FP[^;]*(E4M3|E5M2)' "$OUT/bench.sass" || true)" >> "$OUT/regs.txt"
fi
cat "$OUT/regs.txt"
