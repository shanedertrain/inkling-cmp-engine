#!/usr/bin/env bash
# Build the S2f drafter harness for sm_80. Pinned image: ink-ninfer-toolchain:cu1331 (nvcc 13.3.73).
# Writes build/s2f_drafter, ptxas.log (registers/spills), res-usage.txt, local-mem.txt.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-build}"
mkdir -p "$OUT"
NVCC="${NVCC:-nvcc}"
"$NVCC" --version | tail -1 > "$OUT/nvcc-version.txt"
"$NVCC" -std=c++20 -O3 -lineinfo -gencode arch=compute_80,code=sm_80 -Xptxas -v -Xcompiler -Wall \
  -o "$OUT/s2f_drafter" harness.cu -ldl -lpthread 2> "$OUT/ptxas.log"
if command -v cuobjdump > /dev/null; then
  cuobjdump -res-usage "$OUT/s2f_drafter" 2>/dev/null | grep -A1 Function | paste - - | \
    sed -E 's/.*Function ([^ :]+).*REG:([0-9]+) STACK:([0-9]+) SHARED:([0-9]+) LOCAL:([0-9]+).*/\1 REG=\2 STACK=\3 SHARED=\4 LOCAL=\5/' \
    > "$OUT/res-usage.txt" || true
  cuobjdump -sass "$OUT/s2f_drafter" | grep -c -E '\b(LDL|STL)\b' > "$OUT/local-mem.txt" || true
fi
grep -E 'spill' "$OUT/ptxas.log" | sort | uniq -c > "$OUT/spill-summary.txt" || true
echo built "$OUT/s2f_drafter"
