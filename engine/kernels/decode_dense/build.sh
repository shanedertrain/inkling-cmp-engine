#!/usr/bin/env bash
# Build the decode_dense harness inside ink-ninfer-toolchain:cu1331. Usage: build.sh SRC_DIR OUT_DIR
# (SRC_DIR = a copy of engine/kernels/decode_dense next to engine/native and engine/kernels: includes are relative).
set -euo pipefail
S=${1:-/dd}; B=${2:-/bdd}
mkdir -p "$B"; rm -f "$B/dd_bench"
nvcc -O3 -std=c++20 -arch=sm_80 -lineinfo -Xptxas -warn-spills -o "$B/dd_bench" "$S/dd_bench.cu" -lcublas -lcublasLt -ldl -lpthread 2>&1 | tee "$B/build.log"
test -x "$B/dd_bench"
