#!/usr/bin/env bash
# Build step_trace inside ink-ninfer-toolchain:cu1331. Usage: build.sh SRC_DIR OUT_DIR
#   libstep_trace.so : CUPTI activity injection (CUDA_INJECTION64_PATH), cupti_trace.cpp
#   floor_bench      : empty-kernel graph of a real step's node list, floor_bench.cu
set -euo pipefail
S=${1:-/st}; B=${2:-/bst}
mkdir -p "$B"; rm -f "$B/libstep_trace.so" "$B/floor_bench"
C=/usr/local/cuda/targets/x86_64-linux
# libcupti.so.13 is copied next to the library (rpath $ORIGIN) so the injection also loads in other images (config B)
cp -L $C/lib/libcupti.so.13 "$B/"
g++ -O2 -std=c++17 -shared -fPIC -o "$B/libstep_trace.so" "$S/cupti_trace.cpp" -I$C/include -L$C/lib -lcupti -lpthread '-Wl,-rpath,$ORIGIN'
nvcc -O3 -std=c++17 -arch=sm_80 -o "$B/floor_bench" "$S/floor_bench.cu" -lnvidia-ml
ls -l "$B/libstep_trace.so" "$B/floor_bench"
