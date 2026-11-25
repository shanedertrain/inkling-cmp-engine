#!/usr/bin/env bash
# Runs INSIDE ink-eng-s4e-g<N> (ink-ninfer-toolchain:cu1331). Mounts: /src (this directory, ro), /out (rw),
# /model, /h1, /hcache, /routing (ro). Usage: run.sh STEP_NAME -- bench args...  (appends to /out/STEP_NAME.jsonl)
set -uo pipefail
step=$1; shift; [ "${1:-}" = "--" ] && shift
n_gpu=$(nvidia-smi -L | grep -c '^GPU ')
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
if [ "$n_gpu" != 1 ] || [ "$uuid" != "${S4_EXPECT_UUID:-unset}" ]; then echo "GPU GUARD FAILED: $n_gpu $uuid"; exit 3; fi
B=/out/build
if [ ! -x "$B/bench" ] || [ -n "$(find /src -newer "$B/bench" -name '*.cu*' -o -newer "$B/bench" -name '*.hpp')" ]; then
  bash /src/build.sh "$B" > /out/build.log 2>&1 || { echo BUILD FAILED; tail -20 /out/build.log; exit 4; }
fi
(cd /src && sha256sum *.cu *.cuh *.hpp *.sh *.py) > "/out/$step.sources.sha256"
echo "-- $step: bench $* ($(date -u +%T))"
timeout 14400 "$B/bench" "$@" >> "/out/$step.jsonl" 2>> "/out/$step.stderr"
echo "$step rc=$?" | tee -a /out/exit_codes.txt
