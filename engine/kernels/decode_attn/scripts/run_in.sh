#!/usr/bin/env bash
# Inside the container: GPU guard, build, then each line of /out/cmds.txt "<step> <bench args...>" -> /out/<step>.jsonl
set -uo pipefail
OUT=/out; B=/tmp/s2bbuild
exec > >(tee -a "$OUT/run.log") 2>&1
echo "== S2b run $(date -u +%FT%TZ) tag=${P6_TAG:-}"
n_gpu=$(nvidia-smi -L | grep -c '^GPU ')
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
if [ "$n_gpu" != 1 ] || [ "$uuid" != "${EXPECT_UUID:-GPU-a1446cdb-cb24-3780-cf94-e600ec4d1819}" ]; then echo "GPU GUARD FAILED $n_gpu $uuid"; exit 3; fi
echo "component=decode_attn version=${DA_VERSION:-unset}" | tee "$OUT/version.txt"
cp -r /src /tmp/s2bsrc && DA_FLAGS="${DA_FLAGS:-}" bash /tmp/s2bsrc/build.sh "$B" > "$OUT/build.log" 2>&1 || { echo BUILD FAILED; cat "$OUT/build.log"; exit 4; }
cat $B/regs.txt; cp $B/regs.txt $B/ptxas.log $B/bench "$OUT/"
(cd /tmp/s2bsrc && find . -name '*.cu' -o -name '*.cuh' -o -name '*.hpp' -o -name '*.sh' | sort | xargs sha256sum) > "$OUT/sources.sha256"
while read -r step args; do
  [ -z "$step" ] && continue; case "$step" in \#*) continue;; esac
  echo "-- $step: bench $args ($(date -u +%T))"; t0=$(date +%s)
  timeout 5400 "$B/bench" $args >> "$OUT/$step.jsonl" 2>> "$OUT/$step.stderr"
  printf '%s rc=%d s=%d\n' "$step" $? "$(( $(date +%s) - t0 ))" | tee -a "$OUT/exit_codes.txt"
done < "$OUT/cmds.txt"
echo "== done $(date -u +%FT%TZ)"
