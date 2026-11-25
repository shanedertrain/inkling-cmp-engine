#!/usr/bin/env bash
# run6.sh <TAG> : S2b bench on GPU 6 with s2b/<TAG>/cmds.txt; sources s2b/src (ro), real KV dump (ro). Foreground.
S=~/inkling-cmp-engine/s2b; TAG=$1; OUTD=$S/$TAG; mkdir -p $OUTD
docker run --rm --name ink-eng-s2b-g6 --gpus '"device=6"' --network none --cpus 6 --memory 48g -u "$(id -u):$(id -g)" \
  -e P6_TAG="$TAG" -e P6_VBIOS=67 -e DA_FLAGS="${DA_FLAGS:-}" \
  -v $S/src:/src:ro -v "$OUTD":/out -v $S/kvcap2-g6/dump:/dump:ro \
  ink-ninfer-toolchain:cu1331 bash /src/scripts/run_in.sh > "$OUTD/docker.log" 2>&1 < /dev/null
echo "run6 $TAG rc=$? $(date -u +%T)" >> $OUTD/chain.log
