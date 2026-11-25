#!/usr/bin/env bash
# m3run6.sh <TAG> : P6 M3 stand-in (probes/p6-attn, copy in s2b/p6src) with $OUTD/cmds.txt, GPU 6, foreground.
B=~/inkling-cmp-engine/s2b; TAG=$1; OUTD=$B/$TAG
docker run --rm --name ink-eng-s2b-g6 --gpus '"device=6"' --network none --cpus 8 --memory 32g -u "$(id -u):$(id -g)" \
  -e P6_TAG="$TAG" -e P6_VBIOS=67 -e P6_EXPECT_UUID=GPU-a1446cdb-cb24-3780-cf94-e600ec4d1819 \
  -v $B/p6src:/p6:ro -v "$OUTD":/out \
  ink-ninfer-toolchain:cu1331 bash /p6/run_cmds.sh > "$OUTD/docker.log" 2>&1 < /dev/null
echo "m3 rc=$? $(date -u +%T)" >> $OUTD/chain.log
