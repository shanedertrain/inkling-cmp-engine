#!/usr/bin/env bash
# cfgb6.sh <TAG> <cfgb_attn.py args...> : config B decode attention standalone (probes/p6-attn/cfgb_attn.py), GPU 6, foreground.
# Config B image + read-only copy of its overlay (p6-attn/cfgb-pkg), no network, no $HOME, own Triton cache copy.
B=~/inkling-cmp-engine/s2b; TAG=$1; shift; OUTD=$B/$TAG; mkdir -p $OUTD
docker run --rm --name ink-eng-s2b-g7 --gpus '"device=7"' --network none --cpus 8 --shm-size 4g \
  -u "$(id -u):$(id -g)" -e HOME=/work/home -e XDG_CACHE_HOME=/work/cache -e TRITON_CACHE_DIR=/work/cache/triton \
  -e VLLM_CACHE_ROOT=/work/cache/vllm -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e P6_TAG="$TAG" -e P6_VBIOS=67 \
  -v ~/inkling-cmp-engine/p6-attn/cfgb-pkg:/usr/local/lib/python3.14/dist-packages/vllm:ro \
  -v $B/p6src:/p6:ro -v $B/scripts/cfgb_attn_kinds.py:/s2b/cfgb_attn_kinds.py:ro -v $B/cfgb-cache:/work -v $B/kvcap2-g6/dump:/dump:ro -v "$OUTD":/out \
  --entrypoint bash vllm/vllm-openai@sha256:16721463e5ae0c81623c65c50f5b8d4e37710d9bc4f6f4f7512ec1939096272f \
  -c "mkdir -p /work/home /work/cache && python3 -I ${CFGB_PY:-/p6/cfgb_attn.py} $* --out /out/cfgb" >> "$OUTD/cfgb-docker.log" 2>&1 < /dev/null
echo "cfgb $* rc=$? $(date -u +%T)" >> $OUTD/chain.log
