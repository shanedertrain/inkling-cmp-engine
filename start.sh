#!/usr/bin/env bash
# Start 1-8 Inkling-2-Mini replicas on 127.0.0.1:<BASE_PORT+GPU>.
# Usage:
#   ./start.sh              # prepare (first run) + start configured GPUS
#   ./start.sh restart      # stop existing replicas and start with current settings
#   ./start.sh 0 1 2 3      # start specific GPU indices
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/config.sh"

RESTART=0
if [ "${1:-}" = "restart" ]; then
  RESTART=1
  shift
fi
if [ "$#" -gt 0 ]; then
  GPUS="$*"
fi

if [ "$RESTART" = "1" ]; then
  "$ROOT_DIR/stop.sh"
fi

"$ROOT_DIR/scripts/prepare.sh"

STARTED_PORTS=()
for G in $GPUS; do
  PORT=$((BASE_PORT + G))
  UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$G" '$1 == g {print $2}')
  if [ -z "$UUID" ]; then
    echo "ERROR: could not resolve GPU UUID for index $G (check nvidia-smi)" >&2
    exit 1
  fi

  NAME="${CONTAINER_PREFIX}${G}"
  docker rm -f "$NAME" >/dev/null 2>&1 || true

  ARGS=(
    /bin_ink/ink-server
    --model-dir /model
    --host 127.0.0.1
    --port "$PORT"
    --backend engine
    --gemm-algos /cfg/gemm-algos.json
    --slots "$SLOTS"
    --kv-pages "$PAGES"
    --max-running "$MAX_RUNNING"
    --enable-prompt-tokens-details
  )
  MOUNTS=(
    -v "$MODEL_DIR:/model:ro"
    -v "$BIN_DIR:/bin_ink:ro"
    -v "$CFG_DIR:/cfg"
  )

  if [ -n "$DFLASH2_DIR" ]; then
    ARGS+=(--draft-dflash2 /dflash2)
    MOUNTS+=(-v "$DFLASH2_DIR:/dflash2:ro")
  else
    ARGS+=(--draft-fold /fold/fold.bin --draft-ctx "$DCTX")
    MOUNTS+=(-v "$FOLD_DIR:/fold:ro")
  fi

  if [ "$KV_FP8" = "1" ]; then
    ARGS+=(--kv-fp8)
  fi
  if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    ARGS+=($EXTRA_ARGS)
  fi

  docker run -d --name "$NAME" \
    --gpus "\"device=$UUID\"" \
    --network host \
    --cpus "$CPUS" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    "${MOUNTS[@]}" \
    "$IMAGE" \
    "${ARGS[@]}" >/dev/null

  echo "Started $NAME on GPU $G ($UUID) -> http://127.0.0.1:$PORT/v1 (slots=$SLOTS pages=$PAGES kv_fp8=$KV_FP8)"
  STARTED_PORTS+=("$PORT")
done

echo "Waiting for replicas to become healthy (first boot auto-tunes gemm-algos.json)..."
for PORT in "${STARTED_PORTS[@]}"; do
  READY=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 1
  done
  if [ "$READY" != "1" ]; then
    echo "ERROR: replica on 127.0.0.1:$PORT did not become healthy in 180s. Check 'docker logs ${CONTAINER_PREFIX}$((PORT - BASE_PORT))'." >&2
    exit 1
  fi
done

FIRST_PORT="${STARTED_PORTS[0]}"
echo "All replicas healthy. Running smoke test on http://127.0.0.1:$FIRST_PORT/v1/chat/completions..."
curl -fsS "http://127.0.0.1:$FIRST_PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":16,"temperature":0}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("Smoke OK:", repr(d["choices"][0]["message"].get("content","")), "usage:", d.get("usage"))'
