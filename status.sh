#!/usr/bin/env bash
# Check live health and model listing for configured GPU replicas.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/config.sh"

for G in $GPUS; do
  PORT=$((BASE_PORT + G))
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    echo "GPU $G  127.0.0.1:$PORT  OK (200)"
  else
    echo "GPU $G  127.0.0.1:$PORT  DOWN ($CODE)"
  fi
done
