#!/usr/bin/env bash
# Stop all running ink-engine-g* replica containers.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/config.sh"

NAMES=$(docker ps -a --format '{{.Names}}' | grep -E "^${CONTAINER_PREFIX}[0-9]+$" || true)
if [ -z "$NAMES" ]; then
  echo "No ${CONTAINER_PREFIX}* containers found."
  exit 0
fi

# shellcheck disable=SC2086
docker rm -f $NAMES
echo "Stopped replicas."
