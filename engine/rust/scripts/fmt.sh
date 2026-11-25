#!/usr/bin/env bash
# fmt.sh [--check]: cargo fmt in the build image (rustfmt runs no crate code). Only the workspace sources are
# writable: vendor/, target/, .cargo, Cargo.lock, scripts/ and docker/ are mounted read-only over the writable workspace.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/docker/IMAGE.lock"
[ "$(docker image inspect --format '{{.Id}}' "$RUST_IMAGE")" = "$RUST_IMAGE_ID" ] || { echo "$RUST_IMAGE is not the image of IMAGE.lock (scripts/image.sh rust)" >&2; exit 1; }
exec docker run --rm --network none --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 --memory "${INK_MEM:-32g}" \
  --read-only --tmpfs /tmp:rw,exec,nosuid,mode=1777 --tmpfs /cargo-home:rw,exec,nosuid,mode=1777 \
  -e HOME=/tmp -e CARGO_HOME=/cargo-home \
  -v "$here:/w" -v "$here/vendor:/w/vendor:ro" -v "$here/target:/w/target:ro" -v "$here/.cargo:/w/.cargo:ro" \
  -v "$here/Cargo.lock:/w/Cargo.lock:ro" -v "$here/scripts:/w/scripts:ro" -v "$here/docker:/w/docker:ro" \
  -w /w "$RUST_IMAGE" cargo fmt --all "$@"
