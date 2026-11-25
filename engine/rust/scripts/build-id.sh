#!/usr/bin/env bash
# build-id.sh [OUT_DIR]: write BUILD_ID.json (default engine/rust/) from git and Cargo.lock.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$here}
lock=$(sha256sum < "$here/Cargo.lock" | cut -d' ' -f1)
vendor="0000000000000000000000000000000000000000000000000000000000000000"
if [ -f "$here/vendor.sha256" ]; then
  vendor=$(cut -d' ' -f1 "$here/vendor.sha256")
fi
cd "$(git -C "$here" rev-parse --show-toplevel)"
commit=$(git rev-parse HEAD 2>/dev/null || echo "0000000000000000000000000000000000000000")
if [ -z "$(git status --porcelain -- engine 2>/dev/null)" ]; then
  dirty=false; diff=null
else
  dirty=true
  diff=\"$({
    git diff HEAD -- engine
    git ls-files --others --exclude-standard -z -- engine | LC_ALL=C sort -z | while IFS= read -r -d '' f; do
      printf 'untracked %s\n' "$f"; cat -- "$f"
    done
  } | sha256sum | cut -d' ' -f1)\"
fi
printf '{"commit":"%s","dirty":%s,"diff":%s,"lock":"%s","vendor":"%s"}\n' "$commit" "$dirty" "$diff" "$lock" "$vendor" > "$out/BUILD_ID.json"
cat "$out/BUILD_ID.json"
