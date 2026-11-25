#!/usr/bin/env bash
# Kernel pack build (package R2, docs/RUST-PLAN.md 3.2, 6). Usage: build.sh OUT_DIR
# Runs inside ink-ninfer-toolchain:cu1331 or ink-rust-cuda:cu1331-r1.98.0 (nvcc 13.3, python3 >= 3.11), no GPU.
#
#   1. keys.toml -> OUT/keep_<tu>.inc (one INK_KEEP(instance) per key), OUT/keys.toml (copy, for xtask gen-kernels)
#   2. kpack id = sha256 over: the sorted "<repo path> <sha256>" lines of every repo file the two TUs, layout_emit.cpp,
#      structs.def and keys.toml consist of (nvcc -M), the nvcc version line and the flag string
#   3. nvcc -cubin of pack_engine.cu / pack_drafter.cu -> OUT/pack_<tu>.cubin, with the C++ engine's CUDA flags
#      (engine/native/CMakeLists.txt as CMake expands them for inkrt, minus host-only -Xcompiler=-fPIC). The kpack id
#      is NOT compiled into the cubins: an extra device global renumbers the c[0x4] address slots of kernels that read
#      a global (k_cos_sin's SASS differed, bench/raw/2026-11-23-rust-R2-pack/pack1); ink-kernels checks the id and
#      the cubin sizes at build time instead (build.rs: manifest.json vs gen.rs)
#   4. layout_emit.cpp (host program, same headers and flags, device code PTX only) -> OUT/layout.json
#   5. OUT/manifest.json: kpack id, nvcc version, flags, inputs, cubins (bytes, sha256), key counts
# `xtask gen-kernels OUT DEST` reads OUT; e0.sh compares OUT's cubins with libinkrt_c.so (gate E0).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
out=${1:?usage: build.sh OUT_DIR}
mkdir -p "$out"
out=$(cd "$out" && pwd)
# the C++ engine's CUDA flags (CMakeLists.txt: CMAKE_CUDA_FLAGS + Release + C++20 + sm_80), the code for sm_80 only
FLAGS="-O3 -lineinfo -Xptxas -warn-spills -O3 -DNDEBUG -std=c++20"
ARCH="-gencode arch=compute_80,code=sm_80"
NVCC_VERSION=$(nvcc --version | grep -E '^Cuda compilation tools')
cd "$here"

python3 -I - "$here/keys.toml" "$out" <<'EOF'
import sys, tomllib, shutil
keys, out = sys.argv[1], sys.argv[2]
doc = tomllib.load(open(keys, "rb"))
if sorted(doc) != ["cooperative", "drafter", "engine"]:
    sys.exit("keys.toml: holds cooperative, [engine] and [drafter], got %s" % sorted(doc))
for tu in ("engine", "drafter"):
    table = doc[tu]
    with open("%s/keep_%s.inc" % (out, tu), "w") as f:
        for key, inst in table.items():
            f.write("INK_KEEP(%s)  // %s\n" % (inst, key))
shutil.copyfile(keys, out + "/keys.toml")
EOF

for src in pack_engine.cu pack_drafter.cu layout_emit.cpp; do
  nvcc -M $FLAGS -I"$out" -x cu -MT dep "$src" > "$out/deps-$src.d"
done
# inputs.txt: "<repo path> <sha256>" of every dependency inside the repo, plus the pack's own non-included files
kpack_id=$(python3 -I - "$repo" "$out" "$NVCC_VERSION" "$FLAGS $ARCH" <<'EOF'
import glob, hashlib, os, sys
repo, out, nvcc, flags = sys.argv[1:5]
paths = {os.path.join(repo, "engine/kernels/pack", f) for f in ("structs.def", "keys.toml", "build.sh")}
for d in glob.glob(out + "/deps-*.d"):
    target, sep, rest = open(d).read().partition(":")   # nvcc writes "dep : a \\\n b ..."
    if target.strip() != "dep" or not sep:
        sys.exit("unexpected nvcc -M output in " + d)
    for t in rest.replace("\\\n", " ").split():
        p = os.path.realpath(os.path.join(repo, "engine/kernels/pack", t))
        if not os.path.isfile(p):
            sys.exit("dependency %s does not exist" % p)
        if p.startswith(repo + "/"):
            paths.add(p)
lines = ["%s %s" % (os.path.relpath(p, repo), hashlib.sha256(open(p, "rb").read()).hexdigest()) for p in sorted(paths)]
open(out + "/inputs.txt", "w").write("\n".join(lines) + "\n")
print(hashlib.sha256(("\n".join(lines) + "\n%s\n%s\n" % (nvcc, flags)).encode()).hexdigest())
EOF
)

# Stage sources into a fixed path with zeroed mtimes so nvcc -lineinfo produces byte-reproducible cubins (V10).
stage=/tmp/ink-kpack-stage
rm -rf "$stage"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/engine" "$stage/inc"
cp -r "$repo/engine/kernels" "$stage/engine/kernels"
cp "$out/keep_engine.inc" "$out/keep_drafter.inc" "$stage/inc/"
find "$stage" -exec touch -h -d @0 {} +

(
  cd "$stage/engine/kernels/pack"
  for tu in engine drafter; do
    TZ=UTC SOURCE_DATE_EPOCH=0 nvcc $FLAGS $ARCH -cubin -I"$stage/inc" -o "$stage/pack_$tu.cubin" "pack_$tu.cu" \
      2> "$out/nvcc-$tu.log" || { cat "$out/nvcc-$tu.log" >&2; exit 1; }
    mv "$stage/pack_$tu.cubin" "$out/pack_$tu.cubin"
  done
)
nvcc $FLAGS -gencode arch=compute_80,code=compute_80 -x cu -o "$out/layout_emit" layout_emit.cpp 2> "$out/nvcc-layout.log" || { cat "$out/nvcc-layout.log" >&2; exit 1; }
"$out/layout_emit" > "$out/layout.json"

python3 -I - "$out" "$kpack_id" "$NVCC_VERSION" "$FLAGS $ARCH" <<'EOF'
import hashlib, json, sys, tomllib
out, kid, nvcc, flags = sys.argv[1:5]
keys = tomllib.load(open(out + "/keys.toml", "rb"))
inputs = [l.split(" ") for l in open(out + "/inputs.txt").read().splitlines()]
cubins = []
for tu in ("engine", "drafter"):
    b = open("%s/pack_%s.cubin" % (out, tu), "rb").read()
    cubins.append({"tu": tu, "file": "pack_%s.cubin" % tu, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
                   "keys": len(keys[tu])})
m = {"format": "ink-kpack-manifest/1", "kpack_id": kid, "nvcc": nvcc, "flags": flags,
     "layout_sha256": hashlib.sha256(open(out + "/layout.json", "rb").read()).hexdigest(),
     "inputs": [{"path": p, "sha256": h} for p, h in inputs], "cubins": cubins}
json.dump(m, open(out + "/manifest.json", "w"), indent=1)
open(out + "/manifest.json", "a").write("\n")
print("kpack %s: %s" % (kid, ", ".join("%s %d B, %d keys" % (c["file"], c["bytes"], c["keys"]) for c in cubins)))
EOF
