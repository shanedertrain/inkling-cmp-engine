#!/usr/bin/env python3
"""Is ink-kernels generated from the kernel sources in this tree? (package R2; no nvcc, stdlib only)

  python3 -I engine/kernels/pack/check_fresh.py

Reads the committed engine/rust/crates/ink-kernels/kpack/manifest.json (written by build.sh: every repo file the pack
is compiled from, with its sha256, plus the nvcc version and flags), hashes those files in this tree, recomputes the
kpack id exactly as build.sh does, and requires it to equal the manifest's and src/gen.rs's KPACK_ID. A kernel header,
keys.toml, structs.def or build.sh edited since the last regeneration makes it fail with the list of changed files.
A new #include is covered too: adding it edits a listed file. It also requires gen.rs's GENERATOR_FNV1A64 to equal the
FNV-1a 64 of the generator's sources (engine/rust/crates/xtask/src/kernels/), so a generator change without a
regeneration fails too (R3). Not covered: a hand edit of gen.rs (it says "do not edit"; review the diff), and a
different CUDA image with the same nvcc version line. Fix: engine/kernels/pack/regen.sh. Exit 0 / 1.
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
CRATE = os.path.join(ROOT, "engine/rust/crates/ink-kernels")
FIX = "run engine/kernels/pack/regen.sh (rebuilds the pack, regenerates ink-kernels, runs E0; see its README)"


GENERATOR = os.path.join(ROOT, "engine/rust/crates/xtask/src/kernels")
# xtask's GENERATOR_SOURCES order (kernels/mod.rs)
GENERATOR_SOURCES = ("mod.rs", "ctype.rs", "cubin.rs", "emit.rs", "sig.rs", "names.rs")


def generator_fnv1a64():
    """FNV-1a 64 over the generator's sources, as xtask computes it into gen.rs."""
    h = 0xCBF29CE484222325
    for f in GENERATOR_SOURCES:
        for b in open(os.path.join(GENERATOR, f), "rb").read():
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    m = json.load(open(os.path.join(CRATE, "kpack/manifest.json")))
    gen = open(os.path.join(CRATE, "src/gen.rs")).read()
    errors = []
    lines = []
    for inp in m["inputs"]:
        path = os.path.join(ROOT, inp["path"])
        if not os.path.isfile(path):
            errors.append("missing: %s" % inp["path"])
            continue
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if h != inp["sha256"]:
            errors.append("changed: %s" % inp["path"])
        lines.append("%s %s" % (inp["path"], h))
    kid = hashlib.sha256(("\n".join(lines) + "\n%s\n%s\n" % (m["nvcc"], m["flags"])).encode()).hexdigest()
    g = re.search(r'pub const KPACK_ID: &str = "([0-9a-f]{64})";', gen)
    if not g:
        errors.append("src/gen.rs has no KPACK_ID")
    elif g.group(1) != m["kpack_id"]:
        errors.append("src/gen.rs KPACK_ID %s != kpack/manifest.json %s" % (g.group(1)[:12], m["kpack_id"][:12]))
    if not errors and kid != m["kpack_id"]:
        errors.append("kpack id of this tree %s != manifest %s" % (kid[:12], m["kpack_id"][:12]))
    g = re.search(r"pub const GENERATOR_FNV1A64: u64 = 0x([0-9a-f]{16});", gen)
    have = generator_fnv1a64()
    if not g:
        errors.append("src/gen.rs has no GENERATOR_FNV1A64")
    elif int(g.group(1), 16) != have:
        errors.append("src/gen.rs was written by another generator (GENERATOR_FNV1A64 %s, xtask/src/kernels/ now %016x)"
                      % (g.group(1), have))
    lay = hashlib.sha256(open(os.path.join(CRATE, "kpack/layout.json"), "rb").read()).hexdigest()
    if lay != m["layout_sha256"]:
        errors.append("kpack/layout.json is not the manifest's")
    if errors:
        print("ink-kernels is STALE (generated from other kernel sources):\n  %s\n%s" % ("\n  ".join(errors), FIX),
              file=sys.stderr)
        return 1
    print("ink-kernels fresh: kpack %s, %d input files unchanged" % (kid[:12], len(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
