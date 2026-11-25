#!/usr/bin/env python3 -I
"""Extracts the config B parser recordings the ink-parse differential test reads (package F3) into
engine/rust/testdata-local/parser/.

    python3 -I engine/rust/crates/ink-parse/tests/prepare_fixtures.py [OUT_DIR]

Reads bench/raw/2026-11-23-S5-server-parity/{parser-cases,parser-ref}.jsonl.zst (needs the `zstd` command) and writes
cases.jsonl and ref.jsonl unchanged (decompressed). Nothing else is read or written.
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
S5 = os.path.join(ROOT, "bench/raw/2026-11-23-S5-server-parity")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "engine/rust/testdata-local/parser")

os.makedirs(OUT, exist_ok=True)
for src, dst in (("parser-cases", "cases.jsonl"), ("parser-ref", "ref.jsonl")):
    with open(os.path.join(OUT, dst), "wb") as f:
        subprocess.run(["zstd", "-dc", os.path.join(S5, src + ".jsonl.zst")], stdout=f, check=True)
    with open(os.path.join(OUT, dst), "rb") as f:
        print(f"{dst}: {sum(1 for _ in f)} lines")
