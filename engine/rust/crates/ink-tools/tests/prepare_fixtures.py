#!/usr/bin/env python3 -I
"""Extracts the config B captures the ink-replay gate (G-FE3, package F4) reads into
engine/rust/testdata-local/replay/{cap1,cap2,cap3}/.

    python3 -I engine/rust/crates/ink-tools/tests/prepare_fixtures.py [OUT_DIR]

Reads bench/raw/2026-11-23-S5-server-parity (cap1 is a tarball: needs `zstd` and `tar`) and copies the per-request
records the replay uses (`*.stream.jsonl`, `*.nonstream.json`, `*.prod.jsonl`, `*.lpids-*`, `*.ids-*`) unchanged;
the render captures and logs are left out. Nothing else is read or written.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
S5 = os.path.join(ROOT, "bench/raw/2026-11-23-S5-server-parity")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "engine/rust/testdata-local/replay")
WANTED = (".stream.jsonl", ".nonstream.json", ".prod.jsonl")


def wanted(name):
    return name.endswith(WANTED) or ".lpids-" in name or ".ids-" in name


def copy(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src_dir)):
        if wanted(name):
            shutil.copyfile(os.path.join(src_dir, name), os.path.join(dst_dir, name))
            n += 1
    print(f"{os.path.basename(dst_dir)}: {n} files")


with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(f"zstd -dc {S5}/cap1.tar.zst | tar -x -C {tmp} --wildcards 'cap1/*.jsonl' 'cap1/*.json'",
                   shell=True, check=True)
    copy(os.path.join(tmp, "cap1"), os.path.join(OUT, "cap1"))
copy(os.path.join(S5, "cap2"), os.path.join(OUT, "cap2"))
copy(os.path.join(S5, "cap3"), os.path.join(OUT, "cap3"))
