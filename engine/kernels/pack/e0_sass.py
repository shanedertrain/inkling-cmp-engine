#!/usr/bin/env python3
"""E0 of docs/RUST-PLAN.md 7.3: sha256 of every kernel's SASS bytes in the cubins of a binary (own code, package A1).

  python3 -I e0_sass.py LIB_OR_DIR OUT.jsonl

LIB_OR_DIR: an ELF with embedded fatbins (libinkrt_c.so; `cuobjdump -xelf all` extracts its cubins into a temporary
directory) or a directory of .cubin files (the Rust kernel pack). One line per kernel:
  {"cubin": "<file>", "arch": "sm_80", "kernel": "<mangled name>", "bytes": n, "sha256": "<hex>"}
from the ELF section `.text.<kernel>` of each cubin (the SASS instruction bytes; relocations are in other sections).
Lines are sorted by kernel, then cubin; the first line is a header. No GPU needed. Compare two outputs by kernel name:
E0 passes when every kernel the engine launches has the same sha256 in both.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile


def sections(data):
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian ELF64")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    hdrs = []
    for i in range(shnum):
        name, typ, flags, addr, off, size = struct.unpack_from("<IIQQQQ", data, shoff + i * shentsize)
        hdrs.append((name, typ, off, size))
    stro, strsz = hdrs[shstrndx][2], hdrs[shstrndx][3]
    strtab = data[stro:stro + strsz]
    for name, typ, off, size in hdrs:
        end = strtab.index(b"\0", name)
        yield strtab[name:end].decode(), typ, data[off:off + size] if typ != 8 else b""   # 8 = SHT_NOBITS


def arch_of(path):   # cuobjdump names extracted cubins <binary>.<n>.sm_<arch>.cubin
    parts = os.path.basename(path).split(".")
    return next((p for p in parts if p.startswith("sm_")), "?")


def cubins(src, tmp):
    if os.path.isdir(src):
        return sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".cubin"))
    subprocess.run(["cuobjdump", "-xelf", "all", os.path.abspath(src)], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
    return sorted(os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".cubin"))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        files = cubins(src, tmp)
        if not files:
            sys.exit("no cubin in " + src)
        for path in files:
            data = open(path, "rb").read()
            arch = arch_of(path)
            for name, typ, body in sections(data):
                if name.startswith(".text."):
                    rows.append({"cubin": os.path.basename(path), "arch": arch, "kernel": name[6:], "bytes": len(body),
                                 "sha256": hashlib.sha256(body).hexdigest()})
    rows.sort(key=lambda r: (r["kernel"], r["cubin"]))
    with open(out, "w") as f:
        f.write(json.dumps({"format": "ink-e0-sass/1", "source": os.path.basename(src), "cubins": len(files),
                            "kernels": len(rows)}, separators=(",", ":")) + "\n")
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print("%s: %d kernels in %d cubins" % (out, len(rows), len(files)))


if __name__ == "__main__":
    main()
