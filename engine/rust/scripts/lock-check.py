"""lock-check.py [VENDOR_DIR]: check every vendor/<name>-<version>/.cargo-checksum.json "package" against the checksum
Cargo.lock lists for that crate, independent of cargo; also that the vendor tree and the lock name the same crates.
Run as `python3 -I scripts/lock-check.py VENDOR_DIR` (stdlib only, reads files as data)."""

import json
import os
import re
import sys

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vend = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "vendor")
with open(os.path.join(here, "Cargo.lock"), encoding="utf-8") as f:
    blocks = f.read().split("[[package]]")[1:]
want = {}
for blk in blocks:
    name = re.search(r'^name = "(.*)"', blk, re.M)
    ver = re.search(r'^version = "(.*)"', blk, re.M)
    src = re.search(r'^source = "(.*)"', blk, re.M)
    chk = re.search(r'^checksum = "(.*)"', blk, re.M)
    if not (name and ver):
        sys.exit("lock-check: unparsable package block in Cargo.lock")
    if src:
        if not src.group(1).startswith("registry+https://github.com/rust-lang/crates.io-index") or not chk:
            sys.exit(f"lock-check: {name.group(1)} has source {src.group(1)!r} / no checksum")
        want[f"{name.group(1)}-{ver.group(1)}"] = chk.group(1)
have = sorted(os.listdir(vend))
bad = []
for d in have:
    try:
        with open(os.path.join(vend, d, ".cargo-checksum.json"), encoding="utf-8") as f:
            pkg = json.load(f).get("package")
    except (OSError, ValueError):
        bad.append(f"{d}: no readable .cargo-checksum.json")
        continue
    if want.get(d) != pkg:
        bad.append(f"{d}: package {pkg!r} != lock {want.get(d)!r}")
bad += [f"{d}: in Cargo.lock, not in vendor" for d in sorted(set(want) - set(have))]
for line in bad:
    print("lock-check:", line, file=sys.stderr)
if bad:
    sys.exit(1)
print(f"lock-check ok: {len(have)} crates match Cargo.lock")
