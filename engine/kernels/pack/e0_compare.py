#!/usr/bin/env python3
"""E0 of docs/RUST-PLAN.md 7.3: compare two e0_sass.py outputs kernel by kernel (own code, package R2).

  python3 -I e0_compare.py REF.jsonl CAND.jsonl OUT.json [--subset]

REF: the C++ engine (`e0_sass.py libinkrt_c.so`), CAND: the kernel pack (`e0_sass.py <pack dir>`) or another build of
the engine. Kernels are matched by mangled name after one normalisation: the per-TU id nvcc puts into the name of an
anonymous namespace (`_GLOBAL__N__<n>_<id>_<len>_<file>_<hash>`, e.g. `41_GLOBAL__N__40219435_9_engine_cu_6d0b4272`) is
replaced by `<anon>`, since the same kernel text compiled in another TU (engine.cu vs pack_engine.cu) gets another id.
Kernels are also keyed by cubin role: REF cubins `engine.*`/`draft.*`, pack cubins `pack_engine*`/`pack_drafter*`.

Pass: every REF kernel is in CAND with the same sha256 (and, without --subset, CAND has no kernel REF lacks).
Writes OUT.json (counts and every differing or missing kernel) and prints the verdict; exit 1 on failure.
"""
import json
import re
import sys

ANON = re.compile(r"\d+_GLOBAL__N__[0-9a-f]+_\d+_\w+?_[0-9a-f]{8}")
ROLE = {"engine": "engine", "draft": "drafter", "pack_engine": "engine", "pack_drafter": "drafter"}


def role(cubin):
    stem = cubin.split(".")[0]
    if stem not in ROLE:
        raise SystemExit("unknown cubin " + cubin)
    return ROLE[stem]


def norm(kernel):
    k, n = ANON.subn("<anon>", kernel)
    if "_GLOBAL__N_" in k:
        raise SystemExit("anonymous-namespace id not normalised: " + kernel)
    return k


def load(path):
    rows = [json.loads(l) for l in open(path)][1:]
    out = {}
    for r in rows:
        key = (role(r["cubin"]), norm(r["kernel"]))
        if key in out:
            raise SystemExit("duplicate kernel %s in %s" % (key, path))
        out[key] = r
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 3:
        sys.exit(__doc__)
    subset = "--subset" in sys.argv
    ref, cand = load(args[0]), load(args[1])
    same = sorted(k for k in ref if k in cand and cand[k]["sha256"] == ref[k]["sha256"])
    diff = sorted(k for k in ref if k in cand and cand[k]["sha256"] != ref[k]["sha256"])
    missing = sorted(k for k in ref if k not in cand)
    extra = sorted(k for k in cand if k not in ref)
    ok = not diff and not missing and (subset or not extra)
    res = {"format": "ink-e0-compare/1", "ref": args[0], "cand": args[1], "ref_kernels": len(ref),
           "cand_kernels": len(cand), "identical": len(same), "differing": len(diff), "missing_in_cand": len(missing),
           "extra_in_cand": len(extra), "pass": ok,
           "by_role": {r: {"ref": sum(1 for k in ref if k[0] == r), "identical": sum(1 for k in same if k[0] == r)}
                       for r in sorted({k[0] for k in ref})},
           "differing_kernels": [{"role": k[0], "kernel": k[1], "ref": ref[k]["sha256"], "cand": cand[k]["sha256"],
                                  "ref_bytes": ref[k]["bytes"], "cand_bytes": cand[k]["bytes"]} for k in diff],
           "missing_kernels": ["%s %s" % k for k in missing], "extra_kernels": ["%s %s" % k for k in extra]}
    with open(args[2], "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print("E0 %s: ref %d, cand %d, identical %d, differing %d, missing %d, extra %d" % (
        "PASS" if ok else "FAIL", len(ref), len(cand), len(same), len(diff), len(missing), len(extra)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
