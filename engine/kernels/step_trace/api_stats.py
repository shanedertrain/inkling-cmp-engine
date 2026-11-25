#!/usr/bin/env python3
"""api_stats.py TRACE.csv [--skip-s S]: host runtime-API durations of a trace recorded with INK_TRACE_KINDS=R (no
kernel records, so no per-node tracing cost on graph launches). Per API function: calls, median and mean us; and the
step loop: interval between consecutive cudaGraphLaunch starts (median), launch duration, synchronize duration, and
the host time from the synchronize's return to the next launch. Records in the first S s after the first graph launch
are skipped (default 2)."""
import csv
import json
import statistics as st
import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    skip = float(sys.argv[sys.argv.index("--skip-s") + 1]) if "--skip-s" in sys.argv else 2.0
    rs = [r for r in csv.DictReader(open(path)) if r["kind"] == "R"]
    for r in rs:
        r["s"], r["e"] = int(r["start_ns"]), int(r["end_ns"])
    rs.sort(key=lambda r: r["s"])
    gl = [r for r in rs if r["name"].startswith("cudaGraphLaunch")]
    t0 = gl[0]["s"] + skip * 1e9
    rs = [r for r in rs if r["s"] >= t0]
    gl = [r for r in gl if r["s"] >= t0]
    by = defaultdict(list)
    for r in rs:
        by[r["name"]].append((r["e"] - r["s"]) / 1e3)
    out = {"trace": path, "graph_launches": len(gl),
           "api_us": {k: {"calls": len(v), "median": st.median(v), "mean": st.mean(v)} for k, v in sorted(by.items(), key=lambda x: -sum(x[1]))}}
    iv = [(b["s"] - a["s"]) / 1e3 for a, b in zip(gl, gl[1:])]
    syncs = [r for r in rs if r["name"].startswith("cudaStreamSynchronize")]
    after = []
    j = 0
    for a, b in zip(gl, gl[1:]):
        last = None
        while j < len(syncs) and syncs[j]["s"] < b["s"]:
            if syncs[j]["s"] >= a["s"]:
                last = syncs[j]
            j += 1
        if last:
            after.append((b["s"] - last["e"]) / 1e3)
    out["step_interval_us_median"] = st.median(iv) if iv else None
    out["launch_us_median"] = st.median([(r["e"] - r["s"]) / 1e3 for r in gl])
    out["sync_return_to_next_launch_us_median"] = st.median(after) if after else None
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
