#!/usr/bin/env python3
"""step_all.py TRACE.csv OUT_PREFIX [--min-launches N]: whole-step view of a trace with several graphs per step (config B:
target graph, drafter graphs, eager kernels; the served engine: drafter, verify, accept). The step boundary is the
launch of the graph with the most kernel nodes among graphs launched at least N times (default 20). Per step: interval,
GPU busy (union of all kernel / copy intervals), idle; per graph id: launches, nodes, busy ms; non-graph kernels, copies;
host runtime-API calls between boundaries. Kernel table over all steps (label = short demangled name; cuBLAS / CUTLASS /
Triton kernels keep their own names). Writes OUT.steps.json and OUT.step.md."""
import csv
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict


def demangle(names):
    try:
        out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True).stdout
        return dict(zip(names, out.split("\n")))
    except Exception:
        return {n: n for n in names}


def short(d):
    s = d.split("(")[0]
    if s.startswith("void "):
        s = s[5:]
    return s.split("<")[0].strip()[:90]


def main():
    a = sys.argv[1:]
    path, out = a[0], a[1]
    minl = int(a[a.index("--min-launches") + 1]) if "--min-launches" in a else 20
    recs = list(csv.DictReader(open(path)))
    for r in recs:
        for k in ("start_ns", "end_ns", "corr", "graph_id", "gx", "gy", "gz", "bx", "by", "bz"):
            r[k] = int(r[k])
    dm = demangle(sorted({r["name"] for r in recs if r["kind"] == "K"}))
    gpu = sorted([r for r in recs if r["kind"] in "KCS"], key=lambda r: r["start_ns"])
    api = sorted([r for r in recs if r["kind"] == "R"], key=lambda r: r["start_ns"])
    # graph launches: (graph id, corr) -> nodes
    launch = defaultdict(list)
    for r in gpu:
        if r["graph_id"]:
            launch[(r["graph_id"], r["corr"])].append(r)
    per_gid = defaultdict(list)
    for (g, c), ns in launch.items():
        per_gid[g].append((min(x["start_ns"] for x in ns), len(ns)))
    cand = {g: v for g, v in per_gid.items() if len(v) >= minl}
    main_g = max(cand, key=lambda g: st.median(n for _, n in cand[g]))
    bounds = sorted(t for t, _ in per_gid[main_g])
    steps = []
    gi = 0
    for b0, b1 in zip(bounds, bounds[1:]):
        while gi < len(gpu) and gpu[gi]["start_ns"] < b0:
            gi += 1
        j = gi
        items = []
        while j < len(gpu) and gpu[j]["start_ns"] < b1:
            items.append(gpu[j])
            j += 1
        busy, cur_s, cur_e = 0, None, None
        for r in items:
            s, e = r["start_ns"], min(r["end_ns"], b1)
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            busy += cur_e - cur_s
        steps.append({"b0": b0, "b1": b1, "items": items, "busy": busy})
    steps = steps[2:-1] if len(steps) > 5 else steps
    n = len(steps)
    gsum = defaultdict(lambda: [0, 0, 0.0])   # graph id -> launches, nodes, busy ns
    eager = [0, 0.0]
    copies = [0, 0.0]
    kt = defaultdict(lambda: [0, 0.0, []])
    kg = defaultdict(lambda: [0, 0.0, []])   # (graph id or 0 = eager, label)
    for s in steps:
        seen = set()
        for r in s["items"]:
            d = r["end_ns"] - r["start_ns"]
            if r["graph_id"]:
                if (r["graph_id"], r["corr"]) not in seen:
                    seen.add((r["graph_id"], r["corr"]))
                    gsum[r["graph_id"]][0] += 1
                gsum[r["graph_id"]][1] += 1
                gsum[r["graph_id"]][2] += d
            elif r["kind"] == "K":
                eager[0] += 1
                eager[1] += d
            else:
                copies[0] += 1
                copies[1] += d
            if r["kind"] == "K":
                lbl = short(dm.get(r["name"], r["name"]))
                kt[lbl][0] += 1
                kt[lbl][1] += d
                kt[lbl][2].append(r["gx"] * r["gy"] * r["gz"])
                g = kg[(r["graph_id"], lbl)]
                g[0] += 1
                g[1] += d
                g[2].append(r["gx"] * r["gy"] * r["gz"])
    apis = defaultdict(lambda: [0, 0.0])
    lo, hi = steps[0]["b0"], steps[-1]["b1"]
    for x in api:
        if lo <= x["start_ns"] < hi:
            apis[x["name"]][0] += 1
            apis[x["name"]][1] += x["end_ns"] - x["start_ns"]
    iv = [s["b1"] - s["b0"] for s in steps]
    res = {
        "trace": path, "steps": n, "main_graph": main_g, "interval_ms_median": st.median(iv) / 1e6,
        "interval_ms_mean": st.mean(iv) / 1e6, "gpu_busy_ms": st.mean(s["busy"] for s in steps) / 1e6,
        "gpu_idle_ms": st.mean(s["b1"] - s["b0"] - s["busy"] for s in steps) / 1e6,
        "kernels_per_step": sum(v[0] for v in kt.values()) / n,
        "graphs": {str(g): {"launches_per_step": v[0] / n, "nodes_per_step": v[1] / n, "busy_ms": v[2] / n / 1e6} for g, v in sorted(gsum.items())},
        "eager_kernels_per_step": eager[0] / n, "eager_kernel_ms": eager[1] / n / 1e6,
        "copies_per_step": copies[0] / n, "copy_ms": copies[1] / n / 1e6,
        "host_api": {k: {"per_step": c / n, "ms_per_step": t / n / 1e6} for k, (c, t) in sorted(apis.items(), key=lambda x: -x[1][1])[:15]},
        "kernels": sorted([{"label": k, "per_step": v[0] / n, "us_per_call": v[1] / v[0] / 1e3, "ms_per_step": v[1] / n / 1e6,
                            "ctas_median": st.median(v[2])} for k, v in kt.items()], key=lambda r: -r["ms_per_step"]),
        "kernels_by_graph": sorted([{"graph": g, "label": k, "per_step": v[0] / n, "us_per_call": v[1] / v[0] / 1e3,
                                     "ms_per_step": v[1] / n / 1e6, "ctas_median": st.median(v[2])} for (g, k), v in kg.items()],
                                   key=lambda r: (r["graph"], -r["ms_per_step"])),
    }
    json.dump(res, open(out + ".steps.json", "w"), indent=1)
    with open(out + ".step.md", "w") as f:
        f.write(f"# {path}: {n} steps, boundary = graph {main_g}\n\n| item | value |\n|---|---|\n")
        for k in ("interval_ms_median", "interval_ms_mean", "gpu_busy_ms", "gpu_idle_ms", "kernels_per_step",
                  "eager_kernels_per_step", "eager_kernel_ms", "copies_per_step", "copy_ms"):
            f.write(f"| {k} | {res[k]:.4f} |\n")
        f.write("\n| graph id | launches/step | nodes/step | busy ms/step |\n|---|---|---|---|\n")
        for g, v in res["graphs"].items():
            f.write(f"| {g} | {v['launches_per_step']:.2f} | {v['nodes_per_step']:.1f} | {v['busy_ms']:.3f} |\n")
        f.write("\n| kernel | per step | us/call | ms/step | CTAs |\n|---|---|---|---|---|\n")
        for r in res["kernels"]:
            f.write(f"| {r['label']} | {r['per_step']:.2f} | {r['us_per_call']:.2f} | {r['ms_per_step']:.3f} | {r['ctas_median']:.0f} |\n")
        f.write("\n| graph (0 = eager) | kernel | per step | us/call | ms/step | CTAs |\n|---|---|---|---|---|---|\n")
        for r in res["kernels_by_graph"]:
            f.write(f"| {r['graph']} | {r['label']} | {r['per_step']:.2f} | {r['us_per_call']:.2f} | {r['ms_per_step']:.3f} | {r['ctas_median']:.0f} |\n")
        f.write("\n| host API | per step | ms/step |\n|---|---|---|\n")
        for k, v in res["host_api"].items():
            f.write(f"| {k} | {v['per_step']:.2f} | {v['ms_per_step']:.4f} |\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("kernels", "host_api", "kernels_by_graph")}, indent=1))


if __name__ == "__main__":
    main()
