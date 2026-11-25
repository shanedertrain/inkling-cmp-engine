#!/usr/bin/env python3
"""analyze.py TRACE.csv OUT_PREFIX [--skip S] [--steps N] [--graph-only]: per-step breakdown of a CUPTI step trace
(cupti_trace.cpp) of ink_bench / ink_spec / any process that replays one CUDA graph per step.

A step = one cudaGraphLaunch of the main graph (correlation id shared by its kernel nodes) plus the GPU work queued
outside the graph before the next launch (forward() table copies, commit kernel, ...). Writes:
  OUT.summary.json   per-step means: wall, graph span, kernel busy, in-graph gaps, outside-graph GPU work, GPU idle
                     between steps, host API time by function, kernel count; per-kernel-class table
  OUT.kernels.md     per kernel (demangled, with the GEMM context label) count per step, us per call, ms per step, CTAs
  OUT.nodes.csv      the node list of the median step for floor_bench (kind,gx,gy,gz,bx,by,bz,smem,bytes)
  OUT.gaps.csv       per in-graph node of the median step: name, start offset, duration, gap before
"""
import bisect
import csv
import json
import statistics as stx
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
    """compact kernel label from a demangled name"""
    s = d.split("(")[0]
    if s.startswith("void "):
        s = s[5:]
    # strip template args but keep the first level for our own kernels
    base = s.split("<")[0]
    return base.strip()


def main():
    a = sys.argv[1:]
    path, out = a[0], a[1]
    skip = int(a[a.index("--skip") + 1]) if "--skip" in a else 5
    nmax = int(a[a.index("--steps") + 1]) if "--steps" in a else 10**9
    recs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("start_ns", "end_ns", "corr", "graph_id", "node_id", "stream", "gx", "gy", "gz", "bx", "by", "bz",
                      "smem_static", "smem_dyn", "regs", "bytes"):
                r[k] = int(r[k])
            recs.append(r)
    names = sorted({r["name"] for r in recs if r["kind"] == "K"})
    dm = demangle(names)
    gpu = sorted([r for r in recs if r["kind"] in "KCS"], key=lambda r: r["start_ns"])
    api = sorted([r for r in recs if r["kind"] == "R"], key=lambda r: r["start_ns"])
    api_t = [x["start_ns"] for x in api]
    # graph launches: runtime API cudaGraphLaunch records, matched to the graph kernels by correlation id
    gk = defaultdict(list)
    for r in gpu:
        if r["graph_id"]:
            gk[r["corr"]].append(r)
    launches = [x for x in api if x["name"].startswith("cudaGraphLaunch") and x["corr"] in gk]
    # main graph = the most frequent graph id among launches (ink_bench replays one graph per step)
    gid_count = defaultdict(int)
    for x in launches:
        gid_count[gk[x["corr"]][0]["graph_id"]] += 1
    main_gid = max(gid_count, key=gid_count.get)
    launches = [x for x in launches if gk[x["corr"]][0]["graph_id"] == main_gid]
    launches = launches[skip:skip + nmax + 1]
    steps = []
    gi = 0
    for i in range(len(launches) - 1):
        L, Ln = launches[i], launches[i + 1]
        nodes = sorted(gk[L["corr"]], key=lambda r: r["start_ns"])
        nodes_next = sorted(gk[Ln["corr"]], key=lambda r: r["start_ns"])
        g0, g1 = nodes[0]["start_ns"], nodes[-1]["end_ns"]
        busy = sum(r["end_ns"] - r["start_ns"] for r in nodes)
        gaps = [max(0, nodes[j + 1]["start_ns"] - nodes[j]["end_ns"]) for j in range(len(nodes) - 1)]
        # outside-graph GPU work between this graph's end and the next graph's start
        while gi < len(gpu) and gpu[gi]["start_ns"] < g1:
            gi += 1
        j = gi
        outside = []
        while j < len(gpu) and gpu[j]["start_ns"] < nodes_next[0]["start_ns"]:
            if not gpu[j]["graph_id"]:
                outside.append(gpu[j])
            j += 1
        out_busy = sum(r["end_ns"] - r["start_ns"] for r in outside)
        idle = nodes_next[0]["start_ns"] - g1 - out_busy
        # host API between the two launches
        apis = api[bisect.bisect_left(api_t, L["start_ns"]):bisect.bisect_left(api_t, Ln["start_ns"])]
        abyf = defaultdict(lambda: [0, 0])
        for x in apis:
            abyf[x["name"]][0] += 1
            abyf[x["name"]][1] += x["end_ns"] - x["start_ns"]
        sync = [x for x in apis if x["name"].startswith("cudaStreamSynchronize")]
        sync_ret = sync[-1]["end_ns"] if sync else None
        steps.append({
            "wall_ns": Ln["start_ns"] - L["start_ns"],
            "graph_span_ns": g1 - g0,
            "busy_ns": busy,
            "gap_ns": sum(gaps),
            "gaps": gaps,
            "nodes": nodes,
            "n_kernels": sum(1 for r in nodes if r["kind"] == "K"),
            "n_copies": sum(1 for r in nodes if r["kind"] != "K"),
            "outside": outside,
            "outside_busy_ns": out_busy,
            "gpu_idle_between_ns": idle,
            "launch_to_first_node_ns": g0 - L["start_ns"],
            "host_after_sync_ns": (Ln["start_ns"] - sync_ret) if sync_ret else None,
            "api": {k: v for k, v in abyf.items()},
            "launch_api_ns": L["end_ns"] - L["start_ns"],
        })
    if not steps:
        sys.exit("no steps found")
    ms = lambda ns: ns / 1e6
    def mean(k):
        v = [s[k] for s in steps if s[k] is not None]
        return stx.mean(v) if v else None
    walls = sorted(s["wall_ns"] for s in steps)
    med = min(steps, key=lambda s: abs(s["wall_ns"] - walls[len(walls) // 2]))
    # per-kernel table with GEMM context labels: a library kernel takes "lt<-<previous own kernel>"
    def own(lbl):
        return any(lbl.startswith(p) for p in ("glue::", "pglue::", "pmoe::", "moedev::", "da::", "projs::", "ink::", "humdev::",
                                               "argmax", "accept", "last_feat", "gather_feats", "verify", "rmsnorm_f32out",
                                               "softcap", "proj_split"))
    def labels(nodes):
        out, prev = [], "start"
        for r in nodes:
            if r["kind"] != "K":
                out.append(f"{r['kind']}:{r['name']}")
                continue
            lbl = short(dm.get(r["name"], r["name"]))
            if own(lbl):
                prev = lbl.split("::")[-1]
                out.append(lbl)
            else:
                out.append(f"lt<-{prev}")
        return out
    agg = defaultdict(lambda: {"n": 0, "ns": 0, "ctas": [], "thr": 0, "gap_ns": 0, "raw": set()})
    for s in steps:
        lb = labels(s["nodes"])
        for k, (r, l) in enumerate(zip(s["nodes"], lb)):
            a_ = agg[l]
            a_["n"] += 1
            a_["ns"] += r["end_ns"] - r["start_ns"]
            a_["ctas"].append(r["gx"] * r["gy"] * r["gz"])
            a_["thr"] = r["bx"] * r["by"] * r["bz"]
            a_["gap_ns"] += s["gaps"][k - 1] if k > 0 else 0
            if r["kind"] == "K":
                a_["raw"].add(short(dm.get(r["name"], r["name"]))[:60])
    ns_ = len(steps)
    rows = []
    for l, v in agg.items():
        rows.append({"label": l, "per_step": v["n"] / ns_, "us_per_call": v["ns"] / v["n"] / 1e3,
                     "ms_per_step": v["ns"] / ns_ / 1e6, "gap_before_ms_per_step": v["gap_ns"] / ns_ / 1e6,
                     "ctas_median": stx.median(v["ctas"]), "threads": v["thr"], "kernels": sorted(v["raw"])})
    rows.sort(key=lambda r: -r["ms_per_step"])
    # outside-graph work by name
    og = defaultdict(lambda: [0, 0])
    for s in steps:
        for r in s["outside"]:
            k = f"{r['kind']}:{short(dm.get(r['name'], r['name'])) if r['kind'] == 'K' else r['name']}"
            og[k][0] += 1
            og[k][1] += r["end_ns"] - r["start_ns"]
    apisum = defaultdict(lambda: [0, 0])
    for s in steps:
        for k, (c, t) in s["api"].items():
            apisum[k][0] += c
            apisum[k][1] += t
    summ = {
        "trace": path, "steps": ns_, "skip": skip, "graph_id": main_gid,
        "wall_ms": ms(mean("wall_ns")), "wall_ms_median": ms(walls[len(walls) // 2]),
        "graph_span_ms": ms(mean("graph_span_ns")), "kernel_busy_ms": ms(mean("busy_ns")), "in_graph_gap_ms": ms(mean("gap_ns")),
        "graph_nodes": med["n_kernels"] + med["n_copies"], "graph_kernels": med["n_kernels"], "graph_copies": med["n_copies"],
        "in_graph_gap_us_median": stx.median(med["gaps"]) / 1e3, "in_graph_gap_us_mean": stx.mean(med["gaps"]) / 1e3,
        "outside_graph_gpu_ms": ms(mean("outside_busy_ns")), "outside_graph_ops_per_step": stx.mean(len(s["outside"]) for s in steps),
        "gpu_idle_between_graphs_ms": ms(mean("gpu_idle_between_ns")),
        "launch_api_to_first_node_ms": ms(mean("launch_to_first_node_ns")),
        "host_after_sync_to_next_launch_ms": ms(mean("host_after_sync_ns")) if mean("host_after_sync_ns") is not None else None,
        "outside_ops": {k: {"per_step": c / ns_, "ms_per_step": t / ns_ / 1e6} for k, (c, t) in sorted(og.items(), key=lambda x: -x[1][1])},
        "host_api": {k: {"per_step": c / ns_, "ms_per_step": t / ns_ / 1e6} for k, (c, t) in sorted(apisum.items(), key=lambda x: -x[1][1])},
        "kernels": rows,
    }
    with open(out + ".summary.json", "w") as f:
        json.dump(summ, f, indent=1)
    with open(out + ".kernels.md", "w") as f:
        f.write(f"# {path}: {ns_} steps (median wall {summ['wall_ms_median']:.3f} ms)\n\n")
        f.write("| step part | ms |\n|---|---|\n")
        for k in ("wall_ms", "graph_span_ms", "kernel_busy_ms", "in_graph_gap_ms", "outside_graph_gpu_ms",
                  "gpu_idle_between_graphs_ms", "launch_api_to_first_node_ms", "host_after_sync_to_next_launch_ms"):
            v = summ[k]
            f.write(f"| {k} | {v:.4f} |\n" if v is not None else f"| {k} | - |\n")
        f.write(f"\ngraph nodes {summ['graph_nodes']} (kernels {summ['graph_kernels']}), in-graph gap median "
                f"{summ['in_graph_gap_us_median']:.2f} us, mean {summ['in_graph_gap_us_mean']:.2f} us\n\n")
        f.write("| kernel (lt = cuBLASLt after) | per step | us/call | ms/step | gap before ms/step | CTAs | threads |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['label']} | {r['per_step']:.1f} | {r['us_per_call']:.2f} | {r['ms_per_step']:.3f} | "
                    f"{r['gap_before_ms_per_step']:.3f} | {r['ctas_median']:.0f} | {r['threads']} |\n")
        f.write("\n| outside graph | per step | ms/step |\n|---|---|---|\n")
        for k, v in summ["outside_ops"].items():
            f.write(f"| {k} | {v['per_step']:.1f} | {v['ms_per_step']:.4f} |\n")
        f.write("\n| host API (between launches) | per step | ms/step |\n|---|---|---|\n")
        for k, v in summ["host_api"].items():
            f.write(f"| {k} | {v['per_step']:.1f} | {v['ms_per_step']:.4f} |\n")
    with open(out + ".nodes.csv", "w") as f:
        f.write("kind,gx,gy,gz,bx,by,bz,smem,bytes\n")
        for r in med["nodes"]:
            f.write(f"{r['kind']},{r['gx']},{r['gy']},{r['gz']},{r['bx']},{r['by']},{r['bz']},{r['smem_static'] + r['smem_dyn']},{r['bytes']}\n")
    with open(out + ".gaps.csv", "w") as f:
        f.write("i,label,start_us,dur_us,gap_before_us,ctas,threads\n")
        lb = labels(med["nodes"])
        t0 = med["nodes"][0]["start_ns"]
        for k, (r, l) in enumerate(zip(med["nodes"], lb)):
            f.write(f"{k},{l},{(r['start_ns'] - t0) / 1e3:.2f},{(r['end_ns'] - r['start_ns']) / 1e3:.2f},"
                    f"{(med['gaps'][k - 1] / 1e3) if k else 0:.2f},{r['gx'] * r['gy'] * r['gz']},{r['bx'] * r['by'] * r['bz']}\n")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("kernels", "outside_ops", "host_api")}, indent=1))


if __name__ == "__main__":
    main()
