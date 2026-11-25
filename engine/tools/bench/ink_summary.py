#!/usr/bin/env python3
"""Summarize one or more ink-server replay summary JSONs (from replay/tools/analyze.py)."""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: ink_summary.py SUMMARY.json [SUMMARY2.json ...]", file=sys.stderr)
        return 2
    hdr = f"{'run/cell':36s} {'M1':>6s} {'wall':>6s} {'J/t':>5s} {'W':>5s} {'MHz':>6s} {'Tmax':>4s} {'p50/p90/p99':>16s} {'cach':>6s} {'err':>3s}"
    print(hdr)
    print("-" * len(hdr))
    for path in sys.argv[1:]:
        rows = json.load(open(path))
        if isinstance(rows, dict):
            rows = [rows]
        for r in rows:
            name = r.get("dir", path).replace("out/", "").replace("/replay/", "/")
            p50 = r.get("ttft_p50", 0.0)
            p90 = r.get("ttft_p90", 0.0)
            p99 = r.get("ttft_p99", 0.0)
            print(
                f"{name:36s} {r.get('M1_tok_s', 0.0):6.2f} {r.get('wall_s', 0.0):5.1f}s "
                f"{r.get('j_per_token', 0.0):5.3f} {r.get('power_w_mean', 0.0):5.1f} "
                f"{r.get('sm_mhz_mean', 0.0):6.1f} {r.get('temp_c_max', 0.0):3.0f}C "
                f"{p50:4.2f}/{p90:4.2f}/{p99:5.2f}s {r.get('cached_share', 0.0):6.4f} {r.get('errors', 0):3d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
