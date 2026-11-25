# analyse_grid.py <dir>... : S2b final grid table from fin-* run folders (config B cfgb/cfgb_attn.jsonl, ours step.jsonl)
# Per cell: config B k=0 / k=7 (both processes, both rounds; sandwich tails included for k=0), ours per arm; median
# (min-max) ms per step-equivalent, mean J per step (energy counter: energy_w_mean * ms), mean SM MHz, cap fraction,
# ratio ours / config B on medians. Labels: measured (M) for medians, derived (D) for ratios.
import glob
import json
import os
import statistics
import sys


def load(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def summ(vals):
    return statistics.median(vals), min(vals), max(vals)


rows = []
for d in sys.argv[1:]:
    cell = os.path.basename(d.rstrip('/'))
    cb = [r for r in load(os.path.join(d, 'cfgb', 'cfgb_attn.jsonl')) if r.get('test') == 'step']
    ours = [r for r in load(os.path.join(d, 'step.jsonl')) if r.get('test') == 'step' and r.get('ok', True)]
    res = {}
    for R in (1, 8):
        xs = [r for r in cb if r['R'] == R]
        if xs:
            ms = [r['ms_per_step'] for r in xs]
            j = [r['energy_w_mean'] * r['ms_per_step'] / 1000 for r in xs if 'energy_w_mean' in r]
            res['cfgB k=%d' % (R - 1)] = (summ(ms), statistics.mean(j) if j else float('nan'),
                                          statistics.mean(r.get('sm_mhz_mean', 0) for r in xs),
                                          statistics.mean(r.get('sw_power_cap_frac', 0) for r in xs), len(xs))
    for arm in sorted({r['arm'] for r in ours}):
        xs = [r for r in ours if r['arm'] == arm]
        ms = [r['ms_per_graph'] for r in xs]
        j = [r['energy_w_mean'] * r['ms_per_graph'] / 1000 for r in xs if r.get('energy_w_mean')]
        res[arm] = (summ(ms), statistics.mean(j) if j else float('nan'),
                    statistics.mean(r.get('sm_mhz_median', 0) or 0 for r in xs),
                    statistics.mean(r.get('sw_power_cap_frac', 0) or 0 for r in xs), len(xs))
    rows.append((cell, res))

print('| cell | arm | runs | ms median (min-max) | J/step | SM MHz | cap | vs config B (same k) |')
print('|---|---|---|---|---|---|---|---|')
for cell, res in rows:
    base = {0: res.get('cfgB k=0'), 7: res.get('cfgB k=7')}
    for arm, ((med, lo, hi), j, mhz, cap, n) in res.items():
        k = 0 if (arm == 'cfgB k=0' or ':1:' in arm) else 7
        b = base.get(k)
        ratio = '' if arm.startswith('cfgB') or b is None else '%.3f' % (med / b[0][0])
        print('| %s | %s | %d | %.3f (%.3f-%.3f) | %.3f | %.0f | %.2f | %s |' % (cell, arm, n, med, lo, hi, j, mhz, cap, ratio))
