# S4 prefill MoE: formal timing tables from bench.cu --mode time JSON lines (python3 -I analyse.py F_time_L0.jsonl ...).
# Per (cell, group, round): x baseline = ours / mean(baseline head, baseline tail) for ms (inverted), TFLOPS, J/token;
# baseline drift = tail / head - 1. Prints markdown.
import json
import sys
from collections import defaultdict

rows = []
for path in sys.argv[1:]:
    for line in open(path):
        d = json.loads(line)
        if d.get('probe') == 'time' and d.get('ok'):
            rows.append(d)
pair = {'ours': 'humming', 'ours_gemm': 'humming_gemm'}
blocks = defaultdict(list)       # (cell, group, round) -> rows in order
for d in rows:
    grp = 'full' if d['test'] in ('ours', 'humming') else 'gemm'
    blocks[(d['cell'], grp, d['round'])].append(d)
out = defaultdict(list)
for (cell, grp, rnd), bl in sorted(blocks.items()):
    base = [b for b in bl if b['test'].startswith('humming')]
    ours = [b for b in bl if b['test'].startswith('ours')]
    if len(base) != 2 or not ours:
        continue
    h, t = base
    for o in ours:
        bt = (h['tflops'] + t['tflops']) / 2
        bj = (h['j_per_token'] + t['j_per_token']) / 2
        bms = (h['ms_per_call'] + t['ms_per_call']) / 2
        out[(cell, grp, o['variant'])].append(dict(o=o, h=h, t=t, x_tflops=o['tflops'] / bt, x_ms=bms / o['ms_per_call'],
                                     x_j=bj / o['j_per_token'], drift=t['tflops'] / h['tflops'] - 1, bt=bt, bms=bms, bj=bj))
print('| Cell | Arms | Variant | ours ms | Humming ms | ours TFLOPS | Humming TFLOPS | x Humming (TFLOPS) r0 / r1 / mean | ours J/token | Humming J/token | x (J) | ours W / MHz / cap | Humming W / MHz | Humming drift r0 / r1 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for (cell, grp, var), lst in sorted(out.items(), key=lambda kv: (kv[0][0].split('/')[0], int(kv[0][0].split('Q')[1]), kv[0][1], kv[0][2])):
    m = lambda k: sum(x[k] for x in lst) / len(lst)
    mo = lambda k: sum(x['o'][k] for x in lst) / len(lst)
    mh = lambda k: sum((x['h'][k] + x['t'][k]) / 2 for x in lst) / len(lst)
    xs = ' / '.join(f"{x['x_tflops']:.3f}" for x in lst)
    dr = ' / '.join(f"{100 * x['drift']:+.1f} %" for x in lst)
    print(f"| {cell} | {grp} | {var} | {mo('ms_per_call'):.3f} | {m('bms'):.3f} | {mo('tflops'):.1f} | {m('bt'):.1f} | {xs} / **{m('x_tflops'):.3f}** | "
          f"{mo('j_per_token') * 1e3:.4f} mJ | {m('bj') * 1e3:.4f} mJ | {m('x_j'):.3f} | {mo('power_w_mean'):.1f} / {mo('sm_mhz_mean'):.0f} / "
          f"{mo('sw_power_cap_frac'):.2f} | {mh('power_w_mean'):.1f} / {mh('sm_mhz_mean'):.0f} | {dr} |")
