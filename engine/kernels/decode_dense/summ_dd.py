# decode_dense: tables from dd_bench / dd_torch JSON lines (own code, stdlib).
#   python3 -I summ_dd.py sweep sweep.jsonl [dd_torch.jsonl]   per op x rows: variant ms / J per step (median over rounds)
#   python3 -I summ_dd.py algos algos.jsonl                    per shape x rows: engine plan, best BI-eligible, gemmEx, skinny
#   python3 -I summ_dd.py bits bits.jsonl
import collections
import json
import statistics
import sys


def lines(paths):
    for p in paths:
        for ln in open(p):
            ln = ln.strip()
            if ln.startswith('{'):
                yield json.loads(ln)


def sweep(paths):
    cell = collections.defaultdict(list)
    for d in lines(paths):
        if d.get('test') != 'sweep' or not d.get('ok', True):
            continue
        cell[(d['op'], d['rows'], d['variant'])].append(d)
    ops = sorted({(k[0], k[1]) for k in cell}, key=lambda x: (x[0], x[1]))
    print('| op | rows | variant | n | ms/step median (min-max over rounds) | J/step | GB/s | W | SM MHz |')
    print('|---|---|---|---|---|---|---|---|---|')
    for op, m in ops:
        for (o, r, v), ds in sorted(cell.items()):
            if (o, r) != (op, m):
                continue
            ms = [d['ms_per_call_median'] for d in ds]
            js = [d['j_per_call_energy'] for d in ds if d.get('j_per_call_energy') is not None]
            gb = ds[0].get('bytes_per_call', 0) / 1e9
            w = [d.get('energy_w_mean') for d in ds if d.get('energy_w_mean')]
            mhz = [d.get('sm_mhz_mean', d.get('sm_mhz_end')) for d in ds if d.get('sm_mhz_mean', d.get('sm_mhz_end'))]
            print(f"| {o} | {r} | {v} | {len(ds)} | {statistics.median(ms):.3f} ({min(ms):.3f}-{max(ms):.3f}) | "
                  f"{statistics.median(js):.4f} | {gb / (statistics.median(ms) * 1e-3):.0f} | "
                  f"{statistics.median(w) if w else float('nan'):.0f} | {statistics.median(mhz) if mhz else float('nan'):.0f} |")


def algos(paths):
    cell = collections.defaultdict(list)
    for d in lines(paths):
        if d.get('test') != 'algo' or not d.get('ok', True):
            continue
        cell[(d['shape'], d['rows'])].append(d)
    print('| shape | N x K | rows | engine plan us/call | best split-K-1, ws 0 (algo/tile/stages) us | best any us (splitK) | gemmEx default us | skinny us | sk2 us | candidates |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for (s, r), ds in sorted(cell.items(), key=lambda x: (x[0][0], x[0][1])):
        per = lambda d: d['ms_per_call_median'] * 1000 / d['calls_per_graph']
        g = {d['kind']: d for d in ds if d['kind'] != 'lt'}
        lt = [d for d in ds if d['kind'] == 'lt']
        bi = [d for d in lt if d['splitk'] <= 1 and d['ws_bytes'] == 0]
        b = min(bi, key=per) if bi else None
        a = min(lt, key=per) if lt else None
        f = lambda k: f"{per(g[k]):.1f}" if k in g else '-'
        bs = f"{per(b):.1f} ({b['algo_id']}/{b['tile']}/{b['stages']})" if b else '-'
        as_ = f"{per(a):.1f} (sk {a['splitk']})" if a else '-'
        print(f"| {s} | {ds[0]['N']} x {ds[0]['K']} | {r} | {f('engine_lt')} | {bs} | {as_} | {f('gemmex_default')} | "
              f"{f('skinny')} | {f('sk2')} | {len(lt)} |")


def bits(paths):
    print('| op | visit | layer | separate inputs | variant | rows | differing elements vs engine path | rows 0-7 vs M=8 |')
    print('|---|---|---|---|---|---|---|---|')
    for d in lines(paths):
        if d.get('test') != 'bits':
            continue
        print(f"| {d['op']} | {d.get('visit', '-')} | {d.get('layer', '-')} | {d.get('separate_inputs', '-')} | {d['variant']} | "
              f"{d['rows']} | {d.get('differ_vs_skinny', d.get('differ_vs_lt'))} / {d['elements']} | {d.get('rows0_7_differ_vs_M8', '-')} |")


if __name__ == '__main__':
    {'sweep': sweep, 'algos': algos, 'bits': bits}[sys.argv[1]](sys.argv[2:])
