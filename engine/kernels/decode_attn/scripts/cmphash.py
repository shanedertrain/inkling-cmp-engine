# cmphash.py <pre.jsonl> <post.jsonl> : per-execution comparison of realkv output bit fingerprints (bf16 out / fp32 o)
import json, sys
def load(f):
    r = {}
    for l in open(f):
        try: d = json.loads(l)
        except Exception: continue
        if d.get('test') == 'realkv': r[d['exec']] = d
    return r
a, b = load(sys.argv[1]), load(sys.argv[2])
for key in ('bf16_out_hash', 'e4m3_out_hash'):
    same = diff = 0; bad = []
    for e in sorted(a):
        if key not in a[e] or e not in b or key not in b[e]: continue
        if a[e][key] == b[e][key]: same += 1
        else: diff += 1; bad.append((e, a[e]['local']))
    print(key, 'identical', same, 'different', diff, bad[:10])
