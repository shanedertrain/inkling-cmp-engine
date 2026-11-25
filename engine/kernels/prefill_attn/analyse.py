"""S4 prefill attention: results tables from the raw jsonl of one formal run folder (ours + config B before/after).
Usage: python3 analyse.py RUN_DIR > RESULTS-tables.md      (RUN_DIR holds ours.jsonl, cfgbA/, cfgbB/, check.jsonl, inv.jsonl)
Everything printed is derived from the raw lines; config B's reference = mean of its two passes (A before, B after ours),
both shown. ms = median per-call time of a >= 22 s window; J/new token = energy counter mean W x busy-span time / Q."""
import json
import os
import sys

R = sys.argv[1]


def lines(p):
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p) if x.strip().startswith('{')]


ours = [d for d in lines(f'{R}/ours.jsonl') if d.get('test') == 'time' and d.get('ok')]
nosp = [d for d in lines(f'{R}/ours_nosplit.jsonl') if d.get('test') == 'time' and d.get('ok')]
v1 = [d for d in lines(f'{R}/ours_v1.jsonl') if d.get('test') == 'time' and d.get('ok')]
ceil = [d for d in lines(f'{R}/ceil.jsonl') if d.get('test') == 'time' and d.get('ok')]
cbA = [d for d in lines(f'{R}/cfgbA/cfgb_prefill.jsonl') if d.get('test') == 'time']
cbB = [d for d in lines(f'{R}/cfgbB/cfgb_prefill.jsonl') if d.get('test') == 'time']


def pick(rows, kind, C, Q, arm=None):
    r = [d for d in rows if d['kind'] == kind and d['C'] == C and d['Q'] == Q and (arm is None or d.get('arm') == arm)]
    return r[0] if r else None


cells = sorted({(d['kind'], d['C'], d['Q']) for d in ours}, key=lambda x: (x[0], x[2] != 2048, x[2], x[1]))
print('## Ours (4x2p1s1m2; global split 8,192 geo, SWA unsplit) vs config B (rel_attention + gate_sink), GPU 4, 140 W\n')
print('| kind | C | Q | ours ms | ours TFLOPS | ours MHz | ours W | ours J/new tok | cfgB ms A / B | cfgB TFLOPS | cfgB MHz | '
      'cfgB J/new tok | **x config B (time)** | x config B (J) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for kind, C, Q in cells:
    o = pick(ours, kind, C, Q)
    a, b = pick(cbA, kind, C, Q, 'pipe'), pick(cbB, kind, C, Q, 'pipe')
    if not (a and b):
        continue
    cms = (a['ms_per_chunk'] + b['ms_per_chunk']) / 2
    cj = (a['j_per_new_token'] + b['j_per_new_token']) / 2
    print(f"| {kind} | {C:,} | {Q:,} | {o['ms_per_chunk']:.3f} | {o['tflops']:.1f} | {o['sm_mhz_median']} | "
          f"{o['energy_w_mean']:.1f} | {o['j_per_new_token'] * 1e3:.3f} m | {a['ms_per_chunk']:.3f} / {b['ms_per_chunk']:.3f} | "
          f"{o['flops'] / (cms * 1e-3) / 1e12:.1f} | {a['sm_mhz_median']} / {b['sm_mhz_median']} | {cj * 1e3:.3f} m | "
          f"**{cms / o['ms_per_chunk']:.3f}** | {cj / o['j_per_new_token']:.3f} |")
print('\nconfig B arm `attn` (rel_attention alone, pass A) for reference:\n')
print('| kind | C | Q | attn ms | pipe ms | gate_sink share |')
print('|---|---|---|---|---|---|')
for kind, C, Q in cells:
    a, p = pick(cbA, kind, C, Q, 'attn'), pick(cbA, kind, C, Q, 'pipe')
    if a and p:
        print(f"| {kind} | {C:,} | {Q:,} | {a['ms_per_chunk']:.3f} | {p['ms_per_chunk']:.3f} | "
              f"{(p['ms_per_chunk'] - a['ms_per_chunk']) / p['ms_per_chunk'] * 100:.1f} % |")
if nosp or v1 or ceil:
    print('\n## Ablation arms (same run, ours only)\n')
    print('| arm | kind | C | Q | ms | TFLOPS | MHz | final arm ms | final / arm speed-up |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name, rows in (('4x2p1s1m2 no split', nosp), ('8x1p0s2m1 (v1 shape) no split', v1),
                       ('ceiling: softmax/bias/mask removed (wrong results)', ceil)):
        for d in rows:
            o = pick(ours, d['kind'], d['C'], d['Q'])
            print(f"| {name} | {d['kind']} | {d['C']:,} | {d['Q']:,} | {d['ms_per_chunk']:.3f} | {d['tflops']:.1f} | "
                  f"{d['sm_mhz_median']} | {o['ms_per_chunk']:.3f} | {d['ms_per_chunk'] / o['ms_per_chunk']:.3f} |")
chk = [d for d in lines(f'{R}/check.jsonl') + lines(f'{R}/check_nosplit.jsonl') if d.get('test') == 'oracle']
if chk:
    print('\n## FP64 oracle (sampled rows x 3,072 outputs; synthetic data)\n')
    print('| kind | C | Q | split | rows | det. 3 runs | ours out rel RMS | cfgB out rel RMS | bf16 floor | ours o32 rel RMS | '
          'cfgB o32 rel RMS | ours lse max abs | cfgB lse max abs | bit-equal to cfgB | pass |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for d in chk:
        print(f"| {d['kind']} | {d['C']:,} | {d['Q']:,} | {d.get('split', 0)} | {d['rows']} | {d['deterministic_3runs']} | "
              f"{d['ours_out_rel_rms']:.3e} | {d.get('cfgb_out_rel_rms', float('nan')):.3e} | {d['bf16_floor_out_rel_rms']:.3e} | "
              f"{d['ours_o32_rel_rms']:.3e} | {d.get('cfgb_o32_rel_rms', float('nan')):.3e} | {d['ours_lse_max_abs']:.2e} | "
              f"{d.get('cfgb_lse_max_abs', float('nan')):.2e} | {d.get('ours_vs_cfgb_out_bit_equal_frac', float('nan')) * 100:.2f} % | "
              f"{d['pass']} |")
inv = [d for d in lines(f'{R}/inv.jsonl') if d.get('test') in ('inv',) or 'check' in d and 'bit_identical' in d]
if inv:
    print('\n## Invariance (bit compare against one 8,192-row chunk at past C)\n')
    print('| kind | C | test | detail | bit-identical |')
    print('|---|---|---|---|---|')
    for d in inv:
        t = d.get('check', d['test'])
        det = d.get('chunk', d.get('var2', d.get('page2', '')))
        if t == 'pages':
            det = f"page {d['page2']}, layout {d['layout2']}"
        print(f"| {d['kind']} | {d['C']:,} | {t} | {det} | {d['bit_identical']} {d.get('diff', '')} |")
