# summ.py <dir> : one line per result of an S2b run folder (oracle, invariance, exec/step timing, realkv, cfgb)
import json, sys, os, glob
d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, '*.jsonl')) + glob.glob(os.path.join(d, 'cfgb', '*.jsonl'))):
    for l in open(f):
        try:
            r = json.loads(l)
        except Exception:
            continue
        t = r.get('test')
        if t == 'oracle':
            print('oracle', r['kind'], r['kv'], r['layout'], 'L', r['L'], 'n', r['n'], 'R', r['R'], 'rel %.2e pre %.2e' % (r['rel_rms_err'], r['pregate_rel_rms_err']),
                  'pass', r['pass'], 'det', r['deterministic_3_runs'], ('vsbf16 %.3e' % r['vs_bf16_truth_rel_rms']) if 'vs_bf16_truth_rel_rms' in r else '')
        elif t == 'invariance':
            print('inv', r['kind'], r['kv'], r['L'], r['bit_identical'], r['detail'])
        elif r.get('probe') == 'cfgb_attn' and t == 'step':
            print('cfgb', r.get('kinds', 'step'), 'R', r['R'], 'n', r['n'], 'L', r['L'], 'r', r['round'], 'tail' if r['sandwich_tail'] else '    ', 'ms %.3f' % r['ms_per_step'],
                  'W %.1f' % r.get('energy_w_mean', 0), 'MHz %.0f' % r.get('sm_mhz_mean', 0), 'cap %s' % r.get('sw_power_cap_frac'), 'GB/s %.0f' % r.get('gbs_span', 0))
        elif t in ('exec', 'step'):
            print(t, r['arm'], r['what'], 'n', r['n'], 'L', r['L'], 'r', r['round'], r['pos'], 'tail' if r['sandwich_tail'] else '    ',
                  'ms %.3f' % r.get('ms_per_graph', float('nan')), 'W %.1f' % r.get('energy_w', r.get('energy_w_mean', 0) or 0),
                  'J %.3f' % (r.get('j_per_call_energy') or 0), 'MHz %s' % r.get('sm_mhz_median', r.get('sm_mhz_mean', '')), 'cap %s' % r.get('sw_power_cap_frac', ''),
                  'GB/s %.0f' % (r.get('rate_span') or 0))
        elif t == 'realkv':
            print('realkv e%02d' % r['exec'], 'local' if r['local'] else 'global', 'cfgb %.3e' % r['cfgb_rel_rms_vs_oracle'],
                  'bf16 %.3e' % r.get('bf16_rel_rms_vs_oracle', -1), 'e4m3 %.3e' % r.get('e4m3_rel_rms_vs_oracle', -1),
                  'ours-vs-cfgb %.3e' % r.get('bf16_rel_rms_vs_cfgb', -1))
        elif t == 'realkv_total':
            print('REALKV_TOTAL', {k: v for k, v in r.items() if k not in ('probe', 'test')})
        elif r.get('probe') == 'cfgb_attn' and t == 'step':
            print('cfgb R', r['R'], 'n', r['n'], 'L', r['L'], 'r', r['round'], 'tail' if r['sandwich_tail'] else '    ', 'ms %.3f' % r['ms_per_step'],
                  'W %.1f' % r.get('energy_w_mean', 0), 'MHz %.0f' % r.get('sm_mhz_mean', 0), 'cap %s' % r.get('sw_power_cap_frac'))
        elif t == 'kernel_info':
            print('kinfo', r)
