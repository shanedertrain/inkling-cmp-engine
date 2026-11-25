# S2b copy (2026-11-23) of probes/p6-attn/cfgb_attn.py with --kinds step|global|swa (graph of 48 visits, or only the
# 8 global / 40 SWA ones) and --data randn|kvgen-like note in the arm line. Otherwise unchanged.
# P6 M3 external reference: config B's own decode attention (rel_decode_attention -> Triton _rel_attention_kernel
# + _rel_decode_combine_kernel on sm_80) launched standalone, kernel only (no prologue, no o-proj), in the same
# timing discipline as the C++ probes (K0 harness: sustained >= 20 s, first 2 s dropped, 10 Hz NVML + energy counter,
# sandwich order, 2 rounds). NOT engine code; written 2026-11-22, never run (no GPU at writing).
#
# Runs inside config B's image with config B's overlay mounted read-only (exactly as serve.sh mounts it):
#   vllm/vllm-openai@sha256:16721463e5ae0c81623c65c50f5b8d4e37710d9bc4f6f4f7512ec1939096272f
#   -v ~/inkling2-cmp170hx/vllm/pkg:/usr/local/lib/python3.14/dist-packages/vllm:ro
# Shapes follow nvidia/attention.py _attention -> _decode_attention: q (T, 24, 128) bf16, paged KV
# (blocks, 64, 8, 128) bf16, sheared rel_logits (T + 128, 24, 1024 + 256), gate (T, 3072), sinks (T, 24); split count
# from plan_decode_attention (the served rule). Content is synthetic (cost-relevant: shapes, masks, splits).
# One step-equivalent = 8 global + 40 SWA calls in visit order, captured in one CUDA graph, distinct KV per call.
import argparse
import json
import os
import statistics
import sys
import threading
import time

import torch
import pynvml

from vllm.models.inkling2.nvidia.ops.attention import plan_decode_attention, rel_decode_attention

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=4)
ap.add_argument('--L', type=int, default=44729)
ap.add_argument('--R', default='1,8', help='verify rows per session (1 = k 0, 8 = k 7)')
ap.add_argument('--secs', type=float, default=24.0)
ap.add_argument('--discard', type=float, default=2.0)
ap.add_argument('--rounds', type=int, default=2)
ap.add_argument('--distinct-global', type=int, default=8)
ap.add_argument('--out', required=True)
ap.add_argument('--kinds', default='step')
ap.add_argument('--kvdump', default='', help='real-KV dump dir: K/V/q of captured executions, tiled (key k <- dump key k %% N)')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = torch.device('cuda')
torch.manual_seed(1)
H, HKV, D, PAGE, EXT = 24, 8, 128, 64, 1024
pynvml.nvmlInit()
NVH = pynvml.nvmlDeviceGetHandleByIndex(0)
OUT = open(os.path.join(args.out, 'cfgb_attn.jsonl'), 'a')


def emit(d):
    OUT.write(json.dumps(d) + '\n')
    OUT.flush()
    print(json.dumps(d), flush=True)


emit({'probe': 'meta', 'test': 'cfgb_attn', 'uuid': pynvml.nvmlDeviceGetUUID(NVH), 'vbios': pynvml.nvmlDeviceGetVbiosVersion(NVH),
      'power_limit_w': pynvml.nvmlDeviceGetPowerManagementLimit(NVH) / 1000, 'torch': torch.__version__, 'argv': sys.argv,
      'tag': os.environ.get('P6_TAG', ''), 'vbios_type': os.environ.get('P6_VBIOS', ''),
      'sms': torch.cuda.get_device_properties(0).multi_processor_count})


class Tele(threading.Thread):  # k0bench.py Tele, unchanged in substance
    def __init__(self, hz=10.0):
        super().__init__(daemon=True)
        self.dt, self.samples, self.stop = 1.0 / hz, [], False

    def run(self):
        nxt = time.monotonic()
        while not self.stop:
            t = time.monotonic()
            try:
                self.samples.append((t, pynvml.nvmlDeviceGetPowerUsage(NVH) / 1000,
                                     pynvml.nvmlDeviceGetClockInfo(NVH, pynvml.NVML_CLOCK_SM),
                                     pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(NVH),
                                     pynvml.nvmlDeviceGetTotalEnergyConsumption(NVH)))
            except Exception:  # noqa: BLE001
                pass
            nxt += self.dt
            time.sleep(max(0.0, nxt - time.monotonic()))

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if len(s) < 3:
            return {'tele_n': len(s)}
        dt = s[-1][0] - s[0][0]
        e = (s[-1][4] - s[0][4]) / 1000.0
        return {'tele_n': len(s), 'power_w_mean': statistics.mean(x[1] for x in s), 'energy_w_mean': e / dt,
                'sm_mhz_mean': statistics.mean(x[2] for x in s), 'sw_power_cap_frac': sum(1 for x in s if x[3] & 0x4) / len(s)}


TELE = Tele()
TELE.start()


def sustained(launch, gb, calls=2):
    torch.cuda.synchronize()
    evs, t0 = [], time.monotonic()
    while time.monotonic() - t0 < args.secs:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(calls):
            launch()
        b.record()
        evs.append((a, b))
        if len(evs) > 3:
            evs[-4][1].synchronize()
    torch.cuda.synchronize()
    t1 = time.monotonic()
    starts = [evs[0][0].elapsed_time(a) for a, _ in evs]
    durs = [a.elapsed_time(b) for a, b in evs]
    keep = [i for i, s in enumerate(starts) if s >= args.discard * 1000]
    span = starts[keep[-1]] + durs[keep[-1]] - starts[keep[0]]
    r = {'chunks_kept': len(keep), 'ms_per_call_median': statistics.median(durs[i] for i in keep) / calls,
         'gbs_span': gb * calls * len(keep) / (span * 1e-3), 'gap_frac': 1 - sum(durs[i] for i in keep) / span}
    r.update(TELE.window(t0 + args.discard, t1))
    if 'energy_w_mean' in r:
        r['j_per_gb_energy'] = r['energy_w_mean'] / r['gbs_span']
    return r


DUMP_G = [5, 11, 17, 23, 29, 35, 41, 47]
DUMP_S = [e for e in range(48) if e not in DUMP_G]
_DC = {}


def dump(e):
    if e not in _DC:
        import numpy as np
        def rd(name, shape):
            a = np.fromfile(os.path.join(args.kvdump, 'e%02d.%s.bin' % (e, name)), dtype=np.uint16)
            return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(dev)
        meta = [json.loads(l) for l in open(os.path.join(args.kvdump, 'meta.jsonl'))][e]
        _DC[e] = (rd('k', (meta['keys'], HKV, D)), rd('v', (meta['keys'], HKV, D)), rd('q', (meta['T'], H, D)))
    return _DC[e]


def make_call(R, is_local, idx):
    n, L = args.n, args.L
    T = n * R
    seqlen = L + R
    nlog = (seqlen + PAGE - 1) // PAGE
    if is_local:
        lo = max(0, L - (EXT - 1)) // PAGE  # blocks the window can touch; the rest map to block 0 (never read)
        nphys_per = nlog - lo + 1
    else:
        lo, nphys_per = 0, nlog
    nblocks = n * nphys_per + 1
    kc = torch.randn(nblocks, PAGE, HKV, D, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(nblocks, PAGE, HKV, D, device=dev, dtype=torch.bfloat16)
    bt = torch.zeros(n, nlog, dtype=torch.int32, device=dev)
    for s in range(n):
        bt[s, lo:nlog] = torch.arange(1 + s * nphys_per, 1 + s * nphys_per + (nlog - lo), dtype=torch.int32)
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    if args.kvdump:
        e = DUMP_G[idx % 8] if not is_local else DUMP_S[idx % 40]
        dk, dv, dq = dump(e)
        N = dk.shape[0]
        for s in range(n):  # same tiling as the C++ bench: session s starts at dump key (sid * 7919) % N
            off = (s * 7919) % N
            keys = (torch.arange(lo * PAGE, nlog * PAGE, device=dev) + off) % N
            blk = 1 + s * nphys_per
            kc[blk:blk + nlog - lo] = dk[keys].view(nlog - lo, PAGE, HKV, D)
            vc[blk:blk + nlog - lo] = dv[keys].view(nlog - lo, PAGE, HKV, D)
        q = dq[torch.arange(T, device=dev) % R].contiguous()
    rel = torch.randn(T + 128, H, EXT + 256, device=dev, dtype=torch.bfloat16)
    gate = torch.randn(T, H * D, device=dev, dtype=torch.bfloat16)
    sinks = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    out = torch.empty(T, H * D, device=dev, dtype=torch.bfloat16)
    seql = torch.full((n,), seqlen, dtype=torch.int32, device=dev)
    cu = torch.arange(0, (n + 1) * R, R, dtype=torch.int32, device=dev)
    splits = plan_decode_attention(is_local, n, R, H, HKV, EXT)
    kv_bytes = 0
    for s in range(n):  # keys the kernel touches: whole BLOCK_N = 128 blocks from n_lo to the last row
        first = max(0, L - (EXT - 1)) // 128 * 128 if is_local else 0
        kv_bytes += (((seqlen + 127) // 128) * 128 - first) * HKV * D * 2 * 2

    def call():
        rel_decode_attention(q, kc, vc, rel, gate, sinks, out, block_table=bt, cache_seqlens=seql, cu_seqlens_q=cu,
                             max_query_len=R, num_actual_tokens=T, softmax_scale=D ** -0.5, is_local=is_local,
                             num_splits=splits)
    return call, splits, kv_bytes


def visit_kinds():
    layers = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    return [l in (5, 11, 17, 23) for l in layers]


arms = []
keep = []
for R in [int(x) for x in args.R.split(',')]:
    calls, gb, splits = [], 0.0, {}
    gi = wi = 0
    glob = [make_call(R, False, i) for i in range(args.distinct_global)]
    loc = [make_call(R, True, i) for i in range(40)]
    for g in visit_kinds():
        if (args.kinds == 'global' and not g) or (args.kinds == 'swa' and g):
            gi += g
            wi += not g
            continue
        c, sp, b = glob[gi % len(glob)] if g else loc[wi]
        calls.append(c)
        gb += b / 1e9
        splits['global' if g else 'swa'] = sp
        gi += g
        wi += not g
    for c in calls:  # compile and warm up outside capture
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    # keep every input tensor alive for the graph's life: the closures in glob/loc own them (CHANGES-during-run.md #4)
    keep.append((glob, loc, calls))
    arms.append((R, graph, gb, splits))
    emit({'probe': 'cfgb_attn', 'test': 'arm', 'R': R, 'n': args.n, 'L': args.L, 'kv_gb_step': gb, 'splits': splits, 'kinds': args.kinds,
          'kvdump': args.kvdump})

order = []
for r in range(args.rounds):
    mid = list(range(1, len(arms)))
    if r % 2:
        mid.reverse()
    order += [(r, 0, False)] + [(r, i, False) for i in mid] + ([(r, 0, True)] if len(arms) > 1 else [])
for rnd, i, tail in order:
    R, graph, gb, splits = arms[i]
    res = sustained(graph.replay, gb)
    res.update({'probe': 'cfgb_attn', 'test': 'step', 'R': R, 'n': args.n, 'L': args.L, 'round': rnd, 'sandwich_tail': tail,
                'kv_gb_step': gb, 'splits': splits, 'ms_per_step': res['ms_per_call_median'], 'kinds': args.kinds,
                'note': 'config B decode attention kernels only (rel_attention + combine), no prologue'})
    emit(res)
TELE.stop = True
