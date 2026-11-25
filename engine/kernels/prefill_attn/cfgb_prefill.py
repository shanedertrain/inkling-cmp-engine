# S4 baseline: config B's own prefill attention launched standalone on the data set written by `bench --mode gen`.
# Config B's sm_80 prefill path (models/inkling2/nvidia/attention.py `_attention` -> `_fa4_attention` ->
# `rel_attention(..., num_splits=1, mode=MODE_OUT_LSE)` (Triton `_rel_attention_kernel`, BLOCK_M 128, BLOCK_N 64,
# 8 warps, 2 stages) then `gate_sink` (Triton `_gate_sink_kernel`) @ 6e9d1cce). Kernel only: no prologue (K/V convs,
# clips, rel-logit GEMV, cache write), no o-projection. Not engine code; our own script.
#
# Runs inside config B's image with a copy of config B's overlay mounted read-only (as probes/p6-attn/cfgb_attn.py):
#   vllm/vllm-openai@sha256:16721463e5ae0c81623c65c50f5b8d4e37710d9bc4f6f4f7512ec1939096272f
#   -v ~/inkling-cmp-engine/p6-attn/cfgb-pkg:/usr/local/lib/python3.14/dist-packages/vllm:ro
# Data: K.bin / V.bin [N][8][128] bf16 by absolute position; qg.bin / ql.bin [8192][24][128]; rel.bin
# [8192][24][1024] unsheared (by distance); gate.bin [8192][3072]; sink.bin [8192][24]; query-side row of position p
# is p % 8192. The KV is put in a 64-key paged cache [blocks][64][8][128] with a shuffled block table; the relative
# logits are sheared into config B's layout ((T + 128, 24, 1024 + 256), attention_sm90.py docstring).
# Timing: the two kernels captured in one CUDA graph, replayed back to back for --secs, first --discard s dropped,
# 10 Hz NVML (power, SM clock, throttle reasons, energy counter); same fields as the C++ bench.
import argparse
import json
import os
import statistics
import sys
import threading
import time

import numpy as np
import torch
import pynvml

from vllm.models.inkling2.nvidia.ops.attention import gate_sink
from vllm.models.inkling2.nvidia.ops.attention_sm90 import MODE_OUT_LSE, rel_attention

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--mode', default='time', choices=['check', 'time'])
ap.add_argument('--kind', default='global,swa')
ap.add_argument('--C', default='0,8192,32768,65536,122880,245760')
ap.add_argument('--Q', default='2048')
ap.add_argument('--secs', type=float, default=24.0)
ap.add_argument('--discard', type=float, default=2.0)
ap.add_argument('--rounds', type=int, default=1)
ap.add_argument('--arms', default='pipe', help='pipe = rel_attention + gate_sink; attn = rel_attention only')
ap.add_argument('--out', required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = torch.device('cuda')
H, HKV, D, PAGE, EXT, QMAX = 24, 8, 128, 64, 1024, 8192
pynvml.nvmlInit()
NVH = pynvml.nvmlDeviceGetHandleByIndex(0)
OUT = open(os.path.join(args.out, 'cfgb_prefill.jsonl'), 'a')


def emit(d):
    OUT.write(json.dumps(d) + '\n')
    OUT.flush()
    print(json.dumps(d), flush=True)


emit({'probe': 'meta', 'test': 'cfgb_prefill', 'uuid': pynvml.nvmlDeviceGetUUID(NVH),
      'vbios': pynvml.nvmlDeviceGetVbiosVersion(NVH),
      'power_limit_w': pynvml.nvmlDeviceGetPowerManagementLimit(NVH) / 1000, 'torch': torch.__version__, 'argv': sys.argv,
      'tag': os.environ.get('S4_TAG', ''), 'sms': torch.cuda.get_device_properties(0).multi_processor_count})


class Tele(threading.Thread):
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
        smv = sorted(x[2] for x in s)
        return {'tele_n': len(s), 'power_w_mean': statistics.mean(x[1] for x in s), 'energy_w_mean': e / dt,
                'sm_mhz_mean': statistics.mean(x[2] for x in s), 'sm_mhz_median': smv[len(smv) // 2],
                'sw_power_cap_frac': sum(1 for x in s if x[3] & 0x4) / len(s)}


TELE = Tele()
TELE.start()


def sustained(launch, calls):
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
    busy = sum(durs[i] for i in keep)
    r = {'chunks_kept': len(keep), 'calls_per_chunk': calls,
         'ms_per_call_median': statistics.median(durs[i] for i in keep) / calls,
         'ms_per_call_mean': busy / (calls * len(keep)), 'gap_frac': 1 - busy / span}
    r.update(TELE.window(t0 + args.discard, t1))
    if 'energy_w_mean' in r:
        r['j_per_call_energy'] = r['energy_w_mean'] * r['ms_per_call_mean'] * 1e-3 * (span / busy)
    return r


def load(name, shape):
    a = np.fromfile(os.path.join(args.data, name), dtype=np.uint16)
    return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).reshape(shape)


Kl = load('K.bin', (-1, HKV, D))
Vl = load('V.bin', (-1, HKV, D))
N = Kl.shape[0]
nlog = (N + PAGE - 1) // PAGE
g = torch.Generator().manual_seed(4242)
perm = torch.randperm(nlog, generator=g)
kc = torch.zeros(nlog, PAGE, HKV, D, dtype=torch.bfloat16, device=dev)
vc = torch.zeros(nlog, PAGE, HKV, D, dtype=torch.bfloat16, device=dev)
pad = nlog * PAGE - N
kpad = torch.cat([Kl, Kl.new_zeros(pad, HKV, D)]).view(nlog, PAGE, HKV, D).to(dev)
vpad = torch.cat([Vl, Vl.new_zeros(pad, HKV, D)]).view(nlog, PAGE, HKV, D).to(dev)
kc[perm.to(dev)] = kpad
vc[perm.to(dev)] = vpad
del kpad, vpad
bt = perm.to(torch.int32).to(dev)[None, :].contiguous()   # logical block -> physical block
QG = load('qg.bin', (QMAX, H, D)).to(dev)
QL = load('ql.bin', (QMAX, H, D)).to(dev)
REL = load('rel.bin', (QMAX, H, EXT)).to(dev)
GATE = load('gate.bin', (QMAX, H * D)).to(dev)
SINK = load('sink.bin', (QMAX, H)).to(dev)
emit({'probe': 'cfgb_prefill', 'test': 'data', 'npos': N, 'blocks': nlog})


def shear(rel, C, Q):
    """unsheared rel [Q, 24, 1024] (by distance) -> config B's (Q + 128, 24, 1280): column c of token t holds key
    c + cdiv(past + tile_end, 128) * 128 - (E + 256), tile_end = end of t's 128-row tile (attention_sm90.py)."""
    out = torch.zeros(Q + 128, H, EXT + 256, dtype=torch.bfloat16, device=dev)
    t = torch.arange(Q, device=dev)
    tile_end = torch.clamp((t // 128 + 1) * 128, max=Q)
    rel_base = (C + tile_end + 127) // 128 * 128 - (EXT + 256)
    cols = torch.arange(EXT + 256, device=dev)
    for s in range(0, Q, 512):
        e = min(Q, s + 512)
        key = cols[None, :] + rel_base[s:e, None]
        d = (C + t[s:e, None]) - key
        ok = (d >= 0) & (d < EXT)
        val = rel[s:e].gather(2, d.clamp(0, EXT - 1)[:, None, :].expand(e - s, H, EXT + 256))
        out[s:e] = torch.where(ok[:, None, :], val, torch.zeros_like(val))
    return out


def cell(kind, C, Q):
    swa = kind == 'swa'
    rows = (C + torch.arange(Q, device=dev)) % QMAX
    q = (QL if swa else QG)[rows].contiguous()
    rel = shear(REL[rows], C, Q)
    gate = GATE[rows].contiguous()
    sink = SINK[rows].contiguous()
    out = torch.empty(Q, H, D, dtype=torch.float32, device=dev)
    lse = torch.empty(H, Q, dtype=torch.float32, device=dev)
    res = torch.empty(Q, H * D, dtype=torch.bfloat16, device=dev)
    seql = torch.tensor([C + Q], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, Q], dtype=torch.int32, device=dev)
    keys = sum(min(C + i + 1, 1024) if swa else C + i + 1 for i in range(Q))
    flops = 4.0 * H * D * keys

    def attn():
        rel_attention(q, kc, vc, out, lse, rel, block_table=bt, cache_seqlens=seql, cu_seqlens_q=cu, max_query_len=Q,
                      softmax_scale=D ** -0.5, window=1024 if swa else 0, num_splits=1, mode=MODE_OUT_LSE)

    def pipe():
        attn()
        gate_sink(out, gate, lse.t(), sink, res)
    return attn, pipe, out, lse, res, flops


for kind in args.kind.split(','):
    for C in [int(x) for x in args.C.split(',')]:
        for Q in [int(x) for x in args.Q.split(',')]:
            if C + Q > N:
                continue
            attn, pipe, out, lse, res, flops = cell(kind, C, Q)
            pipe()
            torch.cuda.synchronize()
            if args.mode == 'check':
                nm = f'{kind}_C{C}_Q{Q}'
                res.view(torch.int16).cpu().numpy().tofile(os.path.join(args.out, f'cfgb_out_{nm}.bin'))
                out.cpu().numpy().tofile(os.path.join(args.out, f'cfgb_o32_{nm}.bin'))
                lse.cpu().numpy().tofile(os.path.join(args.out, f'cfgb_lse_{nm}.bin'))
                emit({'probe': 'cfgb_prefill', 'test': 'check_dump', 'kind': kind, 'C': C, 'Q': Q, 'ok': True})
                continue
            graphs = {}
            for arm in args.arms.split(','):
                fn = pipe if arm == 'pipe' else attn
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    fn()
                torch.cuda.current_stream().wait_stream(s)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn()
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record()
                for _ in range(3):
                    gr.replay()
                en.record()
                torch.cuda.synchronize()
                calls = max(1, int(40.0 / (st.elapsed_time(en) / 3)))
                graphs[arm] = (gr, calls)
            for rnd in range(args.rounds):
                for arm, (gr, calls) in graphs.items():
                    r = sustained(gr.replay, calls)
                    ms = r['ms_per_call_median']
                    r.update({'probe': 'cfgb_prefill', 'test': 'time', 'kind': kind, 'C': C, 'Q': Q, 'arm': arm,
                              'round': rnd, 'flops': flops, 'ms_per_chunk': ms, 'tflops': flops / (ms * 1e-3) / 1e12})
                    if 'j_per_call_energy' in r:
                        r['j_per_new_token'] = r['j_per_call_energy'] / Q
                    emit(r)
            del graphs
            torch.cuda.synchronize()
TELE.stop = True
