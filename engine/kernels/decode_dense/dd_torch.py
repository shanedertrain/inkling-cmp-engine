# decode_dense: config B's dense projections as config B runs them (own code). Runs inside config B's vLLM image
# (its torch + cuBLAS), python -I, GPU 7 only, --network none. Same weights, visit plan, rows and timing method as
# dd_bench.cu (CUDA graph of one op over the 48 visits, replayed back to back; NVML energy counter):
#   proj : models/inkling2/nvidia/attention.py _project on sm_80: torch.nn.functional.linear over the merged qkvr weight
#          [q | r | gate | sink | k | v] (8,600 x 2,304) when q / k / v share one input, else three linears (q side
#          padded to 6,560 rows, k, v); full layers add the two dynamic-tap linears (sconv.dynamic_taps, [32 x 2,304])
#   wo   : F.linear(attn [M, 3072], wo_ud [2304, 3072])
#   head : F.linear(h [M, 2304], unembed [201024, 2304]) (bf16 out, config B's UnquantizedEmbeddingMethod.apply)
# Also: one eager pass per (op, rows) under torch.profiler -> kernel names and device time per kernel.
#   python3 -I dd_torch.py --model /model --rows 8,32,64 --ops proj,wo,head --secs 6 --rounds 3 --out /out
import argparse
import ctypes
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

H, QD, KD = 2304, 3072, 1024
FULL = (5, 11, 17, 23)


class Nvml:
    def __init__(self):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        assert self.lib.nvmlInit_v2() == 0
        self.dev = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.dev)) == 0
        buf = ctypes.create_string_buffer(96)
        self.lib.nvmlDeviceGetUUID(self.dev, buf, 96)
        self.uuid = buf.value.decode()
        self.lib.nvmlDeviceGetVbiosVersion(self.dev, buf, 96)
        self.vbios = buf.value.decode()
        lim = ctypes.c_uint()
        self.lib.nvmlDeviceGetPowerManagementLimit(self.dev, ctypes.byref(lim))
        self.limit_w = lim.value / 1000

    def energy_mj(self):
        e = ctypes.c_ulonglong()
        assert self.lib.nvmlDeviceGetTotalEnergyConsumption(self.dev, ctypes.byref(e)) == 0
        return e.value

    def sm_mhz(self):
        c = ctypes.c_uint()
        self.lib.nvmlDeviceGetClockInfo(self.dev, 1, ctypes.byref(c))
        return c.value


def visit_plan():
    v = [(l, False) for l in range(6)]
    for p in range(3):
        v += [(l, p > 0) for l in range(6, 18)]
    v += [(l, False) for l in range(18, 24)]
    return v


def load(model):
    f = safe_open(os.path.join(model, 'model-00001-of-00001.safetensors'), 'pt', device='cpu')
    L = []
    for l in range(24):
        p = f'model.llm.layers.{l}.attn.'
        g = lambda n: f.get_tensor(p + n).to(torch.bfloat16)
        qkvr = torch.cat([g('wq_du.weight'), g('wr_du.weight'), g('gate_attn.weight'), g('attn_sink_proj.weight'),
                          g('wk_dv.weight'), g('wv_dv.weight')]).cuda().contiguous()
        d = {'qkvr': qkvr, 'wo': g('wo_ud.weight').cuda().contiguous()}
        if l in FULL:
            d['kd'] = g('k_sconv.weight_proj').cuda().contiguous()
            d['vd'] = g('v_sconv.weight_proj').cuda().contiguous()
        L.append(d)
    unembed = f.get_tensor('model.llm.unembed.weight').to(torch.bfloat16).cuda().contiguous()
    return L, unembed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/model')
    ap.add_argument('--rows', default='8,32,64')
    ap.add_argument('--ops', default='proj,wo,head')
    ap.add_argument('--secs', type=float, default=6)
    ap.add_argument('--discard', type=float, default=1)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', default='/out')
    a = ap.parse_args()
    nv = Nvml()
    torch.backends.cuda.matmul.allow_tf32 = False
    L, U = load(a.model)
    plan = visit_plan()
    gen = torch.Generator(device='cuda').manual_seed(1)
    xq, xk, xv = (torch.randn(64, H, device='cuda', generator=gen).to(torch.bfloat16) for _ in range(3))
    xo = (torch.randn(64, QD, device='cuda', generator=gen) * 0.5).to(torch.bfloat16)
    xm = torch.randn(64, H, device='cuda', generator=gen).to(torch.bfloat16)
    q_side = 3072 + 384 + 3072 + 24
    q_pad = q_side + (-q_side % 8)
    out = open(os.path.join(a.out, 'dd_torch.jsonl'), 'a')

    def emit(d):
        d.update(probe='dd_torch', torch=torch.__version__, cuda=torch.version.cuda, uuid=nv.uuid, vbios=nv.vbios,
                 power_limit_w=nv.limit_w)
        out.write(json.dumps(d) + '\n')
        out.flush()

    def op_fn(op, M):
        x_q, x_k, x_v, x_o, x_m = xq[:M], xk[:M], xv[:M], xo[:M], xm[:M]

        def proj():
            for l, msgs in plan:
                w = L[l]['qkvr']
                if l not in FULL and not msgs:
                    F.linear(x_q, w)
                else:
                    F.linear(x_q, w[:q_pad])
                    F.linear(x_k, w[q_side:q_side + KD])
                    F.linear(x_v, w[q_side + KD:q_side + 2 * KD])
                if l in FULL:
                    F.linear(x_k, L[l]['kd'])
                    F.linear(x_v, L[l]['vd'])

        def wo():
            for l, _ in plan:
                F.linear(x_o, L[l]['wo'])

        def head():
            F.linear(x_m, U)
        return {'proj': proj, 'wo': wo, 'head': head}[op]

    def nbytes(op):
        if op == 'head':
            return U.numel() * 2
        if op == 'wo':
            return sum(L[l]['wo'].numel() * 2 for l, _ in plan)
        tot = 0
        for l, msgs in plan:
            tot += 8600 * H * 2 + (q_pad - q_side) * H * 2 * (l in FULL or msgs)
            tot += 2 * 32 * H * 2 * (l in FULL)
        return tot

    for op in a.ops.split(','):
        for M in [int(r) for r in a.rows.split(',')]:
            fn = op_fn(op, M)
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            # kernel names (eager pass)
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            ks = {}
            for e in prof.events():
                if e.device_type.name == 'CUDA':
                    k = ks.setdefault(e.name, [0, 0.0])
                    k[0] += 1
                    k[1] += e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total
            emit({'test': 'kernels', 'op': op, 'rows': M,
                  'kernels': [{'name': n, 'count': c, 'us_total': t} for n, (c, t) in sorted(ks.items(), key=lambda z: -z[1][1])]})
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            for rnd in range(a.rounds):
                chunks, evs = [], []
                t0 = time.monotonic()
                e_start = None
                while time.monotonic() - t0 < a.secs:
                    if e_start is None and time.monotonic() - t0 >= a.discard:
                        torch.cuda.synchronize()
                        e_start, t_e0, n_e0 = nv.energy_mj(), time.monotonic(), len(evs)
                    b, c = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    b.record()
                    for _ in range(4):
                        g.replay()
                    c.record()
                    evs.append((b, c))
                    if len(evs) > 3:
                        evs[-4][1].synchronize()
                torch.cuda.synchronize()
                e_end, t_e1 = nv.energy_mj(), time.monotonic()
                kept = evs[n_e0:]
                per = sorted(b.elapsed_time(c) / 4 for b, c in kept)
                calls = 4 * len(kept)
                j = (e_end - e_start) / 1000
                emit({'test': 'sweep', 'op': op, 'variant': 'torch_linear', 'rows': M, 'round': rnd,
                      'ms_per_call_median': per[len(per) // 2], 'ms_per_call_mean': statistics.mean(per),
                      'calls': calls, 'energy_j': j, 'j_per_call_energy': j / calls, 'window_s': t_e1 - t_e0,
                      'energy_w_mean': j / (t_e1 - t_e0), 'sm_mhz_end': nv.sm_mhz(), 'bytes_per_call': nbytes(op)})
            del g
    out.close()


if __name__ == '__main__':
    main()
