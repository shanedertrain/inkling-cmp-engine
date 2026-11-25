# inkling-cmp-engine S4 prefill MoE: capture REAL prefill routing and MoE inputs with the S1 reference (engine/ref).
# Runs the reference's chunked prefill of real token ids (job Q's pool = the frozen panel's texts tokenised by the
# server) and records, for every one of the 48 visits, the router's top-8 expert ids and routing weights per token, and
# for selected visits the bf16 MoE input x (the normalised hidden state the experts see). Output: one safetensors file
# per prompt: v{VV}.idx int32 [N, 8], v{VV}.wts fp32 [N, 8], v{VV}.x bf16 [N, 2304] (selected visits).
# A token's routing does not depend on prefill chunking (causal), so chunks of 1,600 / 2,048 / 6,000 are slices.
#
# Server, config B image (torch + safetensors), own GPU only, --network none (see run_capture.sh).
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from engine.ref import weights as W                       # noqa: E402
from engine.ref.engine import RefEngine, Request          # noqa: E402
from engine.ref import model as M                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model-dir', default='/model')
ap.add_argument('--pool', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--prompts', default='0:8192,200000:6144,100000:6144,300000:6144', help='offset:length,... into the pool')
ap.add_argument('--x-visits', default='0,7,20,44,47')
ap.add_argument('--chunk', type=int, default=2048)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
pool = json.load(open(a.pool))
xv = {int(v) for v in a.x_visits.split(',')}

rec = {}
state = {'v': -1}
orig_visit, orig_route, orig_moe = M.Model._visit, M.Model._route, M.Model._moe


def visit(self, v, *args, **kw):
    state['v'] = v
    return orig_visit(self, v, *args, **kw)


def route(self, layer, xb):
    idx, wts = orig_route(self, layer, xb)
    r = rec.setdefault(state['v'], {'idx': [], 'wts': [], 'x': []})
    r['idx'].append(idx.to(torch.int32).cpu())
    r['wts'].append(wts.float().cpu())
    if state['v'] in xv:
        r['x'].append(xb.to(torch.bfloat16).cpu())
    return idx, wts


M.Model._visit, M.Model._route = visit, route

t0 = time.time()
w = W.load(a.model_dir, 'cuda')
print(f'loaded {time.time() - t0:.0f} s', flush=True)
from safetensors.torch import save_file   # noqa: E402
meta = {'pool': a.pool, 'chunk': str(a.chunk), 'prompts': a.prompts, 'revision': json.dumps(w.revision, default=str)[:2000]}
for pi, spec in enumerate(a.prompts.split(',')):
    off, n = (int(s) for s in spec.split(':'))
    ids = pool[off:off + n]
    rec.clear()
    e = RefEngine(w, 'lossless', 'cuda', num_pages=max(4, n // 1024 + 2), page_size=1024, chunk=a.chunk,
                  use_spec=False)
    t1 = time.time()
    r = e.generate([Request('s', ids, 1, record_top=1)])['s']
    print(f'prompt {pi}: {n} tokens prefilled in {time.time() - t1:.0f} s, next token {r.tokens}', flush=True)
    out = {}
    for v, d in sorted(rec.items()):
        idx = torch.cat(d['idx'])[:n].contiguous()
        out[f'v{v:02d}.idx'] = idx
        out[f'v{v:02d}.wts'] = torch.cat(d['wts'])[:n].contiguous()
        if d['x']:
            out[f'v{v:02d}.x'] = torch.cat(d['x'])[:n].contiguous()
    assert len(rec) == 48, len(rec)
    path = os.path.join(a.out, f'p{pi}.safetensors')
    save_file(out, path, metadata={**meta, 'offset': str(off), 'n': str(n)})
    print(f'wrote {path}', flush=True)
    del e
    torch.cuda.empty_cache()
print(f'done {time.time() - t0:.0f} s', flush=True)
