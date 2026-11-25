#!/usr/bin/env python3
"""Generate DSpark folded head weights (fold.bin + lowrank16.bin) from Inkling-2-Mini.

Pure PyTorch + safetensors, zero vLLM dependency. Produces the exact fp64-folded
Markov head (`fold.bin`) and rank-16 pruning basis (`lowrank16.bin`) expected by
`ink-server --draft-fold <OUT_DIR>/fold.bin`.

Usage:
    python3 scripts/fold-dspark.py --model-dir /path/to/Inkling-2-Mini --out-dir build/dspark-fold
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors import safe_open

RANK = 256
LOWRANK_K = 16


def main() -> None:
    ap = argparse.ArgumentParser(description="Fold DSpark MarkovHead weights for ink-server.")
    ap.add_argument("--model-dir", required=True, help="Directory with model-00001-of-00001.safetensors and mtp.safetensors")
    ap.add_argument("--out-dir", required=True, help="Output directory for fold.bin and lowrank16.bin")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device (default: cuda if available)")
    args = ap.parse_args()

    model_path = os.path.join(args.model_dir, "model-00001-of-00001.safetensors")
    mtp_path = os.path.join(args.model_dir, "mtp.safetensors")
    for p in (model_path, mtp_path):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: missing checkpoint file: {p}")

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)
    print(f"Loading unembed & MarkovHead tensors on {dev}...", flush=True)

    with safe_open(model_path, framework="pt", device=str(dev)) as f:
        lm_head = f.get_tensor("model.llm.unembed.weight")  # [VP, H] bf16
    with safe_open(mtp_path, framework="pt", device=str(dev)) as f:
        hw = f.get_tensor("markov_head.hidden_out.weight").double()  # [H, 256]
        hb = f.get_tensor("markov_head.hidden_out.bias").double()    # [H]

    vp = lm_head.shape[0]
    assert hw.shape[1] == RANK, f"expected Markov rank {RANK}, got {hw.shape}"

    print(f"Folding MarkovHead in fp64 ({vp} vocab rows)...", flush=True)
    fold_w = torch.empty((vp, RANK), dtype=torch.bfloat16, device=dev)
    fold_b = torch.empty((vp,), dtype=torch.bfloat16, device=dev)
    for s in range(0, vp, 4096):
        hd = lm_head[s : s + 4096].double()
        fold_w[s : s + 4096].copy_(hd @ hw)
        fold_b[s : s + 4096].copy_(hd @ hb)
    del lm_head, hw, hb

    fold_path = os.path.join(args.out_dir, "fold.bin")
    with open(fold_path, "wb") as fo:
        fo.write(fold_w.view(torch.int16).cpu().numpy().tobytes())
        fo.write(fold_b.view(torch.int16).cpu().numpy().tobytes())
    print(f"Wrote {fold_path} ({os.path.getsize(fold_path):,} bytes)", flush=True)

    print(f"Computing rank-{LOWRANK_K} orthonormal pruning basis in fp64...", flush=True)
    _, _, vh = torch.linalg.svd(fold_w.float(), full_matrices=False)
    q, _ = torch.linalg.qr(vh[:LOWRANK_K].double().T)
    vk = q.T.contiguous()  # [16, 256] fp64
    uk = (fold_w.double() @ vk.T).to(torch.bfloat16)  # [VP, 16] bf16
    res = fold_w.double() - uk.double() @ vk
    rp = (res @ vk.T).norm(dim=1)
    rq = (res - (res @ vk.T) @ vk).norm(dim=1)

    inf = torch.tensor(float("inf"), device=dev)
    up = lambda t: torch.nextafter(t.float(), inf)

    lr_path = os.path.join(args.out_dir, f"lowrank{LOWRANK_K}.bin")
    with open(lr_path, "wb") as fo:
        fo.write(vk.float().cpu().numpy().tobytes())
        fo.write(uk.view(torch.int16).cpu().numpy().tobytes())
        fo.write(up(rp * (1.0 + 1e-9)).cpu().numpy().tobytes())
        fo.write(up(rq * (1.0 + 1e-9)).cpu().numpy().tobytes())
    print(
        f"Wrote {lr_path} ({os.path.getsize(lr_path):,} bytes; "
        f"resQ_mean={float(rq.mean()):.4f}, resP_max={float(rp.max()):.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
