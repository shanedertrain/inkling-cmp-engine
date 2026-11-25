# Config B numerics, op by op, and the S1 reference against them

Written 2026-11-22, desk only (no card free). Source read: config B's served overlay (Hiutaky vLLM Ampere port
`6e9d1cce`) and the Humming package from its image. Labels: [S] read in source, [M] measured, [I] inferred.

## Source and provenance

| Prefix | Path (desktop, read-only copies) | Check |
|---|---|---|
| `S/` | `Hiutaky/vllm` @ `6e9d1cce` (`models/inkling2/nvidia/`) | sha256 of all 103 `*.py` under `models/inkling2` and `v1/worker/gpu/spec_decode` verified against the served vLLM Config B overlay (2026-11-22) [M] |
| `V/` | `.../src-ref/vllm-pkg/` (rest of the overlay) | wave0 source-audit: byte-identical to the clone |
| `H/` | `.../src-ref/humming-and-config/humming/` (Humming from the image; kernels are JIT-compiled from these headers) | `docs/stage-p/humming-from-cpp.md` |
| checkpoint | `thinkingmachines/Inkling-2-Mini` (`model-00001-of-00001.safetensors`) header | every small parameter is BF16 (norms, convs, scales, rel_proj, controllers, router weight), queries and router bias FP32 [M] |

Path config B takes on the CMP 170HX (sm_80, TP 1, loop model): `Inkling2LoopModel` with `conv_mix` on the Triton
fallback (`S/ops/conv_mix.py:92-122`: `short_conv` then `attn_res`); attention prefill = `rel_attention`
`MODE_OUT_LSE` + `gate_sink`, decode/verify (<= 21 query rows) = split-K `rel_attention` + combine
(`S/attention.py:442-534`, `S/ops/attention.py:1195-1301`); MoE = `select_experts` + Humming + `swiglu` +
`combine` (`S/moe.py:385-395`, `S/ops/moe_sm90.py`); dense = `torch.nn.functional.linear` (`S/ops/linear.py:33-38`);
head = `torch.mm(..., out_dtype=float32)` (`V/model_executor/layers/logits_processor.py:158-171`, `head_dtype`
defaulted to float32 by `V/models/inkling2/configs.py:219-220`).

## Per-op table

In = operand dtype, Acc = accumulation, Round = where a value is rounded. "Ref" = what `engine/ref` does now.

| Op | In | Acc | Round | Order / detail that affects bits | Config B source | Ref |
|---|---|---|---|---|---|---|
| Embedding + embed_norm | bf16 row, bf16 weight | fp32 | bf16 out | `rsqrt.approx(sum x^2 / N + eps)`, x * rstd * w | `S/model.py:268-270,310`; `S/ops/norm.py:14-44` | same (torch rsqrt) |
| Residual stream | — | — | **bf16 storage** | blocks and partial block live in a bf16 stack | `S/loop.py:327`, `S/model.py:8,355` | **fixed**: `stream_bf16` (was fp32) |
| AttnRes mix (attn and MLP inputs, messages, final) | sources fp32 of bf16; partial + addend summed in fp32 unrounded | fp32 | bf16 after the fused RMSNorm; fp32 when no norm (loop messages, pass > 0 inputs) | queries pre-scaled by fp32(H^-0.5) at load; rstd = rsqrt(fma(sum x^2, 1/H, eps)); logit = (q.x) * rstd; p = exp(l - max); out = sum_n fma(p_n * rstd_n, x_n) * (1/sum p); `enable_fp_fusion=False` | `S/ops/attn_res.py:21-169,246-307`; `S/model.py:159-162`; `S/loop.py:176` | **fixed**: `attn_res` (was softmax(q.norm(x)) * scale order) |
| Loop message controller | dest, message fp32 | fp32 / bf16 GEMMs | pair RMS -> bf16; down bmm -> bf16; gain/forget bmm -> fp32; blend + sublayer RMSNorm -> bf16 | (1 - tanh f) * dest + tanh g * message, unfused, libdevice tanh | `S/loop.py:112-130`; `S/ops/loop.py:19-132` | same |
| Pass restart | — | — | — | each pass restarts from the prelude's mixes (computed once, fp32), bank = the 2 entries after the prelude's | `S/loop.py:349-371` | same |
| q/k/v/r/gate/sink projections | bf16 | fp32 (cuBLAS) | bf16 | full layers: one GEMM [q, r, gate, sink] of x_q, k and v separate; local: one GEMM of all six (+pad to 8) | `S/attention.py:217-239,333-351` | separate GEMMs (cuBLAS picks per N: bits, not rounding points) |
| cuBLAS split-K reduction | — | — | **bf16 allowed** | torch default `allow_bf16_reduced_precision_reduction=True`, vLLM changes it only under `VLLM_BATCH_INVARIANT` | `V/model_executor/determinism/batch_invariant.py:1473` | **fixed**: `gemm_bf16_reduction=True` (was False) |
| q/r clip, tau | bf16 proj x bf16 scale | fp32 | bf16 after the clip; x tau then bf16 again (full layers) | norm = s * rsqrt(s); x * min(thr / (norm + 1e-6), 1); tau = 1 + 0.1 log(max((p+1)/8192, 1)) | `S/ops/norm.py:73-82,133-148`; `S/ops/attention.py:342-367` | same (now s * rsqrt(s)) |
| Relative logits | r bf16, rel_proj bf16 | fp32 dot | **bf16** (sheared buffer) | | `S/ops/norm.py:173-185`; `S/attention.py:450-452` | same |
| Dynamic K/V taps | bf16 | fp32 | bf16 | K = 2304 < 2560: plain `F.linear` | `S/sconv.py:96-107` | same |
| K/V short conv (full: dynamic, local: static) | bf16 inputs and cache, bf16 weights | fp32 | **bf16 conv out**, then K clipped / V RMS-normed in fp32, stored bf16 | static: fma oldest tap first, (acc + x) * fp32(5^-0.5); dynamic: w = static + dyn (fp32), lag 0 product then fma, (acc + x) * fp32(1/3) | `S/ops/sconv.py:115-203,238-318,787-840` | **fixed**: multiply by fp32 constants (was division), fma in the dynamic conv |
| KV cache | bf16 (lossless) / unit-scale E4M3, RNE, saturate 448, software codec on sm_80 (lossy) | | | decoded E4M3 -> f16 -> bf16 exactly | `S/ops/sconv.py:37-48,833-840`; `S/ops/attention_sm90.py:42-74` | same codec; ref's lossy mode puts FP8 on the 8 full visits only, config B's `fp8` KV on every layer |
| Attention scores | q, k bf16 | fp32 `tl.dot` | — | log2 domain: (q.k) * fp32(scale * log2e) + bias * log2e; window dist < 1024 (local), bias for dist < 1024 | `S/ops/attention_sm90.py:165-196` | **fixed**: log2 domain |
| Attention softmax / P.V | p fp32 | fp32 | **p -> bf16 for P.V**, row sum fp32 | key blocks aligned to absolute positions; **BLOCK_N 64 for prefill (>= 22 rows) and 64-row tiles, 128 for decode/verify (<= 10 rows)**; online max per split | `S/ops/attention_sm90.py:197-211,252-254,358` | **fixed**: per-shape block (was 64 always), absolute alignment |
| Attention split-K | | fp32 | o_s = acc / l, lse2 | **decode/verify: 9 splits (full, n <= 4; 6 at n = 8), local 9 (n = 1) / 3 (n = 4, verify) / ...; split s takes blocks n_lo + s + i*S**; combine w = 2^(lse_s - max), o = sum o_s w_s / sum w | `S/ops/attention.py:377-405,444-494`; `S/ops/attention_sm90.py:145-152` | **fixed**: `attn_plan` + split emulation (was one split) |
| Gate and sink | o fp32, gate/sink bf16 | fp32 | **one bf16 rounding** | o * sigmoid(gate), then * sigmoid(lse - sink); sigmoid = rcp(1 + ex2(-x log2e)) | `S/ops/attention.py:254-301,490-494`; `S/ops/attention_sm90.py:230-245` | **fixed**: that order (was sink first) |
| o projection (wo) | bf16 | fp32 | bf16 | | `S/attention.py:407` | same |
| Hidden short convs (attn, MLP) + residual update | bf16 x, bf16 cache, bf16 weights | fp32 | **bf16 conv out**; partial = **bf16((residual + attn_out) + y_mlp)** in fp32 | (acc + x) * fp32(5^-0.5) | `S/ops/sconv.py:321-405,469` | **fixed** (was fp32 out, fp32 partial, division) |
| Router | x bf16, W bf16 | fp32 per 1,024-wide K group, groups summed in order | — | key = bits(1/(1+exp(-l)) + bias), reversed index in the low bits (ties -> lower expert); top-8 by `tl.topk` (<= 128 tokens) or 8 max passes | `S/ops/moe.py:33-147`; `S/ops/linear.py:44-100,204-277` | **fixed**: K-group sum |
| Routing weights | fp32 | fp32 | — | lp = min(l, 0) - log1p(exp(-\|l\|)); w = exp(lp - (max + log(sum8))) * 8, sum8 = ((x0+x2)+(x1+x3))+((x4+x6)+(x5+x7)) | `S/ops/moe.py:22-30,53-66` | **fixed**: sum8 order |
| Expert gate/up, down (Humming) | x bf16; NVFP4 e2m1 x e4m3 per-16 scale formed **in bf16** (exact) on B; MXFP8 e4m3 x e8m0 per-32 (exact) | fp32 mma.sync (`use_f16_accum=False`), stream-K off | **bf16** out | NVFP4 per-expert factor amax/denominator multiplies the **fp32 accumulator** before the bf16 conversion | `H/include/humming/arith/mainloop_arith.cuh:156-240,358`; `H/.../epilogue_arith.cuh:139-153`; `H/.../epilogue/smem_writer.cuh:157-163`; `S/ops/moe_sm90.py:228-231` | same (`expert_math exact`) |
| SiTU x routing weight | gate, up bf16 | fp32 | **bf16** | ((softcap4(g) * sigmoid(g)) * softcap25(u)) * gamma; sigmoid = rcp.approx.ftz(1 + ex2.approx.ftz(g * -log2e)); softcap = cap * **tanh.approx**(x / cap) (exact division) | `S/ops/activation.py:21-80`; `S/ops/moe_sm90.py:111-157` | exact torch functions by default; **`situ_lut`** reproduces config B bit for bit once the tables exist |
| Combine | 8 bf16 rows | fp32, 0 + slot 0 + slot 1 ... | **bf16** | slot order = top-k order | `S/ops/moe.py:1060-1107`; `S/moe.py:311` | **fixed**: sequential |
| Final mix + norm | | fp32 | bf16 | AttnRes with output_query and the final norm | `S/loop.py:375,395` | same |
| Unembedding + softcap | bf16 x bf16 (201,024 padded rows) | fp32 cuBLAS | **fp32 out**, sliced to 200,058, tanh(x / 30) * 30 in fp32 | | `V/.../logits_processor.py:113-116,158-171,196-197` | **fixed**: one GEMM over the padded vocabulary on CUDA (was 32K-row chunks of the unpadded rows) |
| DSpark taps | — | — | bf16 (stack entries) | partial block entering visits 5, 23, 41 + final output | `S/loop.py:372-373,396-397` | same |
| DSpark block attention | bf16 | fp32 | p -> bf16; out bf16 | DRAFT mode of the same kernel: all rows at the block's last position, local window 1,024 + 8, splits 9/10 at n = 1 | `S/dspark.py:173,194-247`; `S/ops/attention.py:1158-1192` | **fixed**: `draft_attn='cfgb'` (was fp32 SDPA) |
| DSpark norms, MLP, Markov head | | | | flashinfer rmsnorm / fused_add_rmsnorm, silu_and_mul, addmm fp32 out | `S/dspark.py:53-68,284-385` | as written in S1 (not re-read here) |
| Verify / accept | fp32 logits | | | greedy: longest matching prefix, argmax first index, NaN as -inf | `V/v1/worker/gpu/sample/greedy_argmax.py` | same (`spec.py`) |

Hardware approximations config B uses that the reference cannot reproduce from source: `rsqrt.approx.f32` in every
norm (2^-22.9 relative), `ex2.approx` / `rcp.approx` in softmax and sigmoids (~2^-22), Triton's `div.full.f32`,
`tanh.approx.f32` in the SiTU (**2^-11 relative**: a quarter of a bf16 ulp; the only large one), and the summation
order inside Triton reductions, `tl.dot`, cuBLAS and Humming tiles. Only the SiTU is a function of a bf16 input, so
only it can be tabulated (65,536 entries per function): `probes/s1-numerics/kernels_vs_ref.py`.

## Open points of STATUS.md: closed / open

| # | Point | State | Evidence / what settles it |
|---|---|---|---|
| 1 | NVFP4 per-expert factor position | **closed** [S]: on the fp32 accumulator before the single bf16 rounding; block scale applied to the weights in bf16 (exact) | epilogue and mainloop lines above; ref's `exact` mode is that |
| 2 | Head output precision | **closed** [S]: bf16 x bf16, fp32 out (`torch.mm(out_dtype=float32)`), padded N | logits_processor; ref now one GEMM, padded N |
| 3 | Attention numerics | **closed in structure** [S]: log2 domain, bf16 P, BLOCK_N 64/128 by shape, absolute block alignment, interleaved split-K and its combine, gate then sink. Residual: fp32 order and ex2/rcp approximations | `kernels_vs_ref.py` `attn_dec`/`attn_pre` measures the residual |
| 4 | Router summation | **closed in structure**: in-order 1,024-wide K groups (inside a group the mma order is not reproduced) | `route` check |
| 5 | Routing weights, tie-break | **closed** [S] (sum8 order added) | `route` check |
| 6 | SiTU | **corrected**: STATUS said ~1e-7; tanh.approx is 2^-11. **Open until the tables are built** (one probe run), then exact | `situ_lut`, `situ_exact` checks |
| 7 | Fused small ops | **closed** [S]: AttnRes, short convs, residual update, RMSNorm, controllers read line by line; the big finding is the **bf16 residual stream** | `conv`, `attn_res` checks |
| 8 | Dense GEMMs | **partly closed**: same `F.linear`, same torch flags now (bf16 split-K reductions allowed); fused vs separate projection GEMMs differ in N, so cuBLAS may pick other kernels (order only) | `gemm` check says whether the flag changes bits at these shapes |
| 9 | Prefill chunking | open (unchanged): 2,048 per source-audit A7 | `--chunk 512` A/B |
| 10 | Drafter attention / norms | attention **closed** [S] (split-K bf16 P, now emulated); norms/Markov head not re-read | acceptance vs config B's `/metrics` |
| 11 | Lossy FP8 KV | **closed** [S]: config B's own `fp8` KV is unit-scale E4M3, RNE, saturating, exact decode = the ref's codec; difference: config B would apply it to every layer | — |
| 12 | FP8 drafter / draft ring | unchanged (not implemented, config B is bf16) | — |
| new | Decode vs prefill kernel by shape | **closed** [S]: <= 21 query rows (64 // group 3) use the split-K kernel, also a short final prefill chunk; mixed batches send <= 8-row requests to it | `attn_plan`, tests |
| new | SM count in the split plan | 74 (torch reports 74 on these cards) [M, cmp170hx-facts.md]; config B calls `num_compute_units()` | if config B's process sees 70, full-layer splits become 8: check `torch.cuda.get_device_properties(0).multi_processor_count` in the image |

## Ranked suspects if the S1 gate misses (after the fixes above)

Each row = a switch of `run_goldens.py` on the teacher-forced set (`--skip greedy`), compared with the default run.

| Rank | Suspect | Expected size [I] | Switch / action |
|---|---|---|---|
| 1 | SiTU `tanh.approx` (2^-11 on every expert activation, 48 visits x 8 experts) | largest remaining systematic difference | build tables (probe), rerun with `--situ-lut <out>/situ_lut.pt` |
| 2 | cuBLAS bf16 split-K reductions (flag now as config B; shapes still differ: separate vs fused projections, M) | 0 if cuBLAS never splits K at these shapes; else a few extra bf16 roundings per projection | probe `gemm` rows; `--gemm-fp32-reduction` |
| 3 | Residual stream (sanity of the main fix) | moving to fp32 should **raise** the KL clearly | `--stream-fp32` |
| 4 | Attention split/block emulation | fp32-order level if the probe agrees | `--attn-nosplit`, `--attn-exact`; probe `attn_*` |
| 5 | NVFP4 math (closed in source; job W: Humming 0.22 % from exact math) | small | `--expert-math hf` |
| 6 | Prefill chunk size | position 0 only | `--chunk 512` |
| 7 | Head | closed in source | `--head-bf16` (should be worse) |

If a miss survives all switches: compare position 0 vs 1-32 and short vs long cases first (a long-context-only miss
points at attention or the rings; a uniform miss at the stream, SiTU or GEMMs), then bisect by visit with config B
hidden states (the P5 overlay can dump them; a desk change).

## GPU probe (when a card frees; ~2 min)

`probes/s1-numerics/kernels_vs_ref.py` (header has the docker command; config B's image and overlay, `--network none`,
no `$HOME`): builds `situ_lut.pt` with config B's own `_sigmoid_approx` / `_softcap`, then runs config B's kernels on
synthetic inputs at the real shapes against the reference's emulation (SiTU, static short conv + residual update,
AttnRes, router, combine, decode and prefill attention for full and local layers, and the dense-GEMM reduction flag).
Expected results are in its header. Never run.

## Second-opinion reference (transformers PR 51050, PLAN S0)

ingine's `tools/inkling_ref.py` is that PR's model with packed experts, pinned to transformers commit `6173a2f`
(`prep/vendor/ingine/tools/requirements.txt`); no copy of that commit is on disk (needs one network fetch, or the
owner's approval to copy it from the ingine machine), and it would run in docker like any third-party code. It loads
dense weights as bf16; "FP32 activations" means `dtype=float32` (dense weights ~3 GB in bf16, ~6 GB in fp32: fits). Its attention is
eager/SDPA over the whole sequence: the 45K-125K cases need chunking added or are skipped. Effort ~0.5 day desk + ~1-2
card-hours. What it buys: a higher-precision anchor that separates "the reference misses config B" from "config B is
itself far from the math" (KL(fp32 ref, config B) vs KL(fp32 ref, S1 ref)); it does not help pass the S1 gate, which
is against config B. The S1 reference's switches (`--stream-fp32 --attn-exact --expert-math hf --head-bf16`) already
give an ingine/transformers-ordered variant on the same code. **Not worth building now**; build it only if the gate
misses and the switch A/Bs above do not attribute the miss.
