# Hill-climb ledger (inference-optimisation skill; head session keeps it current)

Target (owner): maximum completed agent work per CMP 170HX card, M1 goodput at 4 sessions, TTFT p99 <= 5 s, lossless
(KL <= 0.0035 nats/token, top-1 >= 98.2 % vs config B). Lossy mode is a labelled switch. 140 W cap fixed. Context
operated at 262,144 (model maximum 1,048,576). Baseline = vLLM config B (`docs/BASELINE.md`).
Local rules: `~/.claude/skills/inference-optimisation/SOURCE.md` (nothing published, no sudo, numbers under `bench/raw/`).

Labels: M measured, D derived, I inferred. Every row's raw data: the folder named.

## Ceilings (what bounds each metric)

| Metric | Bound | Evidence |
|---|---|---|
| Decode step | **Energy**: time = joules / 140 W. Config B (4 sessions, depth 7, 45K): 45.7 ms, 6.4 J, 915 MHz of 1,695 (M) | `bench/raw/2026-11-22-P5-ablations`, `P5b-byte-levers` |
| Decode bytes | 1.3-1.4 ms and 0.2 J returned per GB removed, every class (M) | `P5b-byte-levers` |
| Prefill | **Energy**; cuBLAS bf16 GEMM at the cap = 101 TFLOPS (M) | `2026-11-22-P9-prefill` |
| Attention with softmax | About 81-84 TFLOPS: softmax costs 19 % of the GEMM rate (M) | `2026-11-23-S4-prefill-attn` |
| Experts, prefill | 100-105 TFLOPS without decode; weight decode costs 14 % (M) | `2026-11-23-S4-prefill-moe` |
| Decode attention, 45K, 8 rows | Loads-only 3.95 ms per 8 global executions; ours 6.4-6.7, config B 7.2 (M) | `2026-11-23-S2b-decode-attn` |

## Levers: accepted

| # | Lever | Result vs config B | Lossless | Card(s) | State | Raw |
|---|---|---|---|---|---|---|
| A1 | Prefill attention kernel (FA2-shape tiles, fixed absolute key splits, aligned bias table) | Global 1.65-1.90x (6D), 1.67-1.95x (67) at C >= 32K; SWA 1.33-1.42x (M) | Invariant; model-level KL still to run | 4, 7 | Kernel done, reviewed, no defect | `S4-prefill-attn`, `-review` |
| A2 | Prefill expert kernel (tokens on the N side, decode once per CTA) | 1.35-1.46x Humming (6D); NVFP4 1.35-1.44x (67) (M) | Bit-equal to Humming at Q >= 2,048 | 4, 6 | Kernel done, review finishing | `S4-prefill-moe` |
| A3 | Decode attention kernel (register kernel, fixed key chunks, head-major KV) | Time 0.919 at 45K, 0.858 at 262K (n=4, k=7, same real KV); n=8 0.923; n=1 1.062 (loss) (M) | Batch-invariant; error = config B's | 6 | Kernel done; review pending | `S2b-decode-attn` |
| A4 | Ragged verify width per session | Mixed step = row sum + 4.3 % (M) | Bits independent of other sessions | 6 | In A3 | same |
| A5 | Drafter: int8 head with exact bound + E4M3 draft KV | Drafter time 0.662 (4, short), 0.691 (4, 45K) (M); acceptance x1.0025 (M) | Output-lossless by construction (target verifies) | 7 | Kernel done; review pending | `S2f-drafter` |
| A6 | Confidence-gated verify width | +8.9 % (5 graphs) to +10.6 % (row sum) tokens/s (D from trace + measured cost curve) | Yes | 7 | Needs the fused step | `P10-verify-width` |
| A7 | FP8 (E4M3) KV, lossy switch | Decode attention time 0.758 at 45K, 0.626 at 262K (M); output error 2.5 % on real KV (M) | **Lossy** | 6 | Kernel path done; hard-suite gate not run | `S2b-decode-attn` |

## Levers: rejected or parked (do not rerun without a new reason)

| # | Lever | Result | Why |
|---|---|---|---|
| R1 | Own decode expert kernel (f16, bf16-out, split-K dispatch, int8 limbs) | 0.91-1.06x Humming; fails the every-T gate (M) | Tie. Launch Humming from C++ (H0-H3 proof). f16 activations pass KL (0.00191) if revisited |
| R2 | Prefill expert kernel at decode shapes | 0.89-0.99x at T 1-16, 1.01 at 32, 1.09 at 64 (M, NVFP4, review interim) | Humming stays for decode; candidate only for T >= 64 |
| R3 | Removing glue | Returns 0.21-0.26 ms per ms removed (M) | Energy-bound: glue is cheap in joules |
| R4 | History-based adaptive width | 0 to +2 % (D) | P2 |
| R5 | Padded mixed widths (config B as is) | +2.5 % (D) | Under the noise line |
| R6 | Drafter exact shortlist to 0.5x | Not achievable: no cheap rigorous bound on rare rows (M) | int8 over all rows is the exact optimum |
| R7 | Drafter id-prefix shortlist (first 49,152 rows) | Acceptance x0.32 (M) | Token ids are not frequency-ordered |
| R8 | Decode attention: fused combine, L2 prefetch, transposed/interleaved KV, more CTAs | Slower or no gain (M) | Ledger in `S2b-decode-attn/LEDGER.md` |
| R9 | Attention 2x in prefill | Ceiling about 1.9x (M) | Softmax share |
| R10 | Per-session-count attention split (n=1 +20 %) | Breaks bit invariance | Kept invariant design; n=1 costs 6 % |
| R11 | `last_loop` KV cache sharing (PR `huggingface/transformers#51161`, pass 2 KV reused by passes 0 & 1) | KL `0.25215` vs `0.00480` (**52.5×**), top-1 `84.97%` (−12.72 pp), greedy `5/20` fail, `47/47` cases degraded; breaks verify invariance (`471–512/8,000` bad argmax, −8–10% acceptance) (M, GPU 2 `6D`, `bench/raw/2026-11-24-last-loop-kv/`) | **Hard kill** on `Inkling-2-Mini`: trained without `last_loop` KV sharing; breaks both quality and speculative verification |
| R12 | Exact-NVFP4 `mma.s8` (`probe_s8_real.cu`) & `acc16` decode MoE | `mma.s8` `687 GB/s` vs bit-exact BF16 `NVFP4_N1S1` `716 GB/s`; `acc16` only `+0.12 ms` and breaks bit-identity (M, GPU 1 `6D`, `bench/raw/2026-11-24-frontier-stack/`) | Per-16 `I2F + FFMA` scale overhead makes `mma.s8` slower than `NVFP4_N1S1` BF16 HMMA on `sm_80` |

## In-engine state (M, native C++ engine at `d57c9e6`, GPU 4 6D) and projection

Corrected 2026-11-23 after the Fable review (`docs/LEVERS-REVIEW.md`): the engine as built is **slower than config B
in both phases**. Every "x config B" above 1 in the accepted table is a standalone-kernel ratio.

| Metric | Config B | Native engine, measured | Note |
|---|---|---|---|
| Depth-7 verify step (4, 45K) | 44.7-45.8 ms incl. drafter | 47.5 ms **without** drafter (about 50 with, D) | glue 15.3 vs 5.1, dense 6.3 vs 3.5 |
| Plain decode step | 22.0 ms | 28.6 ms | |
| Cold prefill | 4,531 tok/s (32K), 3,061 (130K) | about 3,430 (45K), 2,940 (120K): 0.80-0.92x (M/D, one session, eager) | kernels 1.3-1.9x standalone not yet realised in-engine |
| Earlier projections (decode about 43 ms, prefill 1.46-1.50x) | | **No in-engine support.** With all glue ablated the verify step is still 43.25 ms before the drafter | withdrawn as estimates of the built engine |

Stage 2 update (M, 2026-11-23, GPU 4 6D, same-card config B controls before and after; `bench/raw/2026-11-23-native-rt/{a1,b12,b16}/`, native 0.2, committed as `3467d5f`):

| 45K contexts | Config B | Native | Ratio (time) |
|---|---|---|---|
| n4, depth 7, step incl. drafter | 45.2 ms | 50.3 (A1 build) | 1.113 |
| n4, depth 7, target forward only | 39.7 | 47.7 (A1 build); b16 build 45.4 for the engine step without drafter (−1.32 ms vs its control 46.7) | 1.20 |
| n4, depth 0 | 21.2 | 27.3 → 26.5 (b16) | 1.29 → 1.25 |
| n1, depth 7, target | 19.3 | 28.2 → 25.0 (b16 step) | 1.47 |
| n8, depth 7, target | 62.7 | 64.7 | 1.03 |
| Drafter | 4.0 | 2.43 | 0.61 |
| Acceptance, 20 golden prompts | 3.11 tokens/round | 3.50 (bf16 draft KV), 3.32 (E4M3: rejected) | |

Gap is a near-fixed 8-9 ms per step at n1 and n4: dense 4.84 vs 3.5 ms, glue 6.35 + router 0.81 vs 5.1, experts 22.99 vs 21.5;
attention 9.15 vs 10.0 (ahead). Rejected: skinny projection kernel for router/gate down-projections (+1.42 ms) and for the
o projection (+2.1 ms). Speculative path check passes (gap rate 11.67 per 10,000 tokens, limit 15).

**Tokens per second on the golden workload** (M, 2026-11-23, GPU 4 6D, same card, 20 golden prompts x 600 tokens, short
contexts; native 0.2 at `6ff2f31`; `bench/raw/2026-11-23-native-rt/{gw1,gw2,cfgb-golden}/`):

| Sessions | Config B tok/s (tok/J) | Native, fixed width 8 | Native, confidence gate tau 1 | Best native / config B |
|---|---|---|---|---|
| 1 | 169.4 (1.218) | 135-138 | 140.4 (1.013) | 0.83 |
| 4 | 389.6 (2.806) | 311 | 324.6-334.5 (2.34-2.41) | 0.83-0.86 |
| 8 | 516.7 (3.689) | 466-507 | 495.1 (3.577) | 0.96 |

**The engine is behind config B in served-equivalent throughput.** Causes: acceptance 0.952x config B's tokens per step per
prompt (14 of 20 lower; the earlier "3.35-3.50 vs 3.11" compared different token budgets and is withdrawn), and round time.
Confidence gate (A6): measured +4 to +8 %, not the derived +8.9 to +10.6 %; the margin predicts acceptance with AUC
0.60-0.72 only; off by default. Process-to-process noise on acceptance-dependent numbers is ±5-8 % until GEMM algorithms
are pinned (start-up timing picks differ in 2-3 shapes per run).

**Acceptance study** (M, 2026-11-23, GPU 4 6D, pinned GEMM algorithms; `bench/raw/2026-11-23-native-rt/acc/`): forced along
config B's own greedy output (identical contexts, 20 prompts x 600 tokens) the native drafter gets 2.271 accepted drafts per
round against config B's 2.261 = **1.004x**. The 0.952x above was greedy trajectory divergence (outputs part after 4-151
tokens; tokens per round on one prompt swing up to 1.7x between runs of the same engine). Free-running golden tokens/s
carries about ±10 % from the path alone and is **not a head-to-head measure**. From now on engines are compared as
**step time x forced acceptance** (both forced along config B's output), or on much larger prompt sets. The acceptance
gap is closed as "none"; the decode gap is step time only. Margin gate rerun on pinned algorithms: +2.0 % tok/s at 4
sessions, +0.7 % at 8 (the +4-8 % was path noise). Config B's own drafter probability predicts acceptance with AUC 0.85
against the margin's 0.60: a probability-based gate is the open form of lever A6.

Also corrected: "101 TFLOPS" is a wave-0 number, not P9; A2 on the 67 card is 1.31x on MXFP8 at Q 6,000 (below the
1.35 target); the KL gate sits at its noise floor (three equally exact configurations: 0.00295, 0.00322, 0.00373).

## In-engine prefill (M, 2026-11-23, GPU 5 6D, 140 W; branch `worktree-agent-ad98827265c0f1eae`, native 0.2 @`3af33d9`, not yet merged)

Records `bench/raw/2026-11-23-native-prefill/` (on the branch). Config B booted before and after (agree within 1 %); native
before = `0de00bd` chunk 2,048, after = `1ff7f31`+ chunk 4,096; 3 rounds, at the cap, so time ratio = energy ratio.

| Cell | Config B s | Native before (x B) | Native after (x B) |
|---|---|---|---|
| cold 32K | 7.217 | 7.331 (0.98) | **5.435 (1.33)** |
| cold 130K | 42.187 | 36.724 (1.15) | **29.071 (1.45)** |
| warm 32K + 1,600 | 0.490 | 0.407 (1.20) | **0.331 (1.48)** |
| warm 32K + 6,000 | 1.612 | 1.490 (1.08) | **1.171 (1.38)** |
| warm 123K + 6,000 | 2.834 | 2.142 (1.32) | **1.800 (1.57)** |
| 32K + 1,600 while 3 decode | 0.550 | | **0.346 (1.59)** (decoders at a fixed 4 tokens per call) |
| 32K + 6,000 while 3 decode | 1.709 | | **1.263 (1.35)** |

Gains: glue written for decode row counts was 34 % of cold 32K (gate tail, relative logits, routing layout, bias table);
fixed without changing bits (teacher-forced output identical, invariance 0 bits for batch, graph, chunks 700/1,111/4,096).
Glue is still 2.2-2.5x config B's; ceiling with config B's glue cost 1.55-1.72x (D). Rejected: unrolled AttnRes loops (0 %),
CUDA graph for prefill (host gaps 0.1 %). Open defect: decode rows inside a mixed call are not bit-equal to a decode-only
call (round 2, item 0). Teacher-forced KL in this configuration (glue 6, moe 1) 0.00378 / 97.73 %, same before and after.

Round 2 (M, GPU 5, merged at `cca45bb`; config B control boot C on the same card after the native runs; cells overlapping
the 13:38-13:51Z CPU-load window rerun): cold 32K 5.492 s = **1.30x** (chunk 8,192: 1.34x), cold 130K 28.825 s = **1.44x**
(1.47x), 32K + 1,600 1.40x, 32K + 6,000 1.36x, 123K + 6,000 **1.55x**, 32K + 6,000 while 3 decode 1.31x. Targets 1.5x /
1.6x not reached: at cold 32K experts 42 %, attention 19 %, dense 13 % (cuBLAS rate), glue 26 %; with config B's glue cost
1.53x, zero glue 1.77x (D). Accepted: decode rows of a mixed call bit-identical to a decode-only call (0 bits; one mixed
call vs two: −4.1 % time, −4.2 % J); prefill_moe defects fixed with an int chunk offset (the `size_t` form cost +4.4 %
expert time). Rejected: AttnRes/norms one warp per row (+4.2 %), tensor-core gate tail (−0.5 %; switch to be deleted),
single GEMM plan class (+6.5 % / +5.7 % prefill time). Chunk 8,192: 0 bits, −2.7 to −4.2 %, but +12 GiB arena and
1.60 s decoder stalls; not adopted (4,096 stays).

Glue, second look (M, GPU 5 6D, `bench/raw/2026-11-23-prefill-glue2/`; branch `worktree-agent-abf2a35c8987044fd` `b18b238` on master
`8bc2d40`, C++ only, Rust port pending). The earlier "fp32-bound" verdict was wrong: AttnRes, RMSNorm and short-conv kernels
ran at 0.13-0.45 TB/s (one 256-thread block per row, 2-byte scalar loads) where config B's run near 1.4 TB/s; and 261 ms of
attention-side work had been counted as glue (like for like 1,347 vs 638 ms at cold 32K). Five bit-identical changes
(teacher-forced output identical, ink-inv 0 bits): AttnRes and norms one warp per row with 16-byte loads −4.4 % at cold 32K,
short convs −1.4 %, gate inputs −0.4 %, q/rel post −0.3 %, gate tail −0.2 %. Against config B booted on the same card at a
matched start temperature (boot B; the earlier boot ran 2-4 % slower hot):

| Cell | Master | Final | Config B | x config B, master → final | + tensor-core gate tail (G7) |
|---|---|---|---|---|---|
| cold 32K | 5.456 s | 5.091 s | 6.874 s | 1.26 → **1.35** | 1.38 |
| cold 130K | 29.12 s | 27.77 s | 41.03 s | 1.41 → **1.48** | 1.51 |
| 32K + 6,000 | 1.163 s | 1.097 s | 1.533 s | 1.32 → 1.40 | 1.43 |
| 123K + 6,000 | 1.806 s | 1.740 s | 2.737 s | 1.52 → 1.57 | 1.59 |
| 32K + 6,000 while 3 decode | 1.290 s | 1.220 s | 1.648 s | 1.28 → 1.35 | 1.38 |

G7 (prefill rows' message-gate tail on bf16 tensor cores; decode rows unchanged; batch-, chunk- and mixed-call-invariant):
changes bits; plus set mean KL 0.00483 vs control 0.00490 (paired −0.00007 [−0.00029, +0.00016]), full envelope PASS, greedy
0/20 fail: **accepted** (head). Decode cell unchanged (44.43 vs 44.40 ms). Remaining glue at cold 32K 908 vs 638 ms; ceiling
with config B's glue cost about 1.43x; beyond that only the expert (42 %) and attention kernels move prefill.

G7 defect, found after merge (M, GPU 5, `bench/raw/2026-11-24-rust-delta3/` "Post-merge defect and fix"): as landed, G7 chose the
tensor-core tail by call size (T > 64), so a prefill row's bits depended on its call's row count; Rust `resume_vs_cold` failed on
master `f5c9169` (E2100 cold ends in a 52-row chunk; E1509 resumes 9 rows). Missed because every chunk-invariance test used
chunks above 64 rows. Fix `37a8f63`: chosen by row class (every prefill row group on the tensor cores), both runtimes; new ink-inv
`small_chunk_*` rows (calls of 9-65 rows, alone, mixed, resumed) 0 bits. Plus set: 99 of 4,072 positions change (3 short cases),
KL 0.00483 unchanged, full envelope PASS, greedy 0/20.

## Served engine, first end-to-end numbers (M, 2026-11-23, `bench/raw/2026-11-23-rust-I1-served/run2/`)

Configuration M3: Rust front end + scheduler 0.1.0 @`20537fc` (dirty; committed as `81f937c`) over `ink_rt.h` v1 on C++ `native`
0.2.0 @`66fe353`, GPU 7 (vBIOS 67), 140 W. **Plain decode, host argmax, no drafter: not comparable to config B's
speculative serving.** Gate: greedy tokens equal to the engine called directly, 20/20 alone and 20/20 at 4 concurrent.

| Case | Result (median of 3) | Note |
|---|---|---|
| Decode, 1 session, 45K | 50.1 tok/s, 20.0 ms/step | 133 W, 1,410 MHz: not at the cap |
| Decode, 4 sessions, 45K | 31.2 tok/s each, 124.8 total, 32.0 ms/step | at the cap, 1,320 MHz; host argmax 2.76 ms of the step |
| TTFT cold 32K | 7.48 s (4,380 tok/s) | config B P9 on GPU 5 (6D): 7.2 s; different card type |
| TTFT warm, 1,600-token suffix | 0.42 s | 32,768 tokens reused (exact resume only) |
| TTFT, 4 cold 45K at once | 11.1 / 23.7 / 36.4 / 49.2 s | prefills serial; no interleave yet |

Found: a prefill chunk in the same engine call changes decode bits (batch-invariance defect in the C++ mixed call; adapter
uses two calls until fixed). Host argmax costs 0.68 ms per row: device argmax through the seam is requested.

## Served engine with speculation (M, 2026-11-23, GPU 6 vBIOS 67, `bench/raw/2026-11-23-rust-I2-spec-served/`)

Rust front end + scheduler 0.1.0 and native 0.2.0 @`453403d` (clean), pinned algorithms; config B booted before and after on
the same card. One 45K prompt, **the same prompt in every session**, greedy, 1,024 tokens, median of 3.

| Sessions | Engine step ms | Config B step ms | Ratio | Engine tok/s | Config B tok/s | Tokens/step E vs B |
|---|---|---|---|---|---|---|
| 1 | 28.80 | 22.25-22.38 | 1.29 | 186.0 | 238-240 | 5.33 vs 5.31 |
| 4 | 47.39 | 37.91-38.09 | 1.25 | 454.8 | 676-679 | 5.36 vs 6.39 |
| 8 | 64.89 | 54.06-54.34 | 1.20 | 666.7 | 676-679 | 5.38 vs 4.56 |

Caveats: (1) identical prompts let config B share KV blocks between sessions through its prefix cache (its 4-session step
is 37.9 ms here against 45.2 ms with distinct contexts on GPU 4); the engine does not share pages. Rerun with distinct
prompts pending (I). (2) Config B's text changes with its batch, so tokens per step differ on the same prompt.
TTFT: cold 32K 6.24 s (config B 6.79), warm +1,600 0.365 s (0.451), 4 cold 45K at once 9.3/20.7/32.2/43.8 s
(10.6/21.2/31.8/42.4), longest decode stall during another prompt's prefill 0.385 s (0.55). Host time 0.24-0.28 ms/step.
Rerun with distinct prompts (M, GPU 0 vBIOS 67, same engine build, `…/distinct/`): config B did NOT share KV in the
same-prompt runs (per-session cache salt); the same prompt simply makes both engines faster. **Distinct 45K prompts: step
31.07 vs 23.35 ms (1.33x) at 1 session, 51.87 vs 47.45 (1.09x) at 4, 70.71 vs 71.63 (0.99x) at 8**; J/token 1.14 vs 0.88,
0.448 vs 0.459, 0.337 vs 0.366. Shared 8K prefix + distinct tails: same step ratios (1.34 / 1.09 / 1.00); config B shares
the prefix (TTFT 9.3 vs 10.9 s per prompt, KV −12 % at 8 sessions), the engine does not (9.56 s cold): prefix page sharing
is worth about 1.5 s TTFT per prompt and KV capacity, nothing in step time. Free-running tok/s (engine 1.02x at 4, 1.09x at
8 on distinct prompts) follows the text each engine happens to generate: compare step ms.
**Defect F1 (fixed on a branch, merge pending; see below):** speculative greedy output equals plain decode on 3/20 prompts only: verify rows give another argmax than
1-row decode at the same context (4 of 8,000 tokens by more than 1.0 nat, worst 10.24, all on 2.7-2.8K agent prompts);
`ink-spec` reproduces it, so it is in the engine. Speculation is not lossless under the greedy clause until fixed.

**F1 resolved (M, GPU 6, `bench/raw/2026-11-23-f1-verify-rows/`, merged at `46d9a18`).** Cause: decode_attn design A put a verify
block's columns in two tile shapes (n8 + m16) with different fp32 summation order, so verify rows 2-7 rounded 1 bf16 ulp
differently from 1-row decode in a few attention outputs (exactly one of 6,416 op-rows differed with write-back); accepted
rows committed that drift into KV and it accumulated; the agent prompts have bifurcation positions where a rounding-size
change flips a confident choice. Fix: three n8 tiles. New permanent test `verify_rows_vs_decode`: 0 differing bits (336 rows,
native and rust runtimes); path check 0 of 8,000 tokens off the plain argmax (was 56, worst 10.24 nat); served speculative ==
plain 20/20 + 20/20; step +0.05 ms at 4x8 (spread 0.06). Config B has the same property: speculation on vs off, forced along
the same tokens, flips at largely the same positions with gaps up to 7.46 nat. So the engine's speculative mode is now
exactly lossless against its own plain decode, which config B's is not.

**Speculative sampling built (M, GPU 6 vBIOS 67, `bench/raw/2026-11-23-spec-sampling/` on branch
`worktree-agent-a1fe201a716453732`, native 0.2.0 @`483a6ae`, before D1/X1; merge pending after the delta port).** Rule:
each verify row's token is sampled on the device (Philox keyed by seed and position, exact integer-weight draw after
temperature / min-p / top-k / top-p), a greedy draft is accepted iff it equals that token: the same acceptance probability as
config B's rule, and the emitted tokens equal plain sampling at the same seed. Gates: speculative == plain tokens 20/20 (tool
and served, alone and 4 concurrent), filtered distributions bit-identical on 7,980/7,980 positions, distribution test passes
(24 tests, min p 0.047), alone == concurrent 20/20 (config B: 0/20), temperature-0 path unchanged. Sampler cost +0.17-0.25 ms
per step. Served at temperature 1.0 / top_p 0.95, distinct 45K prompts, 1 / 4 / 8 sessions: engine speculative 143.6 / 275.8 /
405.9 tok/s (plain 56.3 / 142.6 / 204.0), config B 171-210 / 276-313 / 360-412; step 29.6 / 50.9 / 68.9 ms vs 22.9 / 44.5 /
66.9 (1.29 / 1.14 / 1.03x); golden prompts step 1.34 / 1.24 / 1.13x. Acceptance equal to config B (2.879 vs 2.880 tokens per
step on the golden prompts).

Merged to master 2026-11-24 with the Rust runtime port (both runtimes give the same tokens at the same seed; host sampler
deleted). On master's tree (D1 in), all windows starting at 45 °C: speculative step 28.88 / 49.48 / 66.73 ms against config B's
23.1 / 43.6 / 66.0 (1.25 / 1.13 / 1.01x) at 1 / 4 / 8 sessions; acceptance 2.799 vs 2.880 tokens per step.

**Sampling gap (found 2026-11-23, closed by the above):** the M1 agent-replay runs at temperature 1.0, top_p 0.95, seeded; the served engine
speculates only at temperature 0 and serves sampled requests one row per step. Speculative sampling is in work (GPU 6).

## Desk lever tests (CPU, 2026-11-23, `bench/raw/2026-11-23-desk-lever-tests/`; gains are D on M data, not additive)

| Lever | Result | Verdict | Revised M1 gain |
|---|---|---|---|
| Keep the previous completion's KV | Previous completion = 22.0 % of config B's uncached prefill tokens (M); re-render identical in 189/193 turns | **Go** | about +3 % |
| Lookup drafts after a fully accepted round | 33.8 % of generated tokens lie in copied spans >= 8 tokens (M); 4.06 → 5.00 tokens per round with 8 lookup rows | Desk: go. **Measured in-engine (GPU 5 6D, forced mode, `bench/raw/2026-11-23-lookup-drafts/`, branch `worktree-agent-a6cd38b5d7b2aeacd` `862f4a0`): +2.6 % tok/s, +2.9 % tok/J at 4 sessions on agent turns (+3.0 % at 1), golden +0.7 % at 4, prose ±0.2 %; lossless (path check pass, batch-invariant). Rows 9-16 cost 0.48 ms per session-row at 4 sessions because decode_attn takes 8 rows per pass and the second pass re-reads the KV; agent at 8 sessions ran out of device memory.** Parked, not merged: small gain against a wide engine change that collides with D1/X1 and the Rust port; revisit with a 16-row single-pass decode_attn (estimate about +4 %, I) | +2.6 % measured (desk said +7.8 %) |
| Standard speculative sampling at temperature 1.0 | 4.26 tokens per round vs 3.95 greedy (+7.9 %; +12.7 % on 45K prompts) (M on temperature-0 contexts) | **Go**; Gumbel coupling is worse | +3.5 to +5 % |
| Lossy: drop small router slots | No small-weight tail: smallest slot carries 6.6 % on average | **Kill** | 0 |
| Skip V for negligible key tiles | 4.9-9.1 % of tiles skippable within 1 % added error; exact bound skips 0 % | **Kill** | about 0 |
| Window the drafter's global attention | Accepted drafts −7.9 to −10.5 % | **Kill** | negative |

Caveat found: running `engine/ref` twice on CPU on the same input gave KL 0.34-0.37 between runs in the tester's harness;
unresolved (harness bug or non-determinism on CPU). Do not use `engine/ref` on CPU for KL work until checked.

## Lossless gate (2026-11-23, `bench/raw/2026-11-23-gate-noise/`)

Config B against itself (M, GPU 6): second boot bit-identical; with 3 concurrent sessions KL 0.0030-0.0033 and top-1
97.99-98.74 %; prefill chunk 8,192 KL 0.0039-0.0049; batched greedy 0/20 identical, 4-5/20 inside the 0.05-nat tie rule.
So the absolute gate (KL <= 0.0035, top-1 >= 98.2 %, greedy 20/20) is failed by config B itself and cannot decide.
Working gate from now (head; owner to confirm): `compare.py band`, the envelope of config B's own served configurations
(alone, batched, speculation off) on the 4,072-position plus set: KL limit about 0.0036, top-1 about 97.5 %, at least
2,000 positions; false pass 5 % at +0.001 nats. Engines scored only on the old 396 positions are **undecidable**.

Engines on the plus set (M, GPU 6, `bench/raw/2026-11-23-gate-noise/plus-engines/`): native `d57c9e6` b5 / g1 / b4 mean KL
0.00480 / 0.00440 / 0.00482, native 0.2 `6ff2d00` 0.00499 (per-class GEMM plans) and 0.00503 (decode-class only: no
difference, interval [−0.0008, +0.0007], so plan class is chosen on speed), PyTorch reference 0.00449 (2,098 positions).
All **fail the strict envelope** (limit 0.0037-0.0041) and **pass the full envelope** that includes config B's own
chunk-8,192 setting (0.0049 against itself; limit 0.0053-0.0056). Top-1 97.7-98.0 %. Per-case KL tracks config B's
chunk-8,192 variant (r 0.78-0.89): the distance looks like prefill numerics, not a defect (I). The old 396 positions
flattered the engines (b5: 0.00295 there, 0.0042-0.0057 on the new long cases).

## Decode gap study (M, 2026-11-23, GPUs 7 and 0 vBIOS 67, `bench/raw/2026-11-23-decode-gap/`)

| # | Finding / lever | Result | State |
|---|---|---|---|
| D1 | Dense gap = our skinny projection kernel: 3.65 ms / 0.507 J standalone per step (32 rows) against config B's 2.54 / 0.358. Projections via cuBLASLt over one contiguous weight, one pinned algorithm | In-engine −1.00 ms / −0.125 J at 4x8, −0.58 at 4x1, −0.55 at 1x8; **bits identical** | Accepted; patch being applied by the runtime owner |
| | o projection, head | 0.957 vs 0.931 ms, 1.220 vs 1.240 ms: parity | no lever |
| | Experts "+1.5 ms" | Measurement mismatch (ablation cost vs stamp duration); measured alike, ours 20.43 ms vs config B 20.8-20.95 + 0.46 align | closed |
| X1 | Experts on one fixed config-B Humming kernel for every row count | −0.56 ms / −0.120 J at 4x8, −1.01 at 4x1, −1.05 at 1x8; batch-invariant; **changes bits**. Plus set (GPU 7, `bench/raw/2026-11-23-x1-band/`): D1 + X1 mean KL 0.00506 vs control 0.00490, paired +0.00016 [+0.00003, +0.00029], top-1 97.67 %, greedy 0/20 fail: full envelope PASS (limit 0.00541, margin 0.00035) | Accepted on quality, then **withdrawn as default (2026-11-23)**: it needs the experts resident a second time in Humming format (+15.7 GB; arena at 8 slots / 256 pages 53.9 GB), which breaks the 4 x 262,144 context budget on a 64 GB card and already ran out of memory at 8 sessions. Parked until one weight copy serves both the decode and the prefill expert kernel |
| G1 | Gate down projections on a pinned split-K cuBLASLt algorithm | 1.85 → 0.70 ms standalone; changes bits | With the glue owner |
| | Rejected | prefetching skinny kernel (slower than D1), per-group tuned fused algorithms (changes bits on 28 visits), Humming batch-invariant set as built (+0.87 ms) | |

Prefill numerics attribution (M, GPUs 6 and 3, `bench/raw/2026-11-23-prefill-numerics/`): the engines' distance from the
goldens (0.0044-0.0050) is **prefill rounding, not a logic defect**. Config B prefilled the goldens in 2,048-token chunks from
position 0 with a 0 % prefix-cache hit rate, the same schedule as ours. Any bit-level change of the prompt's stored state
costs about the full distance: config B with only its chunk size changed scores 0.0044 (1,024) and 0.0049 (8,192); a last-bit
flip in 0.3 % of one tensor at the first layer visit during prefill gives 0.0079-0.0085, the same flip on decode rows only
0.00065; all independent prefills (config B's, 5 native builds, the reference) sit 0.0042-0.0057 from each other. Config B
reproduces the gold bit for bit on another card of the same vBIOS. Matching the batched range would need a bit-exact copy
of config B's prefill kernels and would give up the 1.3-1.55x prefill lead: not done. Gate = full envelope (owner);
decode-side numerics are checked separately (decode-only changes cost 0.0005-0.0007).

In-engine confirmation on a 6D card (M, GPU 4, committed `1e161b3` and `25f1fcc`, `bench/raw/2026-11-23-native-rt/glue1/`): C1 (unrolled
glue row loops) −0.15 / −0.12 / −0.25 ms at 1x8 / 4x1 / 4x8-45K; D1 + C3 on top −0.54 / −0.50 / −1.03, bits identical; X1 on top
−1.03 / −0.97 / −0.80 (opt-in `--moe 2` with humming-cache-B until the default switch). 4x8 at 45K: 45.1 → about 43.4 ms
(two chained sandwiches). Rejected: warp-per-output kernel for gate projections and router (+0.83 / −0.32 / +0.97). D1's bits
equal the old kernel's only with its four pin entries (`glue1/pins-glue.json`).
**Short contexts (M, GPU 4, before these changes): native / config B step = 1.41 (1 session, depth 7), 1.39 (4, depth 7),
1.56 (4, depth 0), 1.62 (1, depth 0): a near-constant 6-8 ms of fixed cost.** Host CPU: the bench tool's 7 cores are start-up
only (16 loader threads, GEMM timing); the timed loop uses one spinning core.
Sync mode (M, GPU 4, `…/native-rt/spin/`): blocking instead of spinning in the step loop costs +0.22 ms at 1x8 and +0.11 ms at
4x8-45K (J/step unchanged) and frees one host core per card (1.06 → 0.07): **keep spin** (8 cards = 8.5 of 48 cores). With
X1: 4x8 at 45K = 43.5 ms, 1x8 at 1K = 22.0 ms (engine step without drafter).
Pending, one package: regenerate the kernel pack and port these deltas to the Rust runtime (the pack on master is stale
since `1e161b3`: `check_fresh.py` fails until then), X1 default, gate_tail deletion, production pin file.

Rust vs C++ step time (M, GPU 3, `bench/raw/2026-11-23-rust-R6-graphs/delta-clock/`): the +0.24-0.38 % of the Rust engine was
**thermal history, not the engine**: the Rust process starts in about 75 s against 145 s for C++, so its timed cell began
3-5 °C hotter and ran a lower clock under the cap. With 60 s idle before each Rust run: decode 4x8 45.177 (C++) vs 45.217 ms
(−0.09 %, spread 0.27 %), speculative step 48.311 vs 48.315 ms (−0.01 %, spread 0.12 %). Excluded: arena address and
alignment, module loading, per-kernel resources (266/266 equal), context and stream flags, host time. New measurement rule
7a in AGENTS.md (equal thermal start).

## M1 agent-replay, the owner's metric (M, 2026-11-23, GPU 7 vBIOS 67, `bench/raw/2026-11-23-M1-replay/`)

Wave-0 harness: 16-session pi panel, 343 requests, 118,056 output tokens and 12.6M prompt tokens per round, temperature 1.0,
top_p 0.95, seeded, recorded (teacher-forced) history, closed loop at N sessions. Engine = master `7199786` (before D1, the
sampling work and F1), Rust front end + scheduler over the C++ runtime. Config B before / after agree within 2-3 % and match
wave 0.

| | N | Config B M1 tok/s | Engine plain | Engine speculative | Engine / config B | TTFT p99 s, B vs engine |
|---|---|---|---|---|---|---|
| T 1.0 (as specified) | 4 | 269.9 / 262.5 | 29.0 | 29.7 | **0.11** | 4.94 vs 41.3 |
| T 1.0 | 8 | 299.2 / 305.6 | 31.6 | | **0.10** | 8.72 vs 37.8 |
| T 0 (labelled) | 4 | 293-298 | | 39.4 | 0.13 | 6.37 vs 57.6 |
| T 0 | 8 | 333-335 | | 40.5 | 0.12 | 7.74 vs 65.9 |

**Why: the engine re-prefills 95 % of all prompt tokens every turn (11.9M of 12.6M per round); config B re-prefills 4 %.** The
engine reuses a parked session only when its whole sequence, its own reply included, is a prefix of the next prompt; the
replay's history carries the recorded reply (and live traffic re-renders it), so almost nothing matches. 72-99 % of the
engine's wall time is in steps with a prefill chunk (config B 32-42 %); prefill speed inside those steps is ahead of config
B (4.0-4.1K vs 3.5-3.6K tok/s). Further: no speculation at T 1.0 and host sampling at 6.15 ms per row (being fixed by the
speculative-sampling package); **the runtime's CUDA-graph cache is unbounded and the server ran out of device memory after
20-30 minutes, twice**; `min_tokens` does not suppress EOS as vLLM does; `finish: tool_calls` on 44-61 forced-length turns
where config B says `length`. Derived projection with config B's prefill volume: about 0.27x at T 1, 0.47x with device
sampling, 0.90x at T 0. **Top of the open list from here: prefix reuse across turns, then the graph-cache bound.** All
step-level comparisons above describe decode and prefill kernels correctly but did not exercise multi-turn reuse.

## Headline served decode comparison, matched start temperatures (M, 2026-11-23, GPU 3 vBIOS 67, `bench/raw/2026-11-23-headline-decode/`)

Engine `432f99f` (= master `8bc2d40`'s runtime: D1, F1, one expert path, production pins; without P1-P4, G1, G2), Rust front
end + scheduler, speculative greedy, config B before and after, every cell started at 49-51 °C (3 of 60 cells 2-3 °C colder).

| Cell | Config B step ms | Engine, C++ runtime | Engine, Rust runtime | C++ / config B | Rust / C++ |
|---|---|---|---|---|---|
| distinct 45K, 1 session | 22.49 / 22.43 | 29.08 | 28.93 | 1.294 | 0.995 |
| distinct 45K, 4 sessions | 44.03 / 43.90 | 49.51 | 49.60 | 1.126 | 1.002 |
| distinct 45K, 8 sessions | 66.66 / 66.56 | 66.62 | 66.64 | 1.000 | 1.000 |
| short (2.7K and 73 tokens), 1 session | 20.44 / 20.43 | 28.61 | 28.56 | 1.400 | 0.998 |
| short, 4 sessions | 35.52 / 35.41 | 42.93 | 42.88 | 1.210 | 0.999 |

The earlier GPU 0 figure of 1.09 at 4 sessions was flattered by that card (config B 47.5 ms there against 44.0 here and on
GPU 6). Known recoverable items at 4 sessions (D, from separate measurements): P1-P4 about −1.4 ms, G1 about −1.5, drafter
drafting only the speculating slots (G2) about −1.6: together about 45.0 against 44.0 (1.02x). The engine wins tok/s nowhere
in this run.

## Fixed per-step cost (M, 2026-11-23, GPUs 0 and 4, `bench/raw/2026-11-23-fixed-step-cost/`; branch `worktree-agent-a3761eb037c6afab1` `ea2fe2e`, merge pending)

Both engines traced (CUPTI, no SYS_ADMIN) at 45K on GPU 0, old base `07efcdb`, ms per step:

| Class | 1x8 ours / config B | 4x8 ours / config B |
|---|---|---|
| Launch floor (empty kernels, our 1,094 nodes and grids) | 1.01 (config B has more nodes: 1,268 + 154 + 37) | 1.19 |
| Small-grid glue: AttnRes, gate down projections, gate tail + norms, router, routing layout | **8.26 / 3.22** | **10.91 / 4.19** |
| Head, attention prologue, other glue | 1.80 / 1.82 | 2.47 / 2.20 |
| Host gaps (served) | 0.43 / 0.20 | 0.50 / 0.20 |
| Experts / attention | 7.58, 3.18 / 8.01, 3.02 | 18.55, 7.62 / 22.61, 9.74 |
| **Target forward** | **24.54 / 19.83** | **44.30 / 42.96** |

The fixed cost is not launch count and not host time: it is about six small glue kernels per layer visit that are slow per
call at any row count (AttnRes 24.7 µs against config B's 6.8; gate down projections two 1-CTA GEMMs of 20 µs against one
8.4 µs kernel). Accepted, all bit-identical (8,396 logit-row hashes equal, ink-inv 0 bits), arms within 1 °C: P1 row kernels
load their row into registers up front, P2b routing-layout prefix sums as warp scans, P3 gate tail reads weights once per 32
rows, P4 independent block sums overlapped: **P1-P3 together −1.14 ms (−4.7 %) at 1x8, −1.26 (−2.9 %) at 4x8, −1.25 at 4x1,
−1.70 at 8x8; P4 −0.13 to −0.18 more.** Rejected: folding the top-8 router selection into the layout kernel (−0.23 / 0).
On master since `d455131` (delta 2), confirmed on a third card with the production pins and equal start temperatures
(M, GPU 2 6D, `bench/raw/2026-11-23-rust-delta2/`): −1.38 ms (−3.1 %) at 4x8 45K, −1.25 (−5.1 %) at 1x8 45K, −1.25 (−5.4 %)
at 1x8 1K against master `8bc2d40`; Rust runtime within 0.07 ms of C++.
Remaining (I): target forward about 3.3 ms behind config B at 1x8, level at 4x8; gate down projections +1.5-1.7 ms (G1,
changes bits, in work); AttnRes; served drafter 3.02 vs 2.53 ms. Launch-fusion hypothesis: refuted (config B has more nodes
and is faster). Global attention reads each key once for all verify rows: confirmed (D).

## G1 gate-down projections + AttnRes query split (M, 2026-11-23/24, GPUs 0 vBIOS 67 timing, 4 and 1 vBIOS 6D quality, `bench/raw/2026-11-23-g1-gate-proj/`; branch `worktree-agent-acc4678dafddeafdb`)

| # | Lever | Result | State |
|---|---|---|---|
| G1 | Decode gate-down GEMMs (n 64, k 4608, 96 per step) on split-K 32 with fp32 partials and cuBLASLt's fixed-order reduction (COMPUTE_TYPE), one GEMM per gate, 16 MB workspace; pin entry `SPLITK_NUM` 32 `REDUCTION_SCHEME` 2 | Standalone 1.84 -> 0.65-0.71 ms per step; batch-invariant, verify rows == 1-row decode; band (plus set): KL 0.00488 vs 0.00490, paired -0.00002 [-0.00009, +0.00005], greedy 0/20, full envelope PASS | Accepted (head), landed in both runtimes |
| C | AttnRes with 3 or 4 queries in decode calls as one block per (row, query) (config B's grid) | bit-identical; -0.12 ms at 1x8, 0 at 4x8 | Accepted, landed |
| A | The 3 gates of a stage as one strided-batch GEMM (config B's `bmm` form) | bit-identical, -0.77 / -0.88 ms on split-K 1; **under G1 never ran**: cuBLASLt refuses every batch of 3 with the split-K algorithm (`land/gd-check/`), the C++ fell back to per-gate calls (the measured and scored path) | Removed at landing |

Package (C + G1 on the D1 control, GPU 0, 3 rounds, rule 7a): -1.10 ms (-4.8 %) / -0.155 J at 1x8 45K, -1.23 (-2.9 %) at 4x8,
-1.19 (-4.6 %) at 4x1, -1.21 (-5.1 %) at 1x8 1K, -1.18 (-2.9 %) at 4x8 1K; 8x8 not resolved (about -0.2 ms, under the 0.5 %
line). Landed on master `f207c29` (`land/timing2/`, B N R B N R B, master's C++ ink_bench as B, commit 4): **C++ -1.06 ms
(-4.4 %) at 1x8 45K and -1.06 (-2.4 %) at 4x8; Rust -1.10 (-4.6 %) and -1.07 (-2.4 %)**. Prefill (cold 32K) bits identical,
time -0.2 %.
**Correction** to the fixed-cost study above and its records: config B's `_k_group_kernel` (48 calls, 8.4 µs, 12 CTAs) is
its **router** GEMM (`k_group_partials`, N 128, K 2,304 in three 1,024 groups), not its gate-down projections; config B
runs those as `torch.bmm` (one 3-CTA cuBLAS GEMM of 23 µs for a 3-gate stage, split-K 24 + reduce for the 1-gate stage).

## Probability-gated verify width (M, 2026-11-23, GPU 1 6D, `bench/raw/2026-11-23-ptok-gate/`; branch `worktree-agent-a0f768fbcaa5b5e09` `79684da`, C++ only, port and served wiring pending)

Each session verifies 1 + the leading drafts whose cumulative drafter probability stays above 0.7 (widths 1/2/4/8); lossless
by construction. Forced mode along config B's output, 4 controls and 3 gate cells alternating, arms within 1 °C (one pair 2 °C):

| Workload | 1 session | 4 sessions | 8 sessions |
|---|---|---|---|
| Short prompts, tok/s | +5.12 % | **+5.46 %** | +4.30 % |
| 45K prompts, tok/s | +2.70 % | **+2.45 %** | +3.56 % |
| J/token | −5.3 / −2.3 % | −5.3 / −2.4 % | −4.3 / −3.4 % |

Probability from a rank-16 fold of the drafter head: +0.10 / +0.16 / +0.43 ms per round at 1 / 4 / 8 sessions; AUC 0.84-0.89 per
draft position (config B's p_tok 0.85-0.92). Threshold scan at 4 sessions: 0.5 +3.6 %, 0.7 +5.4 %, 0.8 +5.3 %, 0.9 +5.1 %.
Path check passes with the gate on (5.83 per 10,000). Offline: a per-position history controller loses in every cell (−0.4 to
−5.0 %); all widths 1..8 would add 1.4-1.7 points (not built). **Correction of earlier records:** the "margin AUC 0.60" was a
bug in the trace tool (one-round lag); the margin's true AUC is 0.83-0.86, and the margin gate's +2.0 % / +0.7 % stands; the
probability gate wins on threshold and cost, not on a better signal. The margin gate and the full-fold variant are deleted.
(The 45K path-check failure this agent saw with the gate off predates the F1 verify-row fix on its base.)

## Drafter drafts only speculating sessions, G2 (M, 2026-11-24, GPU 3 vBIOS 67, 140 W, `bench/raw/2026-11-23-g2-drafter-slots/`; merged `f207c29`, code `0b5548aa`, base `f981a4d`)

Served native 0.2.0, 8 slots, base / new / base, every cell started 44-47 °C, controls agree within 0.2 %. Step ms; config B = one control run on the same card.

| Cell | base ms | new ms | new / base | new J / base J (D) | rust / new | config B ms | engine / config B before -> after |
|---|---|---|---|---|---|---|---|
| distinct 45K N=1 | 27.39 | 25.53 | 0.932 | 0.935 | 1.003 | 22.40 | 1.223 -> 1.140 |
| distinct 45K N=2 | 36.41 | 33.98 | 0.933 | 0.931 | - | - | - |
| distinct 45K N=4 | 47.72 | 46.05 | 0.965 | 0.966 | 1.000 | 43.16 | 1.106 -> 1.067 |
| distinct 45K N=8 | 63.97 | 63.91 | 0.999 | 0.995 | - | - | - |
| short N=1 | 27.04 | 24.30 | 0.899 | 0.892 | 1.000 | 20.26 | 1.335 -> 1.199 |
| short N=2 | 32.80 | 31.06 | 0.947 | 0.952 | - | - | - |
| short N=4 | 41.20 | 40.14 | 0.974 | 0.947 | - | 35.02 | 1.176 -> 1.146 |

Drafter ms per round at 8 slots x 45K (CUDA events): 4.30 -> 1.73 (1 speculating), 4.39 -> 2.04 (2), 4.50 -> 2.84 (4), equal at 8. A device-mask variant was slower (3.44 / 3.59 / 4.09 ms) and used more energy: rejected. Not done: speculative mixed step in one call (changes the accept kernel on the sampling seam; mixed steps 294.8 -> 294.4 ms, no gain shown). Noted for prefill: the drafter commit of a 2,048-row prefill chunk costs about 41 ms of host time per chunk-only step.

## M1 replay with prefix reuse across turns (M, 2026-11-24, GPUs 6 and 7 vBIOS 67, 140 W, `bench/raw/2026-11-23-prefix-reuse/`; merged `e7a0dd61`)

Rust 0.1.0 @ `332b1363` (gates re-run on the merged tree `be0cfaac`), 504 KV pages, speculative at T 0 and T 1.0; one round per cell; ratio = engine M1 tok/s / mean of config B before and after on the same card. S = resume at the previous prompt's end (bit-identical to cold prefill: `resume_vs_cold` 90/90 rows, panel 327/327 turns, 0 bits). R = S plus reuse of the reply's decode-row K/V when the reply re-renders identically (28 panel turns: 0 argmax flips, KL mean 9.4e-9, max 2.4e-7; not bit-identical).

| Card, history | Variant | T 0 N 4 | T 0 N 8 | T 1.0 N 4 | T 1.0 N 8 |
|---|---|---|---|---|---|
| 7, recorded | S | 0.766 | 0.794 | 0.778 | 0.783 |
| 6, self-fed | S | 0.772 | 0.803 | 0.760 | 0.798 |
| 6, self-fed | R | 0.800 | 0.809 | 0.782 | 0.828 |

Before reuse: 0.10-0.13. Cached share S 0.9570, R 0.9588, config B 0.958. TTFT p99 at N 4: engine 4.49-4.66 s (R 4.47-5.04 s), config B 4.20-6.10 s; at N 8 both are 7.7-9.6 s. 0 failed requests. Self-fed: 171-205 of 343 replies can be fed back (the rest are length cuts inside a tool call, rejected by both servers). Remaining gap = prefill chunk policy under load: engine about 270 tokens per prefill step at 1.95K tok/s vs config B about 1,200 at 3.6K tok/s; at config B's rate S would be about 1.02x config B (D).

## Delta 3: prefill glue rewrites + tensor-core gate tail G7 on master, both runtimes (M, 2026-11-24, GPU 5 vBIOS 6D, 140 W, `bench/raw/2026-11-24-rust-delta3/`; merged `f5c91696`, gated tree `d2675cf`, base master `f207c29`)

Bits: merge without G7 == master (`8ad8302e…`); with G7 native and Rust == the band-scored G7 build (`c1f22d82…`, KL 0.00483 full envelope PASS). Timing B N R B N R B, launches gated on card temperature (rule 7a).

| Cell | master C++ | new C++ | new Rust | config B (start 37-39 °C) | new vs config B (C++ / Rust) |
|---|---|---|---|---|---|
| cold 32K, s | 5.326 | 4.991 | 5.016 | 6.920 | 1.39x / 1.38x (master 1.30x) |
| cold 130K, s | 28.694 | 27.404 | 27.562 | 40.986 | 1.50x / 1.49x (master 1.43x) |
| 32K + 6,000 while 3 decode, s | 1.265 | 1.204 | 1.201 | - | - |
| decode 4x8 45K, step ms | 43.10 | -0.10 % | -0.09 % | - | - |

Rust prefill is 0.5-0.9 % slower than C++ at cold 32K (master's Rust already +0.3 %); cause not located; equal at 130K.

## Rust Runtime Cut-Over (Steps 0 & 2–5, M, 2026-11-24, GPU 2 vBIOS 6D, `bench/raw/2026-11-24-rust-cutover/`)

- **Step 0 (`b98aee70`):** Ptok3 gated verify width merged (`VERIFY_CALL_ROWS = "1,2,4,6,8"`, `--verify-call-rows` configurable; GPU 3 A/B `long-n4` `227.8 -> 240.6` `+5.62%`, `m6` `241.2` `+5.85%`, `ragged` `243.3` `+6.78%`).
- **Step 2 (`fd6830ec`):** Final C++ oracle + SASS hashes at `b98aee70`: E0 **295/295**, E1/E3/E4 PASS, `spec:g4` **767/767**, `spec:l4` **183/183**, `spec:e4` **102/102**, served `acc-t0`/`acc-t1` **80/80** identical.
- **Step 3 (`007748f7`):** 11 kernel headers moved out of `engine/native/` into `engine/kernels/`; `drafter_pack.cuh` split; `kpack a63fbb93f145` (22 inputs, 0 from `engine/native/`), E0/E0' **295/295**, GPU 2 E1/E3/E4 **7/7 byte-identical**, `ink-inv` exit 0 (`666/666`).
- **Step 4 (`e2b33c5e`):** C++ runtime (`engine/native/`, `ink-rt-api::capi`, `--runtime`/`--rt-lib`, `-10,516` lines) deleted; Rust-only `regen.sh` E0 **295/295** vs `e0-ref.jsonl`; GPU 2 E1/E3/E4 **7/7**, `spec-g4` **767/767**, served `acc-t0`/`acc-t1` **80/80** byte-identical to Step 2 C++ oracle.
- **Step 5 (`85d9c82f`):** V10 byte-reproducible cubins (`build.sh` stages into fixed path with zeroed mtimes); two clean builds produce byte-identical `pack_engine.cubin` (`5257d55b…`), `pack_drafter.cubin` (`214163ae…`), `manifest.json` (`95d34f2c…`), E0 **295/295**.

## DFlash2 + Exact Leviathan Rejection Sampling & Combined M1 Agent Replay (M, 2026-11-24, GPUs 1, 2, 5 `6D` & 7 `67`, 140 W, `bench/raw/2026-11-24-dflash2/`)

- **DFlash2 Leviathan (`sample_leviathan_kernel` + `selector_walk_kernel`, 128-bit exact on `W * 2^22`, 0 spills):** `sdist` 4/4 PASS (`0` zero-prob draws, TV `0.0030–0.0115`, `chi2_z` `[-0.64, +0.85]`); 20-case `sampled` `T=1.0 top-p=0.95` **`3.864 tok/round`** (**+9.9%** over coupled DFlash2 `3.515`, **+32.1%** over DSpark `2.925`); 20-case `greedy` `0/8,000` (`3.9880 tok/round`).
- **M1 Agent Replay (343 reqs / 118,056 out tok, 0 err, 140 W):**
  - **GPU 5 (`6D`, self-fed):** `self-t1-n4`: Config B `257.9` (`p99 5.35s`) | DFlash2 coupled `258.4` (`3.85s`) | **DFlash2 Leviathan `272.3` (+5.6%, `p99 2.87s`)** | DSpark w8 `276.8` (`3.36s`) | **DSpark Ptok3 `280.9` (+8.9%, `4.23s`)**. `self-t0-n4`: Config B `303.1` (`6.61s`) | DSpark Ptok3 **`324.4` (+7.0%, `4.19s`)**. `self-t1-n8`: Config B `287.8` | **DFlash2 Leviathan `313.4` (+8.9%, `7.23s`)** | DSpark w8 `320.7`.
  - **GPU 7 (`67`, recorded):** `rec-t1-n4`: Config B `266.6` (`p99 6.06s`) | DFlash2 coupled `255.2` (`3.61s`) | **DFlash2 Leviathan `270.7` (+1.5%, `p99 2.85s`)** | DSpark w8 `274.8` (`3.03s`) | **DSpark Ptok3 `278.4` (+4.4%, `4.15s`)**. `rec-t0-n4`: Config B `293.8` (`4.98s`) | DSpark Ptok3 **`306.4` (+4.3%, `3.84s`)**. `rec-t1-n8`: Config B `301.2` | **DFlash2 Leviathan `303.4` (+0.7%, `7.77s`, beats both Config B and DSpark `286.9`)**.

## DSpark Full-Vocab `top_p`-Filtered Leviathan + Ptok3 `m6` + Fast cuBLASLt `commit_rows` (M, 2026-11-24, GPUs 0–7, 140 W, `bench/raw/2026-11-24-dspark-leviathan/`)

- **Kernels (`kpack 63a22334630d`, 299 kernels, 0 spills):** `s2f::markov_proj_only`, `s2f::markov_lev_scan`, `s2f::markov_lev_finish` (in-place shared-memory sort + full-vocab `top_p` cutoff via `z_tot` at `T=1.0` + exact `2^22` weights + Philox draw), reusing target `ink::sample_leviathan_kernel` (128-bit exact). Separate `graphs_sampled` leaves greedy `T=0` bit-identical (`0/8,000` bad argmax). Draft overhead `+0.12 ms/step` (`2.83 → 2.95 ms`).
- **Gates (`GPU 2 6D`, `T=1.0 top_p=0.95`):** `sdist` 4/4 PASS (`0` zero-prob draws, TV `0–0.0105`, `chi2_z [-0.71, -0.10]`, `CONC1@80` `5.586 tok/rd`); 20-case `sampled` **`3.1184 tok/rd`** (**+6.0%** over coupled `2.9425` and unfiltered Leviathan `2.9436`, recovering `a=7: 187 → 250` while cutting `a=0: 946 → 816`); fast-commit `greedy` `0/8,000` (`3.1467 tok/rd`).
- **Fast cuBLASLt `commit_rows` (`COMMIT_CHUNK = 1024`, `SK = 1`):** cuts prefill drafter commit **12.8×** (`5.1 s → 0.4 s`, total prefill `123.8 → 119.9 s`), `+1.3 tok/s` (`+0.4%`) on deterministic `T=0` on both GPUs 5 & 7.
- **Same-card M1 replay (`343` reqs / `118,056` tok, `0` err, 140 W):**
  - **GPU 5 (`6D`, `1,082 MHz`, self-fed):** `self-t1-n4`: Config B `257.9` (`p99 5.35s`) | Ptok3 `m5` `280.9` (+8.9%) | Ptok3 `m6` `284.1` (+10.2%) | **DSpark `top_p`-Lev + `m6`: `294.2 tok/s` (+14.1%, `p99 3.43s`, `acc/d 0.530`)**. `self-t0-n4`: Config B `303.1` | **+ `fast_commit`: `326.1` (+7.6%, `p99 3.46s`)**. (`N=8` on GPU 4 `6D`: **`345.1 tok/s`**, +19.9% raw / ~+18% clock-adj vs Config B `287.8`.)
  - **GPU 7 (`67`, `1,121 MHz`, recorded):** `rec-t1-n4`: Config B `266.6` (`p99 6.06s`) | Ptok3 `m5` `278.4` (+4.4%) | Ptok3 `m6` `281.4` (+5.6%) | **DSpark `top_p`-Lev + `m6` + `fast_commit`: `288.6` (+8.3%, `p99 3.08s`)** (`w8` `LEV_FULL=1`: **`289.9`**, +8.7%). `rec-t0-n4`: Config B `293.8` | **+ `fast_commit`: `308.1` (+4.9%, `p99 3.40s`)**. (`N=8` on slowest `67` GPU 0: **`313.5 tok/s`**, +4.1% vs G7 Config B `301.2`.)
- **Defaults landed:** `VERIFY_CALL_ROWS = "1,2,3,4,6,8"` (`m6`), DSpark full-vocab `top_p` Leviathan on by default, fast cuBLASLt `commit_rows` on by default; `VERIFY_TAU = 0.7` and `CHUNK = 2048` (`504` pages) kept as defaults after same-card confirmation showed `tau=0.5`/`c4096` did not beat them on GPUs 5 & 7 (cross-card `+6–10 tok/s` on GPUs 1/4/6 traced to `+1.6–3.0%` higher silicon SM clocks at 140 W).

## DFlash2 Ptok3 Confidence-Gated Verify Width (`tau = 0.25`, M, 2026-11-24, GPUs 1, 2, 4, 5 `6D` & 3, 7 `67`, 140 W, `bench/raw/2026-11-24-dflash2/`)

- **Kernels & Wiring (`kpack 7cfc3e5268d9`, 299 kernels, 0 spills):** `df2::topk16_chunk_max_kernel` + `topk16_stage2_kernel` accumulate full-vocab softcapped `row_z[r]`; `selector_walk_kernel` writes cumulative `conf[s][0..6]` using corrected `z_corr` (in-place shared-memory sort, 0 spills). Added `Drafter::default_verify_tau()` (`0.25` for DFlash2, `0.70` for DSpark) and enabled `--verify-width gated` (`m6`) for `--draft-dflash2`. DSpark path untouched.
- **Gates (`GPU 1 6D`):** `sdist` 4/4 PASS (`0` zero-prob draws, TV `0.0000–0.0085`); 20-case `sampled` `3.8984 tok/rd` (unfiltered top-16 default; `INK_DF2_TOPP=1` opt-in drops to `3.7856`); 20-case `greedy --gate on` `0/8,000` (`largest_gap 0.0000`, `3.9880 tok/rd`).
- **M1 Replay (`343` reqs / `118,056` tok, `0` err, 140 W):** `tau = 0.70` over-truncated parallel DFlash2 (`266.1` G5, `258.3` G7); `tau = 0.25` lifts DFlash2 across cells:
  - **GPU 5 (`6D`, `1,084 MHz`, self-fed):** `self-t1-n4` `272.3 -> 276.6 tok/s` (`+1.6%` vs `w8`, `+7.3%` vs Config B; `289.2` on GPU 4 `1,106 MHz`); `self-t0-n4` `296.7 -> 302.1 tok/s` (`+1.8%`).
  - **GPU 4 (`6D`, `1,079 MHz`, clock-matched to G5):** `self-t1-n8` **`313.4 -> 335.0 tok/s` (`+6.9%` / `+21.6 tok/s`, `+16.4%` vs Config B `287.8`, within `3%` of DSpark `345.1` at `−3.3 GiB` VRAM)**.
  - **GPU 7 (`67`, `1,118 MHz`, recorded):** `rec-t1-n4` `270.7 -> 268.7 tok/s` (`+0.8%` vs Config B); `rec-t0-n4` `277.3 -> 284.2 tok/s` (`+2.5%`). DSpark (`294.2` G5 / `288.6` G7) remains the `N=4` default.

## 6-Track Frontier Stack: Decode MoE + 16-Row Lookup + Glue/Router Fusion + Prefill TC Bias + Opt-In FP8 KV + Fused Drafter (M, 2026-11-24, GPUs 0–7, 140 W, `bench/raw/2026-11-24-frontier-stack/`)

- **Kernels & Footprint (`kpack de0e981af145`, 367 kernels, 0 new spills, `check.sh` PASS):**
  - **Track A (Decode MoE):** `pmoe::moe_gemm_dec<FMT, MODE, 5, 4>` (`BN_DEC=16`, `ldmatrix.x2` for `NT=1`, 5 stages) + 1-byte `NVFP4_N1S1` (`FMT=3`) scales (**−1.27 GB VRAM**, weights `20.25 → 18.98 GB`, eliminates 8 B spill) + 512-thread `moe_build_dec_kernel`.
  - **Track B (16-row decode attn + lookup):** Single-pass `decode_attn_kernel16<KV>` + 6-warp `decode_attn_ring_kernel` + Ptok3-gated verbatim lookup drafts (`--lookup 8 --lookup-min 8` default, `DEFAULT_GRAPH_MEM = 1024 MiB`, `0` refusals).
  - **Track C (Decode router & glue):** Fused smem-tiled `router_decode_kernel<1>` / `route_select9`, `mg_down_dec_kernel`, `mg_tail_tc_dec_kernel`, 288-thread `ar288`/`conv_ar_*`, weight-stationary `rel_logits_decode`, `__match_any_sync` `moe_build` (graph mem `10.5 → 8.4 MB`).
  - **Track D (Prefill):** In-kernel BF16 TC relative bias (`rel_proj_t`, removes `10.3 GB/chunk` `relk` HBM traffic, also applied to `prefill_attn_fp8_kernel`), SMEM `msgs`, faster `moe_build`/`mg_tail_tc`, fused prefill `wqkv`.
  - **Track E (Opt-in `--kv-fp8`, `default: false`):** Global-only E4M3 KV pool (2× capacity: `1,008` pages / `1.03M` tok in `47.9 GiB`; strictly lossy KL `0.02618`, documented in `README.md`; default BF16 untouched).
  - **Track F (Drafter):** Fused DSpark sampled Markov loop (`68 → 58` graph nodes, `draft_ms 3.06 → 2.76 ms`, `−9.9%`) + `DEFAULT_DRAFT_TOPP = 0.90`.
- **Unified Lossless Gates (`0857d0b0`):** `ink-inv` BF16 & `--kv-fp8` **`0` diff bits**; `sdist` **`13/13` PASS** (`425,984` draws, **`0` zero-prob draws**); `spec-greedy --lookup 8` **`0/8,000`** (`max_gap 0.0000`, `3.137 tok/rd`); `spec-sampled --lookup 8` **`20/20` PASS** (`flags_or=0`, `a=15: 37`); **4,072-pos `ink-gold` Config B full-envelope `band` PASS** (**mean KL `0.00477`** `<= 0.00535`, **top-1 `98.04%`**, **max KL `0.4234`**, **greedy `0/20` fail**).
- **Stacked Same-Card M1 Replay (`343` reqs / `118,056` tok, `0` err, 140 W):**
  - **GPU 5 (`6D`, self-fed):** `self-t1-n4`: Config B `257.9` → prior `master` `294.2` → `--lookup 0` `316.9` → **default `--lookup 8` `323.8 tok/s` (`+25.6%` vs Config B, `+10.1%` vs prior `master`, `p99 3.56s`, `0.432 J/tok`)**. `self-t0-n4`: Config B `303.1` → prior `master` `326.1` → **`360.6 tok/s` (`+19.0%` vs Config B, `+10.6%` vs prior `master`, `p99 3.07s`, `0.388 J/tok`)**.
  - **GPU 7 (`67`, recorded):** `rec-t1-n4`: Config B `266.6` → prior `master` `288.6–289.9` → `--lookup 0` `326.9` → **default `--lookup 8` `333.7 tok/s` (`+25.2%` vs Config B, `+15.6%` vs prior `master`, `p99 2.94s`, `0.418 J/tok`)**. `rec-t0-n4`: Config B `293.8` → prior `master` `308.1` → **`347.6 tok/s` (`+18.3%` vs Config B, `+12.8%` vs prior `master`, `p99 2.77s`, `0.402 J/tok`)**. `rec-t1-n8`: **`348.4 tok/s`** (`+15.7%` vs Config B `301.2`).
  - **GPU 4 (`6D`, `N=8`, `--lookup 0`):** `self-t1-n8` **`380.2 tok/s`** (**`+32.1%` vs Config B `287.8`**, `0.368 J/tok`, `p99 7.02s`, `0` cache evictions).
  - **GPU 0 (`67`, opt-in `--kv-fp8`):** `4×8 120K` **`45.00 ms`** (`−16.6%` vs `master` `53.95 ms`); `rec-t1-n8` (`1,008p`) **`327.3 tok/s`** (`+8.7%` vs G7 Config B `301.2`, `+4.4%` vs G0 `master` `313.5`).
  - **Synergy:** Tracks A + C lowered 10–16-row verify step cost enough that Track B's `--lookup 8` switched from standalone break-even at `N=4 T=1.0` to **`+2.18%` on GPU 5 and `+2.08%` on GPU 7**.
- **Stacked Microbenchmarks (GPU 2 `6D`, `--fine 1`):** `4×8 45K` `42.31 → 40.07 ms` (`−5.3%`; eager `47.58 → 44.14 ms`, `−7.2%`), `4×1 45K` `23.83 → 20.53 ms` (`−13.8%`), `1×8 45K` `~22.7 → 18.89 ms` (`−16.8%`), `4×16 45K` `54.02 ms` (`1,172.5 rows/s`), `cold 32K` `5.49 → 5.316 s` (`6,164 tok/s` with `--fine 1`; `4.959 s` / `6,608 tok/s` no-fine on G6).

## `acquire_slot` Fix & MiaAI-Lab Recipe Table vs Config B (M, 2026-11-25, GPUs 1, 3, 4, 5, 6, 7, 140 W, `bench/raw/2026-11-25-steal-slot-and-recipe-table/`, `eaa20598`)

- **`ink-sched` `acquire_slot` fix (`slot_steal_grace_ms = 150.0`):** stale parked slots (`age >= 150 ms`) stolen first by LRU; within grace, smallest `num_computed` stolen first so returning multi-turn 70K–80K prefixes survive `N == slots`.
  - **GPU 5 (`6D`) `self-t1-n8` (`--slots 8, --lookup 8`):** `345.1 → 358.9 tok/s` (**`+4.0%`**, **`+24.7%` vs Config B**), TTFT p99 `12.08 → 7.65 s` (**`−36.7%`**), **`0` zero-cache evictions** (`cached 0.960`, `0.388 J/tok`).
  - **GPU 7 (`67`) `rec-t1-n8` (`--slots 8, --lookup 8`):** `348.4 → 362.3 tok/s` (**`+4.0%`**, **`+20.3%` vs Config B**), TTFT p99 `10.28 → 7.44 s` (**`−27.6%`**), `cached 0.957`, `0.384 J/tok`.
- **MiaAI-Lab protocol (`T=0`, 400-tok cap, 1 card, 140 W) vs Config B:** beats Config B on **all 12/12 cells** on both `6D` and `67`: 1-user structured/code/prose `357.0 / 225.7 / 146.5` vs `281.5 / 179.4 / 110.8` (`+25.8% to +32.2%`); 4-user `728.2 / 435.1 / 341.1` vs `653.3 / 363.2 / 265.0` (`+11.5% to +28.7%`); 8-user `1,003.8 / 582.2 / 485.1` (`6D`) & `1,026.9 / 604.3 / 497.9` (`67`).
- **Cold prefill (`4K–120K`, `--prof 0`) vs Config B (`6D`):** `7,665` vs `5,745` at 4K (`+33.4%`), `6,649` vs `4,880` at 32K (`+36.3%`), `5,981` vs `4,187` at 64K (`+42.8%`), `5,052` vs `3,307` at 120K (`+52.8%`).

## Open, ranked by (gap × likelihood)

1. **Remaining ~3–6% to hard lossless 140 W ceiling:** `moe_gemm_dec` (`19.2 ms` of `40.1 ms` `4×8`), global `decode_attn` (`6.7 ms`), and dense `proj + wo` (`4.8 ms`) are already at `~91–97%` of the practical lossless `sm_80` ceiling at `N=4`. Further leaps require lossy weight/activation compression or stronger drafters.

## 2026-11-25 — Stock 250 W Profile (`6D` GPU 5 & `67` GPU 7, `[M]`)

- **Setup:** Temporary runtime `-pl 250` on GPU 5 (`6D`, 1,728 MHz HBM) and GPU 7 (`67`, 1,458 MHz HBM) under `default` BMC fans (`0` thermal-guard trips; peak core `71 °C`, HBM `73 °C`), running `master @ eaa20598` (`v1.0.0`, lossless BF16 KV + DSpark `top_p=0.90` + `--lookup 8 --lookup-min 8`). Restored to `140 W` + `cmp-quiet3` immediately after.
- **MiaAI-Lab Decode (`T=0`, 400-tok cap, `6D` / `67`):** 1u `435.1 / 393.4` (structured), `272.2 / 247.3` (code), `170.1 / 154.6` (prose); 4u total `921.8 / 869.6`, `560.7 / 523.4`, `427.2 / 401.8`; 8u total **`1,323.2 / 1,276.1`**, `779.9 / 746.5`, `650.5 / 620.9` (`+32–34%` over 140 W on `6D`, `~10.4K tok/s` across 8 cards).
- **Cold Prefill (`4K–120K`, `6D` / `67`):** `9,455 / 9,308` at 4K (`0.43 s`), `8,360 / 8,304` at 32K (`3.92 s`), `7,512 / 7,468` at 64K (`8.72 s`), **`6,378 / 6,336 tok/s`** at 120K (`19.27 s`, `+26.2%` over 140 W).
- **M1 Multi-Turn Agent Replay (`343` reqs, `0` err):** `6D` `self-t1-n4` **`440.2 tok/s`** (`+35.9%` vs 140 W, p99 `2.32 s`, `0.563 J/t`), `self-t0-n4` **`493.0 tok/s`** (`+36.7%`), `self-t1-n8` **`483.7 tok/s`** (`+34.8%`); `67` `rec-t1-n4` **`413.7 tok/s`** (`+24.0%`), `rec-t0-n4` **`435.8 tok/s`** (`+25.4%`), `rec-t1-n8` **`444.4 tok/s`** (`+22.7%`).

## 2026-11-26 — 4-Track Lossless 140 W Push & Stacked Tracks 3 + 4 (`f6164f51`, `kpack 9af3241f6232`, `[M]`)

Full record: `bench/raw/2026-11-26-stacked-t34/README.md` (with per-track records in `2026-11-26-{tree-spec,prefill-opt,drafter-async}/`).

- **Rejected / Killed (do not rerun without a new drafter):**
  - **R13 — Dynamic Speculative Tree Verification (`2026-11-26-tree-spec`):** Exact closed-loop GPU sim (`>150k` live draws, `0` zero-prob) proved trees lose `−0.15%` to `−13.1%` vs `L7-full+lkp8` (`~−2.3%` vs `Ptok3 m6`) due to the `R=8 → 9+` attention cliff (`+4.45 ms`) and single-child `top_p=0.90` Leviathan already capturing ~90% of DSpark's `T=1.0` acceptance.
  - **R14 — Anchor-Triggered Global Lookup Override on DSpark (`feat/wt-hybrid-lookup @ bd8c12c0`):** Same-card `B1/N/B2` on GPU 2 (`6D`): DSpark `+0.37%` (`t0-n4`), `−0.37%` (`t1-n4`), `−2.6%` (`t1-n8`). Helped DFlash2 (`+2.2–2.3%`), but DFlash2 still trails DSpark by `−4–5.5%` on 45K M1.
  - **R15 — Cross-Session Prefix Sharing on M1 (`2026-11-26-prefill-opt` 3B):** Only `16,384` reusable tokens after `1,024`-token SWA recompute (`~1.0%` M1, below threshold).
- **Merged & Verified (`f6164f51`, `dirty=0`, `kpack 9af3241f6232`, `386` launchers, `0` new spills):**
  - **Track 3 (Prefill):** Virtual messages (`vmsg_weights_kernel`, `msgs 7.50×`, eliminates ~87 GB HBM traffic/32K), fused multi-query AttnRes (`−25.2%`), fast `__match_any_sync` prefill `moe_build`, fixed silent `--prefill-chunk` clamping + default `chunk=3072` (recalibrated prefill cost model).
  - **Track 4 (Decode/Runtime):** Single-pass `__match_any_sync` decode `moe_build` (`−41.9%` `4×8`, `−50.1%` `4×16`), cached `CUgraphExec` retargeting (`verify_us >= 200 µs`: `36.0% → 3.5%`, p90 `−70%`), coalesced 1-sync `verify_widths_and_drafts` + lazy lookup H2D + vectorized `softcap` (early-exit Markov loop opt-in via `INK_DSPARK_EARLY_TAU=0.70`).
- **Unified Lossless Gates (`GPU 1 6D`, 140 W):** `ink-inv` BF16 & `--kv-fp8` **`0` diff bits** (`0` fallbacks, `0` reply-reuse fails); `ink-spec` greedy `--lookup 8` **`0/8,000`** (`3.137 tok/rd`); sampled `--lookup 8` **`20/20`** (`8,000/8,000`, `3.003 tok/rd`); `sdist` **`4/4` PASS (`65,536` draws, `0` zero-prob)**.
- **Stacked Microbenchmarks (`GPU 1 6D`, `B1/B2 @ eaa20598 → N @ f6164f51`, 140 W):** Cold 32K `6,447 → 6,965 tok/s` (**`7,008.6 tok/s`** rep 0, **`+8.0%`**); Warm +6000 `5,665 → 6,075` (**`+7.2%`**); Warm +1600 `5,508 → 5,674` (**`+3.0%`**); Cold 120K **`5,248.3 tok/s`** (`23.41 s`); Decode `1×8` `18.73 → 18.56 ms`, `4×16` `52.46 → 52.15 ms`, `4×8` `39.01 → 38.99 ms`.
- **Canonical Same-Card `B1 → N → B2` M1 Replay (`140 W`, `343` reqs, `0` err):**
  - **GPU 5 (`6D`, self-fed):** `self-t0-n4` `366.6 → 373.3 tok/s` (**`+1.83%`**, **`+23.2%`** vs Config B, p99 `2.64 s`, `0.374 J/t`); `self-t1-n4` `321.8 → 328.0` (**`+1.94%`**, **`+27.2%`** vs Config B, p99 `3.95 → 3.34 s` `−15.3%`, `0.426 J/t`); `self-t1-n8` `360.9` (p99 `7.22 → 5.51 s` **`−23.7%`**, **`+25.4%`** vs Config B).
  - **GPU 7 (`67`, recorded):** `rec-t0-n4` `348.1 → 355.4 tok/s` (**`+2.11%`**, **`+21.0%`** vs Config B, p99 `3.15 → 2.50 s` `−20.5%`, `0.393 J/t`); `rec-t1-n4` `325.4 → 338.1` (**`+3.90%`**, **`+26.8%`** vs Config B, p99 `3.89 → 2.91 s` `−25.1%`, `0.413 J/t`); `rec-t1-n8` `352.5 → 366.9` (**`+4.10%`**, **`+21.8%`** vs Config B, p99 `7.98 → 5.84 s` **`−26.8%`**, `0.378 J/t`).
- **Conclusion:** Stock-weights lossless 140 W engine is now empirically at the practical ceiling (~96–98%). Next >10% lossless 140 W leap requires Track 5 (custom distilled/trained lossless drafter, e.g., `DFlash2-Global`).
