# engine/kernels/drafter: batched sm_80 DSpark drafter (S2f)

Kernels: `drafter_kernels.cuh` (header-only, namespace `s2f`). Reference orchestration and standalone harness:
`harness.cu` (struct `Drafter`, modes `time`, `kbench`, `eqhead`, `replay`). Build: `build.sh` in
`ink-ninfer-toolchain:cu1331` (`-gencode arch=compute_80,code=sm_80`, no spills). Measurements, ledger and the exact
measured build: `bench/raw/2026-11-23-S2f-drafter/` (README, LEDGER, `round2/`). Our own code; numerics follow config
B's served drafter (vLLM `models/inkling2/nvidia/dspark.py`, `nvidia/ops/dspark.py`, `models/inkling2/speculator.py`
@ 6e9d1cce, Apache-2.0) at its rounding points.

## Engine contract (for `engine/native/`)

### Model constants

H 2304, 24 query / 8 KV heads x 128, 3 layers (0 and 2 local: last 1,024 + 8 keys; 1 global: all keys), MLP 9,216,
block 8 rows (anchor + 7 masks), 7 drafts, Markov rank 256, padded vocabulary 201,024 rows (argmax over the first
200,058), logit softcap 30, RoPE theta 1e6, eps 1e-6. Batch n <= 8 sessions (GEMM M <= 64 rows).

### Per-session state the engine owns

| Item | Layout | Notes |
|---|---|---|
| Draft K/V, 3 layers | per layer two arrays K and V `[blocks][8 KV heads][64 tokens][128]` bf16 (`CacheL{k, v}`); optional E4M3 copy of layer 1 (`k8, v8`, same index, 1 byte) | paged, 64-token pages; session r's page table row `block_table[r * max_blocks + pos / 64]` (`Step`) |
| Anchor | `anchor[n]` int32 token id, `P[n]` int32 anchor position | **per session** (ragged positions are supported by every kernel: rows read `st.P[r]`) |
| f_rnn | `f_rnn[n][H]` bf16 | = mix of the committed features of the row at `P[r] - 1` (`k_frnn`, alpha row 3, `markov_head.dflare_norm`) |

Nothing else persists between rounds: the Markov recurrence (state, latent, keys) restarts every round.

### Inputs per round

1. **Commit** (`Drafter::commit_rows`): every target row the step verified (and every prefill row), as committed
   features `feats[R][4][H]` bf16 = the partial block entering target visits 5, 23, 41 and the final normed hidden
   (config B's `aux_hidden_states` order), with `pos[R]` and `sess[R]`. Kernels: `k_commit_mix` (grid R x 3) ->
   3 x `gemm_frag` (kv_ctx, split-K 9) -> `epi_kvctx(..., row_pos, row_sess)` (k_norm_ctx + RoPE, writes K/V at
   `pos`). Chunks of <= 64 rows.
2. **f_rnn**: `k_frnn` on the feature row at `P[r] - 1` of each session.
3. **Draft step** (`Drafter::layers` + `Drafter::head`): `k_embed_block`, then per layer qkv GEMM -> `epi_qkv`
   (q/k norm + RoPE, writes the block's own K/V at P..P+7) -> `attn_block` (+ `attn_combine`) -> o_proj GEMM ->
   `epi_addnorm` -> gate_up GEMM -> `epi_silu` -> down GEMM -> `epi_addnorm` (the last one emits the head inputs
   `hout[n*7][H]`, f16 copy, `x7 = [h, f_rnn]`, ||h||), then the head: vocabulary GEMM (int8 + bound by default) ->
   rnn context GEMM + `epi_ctx` -> `markov_seed` -> 7 x (`markov_recur`, `markov_zref`, `markov_argmax3`) ->
   `markov_finish`.
4. **Round batch** (the engine's DraftRT, `engine/native/src/draft_sel.cuh`, package G2): the draft step runs for the
   k sessions that speculate in the round only: `draft_gather` copies their anchor, P, page-table row and f_rnn into
   compact rows 0..k-1, step 3 runs on those k rows (T = 8 k, one CUDA graph per k), `draft_scatter` writes their
   drafts back. Every kernel of step 3 computes a session's rows from that session's inputs only (GEMMs per element,
   attention per session with the drafter's split count, which depends on the drafter's n, not on k), so a session's
   drafts are bit-identical whichever sessions draft with it.

### Outputs

`drafts[n][7]` int32 token ids (`-1` past `ksteps`). Drafts equal config B's greedy drafts up to fp32 summation order
(17,646 / 17,647 on captured head inputs; 98.75 % of tokens on the real-K/V replay, the same as config B's own ops
on that replay (98.78 %); acceptance x 0.9996; see `bench/raw/2026-11-23-S2f-drafter/round2/`); the argmax is exact over the full vocabulary (pruning bounds are rigorous).

### Commit / rollback rules

* Commit only real rows: the accepted rows of a verify step (anchor + accepted drafts) and prefill rows. Config B's
  own buffer pads rejected rows with position 0; committing those corrupts position 0 (found in the replay, ledger 45).
* K/V at a position is simply overwritten by the next commit of that position. Rejected rows need **no rollback**:
  a draft for anchor P reads context positions < P (all accepted, last write wins) and its own block rows P..P+7,
  which `epi_qkv` writes fresh every round; stale rows at >= P are never read before being rewritten.
* Rolling back a session to an earlier position P' only requires setting `P = P'` and recomputing `f_rnn` from the
  committed row at P' - 1 (the engine must keep that row's features, or the f_rnn itself, per checkpoint).
* Prefix reuse: committed K/V depends only on (features, position); pages can be shared copy-on-write.

### Workspaces (per `Drafter`, n sessions)

split-K partials 32 MB (`part`), commit mix `3 x 64 x H`, attention split partials `n x 8 x splits x 32 x 128` fp32,
head: `B` (bf16) or `Bq` (fp32) `[n*7][201,024]`, Markov keys/state/projection (KB). Weights: fragment-layout copies
of every drafter matrix and the head (bf16, int8), config B's fp64 fold (`fold.bin`), rank-16 fold basis
(`lowrank16.bin`), int8 fold rows and exact error norms, all built at load (`load_weights`).

### Options (harness flags; defaults = the measured build)

`--variants` exact (bf16 head) | int8 (default best) | immaO (int8 x int8 head, 32 exact outlier channels) |
sl49k / sl49k8 (first 49,152 rows: acceptance x 0.32, do not use) | *kv8 (E4M3 global-layer draft K/V: acceptance
x 1.003 vs bf16 K/V on real replay); `--overlap 1` (commit and rnn context on a second graph branch, default on);
`--ksteps k` (draft only k tokens, exact prefix). Off by default and measured slower: `--coop 1`, `--gemm3 1|4`,
`--head-rt2 1`, `--argmax2 0|1`.
