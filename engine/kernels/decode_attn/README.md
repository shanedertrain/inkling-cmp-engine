# engine/kernels/decode_attn: batched paged decode / verify attention (S2b)

Inkling-2-Mini decode and speculative-verify attention for the CMP 170HX (sm_80), batch-invariant, lossless (bf16 KV)
with a lossy FP8 (E4M3) KV switch. Measurements and the variant ledger: `bench/raw/2026-11-23-S2b-decode-attn/`.

## Files

| File | What |
|---|---|
| `decode_attn.cuh` | Kernels + host launch helpers. **Design A** (`decode_attn_kernel`, `launch`): 4 warps per CTA, each warp owns whole fixed chunks of one (session, KV head) for all 24 columns, partial straight to global through a coalesced smem stage; columns in up to three n8 tiles (0-7, 8-15, 16-23; K as the A operand, K/V reused from the same registers by every tile, one instruction sequence per tile, so a column's bits do not depend on its index; until F1 columns 8-23 sat in an m16 tile with another summation order, `bench/raw/2026-11-23-f1-verify-rows/`); lazy rescale (tau 0); K/V register prefetch across chunks; Q staged in smem for R >= 3. **Design B** (`decode_attn_ring_kernel`, `ring_launch`; SWA): persistent workers, shared S-stage cp.async ring (the name refers to that ring, not to the KV layout), one warp per 8 columns (n8 tiles, ldmatrix). Both read paged KV and mask out-of-range keys at load. `combine_kernel`: fixed-order fold of chunk partials + config B's gate-then-sink epilogue, one bf16 rounding |
| `bench/bench.cu` | FP64 oracle (synthetic and real KV), batch-invariance test, sustained timing (P6 harness discipline) |
| `bench/common.hpp` | Copy of the P6 harness (our code) |
| `build.sh` | sm_80 build, registers/spills, FP8-convert count |
| `scripts/` | runners for GPU 6 / GPU 7 (`run6.sh` / `run7.sh`, `run_in.sh`, `cfgb6.sh` / `cfgb7.sh`, `final.sh`, `realgrid7.sh`, reviewer harness `rev7.sh`), real-KV hash compare (`cmphash.py`), config B per-kind timing (`cfgb_attn_kinds.py`), summariser (`summ.py`), grid table (`analyse_grid.py`) |

## KV contract (for the runtime author; S2b-D2, 2026-11-23)

There is **one addressing mode** for both kinds: paged, through a per-launch page table. Ring mode
(`ring_cap`, `ring_stride`) was removed; an SWA ring is expressed as page-table entries.

| Item | Contract |
|---|---|
| Address of key `k`, KV head `g`, session `s` (s = index of the session in this launch, 0..n-1) | `kv + page_table[s * pt_stride + (k >> page_shift)] * page_stride + g * head_stride + (k & (2^page_shift - 1)) * row_stride` |
| Page layout | head-major `[8 heads][2^page_shift keys][128]` (bf16: 256 B rows; E4M3: 128 B rows): `row_stride = 128 * elem`, `head_stride = 2^page_shift * row_stride`, `page_stride = 8 * head_stride`; page_shift >= 4 (a 16-key step never straddles a page); design B **requires** this layout (`ring_launch` returns `cudaErrorInvalidValue` otherwise); design A also accepts row-interleaved pages |
| Global layers | row `s` of the page table = the session's pages, logical page `j` -> physical page |
| SWA layers with persistent per-slot rings | each slot owns P physical pages (native: 3 of 1024 keys); the session's page-table row maps logical page `j` to `slot * P + j % P` (exactly what `engine/native` already writes for `d_pt_r_`). Requirement: the keys a call can touch, `max(0, pos0 - 1023) & ~15 .. pos0 + R - 1`, span at most P pages without aliasing: (P - 1) * 2^page_shift >= window + R_max + 15 - 1 (P = 3, page 1024: fine for R <= 1,033). The kernel never wraps on its own; wrap = the `j % P` in the table |
| Which session a row belongs to | `row0[s]`, `ctx[s]`, `nrows[s]`, `chunk_shift[s]`, page-table row `s`: all indexed by the launch index; nothing in the kernel depends on slot ids, so sessions may change slots or launch positions between calls |
| Unwritten memory | no longer a precondition (review D3): keys of a touched 16-key step outside `[k_lo, qlast]` are masked at load (design A: zeroed V registers in boundary steps, in a separate EDGE instance of the step body so the hot loop carries no selects; design B: cp.async zero-fill), so NaN / Inf there cannot reach a row (reviewer strict-poison test: 0 non-finite). Cost vs no masking, n 4 45K: global +0.9 %, step 0; LEDGER 12a. Keys inside `[k_lo, qlast]` must be written (they are visible) |
| Chunk sizes (bits depend on them) | SWA 128 keys (`chunk_shift` 7) with design B; global 1024 keys below 96K context, 4096 at or above (design A). Fixed per kind and context bucket of the session's own `ctx` |
| Grid and buffers larger than the call | allowed: grid x (workers) may exceed the chunks (empty workers exit), `max_chunks` may exceed every session's chunk count, `R` (partial stride, combine grid) may exceed every `nrows[s]`; partials are sized `n * 8 * max_chunks * 3R * 128` fp32. A graph captured with bounds (n, max_chunks, R) serves any call within them |
| Q staging | internal: R >= 3 selects the Q-in-smem kernel (review D5; no host switch) |

Native runtime change needed (not made here: `engine/native` has another agent's uncommitted work): in
`Engine::attention`, delete the `p.ring_stride`, `p.ring_cap`, `p.counters` assignments and the two `da::g_qs`
lines; for SWA (`!full`) call `da::ring_launch(p, da::KV_BF16, n, W, 3, st_)` (after `da::ring_prepare(KV_BF16, R, 3)`
once) with `chunk_shift` 7 for SWA sessions and W from the same worker search over `74 * ring_ctas_per_sm(...)` slots;
the page table (`d_pt_r_`) and pools stay as they are.

### Rows

Session `s` owns rows `row0[s] .. row0[s] + R_s - 1` at positions `ctx[s] + t`; q `[T][24][128]` bf16 (clipped,
tau-scaled), rel `[T][24][1024]` bf16 by distance, gate `[T][3072]`, sink `[T][24]` bf16. Bits of a row do not depend
on the batch, the grid, other sessions' widths or data (tested), nor on the row's index t inside its verify block:
verify row t equals the one-row step at `ctx + t` bit for bit (design B always; design A since F1 replaced the m16
tile of columns 8-23 by two n8 tiles; `ink-inv` test `verify_rows_vs_decode`).

## Recommended configuration (S2b, 2026-11-23)

| Kind | Kernel | Chunk | Notes |
|---|---|---|---|
| global, R >= 3 | design A (Q staged in smem automatically) | 1024 keys below 96K context, 4096 above (LEDGER C1, t17) | workers per (session, head) from `pick_workers_warp` (physical choice, bits unchanged) |
| global, R <= 2 | design A (registers) | same | |
| SWA | design B, S = 3, paged ring pages | 128 keys | `ring_workers` |
| lossy | same kernels with E4M3 KV on the global layers (placement decode, never `cvt`), SWA stays bf16 | | |

Bits depend on (kind, KV dtype, chunk size, R-independent column layout) only. Switching A <-> B is not bit-safe
(different column layout), so each kind uses one design for every batch size (engine: A global, B SWA).

## Removed in S2b-D2 (2026-11-23; engineering rule 1: no dead paths)

Ring mode (`ring_cap`, `ring_stride`: replaced by page-table rings), the twin mode, the SWA "S" variant, the fused
combine (`counters`), `g_qs`, and the measurement macros (`DA_TAU`, `DA_STAGE`, `DA_PFD`, `DA_BPF`, `DA_NOREGPF`,
`DA_NOCOMBINE`, `DA_TWQS`, `DA_TWMIN`, `DA_ABL_*`, load-hint variants). Their measurements stay in the LEDGER
(sections 4-10); the code is in git history (commit e423228 and before). Kept and now fixed: tau 0, staged partial
stores, `L1::no_allocate.L2::256B` K/V loads in design A.

## Ragged rows

`Params::nrows` (optional per-session widths 1..R; R = max sets the partial column stride). Bits independent of
other sessions' widths (tested). Design A runs only the n8 tiles that hold columns of the session. A mixed step costs its
rows' sum + 4 % (LEDGER R2).

Bench `--kvdump <dir>`: timing on the real-KV dump tiled to length (both harnesses: `scripts/cfgb_attn_kinds.py
--kvdump`, `scripts/realgrid7.sh`).
