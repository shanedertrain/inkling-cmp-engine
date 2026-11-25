# ink-runtime PORT-MAP: C++ engine → Rust engine (packages R5, R6)

Source ported: `engine/native/src/engine.cu` / `engine.hpp` @ `1c7ac46` (tag `rust-r5-base`), plus the deltas of the
section "Deltas ported since rust-r5-base" (C++ engine as of delta 3, `4c6b54b`). The next
delta is ported by walking this table: each row names the C++ function or launch site and its Rust counterpart.
Kernels are not ported (kernel pack, `ink-kernels`); their launch geometry is (`ops.rs`).

## Deltas ported since rust-r5-base

| C++ commit | Change | Rust (delta 1, `bench/raw/2026-11-23-rust-delta1/`) |
|---|---|---|
| `d4d5889d` (F1) | decode_attn design A: verify columns in three n8 tiles | kernel only (pack, regenerated at the F1 merge `b62831e5`) |
| `1e161b38` D1 | decode-row input projections: cuBLASLt over one contiguous weight `[q, gate, r, sink, k (, k_dyn), v (, v_dyn)]` (one GEMM when the inputs are one buffer, else the q side plus `LtGemm::run2`, a strided batch of 2 for k / v), then `proj_split_kernel`; `proj_small.cuh` deleted; four decode-class pin entries N = 8600, 6552, 1056, 1024 on the N = 8600 algorithm (`LtGemm::adopt`, overridden by the pin file); `qkvf_` take after `xm`; loader order `wq, wgate, wr, wsink, wk, (wk_dyn,) wv, (wv_dyn,) wo` with a contiguity check | `forward.rs: Fwd::visit` (projections), `ops.rs: Dev::proj_split`, `ink-cuda gemm.rs: LtGemm::run2` (+ batch layouts, `cublasLtMatrixLayoutSetAttribute` in the ABI table), `mod.rs: planned_shapes` (25 shapes), `Engine::new` (`xkv` view: xk and xv adjacent, checked), `ink-model weights.rs: Loader::layer` (order, `LayerW::wqkv` view, `contiguous`), `plan.rs` (`qkv fused`, `FUSED_ROWS`, `FUSED_COLS`) |
| `1e161b38` C1 | glue row loops unrolled (`GLUE_FOR_H`) | kernel only |
| `1e161b38` C3 | `attn_res_kernel<Q, G>` also writes the G loop-gate inputs (`GateIn`); `message_gates(..., gin_ready)` then skips `mg_in`; G forced 0 on the reference path | `ops.rs: Dev::attn_res` (Q, G dispatch, `GateInputs`), `forward.rs: Fwd::attn_res`, `Fwd::message_gates` and their four call sites in `visit` |
| `25f1fcc9` X1 | decode experts on ONE fixed served Humming kernel per projection and format | ported (`403f6920`: `engine/humming.rs`, `ink-model humming.rs`, `Dev::hum_align` / `hum_gemm` / `moe_swiglu`, `Fwd::hum_rows`), gated bit-identical (E1-E4, records `x1-default/`), then **deleted with the C++ path**: X1 parked (head, 2026-11-23, delta 1): a second resident copy of the experts in Humming format (+15.7 GB) breaks the 4 x 262,144 KV budget of the 64 GB card; X1 needs a single weight copy (the fixed Humming kernel reading our repacked expert format, or prefill reading Humming's) before it can return. Code: git history `25f1fcc`, `403f692`. |
| delta 1 (this package) | one expert path: prefill_moe for every row (`moe_mode` deleted: mode 0 was the same code as mode 1, mode 2 = X1 parked); `hum_cache` / `hum_weights` deleted; `gate_tail` and its kernels deleted; graph key `glue|...` without the moe field; C ABI 3 | `mod.rs` (no `moe` field, `set_moe_mode` gone from the `Runtime` trait), `meta.rs: graph_key(glue, call)`, `ink-rt-api` (no `Humming`, `GateTail`, `MoeMode`; `RawConfig` 64 bytes, ABI 3), tools (`--moe`, `--gate-tail`, `--hum-cache`, `--hum-weights` gone) |
| `ca75e72d`..`ea2fe2ef` (delta 2, fixed step cost) | P1 row kernels register-staged (`attn_res_kernel`, `attn_res_norm_kernel`, `rmsnorm_kernel`, `rmsnorm_g_kernel`, `mg_in_kernel`; `__launch_bounds__(256)`), P2b `moe_build_kernel` warp scans (+512 B static smem), P4 `block_sum` reorder: kernel only; **P3**: the message-gate tail of every row count on `pglue::mg_tail_rows_kernel<<<(H / 128, ceil(T / 32), G), 128>>>` (was `glue::mg_fused_kernel` at T <= 64), `mg_fused_kernel` deleted | `ops.rs: Dev::mg_tail` (one launch for every T); pack regenerated (`bench/raw/2026-11-23-rust-delta2/`) |
| `bce2e87`..`01e2436` (delta 3, prefill glue second look, `bench/raw/2026-11-23-prefill-glue2/`), merged at `f1fed2e` | five bit-identical prefill glue rewrites, every one selected by `T > 64` on the production path: AttnRes (attention and MLP inputs with gate inputs, loop messages, head) on `pglue::ar_rows_kernel<Q, G>` (warp per row, `Q == 4` also in the 256-thread fallback chain), plain-visit AttnRes + RMSNorm on `ar_norm_rows_kernel<1 / 3>`, `norm3` and the message visits' `mlp_norm` on `norm_rows_kernel` (G = 3 / 1), attention and MLP conv + residual on `conv_rows_kernel<0 / 1>` (no `y_` store); `mg_tail_rows_kernel` (two rows per step) and `qkv_post_rows_kernel` changed in place: kernel only | `ops.rs: Dev::ar_rows`, `ar_norm_rows`, `norm_rows`, `conv_rows` (`ConvRows::Attn / Mlp`), `gate_in` (the GateIn checks shared with `attn_res`); `forward.rs`: `PGLUE_ROWS`, `Fwd::attn_res`, `norm3`, `visit` (attention inputs, attention conv, MLP input, `mlp_norm`, MLP conv), `run_model` (loop messages); pack regenerated (`bench/raw/2026-11-24-rust-delta3/`, 13 new keys incl. the two G7 kernels, struct `pglue::MgTc`) |
| `df0c813` G7 (delta 3, bit-changing, accepted by the head; re-applied as `4c6b54b`) | the message-gate tail of prefill row groups at `T > 64` on bf16 tensor cores, `pglue::mg_tail_tc_kernel<<<(H / 128, ceil(n / 64), G), 128>>>(MgTc, n, ldh, ldm, ldo)` per row group; decode row groups of a mixed call and every call of `T <= 64` keep `mg_tail_rows_kernel`; weights `tc_w_` bf16 [2][H][64] per loop layer and gate, taken right after `Engine::alloc` ("gate tail tc W", 48 takes), filled by `mg_tc_weights_kernel<<<2 H, 64>>>`; arena term `NLAYERS x 4 x 2 x H x 64 x 2 + 1 MiB` | `ink-model plan.rs: GateTailW` (+ `MemConfig::arena_bytes` term), `mod.rs: Engine::new` (take after `Activations::alloc`, `Dev::mg_tc_weights` per gate), `ops.rs: Dev::mg_tail_tc`, `mg_tc_weights`; `forward.rs: Fwd::message_gates` (row groups); `ink-load` takes the same weights (arena map) |
| `cfe2a2a5`, `412e14b9` (G1 package: C + G1; the batch of A, `b7a498a4`, never ran under G1 and is removed) | **C**: decode-only calls launch the 3- and 4-query AttnRes as one block per (row, query) (`attn_res_kernel<1>` grid (T, Q), `<1, 1>` grid (T, 3) with qsel, `attn_res_norm_kernel<1>` grid (T, 3); kernels take the query offset `blockIdx.y * Q`). **G1**: the decode gate-down shape (n 64, k 4608) pinned on split-K 32 with REDUCTION_SCHEME 2 (the pin file carries it; the C++ `LtGemm::split_k` sets it in write mode), one GEMM per gate as before (cuBLASLt refuses every strided batch of 3 with the split-K algorithm: `bench/raw/2026-11-23-g1-gate-proj/land/gd-check/`), split-K plans on a 16 MB cuBLASLt workspace (`kLtWorkspace`, arena take `lt workspace` after `gd`); pin format: split-K only with reduction 2 or 4 | `ink-cuda gemm.rs`: `set_workspace`, workspace-aware AlgoCheck in `run` and `run2` (`run2` on a private `run_batched`, the C++ `runb`); `pins.rs: AlgoPin::deterministic` + tests; `ink-model plan.rs` (`LT_WORKSPACE`, `lt_ws`, `arena_bytes`); `mod.rs: Engine::new` (`set_workspace`); `forward.rs: Fwd::decode_only` and the AttnRes sites; `ops.rs: Dev::attn_res` / `attn_res_norm` (`qsplit`); pack regenerated (`bench/raw/2026-11-23-g1-gate-proj/land/regen/`) |
| `engine.cu: verify_widths` (`verify_width_kernel`, one host sync); `spec_accept` over ragged verify segments (1..k + 1 rows) | `specrt.rs: verify_widths`, `ops.rs: Dev::verify_widths`; `spec_accept`'s check | gated verify width (ptok) |
| not ported (by brief) | `--mode forced` reference tokens in `spec_accept` | |


## Host logic

| C++ (file: function / site) | Rust | Notes |
|---|---|---|
| `engine.cu: pick_workers`, `pick_workers_warp`, `chunk_shift_for` | `engine/meta.rs: pick_workers`, `pick_workers_warp`, `chunk_shift_for` | f64 arithmetic as the C++ |
| `engine.cu: Engine::decode_gx`, `attention` (decode R / gx bound) | `meta.rs: decode_gx`; `forward.rs: Fwd::attention` | |
| `engine.cu: Engine::forward` (segments, ids/pos/seg/row0/slot/len/lrows, pages, row groups, page tables call + decode order, decode metadata, `pfa::Seg`, prefill items with key splits and stable sorts) | `meta.rs: build_call`, `pa_items`; `mod.rs: Engine::forward`, `Engine::upload` | validation before any page is taken (C++ takes pages, then may fail) |
| `engine.cu: Engine::forward` graph key string | `meta.rs: graph_key` | glue (the moe field went with the default switch, delta 1), decode key chunks (global / sliding), rows and logit rows per segment (R6) |
| `engine.cu: Engine::forward` `graphable`, capture at first use (`cudaStreamBeginCapture` thread-local, `run_model`, end, instantiate), `cudaGraphLaunch` | `mod.rs: Runtime::forward` (`graphable`: graphs on, no tap, no profile, no armed eager trace, decode segments only); `graphs.rs: GraphCache` | R6; the per-call metadata is uploaded before capture or replay |
| `engine.cu: Engine::forward` region walk (stack of roots, last mark wins); `Engine::mark` while capturing | `ink-cuda/src/graph.rs: assign_regions`, `Capture::mark`; `forward.rs: Fwd::mark` | R6 |
| `engine.cu: Engine::apply_ablation`, `set_ablation`, `graph_region_counts`; `capi: ink_rt_set_ablation` (refuses a region no captured graph has) | `graphs.rs: apply`, `GraphCache::set_ablation`, `region_counts`; `mod.rs: Runtime::set_ablation`, `Engine::graph_region_counts` | R6 |
| `engine.hpp: set_graphs`, `graph_captures` | `mod.rs: Runtime::set_graphs` (cache kept), `Runtime::stats` | R6 |
| `engine.hpp: Slot`, `free_pages_`; `engine.cu: Engine::reset_slot` | `meta.rs: Slots` (`reset`, `advance`, `reserve`); `mod.rs: Runtime::reset_slot` | free list order: `num_pages - 1 .. 0` pushed, popped from the back |
| `engine.cu: Engine::Engine` (arena size, weights, `alloc`, kernel attributes, `da::prepare` / `ring_prepare` / `ring_ctas_per_sm`, GEMM shapes planned, pins) | `mod.rs: Engine::new`, `planned_shapes`; `ink-model` (`MemConfig::arena_bytes`, `Weights::load`, `Activations::alloc`) | pins required (no timing tune) |
| `engine.cu: Engine::alloc` | `ink-model/src/plan.rs: Activations::alloc` (R4) | E1 |
| `engine.cu: Engine::commit` | `mod.rs: Runtime::commit`; `ops.rs: Dev::conv_commit` | once per forward call |
| `engine.cu: Engine::argmax`, `logits_host` | `mod.rs: Runtime::argmax_host`, `logits_host`; `ops.rs: Dev::argmax` | |
| `engine.cu: Engine::mark`, `fmark`; `capi: ink_rt_marks` | `forward.rs: Fwd::mark`, `Fwd::fmark`; `mod.rs: Engine::set_profile`, `Runtime::set_profile_fine`, `Runtime::marks` | `fmark` (profile_fine) ported in cut-over prep (`bench/raw/2026-11-24-cutover-prep/`): the 14 C++ sites, same names |
| `engine.cu: Engine::tapv`; `capi: tap trampoline` (host copy, scratch write-back) | `forward.rs: Fwd::tap` | |
| `engine.cu: Engine::gemm`, `for_groups`, `gemm_f32` | `forward.rs: Fwd::gemm`, `Fwd::run` (class per row group), `Fwd::gemm_f32` (call class) | C++ AlgoCheck fallback → Rust error |
| `engine.cu: Engine::run_model` | `forward.rs: Fwd::run_model` | feats copies at visits 5, 23, 41; msrc copies per pass |
| `engine.cu: Engine::visit` | `forward.rs: Fwd::visit` | |
| `engine.cu: Engine::attn_res`, `message_gates`, `norm3`, `attention`, `moe` | `forward.rs: Fwd::attn_res`, `message_gates`, `norm3`, `attention`, `moe` | |
| `capi/dumps.cpp: dump_launch_trace` (Kahn order, kernel / memcpy / memset nodes) of `last_graph_key` | `ink-cuda/src/trace.rs: Graph::launch_trace`; `mod.rs: Runtime::dump(LaunchTrace)` (the newest captured graph) | R6; memcpy fields read from the driver's `CUDA_MEMCPY3D` |
| (no C++ counterpart) eager launch trace | `ink-cuda/src/trace.rs` (eager kernels, dtod, cuBLASLt via one-call capture) | a call armed by `Runtime::set_launch_trace` runs eagerly |
| `capi/dumps.cpp: dump_weight_sums`, `dump_arena_map` | `ink-model/src/dump.rs` (R4) | |
| `engine.cu: Engine::spec_accept`, `build_verify_ids`, `gather_feats`; `forward` `dev_ids` segments | `engine/specrt.rs: impl SpecRuntime for Engine` (`spec_accept`, `build_verify_ids`, `gather_feats`, `forward_dev`; `SpecState::copy_dev_ids` called from `Engine::upload`, before a capture or replay) | package R9; checks of `ink_rt_capi.cpp` (`ink_rt_spec_accept`, `_forward`, `_build_verify_ids`) |
| `capi: ink_rt_dev_alloc`, `_h2d`, `_d2h`, `ink_draft_create` | `specrt.rs: alloc`, `upload`, `download`, `create_drafter`; `devbufs.rs` (ranges shared with the drafter) | |

## Launch sites (`ops.rs`, one safe function each)

| C++ site (`engine.cu` function: kernel) | `ops.rs` |
|---|---|
| `run_model`: `glue::embed_norm_kernel` | `Dev::embed_norm` |
| `run_model`: `glue::feat_copy_kernel` (visits 5/23/41, head) | `Dev::feat_copy` |
| `run_model`: `cudaMemcpyAsync` msrc (D2D) | `Stream::dtod` in `Fwd::run_model` |
| `run_model` / `attn_res`: `glue::attn_res_kernel<1|3|4, 0>`, `<1, 1>`, `<1, 3>`, `<3, 3>` (C3) at T <= 64 or on the reference path; `pglue::ar_rows_kernel<Q, G>` (same instances) at T > 64 (delta 3) | `Dev::attn_res`, `Dev::ar_rows` |
| `run_model`: `rmsnorm_f32out_kernel`, `gather_rows_bf16_kernel`, `softcap_kernel` | `Dev::rmsnorm_f32out`, `gather_rows`, `softcap` |
| `argmax`: `argmax_rows_kernel` | `Dev::argmax` |
| `visit`: `glue::attn_res_norm_kernel<1|3>`; `pglue::ar_norm_rows_kernel<1|3>` at T > 64 (delta 3) | `Dev::attn_res_norm`, `Dev::ar_norm_rows` |
| `visit`: `glue::rmsnorm_kernel<bf16|float>`; the message visits' `mlp_norm` on `pglue::norm_rows_kernel` (G = 1) at T > 64 | `Dev::rmsnorm_bf16`, `rmsnorm_f32`, `norm_rows` |
| `norm3`: `glue::rmsnorm_g_kernel`; `pglue::norm_rows_kernel` (G = 3) at T > 64 | `Dev::rmsnorm_g3`, `Dev::norm_rows` |
| `message_gates`: `glue::mg_in_kernel` (production, reference) | `Dev::mg_in`, `mg_in_one` |
| `message_gates` (reference): `bf16_to_f32_kernel`, `mg_out_kernel` | `Dev::bf16_to_f32`, `mg_out` |
| `message_gates`: `pglue::mg_tail_rows_kernel` (T <= 64, and the decode row groups at T > 64); `pglue::mg_tail_tc_kernel` (prefill row groups at T > 64, G7) | `Dev::mg_tail`, `Dev::mg_tail_tc` |
| `Engine::Engine`: `pglue::mg_tc_weights_kernel` (G7 weights) | `Dev::mg_tc_weights` in `Engine::new` |
| `visit`: decode projections `lt_.run` (fused weight) / `lt_.run2` (k / v batch), `proj_split_kernel` (D1) | `Fwd::run`, `LtGemm::run2`, `Dev::proj_split` |
| `visit`: `pglue::qkv_post_rows_kernel`, `glue::qkv_post_kernel` | `Dev::qkv_post_rows`, `qkv_post` |
| `visit` (reference): `qpost_kernel` + `rel_logits_kernel`, `kv_conv_kernel`, `kv_write_kernel` | `Dev::qpost_ref`, `kv_conv_ref`, `kv_write_ref` |
| `visit`: `pglue::relk_direct_kernel` | `Dev::relk_direct` |
| `attention`: `da::launch` / `da::ring_launch` (+ `combine_kernel`) | `Dev::decode_attn` |
| `attention`: `pfa::launch<4, 2, 1, 1, 2>` | `Dev::prefill_attn` |
| `visit`: `attn_conv_partial_kernel`; reference `hidden_conv_kernel` + `partial_kernel`; `hidden_conv_kernel` (MLP); `pglue::conv_rows_kernel<0|1>` at T > 64 (delta 3) | `Dev::attn_conv_partial`, `hidden_conv`, `partial_ref`, `conv_rows` |
| `moe`: `router_kernel` (reference), cuBLASLt K groups + `route_select_kernel` | `Dev::router_ref`, `Fwd::run`, `Dev::route_select` |
| `moe` `pmoe_rows`: `moe_count/plan/scatter_kernel` (n > 64) or `moedev::moe_build_kernel` | `Dev::moe_build` |
| `moe` `moe_launch_dev<FMT, MODE>` x 2 | `Dev::moe_gemm` |
| `moe`: `pmoe::moe_combine` | `Dev::moe_combine` |
| `commit`: `glue::conv_commit_kernel` | `Dev::conv_commit` |
| `spec_accept`: `argmax_rows_kernel`, `accept_kernel`, `last_feat_kernel`, `gather_feats_kernel`; `gather_feats`: `gather_feats_kernel`; `build_verify_ids`: `verify_ids_kernel`; `forward`: `cudaMemcpyAsync` D2D of `dev_ids` | `Dev::argmax`, `accept`, `last_feat`, `gather_feats`, `verify_ids`; `Stream::dtod` in `SpecState::copy_dev_ids` |
| `Engine::Engine`: `cudaFuncSetAttribute` (moe x4, da x2, ring x3, pfa) | `Engine::new` (`Function::set_max_dynamic_smem`) |

## Drafter (package R8): `draft.cu` / `draft.hpp` / host code of `drafter_rt.cuh` @ `1c7ac46` → `src/draft/`

DraftRT runs the S2f harness's `Drafter` on one stream (the second-stream branch and `g_overlap` are deleted since
package G2; review finding F12 does not apply), `g_head_warps 8`, `g_sk_wo 6`, `g_sk_down 6`, `g_attn_ctas 384`, `g_gemm3 0`,
`g_head_rt2 0`, `g_argmax2 3`, `g_coop 0`, `ksteps 7`, `vlim = VU`, no IMMA head; only that path is ported. Kernels are
not ported (pack TU `drafter`).

### Host logic

| C++ (file: function / site) | Rust | Notes |
|---|---|---|
| `draft.hpp/.cu: DraftRT::DraftRT` (attributes, `load_weights`, `Drafter` ctor, sync) | `draft/mod.rs: RustDrafter::new` | own primary-context handle and stream; one arena for weights and state, sized by an `ArenaPlan` dry run (`arena_bytes`) |
| `drafter_rt.cuh: load_weights` (tensor names, `load_bf16`, `cat_rows`, `to_frag`, `k_copy_cols`, fuse_alpha softmax, rnn bias, head norms, int8 head, `fold.bin`, `lowrank16.bin`, RoPE table) | `ink-model/src/drafter.rs` (files, `bf16_bytes`, `fuse_alpha`, `widen_bf16`, `bf16_cols`, `read_fold`, `read_lowrank`, `lowrank_path`); `draft/mod.rs: RustDrafter::load` | `cat_rows` and `k_copy_cols` are copies, done on the host; temporaries in scratch allocations freed after use; `head_f` only for the bf16 head, `head_q8k32` / `head_qn` (IMMA) not built |
| `drafter_rt.cuh: struct Weights`, `struct Layer` | `draft/mod.rs: DraftW`, `LayerW` (generic over `ink_model::Alloc`) | |
| `drafter_rt.cuh: Drafter::Drafter` (sizes, `split_attn`, allocations, initial page table / P / anchors, noise fill of the draft K/V) | `draft/mod.rs: Shape::new`, `DraftS::take`, `RustDrafter::init_state` | `feats`, `dbg`, `hq`, `hsx`, `hdn`, `hn_in` not allocated (unused on DraftRT's path) |
| `Drafter::step`, `bstep`, `cache`, `markov(k)` | `RustDrafter::step`, `bstep`, `cache`, `markov` | struct padding explicit zero; `bstep` / `markov(k)` over the round batch's compact rows (G2) |
| `Drafter::commit_rows` (chunks of 64) | `RustDrafter::commit` (via `Drafter::commit_rows`) | R = 0 is a no-op, as DraftRT |
| `Drafter::draft_batch(k)` (gather, layers, head, scatter: draft_sel.cuh, package G2) | `RustDrafter::draft_batch` | |
| `Drafter::layers(k)` | `RustDrafter::layers` | k compact rows; attention splits of the drafter (n) |
| `Drafter::head(k)` (`head_gemm`, `head_rest` without coop) | `RustDrafter::head` | memsets of the batch's 7 k rows |
| `gemm`, `gemm_t`, `gemm_w`, `ga` | `draft/ops.rs: Dev::gemm`, `Gemm` | the split check is an error |
| `DraftRT::commit_rows`, `frnn`, `set_state`, `draft_sessions` (list checks, upload of `sel`, per k: eager once, capture relaxed, instantiate, launch), `draft` (= all sessions), `drafts` | `impl ink_rt_api::Drafter for RustDrafter` | capture in thread-local mode on the drafter's stream (ink-cuda); `graphs[k - 1]` |
| `capi: ink_draft_*` device-range checks | `RustDrafter::alloc`, `upload`, `download`, `resolve` | handles (buffer number, offset) instead of device addresses |
| `draft_conf.cuh` (`ConfBuf`, `conf_save_kernel`, `conf_sum_kernel<N>` / `conf_sum_launch`, `conf_finish_kernel`), `Drafter::conf_on`, `DraftRT::set_conf`, `conf()` | `Dev::conf_save`, `conf_sum`, `conf_finish`; `RustDrafter::conf_buf`, `Shape::conf` (from `DraftConfig::verify`), `Drafter::conf` | over the round batch's compact rows (N = k), scattered to `conf [n][7]` with the drafts |
| `draft_sel.cuh` (`draft_gather`, `draft_scatter`) | launch sites `Dev::draft_gather`, `draft_scatter` | package G2 |

### Launch sites (`draft/ops.rs`, one safe function each)

| C++ site (function: kernel) | `ops.rs` |
|---|---|
| `to_frag`: `permute_frag_bf16` | `Dev::permute_frag_bf16` |
| `load_weights`: `permute_frag_i8`, `quant_rows_i8`, `row_norms`, `row_norms_i8`, `k_cos_sin` | `Dev::permute_frag_i8`, `quant_rows_i8`, `row_norms`, `row_norms_i8`, `cos_sin` |
| `Drafter::Drafter`: `k_fill_bf16` (`fill`), `k_fill_e4m3` | `Dev::fill_bf16`, `fill_e4m3` |
| `gemm_w`: `gemm_frag<MT, I8, EPI, 8>` (24 instances) | `Dev::gemm` (macro `gemm_frag!`) |
| `commit_rows`: `k_commit_mix`, `epi_kvctx` | `Dev::commit_mix`, `epi_kvctx` |
| `DraftRT::frnn`: `k_frnn` | `Dev::frnn` |
| `layers`: `k_embed_block`, `epi_qkv`, `attn_block` / `attn_block8`, `attn_combine`, `epi_addnorm`, `epi_silu` | `Dev::embed_block`, `epi_qkv`, `attn_block`, `attn_combine`, `epi_addnorm`, `epi_silu` |
| `head_gemm`: 4 x `cudaMemsetAsync` | `Stream::memset_zero` in `RustDrafter::head` |
| `head_rest`: `epi_ctx`, `markov_seed<I8>`, `markov_recur`, `markov_zref<I8>`, `markov_argmax3<I8>`, `conf_save_kernel`, `markov_finish`, `conf_sum_kernel<N>`, `conf_finish_kernel` | `Dev::epi_ctx`, `markov_seed`, `markov_recur`, `markov_zref`, `markov_argmax3`, `conf_save`, `markov_finish`, `conf_sum`, `conf_finish` |
| `DraftRT::set_state`: 2 x `cudaMemcpyAsync` D2D | `Stream::dtod` in `Drafter::set_state` |
| `DraftRT::DraftRT`, `gemm_w`: `cudaFuncSetAttribute` (attn x 2, gemm_frag per instance) | `RustDrafter::new` (`Function::set_max_dynamic_smem`) |
| not ported (other harness variants): `gemm_frag2/3/4`, `gemm_imma`, `k_quant_h`, `k_head_outliers`, `markov_argmax`/`argmax2`, `markov_coop`, `k_prep_head`, `k_real_head`, `k_bump`, `k_gather_cols`, `permute_frag_i8k32` | |
