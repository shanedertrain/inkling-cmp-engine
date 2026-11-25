# ink-runtime: the Rust engine and the runtime side of serving (packages R5, R6, R8, R9, I1, I2)

| Module | Content |
|---|---|
| `engine` | `Engine` (package R5): the C++ `ink::Engine` forward pass (eager, decode calls as CUDA graphs) over the kernel pack, bit for bit; implements `ink_rt_api::Runtime`. Port map: `PORT-MAP.md` (C++ @ `1c7ac46`) |
| `engine::meta` | host state and per-call metadata, no device access: `Slots` (lengths, pages, free list), `build_call` (segments, row groups, page tables in call and decode order, sliding-window rings, decode-attention metadata, prefill-attention items with key splits), `decode_gx`, `pick_workers*`, `graph_key` |
| `engine::ops` | every kernel launch: one safe function per C++ launch site (extents checked where the host can bound them, launch geometry as the C++, `// SAFETY:` with the kernel's contract); with `draft::ops` the only modules with `#[allow(unsafe_code)]` under the crate's `#![deny(unsafe_code)]` (review R3 deviation 4) |
| `engine::graphs` | `GraphCache` (R6): captured decode graphs by key, the ablation over their regions (`set_ablation`, region counts) |
| `engine::forward` | one call (eager or into an open capture): 48 visits, AttnRes, message gates, attention (decode_attn design A/B, prefill_attn), router + experts (device item list), head; dense GEMMs through `LtGemm` with the row group's plan class; taps; profile marks |
| `adapter` | `RuntimeAdapter` (I1/I2): the scheduler's `ModelRuntime` over any `Runtime`, speculative rounds over a drafter |
| `draft` | `RustDrafter` (package R8): the C++ DSpark drafter `ink::DraftRT` over the kernel pack, bit for bit (`draft::ops`: its launch sites, the second module with `#[allow(unsafe_code)]`); implements `ink_rt_api::Drafter`; its own context handle, stream and arena. Port map: `PORT-MAP.md` "Drafter". Gate E8: `bench/raw/2026-11-23-rust-R8-drafter/` |
| `spec`, `sampler` | the speculation round of the adapter; host sampling and logprobs |

## Engine: what R5 covers and what it refuses

| Covered | Refused at start or call (later packages) |
|---|---|
| eager forward of decode, ragged verify, prefill and mixed calls; chunk 2,048 and 4,096; glue 6 and the per-op reference glue (glue 1, always with a tap); prefill_moe for every expert row, D1 fused decode projections, C3 (delta 1, `PORT-MAP.md` "Deltas ported since rust-r5-base"); commit, reset, argmax, logits; device-side speculation (`Runtime::spec`, package R9: device ranges, verify ids, forward with device ids, greedy accept, feature gathers, the Rust drafter on the engine's context and stream); taps with write-back; weight sums, arena map, eager launch traces (`Runtime::set_launch_trace`); profile marks (`Engine::set_profile`); CUDA graphs of decode and verify calls (R6: capture per key at first use, replay, region map, `set_ablation`, launch traces of captured graphs) | a missing pin file (algorithms must be pinned); forced-mode accept and the confidence gate |

Drop order (struct field order): cuBLASLt handle, activation and weight buffers, kernels, arena, stream, context.
`ink-tools` (`--runtime rust`) and `ink-server --backend engine --runtime rust` create it.

## Engine tests

`scripts/build.sh cargo test -p ink-runtime --offline --locked`: no GPU. `meta` (free list order and return, exhausted
pool takes nothing, row and logit groups of a mixed call, page tables and rings, decode metadata, decode chunk bounds,
worker choice, graph key, prefill items with global key splits and their sort, sliding-window items: expected values
derived by hand from `engine.cu`), launch constants (shared memory sizes = the values the C++ trace shows), the 25
planned GEMM shapes; then the adapter tests. On the card: `bench/raw/2026-11-23-rust-R5-forward/` (gates E1-E4).

## Adapter (I1, I2)

`RuntimeAdapter` is the scheduler's `ink_sched::ModelRuntime` over any forward `ink_rt_api::Runtime`, written once
over `dyn Runtime` (docs/RUST-PLAN.md 3.3): today it drives the C++ engine through the C ABI (`CapiRuntime`,
configuration `M3`, `ink-server --backend engine`) and the Rust `Engine` (`--runtime rust`, configuration `M4`, no
drafter yet). `sampler` is host sampling and logprobs over one logit row. Pattern: vLLM v1's model-runner step behind its
`EngineCore` loop (gather the scheduled segments, one forward, sample the last row of every sampling request).

| Module | Content |
|---|---|
| `adapter` | `RuntimeAdapter`: caps from the `EngineConfig` the runtime was created with (`new`: no drafter; `with_drafter`: the runtime's DSpark drafter, caps `Drafting::FixedGreedy { k: 7, ctx }`); per step: reset stale slots (`StepPlan::stale_slots`) that still hold state, check every segment against the runtime's slot state (reset for a new occupant at position 0, `length(slot) == pos0` otherwise; a verify segment also against the drafter's anchor and position), the verify segments as one speculative round, the other segments (prefill chunk, one-row decode rows) as one `forward`: read the device argmax (all sampling segments greedy without logprobs) or copy the logit rows and sample on the host, `commit` every row, commit the rows to the drafter; then check that the runtime's free KV pages equal its pool minus the pages of the scheduler's segments its slots hold; one `ink_step` debug event per step with host timings |
| `spec` | `Speculation`: the drafter, the round's device buffers and the host copy of every slot's anchor and position. The round is `engine/native/tools/spec.cpp`'s call for call (`set_state`, `frnn`, `draft`, `build_verify_ids`, one `forward_dev` over all verify segments, `spec_accept`, engine `commit` of the accepted rows, drafter `commit_rows`); the emitted tokens end at the request's stop (`ink_sched::Stop`: stop token once `min_tokens` are out, or the token budget) and only those rows are committed. Rows committed outside a round go to the drafter after their call; every slot's drafter position stays at its committed length so the drafter's block writes of slots that do not speculate land on rows not yet committed. Optional trace: one JSON line per round |
| `sampler` | `sample` (vLLM v1 semantics: greedy below temperature 1e-5, else temperature, min-p, top-k, top-p, draw; per-sequence `SmallRng` seeded with the request's seed) and `logprobs` (raw log-softmax, rank, top-k) |

## What the seam serves, and what it does not

| Serves | Through |
|---|---|
| Decode, one row per session | `forward` with logits 1 per decode segment |
| Speculative decode for greedy requests without logprobs (with `with_drafter`): 8 rows per session per round (anchor + 7 DSpark drafts), greedy accept on the device, while the rows stay below the drafter's positions per slot | `SpecRuntime` (`build_verify_ids`, `forward_dev`, `spec_accept`, `gather_feats`) and `Drafter`; the scheduler plans width 8 or 1 from the caps |
| Chunked prefill as the scheduler plans it (interleave rule; chunks up to 2,048 rows, in the same step as decode rows) | one call with the step's one-row decode rows, after the round; logits on the last chunk only |
| New occupant of a slot | `reset_slot`, position 0 |
| Prefix reuse: exact resume of a parked sequence that is a prefix of the new prompt, in its own slot with a valid ring | no reset; `length(slot) == pos0` is checked; the round keeps no row past the request's stop, so the parked KV ends where the client's output ends |
| Slots nobody holds, or whose new occupant has not run yet | `reset_slot` once (their pages go back to the engine's pool before the scheduler hands them out again) |
| Greedy without logprobs | `argmax_host`: the engine's device argmax of the logit rows (no logit copy) |
| Other sampling, logprobs | one row per step, on the host from the 201,024-wide fp32 rows `logits_host` returns (first 200,058 valid) |
| KV page accounting | `free_pages` checked after every step against the scheduler's pages of the occupied slots |
| Vocabulary for the front end's prompt-id check | `RuntimeCaps::vocab` = 200,058 |
| Cancel between steps | the scheduler frees the slot; the next step resets it |

Not served (each fails the step's requests with `StepError` if a plan asked for it; the scheduler is configured from
the caps so that it does not): speculation for sampled requests or with logprobs, verify widths other than 8, page
sharing across sessions and resume at a page boundary in another slot (needs scheduler-owned page tables and conv
checkpoints, `SCHEDULER.md` needs 1-3), device sampling (need 5), a mid-step abort. Requests to the C ABI: the I1
record (`bench/raw/2026-11-23-rust-I1-served/README.md`) and the I2 record
(`bench/raw/2026-11-23-rust-I2-spec-served/README.md`).

## Adapter tests

`scripts/build.sh cargo test -p ink-runtime --offline --locked`: sampler properties (greedy ties and NaN, seed
determinism, softmax frequencies, top-k/top-p/min-p cuts, top-p beyond the sorted head, raw logprobs); the adapter
over a model-free `Runtime` (caps, prefill + decode + greedy, idle-slot reset and exact resume, refused plans,
logprobs and seeded draws, device argmax for greedy rows, a page count out of step fails the step, a mixed plan as
one call, scheduler and runtime page pools in step over 4,000 random steps with resumes, cancels and evictions (the
fake engine refuses a forward when its own pool is exhausted, as `ink::Engine` does), four concurrent requests
through `EngineBackend`); and `tests/spec.rs`, the adapter with a drafter over a model-free runtime with device-side
speculation and a fake drafter that checks the real drafter's preconditions (every position below its `P` committed
with the target's row, `f_rnn` from the row at `P - 1`, block writes inside its positions): greedy output equal to
the target's chain with staggered arrivals and a temperature request beside them, a stop token inside a round, exact
resume of both kinds of finished sessions, sessions past the drafter's positions, refused verify segments. On the
card: the I1 and I2 gate records.
