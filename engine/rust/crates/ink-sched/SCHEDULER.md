# ink-sched: scheduler design and policy (S5 design; Rust since cut-over U1)

CPU logic only, no CUDA. Moved here from `engine/server/SCHEDULER.md` with the C++ scheduler's deletion (cut-over U1,
gate G-S: `bench/raw/2026-11-23-rust-sched-gate/`). Files: `src/scheduler.rs` (policy), `src/pool.rs` (page pool),
`src/engine_backend.rs` (`EngineBackend` loop thread over a `ModelRuntime`), `src/fake.rs` (`FakeRuntime`),
tests `tests/sched.rs`, plan-stream fixtures `tests/fixtures/`. The real runtime's side: `ink-runtime::RuntimeAdapter`.
Labels: **[M]** measured (cited), **[D]** derived, **[S]** simulated with the cost model (not measured), **[I]** inferred.

## Shape

```
front end ──submit/finish/cancel──▶ EngineBackend (1 thread per card) ──plan──▶ ModelRuntime.step(plan)
          ◀──on_step(tokens, cached, done)──┘     Scheduler: admit · decode rows · ≤1 prefill chunk · pages
```

Each loop iteration: apply queued commands → `Scheduler::plan(now)` → `runtime.step(plan)` (forward + drafter + sampling +
accept; returns accepted tokens per sampling segment) → `Scheduler::commit` → `on_step` callbacks. The front end owns
EOS / stop strings / `max_tokens` and calls `finish`; the back end may overshoot by one step (dropped by the front end,
as vLLM's engine core does). A runtime error (`StepError`) fails the requests of that step with an error StepOutput;
their KV and slot are freed, never parked (the slot state is unknown after a failed step).

A step (`StepPlan`) = one decode segment per decoding session (`width` = 1 + k rows, input = last accepted token; the
drafts are the runtime's) + at most one prefill segment (a chunk of one prompt; `samples` on the last chunk). Every
segment carries its slot, `pos0`, page list, `reset_slot` and the request's sampling parameters.

## Policy and defaults (all are `Config` launch parameters; tune on the real engine)

| Item | Rule | Default | Why |
|---|---|---|---|
| Admission | priority, then arrival; head-of-line (FIFO fairness); a preempted request keeps its place; admitted when a slot is free or stealable and the free pages plus the pages eviction of every parked sequence would free cover the first chunk | `max_running` 4 (8 qualified) | owner target 4/card; fleet concurrency p99 3-4 per 4-GPU instance [M, P3 `docs/stage-p/P3-fleet-prompt-lengths.md:58-63`] |
| Slots | runtime slots (SWA ring + conv state); parked sessions keep theirs while free; a new runner steals the LRU parked slot | 8 (= `EngineConfig::max_slots`) | engine/native README "Contracts" |
| KV pool | refcounted pages of `page_tokens`; pool = `pool_bytes / kv_bytes_per_token` or `pool_tokens` | 36 KiB/token, 1.14M tokens, 1,024-token pages | PLAN 5.1; P3:95; engine/native global page = 1,024 keys |
| Session cap | prompt + output ≤ cap; prompt + 1 > cap → rejected at `submit` with "prompt of N tokens exceeds the per-session cap of C tokens"; generation stops at the cap (`StepOutput::session_cap`, finish_reason "length"); the cap may not exceed the pool (`ConfigError::SessionCapOverPool`, as vLLM refuses a max length the KV cache cannot hold) | 262,144 (max 1,048,576) | owner decision (operate at 262,144); model maximum 1,048,576 |
| Decode rows | every decoding session every step, width 1 + k; with `Drafting::Any` k from `width_hook(seq, spec_k)` clamped to [0, min(spec_k, rows−1)] and to the token budget; with `Drafting::Fixed { k, ctx, sampled }` (a drafter that always drafts k: the C++ runtime's DSpark; greedy accept, and with `sampled` the device sampled accept) width 1 + k for requests without logprobs that are greedy (or any, with `sampled`) whose rows stay below `ctx` and the session cap, also past the token budget (the round keeps no token past the request's stop, `Stop`), else 1 | k = 7 static | static k = 7 is best at 45K n = 4 [M/D, P10 `bench/raw/2026-11-22-P10-verify-width/README.md:89-91`]; history rules gain 0-2 %, confidence rule undecided [D, P2 `docs/stage-p/P2-adaptive-width.md:190-194`] → hook only |
| Prefill chunk | **unthrottled `max_chunk`** by default (`stall_budget_ms = inf`); finite `stall_budget_ms` enables the interleave budget/urgency rule (below); FIFO or SRPT among prefills | budget `inf`, min 128, max 2,048, urgency 0.8 × 5 s, urgent cap 300 ms, FIFO | measured on GPU 6/7 agent replay (`bench/raw/2026-11-24-chunk-policy/`): fixed 2,048 beats 60 ms budget by +24–26 % M1 and cuts TTFT p50/p90/p99 (each small chunk pays the full weight-read step cost) |
| Step limits | rows ≤ `t_max`; decode segment ≤ 8 rows; prefill segment ≤ 2,048 rows | 2,112 / 8 / 2,048 | engine/native API contract |
| Prefix reuse | at admission: (1) a parked sequence that is a prefix of the prompt is taken over (exact resume when its slot ring is still valid: everything reused, no recompute); (2) otherwise share the full pages of the longest resident match with the same `cache_salt` (any session), minus the SWA window, page aligned | `swa_recompute_tokens` 1,024 | 94.7 % of fleet prompt tokens hit the prefix cache [M, P9 `bench/raw/2026-11-22-P9-prefill/README.md:71`]; sessionless traffic is 22 % of requests [M, P3:19] so matching is by token prefix, session only ranks |
| Parking | a finished request's KV stays resident (tokens whose KV exists); an older parked copy of the same session that is a prefix of the new one is dropped | `park_on_finish` on, `park_on_cancel` off | think time p50 0 s, p90 4-13 s [M, P3:52]: the next turn usually arrives at once |
| Eviction | only parked sequences, highest score first: age_ms + `evict_size_weight` × pages (pure LRU by default); never a running one | weight 0 | |
| Pool exhausted mid-generation | evict parked; then preempt the running request with the lowest priority, youngest first (not one already in this step's plan): pages freed, back to the queue, recomputed later (its prompt + generated tokens; the next sampled token continues the stream, nothing is re-sent) | — | |
| Cancel | stop, free (or park if `park_on_cancel`), `done` | — | |
| Stop | a segment's emitted tokens end after the first stop token id (EOS unless ignored, `stop_token_ids`) once `min_tokens` are out, or at `max_tokens` (`Stop::keep`, vLLM's `check_stop`); the runtime commits the rows of the kept tokens only and the scheduler finishes the request there, so a parked session's KV ends where the client's output ends | — | exact resume of the next turn after a speculative round that ran past the stop |
| Parked pages | a parked sequence returns the pages past its computed tokens (`trim_parked_pages`); off for a runtime with its own page tables, which cannot free part of a slot | on (off over `ink_rt.h`) | verify rows can reach a page the kept tokens do not |

**Page sharing rule:** a page with refcount > 1 is never written. Sharing therefore stops `swa_recompute_tokens` before the
match end (rounded down to a page): the reused sequence recomputes the last window into its own pages, which rebuilds its
SWA ring; exact resume (same slot, ring valid) needs no recompute. Partial reuse needs a conv-state checkpoint at the page
boundary from the runtime (gap list below).

## Interleave rule (minimise TTFT p99 without stalling decoders)

Step cost estimate (scheduler side; defaults = config B measurements, to be refit on the engine):
`decode(R) = 25.4 + 0.617·R ms` [M, P10:41 linear fit, T 8-32 at 45K, n = 4] and
`prefill(c, ctx) = c·(0.171 + 0.00245·(ctx + c/2)/1000) ms` [M/D, P9:95-98: 105+24+22+12+6+2 µs per new token excluding
global attention, 2.45 µs per token per 1,000 cached keys]. Check: a 2,048 chunk at 32K → 515 ms, measured 390-525 ms per
chunk step [M, P9:63-66].

1. No session decoding → largest chunk (2,048): nothing to stall, best prefill efficiency (8K chunks are −10-12 % wall
   [M, P9:99-101]; capped at the engine's 2,048-row segment).
2. Otherwise **budget chunk**: the largest c ≥ `min_chunk` with `prefill(c, ctx) ≤ stall_budget_ms`.
3. **Urgency override**: project the TTFT of every queued prefill in order, `elapsed_i + ⌈cum_remaining_i / c⌉ ·
   (decode(R) + prefill(c, ctx))` (waiting requests counted with their expected cache hit). If the worst exceeds
   `urgency_frac × ttft_target`, grow c in 64-token steps to the smallest chunk that meets it, but never past
   `prefill(c) ≤ urgent_stall_cap_ms` (a hopeless TTFT, e.g. a 19K suffix or a cold 40K prompt, must not freeze decoders).
4. One prefill segment per step; `prefill_srpt` switches FIFO → shortest-remaining-first.

What it replaces: config B gives a prefill 2,024 of every 2,048-row step, so the other sessions advance once per
390-525 ms (3.2-10.1x slower, longest gap 0.58 s; 20.9x and 1.73 s with 8K chunks) [M, P9:56-61, 63-67].

## Simulated results [S] (cost model above, `FakeRuntime`; `tests/sched.rs`; measured with the C++ scheduler at `4171493`, reproduced bit for bit by the Rust port: `tests/fixtures/interleave_*.plan`, `closed_loop_*.plan` summary lines)

Not measured: the coefficients are config B's, the engine's will differ; decode cost ignores context length.

| Scenario | Rule (defaults) | Fixed 2,048 chunk (config B shape) |
|---|---|---|
| 3 sessions decode at 45K (k = 7) + a 6,000-token turn on a 45,056-token cached prefix: TTFT | **2.90 s** | 1.85 s |
| same: worst step during the prefill (decode-only step 40.2 ms) | **100 ms (2.5x)** | 632 ms (15.7x) |
| closed loop, 4 sessions × 20 turns (suffixes 13×1.6K, 3×3K, 3×6K, 1×19K per session; outputs 124/344/871): warm TTFT p50 / p90 / p99 | **1.27 / 3.97 / 11.1 s** | 0.71 / 2.64 / 8.63 s |
| same: 6K turns, max TTFT | **3.98 s** | 2.70 s |
| same: 19K turns TTFT | 6.0-10.0 s | 5.7-8.6 s |
| same: steps carrying decode rows, p50 / p99 / max | **45 / 334 / 340 ms** | 45 / 830 / 1,000 ms |
| same, runtime 1.5x slower than the estimate (robustness) | warm p90 6.97 s, 6K max 5.33 s, step max 510 ms | — |
| SRPT instead of FIFO | p50 1.10 s, p90 3.96 s (≈ FIFO) | — |

Reading [D]: the rule keeps decoders at ≤ 2.5x (no urgency) / ≤ 340 ms steps (urgency) where config B's shape stalls
them 15x / 1 s, and still meets 5 s for the p90 fleet turn (6K) under 4-session load. The p99 turn (19K uncached on
50-90K context) costs ~6.7 s even alone in this model [D] — the 5 s p99 needs faster prefill (P9 section 5: 1.3x needs
attention ≈ 2x), not scheduling. Cold first turns above ~22K miss 5 s alone at config B rates [D, P9:170-171].
`stall_budget_ms`, `urgency_frac` and `urgent_stall_cap_ms` set the trade between TTFT and decode goodput (M1); tune them
with agent-replay M1/M3 on the real engine.

## Tests (`tests/sched.rs`, 20 tests, all virtual-clock and deterministic except the two threaded ones)

`engine/rust/scripts/build.sh cargo test -p ink-sched --offline --locked`. The 14 tests of the C++ scheduler: admission at 4 and 8; priority/FIFO order; exact token streams and pages back to baseline after finish/cancel (invariants
checked every step: refcounts = owners, free list exact, slots consistent); over-cap rejection and cap 1,048,576;
multi-turn prefix reuse (exact resume 5,049 cached; divergence → 1,024; other session same salt → 3,072; other salt → 0);
LRU eviction; preemption (121 preemptions, outputs identical to the unpreempted sequence); cancel frees; step shape limits
(≤ 1 prefill segment, ≤ 8 decode rows, ≤ 2,112 rows); width hook; interleave scenario and closed loop (above);
`EngineBackend` threaded smoke incl. a rejected request (the 14th, the `NativeRuntimeStub` gap report, went with the
stub at I1). Added in the S2 review:
`mixed_paths` (priorities, width hook, session cap, slot stealing, size-weighted eviction, salts, 256-token pages,
SRPT, parking on cancel; fixture written by the C++ scheduler), `engine_backend_cap_and_dead_loop`,
`abort_never_parks`, `config_rejects_unschedulable_shapes`, `admission_counts_pages_shared_by_parked`, and the
property test `invariants_hold_every_step` (page and slot invariants after every step, every plan writes only
unshared pages it owns, `Scheduler::check_plan`). Every scenario that the C++ dump tool also ran is compared line by
line with its `tests/fixtures/*.plan` (f64 as bit patterns): after the cut-over these files are the oracle;
regenerate them only with a reason recorded.

## The C++ runtime through `ink_rt.h` (`ink-runtime::RuntimeAdapter`, package I1) and what we need from its API

Mapping (`engine/native/src/engine.hpp` through the C ABI): segment → `SegIn{slot, ids, logits}` (decode: last token,
logits 1; prefill: chunk, logits 1 on the last chunk), `reset_slot(slot)` for a new occupant at `pos0 == 0` and for
every slot in `StepPlan::stale_slots` that still holds state; verify segments (width 8, package I2) as one
speculative round (`ink-runtime` `spec`: drafter, `forward_dev`, `spec_accept`, commits), then the other segments
(prefill chunk, one-row decode rows) as one `forward(segs)` (I1 ran them as two calls; the I2 gate measured one call
against the oracle, `bench/raw/2026-11-23-rust-I2-spec-served/`), device argmax or host sampling of the logit rows
(greedy, temperature, top-k, top-p, min-p, seed; raw logprobs), `commit({rows kept})`. Served: speculation for greedy
requests without logprobs (need 4, fixed width), chunked prefill, exact resume of a parked sequence in its own slot
(`PrefixReuse::ExactResume`, selected from the caps), the pool = the engine's pages. Anything else in a plan (a
width other than 1 or 8, a reset above position 0, a slot whose length is not `pos0`) returns `StepError` and fails
that step's requests. The needs below are built once, in Rust (R11), never in C++.

| # | Need from the runtime API | Why (scheduler feature) |
|---|---|---|
| 1 | **Scheduler-owned page tables**: `forward` takes each segment's page list (page ids from a pool the scheduler allocates), instead of the Engine's per-slot free list; `release` is the scheduler's | prefix sharing (refcounted pages across sessions), adoption across slots, preemption, eviction |
| 2 | **Session/slot binding**: `bind(slot, seq)` with `pos0` given per segment (length not implied by the slot) | a request resumes in any slot at any page boundary |
| 3 | **Conv-state checkpoints** at page boundaries (48 visits × 3 × 6,656 bf16 ≈ 1.9 MB per 1,024 tokens, ~5 % of the page bytes) or an equivalent restore | partial prefix reuse: recompute starts at a page boundary with the right conv state; SWA ring rebuilt by recomputing ≥ 1,024 tokens |
| 4 | **Drafter + verify rows on device**: decode segment of width 1 + k, runtime drafts (DSpark block of 8, cost flat in k [M, P10:44-46]), verifies, accepts, commits per segment; returns accepted tokens | speculative decoding, ragged widths per session |
| 5 | **On-device sampling** (temperature, top_p, top_k, min_p, per-request seed) returning token ids, not 201,024-wide logits | host copy of logits per row is 0.8 MB |
| 6 | **Per-row logprobs**: sampled token logprob + rank + top-k (k ≤ 20) ids/logprobs | OpenAI `logprobs`/`top_logprobs` |
| 7 | **Cancel/abort between steps** is enough (no mid-step abort); a failed step must leave other slots consistent | client disconnect |
| 8 | **Page-table / slot export** for metrics and invariant checks (pages per slot, ring validity) | `/metrics`, soak test |
| 9 | **Measured step costs** (decode rows, prefill tokens × context) to refit the scheduler's cost model online | the interleave rule's estimate |
| 10 | Mixed step = decode segments + one prefill segment in one `forward` (already supported: "a call may mix both") | interleave |

## Not done

Real runtime: the C++ engine through `ink_rt.h` (I1), plain decode only; cost model is config B's (refit on the engine); decode cost has no context term; no lossy-mode
pool sizing (FP8 KV halves bytes/token: set `kv_bytes_per_token`); no per-priority preemption fairness beyond
priority/youngest; no admission control by projected memory of max_tokens (pages are allocated as KV grows, preemption
covers overflow).
