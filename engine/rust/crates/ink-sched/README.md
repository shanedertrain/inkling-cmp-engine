# ink-sched: step scheduler (package S1, gate G-S)

Port of the C++ scheduler `engine/server/src/sched/` and `engine/server/src/backend/native_adapter.{h,cpp}` at
`4171493`; the C++ was deleted at cut-over U1 (gate record `bench/raw/2026-11-23-rust-sched-gate/`). Policy, defaults
and the measurements behind them: `SCHEDULER.md` (moved here from `engine/server/` at the cut-over). Patterns: vLLM v1
scheduler and block pool (queues, token budget, priority preemption, refcounted pages, prefix reuse), llama.cpp server slots (reuse of a parked slot), vLLM `EngineCore` loop behind a channel.

| Module | Content |
|---|---|
| `scheduler` | `Scheduler`: admission, decode rows, interleave rule (`choose_chunk`), prefix reuse, parking, eviction, preemption, `check_invariants`, `check_plan` |
| `pool` | refcounted page pool (free list, share, release) |
| `config` | `Config` (all `sched::Config` parameters, same defaults), `PrefillOrder`, `PrefixReuse`, `WidthHook` |
| `plan` | `StepPlan` (with `stale_slots`), `Segment`, `SegResult`, `SeqEvent`, `Finish` |
| `runtime` | `ModelRuntime` trait, `RuntimeCaps` (incl. `vocab`, `kv_pages`), `StepError` |
| `fake` | `FakeRuntime` (hash tokens, cost model, virtual clock), `FakeCost` |
| `engine_backend` | `EngineBackend`: `ink_backend::Backend` over scheduler + runtime, one loop thread; per-step timing events (`tracing` target `ink_step`, debug) |

The real runtime's `ModelRuntime` is `ink-runtime::RuntimeAdapter` (package I1, over any `ink_rt_api::Runtime`; the C++
engine through `ink_rt.h` today). It replaced `NativeRuntimeStub`, which was deleted.

## Runtimes with their own page tables (package I1)

A runtime that reports `external_pages: false` or `conv_checkpoints: false` (the C++ engine through the C ABI) gets
`PrefixReuse::ExactResume` from `EngineBackend::spawn`: a new request reuses resident KV only by the exact resume of
a parked sequence that is a prefix of its prompt, in its own slot with a valid ring (same salt). No page is shared
and no slot is reset above position 0, so every plan is one the engine can run. With `kv_pages: Some(n)` the pool is
narrowed to `n` pages and the session cap to the pool. Every plan lists `stale_slots` (slots no sequence holds, or
whose sequence starts over in them, also when its first segment comes in a later step); the runtime frees their state
before the step, since the engine's pages belong to its slots and the scheduler has already freed or reassigned
them. `PrefixReuse::Shared` (the default, and
every fixture) is unchanged.

## Gate G-S

`scripts/build.sh cargo test -p ink-sched --offline --locked`: the 14 C++ tests (same names without `sched_`), the
S2 review's additions (`mixed_paths`, `engine_backend_cap_and_dead_loop`, `abort_never_parks`,
`config_rejects_unschedulable_shapes`, `admission_counts_pages_shared_by_parked`), I1's
(`exact_resume_reuses_parked_prefixes_only`, `engine_backend_narrows_to_a_runtime_with_own_pages`; the 14 C++ tests
less `native_stub_reports_gaps`, deleted with the stub) and `invariants_hold_every_step`
(proptest: random arrivals, shared prefixes, small pools, 1-10 slots, 64/256/1,024-token pages, recompute windows,
eviction weights, cancels/finishes; `check_invariants` after every step and `check_plan` on every plan).

Plan-stream differential: `tests/fixtures/*.plan` were written by the C++ scheduler at `4171493` through
`engine/server/tools/sched_plan_dump.cpp` (deleted at cut-over U1; the version used, with the review's `mixed`
scenario, is `bench/raw/2026-11-23-rust-sched-gate/sched_plan_dump.cpp`). Each file holds every `StepPlan` of one
scenario (f64 as bit patterns, token and page lists as FNV-1a 64), the final counters, the scenario's summary numbers
as f64 bits and, in `mixed`, one line per finished sequence with its reason. The Rust tests replay the same scenarios
and compare byte for byte: `tokens_exact`, `prefix_reuse`, `eviction`, `preemption`, `interleave_rule`,
`interleave_fixed`, `closed_loop_{fifo,srpt,slow,fixed}`, `mixed` (14,137 plans). The comparison sees a 1-ulp change:
the same C++ built with FMA contraction (`-march=native -ffp-contract=fast`) differs in 7 of the 10 original files
(gate record). The fixtures are the oracle now; regenerate them only with a reason recorded.

## Port notes

| C++ | Rust |
|---|---|
| `sched::Config` | `Config`; `prefill_srpt` → `PrefillOrder`; `width_hook` → `Option<WidthHook>` |
| `Scheduler::add` returning `optional<string>` | `Result<(), AddError>`; messages unchanged |
| constructor throwing `invalid_argument` | `Scheduler::new -> Result<_, ConfigError>`; messages unchanged |
| `check_invariants` throwing `logic_error` | `Result<(), InvariantError>`; messages unchanged |
| `Seq*` pointers into `std::map` | ids into a `BTreeMap` (same iteration order, so the same tie-breaks) |
| slot `-1`, `slot_owner_` 0 = free | `Option<usize>`, `Option<u64>` |
| `SeqEvent{done, error, error_message}` | `finish: Option<Finish>` (`Done`, `SessionCap`, `Error(msg)`) |
| `ModelRuntime::step` throwing | `Result<_, StepError>`; the back end fails that step's requests with the message |
| `EngineBackend(cfg, shared_ptr<rt>, threaded)` | `EngineBackend::spawn(cfg, make_rt) -> Result<_, SpawnError>`; the loop thread builds and owns the runtime (no `Send` bound on it), always threaded (mpsc channel, 20 ms idle wait) |

Deliberate differences:

| Difference | Why |
|---|---|
| The runtime is built on the loop thread (`spawn` takes a `FnOnce() -> Result<R, E> + Send`), not on the caller's (F5; S2 review's open point) | a CUDA context belongs to the thread that made it; the runtime need not be `Send`. `new` is deleted. The caps come back with the ready signal (`EngineBackend::caps`) |
| A panic in the scheduler, the runtime or a callback is fatal (F5) | the loop thread ends, `healthy()` is false, a later submit is answered with "scheduler loop stopped", callbacks of requests in flight are dropped (the front end reports an error for them). Per-callback isolation would mean `catch_unwind` around user code and a scheduler left in an unknown state; tests `a_panicking_callback_is_fatal`, `engine_backend_cap_and_dead_loop`, and `http_loop_panic_is_fatal` in `ink-api` |
| `EngineBackend` has no unthreaded mode (`run_once`/`run_until_idle`, `scheduler()`) | no caller or test used it; the deterministic tests drive `Scheduler` + `FakeRuntime` directly |
| `FakeRuntime` has no `accept` override and no `real_sleep` | unused by any test or caller |
| `pool_bytes > 0` with `kv_bytes_per_token == 0` is a `ConfigError` | the C++ divides by zero |
| `session_cap_tokens` above the pool is a `ConfigError` (S2 review) | the C++ accepted it, and a lone session growing past the pool preempted itself and recomputed forever; vLLM refuses the same at start-up. The `eviction` and `preemption` scenarios now set the cap to their pool (plans unchanged, fixtures regenerated identical) |
| a zero row limit or `max_decode_rows > t_max` is a `ConfigError` (S2 review) | nothing could ever be scheduled (requests wait forever) |
| admission counts pages whose every owner is parked as evictable (S2 review) | the C++ counted refcount-1 pages only: pages shared among parked sequences were never counted, and a request needing them waited forever (found by the property test; `admission_counts_pages_shared_by_parked`). No fixture changed |
| a failed step frees its sequences (`Scheduler::abort`), never parks them (S2 review) | the C++ cancelled them, which parks with `park_on_cancel`; an exact resume would then continue from the unknown slot state of the failed step |
| `EngineBackend::healthy` is false once the loop thread ended; a submit after that is answered with an error (S2 review) | the C++ reported healthy forever and dropped the request |
| A request id 0 is an ordinary id | the C++ slot table used 0 as "free" |
| "session cap reached" is `StepOutput::session_cap` (S2 review) | the C++ sent it as `error_message` with `error` unset; the front end maps it to `finish_reason: "length"`. It arrives on the event carrying the last tokens (`done` and tokens together), not on an empty event |
| Threaded smoke test waits for the metrics of the last iteration | the C++ test read them right after the last callback, a race with the metrics update |

Preserved quirks: a prefill result with several tokens is cut to one but its logprobs are passed through whole; a
duplicate `SegResult` for one sequence: the last wins; `admit_one` can run prefix reuse and then fail on the slot
(unreachable after its capacity check); after a step error `commit` still runs with no results (counts the step).

PORT-MAP rows (for `engine/rust/PORT-MAP.md`, source commit `4171493`, state: ported, gate G-S passed):
`src/sched/scheduler.{h,cpp}` → `ink-sched::{scheduler,pool,config,plan}`; `src/backend/native_adapter.h`
`RuntimeCaps`/`ModelRuntime` → `ink-sched::runtime`; `FakeRuntime` → `ink-sched::fake`; `NativeRuntimeStub` →
`ink-runtime::RuntimeAdapter` (I1; the stub is deleted); `EngineBackend` → `ink-sched::engine_backend`; `tests/test_sched.cpp` → `tests/sched.rs`.
Not ported: `make_engine_backend` (always throws; the `--backend engine` branch goes with U1, I1 builds the real one).
