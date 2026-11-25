//! Step scheduler of one card: admission, paged KV pool, prefill/decode interleave, prefix reuse, parking, eviction
//! and preemption, plus the loop thread that drives a [`ModelRuntime`] for the front end.
//!
//! Port of `engine/server/src/sched/` and `engine/server/src/backend/native_adapter.{h,cpp}` at `4171493`. Policy,
//! defaults and the measurements behind them: `SCHEDULER.md` in this crate (moved from `engine/server/` at cut-over U1).
//!
//! | Item | What it is |
//! |---|---|
//! | [`Scheduler`] | CPU policy: `add`/`finish`/`cancel`, `plan` a step, `commit` its results |
//! | [`ModelRuntime`] | what the scheduler needs from a model: run one [`StepPlan`] |
//! | [`FakeRuntime`] | deterministic tokens and the cost model on a virtual clock (tests, policy simulation) |
//! | [`EngineBackend`] | [`ink_backend::Backend`] over a scheduler and a runtime, one loop thread per card |
#![forbid(unsafe_code)]

mod config;
mod engine_backend;
mod fake;
mod plan;
mod pool;
mod runtime;
mod scheduler;

pub use config::{Config, PrefillOrder, PrefixReuse, WidthHook};
pub use engine_backend::{EngineBackend, SpawnError};
pub use fake::{FakeCost, FakeRuntime};
pub use plan::{Finish, SegKind, SegResult, Segment, SeqEvent, StepPlan, Stop};
pub use runtime::{Drafting, ModelRuntime, RuntimeCaps, StepError};
pub use scheduler::{AddError, ConfigError, Counters, InvariantError, PromptLimits, Scheduler};
