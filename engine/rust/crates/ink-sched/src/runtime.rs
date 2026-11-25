// Ported from engine/server/src/backend/native_adapter.h @ 4171493 (RuntimeCaps, ModelRuntime).
//! The boundary between the scheduler and a model.

use crate::plan::{SegResult, StepPlan};

/// The verify rows a runtime can run: what width a decode segment may have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Drafting {
    /// No drafter: every decode segment has one row.
    #[default]
    None,
    /// Any width from 1 to `1 + spec_k` for any request.
    Any,
    /// A drafter that always proposes `k` tokens (the C++ runtime's DSpark through `ink_rt.h`): a decode segment has
    /// one row or exactly `1 + k`, and `1 + k` only for a request without logprobs whose rows stay below position
    /// `ctx` (the drafter's positions per slot) and that is greedy, or sampled when the runtime accepts sampled rows on
    /// the device (`sampled`).
    Fixed {
        /// Drafts per round.
        k: u32,
        /// Positions per slot the drafter holds.
        ctx: u32,
        /// Sampled requests speculate too (device sampled accept).
        sampled: bool,
    },
}

/// Capabilities and limits a runtime reports to the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeCaps {
    /// Engine slots.
    pub max_slots: u32,
    /// Rows per forward call.
    pub t_max: u32,
    /// Tokens per KV page.
    pub page_tokens: u32,
    /// Most rows in one decode segment.
    pub max_decode_rows: u32,
    /// Most rows in one prefill segment.
    pub max_prefill_rows: u32,
    /// Accepts the scheduler's page tables (needed for prefix sharing).
    pub external_pages: bool,
    /// What verify rows (decode width above 1) the runtime runs.
    pub drafter: Drafting,
    /// Conv state can be restored at page boundaries (partial prefix reuse).
    pub conv_checkpoints: bool,
    /// Rolls a slot back to its checkpoint: the length after the slot's last committed prefill segment that samples
    /// (a prompt's end). A new request that continues a parked sequence's slot at that position below the slot's
    /// length makes the runtime roll the slot back first (prefix reuse across turns with the runtime's own pages).
    pub slot_rollback: bool,
    /// Token ids the model can take and produce: `0..vocab` (prompt ids at or above it are refused by the front end).
    pub vocab: u32,
    /// KV pages of `page_tokens` the runtime owns itself, when it keeps its own page tables (`external_pages` false):
    /// the scheduler's pool is narrowed to them. `None`: the runtime takes the scheduler's pool as configured.
    pub kv_pages: Option<u32>,
}

impl Default for RuntimeCaps {
    /// `engine/native` limits (README "API", `EngineConfig` defaults), its vocabulary (200,058 valid logits), no
    /// optional capability and no own page count.
    fn default() -> Self {
        Self {
            max_slots: 8,
            t_max: 2112,
            page_tokens: 1024,
            max_decode_rows: 8,
            max_prefill_rows: 2048,
            external_pages: false,
            drafter: Drafting::None,
            conv_checkpoints: false,
            slot_rollback: false,
            vocab: 200_058,
            kv_pages: None,
        }
    }
}

/// A step the runtime could not run; the message names the cause.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StepError {
    /// The step's requests cannot run; the runtime is usable and later steps can succeed.
    #[error("{0}")]
    Request(String),
    /// The runtime is unusable (a device failure): no later step can be trusted.
    #[error("{0}")]
    Device(String),
}

/// A model that runs the scheduler's plans, one call per step.
pub trait ModelRuntime {
    /// Limits and capabilities.
    fn caps(&self) -> RuntimeCaps;

    /// Runs one step (forward, drafter, sampling, speculative accept) and returns one [`SegResult`] per sampling
    /// segment.
    ///
    /// # Errors
    ///
    /// Returns [`StepError::Request`] when the step cannot run; the back end then fails the requests of that step.
    /// Returns [`StepError::Device`] when the runtime is unusable; the back end then fails every request and stops.
    fn step(&mut self, plan: &StepPlan) -> Result<Vec<SegResult>, StepError>;

    /// Milliseconds since start: virtual for a fake runtime, steady clock otherwise.
    fn now_ms(&self) -> f64;
}
