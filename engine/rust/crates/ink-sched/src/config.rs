// Ported from engine/server/src/sched/scheduler.h @ 4171493 (sched::Config).
//! Scheduler launch parameters. Defaults and their sources: this crate's `SCHEDULER.md` "Policy and defaults".

use std::fmt;
use std::sync::Arc;

use crate::runtime::Drafting;

/// Order among prompts waiting for their next prefill chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillOrder {
    /// Priority, then arrival.
    #[default]
    Fifo,
    /// Shortest remaining prompt first.
    Srpt,
}

/// How much resident KV a new request may reuse (`SCHEDULER.md` "Prefix reuse").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefixReuse {
    /// Exact resume of a parked sequence, adoption of a parked sequence at a page boundary, and full pages shared
    /// with any resident sequence of the same salt. Needs a runtime with external pages and conv checkpoints.
    #[default]
    Shared,
    /// Only the exact resume of a parked sequence that is a prefix of the prompt, in its own slot with its ring
    /// still valid: no page is shared and no slot is reset at a position above 0. What a runtime with its own page
    /// tables can do; [`crate::EngineBackend`] selects it for such a runtime.
    ExactResume,
    /// [`PrefixReuse::ExactResume`], or a parked sequence whose tokens up to its checkpoint (the end of its last
    /// prompt) are a prefix of the prompt is taken over in its own slot, rolled back to the checkpoint: the prompt
    /// is prefilled from there. What a runtime with its own page tables and `slot_rollback` can do;
    /// [`crate::EngineBackend`] selects it for such a runtime.
    SlotRollback,
}

/// Verify-width hook: `(sequence id, default k) -> k`, clamped by the scheduler to `[0, spec_k]`.
#[derive(Clone)]
pub struct WidthHook(pub Arc<dyn Fn(u64, u32) -> u32 + Send + Sync>);

impl fmt::Debug for WidthHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WidthHook(..)")
    }
}

/// Scheduler launch parameters (one scheduler per card).
#[derive(Debug, Clone)]
pub struct Config {
    /// Concurrent generating sessions (owner target 4; 8 qualified too).
    pub max_running: u32,
    /// Runtime slots (SWA rings + conv state); parked sessions keep one while it is free.
    pub slots: u32,
    /// KV bytes per token (lossless: 36 KiB, PLAN 5.1).
    pub kv_bytes_per_token: u64,
    /// KV pool size in tokens (about 1.14M per card); rounded down to whole pages.
    pub pool_tokens: u64,
    /// KV pool size in bytes; when above 0 it wins over `pool_tokens`.
    pub pool_bytes: u64,
    /// Tokens per KV page (= `engine/native` global page, 1,024 keys).
    pub page_tokens: u32,
    /// Per-session cap (prompt + output), at most 1,048,576 (the model's maximum context).
    pub session_cap_tokens: u64,
    /// Rows per forward call.
    pub t_max: u32,
    /// Rows per decode/verify segment.
    pub max_decode_rows: u32,
    /// Rows per prefill segment.
    pub max_prefill_rows: u32,
    /// Drafts per verify row group (width = 1 + k).
    pub spec_k: u32,
    /// Confidence-gated width hook; `None` is static k (P2/P10: static k = 7 is best at 45K, n = 4).
    pub width_hook: Option<WidthHook>,
    /// The verify rows the runtime runs ([`crate::EngineBackend`] takes it from the runtime's caps).
    pub drafting: Drafting,
    /// A parked sequence returns the pages past its computed tokens (rows of rejected drafts) to the pool. Off for a
    /// runtime with its own page tables, which cannot free part of a slot's pages: the scheduler then keeps counting
    /// them until the slot is reset ([`crate::EngineBackend`] sets it from the caps).
    pub trim_parked_pages: bool,
    /// Prefill cost added to a step while at least one session decodes, in ms (`f64::INFINITY`: no stall cap,
    /// always use `max_chunk`; measured winner on GPU 6/7 agent replay, `bench/raw/2026-11-24-chunk-policy/`).
    pub stall_budget_ms: f64,
    /// Smallest prefill chunk (progress guarantee).
    pub min_chunk: u32,
    /// Largest prefill chunk (capped at `max_prefill_rows`).
    pub max_chunk: u32,
    /// TTFT target in ms (owner target, TTFT p99 at 4 sessions).
    pub ttft_target_ms: f64,
    /// Escalate the chunk when a queued prefill's projected TTFT exceeds this fraction of the target.
    pub urgency_frac: f64,
    /// Escalated chunks still cost at most this per step, in ms.
    pub urgent_stall_cap_ms: f64,
    /// Order among prefills.
    pub prefill_order: PrefillOrder,
    /// Step-cost estimate: fixed cost per step, ms (P10 README:41 intercept).
    pub cost_base_ms: f64,
    /// Step-cost estimate: per verified row, ms (P10 README:41).
    pub cost_row_ms: f64,
    /// Step-cost estimate: per new prefill token excluding global attention, ms (P9 README:96-97).
    pub cost_prefill_tok_ms: f64,
    /// Step-cost estimate: per new token per 1,000 cached keys, ms (P9 README:97-98).
    pub cost_attn_ms_per_tok_kkey: f64,
    /// Keep a finished session's KV resident (prefix cache).
    pub park_on_finish: bool,
    /// Keep a cancelled session's KV resident.
    pub park_on_cancel: bool,
    /// What resident KV a new request may reuse.
    pub prefix_reuse: PrefixReuse,
    /// Partial reuse recomputes this window (the SWA ring) from a page-boundary checkpoint.
    pub swa_recompute_tokens: u32,
    /// Eviction score = age_ms + weight x pages (0 = pure LRU).
    pub evict_size_weight: f64,
    /// Grace window (ms) protecting freshly-parked multi-turn sessions in `acquire_slot`:
    /// stale parked slots (age >= grace) are stolen first by LRU; when all stealable slots are
    /// within the grace window, the smallest `num_computed` slot is stolen first to protect long prefixes.
    pub slot_steal_grace_ms: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_running: 4,
            slots: 8,
            kv_bytes_per_token: 36 * 1024,
            pool_tokens: 1_140_000,
            pool_bytes: 0,
            page_tokens: 1024,
            session_cap_tokens: 262_144,
            t_max: 2112,
            max_decode_rows: 8,
            max_prefill_rows: 2048,
            spec_k: 7,
            width_hook: None,
            drafting: Drafting::Any,
            trim_parked_pages: true,
            stall_budget_ms: f64::INFINITY,
            min_chunk: 128,
            max_chunk: 2048,
            ttft_target_ms: 5000.0,
            urgency_frac: 0.8,
            urgent_stall_cap_ms: 300.0,
            prefill_order: PrefillOrder::Fifo,
            cost_base_ms: 25.4,
            cost_row_ms: 0.617,
            cost_prefill_tok_ms: 0.171,
            cost_attn_ms_per_tok_kkey: 0.00245,
            park_on_finish: true,
            park_on_cancel: false,
            prefix_reuse: PrefixReuse::Shared,
            swa_recompute_tokens: 1024,
            evict_size_weight: 0.0,
            slot_steal_grace_ms: 150.0,
        }
    }
}
