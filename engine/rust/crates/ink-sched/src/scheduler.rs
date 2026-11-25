// Ported from engine/server/src/sched/scheduler.{h,cpp} @ 4171493.
//! The step scheduler: admission, decode rows, one prefill chunk per step, and the KV page accounting.

use std::collections::{BTreeMap, BTreeSet};

use ink_backend::{BackendMetrics, GenRequest, SamplingParams};

use crate::config::{Config, PrefillOrder, PrefixReuse};
use crate::plan::{Finish, SegKind, SegResult, Segment, SeqEvent, StepPlan, Stop};
use crate::pool::PagePool;
use crate::runtime::Drafting;

/// Largest session cap: the model's maximum context.
const MAX_SESSION_CAP: u64 = 1_048_576;

/// Escalated chunks grow in steps of this many tokens.
const URGENT_CHUNK_STEP: usize = 64;

/// A configuration the scheduler refuses.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// `page_tokens` is 0.
    #[error("page_tokens must be > 0")]
    PageTokens,
    /// `session_cap_tokens` is outside `[1, 1048576]`.
    #[error("session_cap_tokens must be in [1, 1048576] (the model's maximum context)")]
    SessionCap,
    /// `max_running` is outside `[1, slots]`.
    #[error("max_running must be in [1, slots]")]
    MaxRunning,
    /// `pool_bytes` is set while `kv_bytes_per_token` is 0.
    #[error("kv_bytes_per_token must be > 0")]
    KvBytesPerToken,
    /// The pool holds no whole page.
    #[error("KV pool smaller than one page")]
    PoolTooSmall,
    /// The pool has more pages than page ids can name.
    #[error("KV pool larger than 2^32 pages")]
    PoolTooLarge,
    /// The session cap is larger than the pool: a lone session could need more pages than exist and would be
    /// preempted and recomputed forever (vLLM refuses the same configuration at start-up).
    #[error("session_cap_tokens must not exceed the KV pool ({pool} tokens)")]
    SessionCapOverPool {
        /// Pool size in tokens, rounded down to whole pages.
        pool: u64,
    },
    /// A step row limit is 0, or a decode segment could not fit in a step (nothing would ever be scheduled).
    #[error("t_max, max_decode_rows and max_prefill_rows must be > 0, and max_decode_rows <= t_max")]
    Rows,
    /// A fixed-width drafter drafts `k` tokens: `spec_k` must be 0 (no speculation) or `k`, within `max_decode_rows`.
    #[error(
        "with a drafter that always drafts {k} tokens, spec_k must be 0 or {k} and max_decode_rows at least {k} + 1"
    )]
    FixedWidth {
        /// Drafts per round of the runtime's drafter.
        k: u32,
    },
}

/// Why a request was rejected at [`Scheduler::add`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AddError {
    /// The prompt has no token.
    #[error("empty prompt")]
    EmptyPrompt,
    /// Prompt plus one generated token is over the per-session cap.
    #[error("prompt of {prompt} tokens exceeds the per-session cap of {cap} tokens")]
    OverSessionCap {
        /// Prompt tokens.
        prompt: usize,
        /// Session cap in tokens.
        cap: u64,
    },
    /// Prompt plus one generated token needs more pages than the pool has.
    #[error("prompt of {prompt} tokens exceeds the KV pool of {pool} tokens")]
    OverPool {
        /// Prompt tokens.
        prompt: usize,
        /// Pool size in tokens.
        pool: u64,
    },
    /// A request with this id is already known.
    #[error("duplicate request id")]
    DuplicateId,
}

/// The admission limits of a scheduler that do not change while it runs (session cap, KV pool): a prompt over them
/// is refused by [`Scheduler::add`] whatever the load, so a front end can refuse it before submitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptLimits {
    session_cap: u64,
    page_tokens: usize,
    pages: usize,
    pool_tokens: u64,
}

impl PromptLimits {
    /// Whether a prompt of `tokens` tokens can be admitted.
    ///
    /// # Errors
    ///
    /// [`AddError::EmptyPrompt`], [`AddError::OverSessionCap`] or [`AddError::OverPool`].
    pub fn check(&self, tokens: usize) -> Result<(), AddError> {
        if tokens == 0 {
            Err(AddError::EmptyPrompt)
        } else if tokens as u64 + 1 > self.session_cap {
            Err(AddError::OverSessionCap {
                prompt: tokens,
                cap: self.session_cap,
            })
        } else if ceil_div(tokens + 1, self.page_tokens) > self.pages {
            Err(AddError::OverPool {
                prompt: tokens,
                pool: self.pool_tokens,
            })
        } else {
            Ok(())
        }
    }
}

/// The page accounting is inconsistent (refcounts, owners, free list or slots); the message says where.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InvariantError(String);

/// Scheduler counters since start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counters {
    /// Prompt tokens looked up in resident KV at first admission.
    pub prefix_queries: u64,
    /// Prompt tokens found in resident KV at first admission.
    pub prefix_hits: u64,
    /// Prompt tokens admitted.
    pub prompt_tokens: u64,
    /// Tokens generated.
    pub generation_tokens: u64,
    /// Draft rows planned.
    pub spec_draft: u64,
    /// Draft rows accepted.
    pub spec_accepted: u64,
    /// Running requests sent back to the queue to free pages.
    pub preemptions: u64,
    /// Parked sequences dropped to free pages.
    pub evictions: u64,
    /// Requests rejected at `add`.
    pub rejections: u64,
    /// Steps committed.
    pub steps: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum St {
    Waiting,
    Running,
    Parked,
}

#[derive(Debug)]
struct Seq {
    id: u64,
    session: String,
    salt: String,
    priority: i32,
    /// Arrival order.
    order: u64,
    arrival_ms: f64,
    last_use_ms: f64,
    /// Prompt, then generated tokens.
    tokens: Vec<i32>,
    prompt_len: usize,
    /// Tokens whose KV is resident.
    num_computed: usize,
    max_new: usize,
    generated: usize,
    pages: Vec<u32>,
    slot: Option<usize>,
    /// The slot's ring and conv state match `tokens[..num_computed]`.
    ring_valid: bool,
    /// Position of the runtime slot's checkpoint (the end of this sequence's last prefill segment that sampled, or
    /// of the one it took over with the slot): the slot can be rolled back there.
    checkpoint: Option<usize>,
    reset_slot_pending: bool,
    st: St,
    admitted_once: bool,
    first_event_sent: bool,
    cached: usize,
    sampling: SamplingParams,
    stop_ids: Vec<i32>,
    min_new: usize,
    min_masked: Vec<i32>,
}

impl Seq {
    fn remaining(&self) -> usize {
        self.tokens.len() - self.num_computed
    }

    /// Back to the queue with nothing resident (its pages and slot are already released).
    fn reset_to_waiting(&mut self) {
        self.num_computed = 0;
        self.cached = 0;
        self.reset_slot_pending = true;
        self.st = St::Waiting;
    }
}

/// Best resident matches for a prompt, found by [`Scheduler::reuse_estimate`].
#[derive(Debug, Default)]
struct Reuse {
    /// Tokens the prompt would get from resident KV.
    best: usize,
    /// A parked sequence that is a prefix of the prompt, and its length.
    adopt: Option<(u64, usize)>,
    /// The longest other resident match, and its length.
    share: Option<(u64, usize)>,
    /// A parked sequence whose slot can be rolled back to a checkpoint inside the match, and that position.
    rollback: Option<(u64, usize)>,
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

fn to_u32(x: usize) -> u32 {
    u32::try_from(x).expect("token counts are bounded by the session cap")
}

/// Length of the common prefix of `a` and `b`.
fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn release_slot(slot_owner: &mut [Option<u64>], s: &mut Seq) {
    if let Some(slot) = s.slot.take() {
        slot_owner[slot] = None;
    }
    s.ring_valid = false;
    s.checkpoint = None;
}

/// Step scheduler of one card.
///
/// Each step it admits waiting requests up to `max_running` (prefix reuse against resident KV), puts every decoding
/// session in the batch with its verify width, adds at most one prefill chunk sized by the interleave rule, and keeps
/// the page pool accounting (refcounted pages, parking, eviction, preemption). The plan goes to a
/// [`crate::ModelRuntime`]; its results come back through [`Scheduler::commit`].
#[derive(Debug)]
pub struct Scheduler {
    cfg: Config,
    seqs: BTreeMap<u64, Seq>,
    pool: PagePool,
    slot_owner: Vec<Option<u64>>,
    order: u64,
    ctr: Counters,
    pending_events: Vec<SeqEvent>,
    /// Sequences with a segment in the plan being built (never preempted).
    in_plan: BTreeSet<u64>,
    now_cache: f64,
}

impl Scheduler {
    /// Validates and normalises `cfg` (chunk bounds, `spec_k`, pool rounded to whole pages).
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] for a zero page size, a session cap outside `[1, 1048576]`, `max_running` outside
    /// `[1, slots]`, a zero row limit or `max_decode_rows > t_max`, a pool without a whole page, a session cap larger
    /// than the pool, or a `spec_k` a fixed-width drafter cannot run.
    pub fn new(mut cfg: Config) -> Result<Self, ConfigError> {
        if cfg.page_tokens == 0 {
            return Err(ConfigError::PageTokens);
        }
        if cfg.session_cap_tokens == 0 || cfg.session_cap_tokens > MAX_SESSION_CAP {
            return Err(ConfigError::SessionCap);
        }
        if cfg.max_running == 0 || cfg.max_running > cfg.slots {
            return Err(ConfigError::MaxRunning);
        }
        if cfg.t_max == 0 || cfg.max_decode_rows == 0 || cfg.max_prefill_rows == 0 || cfg.max_decode_rows > cfg.t_max {
            return Err(ConfigError::Rows);
        }
        cfg.max_chunk = cfg.max_chunk.min(cfg.max_prefill_rows);
        cfg.min_chunk = cfg.min_chunk.min(cfg.max_chunk).max(1);
        cfg.spec_k = cfg.spec_k.min(cfg.max_decode_rows.saturating_sub(1));
        if let Drafting::Fixed { k, .. } = cfg.drafting
            && cfg.spec_k != 0
            && cfg.spec_k != k
        {
            return Err(ConfigError::FixedWidth { k });
        }
        let pool_tokens = if cfg.pool_bytes > 0 {
            if cfg.kv_bytes_per_token == 0 {
                return Err(ConfigError::KvBytesPerToken);
            }
            cfg.pool_bytes / cfg.kv_bytes_per_token
        } else {
            cfg.pool_tokens
        };
        let pages = pool_tokens / u64::from(cfg.page_tokens);
        if pages == 0 {
            return Err(ConfigError::PoolTooSmall);
        }
        cfg.pool_tokens = pages * u64::from(cfg.page_tokens);
        if cfg.session_cap_tokens > cfg.pool_tokens {
            return Err(ConfigError::SessionCapOverPool { pool: cfg.pool_tokens });
        }
        let pages = u32::try_from(pages).map_err(|_| ConfigError::PoolTooLarge)?;
        let slots = cfg.slots as usize;
        Ok(Self {
            cfg,
            seqs: BTreeMap::new(),
            pool: PagePool::new(pages),
            slot_owner: vec![None; slots],
            order: 0,
            ctr: Counters::default(),
            pending_events: Vec::new(),
            in_plan: BTreeSet::new(),
            now_cache: 0.0,
        })
    }

    fn page_tokens(&self) -> usize {
        self.cfg.page_tokens as usize
    }

    /// Estimated cost in ms of a prefill chunk of `chunk` tokens on `ctx` cached tokens.
    #[must_use]
    pub fn prefill_cost_ms(&self, chunk: usize, ctx: usize) -> f64 {
        let keys_k = (ctx as f64 + chunk as f64 / 2.0) / 1000.0;
        chunk as f64 * (self.cfg.cost_prefill_tok_ms + self.cfg.cost_attn_ms_per_tok_kkey * keys_k)
    }

    /// Estimated cost in ms of a step with `rows` decode rows.
    #[must_use]
    pub fn decode_cost_ms(&self, rows: usize) -> f64 {
        self.cfg.cost_base_ms + self.cfg.cost_row_ms * rows as f64
    }

    /// The interleave rule: chunk for a prefill at context `ctx` while `decode_rows` rows decode.
    ///
    /// `queue` holds `(elapsed_ms, cumulative remaining tokens)` of every queued prefill in order; it drives the
    /// urgency override (`SCHEDULER.md` "Interleave rule").
    #[must_use]
    pub fn choose_chunk(&self, decode_rows: usize, ctx: usize, queue: &[(f64, usize)]) -> usize {
        let c = &self.cfg;
        let maxc = (c.max_chunk.min(c.max_prefill_rows) as usize)
            .min((c.t_max as usize).saturating_sub(decode_rows))
            .max(1);
        if decode_rows == 0 || c.stall_budget_ms == f64::INFINITY {
            return maxc; // nobody to stall, or stall budget disabled: largest chunk (best prefill efficiency)
        }
        // 1) budget chunk: the largest chunk whose cost stays within the per-step stall budget, at least min_chunk
        let min_c = (c.min_chunk as usize).min(maxc);
        let (mut lo, mut hi, mut chunk) = (min_c, maxc, min_c);
        while lo <= hi {
            let mid = (lo + hi) / 2;
            if self.prefill_cost_ms(mid, ctx) <= c.stall_budget_ms {
                chunk = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        chunk = chunk.max(min_c);
        // 2) urgency override: projected TTFT of every queued prefill (cumulative work ahead of it)
        let limit = c.urgency_frac * c.ttft_target_ms;
        let worst = |cc: usize| {
            let step = self.decode_cost_ms(decode_rows) + self.prefill_cost_ms(cc, ctx);
            queue.iter().fold(0.0_f64, |w, &(elapsed, cum)| {
                w.max(elapsed + ceil_div(cum, cc) as f64 * step)
            })
        };
        if queue.is_empty() || worst(chunk) <= limit {
            return chunk;
        }
        // escalate, but never past the urgent stall cap (a hopeless TTFT must not stall the decoders without bound)
        let mut cap_c = chunk;
        for cc in (chunk..=maxc).step_by(URGENT_CHUNK_STEP) {
            if self.prefill_cost_ms(cc, ctx) > c.urgent_stall_cap_ms {
                break;
            }
            cap_c = cc;
            if worst(cc) <= limit {
                return cc;
            }
        }
        cap_c
    }

    /// Rows of the sequence's next decode segment: 1 + the drafts it verifies.
    fn width_for(&self, s: &Seq) -> usize {
        let c = &self.cfg;
        let room = (c.session_cap_tokens as usize).saturating_sub(s.num_computed + 1);
        match c.drafting {
            Drafting::None => 1,
            Drafting::Any => {
                let k = c.width_hook.as_ref().map_or(c.spec_k, |hook| (hook.0)(s.id, c.spec_k));
                let k = k.min(c.spec_k.min(c.max_decode_rows.saturating_sub(1))) as usize;
                let k = k.min(s.max_new.saturating_sub(s.generated + 1)); // no drafts past the token budget
                1 + k.min(room)
            }
            // all k drafts or none: drafts past the token budget are verified and dropped by the front end
            Drafting::Fixed { ctx, sampled, .. } => {
                let k = c.spec_k as usize;
                let speculates = k > 0
                    && (sampled || s.sampling.is_greedy())
                    && s.sampling.num_logprobs.is_none()
                    && k <= room
                    && s.num_computed + 1 + k <= ctx as usize;
                if speculates { 1 + k } else { 1 }
            }
        }
    }

    fn count(&self, st: St) -> usize {
        self.seqs.values().filter(|s| s.st == st).count()
    }

    /// Requests generating now.
    #[must_use]
    pub fn running(&self) -> usize {
        self.count(St::Running)
    }

    /// Requests waiting for admission (new or preempted).
    #[must_use]
    pub fn waiting(&self) -> usize {
        self.count(St::Waiting)
    }

    /// Finished sequences whose KV stays resident for prefix reuse.
    #[must_use]
    pub fn parked(&self) -> usize {
        self.count(St::Parked)
    }

    /// Pages in the pool.
    #[must_use]
    pub fn pages_total(&self) -> usize {
        self.pool.total()
    }

    /// Free pages.
    #[must_use]
    pub fn pages_free(&self) -> usize {
        self.pool.free_count()
    }

    /// Pages owned by at least one sequence.
    #[must_use]
    pub fn pages_used(&self) -> usize {
        self.pages_total() - self.pages_free()
    }

    /// Counters since start.
    #[must_use]
    pub fn counters(&self) -> &Counters {
        &self.ctr
    }

    /// The normalised configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Gauges and counters for `/metrics`.
    #[must_use]
    pub fn metrics(&self) -> BackendMetrics {
        let gauge = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        BackendMetrics {
            running: gauge(self.running()),
            waiting: gauge(self.waiting()),
            parked: gauge(self.parked()),
            kv_pages_used: self.pages_used() as u64,
            kv_pages_total: self.pages_total() as u64,
            prefix_cache_queries: self.ctr.prefix_queries,
            prefix_cache_hits: self.ctr.prefix_hits,
            prompt_tokens: self.ctr.prompt_tokens,
            generation_tokens: self.ctr.generation_tokens,
            spec_draft_tokens: self.ctr.spec_draft,
            spec_accepted_tokens: self.ctr.spec_accepted,
        }
    }

    /// The admission limits [`Scheduler::add`] checks every prompt against.
    #[must_use]
    pub fn prompt_limits(&self) -> PromptLimits {
        PromptLimits {
            session_cap: self.cfg.session_cap_tokens,
            page_tokens: self.page_tokens(),
            pages: self.pages_total(),
            pool_tokens: self.cfg.pool_tokens,
        }
    }

    /// Queues a request.
    ///
    /// # Errors
    ///
    /// Returns [`AddError`] for an empty prompt, a prompt over the session cap or the pool, or a known id.
    pub fn add(&mut self, r: GenRequest, now_ms: f64) -> Result<(), AddError> {
        let n = r.prompt.len();
        let err = match self.prompt_limits().check(n) {
            Err(e) => Some(e),
            Ok(()) if self.seqs.contains_key(&r.id) => Some(AddError::DuplicateId),
            Ok(()) => None,
        };
        if let Some(e) = err {
            self.ctr.rejections += 1;
            return Err(e);
        }
        self.order += 1;
        let max_new = (r.max_new_tokens.max(1) as usize).min(self.cfg.session_cap_tokens as usize - n);
        let s = Seq {
            id: r.id,
            session: r.session,
            salt: r.cache_salt,
            priority: r.priority,
            order: self.order,
            arrival_ms: now_ms,
            last_use_ms: now_ms,
            tokens: r.prompt,
            prompt_len: n,
            num_computed: 0,
            max_new,
            generated: 0,
            pages: Vec::new(),
            slot: None,
            ring_valid: false,
            checkpoint: None,
            reset_slot_pending: false,
            st: St::Waiting,
            admitted_once: false,
            first_event_sent: false,
            cached: 0,
            sampling: r.sampling,
            stop_ids: r.stop_token_ids,
            min_new: r.min_new_tokens as usize,
            min_masked: r.min_tokens_masked,
        };
        self.seqs.insert(s.id, s);
        Ok(())
    }

    fn acquire_slot(&mut self, protect: u64, now_ms: f64) -> Option<usize> {
        if let Some(free) = self.slot_owner.iter().position(Option::is_none) {
            return Some(free);
        }
        // Steal a parked session's slot (its ring becomes stale; its pages stay).
        // Prefer stale parked sessions (age >= slot_steal_grace_ms) by oldest last_use_ms, breaking ties by
        // smallest num_computed. If all parked sessions are freshly parked (within the grace window, e.g. two
        // workers finishing on the same or adjacent decode steps at N == slots), steal the smallest num_computed
        // first so a short/finished conversation is evicted instead of a 70K+ live multi-turn prefix.
        let grace = self.cfg.slot_steal_grace_ms.max(0.0);
        let better = |cand: &Seq, best: &Seq| -> bool {
            let c_stale = now_ms - cand.last_use_ms >= grace;
            let b_stale = now_ms - best.last_use_ms >= grace;
            match (c_stale, b_stale) {
                (true, false) => true,
                (false, true) => false,
                (true, true) => {
                    cand.last_use_ms < best.last_use_ms
                        || (cand.last_use_ms == best.last_use_ms && cand.num_computed < best.num_computed)
                }
                (false, false) => {
                    cand.num_computed < best.num_computed
                        || (cand.num_computed == best.num_computed && cand.last_use_ms < best.last_use_ms)
                }
            }
        };
        let mut victim: Option<&Seq> = None;
        for (&id, s) in &self.seqs {
            if s.st == St::Parked && s.slot.is_some() && id != protect && victim.is_none_or(|v| better(s, v)) {
                victim = Some(s);
            }
        }
        let vid = victim?.id;
        let v = self.seqs.get_mut(&vid).expect("victim id taken from the map");
        let slot = v.slot;
        release_slot(&mut self.slot_owner, v);
        slot
    }

    fn evict_one(&mut self, protect: u64) -> bool {
        let mut victim = None;
        let mut best = -1.0_f64;
        for (&id, s) in &self.seqs {
            if s.st != St::Parked || id == protect {
                continue;
            }
            let score = (self.now_cache - s.last_use_ms) + self.cfg.evict_size_weight * s.pages.len() as f64;
            if score > best {
                best = score;
                victim = Some(id);
            }
        }
        let Some(vid) = victim else { return false };
        let mut v = self.seqs.remove(&vid).expect("victim id taken from the map");
        self.pool.release(&mut v.pages, 0);
        release_slot(&mut self.slot_owner, &mut v);
        self.ctr.evictions += 1;
        true
    }

    fn alloc_pages(&mut self, n: usize, protect: u64) -> bool {
        while self.pool.free_count() < n {
            if !self.evict_one(protect) {
                return false;
            }
        }
        true
    }

    fn ensure_pages(&mut self, id: u64, tokens_needed: usize, protect: u64) -> bool {
        let have = self.seqs[&id].pages.len();
        let need = ceil_div(tokens_needed, self.page_tokens()).saturating_sub(have);
        if need == 0 {
            return true;
        }
        if !self.alloc_pages(need, protect) {
            return false;
        }
        let s = self.seqs.get_mut(&id).expect("sequence with a page request exists");
        for _ in 0..need {
            s.pages.push(self.pool.take());
        }
        true
    }

    fn aligned_reuse(&self, m: usize) -> usize {
        let pg = self.page_tokens();
        m.saturating_sub(self.cfg.swa_recompute_tokens as usize) / pg * pg
    }

    /// Tokens a prompt would get from resident KV (exact resume, adoption or full-page sharing; with
    /// [`PrefixReuse::ExactResume`] the exact resume only).
    fn reuse_estimate(&self, s: &Seq) -> Reuse {
        let limit = s.tokens.len() - 1; // the last token is always computed
        let exact_only = self.cfg.prefix_reuse != PrefixReuse::Shared;
        let rollback_ok = self.cfg.prefix_reuse == PrefixReuse::SlotRollback;
        let mut adopt: Option<(u64, usize)> = None;
        let mut share: Option<(u64, usize)> = None;
        let mut rollback: Option<(u64, usize)> = None;
        for (&id, q) in &self.seqs {
            if id == s.id || q.salt != s.salt || q.num_computed == 0 || q.st == St::Waiting {
                continue;
            }
            let m = common_prefix(&q.tokens[..q.num_computed], &s.tokens).min(limit);
            // a parked sequence that is a prefix of this prompt is taken over (moved); any other match shares pages
            // a parked sequence that is a prefix of the prompt is resumed whole, its reply's decode-row K/V included
            // (owner, 2026-11-24: not bit-equal to a cold prefill of those tokens, KL <= 1e-5 gated by ink-inv)
            let adoptable =
                q.st == St::Parked && m == q.num_computed && (!exact_only || (q.ring_valid && q.slot.is_some()));
            if adoptable && m > adopt.map_or(0, |a| a.1) {
                adopt = Some((id, m));
            } else if !exact_only && m > share.map_or(0, |a| a.1) {
                share = Some((id, m));
            }
            if rollback_ok
                && q.st == St::Parked
                && q.ring_valid
                && q.slot.is_some()
                && let Some(c) = q.checkpoint.filter(|&c| c <= m && c < q.num_computed)
                && c > rollback.map_or(0, |r| r.1)
            {
                rollback = Some((id, c));
            }
        }
        let mut best = self.aligned_reuse(share.map_or(0, |a| a.1));
        if let Some((aid, adopt_m)) = adopt {
            let a = &self.seqs[&aid];
            let usable = if a.ring_valid && a.slot.is_some() {
                adopt_m
            } else {
                self.aligned_reuse(adopt_m)
            };
            best = best.max(usable);
        }
        best = best.max(rollback.map_or(0, |r| r.1));
        Reuse {
            best,
            adopt,
            share,
            rollback,
        }
    }

    fn prefix_reuse(&mut self, id: u64) {
        let reuse = self.reuse_estimate(&self.seqs[&id]);
        let pg = self.page_tokens();
        let share_m = reuse.share.map_or(0, |a| a.1);
        if let Some((rid, c)) = reuse.rollback
            && c == reuse.best
            && reuse.adopt.is_none_or(|a| a.1 < c)
        {
            // take over the parked sequence's slot rolled back to its checkpoint: pages, slot, ring and conv state
            // up to `c` carry over, the runtime frees the rest
            let mut p = self.seqs.remove(&rid).expect("rollback id taken from the map");
            let slot = p.slot.take().expect("a rollback candidate holds its slot");
            let s = self.seqs.get_mut(&id).expect("admitted sequence exists");
            s.pages = std::mem::take(&mut p.pages);
            self.pool.release(&mut s.pages, ceil_div(c, pg));
            s.slot = Some(slot);
            self.slot_owner[slot] = Some(id);
            s.ring_valid = true;
            s.checkpoint = Some(c);
            s.num_computed = c;
            s.cached = c;
            return;
        }
        if let Some((aid, adopt_m)) = reuse.adopt {
            let p = &self.seqs[&aid];
            let exact = p.ring_valid && p.slot.is_some();
            let usable = if exact { adopt_m } else { self.aligned_reuse(adopt_m) };
            if usable > 0 && usable >= self.aligned_reuse(share_m) {
                let mut p = self.seqs.remove(&aid).expect("adopted id taken from the map");
                let s = self.seqs.get_mut(&id).expect("admitted sequence exists");
                s.pages = std::mem::take(&mut p.pages);
                if exact {
                    // exact resume: pages, slot, ring and conv state carry over
                    let slot = p.slot.take().expect("exact resume has a slot");
                    s.slot = Some(slot);
                    self.slot_owner[slot] = Some(id);
                    s.ring_valid = true;
                    s.checkpoint = p.checkpoint;
                } else {
                    release_slot(&mut self.slot_owner, &mut p);
                    self.pool.release(&mut s.pages, usable / pg);
                    s.reset_slot_pending = true;
                }
                s.num_computed = usable;
                s.cached = usable;
                return;
            }
        }
        if let Some((sid, _)) = reuse.share {
            let usable = self.aligned_reuse(share_m);
            if usable > 0 {
                let shared: Vec<u32> = self.seqs[&sid].pages[..usable / pg].to_vec();
                for &page in &shared {
                    self.pool.share(page);
                }
                let s = self.seqs.get_mut(&id).expect("admitted sequence exists");
                s.pages.extend_from_slice(&shared);
                s.num_computed = usable;
                s.cached = usable;
                s.reset_slot_pending = true;
            }
        }
    }

    fn admit_one(&mut self, id: u64, now_ms: f64) -> bool {
        // capacity check before any reuse: a free (or stealable) slot and pages for the first chunk
        let free_slots = self.slot_owner.iter().filter(|o| o.is_none()).count();
        let stealable = self
            .seqs
            .values()
            .filter(|q| q.st == St::Parked && q.slot.is_some())
            .count();
        if free_slots + stealable == 0 {
            return false;
        }
        // pages that evicting every parked sequence would free: all their owners are parked (a page shared only among
        // parked sequences counts; the C++ counted refcount-1 pages only and could block admission forever)
        let mut parked_refs = vec![0_u32; self.pool.total()];
        for q in self.seqs.values().filter(|q| q.st == St::Parked) {
            for &p in &q.pages {
                parked_refs[p as usize] += 1;
            }
        }
        let evictable = parked_refs
            .iter()
            .zip(self.pool.refcounts())
            .filter(|&(&n, &rc)| n > 0 && n == rc)
            .count();
        let first = (self.cfg.max_chunk as usize).min(self.seqs[&id].tokens.len());
        if self.pool.free_count() + evictable < ceil_div(first, self.page_tokens()) {
            return false;
        }

        let s = &self.seqs[&id];
        let first_admission = !s.admitted_once;
        if s.num_computed == 0 {
            self.prefix_reuse(id);
        }
        if self.seqs[&id].slot.is_none() {
            let Some(slot) = self.acquire_slot(id, now_ms) else {
                return false; // cannot happen after the check above
            };
            self.slot_owner[slot] = Some(id);
            let s = self.seqs.get_mut(&id).expect("admitted sequence exists");
            s.slot = Some(slot);
            s.reset_slot_pending = true;
            s.ring_valid = false;
            s.checkpoint = None;
        }
        let s = self.seqs.get_mut(&id).expect("admitted sequence exists");
        if first_admission {
            self.ctr.prefix_queries += s.tokens.len() as u64;
            self.ctr.prefix_hits += s.cached as u64;
            self.ctr.prompt_tokens += s.prompt_len as u64;
        }
        s.admitted_once = true;
        s.st = St::Running;
        s.last_use_ms = now_ms;
        true
    }

    /// Frees all pages and the slot of `id` and sends it back to the queue.
    fn preempt(&mut self, id: u64) {
        let s = self.seqs.get_mut(&id).expect("preempted sequence exists");
        self.pool.release(&mut s.pages, 0);
        release_slot(&mut self.slot_owner, s);
        s.reset_to_waiting();
        self.ctr.preemptions += 1;
    }

    /// Preempts the running request with the lowest priority, youngest first, that has no segment in this plan.
    fn preempt_one(&mut self) -> bool {
        let mut victim: Option<&Seq> = None;
        for (id, s) in &self.seqs {
            if s.st != St::Running || self.in_plan.contains(id) {
                continue;
            }
            if victim.is_none_or(|v| s.priority < v.priority || (s.priority == v.priority && s.order > v.order)) {
                victim = Some(s);
            }
        }
        let Some(vid) = victim.map(|v| v.id) else { return false };
        self.preempt(vid);
        true
    }

    /// Makes `id` own pages for `tokens_needed` tokens, preempting others while the pool is short. Returns whether
    /// it succeeded with `id` still running.
    fn ensure_pages_or_preempt(&mut self, id: u64, tokens_needed: usize) -> bool {
        let mut ok = self.ensure_pages(id, tokens_needed, id);
        while !ok && self.preempt_one() {
            if self.seqs[&id].st != St::Running {
                break;
            }
            ok = self.ensure_pages(id, tokens_needed, id);
        }
        ok && self.seqs[&id].st == St::Running
    }

    fn segment(s: &Seq, kind: SegKind, tokens: Vec<i32>, width: usize, samples: bool) -> Segment {
        Segment {
            seq: s.id,
            slot: to_u32(s.slot.expect("running sequence has a slot")),
            kind,
            pos0: to_u32(s.num_computed),
            tokens,
            width: to_u32(width),
            samples,
            reset_slot: s.reset_slot_pending,
            pages: s.pages.clone(),
            sampling: s.sampling.clone(),
            stop: Stop {
                ids: s.stop_ids.clone(),
                generated: to_u32(s.generated),
                min_tokens: to_u32(s.min_new),
                max_tokens: to_u32(s.max_new),
                masked: s.min_masked.clone(),
            },
        }
    }

    /// Plans the next step: admission, one decode segment per decoding session, at most one prefill chunk.
    pub fn plan(&mut self, now_ms: f64) -> StepPlan {
        self.now_cache = now_ms;
        self.in_plan.clear();
        let mut plan = StepPlan::default();
        let t_max = self.cfg.t_max as usize;
        // 1) admission, priority then arrival (preempted requests keep their place)
        let mut wait: Vec<&Seq> = self.seqs.values().filter(|s| s.st == St::Waiting).collect();
        wait.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.order.cmp(&b.order)));
        let wait: Vec<u64> = wait.iter().map(|s| s.id).collect();
        for &id in &wait {
            if self.running() >= self.cfg.max_running as usize {
                break;
            }
            if !self.admit_one(id, now_ms) {
                break; // head of line: keep FIFO fairness
            }
        }
        // 2) decode rows
        let mut run: Vec<&Seq> = self.seqs.values().filter(|s| s.st == St::Running).collect();
        run.sort_by_key(|s| s.order);
        let run: Vec<u64> = run.iter().map(|s| s.id).collect();
        let mut rows = 0;
        for &id in &run {
            let s = &self.seqs[&id];
            if s.st != St::Running || s.num_computed != s.tokens.len() - 1 {
                continue; // preempted by an earlier row, or still prefilling
            }
            let w = self.width_for(s);
            if rows + w > t_max {
                break;
            }
            let needed = s.num_computed + w;
            if !self.ensure_pages_or_preempt(id, needed) {
                if self.seqs[&id].st == St::Running {
                    // nobody else to preempt: preempt itself
                    self.preempt(id);
                }
                continue;
            }
            let s = &self.seqs[&id];
            let last = *s.tokens.last().expect("prompts are never empty");
            plan.segs.push(Self::segment(s, SegKind::Decode, vec![last], w, true));
            self.in_plan.insert(id);
            rows += w;
        }
        plan.decode_rows = to_u32(rows);
        // 3) one prefill chunk
        let mut pre: Vec<&Seq> = run
            .iter()
            .map(|id| &self.seqs[id])
            .filter(|s| s.st == St::Running && s.num_computed < s.tokens.len() - 1)
            .collect();
        if !pre.is_empty() {
            match self.cfg.prefill_order {
                PrefillOrder::Srpt => pre.sort_by_key(|s| s.remaining()),
                PrefillOrder::Fifo => pre.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.order.cmp(&b.order))),
            }
            // queue for the urgency projection: running prefills, then waiting requests (their whole uncached prompt)
            let mut queue = Vec::new();
            let mut cum = 0;
            for s in &pre {
                cum += s.remaining();
                queue.push((now_ms - s.arrival_ms, cum));
            }
            for id in &wait {
                let Some(s) = self.seqs.get(id) else { continue };
                if s.st != St::Waiting {
                    continue;
                }
                cum += s.tokens.len() - self.reuse_estimate(s).best;
                queue.push((now_ms - s.arrival_ms, cum));
            }
            let s = pre[0];
            let (id, nc, rem) = (s.id, s.num_computed, s.remaining());
            let mut c = self.choose_chunk(rows, nc, &queue).min(rem);
            // never leave 1..=max_decode_rows prompt tokens for a later segment: a segment that short runs as a
            // decode segment (decode attention and decode-class plans), so the prompt's K/V would depend on where
            // the chunks fall, and a resumed prefill would differ in bits from a cold one
            let short = self.cfg.max_decode_rows as usize;
            if rem > c && rem - c <= short {
                let room = (self.cfg.max_prefill_rows as usize).min(t_max.saturating_sub(rows));
                c = if rem <= room {
                    rem
                } else {
                    c.saturating_sub(short + 1 - (rem - c)).max(1)
                };
            }
            if c > 0 && rows + c <= t_max && self.ensure_pages_or_preempt(id, nc + c) {
                let s = &self.seqs[&id];
                let chunk = s.tokens[nc..nc + c].to_vec();
                let samples = nc + c == s.tokens.len();
                plan.segs.push(Self::segment(s, SegKind::Prefill, chunk, c, samples));
                plan.prefill_rows = to_u32(c);
                plan.predicted_ms += self.prefill_cost_ms(c, nc);
                self.in_plan.insert(id);
            }
        }
        if !plan.segs.is_empty() {
            plan.predicted_ms += self.decode_cost_ms(rows);
        }
        plan.stale_slots = (0..self.slot_owner.len())
            .filter(|&i| self.slot_owner[i].is_none_or(|o| self.seqs[&o].reset_slot_pending))
            .map(to_u32)
            .collect();
        plan
    }

    fn park_or_free(&mut self, id: u64, park: bool, now_ms: f64) {
        let pg = self.page_tokens();
        let mut s = self.seqs.remove(&id).expect("finished sequence exists");
        if !park || s.num_computed == 0 {
            self.pool.release(&mut s.pages, 0);
            release_slot(&mut self.slot_owner, &mut s);
            return;
        }
        s.tokens.truncate(s.num_computed);
        if self.cfg.trim_parked_pages {
            self.pool.release(&mut s.pages, ceil_div(s.num_computed, pg));
        }
        s.st = St::Parked;
        s.last_use_ms = now_ms;
        s.ring_valid = s.slot.is_some() && !s.reset_slot_pending;
        // an older parked copy of the same session that is a prefix of this one is redundant
        let drop_ids: Vec<u64> = self
            .seqs
            .values()
            .filter(|q| q.st == St::Parked && !q.session.is_empty() && q.session == s.session && q.salt == s.salt)
            .filter(|q| common_prefix(&q.tokens[..q.num_computed], &s.tokens) == q.num_computed)
            .map(|q| q.id)
            .collect();
        self.seqs.insert(id, s);
        for qid in drop_ids {
            let mut q = self.seqs.remove(&qid).expect("dropped id taken from the map");
            self.pool.release(&mut q.pages, 0);
            release_slot(&mut self.slot_owner, &mut q);
        }
    }

    /// Applies the runtime's results of `plan` and returns events for the front end (tokens, first-step cache hits,
    /// done).
    pub fn commit(&mut self, plan: &StepPlan, results: &[SegResult], now_ms: f64) -> Vec<SeqEvent> {
        self.now_cache = now_ms;
        self.ctr.steps += 1;
        let by_seq: BTreeMap<u64, &SegResult> = results.iter().map(|r| (r.seq, r)).collect();
        let mut events = Vec::new();
        for g in &plan.segs {
            let Some(s) = self.seqs.get_mut(&g.seq) else { continue };
            if s.st != St::Running {
                continue;
            }
            let r = by_seq.get(&g.seq);
            let mut toks: Vec<i32> = r.map(|r| r.tokens.clone()).unwrap_or_default();
            let width = g.width as usize;
            match g.kind {
                SegKind::Decode => {
                    if toks.is_empty() {
                        continue; // runtime dropped the row (treated as not run)
                    }
                    toks.truncate(width);
                    toks.truncate(g.stop.keep(&toks));
                    let n = toks.len();
                    s.num_computed += n;
                    self.ctr.spec_draft += (width - 1) as u64;
                    self.ctr.spec_accepted += (n - 1) as u64;
                }
                SegKind::Prefill => {
                    s.num_computed += width;
                    if g.samples {
                        // the runtime takes a checkpoint after a prefill segment that samples (`slot_rollback`)
                        s.checkpoint = Some(s.num_computed);
                    }
                    toks.truncate(usize::from(g.samples));
                }
            }
            s.reset_slot_pending = false;
            s.ring_valid = true;
            s.last_use_ms = now_ms;
            if toks.is_empty() {
                continue;
            }
            s.tokens.extend_from_slice(&toks);
            s.generated += toks.len();
            self.ctr.generation_tokens += toks.len() as u64;
            let num_cached_tokens = if s.first_event_sent {
                None
            } else {
                s.first_event_sent = true;
                Some(to_u32(s.cached))
            };
            let cap = s.tokens.len() as u64 >= self.cfg.session_cap_tokens;
            let stopped = toks.last().is_some_and(|&t| g.stop.stops_at(t, s.generated));
            let finish = if cap {
                Some(Finish::SessionCap)
            } else if s.generated >= s.max_new || stopped {
                Some(Finish::Done)
            } else {
                None
            };
            events.push(SeqEvent {
                seq: g.seq,
                tokens: toks,
                logprobs: r.map(|r| r.logprobs.clone()).unwrap_or_default(),
                num_cached_tokens,
                finish: finish.clone(),
            });
            if finish.is_some() {
                self.park_or_free(g.seq, self.cfg.park_on_finish, now_ms);
            }
        }
        events
    }

    fn end(&mut self, id: u64, park: bool, now_ms: f64) {
        let Some(s) = self.seqs.get(&id) else { return };
        if s.st == St::Parked {
            return;
        }
        self.park_or_free(id, park, now_ms);
        self.pending_events.push(SeqEvent::finished(id, Finish::Done));
    }

    /// The front end found the request finished: stop it, park its KV (`park_on_finish`) and report done.
    pub fn finish(&mut self, id: u64, now_ms: f64) {
        self.end(id, self.cfg.park_on_finish, now_ms);
    }

    /// The client disconnected: stop, free (or park if `park_on_cancel`) and report done.
    pub fn cancel(&mut self, id: u64, now_ms: f64) {
        self.end(id, self.cfg.park_on_cancel, now_ms);
    }

    /// The runtime failed a step that contained `id`: free its pages and slot and report done. Never parked, whatever
    /// `park_on_cancel` says: after a failed step the slot's ring and conv state are unknown, so an exact resume from
    /// them could continue from corrupt state.
    pub fn abort(&mut self, id: u64, now_ms: f64) {
        self.end(id, false, now_ms);
    }

    /// Events produced outside [`Scheduler::commit`] (finish and cancel), drained by the back-end loop.
    pub fn take_events(&mut self) -> Vec<SeqEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Checks a plan just returned by [`Scheduler::plan`] (before its commit): every segment's sequence is running in
    /// the segment's slot with the segment's page list, the pages cover `pos0 + width`, every page that receives a row
    /// in `[pos0, pos0 + width)` has refcount 1 (a shared page is never written), and the rows fit in `t_max`.
    ///
    /// # Errors
    ///
    /// Returns [`InvariantError`] naming the first segment that breaks a rule.
    pub fn check_plan(&self, plan: &StepPlan) -> Result<(), InvariantError> {
        let fail = |m: String| Err(InvariantError(m));
        let pg = self.page_tokens();
        let mut rows = 0_usize;
        for g in &plan.segs {
            let id = g.seq;
            let Some(s) = self.seqs.get(&id) else {
                return fail(format!("plan names unknown seq {id}"));
            };
            if s.st != St::Running || s.slot != Some(g.slot as usize) {
                return fail(format!("seq {id} is not running in slot {}", g.slot));
            }
            if g.pages != s.pages {
                return fail(format!("seq {id}: segment pages differ from the sequence's"));
            }
            let (pos0, end) = (g.pos0 as usize, g.pos0 as usize + g.width as usize);
            if g.width == 0 || s.pages.len() < ceil_div(end, pg) {
                return fail(format!("seq {id}: pages do not cover rows {pos0}..{end}"));
            }
            if let Some(&p) = s.pages[pos0 / pg..ceil_div(end, pg)]
                .iter()
                .find(|&&p| self.pool.refcount(p) != 1)
            {
                return fail(format!("seq {id} writes shared page {p}"));
            }
            rows += g.width as usize;
        }
        if rows > self.cfg.t_max as usize || rows != (plan.decode_rows + plan.prefill_rows) as usize {
            return fail(format!("plan rows {rows} over t_max or not the plan's row counts"));
        }
        Ok(())
    }

    /// Checks the page and slot accounting: refcounts equal owners, the free list is exactly the unowned pages,
    /// every sequence owns pages for its computed tokens, slot owners and sequences agree.
    ///
    /// # Errors
    ///
    /// Returns [`InvariantError`] naming the first inconsistency found.
    pub fn check_invariants(&self) -> Result<(), InvariantError> {
        let fail = |m: String| Err(InvariantError(m));
        let total = self.pool.total();
        let mut rc = vec![0_u32; total];
        for (&id, s) in &self.seqs {
            if s.pages.len() < ceil_div(s.num_computed, self.page_tokens()) {
                return fail(format!("seq {id} has fewer pages than computed tokens"));
            }
            for &p in &s.pages {
                let Some(r) = rc.get_mut(p as usize) else {
                    return fail("bad page id".into());
                };
                *r += 1;
            }
            if let Some(slot) = s.slot
                && self.slot_owner[slot] != Some(id)
            {
                return fail("slot owner mismatch".into());
            }
            if s.st == St::Running && s.slot.is_none() {
                return fail("running seq without slot".into());
            }
        }
        if let Some(i) = (0..total).find(|&i| rc[i] != self.pool.refcounts()[i]) {
            return fail(format!("refcount mismatch on page {i}"));
        }
        let mut seen = vec![false; total];
        for &p in self.pool.free_list() {
            if self.pool.refcount(p) != 0 || seen[p as usize] {
                return fail("bad free list".into());
            }
            seen[p as usize] = true;
        }
        let zero = self.pool.refcounts().iter().filter(|&&r| r == 0).count();
        if zero != self.pool.free_count() {
            return fail("leaked pages".into());
        }
        for (i, owner) in self.slot_owner.iter().enumerate() {
            if let Some(o) = owner
                && self.seqs.get(o).is_none_or(|s| s.slot != Some(i))
            {
                return fail("dangling slot owner".into());
            }
        }
        Ok(())
    }
}
