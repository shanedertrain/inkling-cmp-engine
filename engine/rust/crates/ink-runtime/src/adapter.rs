//! [`RuntimeAdapter`]: the scheduler's [`ModelRuntime`] over a forward [`Runtime`] (packages I1, I2).
//!
//! Written once over `dyn Runtime`: it drives the C++ engine through the C ABI (`CapiRuntime`, configuration `M3`)
//! and, unchanged, the Rust engine once it implements [`Runtime`]. Pattern: vLLM v1's model runner step (inputs of
//! the scheduled requests, one forward, sample the last row of each sampling request) behind the `EngineCore` loop.

use std::collections::hash_map::RandomState;
use std::fs::File;
use std::hash::{BuildHasher, Hasher};
use std::time::{Duration, Instant};

use ink_rt_api::{
    DeviceSampling, DraftConfig, EngineConfig, ErrorKind, LOGIT_ROW, Runtime, RuntimeError, SegIn, VOCAB,
};
use ink_sched::{Drafting, ModelRuntime, RuntimeCaps, SegKind, SegResult, Segment, StepError, StepPlan, Stop};

use crate::sampler;
use crate::spec::{Plain, RoundTimes, Speculation, Verify};

/// Tokens per page of the runtime's own KV pool (`engine/native` global page).
const PAGE_TOKENS: u32 = 1024;

/// Most rows of a decode segment (up to 16 rows: anchor + 7 drafter drafts + up to 8 lookup drafts).
const MAX_DECODE_ROWS: u32 = 16;

/// The sequence whose state a runtime slot holds, its sampling seed and the KV pages the scheduler gave it.
struct Occupant {
    seq: u64,
    /// Key of the device sampler.
    seed: u64,
    /// Pages of the sequence's last segment (`Segment::pages`): what the runtime's slot holds after that segment.
    pages: usize,
}

impl Occupant {
    /// The request's seed, or one from the process's random hasher state without one.
    fn new(g: &Segment) -> Self {
        let seed = g.sampling.seed.map_or_else(
            || RandomState::new().build_hasher().finish(),
            // two's complement: every i64 seed is a distinct u64 seed
            |s| s as u64,
        );
        Self {
            seq: g.seq,
            seed,
            pages: 0,
        }
    }
}

/// The ids of `stop.masked` inside the vocabulary (an id beyond it is never sampled).
pub(crate) fn masked_ids(stop: &Stop) -> Vec<i32> {
    stop.masked
        .iter()
        .copied()
        .filter(|&t| usize::try_from(t).is_ok_and(|t| t < VOCAB))
        .collect()
}

/// The device sampler's controls of segment `g` (temperature 0 for a greedy request: the device argmax).
fn device_sampling(g: &Segment, seed: u64) -> DeviceSampling {
    let s = &g.sampling;
    DeviceSampling {
        seed,
        temperature: if s.is_greedy() { 0.0 } else { s.temperature as f32 },
        top_p: s.top_p as f32,
        min_p: s.min_p as f32,
        top_k: s.top_k,
    }
}

/// A runtime error as the scheduler's step error of the same kind (a device failure stops the back end).
fn step_error(e: RuntimeError) -> StepError {
    match e.kind() {
        ErrorKind::Request => StepError::Request(e.to_string()),
        ErrorKind::Device => StepError::Device(e.to_string()),
    }
}

fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// A [`ModelRuntime`] that runs each [`StepPlan`] on a [`Runtime`]: a speculative round for the verify segments, one
/// `forward` + sampling + `commit` for the rest.
///
/// What it serves (the subset of `docs/RUST-PLAN.md` 5.3 the runtime's API allows): chunked prefill, decode segments
/// of one row, and with a drafter ([`RuntimeAdapter::with_drafter`]) verify segments of exactly 8 rows (anchor + 7
/// DSpark drafts, greedy accept on the device: `engine/native/tools/spec.cpp`'s round); a new occupant of a slot from
/// position 0 (`reset_slot`), the exact resume of a parked sequence in its own slot, device sampling (greedy,
/// temperature, top-k, top-p, min-p, seed) and raw logprobs from the logit rows. The runtime keeps its own page
/// tables, so the caps report no external pages and no conv checkpoints (the scheduler then plans exact resumes only)
/// and the runtime's page count as the KV pool. A slot that no sequence holds any more, or whose new occupant starts
/// over in it (`StepPlan::stale_slots`), is reset before the forward call, so its pages return to the runtime when
/// the scheduler frees them (a new occupant's first segment can come steps after it took the slot).
///
/// A step with verify segments runs them first as one speculative round (the C ABI's accept needs a call of verify
/// segments only); the other segments (the prefill chunk, one-row decode rows of requests that do not speculate)
/// follow in one call. With a drafter, every row the runtime commits outside a round is committed to the drafter
/// after its call, so a slot can speculate whatever it ran before.
///
/// A call whose sampling segments are all greedy without logprobs reads only the runtime's device argmax of its
/// logit rows (`Runtime::argmax_host`). Any other call samples on the device (`Runtime::sample_host`, seeded per
/// request, keyed by position: the tokens equal those of the speculative round; a runtime without a device sampler
/// fails such a step) and copies the logit rows only for logprobs. With a runtime that accepts sampled verify rows (`SpecRuntime::accepts_sampled`), sampled requests without logprobs speculate too.
/// After every step the runtime's free KV pages must equal its pool minus the pages its slots hold by the runtime's
/// rule (`ceil(rows reached / page)` per slot, the most a slot's calls reached since its reset), and no slot may hold
/// more than the scheduler gave its sequence, or the step fails. The two counts differ only when a verify segment of
/// the gated width ran fewer than the 8 rows the scheduler planned.
///
/// Each step emits one `tracing` debug event with target `ink_step` (`phase = "runtime"`): the number of calls and
/// the microseconds spent in slot preparation, `forward`, the logit copy, sampling, `commit` and the drafter commits
/// after the plain call, and for the round: drafting, the verify forward, the accept (which waits for the GPU) and
/// the round's commits.
pub struct RuntimeAdapter {
    rt: Box<dyn Runtime>,
    caps: RuntimeCaps,
    start: Instant,
    /// Per slot: the sequence whose state the runtime holds there.
    slots: Vec<Option<Occupant>>,
    /// Per slot: the KV pages the runtime holds (`ceil(rows / page)` of the furthest row a call reached since the
    /// slot's reset; the engine takes pages in `forward`, frees them in `reset_slot` and those past a rollback).
    rt_pages: Vec<usize>,
    /// The drafter and the round's state, when speculating.
    spec: Option<Speculation>,
    /// Logit rows of the last forward call.
    logits: Vec<f32>,
    /// Device argmax of the last forward call's logit rows.
    argmax: Vec<i32>,
}

impl std::fmt::Debug for RuntimeAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeAdapter")
            .field("identity", &self.rt.identity())
            .field("caps", &self.caps)
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

impl RuntimeAdapter {
    /// Wraps `rt`, which was created with `config` (its slots, pages and row limits become the caps); decode segments
    /// have one row. Every slot of `rt` must be empty.
    #[must_use]
    pub fn new(mut rt: Box<dyn Runtime>, config: &EngineConfig) -> Self {
        let slot_rollback = rt.checkpoints().is_some();
        let caps = RuntimeCaps {
            max_slots: config.max_slots,
            t_max: config.t_max,
            page_tokens: PAGE_TOKENS,
            max_decode_rows: MAX_DECODE_ROWS,
            max_prefill_rows: config.max_chunk.min(config.t_max),
            external_pages: false,
            drafter: Drafting::None,
            conv_checkpoints: false,
            slot_rollback,
            vocab: u32::try_from(VOCAB).expect("the vocabulary fits u32"),
            kv_pages: Some(config.num_pages),
        };
        Self {
            rt,
            caps,
            start: Instant::now(),
            slots: (0..config.max_slots).map(|_| None).collect(),
            rt_pages: vec![0; config.max_slots as usize],
            spec: None,
            logits: Vec::new(),
            argmax: Vec::new(),
        }
    }

    /// As [`RuntimeAdapter::new`], with the runtime's DSpark drafter (`draft`, one drafter session per runtime slot):
    /// greedy requests without logprobs speculate 7 tokens per round while their rows stay below `draft.max_ctx`.
    /// `trace` receives one JSON line per round (slots, sequences, positions, every slot's drafts, accept results).
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `draft.sessions` is not the runtime's slot count, the runtime has no device-side
    /// speculation, or the drafter or its buffers cannot be created.
    pub fn with_drafter(
        mut rt: Box<dyn Runtime>,
        config: &EngineConfig,
        draft: &DraftConfig,
        trace: Option<File>,
    ) -> Result<Self, RuntimeError> {
        if draft.sessions != config.max_slots {
            return Err(RuntimeError::request(format!(
                "the drafter has {} sessions for {} runtime slots",
                draft.sessions, config.max_slots
            )));
        }
        let sampled = rt.samples_on_device() && rt.spec().is_some_and(|s| s.accepts_sampled());
        let spec = Speculation::new(rt.as_mut(), draft, config.t_max as usize, trace)?;
        let mut a = Self::new(rt, config);
        a.caps.drafter = Drafting::Fixed {
            k: u32::try_from(spec.k()).expect("7 drafts"),
            ctx: draft.max_ctx,
            sampled,
        };
        a.spec = Some(spec);
        Ok(a)
    }

    /// The runtime underneath.
    #[must_use]
    pub fn runtime(&self) -> &dyn Runtime {
        self.rt.as_ref()
    }

    fn slot_index(&self, slot: u32) -> Result<usize, StepError> {
        let i = slot as usize;
        if i < self.slots.len() {
            Ok(i)
        } else {
            Err(StepError::Request(format!(
                "slot {slot} outside the runtime's {} slots",
                self.slots.len()
            )))
        }
    }

    fn reset(&mut self, slot: usize) -> Result<(), StepError> {
        self.rt.reset_slot(slot).map_err(step_error)?;
        self.rt_pages[slot] = 0;
        if let Some(spec) = &mut self.spec {
            spec.reset(slot);
        }
        Ok(())
    }

    /// Whether the runtime can run decode segment `g` of width above 1 as a verify segment.
    fn check_verify(&self, g: &Segment) -> Result<(), StepError> {
        let Some(spec) = &self.spec else {
            return Err(StepError::Request(format!(
                "not wired: decode width {} needs the drafter (seq {})",
                g.width, g.seq
            )));
        };
        let max_verify = spec.max_verify_rows();
        let why = if g.width as usize != max_verify {
            "the drafter verifies its configured fixed width"
        } else if g.sampling.num_logprobs.is_some() {
            "verify rows return no logprobs"
        } else if !g.sampling.is_greedy() && !matches!(self.caps.drafter, Drafting::Fixed { sampled: true, .. }) {
            "verify rows accept greedily on this runtime"
        } else if g.pos0 as usize + max_verify > spec.ctx() {
            "the rows pass the drafter's positions"
        } else if !spec.anchored(g.slot as usize, g.tokens[0], g.pos0 as usize) {
            "the drafter's anchor or position is out of step with the plan"
        } else {
            return Ok(());
        };
        Err(StepError::Request(format!(
            "seq {}: decode segment of width {} at position {} cannot run: {why}",
            g.seq, g.width, g.pos0
        )))
    }

    /// Resets stale slots, then checks every segment against the runtime's slot state and resets new occupants.
    fn prepare(&mut self, plan: &StepPlan) -> Result<(), StepError> {
        for &slot in &plan.stale_slots {
            let i = self.slot_index(slot)?;
            if self.slots[i].take().is_some() {
                self.reset(i)?;
            }
        }
        for g in &plan.segs {
            let i = self.slot_index(g.slot)?;
            let tokens = if g.kind == SegKind::Decode { 1 } else { g.width as usize };
            if g.tokens.len() != tokens {
                return Err(StepError::Request(format!(
                    "seq {}: {} tokens for a {:?} segment of width {}",
                    g.seq,
                    g.tokens.len(),
                    g.kind,
                    g.width
                )));
            }
            if g.reset_slot {
                if g.pos0 != 0 {
                    return Err(StepError::Request(format!(
                        "not wired: seq {} resumes at position {} in a reset slot (needs scheduler-owned page \
                         tables and conv checkpoints)",
                        g.seq, g.pos0
                    )));
                }
                // a stale slot was reset above (or has held nothing since its last reset)
                if self.slots[i].is_some() {
                    self.reset(i)?;
                }
                let mut occupant = Occupant::new(g);
                occupant.pages = g.pages.len();
                self.slots[i] = Some(occupant);
            } else {
                let mut length = self.rt.length(i).map_err(step_error)?;
                let new_occupant = self.slots[i].as_ref().is_none_or(|o| o.seq != g.seq);
                if new_occupant && (g.pos0 as usize) < length {
                    // the parked sequence's slot rolled back to its prompt-end checkpoint (prefix reuse)
                    let pos = g.pos0 as usize;
                    let Some(ck) = self.rt.checkpoints() else {
                        return Err(StepError::Request(format!(
                            "seq {} resumes at position {pos} below slot {i}'s length {length}: the runtime \
                             cannot roll a slot back",
                            g.seq
                        )));
                    };
                    ck.rollback(i, pos).map_err(step_error)?;
                    // the rollback returns the pages past `pos` (`Slots::truncate`: ceil(pos / page) stay)
                    self.rt_pages[i] = self.rt_pages[i].min(pos.div_ceil(PAGE_TOKENS as usize));
                    if let Some(spec) = &mut self.spec {
                        spec.rollback(i, pos);
                    }
                    length = pos;
                }
                if length != g.pos0 as usize {
                    return Err(StepError::Request(format!(
                        "seq {} continues at position {} but slot {i} holds {length} tokens",
                        g.seq, g.pos0
                    )));
                }
                // an exact resume hands the parked sequence's slot state to a new request
                let occupant = match &mut self.slots[i] {
                    Some(o) if o.seq == g.seq => o,
                    other => other.insert(Occupant::new(g)),
                };
                occupant.pages = g.pages.len();
            }
            if g.kind == SegKind::Decode && g.width != 1 {
                self.check_verify(g)?;
            }
        }
        Ok(())
    }

    /// The runtime ran `rows` rows of `slot` from position `pos0`: the pages it holds for the slot.
    fn reached(&mut self, slot: usize, pos0: usize, rows: usize) {
        let p = (pos0 + rows).div_ceil(PAGE_TOKENS as usize);
        self.rt_pages[slot] = self.rt_pages[slot].max(p);
    }

    /// The runtime's free KV pages must equal its pool minus the pages its slots hold (`rt_pages`), and no slot may
    /// hold more than the scheduler gave its occupant: both sides take `ceil((pos0 + rows) / page)` pages per segment
    /// (the scheduler for the rows it planned, the runtime for the rows it ran) and free a slot's pages when it is
    /// reset.
    fn check_pages(&self) -> Result<(), StepError> {
        let Some(pool) = self.caps.kv_pages else {
            return Ok(());
        };
        for (i, (o, &r)) in self.slots.iter().zip(&self.rt_pages).enumerate() {
            let planned = o.as_ref().map_or(0, |o| o.pages);
            if r > planned {
                return Err(StepError::Request(format!(
                    "KV pages out of step: slot {i} holds {r} pages, its sequence was given {planned}"
                )));
            }
        }
        let held: usize = self.rt_pages.iter().sum();
        let free = self.rt.free_pages().map_err(step_error)?;
        let want = (pool as usize).checked_sub(held);
        if want != Some(free) {
            return Err(StepError::Request(format!(
                "KV pages out of step: the runtime has {free} of {pool} free, its slots hold {held}"
            )));
        }
        Ok(())
    }
}

/// Time spent in the runtime calls of one step.
#[derive(Default)]
struct CallTimes {
    calls: u32,
    forward: Duration,
    logits: Duration,
    sample: Duration,
    commit: Duration,
    /// Drafter commits after the plain call.
    feats: Duration,
    /// The round's parts ([`RoundTimes`]).
    round: RoundTimes,
}

impl RuntimeAdapter {
    /// One `forward` over `segs`, the logit copy, sampling of the sampling segments into `out`, `commit` of every row,
    /// and with a drafter the drafter's commit of those rows.
    fn call(&mut self, segs: &[&Segment], out: &mut Vec<SegResult>, t: &mut CallTimes) -> Result<(), StepError> {
        let ins: Vec<SegIn<'_>> = segs
            .iter()
            .map(|g| SegIn {
                slot: g.slot as usize,
                ids: &g.tokens,
                logits: usize::from(g.samples),
            })
            .collect();
        let t0 = Instant::now();
        let rows = self.rt.forward(&ins).map_err(step_error)?;
        let t1 = Instant::now();
        let sampling = segs.iter().filter(|g| g.samples).count();
        if rows != sampling {
            return Err(StepError::Request(format!(
                "forward returned {rows} logit rows for {sampling} sampling segments"
            )));
        }
        // greedy rows need only the device argmax, sampled rows the device sampler (greedy segments in it take the
        // same argmax); the logit rows are copied only for logprobs
        let sampling = || segs.iter().filter(|g| g.samples);
        let device_argmax = sampling().all(|g| g.sampling.is_greedy());
        let need_rows = sampling().any(|g| g.sampling.num_logprobs.is_some());
        if rows > 0 {
            // logprobs are of the raw logits (vLLM): copied before the min-tokens mask
            if need_rows {
                self.rt.logits_host(rows, &mut self.logits).map_err(step_error)?;
            }
            // the min-tokens mask of each sampling row (row k = the k-th sampling segment)
            for (k, g) in sampling().enumerate() {
                if g.stop.masked_rows(1) == 1 {
                    self.rt.mask_logits(k, 1, &masked_ids(&g.stop)).map_err(step_error)?;
                }
            }
            if device_argmax {
                self.rt.argmax_host(rows, &mut self.argmax).map_err(step_error)?;
            } else {
                let params: Vec<DeviceSampling> = segs
                    .iter()
                    .map(|g| {
                        let seed = self.slots[g.slot as usize].as_ref().map_or(0, |o| o.seed);
                        device_sampling(g, seed)
                    })
                    .collect();
                self.rt
                    .sample_host(&params, rows, &mut self.argmax)
                    .map_err(step_error)?;
            }
        }
        let t2 = Instant::now();
        let mut plain = Vec::with_capacity(segs.len());
        let (mut row0, mut k) = (0, 0);
        for g in segs {
            let mut sampled = None;
            if g.samples {
                let token = self.argmax[k];
                let logprobs = g
                    .sampling
                    .num_logprobs
                    .map(|top| sampler::logprobs(&self.logits[k * LOGIT_ROW..k * LOGIT_ROW + VOCAB], token, top))
                    .into_iter()
                    .collect();
                out.push(SegResult {
                    seq: g.seq,
                    tokens: vec![token],
                    logprobs,
                });
                sampled = Some(token);
                k += 1;
            }
            let (slot, pos0, rows) = (g.slot as usize, g.pos0 as usize, g.width as usize);
            self.rt_pages[slot] = self.rt_pages[slot].max((pos0 + rows).div_ceil(PAGE_TOKENS as usize));
            plain.push(Plain {
                slot: g.slot as usize,
                row0,
                pos0: g.pos0 as usize,
                rows: g.width as usize,
                tokens: &g.tokens,
                sampled,
            });
            row0 += g.width as usize;
        }
        let t3 = Instant::now();
        let kept: Vec<u32> = segs.iter().map(|g| g.width).collect();
        self.rt.commit(&kept).map_err(step_error)?;
        if self.caps.slot_rollback {
            // the end of a prompt is where the next turn of the session can resume
            for g in segs.iter().filter(|g| g.kind == SegKind::Prefill && g.samples) {
                let ck = self
                    .rt
                    .checkpoints()
                    .expect("slot_rollback means the runtime has checkpoints");
                ck.checkpoint(g.slot as usize).map_err(step_error)?;
            }
        }
        let t4 = Instant::now();
        if let Some(spec) = &mut self.spec {
            spec.after_plain(self.rt.as_mut(), &plain).map_err(step_error)?;
        }
        let t5 = Instant::now();
        t.calls += 1;
        t.forward += t1 - t0;
        t.logits += t2 - t1;
        t.sample += t3 - t2;
        t.commit += t4 - t3;
        t.feats += t5 - t4;
        Ok(())
    }

    /// After a call of `segs` segments failed on a request error: drops the rows of its forward call if that call is
    /// still uncommitted (commit of 0 rows each: no slot advances, the conv state stays), so the back end's aborts can
    /// reset its slots next step (`Runtime::reset_slot` refuses a slot of an uncommitted call). With no such call the
    /// runtime refuses the commit and nothing changes. After a device error nothing is called: the runtime is dead.
    fn drop_failed_call(&mut self, r: &Result<(), StepError>, segs: usize) {
        if let Err(StepError::Request(_)) = r {
            // refused when the failure came before the forward call or after its commit: nothing to drop then
            let _ = self.rt.commit(&vec![0; segs]);
        }
    }

    /// The verify segments as one speculative round: per segment its emitted tokens into `out`.
    fn round(&mut self, segs: &[&Segment], out: &mut Vec<SegResult>, t: &mut CallTimes) -> Result<(), StepError> {
        let spec = self
            .spec
            .as_mut()
            .expect("prepare checked the verify segments against the drafter");
        let slots = &self.slots;
        let verify: Vec<Verify> = segs
            .iter()
            .map(|g| Verify {
                slot: g.slot as usize,
                seq: g.seq,
                pos0: g.pos0 as usize,
                stop: &g.stop,
                sampling: device_sampling(g, slots[g.slot as usize].as_ref().map_or(0, |o| o.seed)),
            })
            .collect();
        let tokens = spec
            .round(self.rt.as_mut(), &verify, &mut t.round)
            .map_err(step_error)?;
        for (g, (tokens, rows)) in segs.iter().zip(tokens) {
            self.reached(g.slot as usize, g.pos0 as usize, rows);
            out.push(SegResult {
                seq: g.seq,
                tokens,
                logprobs: Vec::new(),
            });
        }
        t.calls += 1;
        Ok(())
    }
}

impl ModelRuntime for RuntimeAdapter {
    fn caps(&self) -> RuntimeCaps {
        self.caps
    }

    /// Runs the verify segments as one speculative round, then the other segments in one call (see the type's
    /// documentation).
    fn step(&mut self, plan: &StepPlan) -> Result<Vec<SegResult>, StepError> {
        let t0 = Instant::now();
        self.prepare(plan)?;
        let prep = t0.elapsed();
        let (verify, plain): (Vec<&Segment>, Vec<&Segment>) = plan
            .segs
            .iter()
            .partition(|g| g.kind == SegKind::Decode && g.width != 1);
        let mut out = Vec::with_capacity(plan.segs.len());
        let mut t = CallTimes::default();
        if !verify.is_empty() {
            let r = self.round(&verify, &mut out, &mut t);
            self.drop_failed_call(&r, verify.len());
            r?;
        }
        if !plain.is_empty() {
            let r = self.call(&plain, &mut out, &mut t);
            self.drop_failed_call(&r, plain.len());
            r?;
        }
        self.check_pages()?;
        let r = &t.round;
        tracing::debug!(
            target: "ink_step",
            phase = "runtime",
            rows = out.len(),
            verify = verify.len(),
            calls = t.calls,
            prepare_us = micros(prep),
            forward_us = micros(t.forward),
            logits_us = micros(t.logits),
            sample_us = micros(t.sample),
            commit_us = micros(t.commit),
            feats_us = micros(t.feats),
            draft_us = micros(r.draft),
            verify_us = micros(r.verify),
            accept_us = micros(r.accept),
            spec_commit_us = micros(r.commit),
            lookup_triggers = r.lookup_triggers,
            lookup_accepted = r.lookup_accepted,
        );
        Ok(out)
    }

    fn now_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}
