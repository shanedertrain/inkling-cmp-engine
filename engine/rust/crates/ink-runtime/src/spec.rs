//! [`Speculation`]: the DSpark/DFlash2 drafter and the device-side speculation round of a [`Runtime`] (packages I2,
//! speculative sampling, and lossless verbatim lookup drafts).
//!
//! The round is `engine/native/tools/spec.cpp`'s (and `ink-spec`'s), call for call: drafter state, `f_rnn`, draft of
//! the round's slots, optional verbatim n-gram lookup extension (`m = 8`, up to +8 copied tokens on full-width rounds),
//! verify ids on the device, one forward over every verify segment, device argmax (or, when a segment samples, the
//! device sampler / exact Leviathan rejection + residual sampler) + accept, engine commit of the accepted rows, drafter
//! commit of the same rows. Everything else the runtime runs (prefill chunks, one-row decode rows) is committed to the
//! drafter and indexed in the per-slot lookup table after its call.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

use ink_rt_api::{
    AcceptBufs, CallRows, DRAFTS, DevBuf, DevSeg, DeviceSampling, DraftConfig, Drafter, FEAT_ROW_BYTES, MAX_DRAFTS,
    Runtime, RuntimeError, VerifyWidth,
};
use ink_sched::Stop;

use crate::lookup::{LookupPolicy, NgramIndex};

/// Base rows of a drafter-only verify segment: the anchor and the 7 drafter drafts.
pub(crate) const VERIFY_ROWS: usize = DRAFTS + 1;

fn spec_of(rt: &mut dyn Runtime) -> Result<&mut dyn ink_rt_api::SpecRuntime, RuntimeError> {
    rt.spec()
        .ok_or_else(|| RuntimeError::request("the runtime has no device-side speculation".to_owned()))
}

fn to_i32(v: usize) -> Result<i32, RuntimeError> {
    i32::try_from(v).map_err(|_| RuntimeError::request(format!("{v} does not fit i32")))
}

fn sub(buf: DevBuf, start: usize, len: usize) -> Result<DevBuf, RuntimeError> {
    buf.i32s(start, len)
        .ok_or_else(|| RuntimeError::request(format!("elements {start}..{} outside a device buffer", start + len)))
}

/// Host time and lookup counters of the parts of a round, summed.
#[derive(Debug, Default)]
pub(crate) struct RoundTimes {
    /// Drafter state, `f_rnn`, draft, optional lookup query, and verify ids.
    pub draft: Duration,
    /// The verify forward call (issue only).
    pub verify: Duration,
    /// Device argmax/sampling and accept; waits for the round's GPU work.
    pub accept: Duration,
    /// Engine commit and drafter commit of the accepted rows.
    pub commit: Duration,
    /// Session-rounds in which verbatim lookup appended extra rows.
    pub lookup_triggers: u32,
    /// Lookup tokens accepted beyond the drafter's 7 drafts.
    pub lookup_accepted: u32,
}

/// One verify segment of a round.
pub(crate) struct Verify<'a> {
    /// Engine slot (= drafter session).
    pub slot: usize,
    /// Sequence id, for the trace.
    pub seq: u64,
    /// Position of the anchor row.
    pub pos0: usize,
    /// Where the request's tokens end: the round keeps (and commits the rows of) the emitted tokens up to it.
    pub stop: &'a Stop,
    /// The request's device sampling controls (temperature 0: greedy).
    pub sampling: DeviceSampling,
}

/// One segment of a non-speculative forward call, every row of which was committed.
pub(crate) struct Plain<'a> {
    /// Engine slot.
    pub slot: usize,
    /// First call row of the segment.
    pub row0: usize,
    /// Position of the first row.
    pub pos0: usize,
    /// Rows of the segment.
    pub rows: usize,
    /// Committed token ids of the segment (indexed for verbatim lookup).
    pub tokens: &'a [i32],
    /// The token sampled from the segment's last row, which becomes the slot's anchor.
    pub sampled: Option<i32>,
}

/// The drafter, its device buffers, per-slot lookup indices, and the host copy of each slot's anchor and position.
pub(crate) struct Speculation {
    drafter: Box<dyn Drafter>,
    /// Drafter positions per slot.
    ctx: usize,
    slots: usize,
    /// Maximum lookup tokens appended after the drafter's 7 (`0` = off, up to `8`).
    lookup_len: usize,
    /// Minimum backward verbatim match length `m` required to trigger lookup.
    lookup_min: usize,
    /// Draft stride `K` (`7` without lookup, `7 + lookup_len` with lookup).
    draft_k: usize,
    /// Maximum verify rows per session (`draft_k + 1`: `8` without lookup, up to `16` with lookup).
    max_verify_rows: usize,
    /// Accept result stride `K + 2` (`9` without lookup, up to `17` with lookup).
    res_stride: usize,
    bufs: AcceptBufs,
    vids: DevBuf,
    /// Verify widths of every slot on the device (gated width) and their host copy.
    wdev: DevBuf,
    widths: Vec<i32>,
    /// Rounding of the verify call's rows (gated width).
    call_rows: CallRows,
    /// Log threshold of the probability-gated verify width (`VERIFY_TAU.ln()`, overridable via `INK_VERIFY_TAU`).
    verify_tau_ln: f32,
    /// `0, 1, 2, ..` (call row indices).
    ramp: DevBuf,
    /// Packed call rows, positions and sessions of a drafter commit after a plain call (`3 t_max` i32).
    meta: DevBuf,
    /// Host copies of the drafter state `anchor` / `P` per slot; `dirty` when the device copies are older.
    anchor: Vec<i32>,
    pos: Vec<i32>,
    dirty: bool,
    /// Per-slot incremental n-gram index (populated when `lookup_len > 0`).
    lookup: Vec<NgramIndex>,
    /// Hybrid anchor-triggered / post-draft lookup policy.
    lookup_policy: LookupPolicy,
    /// Optional device copy of `[slots][DRAFTS]` proposal drafts with `-1` masking lookup-overridden slots `j < 7`
    /// so `sample_leviathan_kernel` falls back to exact plain target draw on overridden positions.
    prop_drafts_dev: Option<DevBuf>,
    prop_drafts_host: Vec<i32>,
    /// Per-slot starting draft index of lookup tokens in the current round (`usize::MAX` if no lookup fired).
    lookup_start: Vec<usize>,
    /// Scratch host buffers for downloading drafter drafts and uploading `[d0..d6, l0..l7]`.
    drafts_host: Vec<i32>,
    all_drafts: Vec<i32>,
    res: Vec<i32>,
    packed: Vec<i32>,
    trace: Option<BufWriter<File>>,
    rounds: u64,
    lookup_triggers_total: u64,
    lookup_accepted_total: u64,
}

impl std::fmt::Debug for Speculation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Speculation")
            .field("ctx", &self.ctx)
            .field("slots", &self.slots)
            .field("lookup_len", &self.lookup_len)
            .field("lookup_min", &self.lookup_min)
            .field("rounds", &self.rounds)
            .field("lookup_triggers", &self.lookup_triggers_total)
            .field("lookup_accepted", &self.lookup_accepted_total)
            .finish_non_exhaustive()
    }
}

impl Speculation {
    /// Creates the drafter of `rt` (`draft.sessions` = the runtime's slots) and the round's buffers for calls of up
    /// to `t_max` rows; `trace` receives one JSON line per round.
    pub(crate) fn new(
        rt: &mut dyn Runtime,
        draft: &DraftConfig,
        t_max: usize,
        trace: Option<File>,
    ) -> Result<Self, RuntimeError> {
        let n = draft.sessions as usize;
        let ctx = draft.max_ctx as usize;
        let lookup_len = draft.lookup as usize;
        let lookup_min = draft.lookup_min as usize;
        if DRAFTS + lookup_len > MAX_DRAFTS {
            return Err(RuntimeError::request(format!(
                "lookup drafts {lookup_len} exceeds maximum {}",
                MAX_DRAFTS - DRAFTS
            )));
        }
        let draft_k = DRAFTS + lookup_len;
        let max_verify_rows = draft_k + 1;
        let res_stride = draft_k + 2;
        if ctx < max_verify_rows {
            return Err(RuntimeError::request(format!(
                "drafter positions {ctx} below one verify segment ({max_verify_rows})"
            )));
        }
        let sp = spec_of(rt)?;
        let drafter = sp.create_drafter(draft)?;
        let drafts_buf = if lookup_len > 0 {
            sp.alloc(draft_k * n * 4)?
        } else {
            drafter.drafts()
        };
        let prop_drafts_dev = if lookup_len > 0 && drafter.proposal().is_some() {
            Some(sp.alloc(DRAFTS * n * 4)?)
        } else {
            None
        };
        let bufs = AcceptBufs {
            drafts: drafts_buf,
            anchor: sp.alloc(n * 4)?,
            pos: sp.alloc(n * 4)?,
            res: sp.alloc((res_stride * n + 1) * 4)?,
            crow: sp.alloc(max_verify_rows * n * 4)?,
            cpos: sp.alloc(max_verify_rows * n * 4)?,
            csess: sp.alloc(max_verify_rows * n * 4)?,
            lastfeat: sp.alloc(n * FEAT_ROW_BYTES)?,
            cfeats: sp.alloc(t_max.max(max_verify_rows * n) * FEAT_ROW_BYTES)?,
        };
        let vids = sp.alloc(max_verify_rows * n * 4)?;
        // Packed device buffer for coalesced widths [0..n] + drafts [n..n + 7*n] single-sync D2H.
        let wdev = sp.alloc((1 + DRAFTS) * n * 4)?;
        let ramp = sp.alloc(t_max * 4)?;
        let meta = sp.alloc(3 * t_max * 4)?;
        let rows: Vec<i32> = (0..t_max).map(to_i32).collect::<Result<_, _>>()?;
        sp.upload(ramp, &rows)?;
        let default_tau = drafter.default_verify_tau();
        let is_swa_only = drafter.is_swa_only();
        let lookup_policy = match std::env::var("INK_LOOKUP_ANCHOR").as_deref() {
            Ok("0") => LookupPolicy::for_drafter(false),
            Ok("1") => LookupPolicy::for_drafter(true),
            _ => LookupPolicy::for_drafter(is_swa_only),
        };
        Ok(Self {
            drafter,
            ctx,
            slots: n,
            lookup_len,
            lookup_min,
            draft_k,
            max_verify_rows,
            res_stride,
            bufs,
            vids,
            wdev,
            widths: vec![VERIFY_ROWS as i32; n],
            call_rows: match draft.verify {
                VerifyWidth::Gated(c) => c,
                VerifyWidth::Full => CallRows::RAGGED,
            },
            verify_tau_ln: std::env::var("INK_VERIFY_TAU")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|&v| v > 0.0 && v <= 1.0)
                .unwrap_or(default_tau)
                .ln(),
            ramp,
            meta,
            anchor: vec![0; n],
            pos: vec![0; n],
            dirty: false,
            lookup: (0..n).map(|_| NgramIndex::new()).collect(),
            lookup_policy,
            prop_drafts_dev,
            prop_drafts_host: vec![-1; n * DRAFTS],
            lookup_start: vec![usize::MAX; n],
            drafts_host: vec![0; n * DRAFTS],
            all_drafts: vec![-1; n * draft_k],
            res: Vec::new(),
            packed: Vec::new(),
            trace: trace.map(BufWriter::new),
            rounds: 0,
            lookup_triggers_total: 0,
            lookup_accepted_total: 0,
        })
    }

    /// Maximum drafts per round (`7` without lookup, `7 + lookup_len` with lookup).
    pub(crate) fn k(&self) -> usize {
        self.draft_k
    }

    /// Maximum verify rows per segment (`8` without lookup, up to `16` with lookup).
    pub(crate) fn max_verify_rows(&self) -> usize {
        self.max_verify_rows
    }

    /// Drafter positions per slot.
    pub(crate) fn ctx(&self) -> usize {
        self.ctx
    }

    /// The drafter position of a slot holding `len` committed rows: `len`, or the last position at which a draft
    /// block still fits.
    fn park(&self, len: usize) -> i32 {
        i32::try_from(len.min(self.ctx - self.max_verify_rows)).expect("drafter positions fit i32")
    }

    /// Whether `slot`'s drafter state is anchored at `token` and position `pos0`.
    pub(crate) fn anchored(&self, slot: usize, token: i32, pos0: usize) -> bool {
        self.anchor.get(slot) == Some(&token) && usize::try_from(self.pos[slot]).ok() == Some(pos0)
    }

    /// The runtime emptied `slot`.
    pub(crate) fn reset(&mut self, slot: usize) {
        self.pos[slot] = 0;
        self.dirty = true;
        if self.lookup_len > 0 {
            self.lookup[slot].clear();
        }
    }

    /// The runtime rolled `slot` back to position `pos`: the drafter's K/V below it is the slot's, its anchor and
    /// features are set again by the slot's next committed sampling row.
    pub(crate) fn rollback(&mut self, slot: usize, pos: usize) {
        self.pos[slot] = self.park(pos);
        self.dirty = true;
        if self.lookup_len > 0 {
            self.lookup[slot].truncate(pos);
        }
    }

    /// One speculative round over `segs` (each slot once); per segment the emitted tokens (accepted drafts and the
    /// target's token after them) and the rows its verify segment ran. Leaves the engine and the drafter committed.
    pub(crate) fn round(
        &mut self,
        rt: &mut dyn Runtime,
        segs: &[Verify<'_>],
        t: &mut RoundTimes,
    ) -> Result<Vec<(Vec<i32>, usize)>, RuntimeError> {
        let t0 = Instant::now();
        if self.dirty {
            let sp = spec_of(rt)?;
            sp.upload(self.bufs.anchor, &self.anchor)?;
            sp.upload(self.bufs.pos, &self.pos)?;
            self.dirty = false;
        }
        #[allow(unsafe_code)] // the drafter's bounds contract
        // SAFETY: the anchors and positions are the host's (`self.anchor`: emitted or sampled tokens, < VOCAB;
        // `self.pos`: committed lengths clamped by `park` to ctx - max_verify_rows) or, when not dirty, those the
        // accept kernel left (an emitted token; the row after the committed ones). A slot is drafted only in a round,
        // whose segments `RuntimeAdapter::check_verify` admits with pos0 = its position and pos0 + max_verify_rows <= ctx.
        let state = unsafe { self.drafter.set_state(self.bufs.anchor, self.bufs.pos) };
        state?;
        self.drafter.frnn(self.bufs.lastfeat)?;
        let slots: Vec<usize> = segs.iter().map(|g| g.slot).collect();
        if self.drafter.proposal().is_some() {
            let mut temps = vec![0.0f32; self.slots];
            let mut top_ps = vec![1.0f32; self.slots];
            let mut seeds = vec![0u64; self.slots];
            for g in segs {
                temps[g.slot] = g.sampling.temperature;
                top_ps[g.slot] = g.sampling.top_p;
                seeds[g.slot] = g.sampling.seed;
            }
            self.drafter.set_sampling(&temps, &top_ps, &seeds)?;
        }
        self.drafter.draft_sessions(&slots)?;
        let sp = spec_of(rt)?;
        let mut any_lookup = false;
        let mut prop_overridden = false;
        self.lookup_start.fill(usize::MAX);
        let round_drafts;
        let round_k;
        if self.lookup_len > 0 {
            // Single-sync coalesced D2H of both gated widths and drafter drafts.
            sp.verify_widths_and_drafts(
                self.drafter.conf(),
                self.drafter.drafts(),
                self.verify_tau_ln,
                self.slots,
                self.wdev,
                &mut self.widths,
                &mut self.drafts_host,
            )?;
            let mut extra = [0i32; VERIFY_ROWS];
            let mut anchor_prop = [-1i32; MAX_DRAFTS];
            let mut prop = [-1i32; MAX_DRAFTS - DRAFTS];
            // Lazily touch all_drafts + H2D upload ONLY if at least one session triggers lookup (or trace is active).
            let mut filled = false;
            let fill_all = |all_drafts: &mut [i32], drafts_host: &[i32], draft_k: usize| {
                all_drafts.fill(-1);
                for g2 in segs {
                    let q2 = g2.slot;
                    let ds2 = &drafts_host[q2 * DRAFTS..(q2 + 1) * DRAFTS];
                    all_drafts[q2 * draft_k..q2 * draft_k + DRAFTS].copy_from_slice(ds2);
                }
            };
            if self.trace.is_some() {
                fill_all(&mut self.all_drafts, &self.drafts_host, self.draft_k);
                filled = true;
            }
            for g in segs {
                let q = g.slot;
                let ds = &self.drafts_host[q * DRAFTS..(q + 1) * DRAFTS];
                if g.pos0 + self.max_verify_rows > self.ctx {
                    continue;
                }
                let w = self.widths[q] as usize;
                // 1. Anchor-triggered global lookup rescue (SWA-blind far DFlash2 matches; disabled on DSpark).
                let mut applied_anchor = false;
                if self.lookup_policy.uses_anchor() {
                    let (cnt_a, back_a, dist_a) =
                        self.lookup[q].propose_dist(&[self.anchor[q]], &mut anchor_prop[..self.draft_k]);
                    if cnt_a > 0 && self.lookup_policy.anchor_triggers(back_a, dist_a) {
                        let use_cnt = cnt_a.min(self.lookup_policy.anchor_cap(back_a, self.lookup_min, self.draft_k));
                        let lcp = if ds.iter().all(|&d| d >= 0) {
                            (0..DRAFTS.min(use_cnt))
                                .take_while(|&j| ds[j] == anchor_prop[j])
                                .count()
                        } else {
                            0
                        };
                        if use_cnt > lcp {
                            if !filled {
                                fill_all(&mut self.all_drafts, &self.drafts_host, self.draft_k);
                                filled = true;
                            }
                            if !prop_overridden {
                                self.prop_drafts_host.fill(-1);
                                for g2 in segs {
                                    let q2 = g2.slot;
                                    let ds2 = &self.drafts_host[q2 * DRAFTS..(q2 + 1) * DRAFTS];
                                    self.prop_drafts_host[q2 * DRAFTS..(q2 + 1) * DRAFTS].copy_from_slice(ds2);
                                }
                            }
                            self.all_drafts[q * self.draft_k + lcp..q * self.draft_k + use_cnt]
                                .copy_from_slice(&anchor_prop[lcp..use_cnt]);
                            if use_cnt < DRAFTS {
                                self.all_drafts[q * self.draft_k + use_cnt..q * self.draft_k + DRAFTS].fill(-1);
                            }
                            for j in lcp..DRAFTS {
                                self.prop_drafts_host[q * DRAFTS + j] = -1;
                                prop_overridden = true;
                            }
                            self.widths[q] = to_i32(1 + use_cnt)?;
                            self.lookup_start[q] = lcp;
                            t.lookup_triggers += 1;
                            self.lookup_triggers_total += 1;
                            any_lookup = true;
                            applied_anchor = true;
                        }
                    }
                }
                // 2. Post-draft verbatim extension on full-width (confident) sessions.
                if !applied_anchor && w == VERIFY_ROWS && ds.iter().all(|&d| d >= 0) {
                    extra[0] = self.anchor[q];
                    extra[1..VERIFY_ROWS].copy_from_slice(ds);
                    let (cnt, back) = self.lookup[q].propose(&extra, &mut prop[..self.lookup_len]);
                    if back >= self.lookup_min && cnt > 0 {
                        if !filled {
                            fill_all(&mut self.all_drafts, &self.drafts_host, self.draft_k);
                            filled = true;
                        }
                        self.all_drafts[q * self.draft_k + DRAFTS..q * self.draft_k + DRAFTS + cnt]
                            .copy_from_slice(&prop[..cnt]);
                        self.widths[q] = to_i32(VERIFY_ROWS + cnt)?;
                        self.lookup_start[q] = DRAFTS;
                        t.lookup_triggers += 1;
                        self.lookup_triggers_total += 1;
                        any_lookup = true;
                    }
                }
            }
            if filled {
                sp.upload(self.bufs.drafts, &self.all_drafts)?;
                if prop_overridden && let Some(pd) = self.prop_drafts_dev {
                    sp.upload(pd, &self.prop_drafts_host)?;
                }
                round_drafts = self.bufs.drafts;
                round_k = self.draft_k;
            } else {
                round_drafts = self.drafter.drafts();
                round_k = DRAFTS;
            }
        } else {
            if let Some(conf) = self.drafter.conf() {
                let wbuf = sub(self.wdev, 0, self.slots)?;
                sp.verify_widths(conf, self.verify_tau_ln, self.slots, wbuf, &mut self.widths)?;
            } else {
                self.widths.fill(VERIFY_ROWS as i32);
            }
            round_drafts = self.bufs.drafts;
            round_k = self.draft_k;
        }
        let round_vstride = round_k + 1;
        let round_rs = round_k + 2;
        let round_bufs = AcceptBufs {
            drafts: round_drafts,
            ..self.bufs
        };
        sp.build_verify_ids(self.bufs.anchor, round_drafts, self.vids, self.slots, round_k)?;
        let max_rows = if any_lookup || self.trace.is_some() {
            self.max_verify_rows
        } else {
            VERIFY_ROWS
        };
        let width = |q: usize| -> Result<usize, RuntimeError> {
            usize::try_from(self.widths[q])
                .ok()
                .filter(|w| (1..=max_rows).contains(w))
                .ok_or_else(|| RuntimeError::request(format!("verify width {} of slot {q}", self.widths[q])))
        };
        // call order: by gated width (stable); the call's rows rounded up by widening the widest
        let mut ord: Vec<usize> = (0..segs.len()).collect();
        ord.sort_by_key(|&i| self.widths[segs[i].slot]);
        let gated: Vec<usize> = ord.iter().map(|&i| width(segs[i].slot)).collect::<Result<_, _>>()?;
        let mut rows = gated.clone();
        self.call_rows.fill(&mut rows);
        let dev: Vec<DevSeg> = ord
            .iter()
            .zip(&rows)
            .map(|(&i, &w)| {
                let q = segs[i].slot;
                Ok(DevSeg {
                    slot: q,
                    ids: sub(self.vids, round_vstride * q, w)?,
                    rows: w,
                    logits: w,
                })
            })
            .collect::<Result<_, RuntimeError>>()?;
        let t1 = Instant::now();
        #[allow(unsafe_code)] // the runtime's bounds contract
        // SAFETY: the verify ids are each slot's anchor (as for set_state above: < VOCAB), its drafter drafts, and
        // verbatim-copied tokens from the slot's own vocabulary-validated history (negative padding becomes 0 in
        // verify_ids_kernel).
        let fwd = unsafe { sp.forward_dev(&dev) };
        fwd?;
        // the min-tokens mask on the verify rows, as on the one-row decode rows (speculative == plain): each segment's
        // logit rows start at the rows of the segments before it in call order
        let mut row0 = 0;
        for (&i, &w) in ord.iter().zip(&rows) {
            let g = &segs[i];
            let masked = g.stop.masked_rows(w);
            if masked > 0 {
                rt.mask_logits(row0, masked, &crate::adapter::masked_ids(g.stop))?;
            }
            row0 += w;
        }
        let sp = spec_of(rt)?;
        let t2 = Instant::now();
        if segs.iter().all(|g| g.sampling.temperature == 0.0) {
            sp.spec_accept(&round_bufs, round_k, &mut self.res)?;
        } else {
            // one per segment of the call, in call order
            let params: Vec<DeviceSampling> = ord.iter().map(|&i| segs[i].sampling).collect();
            if let Some(mut prop) = self.drafter.proposal() {
                if prop_overridden {
                    prop.drafts = self
                        .prop_drafts_dev
                        .expect("prop_drafts_dev allocated when lookup_len > 0 and drafter has proposal");
                }
                sp.spec_accept_leviathan(&round_bufs, &prop, round_k, &params, &mut self.res)?;
            } else {
                sp.spec_accept_sampled(&round_bufs, round_k, &params, &mut self.res)?;
            }
        }
        let t3 = Instant::now();
        if self.trace.is_some() {
            let called: Vec<&Verify<'_>> = ord.iter().map(|&i| &segs[i]).collect();
            self.write_trace(rt, &called, &dev, &gated)?;
        }
        self.rounds += 1;
        let rs = round_rs;
        let mut out = vec![(Vec::new(), 0); segs.len()];
        let mut kept = Vec::with_capacity(segs.len());
        let mut old_anchors = Vec::with_capacity(if self.lookup_len > 0 { segs.len() } else { 0 });
        for (s, &i) in ord.iter().enumerate() {
            let g = &segs[i];
            let a = usize::try_from(self.res[rs * s])
                .ok()
                .filter(|&a| a <= round_k)
                .ok_or_else(|| RuntimeError::request(format!("accepted count {} out of range", self.res[rs * s])))?;
            let mut emitted = self.res[rs * s + 1..rs * s + 2 + a].to_vec();
            emitted.truncate(g.stop.keep(&emitted));
            let n = emitted.len();
            if self.lookup_len > 0 {
                old_anchors.push((g.slot, self.anchor[g.slot], i));
                let lk_acc = (n - 1).saturating_sub(self.lookup_start[g.slot]);
                if lk_acc > 0 {
                    t.lookup_accepted += lk_acc as u32;
                    self.lookup_accepted_total += lk_acc as u64;
                }
            }
            // spec_accept moved the device anchor and position to the last emitted token and the row after the
            // committed ones; a request that stops inside the round commits fewer rows, and its device state is
            // replaced by the host's before the next round (its slot runs a prefill before it speculates again)
            self.anchor[g.slot] = *emitted.last().expect("a round emits at least one token");
            if n == a + 1 {
                self.pos[g.slot] = to_i32(g.pos0 + n)?;
            } else {
                self.pos[g.slot] = self.park(g.pos0 + n);
                self.dirty = true;
            }
            kept.push(u32::try_from(n).expect("at most 16 rows"));
            out[i] = (emitted, dev[s].rows);
        }
        let t4 = Instant::now();
        rt.commit(&kept)?;
        let rows = usize::try_from(self.res[rs * segs.len()]).unwrap_or(0);
        #[allow(unsafe_code)] // the drafter's bounds contract
        // SAFETY: the accept kernel packed the committed rows of the round's segments: positions pos0 .. pos0 + a <
        // pos0 + max_verify_rows <= ctx (check_verify) and sessions = the segments' slots < the drafter's sessions.
        let commit = unsafe {
            self.drafter
                .commit_rows(self.bufs.cfeats, self.bufs.cpos, self.bufs.csess, rows)
        };
        commit?;
        // Overlap CPU n-gram index updates behind GPU commit + drafter.commit_rows kernels.
        if self.lookup_len > 0 {
            for (slot, old_anchor, idx) in old_anchors {
                let emitted = &out[idx].0;
                self.lookup[slot].push(old_anchor);
                self.lookup[slot].extend_from_slice(&emitted[..emitted.len() - 1]);
            }
        }
        t.draft += t1 - t0;
        t.verify += t2 - t1;
        t.accept += t3 - t2;
        t.commit += Instant::now() - t4;
        Ok(out)
    }

    /// After a committed non-speculative call: its rows below the drafter's positions go to the drafter, its tokens
    /// go to the per-slot lookup index, and each slot's anchor, position and last features follow its segment.
    pub(crate) fn after_plain(&mut self, rt: &mut dyn Runtime, segs: &[Plain<'_>]) -> Result<(), RuntimeError> {
        self.packed.clear();
        let (mut rows, mut pos, mut sess) = (Vec::new(), Vec::new(), Vec::new());
        for g in segs {
            if self.lookup_len > 0 {
                self.lookup[g.slot].extend_from_slice(g.tokens);
            }
            for j in 0..g.rows.min(self.ctx.saturating_sub(g.pos0)) {
                rows.push(to_i32(g.row0 + j)?);
                pos.push(to_i32(g.pos0 + j)?);
                sess.push(to_i32(g.slot)?);
            }
        }
        let r = rows.len();
        if r > 0 {
            self.packed.extend(rows);
            self.packed.extend(pos);
            self.packed.extend(sess);
            let sp = spec_of(rt)?;
            sp.upload(sub(self.meta, 0, 3 * r)?, &self.packed)?;
            let (rows, pos, sess) = (sub(self.meta, 0, r)?, sub(self.meta, r, r)?, sub(self.meta, 2 * r, r)?);
            #[allow(unsafe_code)] // the runtime's bounds contract
            // SAFETY: the rows just uploaded are rows `row0 + j < row0 + rows` of the committed call's segments.
            let gathered = unsafe { sp.gather_feats(rows, r, self.bufs.cfeats) };
            gathered?;
            #[allow(unsafe_code)] // the drafter's bounds contract
            // SAFETY: the positions uploaded are pos0 + j < ctx (j < ctx - pos0 above), the sessions the segments'
            // slots, < the drafter's sessions (= runtime slots: `with_drafter` checks it).
            let commit = unsafe { self.drafter.commit_rows(self.bufs.cfeats, pos, sess, r) };
            commit?;
        }
        for g in segs {
            let len = g.pos0 + g.rows;
            self.pos[g.slot] = self.park(len);
            if let Some(token) = g.sampled {
                self.anchor[g.slot] = token;
                let dst = self
                    .bufs
                    .lastfeat
                    .range(g.slot * FEAT_ROW_BYTES, FEAT_ROW_BYTES)
                    .ok_or_else(|| RuntimeError::request(format!("slot {} outside the drafter's sessions", g.slot)))?;
                let row = sub(self.ramp, g.row0 + g.rows - 1, 1)?;
                let sp = spec_of(rt)?;
                #[allow(unsafe_code)] // the runtime's bounds contract
                // SAFETY: `ramp[i] = i`, so the row is row0 + rows - 1, the segment's last row in the committed call.
                let gathered = unsafe { sp.gather_feats(row, 1, dst) };
                gathered?;
            }
        }
        self.dirty |= !segs.is_empty();
        Ok(())
    }

    /// `{"round", "slots", "seqs", "pos", "widths", "drafts", "res"}`: the round's segments in call order with their
    /// verify rows, every slot's drafts (new for the round's slots only) and the accept results (`spec.cpp --trace`
    /// plus the sequences, positions and widths).
    fn write_trace(
        &mut self,
        rt: &mut dyn Runtime,
        segs: &[&Verify<'_>],
        dev: &[DevSeg],
        gated: &[usize],
    ) -> Result<(), RuntimeError> {
        let mut drafts = vec![0; self.draft_k * self.slots];
        spec_of(rt)?.download(self.bufs.drafts, &mut drafts)?;
        let list = |v: &mut dyn Iterator<Item = String>| v.collect::<Vec<_>>().join(",");
        let line = format!(
            "{{\"round\":{},\"slots\":[{}],\"seqs\":[{}],\"pos\":[{}],\"gated\":[{}],\"widths\":[{}],\"drafts\":[{}],\
             \"res\":[{}]}}",
            self.rounds,
            list(&mut segs.iter().map(|g| g.slot.to_string())),
            list(&mut segs.iter().map(|g| g.seq.to_string())),
            list(&mut segs.iter().map(|g| g.pos0.to_string())),
            list(&mut gated.iter().map(usize::to_string)),
            list(&mut dev.iter().map(|d| d.rows.to_string())),
            list(&mut drafts.iter().map(i32::to_string)),
            list(&mut self.res[..=self.res_stride * segs.len()].iter().map(i32::to_string)),
        );
        let trace = self.trace.as_mut().expect("tracing");
        writeln!(trace, "{line}")
            .and_then(|()| trace.flush())
            .map_err(|e| RuntimeError::request(format!("writing the speculation trace: {e}")))
    }
}
