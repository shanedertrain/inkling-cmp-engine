//! [`RuntimeAdapter`] over a model-free [`Runtime`]: slot handling, sampling and logprobs, refused plans, and the
//! scheduler + adapter loop end to end.

use std::cell::RefCell;
use std::convert::Infallible;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ink_backend::{Backend, GenRequest, SamplingParams, StepOutput};
use ink_record::Source;
use ink_rt_api::{
    DeviceSampling, Dump, EngineConfig, Glue, Graphs, LOGIT_ROW, Runtime, RuntimeError, RuntimeStats, SegIn,
    SlotCheckpoints, SpecRuntime, TapFn, VOCAB,
};
use ink_runtime::RuntimeAdapter;
use ink_sched::{Config, Drafting, EngineBackend, ModelRuntime, SegKind, Segment, StepError, StepPlan, Stop};

/// What the fake engine saw, shared with the test.
#[derive(Debug, Default)]
struct Log {
    resets: Vec<usize>,
    forwards: Vec<Vec<(usize, Vec<i32>, usize)>>,
    commits: Vec<Vec<u32>>,
    /// Calls that copied logit rows, read only the device argmax, and used the device sampler.
    logit_copies: usize,
    argmax_reads: usize,
    /// `mask_logits` calls: first row, rows, ids.
    masks: Vec<(usize, usize, Vec<i32>)>,
    sample_reads: usize,
    /// Checkpoints taken and rollbacks made.
    checkpoints: Vec<(usize, usize)>,
    rollbacks: Vec<(usize, usize)>,
    /// The next `argmax_host` fails with this error (a failure after the forward call).
    fail_argmax: Option<RuntimeError>,
}

/// A model-free engine: per-slot lengths as `ink::Engine` keeps them; the logit row of input id `x` peaks at
/// `(x + 1) % 1000`, with `x % 7` as the runner-up.
struct FakeEngine {
    lengths: Vec<usize>,
    /// Pages held per slot and pages free, as `ink::Engine` counts them (1,024 tokens each).
    pages: Vec<usize>,
    free_pages: usize,
    pending: Vec<(usize, usize)>,
    /// The pending call was committed (as `ink::Engine`'s flag).
    committed: bool,
    rows: Vec<i32>,
    /// Ids masked per logit row of the last call (the device argmax honours them).
    masked: Vec<Vec<i32>>,
    /// Per logit row: its segment and the position its token is sampled at.
    row_seg: Vec<(usize, usize)>,
    log: Rc<RefCell<Log>>,
    /// Per slot the checkpoint position, when the engine has slot checkpoints.
    ckpt: Option<Vec<Option<usize>>>,
}

impl FakeEngine {
    fn new(slots: usize, log: Rc<RefCell<Log>>) -> Self {
        Self {
            lengths: vec![0; slots],
            pages: vec![0; slots],
            free_pages: 16,
            pending: Vec::new(),
            committed: true,
            rows: Vec::new(),
            masked: Vec::new(),
            row_seg: Vec::new(),
            log,
            ckpt: None,
        }
    }

    fn with_checkpoints(mut self) -> Self {
        self.ckpt = Some(vec![None; self.lengths.len()]);
        self
    }
}

impl SlotCheckpoints for FakeEngine {
    fn checkpoint(&mut self, slot: usize) -> Result<(), RuntimeError> {
        let len = self.lengths[slot];
        assert!(len > 0, "checkpoint of an empty slot");
        self.ckpt.as_mut().expect("checkpoints")[slot] = Some(len);
        self.log.borrow_mut().checkpoints.push((slot, len));
        Ok(())
    }
    fn checkpoint_pos(&self, slot: usize) -> Option<usize> {
        self.ckpt.as_ref().and_then(|c| c[slot])
    }
    fn rollback(&mut self, slot: usize, pos: usize) -> Result<(), RuntimeError> {
        if self.checkpoint_pos(slot) != Some(pos) || pos > self.lengths[slot] {
            return Err(RuntimeError::request(format!("slot {slot}: no checkpoint at {pos}")));
        }
        self.lengths[slot] = pos;
        let keep = pos.div_ceil(1024).min(self.pages[slot]);
        self.free_pages += self.pages[slot] - keep;
        self.pages[slot] = keep;
        self.log.borrow_mut().rollbacks.push((slot, pos));
        Ok(())
    }
}

fn next_of(x: i32) -> i32 {
    (x + 1) % 1000
}

impl Runtime for FakeEngine {
    fn identity(&self) -> Source {
        Source::this_build()
    }
    fn reset_slot(&mut self, slot: usize) -> Result<(), RuntimeError> {
        // as the Rust engine: a slot of the uncommitted last call is not reset
        if !self.committed && self.pending.iter().any(|&(s, _)| s == slot) {
            return Err(RuntimeError::request(format!("slot {slot} is in the uncommitted call")));
        }
        self.lengths[slot] = 0;
        self.free_pages += std::mem::take(&mut self.pages[slot]);
        if let Some(c) = &mut self.ckpt {
            c[slot] = None;
        }
        self.log.borrow_mut().resets.push(slot);
        Ok(())
    }
    fn length(&self, slot: usize) -> Result<usize, RuntimeError> {
        self.lengths
            .get(slot)
            .copied()
            .ok_or_else(|| RuntimeError::request(format!("no slot {slot}")))
    }
    fn forward(&mut self, segs: &[SegIn<'_>]) -> Result<usize, RuntimeError> {
        for s in segs {
            let need = (self.lengths[s.slot] + s.ids.len()).div_ceil(1024);
            while self.pages[s.slot] < need {
                if self.free_pages == 0 {
                    return Err(RuntimeError::request(format!(
                        "KV page pool exhausted pages {:?} lengths {:?}",
                        self.pages, self.lengths
                    )));
                }
                self.free_pages -= 1;
                self.pages[s.slot] += 1;
            }
        }
        self.pending = segs.iter().map(|s| (s.slot, s.ids.len())).collect();
        self.committed = false;
        self.rows = segs
            .iter()
            .filter(|s| s.logits == 1)
            .map(|s| *s.ids.last().expect("rows"))
            .collect();
        self.masked = vec![Vec::new(); self.rows.len()];
        self.row_seg = segs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.logits == 1)
            .map(|(i, s)| (i, self.lengths[s.slot] + s.ids.len()))
            .collect();
        self.log
            .borrow_mut()
            .forwards
            .push(segs.iter().map(|s| (s.slot, s.ids.to_vec(), s.logits)).collect());
        Ok(self.rows.len())
    }
    fn commit(&mut self, kept: &[u32]) -> Result<(), RuntimeError> {
        if self.committed || kept.len() != self.pending.len() {
            return Err(RuntimeError::request("commit does not match the last call".to_owned()));
        }
        self.committed = true;
        for (&(slot, m), &k) in self.pending.iter().zip(kept) {
            assert!(k as usize <= m);
            self.lengths[slot] += k as usize;
        }
        self.log.borrow_mut().commits.push(kept.to_vec());
        Ok(())
    }
    fn free_pages(&self) -> Result<usize, RuntimeError> {
        Ok(self.free_pages)
    }
    fn samples_on_device(&self) -> bool {
        true
    }
    /// A stand-in device sampler: greedy rows take the fake model's next token; sampled rows a token that depends
    /// only on the row's input token, the seed and the position (the real kernel's contract).
    fn sample_host(&mut self, params: &[DeviceSampling], rows: usize, out: &mut Vec<i32>) -> Result<(), RuntimeError> {
        if params.len() != self.pending.len() {
            return Err(RuntimeError::request(
                "sampling controls do not match the last call".to_owned(),
            ));
        }
        self.log.borrow_mut().sample_reads += 1;
        out.clear();
        for (&x, &(seg, pos)) in self.rows.iter().zip(&self.row_seg).take(rows) {
            let p = params[seg];
            out.push(if p.temperature == 0.0 {
                next_of(x)
            } else {
                let h = (x as u64).wrapping_mul(0x9E37_79B9) ^ p.seed.wrapping_mul(31) ^ (pos as u64).wrapping_mul(131);
                i32::try_from(h % 997).expect("small")
            });
        }
        Ok(())
    }
    fn argmax_host(&self, rows: usize, out: &mut Vec<i32>) -> Result<(), RuntimeError> {
        if let Some(e) = self.log.borrow_mut().fail_argmax.take() {
            return Err(e);
        }
        self.log.borrow_mut().argmax_reads += 1;
        out.clear();
        out.extend(
            self.rows.iter().zip(&self.masked).take(rows).map(
                |(&x, m)| {
                    if m.contains(&next_of(x)) { x % 7 } else { next_of(x) }
                },
            ),
        );
        Ok(())
    }
    fn mask_logits(&mut self, first: usize, rows: usize, ids: &[i32]) -> Result<(), RuntimeError> {
        for m in &mut self.masked[first..first + rows] {
            m.extend_from_slice(ids);
        }
        self.log.borrow_mut().masks.push((first, rows, ids.to_vec()));
        Ok(())
    }
    fn logits_host(&self, rows: usize, out: &mut Vec<f32>) -> Result<(), RuntimeError> {
        self.log.borrow_mut().logit_copies += 1;
        out.clear();
        out.resize(rows * LOGIT_ROW, -10.0);
        for (r, &x) in self.rows.iter().take(rows).enumerate() {
            out[r * LOGIT_ROW + usize::try_from(next_of(x)).expect("id")] = 5.0;
            out[r * LOGIT_ROW + usize::try_from(x % 7).expect("id")] = 4.0;
        }
        Ok(())
    }
    fn set_graphs(&mut self, _: Graphs) -> Result<(), RuntimeError> {
        Ok(())
    }
    fn set_glue(&mut self, _: Glue) -> Result<(), RuntimeError> {
        Ok(())
    }
    fn set_tap(&mut self, _: Option<TapFn>) -> Result<(), RuntimeError> {
        Ok(())
    }
    fn stats(&self) -> Result<RuntimeStats, RuntimeError> {
        Ok(RuntimeStats::default())
    }
    fn marks(&mut self) -> Result<Vec<(String, f32)>, RuntimeError> {
        Ok(Vec::new())
    }
    fn dump(&mut self, _: Dump, _: &Path) -> Result<(), RuntimeError> {
        Ok(())
    }
    fn spec(&mut self) -> Option<&mut dyn SpecRuntime> {
        None
    }
    fn checkpoints(&mut self) -> Option<&mut dyn SlotCheckpoints> {
        if self.ckpt.is_some() { Some(self) } else { None }
    }
}

fn config(slots: u32) -> EngineConfig {
    EngineConfig {
        max_slots: slots,
        num_pages: 16,
        ..EngineConfig::default()
    }
}

fn adapter(slots: u32) -> (RuntimeAdapter, Rc<RefCell<Log>>) {
    let log = Rc::new(RefCell::new(Log::default()));
    let rt = FakeEngine::new(slots as usize, Rc::clone(&log));
    (RuntimeAdapter::new(Box::new(rt), &config(slots)), log)
}

fn seg(seq: u64, slot: u32, kind: SegKind, pos0: u32, tokens: Vec<i32>, samples: bool, reset: bool) -> Segment {
    // the scheduler's pages for the segment: ceil((pos0 + width) / 1024)
    let pages = (0..(pos0 as usize + tokens.len()).div_ceil(1024))
        .map(|p| u32::try_from(p).expect("page id"))
        .collect();
    Segment {
        seq,
        slot,
        kind,
        pos0,
        width: u32::try_from(tokens.len()).expect("width"),
        tokens,
        samples,
        reset_slot: reset,
        pages,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        stop: Stop::default(),
    }
}

fn plan(segs: Vec<Segment>, idle: Vec<u32>) -> StepPlan {
    StepPlan {
        segs,
        stale_slots: idle,
        ..StepPlan::default()
    }
}

#[test]
fn caps_describe_a_runtime_with_its_own_pages() {
    let (a, _) = adapter(8);
    let c = a.caps();
    assert_eq!((c.max_slots, c.t_max, c.page_tokens), (8, 2112, 1024));
    assert!(!c.external_pages && !c.conv_checkpoints);
    assert_eq!(c.drafter, Drafting::None);
    assert_eq!((c.vocab as usize, c.kv_pages), (VOCAB, Some(16)));
}

#[test]
fn prefill_decode_and_greedy_sampling() {
    let (mut a, log) = adapter(4);
    // chunk 1 of a prompt (no sample), then the last chunk together with another session's first chunk
    let r = a
        .step(&plan(
            vec![seg(1, 0, SegKind::Prefill, 0, vec![10, 11, 12], false, true)],
            vec![1, 2, 3],
        ))
        .expect("step");
    assert!(r.is_empty());
    let r = a
        .step(&plan(
            vec![
                seg(1, 0, SegKind::Prefill, 3, vec![13, 14], true, false),
                seg(2, 1, SegKind::Prefill, 0, vec![500], true, true),
            ],
            vec![2, 3],
        ))
        .expect("step");
    assert_eq!(
        r.iter().map(|r| (r.seq, r.tokens.clone())).collect::<Vec<_>>(),
        [(1, vec![15]), (2, vec![501])]
    );
    let r = a
        .step(&plan(
            vec![
                seg(1, 0, SegKind::Decode, 5, vec![15], true, false),
                seg(2, 1, SegKind::Decode, 1, vec![501], true, false),
            ],
            vec![2, 3],
        ))
        .expect("step");
    assert_eq!(r[0].tokens, [16]);
    assert_eq!(r[1].tokens, [502]);
    let log = log.borrow();
    assert!(log.resets.is_empty(), "slots that never held state need no reset");
    assert_eq!(log.commits, [vec![3], vec![2, 1], vec![1, 1]]);
    assert_eq!(log.forwards[1][0], (0, vec![13, 14], 1));
    assert_eq!(
        (log.argmax_reads, log.logit_copies),
        (2, 0),
        "greedy rows without logprobs read only the device argmax"
    );
}

#[test]
fn a_page_count_out_of_step_fails_the_step() {
    let (mut a, _) = adapter(2);
    let mut g = seg(1, 0, SegKind::Prefill, 0, vec![1, 2], true, true);
    g.pages.clear(); // the scheduler would hold no page for a slot that holds one
    let e = a.step(&plan(vec![g], vec![1])).expect_err("pages out of step");
    assert!(e.to_string().contains("KV pages out of step"), "{e}");
}

#[test]
fn stale_slots_are_reset_once_and_resume_keeps_state() {
    let (mut a, log) = adapter(2);
    a.step(&plan(
        vec![seg(1, 0, SegKind::Prefill, 0, vec![1, 2], true, true)],
        vec![1],
    ))
    .expect("step");
    // the sequence parks (slot 0 held), then a new request resumes it exactly in slot 0: no reset
    a.step(&plan(
        vec![seg(9, 0, SegKind::Prefill, 2, vec![3, 4], true, false)],
        vec![1],
    ))
    .expect("step");
    assert!(
        log.borrow().resets.is_empty(),
        "a fresh slot and an exact resume need no reset"
    );
    // slot 0 becomes stale (evicted): reset once, however many plans name it
    a.step(&plan(
        vec![seg(3, 1, SegKind::Prefill, 0, vec![7], true, true)],
        vec![0],
    ))
    .expect("step");
    a.step(&plan(
        vec![seg(3, 1, SegKind::Decode, 1, vec![8], true, false)],
        vec![0],
    ))
    .expect("step");
    assert_eq!(log.borrow().resets, [0], "stale slot 0 once; slot 1 never held state");
    assert_eq!(a.runtime().length(0).expect("slot"), 0);
}

#[test]
fn plans_the_runtime_cannot_run_are_refused() {
    let (mut a, _) = adapter(2);
    let e = a
        .step(&plan(
            vec![seg(1, 0, SegKind::Prefill, 1024, vec![1], true, true)],
            vec![],
        ))
        .expect_err("reset above 0");
    assert!(e.to_string().contains("not wired"), "{e}");
    let mut verify = seg(1, 0, SegKind::Decode, 0, vec![1], true, false);
    verify.width = 3;
    let e = a.step(&plan(vec![verify], vec![])).expect_err("verify rows");
    assert!(e.to_string().contains("drafter"), "{e}");
    let e = a
        .step(&plan(
            vec![seg(1, 0, SegKind::Decode, 0, vec![1, 2, 3], true, false)],
            vec![],
        ))
        .expect_err("two tokens in a decode segment");
    assert!(e.to_string().contains("3 tokens for a Decode segment"), "{e}");
    let e = a
        .step(&plan(vec![seg(1, 1, SegKind::Decode, 5, vec![1], true, false)], vec![]))
        .expect_err("length mismatch");
    assert!(e.to_string().contains("holds 0 tokens"), "{e}");
    let e = a
        .step(&plan(vec![seg(1, 2, SegKind::Prefill, 0, vec![1], true, true)], vec![]))
        .expect_err("slot out of range");
    assert!(e.to_string().contains("outside"), "{e}");
}

#[test]
fn logprobs_and_seeded_sampling() {
    let (mut a, _) = adapter(1);
    let mut g = seg(1, 0, SegKind::Prefill, 0, vec![20], true, true);
    g.sampling.num_logprobs = Some(2);
    let r = a.step(&plan(vec![g], vec![])).expect("step");
    let lp = &r[0].logprobs[0];
    assert_eq!((r[0].tokens[0], lp.rank), (21, 1));
    assert_eq!(lp.top_ids, [21, 20 % 7]);
    assert!(lp.logprob < 0.0 && lp.logprob > -1.0, "{lp:?}");
    // temperature 1 with a seed: the same draws in a fresh adapter
    let draws = |seed| {
        let (mut a, _) = adapter(1);
        let mut out = Vec::new();
        let mut last = 3;
        for pos in 0..32 {
            let mut g = seg(1, 0, SegKind::Decode, pos, vec![last], true, pos == 0);
            g.sampling.temperature = 1.0;
            g.sampling.seed = Some(seed);
            last = a.step(&plan(vec![g], vec![])).expect("step")[0].tokens[0];
            out.push(last);
        }
        out
    };
    assert_eq!(draws(5), draws(5));
    assert_ne!(draws(5), draws(6));
}

/// Scheduler + adapter + fake engine through [`EngineBackend`]: four concurrent greedy requests get the stream the
/// fake model defines, whatever the interleaving.
#[test]
fn engine_backend_over_the_adapter() {
    let be = EngineBackend::spawn(Config::default(), || {
        let log = Rc::new(RefCell::new(Log::default()));
        Ok::<_, Infallible>(RuntimeAdapter::new(Box::new(FakeEngine::new(8, log)), &config(8)))
    })
    .expect("spawns");
    assert_eq!(be.caps().kv_pages, Some(16));
    let outs: Arc<Mutex<Vec<(u64, StepOutput)>>> = Arc::default();
    for id in 0..4_u64 {
        let prompt: Vec<i32> = (0..3000)
            .map(|i| i32::try_from((i + id * 7) % 900).expect("id"))
            .collect();
        let outs = Arc::clone(&outs);
        be.submit(
            GenRequest {
                id,
                prompt,
                max_new_tokens: 20,
                sampling: SamplingParams {
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                ..GenRequest::default()
            },
            Box::new(move |o| outs.lock().expect("lock").push((id, o))),
        );
    }
    let t0 = Instant::now();
    while outs.lock().expect("lock").iter().filter(|(_, o)| o.done).count() < 4
        && t0.elapsed() < Duration::from_secs(10)
    {
        std::thread::sleep(Duration::from_millis(2));
    }
    let outs = outs.lock().expect("lock");
    for id in 0..4_u64 {
        let toks: Vec<i32> = outs
            .iter()
            .filter(|(i, _)| *i == id)
            .flat_map(|(_, o)| {
                assert!(o.error.is_none(), "{o:?}");
                o.tokens.clone()
            })
            .collect();
        let last = i32::try_from((2999 + id * 7) % 900).expect("id");
        let want: Vec<i32> = (1..=20).map(|k| (last + k) % 1000).collect();
        assert_eq!(toks, want, "request {id}");
    }
}

/// Without verify segments, a plan with decode segments and a prefill chunk runs as one call, every row committed.
#[test]
fn mixed_plans_run_as_one_call() {
    let (mut a, log) = adapter(2);
    a.step(&plan(
        vec![seg(1, 0, SegKind::Prefill, 0, vec![1, 2], true, true)],
        vec![1],
    ))
    .expect("step");
    let r = a
        .step(&plan(
            vec![
                seg(1, 0, SegKind::Decode, 2, vec![3], true, false),
                seg(2, 1, SegKind::Prefill, 0, vec![9, 10, 11], false, true),
            ],
            vec![],
        ))
        .expect("step");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].tokens, [4]);
    let log = log.borrow();
    assert_eq!(log.forwards.len(), 2);
    assert_eq!(log.forwards[1], [(0, vec![3], 1), (1, vec![9, 10, 11], 0)]);
    assert_eq!(log.commits, [vec![2], vec![1, 3]]);
}

/// The scheduler (narrowed by the adapter's caps) and the adapter keep the runtime's own page pool in step: random
/// sessions, multi-turn resumes, cancels and evictions on a pool of 16 pages never exhaust the runtime's pages.
#[test]
fn scheduler_and_runtime_pools_agree() {
    let log = Rc::new(RefCell::new(Log::default()));
    let a = RuntimeAdapter::new(Box::new(FakeEngine::new(4, Rc::clone(&log))), &config(4));
    assert!(!a.caps().slot_rollback);
    pools_agree(a, ink_sched::PrefixReuse::ExactResume);
}

/// As [`scheduler_and_runtime_pools_agree`] with slot checkpoints: follow-up prompts that extend an earlier prompt
/// but not its reply roll the parked slot back to the prompt's end; pages and lengths stay in step.
#[test]
fn scheduler_and_runtime_pools_agree_with_rollbacks() {
    let log = Rc::new(RefCell::new(Log::default()));
    let rt = FakeEngine::new(4, Rc::clone(&log)).with_checkpoints();
    let a = RuntimeAdapter::new(Box::new(rt), &config(4));
    assert!(a.caps().slot_rollback);
    pools_agree(a, ink_sched::PrefixReuse::SlotRollback);
    let log = log.borrow();
    assert!(!log.rollbacks.is_empty(), "rollbacks exercised");
    assert!(log.rollbacks.iter().all(|r| log.checkpoints.contains(r)));
}

fn pools_agree(mut a: RuntimeAdapter, reuse: ink_sched::PrefixReuse) {
    use ink_sched::Scheduler;
    let caps = a.caps();
    let mut cfg = Config {
        slots: caps.max_slots,
        max_running: 3,
        pool_tokens: u64::from(caps.kv_pages.expect("own pages")) * 1024,
        session_cap_tokens: 16 * 1024,
        spec_k: 0,
        prefix_reuse: reuse,
        trim_parked_pages: false,
        ..Config::default()
    };
    cfg.max_chunk = 2048;
    let mut s = Scheduler::new(cfg).expect("config");
    let mut x: u64 = 0x1234_5678;
    let mut rnd = |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    let mut history: Vec<Vec<i32>> = Vec::new();
    let (mut next, mut now) = (0_u64, 0.0);
    let mut resumed = 0;
    for step in 0..4000 {
        if rnd(20) == 0 {
            // a new prompt, or a previous one plus a suffix (exact resume when still parked in its slot)
            let mut prompt: Vec<i32> = if !history.is_empty() && rnd(2) == 0 {
                history[usize::try_from(rnd(history.len() as u64)).expect("index")].clone()
            } else {
                Vec::new()
            };
            let extra = 1 + rnd(3000) as usize;
            prompt.extend((0..extra).map(|_| i32::try_from(rnd(900)).expect("id")));
            if prompt.len() < 12_000 {
                history.push(prompt.clone());
                let r = GenRequest {
                    id: next,
                    prompt,
                    // one token parks exactly the prompt, so a later prompt extending it resumes exactly
                    max_new_tokens: if rnd(2) == 0 {
                        1
                    } else {
                        1 + u32::try_from(rnd(300)).expect("n")
                    },
                    sampling: SamplingParams {
                        temperature: 0.0,
                        ..SamplingParams::default()
                    },
                    ..GenRequest::default()
                };
                let _ = s.add(r, now);
                next += 1;
            }
        }
        if rnd(50) == 0 && next > 0 {
            s.cancel(rnd(next), now);
        }
        s.take_events();
        let p = s.plan(now);
        if p.is_empty() {
            continue;
        }
        let r = a.step(&p).unwrap_or_else(|e| {
            panic!(
                "step {step}: {e}\nplan {:?}\nused {}",
                p.segs
                    .iter()
                    .map(|g| (g.seq, g.slot, g.kind, g.pos0, g.width, g.reset_slot, g.pages.len()))
                    .collect::<Vec<_>>(),
                s.pages_used()
            )
        });
        s.commit(&p, &r, now);
        resumed += p
            .segs
            .iter()
            .filter(|g| !g.reset_slot && g.kind == SegKind::Prefill && g.pos0 > 0)
            .count();
        s.check_invariants().expect("invariants");
        now += 10.0;
    }
    assert!(s.counters().prefix_hits > 0 && resumed > 0, "exact resumes exercised");
    assert!(s.counters().evictions > 0, "evictions exercised");
}

/// A request error after the forward call (the argmax read fails): the step fails, the adapter drops the call's rows
/// (commit of 0 rows: the slot does not advance), so the next step can reset the slot the back end aborted.
#[test]
fn a_request_error_after_the_forward_call_leaves_the_slot_resettable() {
    let (mut a, log) = adapter(2);
    log.borrow_mut().fail_argmax = Some(RuntimeError::request("argmax refused"));
    let e = a
        .step(&plan(
            vec![seg(1, 0, SegKind::Prefill, 0, vec![1, 2], true, true)],
            vec![],
        ))
        .expect_err("argmax fails");
    assert!(matches!(e, StepError::Request(_)), "{e:?}");
    assert_eq!(
        log.borrow().commits.last().map(Vec::as_slice),
        Some(&[0][..]),
        "the failed call's rows are dropped"
    );
    assert_eq!(a.runtime().length(0).expect("slot"), 0);
    // the back end aborted seq 1: slot 0 is stale in the next plan and is reset before the new occupant's prefill
    let r = a
        .step(&plan(
            vec![seg(2, 0, SegKind::Prefill, 0, vec![5, 6], true, true)],
            vec![0],
        ))
        .expect("the slot is reset and reused");
    assert_eq!(r[0].tokens, [next_of(6)]);
    assert_eq!(log.borrow().resets, [0]);
}

/// A device error from the runtime is a device step error; the adapter makes no further runtime call for it.
#[test]
fn a_device_error_is_a_device_step_error() {
    let (mut a, log) = adapter(1);
    log.borrow_mut().fail_argmax = Some(RuntimeError::device("cuStreamSynchronize: CUDA_ERROR_ILLEGAL_ADDRESS"));
    let e = a
        .step(&plan(vec![seg(1, 0, SegKind::Prefill, 0, vec![1], true, true)], vec![]))
        .expect_err("device failure");
    assert!(
        matches!(&e, StepError::Device(m) if m.contains("ILLEGAL_ADDRESS")),
        "{e:?}"
    );
    assert!(
        log.borrow().commits.is_empty(),
        "nothing is committed on a dead runtime"
    );
}

/// Through [`EngineBackend`]: a device error fails every request the loop knows (the running one and the waiting
/// ones), the back end turns unhealthy and a later submit is answered with an error at once.
#[test]
fn a_device_error_stops_the_engine_backend() {
    let be = EngineBackend::spawn(
        Config {
            max_running: 1,
            ..Config::default()
        },
        || {
            let log = Rc::new(RefCell::new(Log::default()));
            log.borrow_mut().fail_argmax = Some(RuntimeError::device("CUDA_ERROR_ILLEGAL_ADDRESS"));
            Ok::<_, Infallible>(RuntimeAdapter::new(Box::new(FakeEngine::new(2, log)), &config(2)))
        },
    )
    .expect("spawns");
    let outs: Arc<Mutex<Vec<(u64, StepOutput)>>> = Arc::default();
    let submit = |id: u64| {
        let outs = Arc::clone(&outs);
        be.submit(
            GenRequest {
                id,
                prompt: vec![1, 2, 3],
                max_new_tokens: 4,
                // greedy: the step reads the device argmax, which fails
                sampling: SamplingParams {
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                ..GenRequest::default()
            },
            Box::new(move |o| outs.lock().expect("lock").push((id, o))),
        );
    };
    for id in 0..3 {
        submit(id);
    }
    let t0 = Instant::now();
    while be.healthy() && t0.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(!be.healthy(), "the loop ends after a device error");
    submit(9);
    let outs = outs.lock().expect("lock");
    for id in [0, 1, 2, 9] {
        let last = outs.iter().rfind(|(i, _)| *i == id).map(|(_, o)| o).expect("an answer");
        assert!(
            last.done && last.error.is_some() && last.tokens.is_empty(),
            "request {id}: {last:?}"
        );
    }
    let first = outs.iter().find(|(i, _)| *i == 0).map(|(_, o)| o).expect("an answer");
    assert!(
        first
            .error
            .as_deref()
            .is_some_and(|m| m.contains("device failure") && m.contains("ILLEGAL_ADDRESS")),
        "{first:?}"
    );
}

/// `min_tokens`: while fewer than `min_tokens` are out, the stop ids of `Stop::masked` are masked in the row that
/// samples, on the device after the logprob copy and before the argmax / device sampler; from `min_tokens` on nothing
/// is masked.
#[test]
fn min_tokens_masks_stop_ids_before_sampling() {
    let with_stop = |mut g: Segment, generated, logprobs| {
        g.stop = Stop {
            ids: vec![21],
            generated,
            min_tokens: 2,
            max_tokens: 10,
            masked: vec![21, 5_000_000],
        };
        g.sampling.num_logprobs = logprobs;
        g
    };
    // device argmax: the peak 21 is masked, the runner-up 20 % 7 = 6 wins; the id beyond the vocabulary is dropped
    let (mut a, log) = adapter(2);
    let g = with_stop(seg(1, 0, SegKind::Prefill, 0, vec![20], true, true), 0, None);
    assert_eq!(a.step(&plan(vec![g], vec![1])).expect("step")[0].tokens, [6]);
    assert_eq!(log.borrow().masks, [(0, 1, vec![21])]);
    // the same at output 2 (generated 1 < 2), and not at output 3
    let g = with_stop(seg(1, 0, SegKind::Decode, 1, vec![20], true, false), 1, None);
    assert_eq!(a.step(&plan(vec![g], vec![1])).expect("step")[0].tokens, [6]);
    let g = with_stop(seg(1, 0, SegKind::Decode, 2, vec![20], true, false), 2, None);
    assert_eq!(a.step(&plan(vec![g], vec![1])).expect("step")[0].tokens, [21]);
    assert_eq!(log.borrow().masks.len(), 2);
    // with logprobs the rows are copied before the mask: the logprobs are of the raw row (21 ranks first)
    let (mut a, log) = adapter(1);
    let g = with_stop(seg(1, 0, SegKind::Prefill, 0, vec![20], true, true), 0, Some(2));
    let r = a.step(&plan(vec![g], vec![])).expect("step");
    assert_eq!(r[0].tokens, [6]);
    assert_eq!(
        (r[0].logprobs[0].rank, r[0].logprobs[0].top_ids.clone()),
        (2, vec![21, 6])
    );
    assert_eq!((log.borrow().logit_copies, log.borrow().masks.len()), (1, 1));
}
