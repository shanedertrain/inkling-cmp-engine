//! [`RuntimeAdapter::with_drafter`] over a model-free runtime with device-side speculation: the scheduler's verify
//! widths, the speculative round, the drafter's commits outside rounds, and greedy output equal to plain decoding.
//!
//! The fake model: the target's argmax after input token `x` is `(x + 1) % 1000`. The fake drafter drafts that chain
//! from its anchor but gets draft `anchor % 7` wrong, so rounds accept 0 to 7 drafts. It checks what the real drafter
//! needs (`engine/kernels/drafter/README.md`): every position below a session's `P` committed with the target's row
//! of that position, `f_rnn` from the row at `P - 1`, block writes at `P .. P + 8` inside its positions; a violation
//! is logged and fails the test.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ink_backend::{GenRequest, SamplingParams};
use ink_record::Source;
use ink_rt_api::{
    AcceptBufs, DRAFTS, DevBuf, DevSeg, DeviceSampling, DraftConfig, Drafter, Dump, EngineConfig, FEAT_ROW_BYTES, Glue,
    Graphs, HeadPrecision, KvPrecision, LOGIT_ROW, Runtime, RuntimeError, RuntimeStats, SegIn, SpecRuntime, TapFn,
    VerifyWidth,
};
use ink_runtime::RuntimeAdapter;
use ink_sched::{Config, Drafting, ModelRuntime, PrefixReuse, Scheduler};

/// i32 words of one feature row; the fake uses the first three: slot, position, input token.
const FEAT_WORDS: usize = FEAT_ROW_BYTES / 4;

fn next_of(x: i32) -> i32 {
    (x + 1) % 1000
}

fn err(m: impl Into<String>) -> RuntimeError {
    RuntimeError::request(m)
}

/// Device memory: allocations of i32 words at made-up addresses.
#[derive(Default)]
struct Dev {
    allocs: Vec<(u64, Vec<i32>)>,
    next: u64,
}

impl Dev {
    fn alloc(&mut self, bytes: usize) -> DevBuf {
        let base = 0x1000 + self.next;
        self.next += (bytes as u64).div_ceil(256) * 256 + 256;
        self.allocs.push((base, vec![0; bytes.div_ceil(4)]));
        DevBuf::new(base, bytes)
    }

    fn words(&mut self, b: DevBuf) -> Result<&mut [i32], RuntimeError> {
        let (base, mem) = self
            .allocs
            .iter_mut()
            .find(|(base, mem)| b.addr() >= *base && b.addr() + b.bytes() as u64 <= *base + 4 * mem.len() as u64)
            .ok_or_else(|| err(format!("{b:?} is not device memory")))?;
        let at = usize::try_from(b.addr() - *base).expect("offset") / 4;
        Ok(&mut mem[at..at + b.bytes() / 4])
    }
}

/// What the fake drafter saw.
#[derive(Debug, Default)]
struct DraftLog {
    violations: Vec<String>,
    /// Per session: whether it was drafted by the last `draft_sessions` with a sound state (false for the others).
    sound: Vec<bool>,
    drafts: usize,
    rows_committed: usize,
}

struct FakeDrafter {
    dev: Rc<RefCell<Dev>>,
    log: Rc<RefCell<DraftLog>>,
    ctx: usize,
    drafts: DevBuf,
    /// Per session: committed input token per position (`None`: overwritten by a draft block).
    kv: Vec<BTreeMap<usize, Option<i32>>>,
    anchor: Vec<i32>,
    pos: Vec<usize>,
    /// Per session: (position, token) of the feature row `f_rnn` came from.
    frnn: Vec<(i32, i32)>,
    perfect: bool,
}

impl std::fmt::Debug for FakeDrafter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FakeDrafter")
    }
}

impl Drafter for FakeDrafter {
    unsafe fn commit_rows(&mut self, feats: DevBuf, pos: DevBuf, sess: DevBuf, n: usize) -> Result<(), RuntimeError> {
        let mut dev = self.dev.borrow_mut();
        let p = dev.words(pos)?[..n].to_vec();
        let s = dev.words(sess)?[..n].to_vec();
        let f = dev.words(feats)?[..n * FEAT_WORDS].to_vec();
        let mut log = self.log.borrow_mut();
        for i in 0..n {
            let row = &f[i * FEAT_WORDS..i * FEAT_WORDS + 3];
            let (q, at) = (
                usize::try_from(s[i]).expect("session"),
                usize::try_from(p[i]).expect("pos"),
            );
            if row[0] != s[i] || row[1] != p[i] || at >= self.ctx {
                log.violations.push(format!(
                    "commit row {i}: features {row:?} for session {} position {}",
                    s[i], p[i]
                ));
            }
            self.kv[q].insert(at, Some(row[2]));
        }
        log.rows_committed += n;
        Ok(())
    }

    fn frnn(&mut self, feats: DevBuf) -> Result<(), RuntimeError> {
        let mut dev = self.dev.borrow_mut();
        let f = dev.words(feats)?;
        for q in 0..self.kv.len() {
            self.frnn[q] = (f[q * FEAT_WORDS + 1], f[q * FEAT_WORDS + 2]);
        }
        Ok(())
    }

    unsafe fn set_state(&mut self, anchor: DevBuf, pos: DevBuf) -> Result<(), RuntimeError> {
        let mut dev = self.dev.borrow_mut();
        self.anchor = dev.words(anchor)?[..self.kv.len()].to_vec();
        self.pos = dev.words(pos)?[..self.kv.len()]
            .iter()
            .map(|&p| usize::try_from(p).expect("position"))
            .collect();
        Ok(())
    }

    fn draft(&mut self) -> Result<(), RuntimeError> {
        let all: Vec<usize> = (0..self.kv.len()).collect();
        self.draft_sessions(&all)
    }

    fn draft_sessions(&mut self, sessions: &[usize]) -> Result<(), RuntimeError> {
        let mut log = self.log.borrow_mut();
        log.drafts += 1;
        // the other sessions' drafts stay; a session not drafted in this call is not sound for its verify rows
        let mut out = self.dev.borrow_mut().words(self.drafts)?.to_vec();
        log.sound.iter_mut().for_each(|x| *x = false);
        for &q in sessions {
            let p = self.pos[q];
            // sound: every position below P committed, f_rnn from the row at P - 1
            let sound = p > 0
                && (0..p).all(|i| matches!(self.kv[q].get(&i), Some(Some(_))))
                && self.frnn[q]
                    == (
                        i32::try_from(p - 1).expect("pos"),
                        self.kv[q][&(p - 1)].expect("committed"),
                    );
            log.sound[q] = sound;
            if p + DRAFTS + 1 > self.ctx {
                log.violations
                    .push(format!("session {q}: draft block at {p} past {} positions", self.ctx));
            }
            for i in p..p + DRAFTS + 1 {
                self.kv[q].insert(i, None); // the block's own K/V
            }
            let mut t = self.anchor[q];
            let wrong = usize::try_from(self.anchor[q].rem_euclid(7)).expect("small");
            for j in 0..DRAFTS {
                t = next_of(t);
                out[q * DRAFTS + j] = if !self.perfect && j == wrong {
                    (t + 500) % 1000
                } else {
                    t
                };
            }
        }
        drop(log);
        self.dev.borrow_mut().words(self.drafts)?.copy_from_slice(&out);
        Ok(())
    }

    fn drafts(&self) -> DevBuf {
        self.drafts
    }
}

/// One row of the last forward call: slot, position, input token.
type Row = (i32, i32, i32);

/// What the engine saw.
#[derive(Debug, Default)]
struct EngineLog {
    /// Rows per forward call and whether its ids came from the device.
    calls: Vec<(usize, bool)>,
    rounds: usize,
}

struct FakeSpecEngine {
    dev: Rc<RefCell<Dev>>,
    draft_log: Rc<RefCell<DraftLog>>,
    log: Rc<RefCell<EngineLog>>,
    lengths: Vec<usize>,
    pages: Vec<usize>,
    free_pages: usize,
    /// Last call: per segment (slot, first row, rows, logit rows), and every row.
    segs: Vec<(usize, usize, usize, usize)>,
    rows: Vec<Row>,
    has_drafter: bool,
    perfect_drafter: bool,
}

impl FakeSpecEngine {
    fn new(slots: usize, pages: usize, draft_log: Rc<RefCell<DraftLog>>, log: Rc<RefCell<EngineLog>>) -> Self {
        Self {
            dev: Rc::default(),
            draft_log,
            log,
            lengths: vec![0; slots],
            pages: vec![0; slots],
            free_pages: pages,
            segs: Vec::new(),
            rows: Vec::new(),
            has_drafter: false,
            perfect_drafter: false,
        }
    }

    fn run(&mut self, segs: &[(usize, Vec<i32>, usize)], dev_ids: bool) -> Result<usize, RuntimeError> {
        self.segs.clear();
        self.rows.clear();
        for (slot, ids, logits) in segs {
            let len = self.lengths[*slot];
            let need = (len + ids.len()).div_ceil(1024);
            while self.pages[*slot] < need {
                if self.free_pages == 0 {
                    return Err(err("KV page pool exhausted"));
                }
                self.free_pages -= 1;
                self.pages[*slot] += 1;
            }
            self.segs.push((*slot, self.rows.len(), ids.len(), *logits));
            for (j, &t) in ids.iter().enumerate() {
                self.rows.push((
                    i32::try_from(*slot).expect("slot"),
                    i32::try_from(len + j).expect("pos"),
                    t,
                ));
            }
        }
        self.log.borrow_mut().calls.push((self.rows.len(), dev_ids));
        Ok(segs.iter().map(|s| s.2).sum())
    }

    /// Argmax of every logit row of the last call, in order.
    fn amax(&self) -> Vec<i32> {
        self.segs
            .iter()
            .flat_map(|&(_, r0, m, l)| (r0 + m - l..r0 + m).map(|r| next_of(self.rows[r].2)))
            .collect()
    }

    fn feat_rows(&self, rows: &[i32], dst: &mut [i32]) {
        for (i, &r) in rows.iter().enumerate() {
            let (s, p, t) = self.rows[usize::try_from(r).expect("row")];
            dst[i * FEAT_WORDS..i * FEAT_WORDS + 3].copy_from_slice(&[s, p, t]);
        }
    }
}

impl Runtime for FakeSpecEngine {
    fn identity(&self) -> Source {
        Source::this_build()
    }
    fn reset_slot(&mut self, slot: usize) -> Result<(), RuntimeError> {
        self.lengths[slot] = 0;
        self.free_pages += std::mem::take(&mut self.pages[slot]);
        Ok(())
    }
    fn length(&self, slot: usize) -> Result<usize, RuntimeError> {
        self.lengths.get(slot).copied().ok_or_else(|| err("no slot"))
    }
    fn free_pages(&self) -> Result<usize, RuntimeError> {
        Ok(self.free_pages)
    }
    fn forward(&mut self, segs: &[SegIn<'_>]) -> Result<usize, RuntimeError> {
        let segs: Vec<_> = segs.iter().map(|s| (s.slot, s.ids.to_vec(), s.logits)).collect();
        self.run(&segs, false)
    }
    fn commit(&mut self, kept: &[u32]) -> Result<(), RuntimeError> {
        if kept.len() != self.segs.len() {
            return Err(err("commit counts"));
        }
        for (&(slot, _, m, _), &k) in self.segs.iter().zip(kept) {
            assert!(k as usize <= m);
            self.lengths[slot] += k as usize;
        }
        Ok(())
    }
    fn logits_host(&self, rows: usize, out: &mut Vec<f32>) -> Result<(), RuntimeError> {
        out.clear();
        out.resize(rows * LOGIT_ROW, -10.0);
        for (r, t) in self.amax().into_iter().take(rows).enumerate() {
            out[r * LOGIT_ROW + usize::try_from(t).expect("id")] = 5.0;
        }
        Ok(())
    }
    fn argmax_host(&self, rows: usize, out: &mut Vec<i32>) -> Result<(), RuntimeError> {
        *out = self.amax().into_iter().take(rows).collect();
        Ok(())
    }
    fn samples_on_device(&self) -> bool {
        true
    }
    /// The fake model's next token for every row (the sampled rows' token does not matter here: this runtime does not
    /// accept sampled verify rows, so sampled requests decode one row per step).
    fn sample_host(&mut self, _: &[DeviceSampling], rows: usize, out: &mut Vec<i32>) -> Result<(), RuntimeError> {
        self.argmax_host(rows, out)
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
        Some(self)
    }
}

impl SpecRuntime for FakeSpecEngine {
    fn alloc(&mut self, bytes: usize) -> Result<DevBuf, RuntimeError> {
        Ok(self.dev.borrow_mut().alloc(bytes))
    }
    fn upload(&mut self, dst: DevBuf, src: &[i32]) -> Result<(), RuntimeError> {
        self.dev.borrow_mut().words(dst)?[..src.len()].copy_from_slice(src);
        Ok(())
    }
    fn download(&mut self, src: DevBuf, dst: &mut [i32]) -> Result<(), RuntimeError> {
        dst.copy_from_slice(&self.dev.borrow_mut().words(src)?[..dst.len()]);
        Ok(())
    }
    unsafe fn forward_dev(&mut self, segs: &[DevSeg]) -> Result<usize, RuntimeError> {
        let segs = segs
            .iter()
            .map(|s| Ok((s.slot, self.dev.borrow_mut().words(s.ids)?[..s.rows].to_vec(), s.logits)))
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        self.run(&segs, true)
    }
    fn build_verify_ids(
        &mut self,
        anchor: DevBuf,
        drafts: DevBuf,
        vids: DevBuf,
        n: usize,
        k: usize,
    ) -> Result<(), RuntimeError> {
        let mut dev = self.dev.borrow_mut();
        let a = dev.words(anchor)?[..n].to_vec();
        let d = dev.words(drafts)?[..n * k].to_vec();
        let v = dev.words(vids)?;
        let stride = k + 1;
        for q in 0..n {
            v[q * stride] = a[q];
            for j in 1..stride {
                v[q * stride + j] = d[q * k + j - 1].max(0);
            }
        }
        Ok(())
    }
    fn spec_accept(&mut self, b: &AcceptBufs, k: usize, res: &mut Vec<i32>) -> Result<(), RuntimeError> {
        if self.segs.iter().any(|&(_, _, m, l)| m == 0 || m > k + 1 || l != m) {
            return Err(err("spec_accept: verify segments with all logits"));
        }
        self.log.borrow_mut().rounds += 1;
        let amax = self.amax();
        let n_slots = self.lengths.len();
        let mut dev = self.dev.borrow_mut();
        let drafts = dev.words(b.drafts)?[..n_slots * k].to_vec();
        let mut anchor = dev.words(b.anchor)?[..n_slots].to_vec();
        let mut pos = dev.words(b.pos)?[..n_slots].to_vec();
        let s_count = self.segs.len();
        let rs = k + 2;
        res.clear();
        res.resize(rs * s_count + 1, 0);
        let mut crow = Vec::new();
        let mut last = Vec::new();
        for (s, &(q, r0, m, _)) in self.segs.iter().enumerate() {
            if !self.draft_log.borrow().sound[q] {
                self.draft_log
                    .borrow_mut()
                    .violations
                    .push(format!("slot {q} verified with unsound drafter state"));
            }
            let lim = m - 1;
            let mut a = 0;
            while a < lim && drafts[q * k + a] >= 0 && amax[r0 + a] == drafts[q * k + a] {
                a += 1;
            }
            res[rs * s] = i32::try_from(a).expect("a");
            for j in 0..a {
                res[rs * s + 1 + j] = drafts[q * k + j];
            }
            res[rs * s + 1 + a] = amax[r0 + a];
            anchor[q] = amax[r0 + a];
            pos[q] = self.rows[r0].1 + i32::try_from(a + 1).expect("a");
            crow.extend((0..=a).map(|j| i32::try_from(r0 + j).expect("row")));
            last.push((q, i32::try_from(r0 + a).expect("row")));
        }
        res[rs * s_count] = i32::try_from(crow.len()).expect("rows");
        dev.words(b.anchor)?[..n_slots].copy_from_slice(&anchor);
        dev.words(b.pos)?[..n_slots].copy_from_slice(&pos);
        let cpos: Vec<i32> = crow
            .iter()
            .map(|&r| self.rows[usize::try_from(r).expect("row")].1)
            .collect();
        let csess: Vec<i32> = crow
            .iter()
            .map(|&r| self.rows[usize::try_from(r).expect("row")].0)
            .collect();
        dev.words(b.crow)?[..crow.len()].copy_from_slice(&crow);
        dev.words(b.cpos)?[..crow.len()].copy_from_slice(&cpos);
        dev.words(b.csess)?[..crow.len()].copy_from_slice(&csess);
        self.feat_rows(&crow, dev.words(b.cfeats)?);
        let lf = dev.words(b.lastfeat)?;
        for (q, r) in last {
            self.feat_rows(&[r], &mut lf[q * FEAT_WORDS..(q + 1) * FEAT_WORDS]);
        }
        dev.words(b.res)?[..res.len()].copy_from_slice(res);
        Ok(())
    }
    unsafe fn gather_feats(&mut self, rows: DevBuf, n: usize, dst: DevBuf) -> Result<(), RuntimeError> {
        let mut dev = self.dev.borrow_mut();
        let r = dev.words(rows)?[..n].to_vec();
        self.feat_rows(&r, dev.words(dst)?);
        Ok(())
    }
    fn create_drafter(&mut self, config: &DraftConfig) -> Result<Box<dyn Drafter>, RuntimeError> {
        if std::mem::replace(&mut self.has_drafter, true) {
            return Err(err("one drafter per engine"));
        }
        let n = config.sessions as usize;
        let drafts = self.dev.borrow_mut().alloc(n * DRAFTS * 4);
        self.draft_log.borrow_mut().sound = vec![false; n];
        Ok(Box::new(FakeDrafter {
            dev: Rc::clone(&self.dev),
            log: Rc::clone(&self.draft_log),
            ctx: config.max_ctx as usize,
            drafts,
            kv: vec![BTreeMap::new(); n],
            anchor: vec![0; n],
            pos: vec![0; n],
            frnn: vec![(-1, -1); n],
            perfect: self.perfect_drafter,
        }))
    }
}

const SLOTS: u32 = 8;
const PAGES: u32 = 64;

fn engine_config() -> EngineConfig {
    EngineConfig {
        max_slots: SLOTS,
        num_pages: PAGES,
        ..EngineConfig::default()
    }
}

fn draft_config(ctx: u32) -> DraftConfig {
    DraftConfig {
        model_dir: PathBuf::from("/model"),
        fold_bin: PathBuf::from("/fold.bin"),
        dflash2_dir: None,
        sessions: SLOTS,
        max_ctx: ctx,
        head: HeadPrecision::Int8,
        global_kv: KvPrecision::Bf16,
        verify: VerifyWidth::Full,
        lookup: 0,
        lookup_min: 8,
    }
}

type Logs = (Rc<RefCell<DraftLog>>, Rc<RefCell<EngineLog>>);

fn spec_adapter(ctx: u32) -> (RuntimeAdapter, Logs) {
    let logs: Logs = (Rc::default(), Rc::default());
    let rt = FakeSpecEngine::new(SLOTS as usize, PAGES as usize, Rc::clone(&logs.0), Rc::clone(&logs.1));
    let a = RuntimeAdapter::with_drafter(Box::new(rt), &engine_config(), &draft_config(ctx), None).expect("drafter");
    (a, logs)
}

#[test]
fn caps_report_a_fixed_greedy_drafter() {
    let (a, _) = spec_adapter(4096);
    assert_eq!(
        a.caps().drafter,
        Drafting::Fixed {
            k: 7,
            ctx: 4096,
            sampled: false
        }
    );
    assert_eq!(a.caps().max_decode_rows, 16);
    let rt = FakeSpecEngine::new(4, 8, Rc::default(), Rc::default());
    let four = EngineConfig {
        max_slots: 4,
        ..engine_config()
    };
    let e = RuntimeAdapter::with_drafter(Box::new(rt), &four, &draft_config(4096), None)
        .expect_err("4 runtime slots, 8 drafter sessions");
    assert!(e.to_string().contains("8 sessions for 4 runtime slots"), "{e}");
}

/// The scheduler configuration `EngineBackend` derives from the adapter's caps.
fn sched_for(a: &RuntimeAdapter) -> Scheduler {
    let c = a.caps();
    let spec_k = match c.drafter {
        Drafting::Fixed { k, .. } => k,
        _ => 0,
    };
    Scheduler::new(Config {
        slots: c.max_slots,
        max_running: 4,
        max_decode_rows: c.max_decode_rows,
        spec_k,
        pool_tokens: u64::from(PAGES) * 1024,
        session_cap_tokens: u64::from(PAGES) * 1024,
        drafting: c.drafter,
        trim_parked_pages: false,
        prefix_reuse: PrefixReuse::ExactResume,
        ..Config::default()
    })
    .expect("config")
}

/// Runs `reqs` (each with the step it arrives at) through the scheduler and `a` until every request is done; returns
/// per request id its tokens, cut at its `max_new_tokens` as the front end does.
fn serve(a: &mut RuntimeAdapter, s: &mut Scheduler, reqs: Vec<(u32, GenRequest)>) -> BTreeMap<u64, Vec<i32>> {
    let limits: BTreeMap<u64, usize> = reqs.iter().map(|(_, r)| (r.id, r.max_new_tokens as usize)).collect();
    let mut out: BTreeMap<u64, Vec<i32>> = BTreeMap::new();
    let mut done = 0;
    let mut pending = reqs;
    let mut step = 0_u32;
    while done < limits.len() {
        let (now, later): (Vec<_>, Vec<_>) = pending.into_iter().partition(|(at, _)| *at <= step);
        pending = later;
        for (_, r) in now {
            s.add(r, f64::from(step)).expect("admitted");
        }
        let p = s.plan(f64::from(step));
        if !p.is_empty() {
            let r = a.step(&p).unwrap_or_else(|e| panic!("step {step}: {e}"));
            for e in s.commit(&p, &r, f64::from(step)) {
                out.entry(e.seq).or_default().extend(e.tokens);
                done += usize::from(e.finish.is_some());
            }
            s.check_invariants().expect("invariants");
        }
        step += 1;
        assert!(step < 100_000, "no progress");
    }
    for (id, t) in &mut out {
        t.truncate(limits[id]);
    }
    out
}

fn request(id: u64, prompt: Vec<i32>, max_new_tokens: u32, temperature: f64) -> GenRequest {
    GenRequest {
        id,
        prompt,
        max_new_tokens,
        sampling: SamplingParams {
            temperature,
            seed: Some(1),
            ..SamplingParams::default()
        },
        ..GenRequest::default()
    }
}

fn prompt(id: u64, len: usize) -> Vec<i32> {
    (0..len as u64)
        .map(|i| i32::try_from((i * 31 + id * 7) % 900).expect("id"))
        .collect()
}

/// The target's greedy continuation of `prompt`.
fn chain(prompt: &[i32], n: usize) -> Vec<i32> {
    let mut t = *prompt.last().expect("prompt");
    (0..n)
        .map(|_| {
            t = next_of(t);
            t
        })
        .collect()
}

/// Greedy requests arriving while others decode speculate, and their tokens are the target's greedy chain; a request
/// with temperature decodes one row per step next to them; the drafter's state is sound at every round and its block
/// writes never leave its positions; a second turn resumes the parked session exactly and speculates again.
#[test]
fn speculative_serving_keeps_greedy_output_and_drafter_state() {
    let (mut a, (draft_log, engine_log)) = spec_adapter(16_384);
    let mut s = sched_for(&a);
    let len = |id: u64| 3000 + 700 * usize::try_from(id).expect("id");
    let mut reqs: Vec<(u32, GenRequest)> = (0..4_u64)
        .map(|id| {
            (
                u32::try_from(id).expect("id") * 3,
                request(id, prompt(id, len(id)), 60, 0.0),
            )
        })
        .collect();
    // request 1 stops on a token inside a round: the 38th of its chain (once 5 are out)
    let stop_at = chain(&prompt(1, len(1)), 38)[37];
    reqs[1].1.stop_token_ids = vec![stop_at];
    reqs[1].1.min_new_tokens = 5;
    reqs.push((2, request(9, prompt(9, 1500), 30, 1.0)));
    let out = serve(&mut a, &mut s, reqs);
    for id in [0, 2, 3] {
        assert_eq!(out[&id], chain(&prompt(id, len(id)), 60), "request {id}");
    }
    assert_eq!(out[&1], chain(&prompt(1, len(1)), 38), "stops after its stop token");
    assert_eq!(out[&9].len(), 30);
    let rounds = engine_log.borrow().rounds;
    assert!(rounds > 0 && rounds < 4 * 60, "{rounds} rounds");
    {
        let d = draft_log.borrow();
        assert!(d.violations.is_empty(), "{:?}", d.violations);
        assert!(d.drafts == rounds && d.rows_committed > 4 * 3000, "{d:?}");
    }
    // every call with device ids is a round of 8-row verify segments
    assert!(engine_log.borrow().calls.iter().filter(|c| c.1).all(|c| c.0 % 8 == 0));
    // second turns of requests 0 (ended by its budget) and 1 (by its stop token): prompt, output and a suffix resume
    // each parked session exactly in its slot (the rounds kept no row past the client's output)
    let turn2 = |id: u64| {
        let mut p = prompt(id, len(id));
        p.extend(&out[&id]);
        p.extend([5, 6, 7]);
        p
    };
    let before = engine_log.borrow().rounds;
    let hits = s.counters().prefix_hits;
    let out2 = serve(
        &mut a,
        &mut s,
        vec![(0, request(20, turn2(0), 40, 0.0)), (0, request(21, turn2(1), 40, 0.0))],
    );
    assert_eq!(out2[&20], chain(&turn2(0), 40));
    assert_eq!(out2[&21], chain(&turn2(1), 40));
    let reused = s.counters().prefix_hits - hits;
    let want = turn2(0).len() + turn2(1).len() - 2 * 4;
    assert_eq!(
        reused, want as u64,
        "exact resumes: everything but the suffix and the last output token"
    );
    assert!(engine_log.borrow().rounds > before);
    assert!(
        draft_log.borrow().violations.is_empty(),
        "{:?}",
        draft_log.borrow().violations
    );
}

/// A session decodes speculatively while its rows stay below the drafter's positions, then one row per step; the
/// drafter's block writes stay inside its positions.
#[test]
fn sessions_past_the_drafter_positions_decode_one_row() {
    let (mut a, (draft_log, engine_log)) = spec_adapter(2048);
    let mut s = sched_for(&a);
    let p = prompt(3, 1990);
    let out = serve(&mut a, &mut s, vec![(0, request(3, p.clone(), 120, 0.0))]);
    assert_eq!(out[&3], chain(&p, 120));
    assert!(engine_log.borrow().rounds > 0, "speculated below 2,048 positions");
    let one_row = engine_log.borrow().calls.iter().filter(|c| !c.1 && c.0 == 1).count();
    assert!(one_row > 60, "{one_row} one-row calls past the drafter's positions");
    assert!(
        draft_log.borrow().violations.is_empty(),
        "{:?}",
        draft_log.borrow().violations
    );
}

/// Logprobs or temperature never get verify rows, and a verify segment whose anchor is not the slot's last token is
/// refused.
#[test]
fn verify_rows_only_for_plain_greedy_and_in_step_anchors() {
    let (mut a, (_, engine_log)) = spec_adapter(4096);
    let mut s = sched_for(&a);
    let mut r = request(1, prompt(1, 100), 4, 0.0);
    r.sampling.num_logprobs = Some(2);
    let out = serve(&mut a, &mut s, vec![(0, r), (0, request(2, prompt(2, 100), 4, 0.7))]);
    assert_eq!(out[&1], chain(&prompt(1, 100), 4));
    assert_eq!(out[&2].len(), 4);
    assert_eq!(engine_log.borrow().rounds, 0);
    let mut s2 = sched_for(&a);
    s2.add(request(5, prompt(5, 50), 8, 0.0), 0.0).expect("add");
    let p = s2.plan(0.0);
    let r = a.step(&p).expect("prefill");
    s2.commit(&p, &r, 0.0);
    let mut p = s2.plan(1.0);
    assert_eq!(p.segs[0].width, 8);
    p.segs[0].tokens[0] += 1;
    let e = a.step(&p).expect_err("anchor out of step");
    assert!(e.to_string().contains("out of step"), "{e}");
}

/// Verbatim n-gram lookup extends 8-row verify segments up to 16 rows on matching spans, accepts up to 15 drafts in a
/// single round (including partial lookup acceptance when the copied span diverges), preserves exact greedy output and
/// drafter soundness, and stays coherent across exact-resume second turns.
#[test]
fn lookup_extends_verify_to_sixteen_rows_and_preserves_greedy_output() {
    let logs: Logs = (Rc::default(), Rc::default());
    let mut rt = FakeSpecEngine::new(SLOTS as usize, PAGES as usize, Rc::clone(&logs.0), Rc::clone(&logs.1));
    rt.perfect_drafter = true;
    let dcfg = DraftConfig {
        lookup: 8,
        lookup_min: 8,
        ..draft_config(16_384)
    };
    let mut a = RuntimeAdapter::with_drafter(Box::new(rt), &engine_config(), &dcfg, None).expect("drafter");
    assert_eq!(a.caps().max_decode_rows, 16);
    assert_eq!(
        a.caps().drafter,
        Drafting::Fixed {
            k: 15,
            ctx: 16_384,
            sampled: false
        }
    );
    let mut s = sched_for(&a);
    // Prompt 0 contains a 40-token run matching the true chain 100..140, plus a diverging copy 140..151 then 999,
    // and ends at 100 so lookup fires immediately and yields both full 16-row acceptance and partial acceptance.
    let mut p0: Vec<i32> = (0..100).map(|i| 500 + (i * 17) % 300).collect();
    p0.extend(100..145);
    p0.extend([999, 998, 997, 996, 995, 994, 993, 992]);
    p0.extend([800, 801, 802, 100]);
    let out = serve(&mut a, &mut s, vec![(0, request(1, p0.clone(), 60, 0.0))]);
    assert_eq!(out[&1], chain(&p0, 60));
    let wide_calls = logs.1.borrow().calls.iter().filter(|c| c.1 && c.0 > 8).count();
    assert!(wide_calls >= 2, "expected 16-row lookup verify calls, saw {wide_calls}");
    assert!(
        logs.1.borrow().rounds < 60 / 8,
        "lookup should beat 8 tok/round: {} rounds",
        logs.1.borrow().rounds
    );
    assert!(
        logs.0.borrow().violations.is_empty(),
        "{:?}",
        logs.0.borrow().violations
    );
    // Second turn resumes the parked slot and re-triggers lookup cleanly.
    let mut p2 = p0;
    p2.extend(&out[&1]);
    p2.extend([810, 811, 100]);
    let out2 = serve(&mut a, &mut s, vec![(0, request(2, p2.clone(), 32, 0.0))]);
    assert_eq!(out2[&2], chain(&p2, 32));
    assert!(
        logs.0.borrow().violations.is_empty(),
        "{:?}",
        logs.0.borrow().violations
    );
}
