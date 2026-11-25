// Ported from engine/server/tests/test_sched.cpp @ 4171493: one test per C++ test (same name without the `sched_`
// prefix), plus plan-stream comparison against the C++ scheduler's output (tests/fixtures/*.plan, written by
// engine/server/tools/sched_plan_dump.cpp, deleted at cut-over U1; the version that wrote them is kept in
// bench/raw/2026-11-23-rust-sched-gate/) and a property test of the page-pool invariants.
//! Scheduler tests on a virtual clock (deterministic except the threaded smoke test).

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ink_backend::{Backend, GenRequest, StepOutput};
use ink_sched::{
    Config, ConfigError, Drafting, EngineBackend, FakeCost, FakeRuntime, Finish, ModelRuntime, PrefillOrder,
    PrefixReuse, RuntimeCaps, Scheduler, SegKind, SegResult, Segment, SeqEvent, SpawnError, StepError, StepPlan,
    WidthHook,
};
use proptest::prelude::*;

fn prompt_of(seed: u64, n: usize) -> Vec<i32> {
    (0..n as i64)
        .map(|i| FakeRuntime::token_at(seed + 1_000_000, i))
        .collect()
}

fn req(id: u64, prompt: Vec<i32>, max_new: u32, session: &str, priority: i32, salt: &str) -> GenRequest {
    GenRequest {
        id,
        session: session.to_owned(),
        cache_salt: salt.to_owned(),
        prompt,
        max_new_tokens: max_new,
        priority,
        ..GenRequest::default()
    }
}

fn plain(id: u64, prompt: Vec<i32>, max_new: u32) -> GenRequest {
    req(id, prompt, max_new, "", 0, "")
}

fn fnv(values: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn pages_fnv(pages: &[u32]) -> u64 {
    let as_i32: Vec<i32> = pages
        .iter()
        .map(|&p| i32::try_from(p).expect("page id fits i32"))
        .collect();
    fnv(&as_i32)
}

/// Writes one plan line in the format of `sched_plan_dump.cpp`.
fn record_plan(out: &mut String, step: usize, now: f64, p: &StepPlan) {
    write!(
        out,
        "P {step} now={:016x} pred={:016x} d={} p={}",
        now.to_bits(),
        p.predicted_ms.to_bits(),
        p.decode_rows,
        p.prefill_rows
    )
    .expect("write to String");
    for g in &p.segs {
        let kind = if g.kind == SegKind::Decode { 'D' } else { 'P' };
        write!(
            out,
            " | {} {} {kind} {} {} {} {} {} {:016x} {} {:016x}",
            g.seq,
            g.slot,
            g.pos0,
            g.width,
            u8::from(g.samples),
            u8::from(g.reset_slot),
            g.tokens.len(),
            fnv(&g.tokens),
            g.pages.len(),
            pages_fnv(&g.pages)
        )
        .expect("write to String");
    }
    out.push('\n');
}

fn record_summary(out: &mut String, name: &str, v: f64) {
    writeln!(out, "S {name} {:016x} {v:.6}", v.to_bits()).expect("write to String");
}

/// Compares a recorded stream with the C++ scheduler's fixture, naming the first differing line.
fn assert_fixture(name: &str, got: &str) {
    let path = format!("{}/tests/fixtures/{name}.plan", env!("CARGO_MANIFEST_DIR"));
    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    if got == want {
        return;
    }
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    let i = g
        .iter()
        .zip(&w)
        .position(|(a, b)| a != b)
        .unwrap_or(g.len().min(w.len()));
    panic!(
        "{name}: plan stream differs from the C++ scheduler at line {} ({} lines vs {} in the fixture)\n  rust: {}\n  c++:  {}",
        i + 1,
        g.len(),
        w.len(),
        g.get(i).unwrap_or(&"<end>"),
        w.get(i).unwrap_or(&"<end>")
    );
}

/// Scheduler + FakeRuntime on a virtual clock, as the C++ tests' `Sim`, with the plan recorder of the dump tool.
struct Sim {
    s: Scheduler,
    rt: FakeRuntime,
    log: String,
    nstep: usize,
    out: BTreeMap<u64, Vec<i32>>,
    done: BTreeMap<u64, bool>,
    cached: BTreeMap<u64, u32>,
    submit_ms: BTreeMap<u64, f64>,
    first_ms: BTreeMap<u64, f64>,
    first_order: Vec<u64>,
    step_ms: Vec<f64>,
    step_decode_rows: Vec<u32>,
    step_prefill_ms: Vec<f64>,
    step_has_prefill: Vec<bool>,
    /// `mixed` scenario: one `F` line per finished sequence.
    log_finish: bool,
    /// Every plan must suit a runtime with its own page tables: a reset slot starts at position 0.
    own_pages_only: bool,
    /// A model of a runtime that keeps its own slots and pages and rolls slots back to checkpoints
    /// (`ink-runtime`'s adapter over the Rust engine): every plan is checked against it.
    slot_model: Option<SlotModel>,
}

/// The slot state of a runtime with its own page tables and prompt-end checkpoints, driven by the plans as the
/// adapter drives the engine: lengths, checkpoints, occupants and pages per slot.
#[derive(Default)]
struct SlotModel {
    len: Vec<usize>,
    ck: Vec<Option<usize>>,
    seq: Vec<Option<u64>>,
    pages: Vec<usize>,
    rollbacks: usize,
}

impl SlotModel {
    fn new(slots: usize) -> Self {
        Self {
            len: vec![0; slots],
            ck: vec![None; slots],
            seq: vec![None; slots],
            pages: vec![0; slots],
            rollbacks: 0,
        }
    }

    fn reset(&mut self, i: usize) {
        (self.len[i], self.ck[i], self.seq[i], self.pages[i]) = (0, None, None, 0);
    }

    /// The adapter's `prepare`: stale slots reset, new occupants at 0, rollbacks to a checkpoint, lengths checked.
    fn before(&mut self, p: &StepPlan) {
        for &i in &p.stale_slots {
            self.reset(i as usize);
        }
        for g in &p.segs {
            let (i, pos0) = (g.slot as usize, g.pos0 as usize);
            if g.reset_slot {
                assert_eq!(pos0, 0, "seq {} resets slot {i} above position 0", g.seq);
                self.reset(i);
            } else if pos0 < self.len[i] {
                assert_ne!(self.seq[i], Some(g.seq), "seq {} rolls back its own slot", g.seq);
                assert_eq!(
                    self.ck[i],
                    Some(pos0),
                    "seq {}: slot {i} has no checkpoint at {pos0}",
                    g.seq
                );
                self.len[i] = pos0;
                self.pages[i] = self.pages[i].min(pos0.div_ceil(1024));
                self.rollbacks += 1;
            }
            assert_eq!(self.len[i], pos0, "seq {} continues slot {i} at {pos0}", g.seq);
            self.seq[i] = Some(g.seq);
            self.pages[i] = self.pages[i].max((pos0 + g.width as usize).div_ceil(1024));
            assert_eq!(self.pages[i], g.pages.len(), "seq {}: runtime pages of slot {i}", g.seq);
        }
    }

    /// The commits: kept rows per segment, and a checkpoint after a prefill segment that samples.
    fn after(&mut self, p: &StepPlan, r: &[SegResult]) {
        for g in &p.segs {
            let i = g.slot as usize;
            let kept = match g.kind {
                SegKind::Prefill => g.width as usize,
                SegKind::Decode => {
                    let toks = r.iter().find(|x| x.seq == g.seq).map_or(&[][..], |x| &x.tokens[..]);
                    g.stop.keep(&toks[..toks.len().min(g.width as usize)])
                }
            };
            self.len[i] += kept;
            if g.kind == SegKind::Prefill && g.samples {
                self.ck[i] = Some(self.len[i]);
            }
        }
    }
}

impl Sim {
    fn new(c: Config) -> Self {
        Self::with_cost(c, FakeCost::default())
    }

    fn with_cost(c: Config, fc: FakeCost) -> Self {
        Self {
            s: Scheduler::new(c).expect("valid config"),
            rt: FakeRuntime::new(fc),
            log: String::new(),
            nstep: 0,
            out: BTreeMap::new(),
            done: BTreeMap::new(),
            cached: BTreeMap::new(),
            submit_ms: BTreeMap::new(),
            first_ms: BTreeMap::new(),
            first_order: Vec::new(),
            step_ms: Vec::new(),
            step_decode_rows: Vec::new(),
            step_prefill_ms: Vec::new(),
            step_has_prefill: Vec::new(),
            log_finish: false,
            own_pages_only: false,
            slot_model: None,
        }
    }

    fn add(&mut self, r: GenRequest) -> Result<(), ink_sched::AddError> {
        self.submit_ms.insert(r.id, self.rt.now_ms());
        self.s.add(r, self.rt.now_ms())
    }

    fn done(&self, id: u64) -> bool {
        self.done.get(&id).copied().unwrap_or(false)
    }

    fn out(&self, id: u64) -> &[i32] {
        self.out.get(&id).map_or(&[], Vec::as_slice)
    }

    fn handle(&mut self, events: Vec<SeqEvent>) {
        for e in events {
            if self.log_finish
                && let Some(f) = &e.finish
            {
                let reason = match f {
                    Finish::Done => "done",
                    Finish::SessionCap => "cap",
                    Finish::Error(_) => "error",
                };
                writeln!(self.log, "F {} {} {reason}", self.nstep, e.seq).expect("write to String");
            }
            if !e.tokens.is_empty() && !self.first_ms.contains_key(&e.seq) {
                self.first_ms.insert(e.seq, self.rt.now_ms());
                self.first_order.push(e.seq);
            }
            self.out.entry(e.seq).or_default().extend_from_slice(&e.tokens);
            if let Some(c) = e.num_cached_tokens {
                self.cached.insert(e.seq, c);
            }
            if e.finish.is_some() {
                self.done.insert(e.seq, true);
            }
        }
    }

    fn check(&self) {
        if let Err(e) = self.s.check_invariants() {
            panic!("invariant broken after step {}: {e}", self.nstep);
        }
    }

    fn step(&mut self) -> bool {
        let events = self.s.take_events();
        self.handle(events);
        let now = self.rt.now_ms();
        let p = self.s.plan(now);
        self.check();
        if let Err(e) = self.s.check_plan(&p) {
            panic!("bad plan at step {}: {e}", self.nstep);
        }
        if self.own_pages_only
            && let Some(g) = p.segs.iter().find(|g| g.reset_slot && g.pos0 > 0)
        {
            panic!(
                "step {}: seq {} resets its slot at position {}",
                self.nstep, g.seq, g.pos0
            );
        }
        if p.is_empty() {
            return false;
        }
        if let Some(g) = p
            .segs
            .iter()
            .find(|g| g.kind == SegKind::Prefill && g.width <= 8 && !g.samples)
        {
            // a segment of <= 8 rows runs as a decode segment: only a prompt's last chunk may be that short
            panic!(
                "step {}: seq {} prefill segment of {} rows before its end",
                self.nstep, g.seq, g.width
            );
        }
        if let Some(m) = &mut self.slot_model {
            m.before(&p);
        }
        record_plan(&mut self.log, self.nstep, now, &p);
        self.nstep += 1;
        let t0 = self.rt.now_ms();
        let r = self.rt.step(&p).expect("fake runtime never fails");
        self.step_ms.push(self.rt.now_ms() - t0);
        self.step_decode_rows.push(p.decode_rows);
        self.step_has_prefill.push(p.prefill_rows > 0);
        self.step_prefill_ms
            .push(p.predicted_ms - self.s.decode_cost_ms(p.decode_rows as usize));
        let events = self.s.commit(&p, &r, self.rt.now_ms());
        if let Some(m) = &mut self.slot_model {
            m.after(&p, &r);
        }
        self.handle(events);
        self.check();
        true
    }

    fn run(&mut self) {
        while self.step() {}
    }

    fn until_done(&mut self, id: u64) -> bool {
        while !self.done(id) {
            if !self.step() {
                break;
            }
        }
        self.done(id)
    }

    fn summary(&mut self, name: &str, v: f64) {
        record_summary(&mut self.log, name, v);
    }

    /// Appends the counter lines and returns the whole stream.
    fn finish_log(mut self) -> String {
        let k = *self.s.counters();
        for (name, v) in [
            ("prefix_queries", k.prefix_queries),
            ("prefix_hits", k.prefix_hits),
            ("prompt_tokens", k.prompt_tokens),
            ("generation_tokens", k.generation_tokens),
            ("spec_draft", k.spec_draft),
            ("spec_accepted", k.spec_accepted),
            ("preemptions", k.preemptions),
            ("evictions", k.evictions),
            ("rejections", k.rejections),
            ("steps", k.steps),
        ] {
            writeln!(self.log, "C {name} {v}").expect("write to String");
        }
        self.log
    }

    /// Drives plan/step/commit directly (as two C++ tests do), checking each plan with `check_plan`.
    fn run_manual(&mut self, mut check_plan: impl FnMut(&StepPlan)) {
        loop {
            let p = self.s.plan(self.rt.now_ms());
            if p.is_empty() {
                break;
            }
            check_plan(&p);
            let r = self.rt.step(&p).expect("fake runtime never fails");
            let events = self.s.commit(&p, &r, self.rt.now_ms());
            self.handle(events);
        }
    }
}

/// Nearest-rank percentile, as the C++ tests' `pct`.
fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    let r = (p / 100.0 * v.len() as f64).ceil() - 1.0;
    v[r.clamp(0.0, (v.len() - 1) as f64) as usize]
}

/// Value of an `S` line in a recorded stream.
fn summary_of(log: &str, name: &str) -> f64 {
    let prefix = format!("S {name} ");
    let line = log
        .lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no summary {name}"));
    let hex = line[prefix.len()..].split(' ').next().expect("hex field");
    f64::from_bits(u64::from_str_radix(hex, 16).expect("hex bits"))
}

#[test]
fn admission_limit_4_and_8() {
    for lim in [4, 8] {
        let c = Config {
            max_running: lim,
            slots: 8,
            ..Config::default()
        };
        let mut sim = Sim::new(c);
        for i in 1..=10 {
            sim.add(plain(i, prompt_of(i, 500), 20)).expect("admitted");
        }
        sim.step();
        assert_eq!(sim.s.running(), lim as usize);
        assert_eq!(sim.s.waiting(), 10 - lim as usize);
        let mut max_run = 0;
        while sim.step() {
            max_run = max_run.max(sim.s.running());
        }
        assert!(max_run <= lim as usize);
        assert!((1..=10).all(|i| sim.done(i)));
    }
}

#[test]
fn queue_order_priority_fifo() {
    let mut sim = Sim::new(Config {
        max_running: 1,
        ..Config::default()
    });
    sim.add(req(1, prompt_of(1, 300), 5, "", 0, "")).expect("ok");
    sim.add(req(2, prompt_of(2, 300), 5, "", 0, "")).expect("ok");
    sim.add(req(3, prompt_of(3, 300), 5, "", 5, "")).expect("ok");
    sim.add(req(4, prompt_of(4, 300), 5, "", 0, "")).expect("ok");
    sim.run();
    assert_eq!(sim.first_order, [3, 1, 2, 4]);
}

#[test]
fn tokens_exact_and_pages_return_to_baseline() {
    let mut sim = Sim::new(Config {
        park_on_finish: false,
        stall_budget_ms: 60.0, // C++ parity fixture
        ..Config::default()
    });
    let base = sim.s.pages_free();
    let plen = |i: u64| 700 + 997 * i as usize;
    let max_new = |i: u64| 50 + 13 * i as u32;
    for i in 1..=12 {
        sim.add(plain(i, prompt_of(i, plen(i)), max_new(i))).expect("ok");
    }
    let mut steps = 0;
    while sim.step() {
        steps += 1;
        if steps == 40 {
            sim.s.cancel(5, sim.rt.now_ms());
        }
        if steps == 60 {
            sim.s.finish(7, sim.rt.now_ms());
        }
    }
    let events = sim.s.take_events();
    sim.handle(events);
    for i in 1..=12 {
        assert!(sim.done(i), "request {i} not done");
        if i == 5 || i == 7 {
            continue;
        }
        let o = sim.out(i);
        assert_eq!(o.len(), max_new(i) as usize);
        let want: Vec<i32> = (0..o.len())
            .map(|k| FakeRuntime::token_at(i, (plen(i) + k) as i64))
            .collect();
        assert_eq!(o, want.as_slice(), "request {i}");
    }
    assert_eq!(sim.s.pages_free(), base);
    assert_eq!(sim.s.pages_used(), 0);
    assert_fixture("tokens_exact", &sim.finish_log());
}

#[test]
fn over_cap_rejected() {
    let mut sim = Sim::new(Config {
        session_cap_tokens: 4096,
        ..Config::default()
    });
    let e = sim.add(plain(1, prompt_of(1, 5000), 10)).expect_err("over the cap");
    assert!(e.to_string().contains("per-session cap of 4096"), "{e}");
    sim.add(plain(2, prompt_of(2, 4000), 500)).expect("ok");
    sim.run();
    assert!(sim.done(2));
    assert_eq!(sim.out(2).len(), 96); // generation stops at the cap (4,096 - 4,000)
    let bad = Config {
        session_cap_tokens: 1_048_577,
        ..Config::default()
    };
    assert_eq!(Scheduler::new(bad).err(), Some(ConfigError::SessionCap));
    let ok = Scheduler::new(Config {
        session_cap_tokens: 1_048_576,
        ..Config::default()
    })
    .expect("the model maximum is allowed");
    assert_eq!(ok.config().session_cap_tokens, 1_048_576);
}

/// Full pages of a match of `m` tokens minus the SWA window (defaults: 1,024-token pages and window).
fn aligned_reuse(m: u32) -> u32 {
    (m - 1024) / 1024 * 1024
}

#[test]
fn prefix_reuse_multi_turn() {
    let mut sim = Sim::new(Config::default());
    let p1 = prompt_of(10, 5000);
    sim.add(req(1, p1.clone(), 50, "S", 0, "")).expect("ok");
    assert!(sim.until_done(1));
    assert_eq!(sim.cached[&1], 0);
    assert_eq!(sim.s.parked(), 1);
    // turn 2: whole previous turn + 2,000 new tokens -> exact resume of everything computed
    let mut p2 = p1;
    p2.extend_from_slice(sim.out(1));
    p2.extend(prompt_of(11, 2000));
    sim.add(req(2, p2.clone(), 30, "S", 0, "")).expect("ok");
    assert!(sim.until_done(2));
    assert_eq!(sim.cached[&2], 5000 + 50 - 1);
    assert_eq!(sim.s.parked(), 1); // turn 1's copy dropped (prefix of turn 2)
    // turn 3 diverges at 3,000: partial reuse at a page boundary minus the SWA window
    let mut p3 = p2;
    p3[3000] ^= 1;
    sim.add(req(3, p3, 10, "S", 0, "")).expect("ok");
    assert!(sim.until_done(3));
    assert_eq!(sim.cached[&3], aligned_reuse(3000));
    // another session, same salt: shares full pages of the resident prefix (no write to shared pages)
    let mut p4 = prompt_of(10, 5000);
    p4.push(42);
    sim.add(req(4, p4, 10, "T", 0, "")).expect("ok");
    assert!(sim.until_done(4));
    assert_eq!(sim.cached[&4], aligned_reuse(5000));
    // different salt: nothing shared
    sim.add(req(5, prompt_of(10, 5000), 10, "U", 0, "other-salt"))
        .expect("ok");
    assert!(sim.until_done(5));
    assert_eq!(sim.cached[&5], 0);
    let k = *sim.s.counters();
    assert_eq!(k.prefix_hits, 5049 + 1024 + 3072);
    println!(
        "  prefix reuse: queries {} tokens, hits {} tokens",
        k.prefix_queries, k.prefix_hits
    );
    for i in 1..=5 {
        let v = f64::from(sim.cached[&i]);
        sim.summary(&format!("cached_{i}"), v);
    }
    assert_fixture("prefix_reuse", &sim.finish_log());
}

#[test]
fn eviction_lru_under_pressure() {
    let mut sim = Sim::new(Config {
        pool_tokens: 16 * 1024,
        session_cap_tokens: 16 * 1024, // one session at the cap must fit the pool
        ..Config::default()
    });
    for i in 1..=4 {
        // 4 parked sessions of 3 pages each (3,000 + 20 tokens)
        sim.add(req(i, prompt_of(i, 3000), 20, &format!("s{i}"), 0, ""))
            .expect("ok");
        assert!(sim.until_done(i));
    }
    assert_eq!(sim.s.parked(), 4);
    assert_eq!(sim.s.pages_used(), 12);
    sim.add(plain(9, prompt_of(9, 9000), 10)).expect("ok"); // needs 9 pages: evicts the two least recently used
    assert!(sim.until_done(9));
    assert!(sim.s.counters().evictions >= 2);
    // session 1 (oldest) lost its KV, session 4 (newest) kept it
    sim.add(req(11, prompt_of(1, 3000), 5, "s1", 0, "")).expect("ok");
    assert!(sim.until_done(11));
    assert_eq!(sim.cached[&11], 0);
    sim.add(req(14, prompt_of(4, 3000), 5, "s4", 0, "")).expect("ok");
    assert!(sim.until_done(14));
    assert!(sim.cached[&14] > 0);
    let (c11, c14) = (f64::from(sim.cached[&11]), f64::from(sim.cached[&14]));
    sim.summary("cached_11", c11);
    sim.summary("cached_14", c14);
    assert_fixture("eviction", &sim.finish_log());
}

#[test]
fn preemption_when_pool_exhausted() {
    let mut sim = Sim::new(Config {
        pool_tokens: 12 * 1024,
        session_cap_tokens: 12 * 1024, // one session at the cap must fit the pool
        park_on_finish: false,
        stall_budget_ms: 60.0, // C++ parity fixture
        ..Config::default()
    });
    for i in 1..=3 {
        sim.add(plain(i, prompt_of(i, 3000), 2500)).expect("ok");
    }
    sim.run();
    let preemptions = sim.s.counters().preemptions;
    assert!(preemptions > 0);
    for i in 1..=3 {
        assert!(sim.done(i));
        let want: Vec<i32> = (0..2500).map(|k| FakeRuntime::token_at(i, 3000 + k)).collect();
        assert_eq!(sim.out(i), want.as_slice(), "request {i}");
    }
    assert_eq!(sim.s.pages_used(), 0);
    assert_eq!(preemptions, 121, "C++ scheduler: 121 preemptions");
    println!("  preemption: {preemptions} preemptions, outputs identical to the unpreempted token sequence");
    assert_fixture("preemption", &sim.finish_log());
}

#[test]
fn cancel_frees() {
    let mut sim = Sim::new(Config::default());
    sim.add(plain(1, prompt_of(1, 4000), 1000)).expect("ok");
    for _ in 0..10 {
        sim.step();
    }
    assert!(sim.s.pages_used() > 0);
    sim.s.cancel(1, sim.rt.now_ms());
    let events = sim.s.take_events();
    sim.handle(events);
    assert!(sim.done(1));
    assert_eq!(sim.s.pages_used(), 0);
    assert_eq!(sim.s.running(), 0);
}

#[test]
fn step_shape_limits() {
    let mut sim = Sim::new(Config {
        max_running: 8,
        ..Config::default()
    });
    for i in 1..=8 {
        sim.add(plain(i, prompt_of(i, 3000 + 100 * i as usize), 200))
            .expect("ok");
    }
    sim.run_manual(|p| {
        let prefill: Vec<&Segment> = p.segs.iter().filter(|g| g.kind == SegKind::Prefill).collect();
        assert!(prefill.len() <= 1);
        assert!(prefill.iter().all(|g| g.width <= 2048));
        assert!(
            p.segs
                .iter()
                .filter(|g| g.kind == SegKind::Decode)
                .all(|g| g.width <= 8)
        );
        assert!(p.decode_rows + p.prefill_rows <= 2112);
    });
    assert!((1..=8).all(|i| sim.done(i)));
    let k = sim.s.counters();
    assert!(k.spec_draft > 0 && k.spec_accepted > 0 && k.spec_accepted <= k.spec_draft);
}

#[test]
fn width_hook() {
    let mut sim = Sim::new(Config {
        width_hook: Some(WidthHook(Arc::new(|_, _| 3))),
        ..Config::default()
    });
    sim.add(plain(1, prompt_of(1, 500), 100)).expect("ok");
    let mut seen = false;
    sim.run_manual(|p| {
        for g in p.segs.iter().filter(|g| g.kind == SegKind::Decode) {
            assert!(g.width <= 4);
            seen |= g.width == 4;
        }
    });
    assert!(seen);
}

/// 3 sessions decode (k = 7, 45K contexts) while a 4th session sends a 6,000-token turn on a 45K cached prefix.
/// Simulated with the cost model (config B coefficients), not measured. Returns the recorded stream.
fn interleave_scenario(c: &Config, label: &str) -> String {
    let mut sim = Sim::new(c.clone());
    let px = prompt_of(100, 45056);
    sim.add(req(100, px.clone(), 1, "X", 0, "")).expect("ok");
    assert!(sim.until_done(100));
    for i in 1..=3 {
        sim.add(plain(i, prompt_of(i, 45056), 200_000)).expect("ok");
    }
    // run until the three decoders are all decoding
    for _ in 0..100_000 {
        sim.step();
        if (1..=3).all(|i| !sim.out(i).is_empty()) {
            break;
        }
    }
    for _ in 0..20 {
        sim.step();
    }
    let decode_only = *sim.step_ms.last().expect("steps ran");
    let mut p2 = px;
    p2.extend_from_slice(sim.out(100));
    p2.extend(prompt_of(101, 6000));
    sim.add(req(200, p2, 50, "X", 0, "")).expect("ok");
    let first_step = sim.step_ms.len();
    while !sim.first_ms.contains_key(&200) {
        sim.step();
    }
    let ttft = sim.first_ms[&200] - sim.submit_ms[&200];
    let (mut worst, mut budget_excess) = (0.0_f64, 0.0_f64);
    for k in first_step..sim.step_ms.len() {
        worst = worst.max(sim.step_ms[k]);
        if sim.step_has_prefill[k] {
            budget_excess = budget_excess.max(sim.step_prefill_ms[k] - c.stall_budget_ms);
        }
    }
    let cached = sim.cached[&200];
    println!(
        "  [simulated with cost model, not measured] {label}: TTFT {ttft:.0} ms (cached {cached}), decode-only step \
         {decode_only:.1} ms, worst step during prefill {worst:.1} ms (decoder stall {:.1}x)",
        worst / decode_only
    );
    sim.summary("ttft_ms", ttft);
    sim.summary("cached_200", f64::from(cached));
    sim.summary("decode_only_ms", decode_only);
    sim.summary("worst_step_ms", worst);
    sim.summary("stall_x", worst / decode_only);
    sim.summary("budget_excess_ms", budget_excess);
    sim.finish_log()
}

#[test]
fn interleave_ttft_and_stall() {
    let c = Config {
        stall_budget_ms: 60.0, // C++ parity fixture: budget 60 ms, min chunk 128, urgency 0.8 x 5 s
        ..Config::default()
    };
    let log = interleave_scenario(&c, "rule (budget 60 ms)");
    assert!(summary_of(&log, "ttft_ms") <= c.ttft_target_ms);
    assert!(summary_of(&log, "budget_excess_ms") <= 1e-6); // no urgency: every chunk stayed inside the budget
    let stall = summary_of(&log, "stall_x");
    assert!(stall < 3.5);
    assert_fixture("interleave_rule", &log);
    let b = Config {
        stall_budget_ms: 1e9, // config B-like: always a 2,048-token chunk
        ..Config::default()
    };
    let log_b = interleave_scenario(&b, "fixed 2,048 chunk (config B shape)");
    assert!(summary_of(&log_b, "stall_x") > stall);
    assert_fixture("interleave_fixed", &log_b);
}

#[test]
fn default_chunk_policy_uses_max_chunk_under_decode_load() {
    let s = Scheduler::new(Config::default()).expect("default config is valid");
    assert!(s.config().stall_budget_ms.is_infinite());
    assert_eq!(s.config().max_chunk, 2048);
    // 4 or 8 sessions decoding (32 or 64 rows): still chooses full max_chunk across all context lengths
    for decode_rows in [0, 1, 8, 32, 64] {
        for ctx in [0, 16_384, 49_152, 90_112, 200_000] {
            assert_eq!(s.choose_chunk(decode_rows, ctx, &[]), 2048);
        }
    }
    // clamps cleanly to t_max - decode_rows when decode rows eat into the 2,048-row prefill room
    assert_eq!(s.choose_chunk(80, 49_152, &[]), 2112 - 80);
    // honours smaller explicit max_chunk
    let s512 = Scheduler::new(Config {
        max_chunk: 512,
        ..Config::default()
    })
    .expect("valid config");
    assert_eq!(s512.choose_chunk(32, 49_152, &[]), 512);
    // end-to-end: default config matches the fixed 2,048-row interleave fixture
    let log_def = interleave_scenario(&Config::default(), "default (unthrottled 2,048 chunk)");
    assert_fixture("interleave_fixed", &log_def);
}

struct Arm {
    name: &'static str,
    label: &'static str,
    scale: f64,
    order: PrefillOrder,
    budget: f64,
}

/// Closed-loop agent workload: 4 sessions x 20 turns, first turns staggered by 15 s (30-51K cold prompts), then
/// uncached suffixes per turn shaped like the fleet's p50 / p90 / p99 (1.6K / 6K / 19K tokens: P9 README:72-73),
/// outputs 124 / 344 / 871 tokens. Simulated with the cost model, not measured. Returns the recorded stream.
fn closed_loop(arm: &Arm) -> String {
    let mut c = Config {
        prefill_order: arm.order,
        stall_budget_ms: arm.budget,
        ..Config::default()
    };
    if arm.budget > 1e8 {
        c.urgent_stall_cap_ms = 1e9;
    }
    let mut sim = Sim::with_cost(
        c.clone(),
        FakeCost {
            scale: arm.scale,
            ..FakeCost::default()
        },
    );
    const SESSIONS: usize = 4;
    const TURNS: usize = 20;
    // per session: 13 x 1.6K, 3 x 3K, 3 x 6K, 1 x 19K (the 19K turn of each session at a different turn)
    let suffix_of = |s: usize, t: usize| {
        const PAT: [usize; 19] = [
            1600, 1600, 3000, 1600, 6000, 1600, 1600, 1600, 3000, 1600, 6000, 1600, 1600, 1600, 3000, 1600, 6000, 1600,
            1600,
        ];
        if t == 4 + 4 * s {
            19000
        } else {
            PAT[(t - 1 + 5 * s) % 19]
        }
    };
    const OUTLEN: [u32; 3] = [124, 344, 871];
    let mut hist: Vec<Vec<i32>> = vec![Vec::new(); SESSIONS];
    let mut turn = [0_usize; SESSIONS];
    let (mut warm, mut cold, mut t6k, mut t19k) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut next_id = 1_u64;
    let mut owner: BTreeMap<u64, usize> = BTreeMap::new();
    let mut suffix: BTreeMap<u64, usize> = BTreeMap::new();
    let mut submit = |sim: &mut Sim, hist: &mut [Vec<i32>], s: usize, t: usize| {
        let add = if t == 0 { 30000 + 7000 * s } else { suffix_of(s, t) };
        hist[s].extend(prompt_of(5000 + s as u64 * 100 + t as u64, add));
        let id = next_id;
        next_id += 1;
        sim.add(req(
            id,
            hist[s].clone(),
            OUTLEN[(t + s) % 3],
            &format!("sess{s}"),
            0,
            "",
        ))
        .expect("ok");
        (id, add)
    };
    let mut started = [false; SESSIONS];
    let mut seen_done: BTreeMap<u64, bool> = BTreeMap::new();
    let mut finished_sessions = 0;
    while finished_sessions < SESSIONS {
        for s in 0..SESSIONS {
            if !started[s] && sim.rt.now_ms() >= 15000.0 * s as f64 {
                started[s] = true;
                let (id, add) = submit(&mut sim, &mut hist, s, turn[s]);
                owner.insert(id, s);
                suffix.insert(id, add);
            }
        }
        if !sim.step() {
            sim.rt.advance(10.0); // idle: let virtual time pass to the next start
        }
        let newly: Vec<u64> = sim
            .done
            .iter()
            .filter(|&(id, &d)| d && !seen_done.contains_key(id))
            .map(|(&id, _)| id)
            .collect();
        for id in newly {
            seen_done.insert(id, true);
            let s = owner[&id];
            let t = sim.first_ms.get(&id).copied().unwrap_or(0.0) - sim.submit_ms[&id];
            if turn[s] == 0 {
                cold.push(t);
            } else {
                warm.push(t);
                match suffix[&id] {
                    6000 => t6k.push(t),
                    19000 => t19k.push(t),
                    _ => {}
                }
            }
            let out = sim.out(id).to_vec();
            hist[s].extend_from_slice(&out);
            turn[s] += 1;
            if turn[s] < TURNS {
                let (nid, add) = submit(&mut sim, &mut hist, s, turn[s]);
                owner.insert(nid, s);
                suffix.insert(nid, add);
            } else {
                finished_sessions += 1;
            }
        }
    }
    let dec_steps: Vec<f64> = sim
        .step_ms
        .iter()
        .zip(&sim.step_decode_rows)
        .filter(|&(_, &rows)| rows > 0)
        .map(|(&ms, _)| ms)
        .collect();
    let k = *sim.s.counters();
    let hit_pct = 100.0 * k.prefix_hits as f64 / k.prefix_queries.max(1) as f64;
    println!(
        "  [simulated with cost model, not measured] {}: warm TTFT (n={}) p50 {:.0} / p90 {:.0} / p99 {:.0} ms; 6K \
         turns max {:.0} ms; 19K turns {:.0}-{:.0} ms; cold first turns {:.0}-{:.0} ms; steps with decode rows p50 \
         {:.1} / p99 {:.1} / max {:.1} ms; prefix hits {hit_pct:.1} %",
        arm.label,
        warm.len(),
        pct(&warm, 50.0),
        pct(&warm, 90.0),
        pct(&warm, 99.0),
        pct(&t6k, 100.0),
        pct(&t19k, 0.0),
        pct(&t19k, 100.0),
        pct(&cold, 0.0),
        pct(&cold, 100.0),
        pct(&dec_steps, 50.0),
        pct(&dec_steps, 99.0),
        pct(&dec_steps, 100.0)
    );
    assert_eq!(warm.len(), SESSIONS * (TURNS - 1));
    if arm.scale == 1.0 && arm.budget < 1e8 {
        assert!(pct(&warm, 90.0) <= c.ttft_target_ms);
        assert!(pct(&t6k, 100.0) <= c.ttft_target_ms);
        assert!(pct(&dec_steps, 100.0) <= c.cost_base_ms + 64.0 * c.cost_row_ms + c.urgent_stall_cap_ms + 1.0);
    }
    for (name, v) in [
        ("warm_n", warm.len() as f64),
        ("warm_p50_ms", pct(&warm, 50.0)),
        ("warm_p90_ms", pct(&warm, 90.0)),
        ("warm_p99_ms", pct(&warm, 99.0)),
        ("ttft_6k_max_ms", pct(&t6k, 100.0)),
        ("ttft_19k_min_ms", pct(&t19k, 0.0)),
        ("ttft_19k_max_ms", pct(&t19k, 100.0)),
        ("cold_min_ms", pct(&cold, 0.0)),
        ("cold_max_ms", pct(&cold, 100.0)),
        ("decode_step_p50_ms", pct(&dec_steps, 50.0)),
        ("decode_step_p99_ms", pct(&dec_steps, 99.0)),
        ("decode_step_max_ms", pct(&dec_steps, 100.0)),
        ("prefix_hit_pct", hit_pct),
    ] {
        sim.summary(name, v);
    }
    sim.finish_log()
}

#[test]
fn interleave_closed_loop_p99() {
    for arm in [
        Arm {
            name: "closed_loop_fifo",
            label: "rule, FIFO",
            scale: 1.0,
            order: PrefillOrder::Fifo,
            budget: 60.0,
        },
        Arm {
            name: "closed_loop_srpt",
            label: "rule, SRPT",
            scale: 1.0,
            order: PrefillOrder::Srpt,
            budget: 60.0,
        },
        Arm {
            name: "closed_loop_slow",
            label: "rule, FIFO, runtime 1.5x slower than estimate",
            scale: 1.5,
            order: PrefillOrder::Fifo,
            budget: 60.0,
        },
        Arm {
            name: "closed_loop_fixed",
            label: "fixed 2,048 chunk (config B shape)",
            scale: 1.0,
            order: PrefillOrder::Fifo,
            budget: 1e9,
        },
    ] {
        let log = closed_loop(&arm);
        assert_fixture(arm.name, &log);
    }
}

/// The paths the other fixtures leave out (review S2): priorities, width hook, session cap, slot stealing, eviction
/// score with a size weight, salts, small pages, SRPT, parking on cancel, rejection, adoption with outputs.
#[test]
fn mixed_paths() {
    let c = Config {
        slots: 4,
        max_running: 3,
        page_tokens: 256,
        pool_tokens: 40 * 256,
        session_cap_tokens: 6000,
        swa_recompute_tokens: 512,
        evict_size_weight: 0.37,
        park_on_cancel: true,
        prefill_order: PrefillOrder::Srpt,
        stall_budget_ms: 45.0,
        width_hook: Some(WidthHook(Arc::new(|id, k| {
            u32::try_from((id * 3 + u64::from(k)) % 9).expect("below 9")
        }))),
        ..Config::default()
    };
    let mut sim = Sim::new(c);
    sim.log_finish = true;
    let n = 16_usize;
    let mut prompts: Vec<Vec<i32>> = vec![Vec::new(); n];
    let mut rejected = 0_u32;
    for step in 0..200_000_usize {
        if step % 6 == 0 && step / 6 < n {
            let i = step / 6;
            let p = if i >= 5 {
                let mut p = prompts[i - 5].clone();
                let prev = (i - 4) as u64;
                if sim.done(prev) {
                    p.extend_from_slice(sim.out(prev));
                }
                p.extend(prompt_of(200 + i as u64, 300 + 211 * (i % 7)));
                p
            } else {
                prompt_of(200 + i as u64, if i == 4 { 6000 } else { 900 + 400 * i })
            };
            prompts[i] = p.clone();
            let max_new = if i == 2 { 100_000 } else { 40 + 37 * (i as u32 % 6) };
            let salt = if i % 4 == 3 { "z" } else { "" };
            let rq = req(i as u64 + 1, p, max_new, &format!("m{}", i % 5), (i % 3) as i32, salt);
            if sim.add(rq).is_err() {
                rejected += 1;
            }
        }
        if step == 50 {
            sim.s.cancel(4, sim.rt.now_ms());
        }
        if step == 90 {
            sim.s.finish(7, sim.rt.now_ms());
        }
        if !sim.step() {
            if step > 6 * n {
                break;
            }
            sim.rt.advance(5.0);
        }
    }
    let events = sim.s.take_events();
    sim.handle(events);
    sim.summary("rejected", f64::from(rejected));
    let k = *sim.s.counters();
    assert!(k.preemptions > 0 && k.evictions > 0 && k.prefix_hits > 0 && k.rejections == 3);
    let log = sim.finish_log();
    assert!(log.contains(" cap\n"), "the session cap ends one request");
    assert_fixture("mixed", &log);
}

#[test]
fn engine_backend_threaded_smoke() {
    let be =
        EngineBackend::spawn(Config::default(), || Ok::<_, Infallible>(FakeRuntime::default())).expect("valid config");
    let out: Arc<Mutex<BTreeMap<u64, Vec<i32>>>> = Arc::default();
    let rejected: Arc<Mutex<Vec<StepOutput>>> = Arc::default();
    let done = Arc::new(AtomicUsize::new(0));
    for i in 1..=6 {
        let (out, done) = (Arc::clone(&out), Arc::clone(&done));
        be.submit(
            plain(i, prompt_of(i, 1000), 64),
            Box::new(move |o: StepOutput| {
                out.lock()
                    .expect("lock")
                    .entry(i)
                    .or_default()
                    .extend_from_slice(&o.tokens);
                if o.done {
                    done.fetch_add(1, Ordering::SeqCst);
                }
            }),
        );
    }
    {
        let (rejected, done) = (Arc::clone(&rejected), Arc::clone(&done));
        be.submit(
            plain(99, prompt_of(99, 300_000), 5),
            Box::new(move |o: StepOutput| {
                rejected.lock().expect("lock").push(o);
                done.fetch_add(1, Ordering::SeqCst);
            }),
        );
    }
    let t0 = Instant::now();
    while done.load(Ordering::SeqCst) < 7 && t0.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(done.load(Ordering::SeqCst), 7);
    let rejected = rejected.lock().expect("lock");
    assert_eq!(rejected.len(), 1);
    assert!(rejected[0].error.is_some() && rejected[0].done);
    let out = out.lock().expect("lock");
    for i in 1..=6 {
        assert_eq!(out[&i].len(), 64);
        assert_eq!(out[&i][0], FakeRuntime::token_at(i, 1000));
    }
    // metrics are published after the iteration that delivered the last callback
    let mut m = be.metrics();
    while m.generation_tokens != 6 * 64 && t0.elapsed() < Duration::from_secs(3) {
        std::thread::sleep(Duration::from_millis(1));
        m = be.metrics();
    }
    assert_eq!(m.running, 0);
    assert_eq!(m.generation_tokens, 6 * 64);
}

/// Review S2: the session cap reaches `StepOutput`; a dead loop thread is unhealthy and answers submits with an error.
type Outs = Arc<Mutex<Vec<StepOutput>>>;

fn collect(outs: &Outs) -> ink_backend::StepFn {
    let outs = Arc::clone(outs);
    Box::new(move |o| outs.lock().expect("lock").push(o))
}

#[test]
fn engine_backend_cap_and_dead_loop() {
    let wait_for = |outs: &Outs, n: usize| {
        let t0 = Instant::now();
        while outs.lock().expect("lock").iter().filter(|o| o.done).count() < n && t0.elapsed() < Duration::from_secs(2)
        {
            std::thread::sleep(Duration::from_millis(1));
        }
    };

    let be = EngineBackend::spawn(
        Config {
            session_cap_tokens: 1100,
            ..Config::default()
        },
        || Ok::<_, Infallible>(FakeRuntime::default()),
    )
    .expect("valid config");
    assert!(be.healthy());
    let outs: Outs = Arc::default();
    be.submit(plain(1, prompt_of(1, 1000), 500), collect(&outs));
    wait_for(&outs, 1);
    {
        let outs = outs.lock().expect("lock");
        let last = outs.last().expect("outputs");
        assert!(last.done && last.session_cap && last.error.is_none());
        assert_eq!(outs.iter().map(|o| o.tokens.len()).sum::<usize>(), 100);
        assert!(outs.iter().rev().skip(1).all(|o| !o.session_cap && !o.done));
    }

    struct Panics;
    impl ModelRuntime for Panics {
        fn caps(&self) -> ink_sched::RuntimeCaps {
            ink_sched::RuntimeCaps::default()
        }
        fn step(&mut self, _: &StepPlan) -> Result<Vec<ink_sched::SegResult>, ink_sched::StepError> {
            panic!("runtime bug (expected by the test)");
        }
        fn now_ms(&self) -> f64 {
            0.0
        }
    }
    let be = EngineBackend::spawn(Config::default(), || Ok::<_, Infallible>(Panics)).expect("valid config");
    be.submit(plain(1, prompt_of(1, 100), 5), Box::new(|_| {}));
    let t0 = Instant::now();
    while be.healthy() && t0.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!be.healthy());
    let outs: Outs = Arc::default();
    be.submit(plain(2, prompt_of(2, 100), 5), collect(&outs));
    let outs = outs.lock().expect("lock");
    assert!(outs.len() == 1 && outs[0].done && outs[0].error.as_deref() == Some("scheduler loop stopped"));
}

/// Review S2: a sequence of a failed step is freed even with `park_on_cancel` (its slot state is unknown).
#[test]
fn abort_never_parks() {
    let mut sim = Sim::new(Config {
        park_on_cancel: true,
        ..Config::default()
    });
    sim.add(plain(1, prompt_of(1, 3000), 50)).expect("ok");
    for _ in 0..3 {
        sim.step();
    }
    sim.s.cancel(1, sim.rt.now_ms());
    assert_eq!(sim.s.parked(), 1, "cancel parks with park_on_cancel");
    sim.add(plain(2, prompt_of(2, 3000), 50)).expect("ok");
    for _ in 0..3 {
        sim.step();
    }
    let parked_before = sim.s.parked();
    sim.s.abort(2, sim.rt.now_ms());
    assert_eq!(sim.s.parked(), parked_before, "abort frees instead of parking");
    assert_eq!(sim.s.running() + sim.s.waiting(), 0);
    sim.check();
    let events = sim.s.take_events();
    assert!(events.iter().any(|e| e.seq == 2 && e.finish == Some(Finish::Done)));
}

#[test]
fn config_rejects_unschedulable_shapes() {
    for c in [
        Config {
            t_max: 0,
            ..Config::default()
        },
        Config {
            max_prefill_rows: 0,
            ..Config::default()
        },
        Config {
            max_decode_rows: 0,
            ..Config::default()
        },
        Config {
            t_max: 4,
            ..Config::default()
        },
    ] {
        assert_eq!(Scheduler::new(c).err(), Some(ConfigError::Rows));
    }
    // a lone session at the cap must fit the pool (it would otherwise preempt itself and recompute forever)
    let c = Config {
        pool_tokens: 16 * 1024 + 100,
        session_cap_tokens: 16 * 1024 + 1,
        ..Config::default()
    };
    assert_eq!(
        Scheduler::new(c).err(),
        Some(ConfigError::SessionCapOverPool { pool: 16 * 1024 })
    );
}

/// Review S2 (found by the property test): pages shared only among parked sequences are freed by eviction, so they
/// count for admission. The C++ counted refcount-1 pages only, and request 4 here waited forever.
#[test]
fn admission_counts_pages_shared_by_parked() {
    let mut sim = Sim::new(Config {
        pool_tokens: 6 * 1024,
        session_cap_tokens: 6 * 1024,
        max_running: 1,
        slots: 1,
        page_tokens: 64,
        swa_recompute_tokens: 0,
        ..Config::default()
    });
    let p2 = prompt_of(9, 4353);
    let mut p3 = p2.clone();
    p3.extend(prompt_of(10, 1));
    sim.add(plain(1, prompt_of(8, 1), 1)).expect("ok");
    sim.add(req(3, p3, 13, "s0", 1, "x")).expect("ok");
    for _ in 0..13 {
        sim.step();
    }
    sim.add(req(2, p2, 1, "s0", 0, "x")).expect("ok"); // shares 68 pages with parked request 3
    sim.add(plain(4, prompt_of(11, 1793), 1)).expect("ok"); // 29 pages: needs the shared pages evicted
    sim.run();
    for id in 1..=4 {
        assert!(sim.done(id), "request {id} not done");
    }
}

/// `PrefixReuse::ExactResume` (a runtime with its own page tables): a parked sequence resumes exactly in its slot;
/// nothing else is reused, no page is shared and no slot is reset above position 0.
#[test]
fn exact_resume_reuses_parked_prefixes_only() {
    let mut sim = Sim::new(Config {
        prefix_reuse: PrefixReuse::ExactResume,
        ..Config::default()
    });
    sim.own_pages_only = true;
    let p1 = prompt_of(10, 5000);
    sim.add(req(1, p1.clone(), 50, "S", 0, "")).expect("ok");
    assert!(sim.until_done(1));
    let mut p2 = p1;
    p2.extend_from_slice(sim.out(1));
    p2.extend(prompt_of(11, 2000));
    sim.add(req(2, p2.clone(), 30, "S", 0, "")).expect("ok");
    assert!(sim.until_done(2));
    assert_eq!(sim.cached[&2], 5000 + 50 - 1, "exact resume of everything computed");
    // divergence, another session with the same prefix, a parked prefix whose slot was taken: all recomputed
    let mut p3 = p2.clone();
    p3[3000] ^= 1;
    sim.add(req(3, p3, 10, "S", 0, "")).expect("ok");
    assert!(sim.until_done(3));
    let mut p4 = prompt_of(10, 5000);
    p4.push(42);
    sim.add(req(4, p4, 10, "T", 0, "")).expect("ok");
    assert!(sim.until_done(4));
    assert_eq!((sim.cached[&3], sim.cached[&4]), (0, 0));
    // turn 2 is still parked in its own slot: its continuation resumes exactly
    let mut p5 = p2;
    p5.extend_from_slice(sim.out(2));
    p5.push(7);
    sim.add(req(5, p5, 5, "S", 0, "")).expect("ok");
    assert!(sim.until_done(5));
    assert_eq!(sim.cached[&5], 5000 + 50 + 2000 + 30 - 1);
}

/// `PrefixReuse::SlotRollback`: a follow-up turn whose prompt extends the previous prompt but not the previous
/// reply resumes in the parked slot at the previous prompt's end (the checkpoint); a whole-sequence prefix still
/// resumes exactly; another session or salt gets nothing. Every plan is checked against a model of the runtime's
/// slots (lengths, rollbacks only to a checkpoint, pages per slot).
#[test]
fn slot_rollback_resumes_at_the_previous_prompt_end() {
    let mut sim = Sim::new(Config {
        prefix_reuse: PrefixReuse::SlotRollback,
        trim_parked_pages: false,
        spec_k: 7,
        drafting: Drafting::Any,
        ..Config::default()
    });
    sim.own_pages_only = true;
    sim.slot_model = Some(SlotModel::new(8));
    let p1 = prompt_of(10, 5000);
    sim.add(req(1, p1.clone(), 300, "S", 0, "")).expect("ok");
    assert!(sim.until_done(1));
    // turn 2 carries another reply than the one generated (teacher-forced history): resume at 5000
    let mut p2 = p1.clone();
    p2.extend(prompt_of(12, 300));
    p2.extend(prompt_of(11, 2000));
    sim.add(req(2, p2.clone(), 40, "S", 0, "")).expect("ok");
    assert!(sim.until_done(2));
    assert_eq!(sim.cached[&2], 5000, "resumed at the previous prompt's end");
    // turn 3 extends turn 2 with its own reply: everything computed resumes, the reply's decode rows included
    let mut p3 = p2.clone();
    p3.extend_from_slice(sim.out(2));
    p3.push(7);
    sim.add(req(3, p3.clone(), 20, "S", 0, "")).expect("ok");
    assert!(sim.until_done(3));
    assert_eq!(sim.cached[&3], 7300 + 40 - 1);
    // turn 4 extends turn 3's prompt but not its reply: back to turn 3's prompt end (its two-token prefill)
    let mut p4 = p3.clone();
    p4.extend(prompt_of(13, 900));
    sim.add(req(4, p4, 10, "S", 0, "")).expect("ok");
    assert!(sim.until_done(4));
    assert_eq!(
        sim.cached[&4] as usize,
        p3.len(),
        "turn 3's checkpoint at the end of its prompt"
    );
    // another salt and a diverging prompt get nothing
    let mut p5 = p1;
    p5.push(1);
    sim.add(req(5, p5, 10, "T", 0, "other")).expect("ok");
    assert!(sim.until_done(5));
    assert_eq!(sim.cached[&5], 0);
    assert!(sim.slot_model.as_ref().expect("model").rollbacks >= 2);
}

/// A runtime with its own page tables and pages narrows the scheduler: exact-resume reuse, its pool, a session cap
/// within it; every plan names the slots nobody holds.
#[test]
fn engine_backend_narrows_to_a_runtime_with_own_pages() {
    struct OwnPages {
        inner: FakeRuntime,
        plans: Arc<Mutex<Vec<StepPlan>>>,
    }
    impl ModelRuntime for OwnPages {
        fn caps(&self) -> RuntimeCaps {
            RuntimeCaps {
                kv_pages: Some(8),
                ..RuntimeCaps::default()
            }
        }
        fn step(&mut self, plan: &StepPlan) -> Result<Vec<SegResult>, StepError> {
            self.plans.lock().expect("lock").push(plan.clone());
            self.inner.step(plan)
        }
        fn now_ms(&self) -> f64 {
            self.inner.now_ms()
        }
    }
    let plans: Arc<Mutex<Vec<StepPlan>>> = Arc::default();
    let be = EngineBackend::spawn(Config::default(), {
        let plans = Arc::clone(&plans);
        move || {
            Ok::<_, Infallible>(OwnPages {
                inner: FakeRuntime::default(),
                plans,
            })
        }
    })
    .expect("the session cap is narrowed to the runtime's 8 pages, not refused");
    let outs: Outs = Arc::default();
    be.submit(plain(1, prompt_of(1, 9000), 5), collect(&outs));
    be.submit(plain(2, prompt_of(2, 3000), 5), collect(&outs));
    let t0 = Instant::now();
    while outs.lock().expect("lock").iter().filter(|o| o.done).count() < 2 && t0.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let outs = outs.lock().expect("lock");
    let errors: Vec<&str> = outs.iter().filter_map(|o| o.error.as_deref()).collect();
    assert_eq!(
        errors,
        ["prompt of 9000 tokens exceeds the per-session cap of 8192 tokens"],
        "{outs:?}"
    );
    assert!(outs.iter().any(|o| o.done && o.error.is_none()), "{outs:?}");
    let plans = plans.lock().expect("lock");
    assert!(!plans.is_empty());
    for p in plans.iter() {
        // unused slots, and the slot of a sequence that starts over in this step
        let kept: Vec<u32> = p.segs.iter().filter(|g| !g.reset_slot).map(|g| g.slot).collect();
        let want: Vec<u32> = (0..8).filter(|s| !kept.contains(s)).collect();
        assert_eq!(p.stale_slots, want, "stale slots of {p:?}");
    }
}

/// One random request of the property test.
#[derive(Debug, Clone)]
struct RandReq {
    prompt: usize,
    max_new: u32,
    session: u8,
    salt: bool,
    priority: i32,
    /// Prefix of another request's prompt (multi-turn shape) when set.
    extends: Option<usize>,
    arrive_step: usize,
}

fn rand_req() -> impl Strategy<Value = RandReq> {
    (
        1_usize..5000,
        1_u32..400,
        0_u8..4,
        any::<bool>(),
        0_i32..3,
        proptest::option::of(0_usize..8),
        0_usize..60,
    )
        .prop_map(
            |(prompt, max_new, session, salt, priority, extends, arrive_step)| RandReq {
                prompt,
                max_new,
                session,
                salt,
                priority,
                extends,
                arrive_step,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Invariants (refcounts = owners, exact free list, consistent slots; every plan writes only unshared pages it
    /// owns) hold after every step under random arrivals, shared prefixes, small pools (eviction, preemption), few
    /// slots (stealing), page sizes, recompute windows, eviction weights, cancels and finishes; at the end every
    /// request is done and only parked sequences own pages.
    #[test]
    fn invariants_hold_every_step(
        reqs in proptest::collection::vec(rand_req(), 1..12),
        pool_pages in 6_u64..40,
        max_running in 1_u32..8,
        extra_slots in 0_u32..3,
        page_tokens in proptest::sample::select(vec![64_u32, 256, 1024]),
        swa_recompute_tokens in proptest::sample::select(vec![0_u32, 512, 1024]),
        evict_size_weight in proptest::sample::select(vec![0.0_f64, 3.0]),
        park_on_cancel in any::<bool>(),
        stops in proptest::collection::vec((0_usize..200, 1_u64..13, any::<bool>()), 0..6),
    ) {
        let c = Config {
            pool_tokens: pool_pages * 1024,
            session_cap_tokens: pool_pages * 1024,
            max_running,
            slots: max_running + extra_slots,
            page_tokens,
            swa_recompute_tokens,
            evict_size_weight,
            park_on_cancel,
            ..Config::default()
        };
        let mut sim = Sim::new(c);
        let mut prompts: Vec<Vec<i32>> = Vec::new();
        for (i, r) in reqs.iter().enumerate() {
            let mut p = match r.extends {
                Some(j) if j < prompts.len() => prompts[j].clone(),
                _ => Vec::new(),
            };
            p.extend(prompt_of(i as u64 + 7, r.prompt));
            prompts.push(p);
        }
        let mut accepted = Vec::new();
        for step in 0..20_000 {
            for (i, r) in reqs.iter().enumerate() {
                if r.arrive_step == step {
                    let id = i as u64 + 1;
                    let rq = req(id, prompts[i].clone(), r.max_new, &format!("s{}", r.session), r.priority,
                                 if r.salt { "x" } else { "" });
                    if sim.add(rq).is_ok() {
                        accepted.push(id);
                    }
                }
            }
            for &(at, id, cancel) in &stops {
                if at == step {
                    if cancel { sim.s.cancel(id, sim.rt.now_ms()) } else { sim.s.finish(id, sim.rt.now_ms()) }
                }
            }
            if !sim.step() && step > 200 {
                break;
            }
        }
        let events = sim.s.take_events();
        sim.handle(events);
        prop_assert!(sim.s.check_invariants().is_ok());
        for id in accepted {
            prop_assert!(sim.done(id), "request {} not done", id);
        }
        prop_assert_eq!(sim.s.running() + sim.s.waiting(), 0);
    }
}

/// A runtime that is not `Send` (it holds an `Rc`, as a CUDA context handle would) is built and used on the loop
/// thread only.
#[test]
fn spawn_builds_the_runtime_on_the_loop_thread() {
    struct Local {
        inner: FakeRuntime,
        built_on: std::thread::ThreadId,
        _not_send: std::rc::Rc<()>,
    }
    impl ModelRuntime for Local {
        fn caps(&self) -> ink_sched::RuntimeCaps {
            ink_sched::RuntimeCaps {
                max_slots: 3,
                ..self.inner.caps()
            }
        }
        fn step(&mut self, plan: &StepPlan) -> Result<Vec<ink_sched::SegResult>, ink_sched::StepError> {
            assert_eq!(
                std::thread::current().id(),
                self.built_on,
                "step runs where the runtime was built"
            );
            self.inner.step(plan)
        }
        fn now_ms(&self) -> f64 {
            self.inner.now_ms()
        }
    }
    let caller = std::thread::current().id();
    let be = EngineBackend::spawn(Config::default(), move || {
        let built_on = std::thread::current().id();
        assert_ne!(built_on, caller, "the constructor does not run on the caller's thread");
        Ok::<_, Infallible>(Local {
            inner: FakeRuntime::default(),
            built_on,
            _not_send: std::rc::Rc::new(()),
        })
    })
    .expect("spawns");
    assert_eq!(be.caps().max_slots, 3, "the caps come back from the loop thread");
    let outs: Outs = Arc::default();
    be.submit(plain(1, prompt_of(1, 100), 5), collect(&outs));
    let t0 = Instant::now();
    while !outs.lock().expect("lock").iter().any(|o| o.done) && t0.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let outs = outs.lock().expect("lock");
    assert!(outs.iter().any(|o| o.done && o.error.is_none()), "{outs:?}");
    assert!(be.healthy());
}

#[test]
fn spawn_reports_constructor_and_config_failures() {
    #[derive(Debug, thiserror::Error)]
    #[error("no device")]
    struct NoDevice;
    let e = EngineBackend::spawn(Config::default(), || Err::<FakeRuntime, _>(NoDevice)).expect_err("fails");
    assert!(
        matches!(&e, SpawnError::Runtime(source) if source.to_string() == "no device"),
        "{e:?}"
    );
    let e = EngineBackend::spawn(Config::default(), || -> Result<FakeRuntime, Infallible> {
        panic!("constructor bug (expected by the test)")
    })
    .expect_err("fails");
    assert!(matches!(e, SpawnError::Panicked), "{e:?}");
    let e = EngineBackend::spawn(
        Config {
            max_decode_rows: 0,
            ..Config::default()
        },
        || Ok::<_, Infallible>(FakeRuntime::default()),
    )
    .expect_err("fails");
    assert!(matches!(e, SpawnError::Config(_)), "{e:?}");
}

/// Panic policy: a panic in the scheduler, the runtime or a callback is fatal for the loop. The back end turns
/// unhealthy, answers new requests with an error and drops the callbacks of requests in flight.
#[test]
fn a_panicking_callback_is_fatal() {
    let be =
        EngineBackend::spawn(Config::default(), || Ok::<_, Infallible>(FakeRuntime::default())).expect("valid config");
    assert!(be.healthy());
    let bystander: Outs = Arc::default();
    be.submit(plain(1, prompt_of(1, 100), 100_000), collect(&bystander));
    be.submit(
        plain(2, prompt_of(2, 100), 5),
        Box::new(|_| panic!("callback bug (expected by the test)")),
    );
    let t0 = Instant::now();
    while be.healthy() && t0.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!be.healthy());
    let outs: Outs = Arc::default();
    be.submit(plain(3, prompt_of(3, 100), 5), collect(&outs));
    let outs = outs.lock().expect("lock");
    assert!(outs.len() == 1 && outs[0].done && outs[0].error.as_deref() == Some("scheduler loop stopped"));
    be.cancel(1);
    be.finish(3); // nothing left to stop; must not panic
}

fn greedy_req(id: u64, n: usize, max_new: u32) -> GenRequest {
    let mut r = plain(id, prompt_of(id, n), max_new);
    r.sampling.temperature = 0.0;
    r
}

/// A fixed-width greedy drafter (`Drafting::Fixed`, the C++ runtime's DSpark): decode segments are 1 or exactly
/// 1 + k rows; 1 + k only for greedy requests without logprobs whose rows stay below the drafter's positions, also
/// past the token budget (the front end drops the overshoot).
#[test]
fn fixed_greedy_drafter_widths() {
    let cfg = Config {
        drafting: Drafting::Fixed {
            k: 7,
            ctx: 1100,
            sampled: false,
        },
        trim_parked_pages: false,
        prefix_reuse: PrefixReuse::ExactResume,
        ..Config::default()
    };
    let mut s = Scheduler::new(cfg.clone()).expect("config");
    let mut lp = greedy_req(3, 300, 6);
    lp.sampling.num_logprobs = Some(1);
    for r in [
        greedy_req(1, 1000, 200),
        greedy_req(2, 300, 3),
        lp,
        plain(4, prompt_of(4, 300), 6),
    ] {
        s.add(r, 0.0).expect("add");
    }
    let mut rt = FakeRuntime::default();
    let mut widths: BTreeMap<u64, Vec<(u32, u32)>> = BTreeMap::new();
    for _ in 0..400 {
        let p = s.plan(rt.now_ms());
        if p.is_empty() {
            break;
        }
        for g in p.segs.iter().filter(|g| g.kind == SegKind::Decode) {
            widths.entry(g.seq).or_default().push((g.pos0, g.width));
        }
        let r = rt.step(&p).expect("fake step");
        s.commit(&p, &r, rt.now_ms());
        s.check_invariants().expect("invariants");
    }
    for (seq, w) in &widths {
        assert!(w.iter().all(|&(_, w)| w == 1 || w == 8), "seq {seq}: {w:?}");
    }
    // greedy: 8 rows while pos0 + 8 <= 1100, then 1
    assert!(
        widths[&1].iter().all(|&(p, w)| w == if p + 8 <= 1100 { 8 } else { 1 }),
        "{:?}",
        widths[&1]
    );
    assert!(widths[&1].iter().any(|&(_, w)| w == 1));
    assert!(
        widths[&2].iter().all(|&(_, w)| w == 8),
        "verify rows past a budget of 3: {:?}",
        widths[&2]
    );
    assert!(widths[&3].iter().all(|&(_, w)| w == 1), "logprobs: {:?}", widths[&3]);
    assert!(
        widths[&4].iter().all(|&(_, w)| w == 1),
        "temperature 1: {:?}",
        widths[&4]
    );
    assert_eq!(
        Scheduler::new(Config { spec_k: 3, ..cfg }).err(),
        Some(ConfigError::FixedWidth { k: 7 })
    );
}

/// With a device sampled accept (`Drafting::Fixed { sampled: true }`) sampled requests speculate too; logprobs still
/// decode one row per step.
#[test]
fn fixed_drafter_speculates_sampled_requests() {
    let mut s = Scheduler::new(Config {
        drafting: Drafting::Fixed {
            k: 7,
            ctx: 4096,
            sampled: true,
        },
        trim_parked_pages: false,
        prefix_reuse: PrefixReuse::ExactResume,
        ..Config::default()
    })
    .expect("config");
    let mut lp = plain(2, prompt_of(2, 300), 20);
    lp.sampling.num_logprobs = Some(1);
    for r in [plain(1, prompt_of(1, 300), 20), lp, greedy_req(3, 300, 20)] {
        s.add(r, 0.0).expect("add");
    }
    let mut rt = FakeRuntime::default();
    let mut widths: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
    for _ in 0..400 {
        let p = s.plan(rt.now_ms());
        if p.is_empty() {
            break;
        }
        for g in p.segs.iter().filter(|g| g.kind == SegKind::Decode) {
            widths.entry(g.seq).or_default().push(g.width);
        }
        let r = rt.step(&p).expect("fake step");
        s.commit(&p, &r, rt.now_ms());
        s.check_invariants().expect("invariants");
    }
    assert!(widths[&1].iter().all(|&w| w == 8), "temperature 1: {:?}", widths[&1]);
    assert!(widths[&2].iter().all(|&w| w == 1), "logprobs: {:?}", widths[&2]);
    assert!(widths[&3].iter().all(|&w| w == 8), "greedy: {:?}", widths[&3]);
}

/// A parked sequence returns the pages past its computed tokens only with `trim_parked_pages`; without it (a runtime
/// with its own page tables) it keeps the page its last verify rows reached.
#[test]
fn parked_pages_are_trimmed_only_when_configured() {
    for trim in [true, false] {
        let mut s = Scheduler::new(Config {
            drafting: Drafting::Fixed {
                k: 7,
                ctx: 4096,
                sampled: false,
            },
            trim_parked_pages: trim,
            ..Config::default()
        })
        .expect("config");
        s.add(greedy_req(1, 1020, 2), 0.0).expect("add");
        let mut rt = FakeRuntime::default();
        let mut decoded = 0;
        while s.running() + s.waiting() > 0 {
            let p = s.plan(rt.now_ms());
            let decode = p.segs.iter().any(|g| g.kind == SegKind::Decode);
            if decode {
                assert_eq!(
                    p.segs[0].pages.len(),
                    2,
                    "8 rows from position 1,020 reach a second page"
                );
            }
            let res = rt.step(&p).expect("fake step");
            for e in s.commit(&p, &res, rt.now_ms()) {
                if decode {
                    decoded += e.tokens.len();
                }
            }
        }
        assert_eq!(s.parked(), 1);
        let computed = 1020 + decoded;
        let want = if trim { computed.div_ceil(1024) } else { 2 };
        assert_eq!(s.pages_used(), want, "trim {trim}, {computed} computed");
    }
}

/// At `N == slots`, when two sessions park within `slot_steal_grace_ms` (e.g. a 10K live prefix and a 512-token
/// finished conversation finishing on the same or adjacent decode steps), a new turn-0 arrival steals the smaller
/// parked slot instead of destroying the large prefix. Once a parked slot ages past the grace window, stale slots
/// are always stolen before fresh ones.
#[test]
fn acquire_slot_protects_fresh_long_prefix_and_steals_stale_first() {
    let mut s = Scheduler::new(Config {
        max_running: 2,
        slots: 2,
        pool_tokens: 65_536,
        session_cap_tokens: 65_536,
        prefix_reuse: PrefixReuse::SlotRollback,
        slot_steal_grace_ms: 150.0,
        ..Config::default()
    })
    .expect("config");

    let long_p = prompt_of(10, 4096);
    let short_p = prompt_of(20, 256);
    // Long finishes slightly BEFORE short (t=100 vs t=120), both inside the 150 ms grace window at t=130.
    s.add(plain(1, long_p.clone(), 1), 0.0).expect("add long");
    let mut rt = FakeRuntime::default();
    while s.running() + s.waiting() > 0 {
        let p = s.plan(100.0);
        let r = rt.step(&p).expect("step");
        s.commit(&p, &r, 100.0);
    }
    s.add(plain(2, short_p, 1), 110.0).expect("add short");
    while s.running() + s.waiting() > 0 {
        let p = s.plan(120.0);
        let r = rt.step(&p).expect("step");
        s.commit(&p, &r, 120.0);
    }
    assert_eq!(s.parked(), 2);

    // At t=130 (ages: long=30 ms, short=10 ms — both inside 150 ms grace), new turn-0 must steal short's slot.
    let new_p = prompt_of(30, 128);
    s.add(plain(3, new_p, 1), 130.0).expect("add new");
    let p3 = s.plan(130.0);
    assert_eq!(p3.segs[0].seq, 3);
    let stolen_slot = p3.segs[0].slot;
    let r3 = rt.step(&p3).expect("step");
    s.commit(&p3, &r3, 130.0);

    // Long's next turn arrives at t=135: its slot was preserved so it gets full prefix reuse!
    let mut long_t2 = long_p.clone();
    long_t2.extend(prompt_of(11, 32));
    s.add(plain(4, long_t2, 1), 135.0).expect("add long t2");
    let p4 = s.plan(135.0);
    assert_eq!(p4.segs[0].seq, 4);
    assert_ne!(p4.segs[0].slot, stolen_slot, "long prefix kept its own slot");
    assert_eq!(p4.segs[0].pos0, 4096, "long prefix resumed without re-prefill");
    let r4 = rt.step(&p4).expect("step");
    s.commit(&p4, &r4, 135.0);

    // Now at t=400, seq 3 (parked at 130, age=270ms >= 150ms) and seq 4 (parked at 135, age=265ms >= 150ms)
    // are both stale; park a fresh tiny seq at t=390 (seq 5 steals seq 3 at t=390 -> age 10ms at t=400).
    let fresh_p = prompt_of(40, 64);
    s.add(plain(5, fresh_p.clone(), 1), 390.0).expect("add fresh");
    let p5 = s.plan(390.0);
    let fresh_slot = p5.segs[0].slot;
    let r5 = rt.step(&p5).expect("step");
    s.commit(&p5, &r5, 390.0);

    // At t=400: seq 4 (large, 4128 tok, age=265ms >= grace) is stale, seq 5 (tiny, 64 tok, age=10ms < grace) is fresh.
    // A new turn-0 arrival must steal the stale seq 4, NOT the freshly-parked seq 5.
    s.add(plain(6, prompt_of(50, 128), 1), 400.0).expect("add turn0");
    let p6 = s.plan(400.0);
    assert_ne!(p6.segs[0].slot, fresh_slot, "stale slot stolen before fresh slot");
    s.check_invariants().expect("invariants");
}
