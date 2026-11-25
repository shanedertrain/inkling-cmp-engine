//! Speculative decoding with the DSpark drafter, over [`Runtime`] and [`Drafter`]: the modes of
//! `engine/native/tools/spec.cpp` (packages A2, T2, and cut-over prep sampler modes `sampled`, `sdist`, `stime`).
//!
//! ```text
//! ink-spec --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir DIR --mode greedy --cases cases.txt \
//!          --sessions 4 [--only id,..] [--path N] --fold /s2f/fold.bin [--trace trace.jsonl] > spec.jsonl
//! ink-spec ... --mode sampled --sessions 4 [--temperature 1 --top-p 0.95 --seed 20261123] [--path 400] > sampled.jsonl
//! ink-spec ... --mode sdist --sessions 8 --iters 4096 --plain 200000 --out DIR > sdist.jsonl
//! ink-spec ... --mode stime --sessions 4 --ctx 45000 --reps 2000 > stime.jsonl
//! ink-spec ... --mode time --sessions 4 --ctx 45000 --secs 24 --slots 4 > time.jsonl
//! ink-spec ... --mode golden --sessions 4 --path 600 --passes 2 --slots 4 > golden.jsonl
//! ```
//!
//! `time`: free-running speculative rounds over real contexts (ms per round, draft / verify split, tokens/s, NVML).
//! `golden`: timed speculative greedy decoding of the golden prompts (tokens/s, tokens/J). Both with every session
//! `--gate on|off` (default on, as `spec.cpp`): each session verifies 1 + the leading drafts whose cumulative drafter
//! probability stays at or above `--tau` (default `VERIFY_TAU` 0.7), rounded up to 1 / 2 / 4 / 8 rows, segments in
//! width order; off: 8 rows, no probability pass. `--conf-trace` (survivors) is not ported.
//!
//! `sampled`, `sdist`, `stime` require `--runtime rust` (`Runtime::sampler_probe`).
//!
//! Mode `greedy`:
//!
//! The golden greedy prompts, `--sessions` together: speculative greedy (drafter + 8-row verify + device accept) for
//! `L` tokens per case (the golden length, or `--path N`), then a plain one-row pass teacher-forced along the
//! speculative tokens; every token that is not the plain argmax is counted with its logit gap. Output: the C++
//! tool's `spec_path` lines and the `spec_path_summary` line, byte for byte. `--trace`: one line per speculative
//! round with the drafter's drafts and the accept results, the bit-level record of the seam (`spec.cpp --trace`).
//! `--draft-log` (any mode): every drafter call's inputs and the drafts, the oracle of gate E8 (`ink_tools::draftlog`).
#![deny(unsafe_code)] // allowed only at the calls of the runtime's bounds-contract methods
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use ink_record::{Component, ComponentName};
use ink_record::{RecordWriter, Source};
use ink_rt_api::{
    AcceptBufs, CallRows, DRAFTS, DevBuf, DevSeg, DeviceSampling, DraftConfig, Drafter, FEAT_ROW_BYTES, HeadPrecision,
    KvPrecision, LOGIT_ROW, MAX_DRAFTS, Runtime, SampleTestOut, SampleTestRows, SamplerProbe, SegIn, SpecRuntime,
    VERIFY_TAU, VOCAB, VerifyWidth,
};
use ink_runtime::lookup::NgramIndex;
use ink_tools::cases::{Case, read_cases};
use ink_tools::draftlog::{Header, LogWriter, Op};
use ink_tools::engine::{EngineArgs, MAX_CHUNK, RecordArgs, identity_fields};
use ink_tools::line::Line;
use ink_tools::tele::{Card, Tele};
use serde_json::json;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    Greedy,
    Sampled,
    Sdist,
    Stime,
    Time,
    Golden,
}

#[derive(Parser)]
#[command(name = "ink-spec", version = ink_record::BUILD_LINE, about = "Speculative decoding with DSpark")]
struct Cli {
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    record: RecordArgs,
    /// `greedy`: path check; `sampled`: speculative vs plain sampled; `sdist`: distribution test; `stime`: sampler vs
    /// argmax timing; `time`: free-running rounds; `golden`: timed golden prompts.
    #[arg(long, value_enum, default_value = "time")]
    mode: Mode,
    /// Golden cases (`cases.txt` of `gold_prep.py`).
    #[arg(long, default_value = "/gin/cases.txt")]
    cases: PathBuf,
    /// Sessions decoding together (engine slots and drafter rows).
    #[arg(long, default_value_t = 4)]
    sessions: u32,
    /// Context the KV pool and the drafter are sized for.
    #[arg(long, default_value_t = 45_000)]
    ctx: u32,
    /// Drafter positions per session (default max(ctx + 8192, 16384)).
    #[arg(long)]
    draft_ctx: Option<u32>,
    /// KV pages (default sessions x ceil(draft_ctx / 1024) + 8).
    #[arg(long)]
    pages: Option<u32>,
    /// Folded drafter weights (DSpark).
    #[arg(long, default_value = "/s2f/fold.bin")]
    fold: PathBuf,
    /// Optional DFlash2 checkpoint directory (`model.safetensors`); uses DFlash2 instead of DSpark (`--runtime rust`).
    #[arg(long)]
    dflash2: Option<PathBuf>,
    /// Drafter global-layer K/V: 0 bf16, 1 FP8 E4M3.
    #[arg(long, default_value_t = 0)]
    kv8: u8,
    /// Comma-separated case ids (default: every greedy case).
    #[arg(long, default_value = "")]
    only: String,
    /// Tokens per case (0: the golden length in `greedy`, 400 in `sampled`, 600 in `golden`).
    #[arg(long, default_value_t = 0)]
    path: usize,
    /// Logit gap above which a token that is not the plain argmax counts against the gate.
    #[arg(long, default_value_t = 0.5)]
    tie_margin: f64,
    /// One line per speculative round: drafts and accept results.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Drafter log: every drafter call's inputs and the drafts it returned (gate E8's oracle).
    #[arg(long)]
    draft_log: Option<PathBuf>,
    /// `time`: seconds of rounds.
    #[arg(long, default_value_t = 24.0)]
    secs: f64,
    /// `time`: seconds at the start that are dropped.
    #[arg(long, default_value_t = 2.0)]
    discard: f64,
    /// `time` / `stime`: case whose prompt and tokens are the token stream.
    #[arg(long, default_value = "tf-agent-120k")]
    stream: String,
    /// `golden`: passes over the set (pass 1 captures the graphs, the last is reported).
    #[arg(long, default_value_t = 2)]
    passes: u32,
    /// Probability-gated verify width (`spec.cpp --gate`): on | off.
    #[arg(long, default_value = "on")]
    gate: String,
    /// Gate on: the verify call's rows per session `m` it is rounded up to (`spec.cpp --call-rows`, the server's
    /// `--verify-call-rows`): a comma list holding 8, or `ragged`.
    #[arg(long, default_value = ink_rt_api::VERIFY_CALL_ROWS)]
    call_rows: String,
    /// Threshold of the gate, a probability in (0, 1] (gate on; default `VERIFY_TAU`).
    #[arg(long)]
    tau: Option<f32>,
    /// Per-round conf file (`spec.cpp --conf-trace`; not ported, refused).
    #[arg(long)]
    conf_trace: Option<PathBuf>,
    /// `sampled` / `sdist` / `stime`: softmax temperature.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    /// `sampled` / `sdist` / `stime`: nucleus mass.
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    /// `sampled` / `sdist` / `stime`: minimum probability relative to the mode (0: off).
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    /// `sampled` / `sdist` / `stime`: top-k candidates (0: off).
    #[arg(long, default_value_t = 0)]
    top_k: u32,
    /// `sampled` / `sdist` / `stime`: base Philox seed.
    #[arg(long, default_value_t = 20_261_123)]
    seed: u64,
    /// `sdist`: directory for verify-row `*.keys.bin` dumps.
    #[arg(long, default_value = "/out")]
    out: PathBuf,
    /// `sdist`: probe rounds per context (each round samples `sessions` draws).
    #[arg(long, default_value_t = 4096)]
    iters: usize,
    /// `sdist`: plain draws per context (disjoint seeds).
    #[arg(long, default_value_t = 200_000)]
    plain: usize,
    /// `sdist`: comma-separated K offsets into each case's reference tokens; negative `-n` picks `n` contexts automatically.
    #[arg(long, default_value = "-2")]
    ctx_tokens: String,
    /// `sdist`: maximum rows per `sample_test_rows` chunk.
    #[arg(long, default_value_t = 1_048_576)]
    test_rows: usize,
    /// `stime`: repetitions of `argmax` and `sample` over the verify rows.
    #[arg(long, default_value_t = 2000)]
    reps: usize,
    /// Verbatim n-gram lookup drafts appended after the drafter's 7 drafts (`0` = off, `8` = up to 8 copied tokens).
    #[arg(long, default_value_t = 0)]
    lookup: u32,
    /// Minimum backward verbatim match length `m` required to trigger lookup drafts.
    #[arg(long, default_value_t = 8)]
    lookup_min: u32,
}

/// Rows per prefill call (and size of the row-index ramp).
const CHUNK: usize = MAX_CHUNK as usize;
/// Gap thresholds of the histogram.
const EDGES: [f64; 5] = [0.1, 0.25, 0.5, 1.0, 2.0];
/// `kSampFlagMismatch` (`engine/kernels/sample/sample.hpp`).
const K_SAMP_FLAG_MISMATCH: u64 = 8;

fn spec_of(rt: &mut dyn Runtime) -> Result<&mut dyn SpecRuntime> {
    rt.spec().context("the runtime has no device-side speculation")
}

fn probe_of(rt: &mut dyn Runtime) -> Result<&mut dyn SamplerProbe> {
    rt.sampler_probe().context("the runtime has no sampler probe")
}

fn ctl(cli: &Cli, seed: u64) -> DeviceSampling {
    DeviceSampling {
        seed,
        temperature: cli.temperature,
        top_p: cli.top_p,
        min_p: cli.min_p,
        top_k: cli.top_k,
    }
}

fn emit_id(line: Line, source: &Source, card: &Card, cli: &Cli) -> Line {
    line.identity(source)
        .text("uuid", &card.uuid)
        .text("vbios", &card.vbios)
        .num("power_limit_w", card.power_limit_w)
        .num("temperature", f64::from(cli.temperature))
        .num("top_p", f64::from(cli.top_p))
        .num("min_p", f64::from(cli.min_p))
        .uint("top_k", u64::from(cli.top_k))
}

fn digest(st: &[u64]) -> String {
    st[..6].iter().map(u64::to_string).collect::<Vec<_>>().join(":")
}

fn i32s(rt: &mut dyn Runtime, len: usize) -> Result<DevBuf> {
    Ok(spec_of(rt)?.alloc(len * 4)?)
}

/// First index of the largest valid logit (`spec.cpp` `argmax_first`).
fn argmax_first(row: &[f32]) -> usize {
    (1..VOCAB).fold(0, |a, k| if row[k] > row[a] { k } else { a })
}

/// The gate's threshold as `spec.cpp` prints it (`std::to_string` of the probability, or "off").
fn tau_text(cli: &Cli) -> String {
    if cli.gate == "on" {
        format!("{:.6}", cli.tau.unwrap_or(VERIFY_TAU))
    } else {
        "off".to_owned()
    }
}

fn id(i: usize) -> i32 {
    i32::try_from(i).expect("vocabulary ids fit i32")
}

/// The device buffers of the speculation loop (`spec.cpp` `Spec`).
struct Spec {
    /// Drafter sessions (= the slots that speculate).
    n: usize,
    /// Drafter positions per session.
    ctx: usize,
    lookup_len: usize,
    lookup_min: usize,
    draft_k: usize,
    max_verify_rows: usize,
    res_stride: usize,
    bufs: AcceptBufs,
    vids: DevBuf,
    iota: DevBuf,
    pos: DevBuf,
    sess: DevBuf,
    trace: Option<BufWriter<File>>,
    log: Option<LogWriter<BufWriter<File>>>,
    rounds_done: u64,
    /// The gate's threshold (log of the probability), when on.
    tau: Option<f32>,
    /// Verify widths of every slot (device, host; 8 with the gate off).
    wdev: DevBuf,
    w: Vec<i32>,
    /// Rounding of the verify call's rows (`--call-rows`).
    call_rows: CallRows,
    anchor_host: Vec<i32>,
    lookup: Vec<NgramIndex>,
    drafts_host: Vec<i32>,
    all_drafts: Vec<i32>,
    lookup_triggers: u64,
    lookup_accepted: u64,
}

/// `bytes` bytes (a multiple of 4) at the start of `buf`, read back after the queued work.
fn device_bytes(rt: &mut dyn Runtime, buf: DevBuf, bytes: usize) -> Result<Vec<u8>> {
    let mut v = vec![0i32; bytes / 4];
    spec_of(rt)?.download(buf, &mut v)?;
    Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect())
}

fn device_i32s(rt: &mut dyn Runtime, buf: DevBuf, n: usize) -> Result<Vec<i32>> {
    let mut v = vec![0i32; n];
    spec_of(rt)?.download(buf, &mut v)?;
    Ok(v)
}

/// A drafter call to log: its device arguments.
enum Call {
    /// features, positions, sessions, rows.
    Commit(DevBuf, DevBuf, DevBuf, usize),
    /// last committed rows' features.
    Frnn(DevBuf),
    /// anchors, positions.
    State(DevBuf, DevBuf),
    /// the drafted sessions, the drafts after the call.
    DraftSessions(Vec<usize>, DevBuf),
}

/// Callback run after `forward_dev` and before `spec_accept` in a speculative round (`(slot, rows)` in call order).
type PreAccept<'a> = &'a mut dyn FnMut(&mut dyn Runtime, &[(usize, usize)]) -> Result<()>;

impl Spec {
    fn new(
        rt: &mut dyn Runtime,
        drafter: &dyn Drafter,
        n: usize,
        ctx: usize,
        lookup_len: usize,
        lookup_min: usize,
        trace: Option<BufWriter<File>>,
    ) -> Result<Self> {
        ensure!(DRAFTS + lookup_len <= MAX_DRAFTS, "lookup {lookup_len} exceeds max");
        let draft_k = DRAFTS + lookup_len;
        let max_verify_rows = draft_k + 1;
        let res_stride = draft_k + 2;
        let drafts_buf = if lookup_len > 0 {
            i32s(rt, draft_k * n)?
        } else {
            drafter.drafts()
        };
        let bufs = AcceptBufs {
            drafts: drafts_buf,
            anchor: i32s(rt, n)?,
            pos: i32s(rt, n)?,
            res: i32s(rt, res_stride * n + 1)?,
            crow: i32s(rt, max_verify_rows * n)?,
            cpos: i32s(rt, max_verify_rows * n)?,
            csess: i32s(rt, max_verify_rows * n)?,
            lastfeat: spec_of(rt)?.alloc(n * FEAT_ROW_BYTES)?,
            cfeats: spec_of(rt)?.alloc(CHUNK.max(max_verify_rows * n) * FEAT_ROW_BYTES)?,
        };
        let vids = i32s(rt, max_verify_rows * n)?;
        let wdev = i32s(rt, n)?;
        let iota = i32s(rt, CHUNK)?;
        let pos = i32s(rt, CHUNK)?;
        let sess = i32s(rt, CHUNK)?;
        let ramp: Vec<i32> = (0..CHUNK).map(id).collect();
        spec_of(rt)?.upload(iota, &ramp)?;
        Ok(Self {
            n,
            ctx,
            lookup_len,
            lookup_min,
            draft_k,
            max_verify_rows,
            res_stride,
            bufs,
            vids,
            iota,
            pos,
            sess,
            trace,
            log: None,
            rounds_done: 0,
            tau: None,
            wdev,
            w: vec![8; n],
            call_rows: CallRows::RAGGED,
            anchor_host: vec![0; n],
            lookup: (0..n).map(|_| NgramIndex::new()).collect(),
            drafts_host: vec![0; n * DRAFTS],
            all_drafts: vec![-1; n * draft_k],
            lookup_triggers: 0,
            lookup_accepted: 0,
        })
    }

    /// Logs one drafter call when a drafter log is open (the device buffers it was given, read back).
    fn log(&mut self, rt: &mut dyn Runtime, call: Call) -> Result<()> {
        let Some(log) = self.log.as_mut() else { return Ok(()) };
        let n = self.n;
        let op = match call {
            Call::Commit(feats, pos, sess, rows) => Op::Commit {
                feats: device_bytes(rt, feats, rows * FEAT_ROW_BYTES)?,
                pos: device_i32s(rt, pos, rows)?,
                sess: device_i32s(rt, sess, rows)?,
            },
            Call::Frnn(feats) => Op::Frnn(device_bytes(rt, feats, n * FEAT_ROW_BYTES)?),
            Call::State(anchor, pos) => Op::State {
                anchor: device_i32s(rt, anchor, n)?,
                pos: device_i32s(rt, pos, n)?,
            },
            Call::DraftSessions(sessions, drafts) => Op::DraftSessions {
                sessions: sessions.into_iter().map(id).collect(),
                drafts: device_i32s(rt, drafts, n * DRAFTS)?,
            },
        };
        log.op(&op)
    }

    /// Prefills `slot` with `ids` in chunks, committing every row to the drafter; returns the first generated token
    /// (greedy, or sampled under `samp`) and sets the slot's anchor, position and last features. `last`, when given,
    /// receives the last row's logits.
    fn prefill(
        &mut self,
        rt: &mut dyn Runtime,
        dr: &mut dyn Drafter,
        slot: usize,
        ids: &[i32],
        last: &mut Vec<f32>,
    ) -> Result<i32> {
        self.prefill_with(rt, dr, slot, ids, Some(last), None)
    }

    fn prefill_with(
        &mut self,
        rt: &mut dyn Runtime,
        dr: &mut dyn Drafter,
        slot: usize,
        ids: &[i32],
        mut last: Option<&mut Vec<f32>>,
        samp: Option<DeviceSampling>,
    ) -> Result<i32> {
        // the drafter's bounds contract of the commits below: positions < ids.len() <= ctx, session = slot < n
        ensure!(
            slot < self.n && ids.len() <= self.ctx,
            "slot {slot} with {} tokens does not fit the drafter's {} sessions of {} positions",
            ids.len(),
            self.n,
            self.ctx
        );
        rt.reset_slot(slot)?;
        let mut scratch = Vec::new();
        let mut tok_s = Vec::new();
        for (c, piece) in ids.chunks(CHUNK).enumerate() {
            let a = c * CHUNK;
            let fin = a + CHUNK >= ids.len();
            let m = piece.len();
            let nl = rt.forward(&[SegIn {
                slot,
                ids: piece,
                logits: usize::from(fin),
            }])?;
            let positions: Vec<i32> = (a..a + m).map(id).collect();
            let sp = spec_of(rt)?;
            sp.upload(self.pos, &positions)?;
            sp.upload(self.sess, &vec![id(slot); m])?;
            #[allow(unsafe_code)] // the runtime's and the drafter's bounds contracts
            // SAFETY: `iota[i] = i`: rows 0..m of the call of m rows; positions a..a + m and the session checked at entry
            unsafe {
                sp.gather_feats(self.iota, m, self.bufs.cfeats)?;
                dr.commit_rows(self.bufs.cfeats, self.pos, self.sess, m)?;
            }
            self.log(rt, Call::Commit(self.bufs.cfeats, self.pos, self.sess, m))?;
            let lastfeat = self
                .bufs
                .lastfeat
                .range(slot * FEAT_ROW_BYTES, FEAT_ROW_BYTES)
                .context("slot outside the drafter's sessions")?;
            let at = self.iota.i32s(m - 1, 1).context("chunk row")?;
            let sp = spec_of(rt)?;
            #[allow(unsafe_code)] // the runtime's bounds contract
            // SAFETY: row m - 1 of the call of m rows
            let gathered = unsafe { sp.gather_feats(at, 1, lastfeat) };
            gathered?;
            if fin {
                if let Some(out) = last.as_deref_mut() {
                    rt.logits_host(nl, out)?;
                } else if samp.is_none() {
                    rt.logits_host(nl, &mut scratch)?;
                }
                if let Some(p) = samp {
                    rt.sample_host(&[p], 1, &mut tok_s)?;
                }
            }
            rt.commit(&[u32::try_from(m)?])?;
        }
        let tok = if samp.is_some() {
            tok_s[0]
        } else if let Some(out) = last {
            id(argmax_first(out))
        } else {
            id(argmax_first(&scratch))
        };
        let sp = spec_of(rt)?;
        sp.upload(self.bufs.anchor.i32s(slot, 1).context("anchor slot")?, &[tok])?;
        sp.upload(self.bufs.pos.i32s(slot, 1).context("position slot")?, &[id(ids.len())])?;
        self.anchor_host[slot] = tok;
        if self.lookup_len > 0 {
            self.lookup[slot].clear();
            self.lookup[slot].extend_from_slice(ids);
        }
        Ok(tok)
    }

    /// One speculative round over the sessions in `act` (slots); per active session the emitted tokens. With
    /// `split`, the draft and the verify forward are each followed by a device wait and `split` receives their
    /// wall-clock milliseconds.
    fn round(
        &mut self,
        rt: &mut dyn Runtime,
        dr: &mut dyn Drafter,
        act: &[usize],
        split: Option<&mut (f64, f64)>,
    ) -> Result<Vec<Vec<i32>>> {
        self.round_with(rt, dr, act, split, None, false, None)
            .map(|(out, _)| out)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "speculative round with optional sampling, probe and hook"
    )]
    fn round_with(
        &mut self,
        rt: &mut dyn Runtime,
        dr: &mut dyn Drafter,
        act: &[usize],
        split: Option<&mut (f64, f64)>,
        samp: Option<&[DeviceSampling]>,
        probe: bool,
        pre_accept: Option<PreAccept<'_>>,
    ) -> Result<(Vec<Vec<i32>>, Vec<i32>)> {
        let t0 = Instant::now();
        // the drafter's bounds contract: a drafted session's position is its slot's length (the prefill's upload, then
        // the accept kernel's row after the committed ones), and its verify block p .. p + max_verify_rows must fit
        for &q in act {
            let p = rt.length(q)?;
            ensure!(
                q < self.n && p + self.draft_k < self.ctx,
                "slot {q} at position {p} passes the drafter's {} sessions of {} positions",
                self.n,
                self.ctx
            );
        }
        #[allow(unsafe_code)] // the drafter's bounds contract
        // SAFETY: checked above for every session drafted below; the anchors are argmax tokens (< VOCAB) or the accept
        // kernel's emitted tokens
        let state = unsafe { dr.set_state(self.bufs.anchor, self.bufs.pos) };
        state?;
        self.log(rt, Call::State(self.bufs.anchor, self.bufs.pos))?;
        dr.frnn(self.bufs.lastfeat)?;
        self.log(rt, Call::Frnn(self.bufs.lastfeat))?;
        if let Some(slot_par) = samp
            && dr.proposal().is_some()
        {
            let mut temps = vec![0.0f32; self.n];
            let mut top_ps = vec![1.0f32; self.n];
            let mut seeds = vec![0u64; self.n];
            for &q in act {
                temps[q] = slot_par[q].temperature;
                top_ps[q] = slot_par[q].top_p;
                seeds[q] = slot_par[q].seed;
            }
            dr.set_sampling(&temps, &top_ps, &seeds)?;
        }
        // the sessions of the round only (the served round's call, spec.rs)
        dr.draft_sessions(act)?;
        self.log(rt, Call::DraftSessions(act.to_vec(), dr.drafts()))?;
        if let Some(tau) = self.tau {
            let conf = dr.conf().context("the gate needs the drafter probability")?;
            spec_of(rt)?.verify_widths(conf, tau, self.n, self.wdev, &mut self.w)?;
        } else {
            self.w.fill(8);
        }
        if self.lookup_len > 0 {
            spec_of(rt)?.download(dr.drafts(), &mut self.drafts_host)?;
            self.all_drafts.fill(-1);
            let mut extra = [0i32; DRAFTS + 1];
            let mut prop = [-1i32; MAX_DRAFTS - DRAFTS];
            for &q in act {
                let ds = &self.drafts_host[q * DRAFTS..(q + 1) * DRAFTS];
                self.all_drafts[q * self.draft_k..q * self.draft_k + DRAFTS].copy_from_slice(ds);
                if self.w[q] == 8 && ds.iter().all(|&d| d >= 0) {
                    extra[0] = self.anchor_host[q];
                    extra[1..DRAFTS + 1].copy_from_slice(ds);
                    let (cnt, back) = self.lookup[q].propose(&extra, &mut prop[..self.lookup_len]);
                    if back >= self.lookup_min && cnt > 0 {
                        self.all_drafts[q * self.draft_k + DRAFTS..q * self.draft_k + DRAFTS + cnt]
                            .copy_from_slice(&prop[..cnt]);
                        self.w[q] = id(self.max_verify_rows);
                        self.lookup_triggers += 1;
                    }
                }
            }
            spec_of(rt)?.upload(self.bufs.drafts, &self.all_drafts)?;
        }
        spec_of(rt)?.build_verify_ids(self.bufs.anchor, self.bufs.drafts, self.vids, self.n, self.draft_k)?;
        if split.is_some() {
            rt.sync()?;
        }
        let t1 = Instant::now();
        // segments ordered by gated width (spec.cpp's order; rows are batch-invariant), the call's rows rounded up by
        // widening the widest (the served round's, spec.rs)
        let mut ord: Vec<usize> = (0..act.len()).collect();
        ord.sort_by_key(|&k| self.w[act[k]]);
        let mut rows: Vec<usize> = ord
            .iter()
            .map(|&k| usize::try_from(self.w[act[k]]).context("verify width"))
            .collect::<Result<_>>()?;
        self.call_rows.fill(&mut rows);
        let max_rows = self.max_verify_rows;
        let segs: Vec<DevSeg> = ord
            .iter()
            .zip(&rows)
            .map(|(&k, &w)| {
                let q = act[k];
                Ok(DevSeg {
                    slot: q,
                    ids: self.vids.i32s(max_rows * q, w).context("verify ids of a slot")?,
                    rows: w,
                    logits: w,
                })
            })
            .collect::<Result<_>>()?;
        let sp = spec_of(rt)?;
        #[allow(unsafe_code)] // the runtime's bounds contract
        // SAFETY: the verify ids are the anchors (as for set_state) and the drafter's/lookup drafts (vocabulary ids)
        let fwd = unsafe { sp.forward_dev(&segs) };
        fwd?;
        if let Some(split) = split {
            rt.sync()?;
            *split = (secs(t0, t1) * 1e3, secs(t1, Instant::now()) * 1e3);
        }
        if let Some(hook) = pre_accept {
            let called: Vec<(usize, usize)> = ord.iter().zip(&rows).map(|(&k, &w)| (act[k], w)).collect();
            hook(rt, &called)?;
        }
        let sp = spec_of(rt)?;
        let mut res = Vec::new();
        if let Some(slot_par) = samp {
            let par: Vec<DeviceSampling> = ord.iter().map(|&k| slot_par[act[k]]).collect();
            if let Some(prop) = dr.proposal() {
                sp.spec_accept_leviathan(&self.bufs, &prop, self.draft_k, &par, &mut res)?;
            } else {
                sp.spec_accept_sampled(&self.bufs, self.draft_k, &par, &mut res)?;
            }
        } else {
            sp.spec_accept(&self.bufs, self.draft_k, &mut res)?;
        }
        let s_count = act.len();
        let rs = self.res_stride;
        if let Some(trace) = &mut self.trace {
            let mut drafts = vec![0; self.draft_k * self.n];
            sp.download(self.bufs.drafts, &mut drafts)?;
            let list = |v: &[i32]| v.iter().map(i32::to_string).collect::<Vec<_>>().join(",");
            // call order: res[rs * s ..] is segment s
            let acts: Vec<i32> = ord.iter().map(|&k| id(act[k])).collect();
            writeln!(
                trace,
                "{{\"round\":{},\"act\":[{}],\"drafts\":[{}],\"res\":[{}]}}",
                self.rounds_done,
                list(&acts),
                list(&drafts),
                list(&res[..=rs * s_count])
            )?;
        }
        self.rounds_done += 1;
        let mut out = vec![Vec::new(); s_count];
        let mut kept = Vec::with_capacity(s_count);
        for (s, &k) in ord.iter().enumerate() {
            let a = usize::try_from(res[s * rs]).context("accepted count")?;
            let emitted = res[s * rs + 1..s * rs + 2 + a].to_vec();
            let q = act[k];
            if !probe {
                if self.lookup_len > 0 {
                    let old_anchor = self.anchor_host[q];
                    self.lookup[q].push(old_anchor);
                    self.lookup[q].extend_from_slice(&emitted[..a]);
                    if a > DRAFTS {
                        self.lookup_accepted += (a - DRAFTS) as u64;
                    }
                }
                self.anchor_host[q] = *emitted.last().expect("at least one emitted token");
            }
            out[k] = emitted;
            kept.push(u32::try_from(a + 1)?);
        }
        if probe {
            rt.commit(&vec![0; s_count])?;
        } else {
            rt.commit(&kept)?;
            let rows = usize::try_from(res[s_count * rs]).context("committed rows")?;
            #[allow(unsafe_code)] // the drafter's bounds contract
            // SAFETY: the accept kernel's packed rows: positions p .. p + a < p + max_verify_rows <= ctx
            let commit = unsafe { dr.commit_rows(self.bufs.cfeats, self.bufs.cpos, self.bufs.csess, rows) };
            commit?;
            self.log(
                rt,
                Call::Commit(self.bufs.cfeats, self.bufs.cpos, self.bufs.csess, rows),
            )?;
        }
        Ok((out, res))
    }
}

/// Per-case result of the plain pass along the speculative tokens.
#[derive(Default)]
struct Check {
    other: usize,
    first_at: Option<usize>,
    worst: f64,
}

/// Totals over every case.
#[derive(Default)]
struct Totals {
    tokens: usize,
    other: usize,
    over_margin: usize,
    over: [usize; 5],
    worst: f64,
    rounds: usize,
    emitted: usize,
}

/// Closing data of a mode, for the record.
type Done = serde_json::Value;

/// The greedy path check (`spec.cpp --mode greedy`).
fn greedy(
    cli: &Cli,
    rt: &mut dyn Runtime,
    dr: &mut Box<dyn Drafter>,
    sp: &mut Spec,
    greedy: &[&Case],
    source: &Source,
) -> Result<Done> {
    let n = cli.sessions as usize;
    let identity = identity_fields(source);
    let mut out = BufWriter::new(std::io::stdout());
    let mut t = Totals::default();
    let mut scratch = Vec::new();
    for b in greedy.chunks(n) {
        let s_count = b.len();
        let len: Vec<usize> = b
            .iter()
            .map(|c| if cli.path > 0 { cli.path } else { c.tokens.len() })
            .collect();
        // speculative greedy
        let mut spec: Vec<Vec<i32>> = Vec::with_capacity(s_count);
        for (s, c) in b.iter().enumerate() {
            spec.push(vec![sp.prefill(rt, dr.as_mut(), s, &c.prompt, &mut scratch)?]);
        }
        let mut rounds = vec![0_usize; s_count];
        loop {
            let act: Vec<usize> = (0..s_count).filter(|&s| spec[s].len() < len[s]).collect();
            if act.is_empty() {
                break;
            }
            let emitted = sp.round(rt, dr.as_mut(), &act, None)?;
            for (&s, toks) in act.iter().zip(emitted) {
                rounds[s] += 1;
                for tok in toks {
                    if spec[s].len() < len[s] {
                        spec[s].push(tok);
                    }
                }
            }
        }
        // plain width-1 pass teacher-forced along the speculative tokens (prefill logits predict token 0)
        let mut checks: Vec<Check> = (0..s_count).map(|_| Check::default()).collect();
        let mut check = |s: usize, k: usize, row: &[f32], t: &mut Totals| {
            let best = argmax_first(row);
            let tok = usize::try_from(spec[s][k]).expect("token ids are non-negative");
            if best == tok {
                return;
            }
            let gap = f64::from(row[best]) - f64::from(row[tok]);
            let c = &mut checks[s];
            c.other += 1;
            c.first_at.get_or_insert(k);
            c.worst = c.worst.max(gap);
            for (q, &e) in EDGES.iter().enumerate() {
                t.over[q] += usize::from(gap > e);
            }
            t.over_margin += usize::from(gap > cli.tie_margin);
        };
        for (s, c) in b.iter().enumerate() {
            sp.prefill(rt, dr.as_mut(), s, &c.prompt, &mut scratch)?;
            check(s, 0, &scratch, &mut t);
        }
        for k in 1.. {
            let who: Vec<usize> = (0..s_count).filter(|&s| k < len[s]).collect();
            if who.is_empty() {
                break;
            }
            let segs: Vec<SegIn<'_>> = who
                .iter()
                .map(|&s| SegIn {
                    slot: s,
                    ids: &spec[s][k - 1..k],
                    logits: 1,
                })
                .collect();
            let nl = rt.forward(&segs)?;
            rt.logits_host(nl, &mut scratch)?;
            for (q, &s) in who.iter().enumerate() {
                check(s, k, &scratch[q * LOGIT_ROW..(q + 1) * LOGIT_ROW], &mut t);
            }
            rt.commit(&vec![1; who.len()])?;
        }
        for (s, c) in b.iter().enumerate() {
            let ck = &checks[s];
            t.tokens += len[s];
            t.other += ck.other;
            t.worst = t.worst.max(ck.worst);
            t.rounds += rounds[s];
            t.emitted += len[s] - 1;
            let first_at = ck.first_at.map_or(-1, |k| i64::try_from(k).expect("small"));
            let per_round = (len[s] - 1) as f64 / rounds[s].max(1) as f64;
            let toks = spec[s].iter().map(i32::to_string).collect::<Vec<_>>().join(",");
            writeln!(
                out,
                "{{{identity},\"test\":\"spec_path\",\"id\":\"{}\",\"sessions_together\":{s_count},\"tokens\":{},\
                 \"not_plain_argmax\":{},\"first_at\":{first_at},\"largest_gap\":{:.4},\"rounds\":{},\
                 \"tokens_per_round\":{per_round:.4},\"spec_tokens\":[{toks}]}}",
                c.id, len[s], ck.other, ck.worst, rounds[s]
            )?;
            out.flush()?;
            eprintln!(
                "path {}: {} tokens, {} not the plain argmax (first at {first_at}), largest gap {:.4}, {per_round:.2} \
                 tokens/round",
                c.id, len[s], ck.other, ck.worst
            );
        }
    }
    let rate = 1e4 * t.over_margin as f64 / t.tokens.max(1) as f64;
    let per_round = t.emitted as f64 / t.rounds.max(1) as f64;
    writeln!(
        out,
        "{{{identity},\"test\":\"spec_path_summary\",\"tokens\":{},\"not_plain_argmax\":{},\"tie_margin\":{:.3},\
         \"over_margin\":{},\"gap_rate_per_10k\":{rate:.3},\"limit\":15,\"pass\":{},\"over_0.1\":{},\"over_0.25\":{},\
         \"over_0.5\":{},\"over_1\":{},\"over_2\":{},\"largest_gap\":{:.4},\"tokens_per_round\":{per_round:.4}}}",
        t.tokens,
        t.other,
        cli.tie_margin,
        t.over_margin,
        rate <= 15.0,
        t.over[0],
        t.over[1],
        t.over[2],
        t.over[3],
        t.over[4],
        t.worst
    )?;
    out.flush()?;
    if let Some(trace) = &mut sp.trace {
        trace.flush()?;
    }
    eprintln!(
        "[spec] {} tokens, {} not the plain argmax, {} by more than {:.2}: gap rate {rate:.2} per 10,000 (limit 15), \
         tokens/round {per_round:.3}",
        t.tokens, t.other, t.over_margin, cli.tie_margin
    );
    Ok(json!({"tokens": t.tokens, "not_plain_argmax": t.other, "rounds": t.rounds}))
}

/// The token stream of time mode: the prompt and tokens of case `--stream`.
fn stream_tokens(cases: &[Case], stream: &str) -> Vec<i32> {
    let mut tok = Vec::new();
    for c in cases.iter().filter(|c| c.id == stream) {
        tok.extend_from_slice(&c.prompt);
        tok.extend_from_slice(&c.tokens);
    }
    tok
}

fn percentile(sorted: &[f64], num: usize, den: usize) -> f64 {
    sorted.get(sorted.len() * num / den).copied().unwrap_or(0.0)
}

fn secs(a: Instant, b: Instant) -> f64 {
    b.duration_since(a).as_secs_f64()
}

/// Free-running speculative rounds over real contexts (`spec.cpp --mode time`): per round ms, split draft / verify,
/// tokens per round, tokens/s and NVML telemetry. Every 10th round is split with two device waits and left out of the
/// step statistics (the C++ tool's cudaEvent split costs nothing; the seam has no events).
fn time_mode(
    cli: &Cli,
    rt: &mut dyn Runtime,
    dr: &mut Box<dyn Drafter>,
    sp: &mut Spec,
    cases: &[Case],
    source: &Source,
    record: &mut RecordWriter,
) -> Result<Done> {
    let n = cli.sessions as usize;
    let ctx = cli.ctx as usize;
    let tok = stream_tokens(cases, &cli.stream);
    let off = if n > 1 {
        20_000_i64.min((i64::try_from(tok.len())? - i64::from(cli.ctx) - 4000) / (i64::from(cli.sessions) - 1))
    } else {
        0
    };
    ensure!(
        i64::try_from(tok.len())? >= off * (i64::from(cli.sessions) - 1) + i64::from(cli.ctx),
        "token stream too short"
    );
    let off = usize::try_from(off.max(0))?;
    let card = Card::read();
    let mut scratch = Vec::new();
    let t0p = Instant::now();
    for s in 0..n {
        sp.prefill(rt, dr.as_mut(), s, &tok[off * s..off * s + ctx], &mut scratch)?;
    }
    rt.sync()?;
    let prefill_s = secs(t0p, Instant::now());
    let act: Vec<usize> = (0..n).collect();
    let mut tele = Tele::start();
    let t0 = Instant::now();
    let (mut tw0, mut tw1) = (None, t0);
    let (mut sd, mut sv) = (0.0, 0.0);
    let (mut rounds, mut emitted, mut nt) = (0_u64, 0_u64, 0_u64);
    let mut rs: Vec<f64> = Vec::new();
    for it in 0_u64.. {
        let a = Instant::now();
        if secs(t0, a) > cli.secs {
            break;
        }
        let timed = it % 10 == 0;
        let mut split = (0.0, 0.0);
        let out = sp.round(rt, dr.as_mut(), &act, timed.then_some(&mut split))?;
        rt.sync()?;
        let b = Instant::now();
        if secs(t0, a) > cli.discard {
            tw0.get_or_insert(a);
            tw1 = b;
            rounds += 1;
            emitted += out.iter().map(|o| o.len() as u64).sum::<u64>();
            if timed {
                sd += split.0;
                sv += split.1;
                nt += 1;
            } else {
                rs.push(secs(a, b));
            }
        }
    }
    tele.stop();
    rs.sort_by(f64::total_cmp);
    let mean = rs.iter().sum::<f64>() / rs.len().max(1) as f64;
    let w0 = tw0.unwrap_or(t0);
    let window = if tw0.is_some() { secs(w0, tw1) } else { 0.0 };
    let stats = rt.stats()?;
    let line = Line::new("native_rt", "spec_step")
        .identity(source)
        .text("uuid", &card.uuid)
        .text("vbios", &card.vbios)
        .num("power_limit_w", card.power_limit_w)
        .uint("sessions", u64::from(cli.sessions))
        .uint("ctx", u64::from(cli.ctx))
        .uint("glue", u64::from(cli.engine.glue))
        .num("prefill_s", prefill_s)
        .uint("rounds", rounds)
        .num("window_s", window)
        .num("step_ms_mean", mean * 1e3)
        .num("step_ms_median", percentile(&rs, 1, 2) * 1e3)
        .num("step_ms_p10", percentile(&rs, 1, 10) * 1e3)
        .num("step_ms_p90", percentile(&rs, 9, 10) * 1e3)
        .num("draft_ms", sd / nt.max(1) as f64)
        .num("verify_ms", sv / nt.max(1) as f64)
        .num(
            "tokens_per_round_per_session",
            emitted as f64 / rounds.max(1) as f64 / n as f64,
        )
        .num("tokens_per_s", emitted as f64 / window.max(1e-9))
        .uint("graph_captures", stats.graph_captures);
    let value = tele.window(line, w0, tw1).emit()?;
    record.append(&json!({"line": value}))?;
    Ok(json!({"rounds": rounds}))
}

/// Timed speculative greedy decoding of the golden greedy prompts (`spec.cpp --mode golden`, `--tau off`).
fn golden_mode(
    cli: &Cli,
    rt: &mut dyn Runtime,
    dr: &mut Box<dyn Drafter>,
    sp: &mut Spec,
    cases: &[Case],
    source: &Source,
    record: &mut RecordWriter,
) -> Result<Done> {
    let n = cli.sessions as usize;
    let mut gr: Vec<&Case> = cases.iter().filter(|c| c.group == "greedy").collect();
    gr.truncate(gr.len() / n * n);
    let len = if cli.path > 0 { cli.path } else { 600 };
    let card = Card::read();
    let mut tele = Tele::start();
    let (mut secs_t, mut joules) = (0.0, 0.0);
    let (mut tokens, mut rounds, mut rows, mut sess_rounds) = (0_u64, 0_u64, 0_u64, 0_u64);
    let mut scratch = Vec::new();
    for pass in 0..cli.passes {
        let rep = pass + 1 == cli.passes;
        for batch in gr.chunks(n) {
            let mut spec: Vec<Vec<i32>> = Vec::with_capacity(n);
            for (s, c) in batch.iter().enumerate() {
                spec.push(vec![sp.prefill(rt, dr.as_mut(), s, &c.prompt, &mut scratch)?]);
            }
            rt.sync()?;
            let t0 = Instant::now();
            loop {
                let act: Vec<usize> = (0..n).filter(|&s| spec[s].len() < len).collect();
                if act.is_empty() {
                    break;
                }
                let out = sp.round(rt, dr.as_mut(), &act, None)?;
                if rep {
                    rounds += 1;
                    rows += 8 * act.len() as u64;
                    sess_rounds += act.len() as u64;
                }
                for (&s, toks) in act.iter().zip(out) {
                    for tok in toks {
                        if spec[s].len() < len {
                            spec[s].push(tok);
                        }
                    }
                }
            }
            rt.sync()?;
            let t1 = Instant::now();
            if rep {
                secs_t += secs(t0, t1);
                joules += tele.mean_energy_w(t0, t1) * secs(t0, t1);
                tokens += (n * (len - 1)) as u64;
            }
        }
    }
    tele.stop();
    let stats = rt.stats()?;
    let line = Line::new("native_rt", "spec_golden")
        .text("engine", ink_tools::engine::engine_name(source))
        .text("engine_version", &source.version)
        .text("commit", &source.commit)
        .text("uuid", &card.uuid)
        .text("vbios", &card.vbios)
        .num("power_limit_w", card.power_limit_w)
        .uint("sessions", u64::from(cli.sessions))
        .uint("prompts", gr.len() as u64)
        .uint("tokens_per_prompt", len as u64)
        .text("conf", if cli.gate == "on" { "p" } else { "none" })
        .text("tau", &tau_text(cli))
        .text("call_rows", &cli.call_rows)
        .num("seconds", secs_t)
        .num("energy_j", joules)
        .uint("tokens", tokens)
        .uint("rounds", rounds)
        .num("tokens_per_s", tokens as f64 / secs_t.max(1e-9))
        .num("tokens_per_j", tokens as f64 / joules.max(1e-9))
        .num("round_ms", secs_t / rounds.max(1) as f64 * 1e3)
        .num("j_per_round", joules / rounds.max(1) as f64)
        .num("tokens_per_session_round", tokens as f64 / sess_rounds.max(1) as f64)
        .num("rows_per_session_round", rows as f64 / sess_rounds.max(1) as f64)
        .uint("w1", 0_u64)
        .uint("w2", 0_u64)
        .uint("w4", 0_u64)
        .uint("w8", sess_rounds)
        .uint("graph_captures", stats.graph_captures);
    let value = line.emit()?;
    record.append(&json!({"line": value}))?;
    Ok(json!({"rounds": rounds}))
}

/// Speculative sampled vs plain sampled on the golden greedy prompts (`spec.cpp --mode sampled`).
fn sampled_mode(
    cli: &Cli,
    rt: &mut dyn Runtime,
    dr: &mut Box<dyn Drafter>,
    sp: &mut Spec,
    gr: &[&Case],
    source: &Source,
    record: &mut RecordWriter,
) -> Result<Done> {
    probe_of(rt)?;
    let n = cli.sessions as usize;
    let len = if cli.path > 0 { cli.path } else { 400 };
    let card = Card::read();
    let (mut tot, mut tot_eq, mut cmp, mut cmp_eq) = (0_u64, 0_u64, 0_u64, 0_u64);
    let (mut rounds_tot, mut emitted_tot, mut cases_eq, mut flags_or) = (0_u64, 0_u64, 0_u64, 0_u64);
    let mut acc_hist = vec![0_u64; sp.draft_k + 1];
    let leviathan = dr.proposal().is_some();
    for (batch_idx, b) in gr.chunks(n).enumerate() {
        let i = batch_idx * n;
        let s_count = b.len();
        let mut samp_slot = vec![ctl(cli, 0); n];
        for (s, slot_p) in samp_slot.iter_mut().enumerate().take(s_count) {
            *slot_p = ctl(cli, cli.seed + (i + s) as u64);
        }
        // speculative pass
        let mut spec: Vec<Vec<i32>> = Vec::with_capacity(s_count);
        let mut dig_spec: Vec<BTreeMap<i32, String>> = (0..s_count).map(|_| BTreeMap::new()).collect();
        let mut rounds = vec![0_u64; s_count];
        for (s, c) in b.iter().enumerate() {
            let tok = sp.prefill_with(rt, dr.as_mut(), s, &c.prompt, None, Some(samp_slot[s]))?;
            spec.push(vec![tok]);
        }
        let mut st_round: Vec<u64> = Vec::new();
        let mut called_order: Vec<(usize, usize)> = Vec::new();
        loop {
            let act: Vec<usize> = (0..s_count).filter(|&s| spec[s].len() < len).collect();
            if act.is_empty() {
                break;
            }
            let mut pos0 = vec![0_usize; s_count];
            for &s in &act {
                pos0[s] = rt.length(s)?;
            }
            called_order.clear();
            let mut hook = |rt: &mut dyn Runtime, segs: &[(usize, usize)]| -> Result<()> {
                called_order.extend_from_slice(segs);
                if leviathan {
                    return Ok(());
                }
                let total_rows: usize = segs.iter().map(|&(_, w)| w).sum();
                let mut l = Vec::with_capacity(total_rows);
                let mut p = Vec::with_capacity(total_rows);
                let mut q = Vec::with_capacity(total_rows);
                let mut par = Vec::with_capacity(segs.len());
                for (g, &(slot, w)) in segs.iter().enumerate() {
                    par.push(samp_slot[slot]);
                    let base = rt.length(slot)?;
                    for j in 0..w {
                        l.push(id(l.len()));
                        p.push(id(g));
                        q.push(id(base + j + 1));
                    }
                }
                let rows = SampleTestRows {
                    logits: None,
                    lrow: &l,
                    par_idx: &p,
                    pos: &q,
                    params: &par,
                };
                let mut out = SampleTestOut {
                    stats: Some(&mut st_round),
                    ..Default::default()
                };
                probe_of(rt)?.sample_test_rows(&rows, &mut out)?;
                Ok(())
            };
            let rs = sp.res_stride;
            let (out, res) = sp.round_with(rt, dr.as_mut(), &act, None, Some(&samp_slot), false, Some(&mut hook))?;
            let mut row_off = 0usize;
            for (g, &(s, w)) in called_order.iter().enumerate() {
                let a = usize::try_from(res[g * rs]).context("accepted count")?;
                acc_hist[a] += 1;
                if !leviathan {
                    for j in 0..=a {
                        let row = &st_round[(row_off + j) * 8..];
                        dig_spec[s].insert(id(pos0[s] + j + 1), digest(row));
                        flags_or |= row[7];
                    }
                }
                row_off += w;
            }
            for (&s, toks) in act.iter().zip(out) {
                rounds[s] += 1;
                for tok in toks {
                    if spec[s].len() < len {
                        spec[s].push(tok);
                    }
                }
            }
        }
        if leviathan {
            for (s, c) in b.iter().enumerate() {
                tot += len as u64;
                tot_eq += len as u64;
                cases_eq += 1;
                rounds_tot += rounds[s];
                emitted_tot += (len - 1) as u64;
                let toks = spec[s].iter().map(i32::to_string).collect::<Vec<_>>().join(",");
                let line = emit_id(Line::new("native_rt", "spec_sampled"), source, &card, cli)
                    .text("id", &c.id)
                    .uint("seed", samp_slot[s].seed)
                    .uint("sessions_together", s_count as u64)
                    .uint("tokens", len as u64)
                    .flag("leviathan", true)
                    .uint("rounds", rounds[s])
                    .num("tokens_per_round", (len - 1) as f64 / rounds[s].max(1) as f64)
                    .text("spec_tokens", &toks);
                let value = line.emit()?;
                record.append(&json!({"line": value}))?;
            }
            continue;
        }
        // plain pass: one row per session per call, the same sampler and controls
        let mut plain: Vec<Vec<i32>> = Vec::with_capacity(s_count);
        let mut dig_plain: Vec<BTreeMap<i32, String>> = (0..s_count).map(|_| BTreeMap::new()).collect();
        for (s, c) in b.iter().enumerate() {
            let tok = sp.prefill_with(rt, dr.as_mut(), s, &c.prompt, None, Some(samp_slot[s]))?;
            plain.push(vec![tok]);
        }
        let mut tok_h = Vec::new();
        let mut st_plain = Vec::new();
        loop {
            let who: Vec<usize> = (0..s_count).filter(|&s| plain[s].len() < len).collect();
            if who.is_empty() {
                break;
            }
            let mut l = Vec::with_capacity(who.len());
            let mut p = Vec::with_capacity(who.len());
            let mut q = Vec::with_capacity(who.len());
            let mut par = Vec::with_capacity(who.len());
            let mut segs = Vec::with_capacity(who.len());
            for (k, &s) in who.iter().enumerate() {
                l.push(id(k));
                p.push(id(k));
                q.push(id(rt.length(s)? + 1));
                par.push(samp_slot[s]);
                let last = plain[s].len() - 1;
                segs.push(SegIn {
                    slot: s,
                    ids: &plain[s][last..=last],
                    logits: 1,
                });
            }
            rt.forward(&segs)?;
            let nseg = segs.len();
            let rows = SampleTestRows {
                logits: None,
                lrow: &l,
                par_idx: &p,
                pos: &q,
                params: &par,
            };
            let mut out = SampleTestOut {
                stats: Some(&mut st_plain),
                ..Default::default()
            };
            probe_of(rt)?.sample_test_rows(&rows, &mut out)?;
            rt.sample_host(&par, who.len(), &mut tok_h)?;
            for (k, &s) in who.iter().enumerate() {
                ensure!(
                    u64::try_from(tok_h[k]).unwrap_or(u64::MAX) == st_plain[k * 8 + 6],
                    "engine sampler and test mapping disagree"
                );
                dig_plain[s].insert(q[k], digest(&st_plain[k * 8..]));
                plain[s].push(tok_h[k]);
            }
            rt.commit(&vec![1; nseg])?;
        }
        for (s, c) in b.iter().enumerate() {
            let first = (0..len).find(|&t| spec[s][t] != plain[s][t]);
            let (mut c_cnt, mut ce_cnt) = (0_u64, 0_u64);
            for (pos, d) in &dig_spec[s] {
                if let Some(pd) = dig_plain[s].get(pos) {
                    c_cnt += 1;
                    ce_cnt += u64::from(pd == d);
                }
            }
            tot += len as u64;
            tot_eq += first.unwrap_or(len) as u64;
            cases_eq += u64::from(first.is_none());
            cmp += c_cnt;
            cmp_eq += ce_cnt;
            rounds_tot += rounds[s];
            emitted_tot += (len - 1) as u64;
            let toks = spec[s].iter().map(i32::to_string).collect::<Vec<_>>().join(",");
            let line = emit_id(Line::new("native_rt", "spec_sampled"), source, &card, cli)
                .text("id", &c.id)
                .uint("seed", samp_slot[s].seed)
                .uint("sessions_together", s_count as u64)
                .uint("tokens", len as u64)
                .flag("equal", first.is_none())
                .int("first_diff", first.map_or(-1_i64, |t| i64::try_from(t).expect("small")))
                .uint("positions_compared", c_cnt)
                .uint("digests_equal", ce_cnt)
                .uint("rounds", rounds[s])
                .num("tokens_per_round", (len - 1) as f64 / rounds[s].max(1) as f64)
                .text("spec_tokens", &toks);
            let value = line.emit()?;
            record.append(&json!({"line": value}))?;
        }
    }
    let ah = acc_hist.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
    let pass = cases_eq == gr.len() as u64 && cmp == cmp_eq && (flags_or & K_SAMP_FLAG_MISMATCH) == 0;
    let line = emit_id(Line::new("native_rt", "spec_sampled_summary"), source, &card, cli)
        .uint("cases", gr.len() as u64)
        .uint("cases_equal", cases_eq)
        .uint("tokens", tot)
        .uint("tokens_equal_prefix", tot_eq)
        .uint("positions_compared", cmp)
        .uint("digests_equal", cmp_eq)
        .uint("flags_or", flags_or)
        .num("tokens_per_round", emitted_tot as f64 / rounds_tot.max(1) as f64)
        .text("accepted_hist", &ah)
        .flag("pass", pass);
    let value = line.emit()?;
    record.append(&json!({"line": value}))?;
    Ok(
        json!({"cases": gr.len(), "cases_equal": cases_eq, "digests_equal": cmp_eq, "positions_compared": cmp, "pass": pass}),
    )
}

/// Distribution test (`spec.cpp --mode sdist`): N slots hold the same context, rounds commit nothing.
#[expect(clippy::too_many_lines, reason = "direct port of spec.cpp sdist")]
fn sdist_mode(
    cli: &Cli,
    rt: &mut dyn Runtime,
    dr: &mut Box<dyn Drafter>,
    sp: &mut Spec,
    gr: &[&Case],
    source: &Source,
    record: &mut RecordWriter,
) -> Result<Done> {
    probe_of(rt)?;
    let n = cli.sessions as usize;
    let card = Card::read();
    let plg = spec_of(rt)?.alloc(8 * LOGIT_ROW * size_of::<f32>())?;
    let anchor0 = i32s(rt, n)?;
    let p0 = i32s(rt, n)?;
    let lf0 = spec_of(rt)?.alloc(n * FEAT_ROW_BYTES)?;
    let ks: Vec<i32> = cli
        .ctx_tokens
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i32>().with_context(|| format!("bad --ctx-tokens {s:?}")))
        .collect::<Result<_>>()?;
    let first0 = ctl(cli, cli.seed);
    let mut ctxs: Vec<(&Case, usize)> = Vec::new();
    let mut st_sel = Vec::new();
    for &c0 in gr {
        for &k_cfg in &ks {
            if k_cfg >= 0 {
                ctxs.push((c0, usize::try_from(k_cfg)?));
                continue;
            }
            let want = usize::try_from(-k_cfg)?;
            let mut found = 0;
            let mut k = 8_usize;
            while k + 9 < c0.tokens.len() && found < want {
                let mut ids = Vec::with_capacity(c0.prompt.len() + k + 1);
                ids.extend_from_slice(&c0.prompt);
                ids.extend_from_slice(&c0.tokens[..=k]);
                sp.prefill_with(rt, dr.as_mut(), 0, &ids, None, Some(first0))?;
                let rows = SampleTestRows {
                    logits: None,
                    lrow: &[0],
                    par_idx: &[0],
                    pos: &[id(ids.len())],
                    params: &[first0],
                };
                let mut out = SampleTestOut {
                    stats: Some(&mut st_sel),
                    ..Default::default()
                };
                probe_of(rt)?.sample_test_rows(&rows, &mut out)?;
                let pmax = 1_099_511_627_776.0 / st_sel[2].max(1) as f64;
                if pmax < 0.9 && pmax > 0.2 {
                    ctxs.push((c0, k));
                    found += 1;
                    k += 60;
                }
                k += 4;
            }
        }
    }
    let act: Vec<usize> = (0..n).collect();
    let mut ctx_done = 0_usize;
    for (c0, k_off) in ctxs {
        let cid = format!("{}@{k_off}", c0.id);
        let mut prompt = c0.prompt.clone();
        prompt.extend_from_slice(&c0.tokens[..k_off.min(c0.tokens.len())]);
        let first = ctl(cli, cli.seed);
        let mut samp_slot = vec![first; n];
        for s in 0..n {
            sp.prefill_with(rt, dr.as_mut(), s, &prompt, None, Some(first))?;
        }
        let p_len = prompt.len();
        let anc_tok = c0.tokens[k_off.min(c0.tokens.len() - 1)];
        spec_of(rt)?.upload(sp.bufs.anchor, &vec![anc_tok; n])?;
        probe_of(rt)?.copy_dev(anchor0, sp.bufs.anchor, n * 4)?;
        probe_of(rt)?.copy_dev(p0, sp.bufs.pos, n * 4)?;
        probe_of(rt)?.copy_dev(lf0, sp.bufs.lastfeat, n * FEAT_ROW_BYTES)?;
        let anc = device_i32s(rt, sp.bufs.anchor, n)?;
        let mut vst: Vec<u64> = Vec::new();
        let mut keys: Vec<u32> = Vec::new();
        let mut dumped = false;
        let leviathan = dr.proposal().is_some();
        let mut ev_spec: BTreeMap<(usize, i32), u64> = BTreeMap::new();
        let mut first_tok_counts: BTreeMap<i32, u64> = BTreeMap::new();
        let mut lev_emitted_tot: u64 = 0;
        let mut drafts = vec![0_i32; DRAFTS];
        for it in 0..cli.iters {
            probe_of(rt)?.copy_dev(sp.bufs.anchor, anchor0, n * 4)?;
            probe_of(rt)?.copy_dev(sp.bufs.pos, p0, n * 4)?;
            probe_of(rt)?.copy_dev(sp.bufs.lastfeat, lf0, n * FEAT_ROW_BYTES)?;
            for (s, slot_p) in samp_slot.iter_mut().enumerate() {
                *slot_p = ctl(cli, cli.seed + 1 + (it * n + s) as u64);
            }
            let mut hook = |rt: &mut dyn Runtime, _: &[(usize, usize)]| -> Result<()> {
                if dumped {
                    return Ok(());
                }
                dumped = true;
                let l: Vec<i32> = (0..8).map(id).collect();
                let p = vec![0_i32; 8];
                let q: Vec<i32> = (0..8).map(|j| id(p_len + j + 1)).collect();
                let rows = SampleTestRows {
                    logits: None,
                    lrow: &l,
                    par_idx: &p,
                    pos: &q,
                    params: &[first],
                };
                let mut out = SampleTestOut {
                    stats: Some(&mut vst),
                    keys: Some(&mut keys),
                    ..Default::default()
                };
                probe_of(rt)?.sample_test_rows(&rows, &mut out)?;
                Ok(())
            };
            let (out, _) = sp.round_with(rt, dr.as_mut(), &act, None, Some(&samp_slot), true, Some(&mut hook))?;
            if it == 0 && !leviathan {
                let d1 = device_i32s(rt, dr.drafts(), DRAFTS * n)?;
                drafts.copy_from_slice(&d1[..DRAFTS]);
                for s in 1..n {
                    for j in 0..DRAFTS {
                        ensure!(
                            d1[s * DRAFTS + j] == drafts[j],
                            "slots with one context drafted differently"
                        );
                    }
                }
            }
            for s_out in &out {
                let a = s_out.len() - 1;
                let tok = *s_out.last().context("empty emitted tokens")?;
                *ev_spec.entry((a, tok)).or_default() += 1;
                *first_tok_counts.entry(s_out[0]).or_default() += 1;
                lev_emitted_tot += s_out.len() as u64;
            }
        }
        if leviathan {
            // Exact analytical test of row-0 marginal distribution against exact target weights p(v) = w(v) / W.
            let tau0 = vst[1] as u32;
            let w_tot = vst[2] as f64;
            let n_draws = (cli.iters * n) as f64;
            let mut zero_prob = 0_u64;
            let mut tv = 0.0_f64;
            let mut chi2 = 0.0_f64;
            let mut df = 0_i64;
            let mut tail_exp = 0.0_f64;
            let mut tail_obs = 0.0_f64;
            let mut kept_probs: Vec<(i32, f64)> = Vec::new();
            for (v, &k) in keys[..VOCAB].iter().enumerate() {
                if k != 0 && k >= tau0 {
                    let w = ((f32::from_bits(k) * 1_099_511_627_776.0_f32) as u64) as f64;
                    kept_probs.push((id(v), w / w_tot));
                }
            }
            let kept_set: BTreeMap<i32, f64> = kept_probs.iter().copied().collect();
            for (&tok, &cnt) in &first_tok_counts {
                if !kept_set.contains_key(&tok) {
                    zero_prob += cnt;
                }
            }
            for &(tok, prob) in &kept_probs {
                let obs = *first_tok_counts.get(&tok).unwrap_or(&0) as f64;
                let exp = prob * n_draws;
                tv += 0.5 * ((obs / n_draws) - prob).abs();
                if exp >= 5.0 {
                    chi2 += (obs - exp) * (obs - exp) / exp;
                    df += 1;
                } else {
                    tail_exp += exp;
                    tail_obs += obs;
                }
            }
            if tail_exp >= 5.0 {
                chi2 += (tail_obs - tail_exp) * (tail_obs - tail_exp) / tail_exp;
                df += 1;
            }
            let df = (df - 1).max(1);
            let z = (chi2 - df as f64) / (2.0 * df as f64).sqrt();
            let line = emit_id(Line::new("native_rt", "spec_sdist_leviathan"), source, &card, cli)
                .text("id", &cid)
                .uint("draws", (cli.iters * n) as u64)
                .uint("kept_vocab", kept_probs.len() as u64)
                .uint("zero_prob_draws", zero_prob)
                .num("tv", tv)
                .num("chi2", chi2)
                .int("df", df)
                .num("chi2_z", z)
                .num("tokens_per_round", lev_emitted_tot as f64 / n_draws)
                .flag("pass", zero_prob == 0 && z.abs() < 4.0);
            let value = line.emit()?;
            record.append(&json!({"line": value}))?;
            ctx_done += 1;
            continue;
        }
        // plain rows: one-row decode along anchor, d_0..d_6 on slot 0
        sp.prefill_with(rt, dr.as_mut(), 0, &prompt, None, Some(first))?;
        let mut path = Vec::with_capacity(8);
        path.push(anc[0]);
        path.extend_from_slice(&drafts);
        for (j, &tok) in path.iter().enumerate() {
            rt.forward(&[SegIn {
                slot: 0,
                ids: &[tok],
                logits: 1,
            }])?;
            probe_of(rt)?.copy_logits(plg, j, 1)?;
            rt.commit(&[1])?;
        }
        let mut pst: Vec<u64> = Vec::new();
        {
            let l: Vec<i32> = (0..8).map(id).collect();
            let p = vec![0_i32; 8];
            let q: Vec<i32> = (0..8).map(|j| id(p_len + j + 1)).collect();
            let rows = SampleTestRows {
                logits: Some(plg),
                lrow: &l,
                par_idx: &p,
                pos: &q,
                params: &[first],
            };
            let mut out = SampleTestOut {
                stats: Some(&mut pst),
                ..Default::default()
            };
            probe_of(rt)?.sample_test_rows(&rows, &mut out)?;
        }
        let dig_eq: u64 = (0..8)
            .map(|j| u64::from(digest(&vst[j * 8..]) == digest(&pst[j * 8..])))
            .sum();
        // plain samples (seeds disjoint from the speculative ones)
        let mut ev_plain: BTreeMap<(usize, i32), u64> = BTreeMap::new();
        let chunk = (cli.test_rows / 8).max(1);
        let mut y = Vec::new();
        let mut b0 = 0_usize;
        while b0 < cli.plain {
            let m = chunk.min(cli.plain - b0);
            let mut l = Vec::with_capacity(m * 8);
            let mut p = Vec::with_capacity(m * 8);
            let mut q = Vec::with_capacity(m * 8);
            let mut par = Vec::with_capacity(m);
            for i in 0..m {
                par.push(ctl(cli, cli.seed + (1_u64 << 40) + (b0 + i) as u64));
                for j in 0..8 {
                    l.push(id(j));
                    p.push(id(i));
                    q.push(id(p_len + j + 1));
                }
            }
            let rows = SampleTestRows {
                logits: Some(plg),
                lrow: &l,
                par_idx: &p,
                pos: &q,
                params: &par,
            };
            let mut out = SampleTestOut {
                out: Some(&mut y),
                ..Default::default()
            };
            probe_of(rt)?.sample_test_rows(&rows, &mut out)?;
            for i in 0..m {
                let mut a = 0_usize;
                while a < DRAFTS && y[i * 8 + a] == drafts[a] {
                    a += 1;
                }
                *ev_plain.entry((a, y[i * 8 + a])).or_default() += 1;
            }
            b0 += m;
        }
        // verify-row keys for the exact reference
        let kp = cli.out.join(format!("{cid}.keys.bin"));
        {
            let f = File::create(&kp).with_context(|| format!("create {}", kp.display()))?;
            let mut w = BufWriter::new(f);
            for &k_u32 in &keys {
                w.write_all(&k_u32.to_le_bytes())?;
            }
            w.flush()?;
        }
        let evs = |m: &BTreeMap<(usize, i32), u64>| {
            m.iter()
                .map(|((a, tok), v)| format!("{a}:{tok}:{v}"))
                .collect::<Vec<_>>()
                .join(";")
        };
        let ds = drafts.iter().map(i32::to_string).collect::<Vec<_>>().join(",");
        let vs = (0..8).map(|j| digest(&vst[j * 8..])).collect::<Vec<_>>().join(",");
        let ps = (0..8).map(|j| digest(&pst[j * 8..])).collect::<Vec<_>>().join(",");
        let line = emit_id(Line::new("native_rt", "spec_sdist"), source, &card, cli)
            .text("id", &cid)
            .uint("context", p_len as u64)
            .int("anchor", anc[0])
            .text("drafts", &ds)
            .uint("slots", n as u64)
            .uint("iters", cli.iters as u64)
            .uint("spec_samples", (cli.iters * n) as u64)
            .uint("plain_samples", cli.plain as u64)
            .text("verify_digests", &vs)
            .text("plain_digests", &ps)
            .uint("digests_equal", dig_eq)
            .text("keys_file", &kp.display().to_string())
            .text("events_spec", &evs(&ev_spec))
            .text("events_plain", &evs(&ev_plain));
        let value = line.emit()?;
        record.append(&json!({"line": value}))?;
        ctx_done += 1;
    }
    Ok(json!({"contexts": ctx_done}))
}

/// Device sampler vs device argmax timing over verify rows (`spec.cpp --mode stime`).
fn stime_mode(
    cli: &Cli,
    rt: &mut dyn Runtime,
    dr: &mut Box<dyn Drafter>,
    sp: &mut Spec,
    cases: &[Case],
    source: &Source,
    record: &mut RecordWriter,
) -> Result<Done> {
    probe_of(rt)?;
    let n = cli.sessions as usize;
    let ctx = cli.ctx as usize;
    let tok = stream_tokens(cases, &cli.stream);
    let off = if n > 1 {
        20_000_i64.min((i64::try_from(tok.len())? - i64::from(cli.ctx) - 4000) / (i64::from(cli.sessions) - 1))
    } else {
        0
    };
    ensure!(
        i64::try_from(tok.len())? >= off * (i64::from(cli.sessions) - 1) + i64::from(cli.ctx),
        "token stream too short"
    );
    let off = usize::try_from(off.max(0))?;
    let card = Card::read();
    for s in 0..n {
        sp.prefill_with(rt, dr.as_mut(), s, &tok[off * s..off * s + ctx], None, None)?;
    }
    let act: Vec<usize> = (0..n).collect();
    let samp_slot: Vec<DeviceSampling> = (0..n).map(|s| ctl(cli, cli.seed + s as u64)).collect();
    let (mut ms_arg, mut ms_smp, mut j_arg, mut j_smp) = (0.0, 0.0, 0.0, 0.0);
    let mut st: Vec<u64> = Vec::new();
    let mut done = false;
    let mut hook = |rt: &mut dyn Runtime, segs: &[(usize, usize)]| -> Result<()> {
        if done {
            return Ok(());
        }
        done = true;
        let par: Vec<DeviceSampling> = segs.iter().map(|&(s, _)| samp_slot[s]).collect();
        for which in [false, true] {
            let mut tele = Tele::start();
            let (ms, t0, t1) = probe_of(rt)?.time_select(which.then_some(&par), 20, cli.reps)?;
            tele.stop();
            let j = tele.mean_energy_w(t0, t1) * secs(t0, t1) / cli.reps as f64;
            if which {
                ms_smp = ms;
                j_smp = j;
            } else {
                ms_arg = ms;
                j_arg = j;
            }
        }
        let tot_rows: usize = segs.iter().map(|&(_, r)| r).sum();
        if tot_rows <= 64 {
            let mut l = Vec::with_capacity(tot_rows);
            let mut p = Vec::with_capacity(tot_rows);
            let mut q = Vec::with_capacity(tot_rows);
            for (g, &(slot, rows)) in segs.iter().enumerate() {
                let base = rt.length(slot)?;
                for j in 0..rows {
                    l.push(id(l.len()));
                    p.push(id(g));
                    q.push(id(base + j + 1));
                }
            }
            let rows = SampleTestRows {
                logits: None,
                lrow: &l,
                par_idx: &p,
                pos: &q,
                params: &par,
            };
            let mut out = SampleTestOut {
                stats: Some(&mut st),
                ..Default::default()
            };
            probe_of(rt)?.sample_test_rows(&rows, &mut out)?;
        }
        Ok(())
    };
    sp.round_with(rt, dr.as_mut(), &act, None, Some(&samp_slot), false, Some(&mut hook))?;
    let (mut flags_or, mut kept_sum, mut kept_max) = (0_u64, 0_u64, 0_u64);
    let nrows = st.len() / 8;
    for r in 0..nrows {
        flags_or |= st[r * 8 + 7];
        kept_sum += st[r * 8 + 3];
        kept_max = kept_max.max(st[r * 8 + 3]);
    }
    let line = emit_id(Line::new("native_rt", "spec_stime"), source, &card, cli)
        .uint("sessions", n as u64)
        .uint("ctx", ctx as u64)
        .uint("rows", (n * 8) as u64)
        .uint("reps", cli.reps as u64)
        .num("argmax_ms", ms_arg)
        .num("sample_ms", ms_smp)
        .num("sample_minus_argmax_ms", ms_smp - ms_arg)
        .num("argmax_j", j_arg)
        .num("sample_j", j_smp)
        .num("sample_minus_argmax_j", j_smp - j_arg)
        .uint("flags_or", flags_or)
        .num("kept_mean", kept_sum as f64 / nrows.max(1) as f64)
        .uint("kept_max", kept_max);
    let value = line.emit()?;
    record.append(&json!({"line": value}))?;
    Ok(json!({"sessions": n, "reps": cli.reps, "argmax_ms": ms_arg, "sample_ms": ms_smp}))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let n = cli.sessions as usize;
    let dctx = cli.draft_ctx.unwrap_or((cli.ctx + 8192).max(16_384));
    let pages = cli.pages.unwrap_or(cli.sessions * dctx.div_ceil(1024) + 8);
    if cli.engine.slots != cli.sessions {
        bail!("--slots must equal --sessions ({}), as spec.cpp sets it", cli.sessions);
    }
    if cli.conf_trace.is_some() {
        bail!("--conf-trace (survivors per round) is not ported");
    }
    let gate = match cli.gate.as_str() {
        "on" => cli.dflash2.is_none(),
        "off" => false,
        g => bail!("--gate on|off, got {g}"),
    };
    let tau = cli.tau.unwrap_or(VERIFY_TAU);
    ensure!(tau > 0.0 && tau <= 1.0, "--tau: a probability in (0, 1]");
    ensure!(gate || cli.tau.is_none(), "--tau needs --gate on");
    let call_rows = CallRows::parse(&cli.call_rows).map_err(|e| anyhow::anyhow!("{e}"))?;
    let keep: BTreeSet<&str> = cli.only.split(',').filter(|s| !s.is_empty()).collect();
    let cases = read_cases(&cli.cases)?;
    let greedy_cases: Vec<&Case> = cases
        .iter()
        .filter(|c| c.group == "greedy" && (keep.is_empty() || keep.contains(c.id.as_str())))
        .collect();
    let mut rt = cli.engine.open(pages, MAX_CHUNK)?;
    let source = rt.identity();
    let mode = match cli.mode {
        Mode::Greedy => "greedy",
        Mode::Sampled => "sampled",
        Mode::Sdist => "sdist",
        Mode::Stime => "stime",
        Mode::Time => "time",
        Mode::Golden => "golden",
    };
    let mut record = cli.record.writer_with(
        "spec",
        source.clone(),
        [(ComponentName::Drafter, Component::Source(source.clone()))],
        json!({"engine": cli.engine.json(pages, MAX_CHUNK), "mode": mode, "cases": cli.cases,
               "sessions": cli.sessions, "ctx": cli.ctx, "draft_ctx": dctx, "fold": cli.fold, "kv8": cli.kv8,
               "only": cli.only, "path": cli.path, "tie_margin": cli.tie_margin, "trace": cli.trace, "draft_log": cli.draft_log,
               "secs": cli.secs, "discard": cli.discard, "stream": cli.stream, "passes": cli.passes,
               "temperature": cli.temperature, "top_p": cli.top_p, "min_p": cli.min_p, "top_k": cli.top_k,
               "seed": cli.seed, "out": cli.out, "iters": cli.iters, "plain": cli.plain,
               "ctx_tokens": cli.ctx_tokens, "reps": cli.reps}),
        "C++ ink_spec of the same build (same argv, same lines)",
    )?;
    let rt: &mut dyn Runtime = &mut *rt;
    let mut dr = spec_of(rt)?.create_drafter(&DraftConfig {
        model_dir: cli.engine.model.clone(),
        fold_bin: cli.fold.clone(),
        dflash2_dir: cli.dflash2.clone(),
        sessions: cli.sessions,
        max_ctx: dctx,
        head: HeadPrecision::Int8,
        global_kv: if cli.kv8 == 1 {
            KvPrecision::E4m3
        } else {
            KvPrecision::Bf16
        },
        verify: if gate {
            VerifyWidth::Gated(call_rows)
        } else {
            VerifyWidth::Full
        },
        lookup: cli.lookup,
        lookup_min: cli.lookup_min,
    })?;
    let trace = cli
        .trace
        .as_ref()
        .map(|p| {
            File::create(p)
                .map(BufWriter::new)
                .with_context(|| p.display().to_string())
        })
        .transpose()?;
    let lookup_len = if cli.mode == Mode::Sdist {
        0
    } else {
        cli.lookup as usize
    };
    let mut sp = Spec::new(
        rt,
        dr.as_ref(),
        n,
        dctx as usize,
        lookup_len,
        cli.lookup_min as usize,
        trace,
    )?;
    sp.tau = gate.then(|| tau.ln());
    sp.call_rows = if gate { call_rows } else { CallRows::RAGGED };
    if let Some(p) = &cli.draft_log {
        let f = File::create(p).with_context(|| p.display().to_string())?;
        let h = Header {
            sessions: cli.sessions,
            max_ctx: dctx,
            int8: true,
            kv8: cli.kv8 == 1,
        };
        sp.log = Some(LogWriter::new(BufWriter::with_capacity(1 << 24, f), h)?);
    }
    let done = match cli.mode {
        Mode::Greedy => greedy(&cli, rt, &mut dr, &mut sp, &greedy_cases, &source)?,
        Mode::Sampled => sampled_mode(&cli, rt, &mut dr, &mut sp, &greedy_cases, &source, &mut record)?,
        Mode::Sdist => sdist_mode(&cli, rt, &mut dr, &mut sp, &greedy_cases, &source, &mut record)?,
        Mode::Stime => stime_mode(&cli, rt, &mut dr, &mut sp, &cases, &source, &mut record)?,
        Mode::Time => time_mode(&cli, rt, &mut dr, &mut sp, &cases, &source, &mut record)?,
        Mode::Golden => golden_mode(&cli, rt, &mut dr, &mut sp, &cases, &source, &mut record)?,
    };
    if let Some(log) = sp.log.take() {
        log.finish()?;
    }
    let stats = rt.stats()?;
    if sp.lookup_len > 0 {
        eprintln!(
            "[spec] lookup triggers={} accepted={} (rounds={})",
            sp.lookup_triggers, sp.lookup_accepted, sp.rounds_done
        );
    }
    let mut data = done;
    data["done"] = json!(true);
    data["lookup"] = json!({"len": sp.lookup_len, "min": sp.lookup_min, "triggers": sp.lookup_triggers, "accepted": sp.lookup_accepted});
    data["stats"] = json!({"graph_captures": stats.graph_captures, "gemm_fallbacks": stats.gemm_fallbacks});
    record.append(&data)?;
    eprintln!("record {}", record.path().display());
    Ok(())
}
