//! The Rust DSpark drafter fed from a drafter log (package R8, gate E8).
//!
//! ```text
//! ink-draft --log draft.log --model /model --fold /s2f/fold.bin --record-dir DIR [--taps-at 0,57]
//!           [--time-iters 200 --time-reps 5]
//! ```
//!
//! Replays every call of the log (`ink_tools::draftlog`, written by `ink-spec --draft-log` with the C++ drafter)
//! into [`RustDrafter`] in order: commit_rows, set_state, frnn as logged, and at each logged draft (all sessions, or
//! the logged sessions of a `draft_sessions` record) the Rust drafter's drafts of the drafted sessions are compared
//! with the logged ones. Writes into the record directory: `rounds.jsonl` (one line per draft:
//! index, equal, the Rust drafts), `trace.jsonl` (the draft graph's launch trace, `ink-launch-trace/1`) and
//! `arena-map.jsonl` (the drafter arena, to relocate the trace's pointers; both of the first draft's batch),
//! `taps-<k>.jsonl` (every weight and state buffer's FNV-1a 64 after draft k), `time.jsonl` (ms per draft graph
//! replay of the last draft's batch size after the last call), and the record
//! `draft.jsonl` with the totals. Exit status 1 when a draft differs.
#![deny(unsafe_code)] // allowed only at the calls of the runtime's bounds-contract methods
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use ink_record::{Component, ComponentName, Source};
use ink_rt_api::{
    DRAFTS, DevBuf, DraftConfig, Drafter, FEAT_ROW_BYTES, HeadPrecision, KvPrecision, VOCAB, VerifyWidth,
};
use ink_runtime::RustDrafter;
use ink_tools::draftlog::{LogReader, Op};
use ink_tools::engine::RecordArgs;
use serde_json::json;

#[derive(Parser)]
#[command(name = "ink-draft", version = ink_record::BUILD_LINE, about = "The Rust DSpark drafter fed from a drafter log")]
struct Cli {
    #[command(flatten)]
    record: RecordArgs,
    /// Drafter log (`ink-spec --draft-log`).
    #[arg(long)]
    log: PathBuf,
    /// Model directory.
    #[arg(long, default_value = "/model")]
    model: PathBuf,
    /// Folded drafter weights.
    #[arg(long, default_value = "/s2f/fold.bin")]
    fold: PathBuf,
    /// Draft indices (0-based, in log order) after which every buffer is hashed.
    #[arg(long, value_delimiter = ',')]
    taps_at: Vec<u64>,
    /// Graph replays per timing repeat after the last call (0: no timing).
    #[arg(long, default_value_t = 0)]
    time_iters: usize,
    /// Timing repeats.
    #[arg(long, default_value_t = 5)]
    time_reps: usize,
}

/// Rows the feature input buffer holds at first (the engine's prefill chunk).
const ROWS0: usize = 2048;

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The drafter's limits and the last state the log set, on the host: the log's values are checked against the drafter's
/// bounds contract (`ink_rt_api::Drafter`) before they reach the device.
struct Bounds {
    sessions: usize,
    max_ctx: usize,
    anchor: Vec<i32>,
    pos: Vec<i32>,
}

impl Bounds {
    /// A commit's positions and sessions.
    fn commit(&self, pos: &[i32], sess: &[i32]) -> Result<()> {
        let bad_pos = pos
            .iter()
            .find(|&&p| usize::try_from(p).map_or(true, |p| p >= self.max_ctx));
        let bad_sess = sess
            .iter()
            .find(|&&s| usize::try_from(s).map_or(true, |s| s >= self.sessions));
        ensure!(
            bad_pos.is_none() && bad_sess.is_none(),
            "commit outside the drafter: position {bad_pos:?} (of {}), session {bad_sess:?} (of {})",
            self.max_ctx,
            self.sessions
        );
        Ok(())
    }

    /// The last state of each drafted session: anchor in the vocabulary, its draft block below `max_ctx`.
    fn draft(&self, sessions: &[usize]) -> Result<()> {
        for &q in sessions {
            let (a, p) = (self.anchor[q], self.pos[q]);
            ensure!(
                usize::try_from(a).is_ok_and(|a| a < VOCAB) && usize::try_from(p).is_ok_and(|p| p <= self.max_ctx),
                "session {q}: anchor {a} or position {p} outside the drafter's bounds"
            );
        }
        Ok(())
    }
}

/// The drafter's input buffers, grown when a commit has more rows.
struct Inputs {
    rows: usize,
    feats: DevBuf,
    pos: DevBuf,
    sess: DevBuf,
    frnn: DevBuf,
    anchor: DevBuf,
    apos: DevBuf,
}

impl Inputs {
    fn new(d: &mut RustDrafter, n: usize) -> Result<Self> {
        Ok(Self {
            rows: ROWS0,
            feats: d.alloc(ROWS0 * FEAT_ROW_BYTES)?,
            pos: d.alloc(ROWS0 * 4)?,
            sess: d.alloc(ROWS0 * 4)?,
            frnn: d.alloc(n * FEAT_ROW_BYTES)?,
            anchor: d.alloc(n * 4)?,
            apos: d.alloc(n * 4)?,
        })
    }

    fn fit(&mut self, d: &mut RustDrafter, rows: usize) -> Result<()> {
        if rows > self.rows {
            self.rows = rows;
            self.feats = d.alloc(rows * FEAT_ROW_BYTES)?;
            self.pos = d.alloc(rows * 4)?;
            self.sess = d.alloc(rows * 4)?;
        }
        Ok(())
    }
}

fn list(v: &[i32]) -> String {
    v.iter().map(i32::to_string).collect::<Vec<_>>().join(",")
}

/// Replays a commit, frnn or set_state record (the draft records are compared by the caller).
fn replay(
    d: &mut RustDrafter,
    inp: &mut Inputs,
    bounds: &mut Bounds,
    op: Op,
    commits: &mut u64,
    rows: &mut u64,
) -> Result<()> {
    match op {
        Op::Commit { feats, pos, sess } => {
            let r = pos.len();
            ensure!(
                sess.len() == r,
                "commit record with {r} positions and {} sessions",
                sess.len()
            );
            bounds.commit(&pos, &sess)?;
            inp.fit(d, r)?;
            if r > 0 {
                d.upload(inp.feats, &feats)?;
                d.upload(inp.pos, &i32_bytes(&pos))?;
                d.upload(inp.sess, &i32_bytes(&sess))?;
            }
            #[allow(unsafe_code)] // the drafter's bounds contract
            // SAFETY: the uploaded positions and sessions were checked against the drafter's above
            let commit = unsafe { d.commit_rows(inp.feats, inp.pos, inp.sess, r) };
            commit?;
            *commits += 1;
            *rows += r as u64;
        }
        Op::Frnn(feats) => {
            d.upload(inp.frnn, &feats)?;
            d.frnn(inp.frnn)?;
        }
        Op::State { anchor, pos } => {
            ensure!(
                anchor.len() == bounds.sessions && pos.len() == bounds.sessions,
                "state record of {} anchors and {} positions for {} sessions",
                anchor.len(),
                pos.len(),
                bounds.sessions
            );
            d.upload(inp.anchor, &i32_bytes(&anchor))?;
            d.upload(inp.apos, &i32_bytes(&pos))?;
            #[allow(unsafe_code)] // the drafter's bounds contract
            // SAFETY: every draft checks its sessions' state (this one, kept in `bounds`) before it runs (main)
            let state = unsafe { d.set_state(inp.anchor, inp.apos) };
            state?;
            (bounds.anchor, bounds.pos) = (anchor, pos);
        }
        Op::Draft(_) | Op::DraftSessions { .. } => unreachable!("draft records are handled by the caller"),
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let file = File::open(&cli.log).with_context(|| cli.log.display().to_string())?;
    let mut log = LogReader::new(BufReader::with_capacity(1 << 24, file))?;
    let h = log.header;
    let n = h.sessions as usize;
    let config = DraftConfig {
        model_dir: cli.model.clone(),
        fold_bin: cli.fold.clone(),
        dflash2_dir: None,
        sessions: h.sessions,
        max_ctx: h.max_ctx,
        head: if h.int8 {
            HeadPrecision::Int8
        } else {
            HeadPrecision::Bf16
        },
        global_kv: if h.kv8 { KvPrecision::E4m3 } else { KvPrecision::Bf16 },
        verify: VerifyWidth::Full,
        lookup: 0,
        lookup_min: 8,
    };
    let me = Source::this_build();
    let mut record = cli.record.writer_with(
        "draft",
        me.clone(),
        [(ComponentName::Drafter, Component::Source(me))],
        json!({"log": cli.log, "model": cli.model, "fold": cli.fold, "sessions": h.sessions, "max_ctx": h.max_ctx,
               "int8": h.int8, "kv8": h.kv8, "taps_at": cli.taps_at, "time_iters": cli.time_iters,
               "time_reps": cli.time_reps}),
        "the drafts in the log (C++ ink::DraftRT through ink_rt.h, ink-spec --draft-log)",
    )?;
    let t0 = std::time::Instant::now();
    let mut d = RustDrafter::new(&config)?;
    eprintln!(
        "[draft] loaded in {:.1} s: {n} sessions, max_ctx {}",
        t0.elapsed().as_secs_f64(),
        h.max_ctx
    );
    let mut inp = Inputs::new(&mut d, n)?;
    // the drafter starts with zero anchors and positions
    let mut bounds = Bounds {
        sessions: n,
        max_ctx: h.max_ctx as usize,
        anchor: vec![0; n],
        pos: vec![0; n],
    };
    let taps_at: BTreeSet<u64> = cli.taps_at.iter().copied().collect();
    let mut rounds = BufWriter::new(File::create(cli.record.file("rounds.jsonl"))?);
    let (mut drafts_n, mut equal, mut commits, mut rows) = (0u64, 0u64, 0u64, 0u64);
    let mut first_diff: Option<u64> = None;
    let mut got = vec![0u8; n * DRAFTS * 4];
    let mut last_k = n;
    while let Some(op) = log.next_op()? {
        // a draft record: the drafted sessions and the logged drafts
        let (sessions, want): (Vec<usize>, Vec<i32>) = match op {
            Op::Draft(want) => ((0..n).collect(), want),
            Op::DraftSessions { sessions, drafts } => (
                sessions
                    .iter()
                    .map(|&q| usize::try_from(q).context("negative session in the log"))
                    .collect::<Result<_>>()?,
                drafts,
            ),
            op => {
                replay(&mut d, &mut inp, &mut bounds, op, &mut commits, &mut rows)?;
                continue;
            }
        };
        let k = drafts_n;
        last_k = sessions.len();
        ensure!(
            sessions.iter().all(|&q| q < n),
            "draft record names a session outside the drafter's {n}"
        );
        bounds.draft(&sessions)?;
        d.draft_sessions(&sessions)?;
        d.download(d.drafts(), &mut got)?;
        let mine: Vec<i32> = got.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
        let rows_of = |v: &[i32]| -> Vec<i32> {
            sessions
                .iter()
                .flat_map(|&q| v[q * DRAFTS..(q + 1) * DRAFTS].iter().copied())
                .collect()
        };
        let same = rows_of(&mine) == rows_of(&want);
        equal += u64::from(same);
        if !same && first_diff.is_none() {
            first_diff = Some(k);
            eprintln!(
                "[draft] draft {k} differs: rust [{}] log [{}]",
                list(&rows_of(&mine)),
                list(&rows_of(&want))
            );
        }
        writeln!(
            rounds,
            "{{\"draft\":{k},\"sessions\":[{}],\"equal\":{same},\"drafts\":[{}]}}",
            list(
                &sessions
                    .iter()
                    .map(|&q| i32::try_from(q).unwrap_or(-1))
                    .collect::<Vec<_>>()
            ),
            list(&mine)
        )?;
        if k == 0 {
            let (lines, map) = d
                .launch_trace(sessions.len())?
                .context("no graph after the first draft")?;
            let mut t = BufWriter::new(File::create(cli.record.file("trace.jsonl"))?);
            writeln!(
                t,
                "{{\"format\":\"ink-launch-trace/1\",\"key\":\"drafter n={n} k={} max_ctx={}\",\"nodes\":{}}}",
                sessions.len(),
                h.max_ctx,
                lines.len()
            )?;
            for l in &lines {
                writeln!(t, "{l}")?;
            }
            t.flush()?;
            std::fs::write(cli.record.file("arena-map.jsonl"), map)?;
        }
        if taps_at.contains(&k) {
            let mut t = BufWriter::new(File::create(cli.record.file(&format!("taps-{k}.jsonl")))?);
            for tap in d.taps()? {
                writeln!(
                    t,
                    "{{\"name\":\"{}\",\"bytes\":{},\"fnv1a64\":\"{:016x}\"}}",
                    tap.name, tap.bytes, tap.fnv1a64
                )?;
            }
            t.flush()?;
        }
        drafts_n += 1;
    }
    rounds.flush()?;
    let mut times = Vec::new();
    if cli.time_iters > 0 && drafts_n > 0 {
        let mut t = BufWriter::new(File::create(cli.record.file("time.jsonl"))?);
        d.draft_ms(last_k, cli.time_iters.min(20))?; // warm-up
        for rep in 0..cli.time_reps {
            let ms = d.draft_ms(last_k, cli.time_iters)?;
            times.push(ms);
            writeln!(
                t,
                "{{\"rep\":{rep},\"iters\":{},\"ms_per_draft\":{ms:.6}}}",
                cli.time_iters
            )?;
        }
        t.flush()?;
    }
    eprintln!(
        "[draft] {drafts_n} drafts, {equal} equal to the log; {commits} commits ({rows} rows); first difference: {first_diff:?}"
    );
    record.append(
        &json!({"done": true, "drafts": drafts_n, "equal": equal, "first_diff": first_diff,
        "commits": commits, "rows": rows, "ms_per_draft": times}),
    )?;
    eprintln!("record {}", record.path().display());
    if equal != drafts_n {
        std::process::exit(1);
    }
    Ok(())
}
