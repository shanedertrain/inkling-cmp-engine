//! Batch- and chunk-invariance tests (package T1; port of `engine/native/tools/inv.cpp`, deleted at cut-over U3a).
//!
//! ```text
//! ink-inv --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir DIR --cases cases.txt [--max-chunk 4096] > inv.jsonl
//! ```
//!
//! Every test compares the fp32 logits of one session bit for bit between two runs that differ only in what else is
//! in the forward calls (other sessions, their widths, a prefill chunk riding along), in the prefill chunking, in
//! graphs against eager launch, in the expert path or in the glue path. One JSON line per compared row, in the C++
//! tool's format. `--max-chunk 4096` (or 8192) adds the 4,096-row (and 8,192-row) chunk test.
//! `verify_layouts_graph_vs_eager`: verify calls of one graph key (segments, rows, decode-attention variant) with
//! other per-segment widths, replayed from one graph, against eager calls.
//! `verify_rows_vs_decode`: every row of 8-row verify segments, committed with varying accept counts, against the
//! one-row decode row at the same position (page, window and ring boundaries of `tf-raw-5000`, the three agent prompts).
//! `small_chunk_{alone,mixed,resumed}_S`: a prefix ended by a prefill call of S = 9 .. 65 rows (alone, mixed with
//! decode rows, or resumed) against the prefix in one call: a row's bits depend on its row class, never on its call size.
//! `resume_vs_cold` (a runtime with slot checkpoints): a slot checkpointed at `P`, written past it (prefill chunks
//! or speculative verify rows with rollbacks, inside and beyond the sliding-window ring), rolled back to `P` and
//! prefilled to `E` must give the logits of a cold prefill of `[0, E)` and of the decode rows after it, bit for bit.
//! `--panel FILE` (from `bench/raw/2026-11-23-prefix-reuse/tools/panel_ids.py`) adds `resume_vs_cold_panel`: every
//! follow-up turn of the agent-replay panel resumed from the previous turn's prompt-end checkpoint after its recorded
//! completion was written, against a cold prefill of the whole prompt (last-row logits), and
//! `reply_reuse_vs_cold_panel`: the turns whose previous reply prefixes the next prompt resumed whole (the reply's
//! decode-row K/V reused, the served behaviour). `reply_reuse_vs_cold` (synthetic) and the panel rows are not bit
//! gates: each next-token row must stay within KL 1e-3 nats of the cold row with the same argmax.
#![forbid(unsafe_code)]
use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use ink_rt_api::{Glue, Graphs, Runtime, SegIn, VOCAB};
use ink_tools::cases::read_cases;
use ink_tools::engine::{EngineArgs, RecordArgs, identity_fields};
use ink_tools::fmt;
use ink_tools::session::{prefill, row};
use serde_json::json;

#[derive(Parser)]
#[command(name = "ink-inv", version = ink_record::BUILD_LINE, about = "Batch- and chunk-invariance tests")]
struct Cli {
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    record: RecordArgs,
    /// Golden cases (`cases.txt` of `gold_prep.py`).
    #[arg(long, default_value = "/gin/cases.txt")]
    cases: PathBuf,
    /// Rows per prefill segment of the engine (`t_max` = this + 64); 4096 and 8192 add the long-chunk tests.
    #[arg(long, default_value_t = 2048)]
    max_chunk: u32,
    /// Agent-replay panel token ids (one JSON line per session: `turns[].prompt`, `turns[].completion`): adds
    /// `resume_vs_cold_panel`.
    #[arg(long)]
    panel: Option<PathBuf>,
}

/// KV pages of the tool's engine (`inv.cpp`).
const PAGES: u32 = 200;
/// Single-row decode calls before the 8-row verify-shaped call.
const STEPS: usize = 6;
/// The sessions of the batch tests: golden cases whose prompt and continuation form the token streams.
const NAMES: [&str; 8] = [
    "tf-chat-prose",
    "tf-chat-code",
    "tf-raw-300",
    "tf-chat-tools-first",
    "tf-raw-2100",
    "gr-code-0",
    "gr-prose-0",
    "tf-chat-structured-low",
];
/// Prompt rows per session.
const PLEN: [usize; 8] = [40, 80, 250, 200, 1500, 300, 90, 50];
/// The prefill chunk of the mixed calls: stream 4, rows 1,000..1,300, after a 1,000-row prefill of its slot.
const MIXED_PREFIX: usize = 1000;
const MIXED_ROWS: usize = 300;

/// `verify_rows_vs_decode`: positions covered per context, and the rows committed by successive verify calls.
const VERIFY_SPAN: usize = 24;
const VERIFY_KEEP: [usize; 9] = [1, 8, 3, 5, 2, 8, 4, 7, 6];
/// Commit sequence for 16-row `verify_rows_vs_decode16` (exercises both `<= 8` and `9..=16` commits).
const VERIFY_KEEP16: [usize; 6] = [16, 1, 9, 13, 5, 16];
/// Its contexts in `tf-raw-5000`: verify blocks straddling the 1,024-key pages (global chunks), the SWA window edge
/// (keys pos - 1,023 cross 128-key chunks and ring pages at every one of these) and the 3-page ring wrap at 3,072.
const VERIFY_STARTS: [usize; 4] = [1012, 2036, 3060, 4084];
/// Its agent prompts (the greedy cases where F1 was found): prompt + golden continuation.
const VERIFY_AGENT: [&str; 3] = [
    "gr-agent-short-MF3-r1",
    "gr-agent-median-CONC1-r3",
    "gr-agent-short-REF1-r2",
];

type Rows = Vec<Vec<f32>>;

/// How `resume_vs_cold` writes past its checkpoint before rolling back.
#[derive(Clone, Copy, Debug)]
enum Tail {
    /// Prefill chunks of this many rows in total.
    Prefill(usize),
    /// 8-row verify segments committing `VERIFY_KEEP` rows in turn, until this many rows are committed.
    Verify(usize),
}

/// `resume_vs_cold` cases: (checkpoint `P`, tail, end `E` of the resumed prefill). Checkpoints below the window, at a
/// page boundary, mid-page, at the ring wrap (3,072); a tail whose last row lands just before the first window key's
/// ring entry (no window copy) and one row more (window saved and restored); long tails across several ring periods;
/// speculative tails; a 9-row resumed suffix.
const RESUME_CASES: [(usize, Tail, usize); 9] = [
    (700, Tail::Prefill(300), 1500),
    (1024, Tail::Verify(40), 2100),
    (2000, Tail::Prefill(2049), 3000),
    (2000, Tail::Prefill(2050), 3000),
    (3072, Tail::Verify(60), 4000),
    (2500, Tail::Prefill(6000), 3500),
    (1500, Tail::Verify(30), 1517),
    (4200, Tail::Prefill(2500), 4900),
    (3000, Tail::Verify(2100), 3700),
];
/// Reply-reuse gate: KL(cold || resumed) per next-token row, nats. Measured: agent panel (28 turns,
/// `bench/raw/2026-11-23-prefix-reuse/` g7-invp) max 2.4e-7; raw web text `tf-raw-5000` (the synthetic row, high-entropy
/// next tokens) max 1.4e-4 (`bench/raw/2026-11-24-chunk-policy/` g7-gateR). The bound keeps 7x over the worst measured
/// row and stays 5x below the lossless envelope's mean-KL limit (about 0.0053).
const REPLY_KL_MAX: f64 = 1e-3;
/// One-row decode steps compared after each resumed prefill.
const RESUME_DECODE: usize = 4;
/// `small_chunk_*`: prefill calls of these many rows end a 1,500-token prefix (row-class invariance across the
/// `T > 64` kernel switches and the tensor-core gate tail of prefill rows: delta 3 defect, `bench/raw/2026-11-24-rust-delta3/`).
/// Segments of 1..=16 rows are decode class (16-row single-pass `decode_attn`), so prefill starts at 17.
const SMALL_CHUNKS: [usize; 5] = [17, 33, 52, 64, 65];
const SMALL_END: usize = 1500;

/// How `small_chunk` runs its last prefill call.
#[derive(Clone, Copy, Debug)]
enum Small {
    /// The call holds the chunk alone.
    Alone,
    /// The chunk rides between a one-row decode segment and an 8-row verify segment of other slots (uncommitted).
    Mixed,
    /// The slot was checkpointed before the chunk, written past it and rolled back first (slot checkpoints only).
    Resumed,
}

/// `ids` in pieces of `chunk` rows, the scheduler's way: no piece but the whole of `ids` is 16 rows or shorter (a
/// segment that short runs as a decode segment, `ink-sched`'s prefill chunk rule).
fn chunked(ids: &[i32], chunk: usize) -> Vec<&[i32]> {
    let mut out = Vec::new();
    let mut rest = ids;
    while !rest.is_empty() {
        let mut c = chunk.min(rest.len());
        let left = rest.len() - c;
        if left > 0 && left <= 16 {
            c -= 17 - left;
        }
        let (a, b) = rest.split_at(c);
        out.push(a);
        rest = b;
    }
    out
}

/// The `--panel` file: per session its turns' prompt and completion ids.
type Panel = Vec<(String, Vec<(Vec<i32>, Vec<i32>)>)>;

fn read_panel(path: &std::path::Path) -> Result<Panel> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let ids = |x: &serde_json::Value| -> Result<Vec<i32>> {
        x.as_array()
            .context("token ids")?
            .iter()
            .map(|t| t.as_i64().and_then(|t| i32::try_from(t).ok()).context("token id"))
            .collect()
    };
    let mut out = Panel::new();
    for line in BufReader::new(f).lines() {
        let v: serde_json::Value = serde_json::from_str(&line?)?;
        let turns = v["turns"]
            .as_array()
            .context("turns")?
            .iter()
            .map(|t| Ok((ids(&t["prompt"])?, ids(&t["completion"])?)))
            .collect::<Result<_>>()?;
        out.push((v["session"].as_str().context("session")?.to_owned(), turns));
    }
    Ok(out)
}

/// A 300-row prefill chunk of slot `slot` (never committed, logits of its last row) inserted at segment index `at` of
/// every call of [`Inv::run_a`]: the mixed calls.
#[derive(Clone, Copy)]
struct Mixed {
    slot: usize,
    at: usize,
}

struct Inv<'a> {
    rt: &'a mut dyn Runtime,
    streams: Vec<Vec<i32>>,
    out: BufWriter<std::io::Stdout>,
    identity: String,
    ndiff_total: usize,
    /// Reply-reuse rows over the KL bound or with another argmax.
    reply_failures: usize,
}

impl Inv<'_> {
    /// Session 0's stream alone or with `others`: prefill, `STEPS` one-row decodes, one 8-row verify-shaped call.
    /// Returns the valid logits of session 0's rows. Session `i` of the call uses stream `i` in slot `slots[i]`.
    fn run_a(&mut self, slot_a: usize, others: &[usize], widths: &[usize], mixed: Option<Mixed>) -> Result<Rows> {
        let mut slots = vec![slot_a];
        slots.extend_from_slice(others);
        for (i, &slot) in slots.iter().enumerate() {
            prefill(self.rt, slot, &self.streams[i][..PLEN[i]], 2048, false)?;
        }
        if let Some(m) = mixed {
            prefill(self.rt, m.slot, &self.streams[4][..MIXED_PREFIX], 2048, false)?;
        }
        let mut posn: Vec<usize> = slots.iter().enumerate().map(|(i, _)| PLEN[i]).collect();
        let mut out = Rows::new();
        let mut lg = Vec::new();
        for st in 0..=STEPS {
            let verify = st == STEPS;
            let mut segs: Vec<SegIn<'_>> = slots
                .iter()
                .enumerate()
                .map(|(i, &slot)| {
                    let w = if !verify {
                        1
                    } else if i == 0 {
                        8
                    } else {
                        widths[(i - 1) % widths.len()]
                    };
                    SegIn {
                        slot,
                        ids: &self.streams[i][posn[i]..posn[i] + w],
                        logits: if verify && i == 0 { 8 } else { 1 },
                    }
                })
                .collect();
            let mut counts: Vec<u32> = segs
                .iter()
                .map(|s| u32::try_from(s.ids.len()))
                .collect::<Result<_, _>>()?;
            if let Some(m) = mixed {
                segs.insert(
                    m.at,
                    SegIn {
                        slot: m.slot,
                        ids: &self.streams[4][MIXED_PREFIX..MIXED_PREFIX + MIXED_ROWS],
                        logits: 1,
                    },
                );
                counts.insert(m.at, 0);
            }
            let n = self.rt.forward(&segs)?;
            self.rt.logits_host(n, &mut lg)?;
            // session A's first logit row: after the prefill chunk's row when that comes first
            let a0 = usize::from(mixed.is_some_and(|m| m.at == 0));
            for r in a0..a0 + if verify { 8 } else { 1 } {
                out.push(row(&lg, r)[..VOCAB].to_vec());
            }
            let session_counts = counts
                .iter()
                .enumerate()
                .filter(|&(i, _)| mixed.is_none_or(|m| m.at != i))
                .map(|(_, &c)| c);
            for (p, c) in posn.iter_mut().zip(session_counts) {
                *p += c as usize;
            }
            self.rt.commit(&counts)?;
        }
        Ok(out)
    }

    /// One JSON line: how many of the fp32 values of `a` and `b` differ in their bits, and the largest difference.
    fn report(&mut self, test: &str, step: usize, a: &[f32], b: &[f32]) -> Result<()> {
        let ndiff = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let max_abs = a
            .iter()
            .zip(b)
            .fold(0.0_f64, |m, (&x, &y)| m.max(f64::from((x - y).abs())));
        self.ndiff_total += ndiff;
        writeln!(
            self.out,
            "{{{},\"test\":\"{test}\",\"step\":{step},\"n\":{},\"ndiff\":{ndiff},\"max_abs\":{}}}",
            self.identity,
            a.len(),
            fmt::g(max_abs, 6)
        )?;
        self.out.flush()?;
        Ok(())
    }

    /// `verify_layouts_graph_vs_eager`: verify calls of 4 segments and 16 rows (logits on every row) whose widths
    /// differ, against the same calls run eagerly, every row bit for bit. With the graph cache keyed by segment
    /// count, rows and decode-attention variant, the second call of each pair replays the graph the first captured
    /// with another row bound `R` (8 then 6; 5 then 4) and another per-segment layout. Nothing is committed: every
    /// call starts from the prompts of slots 0-3.
    fn verify_layouts(&mut self, graphs: Graphs) -> Result<()> {
        const LAYOUTS: [[usize; 4]; 6] = [
            [8, 4, 2, 2],
            [2, 2, 6, 6],
            [5, 5, 5, 1],
            [4, 4, 4, 4],
            [16, 4, 2, 2],
            [2, 2, 6, 14],
        ];
        for (i, &p) in PLEN.iter().enumerate().take(4) {
            prefill(self.rt, i, &self.streams[i][..p], 2048, false)?;
        }
        let run = |t: &mut Self| -> Result<Vec<Rows>> {
            let mut lg = Vec::new();
            let mut all = Vec::new();
            for w in LAYOUTS {
                let segs: Vec<SegIn<'_>> = (0..4)
                    .map(|i| SegIn {
                        slot: i,
                        ids: &t.streams[i][PLEN[i]..PLEN[i] + w[i]],
                        logits: w[i],
                    })
                    .collect();
                let n = t.rt.forward(&segs)?;
                t.rt.logits_host(n, &mut lg)?;
                t.rt.commit(&[0; 4])?;
                all.push((0..n).map(|r| row(&lg, r)[..VOCAB].to_vec()).collect());
            }
            Ok(all)
        };
        self.rt.set_graphs(Graphs::Off)?;
        let eager = run(self)?;
        self.rt.set_graphs(graphs)?;
        let graphed = run(self)?;
        let mut step = 0;
        for (k, (a, b)) in eager.iter().zip(&graphed).enumerate() {
            let want: usize = LAYOUTS[k].iter().sum();
            ensure!(
                a.len() == want && b.len() == want,
                "layout {k}: {} and {} rows (expected {want})",
                a.len(),
                b.len()
            );
            for (x, y) in a.iter().zip(b) {
                self.report("verify_layouts_graph_vs_eager", step, x, y)?;
                step += 1;
            }
        }
        Ok(())
    }

    fn report_rows(&mut self, test: &str, a: &Rows, b: &Rows) -> Result<()> {
        for (r, (x, y)) in a.iter().zip(b).enumerate() {
            self.report(test, r, x, y)?;
        }
        Ok(())
    }

    /// Verify rows against one-row decode rows (`verify_rows_vs_decode`, finding F1): `ids[..start]` prefilled in two
    /// slots; slot 0 decodes `ids[start..start + VERIFY_SPAN + 7]` one row per call, slot 1 runs 8-row verify segments
    /// of the same tokens and commits `VERIFY_KEEP[i]` rows of call `i` (the speculative accept / rollback path). Every
    /// verify row is compared bit for bit with the decode row at its position: the rows and the history they commit
    /// must equal one-row decode exactly.
    fn verify_vs_decode(&mut self, label: &str, ids: &[i32], start: usize) -> Result<()> {
        let end = start + VERIFY_SPAN;
        ensure!(ids.len() >= end + 7, "{label}: {} tokens, need {}", ids.len(), end + 7);
        prefill(self.rt, 0, &ids[..start], 2048, false)?;
        prefill(self.rt, 1, &ids[..start], 2048, false)?;
        let mut lg = Vec::new();
        let mut dec = Rows::new();
        for &tok in &ids[start..end + 7] {
            let n = self.rt.forward(&[SegIn {
                slot: 0,
                ids: &[tok],
                logits: 1,
            }])?;
            self.rt.logits_host(n, &mut lg)?;
            dec.push(row(&lg, 0)[..VOCAB].to_vec());
            self.rt.commit(&[1])?;
        }
        let (mut pos, mut call) = (start, 0);
        while pos < end {
            let n = self.rt.forward(&[SegIn {
                slot: 1,
                ids: &ids[pos..pos + 8],
                logits: 8,
            }])?;
            self.rt.logits_host(n, &mut lg)?;
            for j in 0..8 {
                let test = format!("verify_rows_vs_decode:{label}@{}", pos + j);
                self.report(&test, j, &dec[pos + j - start], &row(&lg, j)[..VOCAB])?;
            }
            let keep = VERIFY_KEEP[call % VERIFY_KEEP.len()];
            self.rt.commit(&[u32::try_from(keep)?])?;
            pos += keep;
            call += 1;
        }
        Ok(())
    }

    /// 16-row single-pass verify rows against one-row decode rows (F1 for 9..=16 rows): every row `0..15` must equal
    /// one-row decode at the same position bit for bit.
    fn verify_vs_decode16(&mut self, label: &str, ids: &[i32], start: usize) -> Result<()> {
        let end = start + VERIFY_SPAN;
        ensure!(
            ids.len() >= end + 15,
            "{label}: {} tokens, need {}",
            ids.len(),
            end + 15
        );
        prefill(self.rt, 0, &ids[..start], 2048, false)?;
        prefill(self.rt, 1, &ids[..start], 2048, false)?;
        let mut lg = Vec::new();
        let mut dec = Rows::new();
        for &tok in &ids[start..end + 15] {
            let n = self.rt.forward(&[SegIn {
                slot: 0,
                ids: &[tok],
                logits: 1,
            }])?;
            self.rt.logits_host(n, &mut lg)?;
            dec.push(row(&lg, 0)[..VOCAB].to_vec());
            self.rt.commit(&[1])?;
        }
        let (mut pos, mut call) = (start, 0);
        while pos < end {
            let n = self.rt.forward(&[SegIn {
                slot: 1,
                ids: &ids[pos..pos + 16],
                logits: 16,
            }])?;
            self.rt.logits_host(n, &mut lg)?;
            for j in 0..16 {
                let test = format!("verify_rows16_vs_decode:{label}@{}", pos + j);
                self.report(&test, j, &dec[pos + j - start], &row(&lg, j)[..VOCAB])?;
            }
            let keep = VERIFY_KEEP16[call % VERIFY_KEEP16.len()];
            self.rt.commit(&[u32::try_from(keep)?])?;
            pos += keep;
            call += 1;
        }
        Ok(())
    }

    /// Prefills `ids` into `slot` from its current length in calls of at most `chunk` rows, without a reset; returns
    /// the last row's logits (`VOCAB` values).
    fn extend(&mut self, slot: usize, ids: &[i32], chunk: usize) -> Result<Vec<f32>> {
        let mut lg = Vec::new();
        let pieces = chunked(ids, chunk);
        for (i, piece) in pieces.iter().enumerate() {
            let last = i + 1 == pieces.len();
            let n = self.rt.forward(&[SegIn {
                slot,
                ids: piece,
                logits: usize::from(last),
            }])?;
            if last {
                self.rt.logits_host(n, &mut lg)?;
            }
            self.rt.commit(&[u32::try_from(piece.len())?])?;
        }
        Ok(row(&lg, 0)[..VOCAB].to_vec())
    }

    /// One-row decode steps of `ids` in `slot`: their logit rows.
    fn decode_rows(&mut self, slot: usize, ids: &[i32]) -> Result<Rows> {
        let mut lg = Vec::new();
        let mut out = Rows::new();
        for &tok in ids {
            let n = self.rt.forward(&[SegIn {
                slot,
                ids: &[tok],
                logits: 1,
            }])?;
            self.rt.logits_host(n, &mut lg)?;
            out.push(row(&lg, 0)[..VOCAB].to_vec());
            self.rt.commit(&[1])?;
        }
        Ok(out)
    }

    /// Writes `tail` rows of `junk` into `slot` past its length.
    fn write_tail(&mut self, slot: usize, junk: &[i32], tail: Tail) -> Result<()> {
        match tail {
            Tail::Prefill(n) => {
                ensure!(junk.len() >= n, "tail of {n} rows from {} tokens", junk.len());
                self.extend(slot, &junk[..n], 2048)?;
            }
            Tail::Verify(n) => {
                let (mut done, mut call) = (0, 0);
                while done < n {
                    ensure!(junk.len() >= done + 8, "verify tail past {} tokens", junk.len());
                    self.rt.forward(&[SegIn {
                        slot,
                        ids: &junk[done..done + 8],
                        logits: 8,
                    }])?;
                    let keep = VERIFY_KEEP[call % VERIFY_KEEP.len()].min(n - done);
                    self.rt.commit(&[u32::try_from(keep)?])?;
                    done += keep;
                    call += 1;
                }
            }
        }
        Ok(())
    }

    fn checkpoint(&mut self, slot: usize) -> Result<()> {
        let ck = self.rt.checkpoints().context("the runtime has no slot checkpoints")?;
        ck.checkpoint(slot)?;
        Ok(())
    }

    fn rollback(&mut self, slot: usize, pos: usize) -> Result<()> {
        let ck = self.rt.checkpoints().context("the runtime has no slot checkpoints")?;
        ck.rollback(slot, pos)?;
        ensure!(
            self.rt.length(slot)? == pos,
            "slot {slot} not at {pos} after the rollback"
        );
        Ok(())
    }

    /// `resume_vs_cold`: slot 1 prefills `ids[..e]` cold and decodes `RESUME_DECODE` rows; slot 0 is prefilled to `p`
    /// and checkpointed, then twice (the checkpoint stays valid after a resume): `tail` written past it, rolled back
    /// to `p`, prefilled to `e`, the same decode rows. Every row compared bit for bit.
    fn resume_vs_cold(&mut self, ids: &[i32], junk: &[i32], p: usize, tail: Tail, e: usize) -> Result<()> {
        ensure!(
            ids.len() >= e + RESUME_DECODE,
            "{} tokens, need {}",
            ids.len(),
            e + RESUME_DECODE
        );
        let dec = &ids[e..e + RESUME_DECODE];
        self.rt.reset_slot(1)?;
        let mut cold = vec![self.extend(1, &ids[..e], 2048)?];
        cold.extend(self.decode_rows(1, dec)?);
        self.rt.reset_slot(0)?;
        self.extend(0, &ids[..p], 2048)?;
        self.checkpoint(0)?;
        for round in 0..2 {
            self.write_tail(0, junk, tail)?;
            self.rollback(0, p)?;
            let mut warm = vec![self.extend(0, &ids[p..e], 2048)?];
            warm.extend(self.decode_rows(0, dec)?);
            let test = format!("resume_vs_cold:P{p}:{tail:?}:E{e}:round{round}");
            self.report_rows(&test, &cold, &warm)?;
        }
        Ok(())
    }

    /// `small_chunk_{alone,mixed,resumed}_S`: `ids[..SMALL_END]` prefilled in one call in slot 1 against slot 0 with
    /// `ids[..SMALL_END - s]` in one call and the last `s` rows as a call of their own (alone, mixed with decode rows
    /// of slots 3 and 4, or resumed after a checkpoint, a written tail and a rollback); the last row and the decode
    /// rows after it, bit for bit. A row's bits must depend on its row class only, never on the size of its call.
    fn small_chunk(&mut self, ids: &[i32], junk: &[i32], s: usize, how: Small) -> Result<()> {
        let (e, p) = (SMALL_END, SMALL_END - s);
        let dec = &ids[e..e + RESUME_DECODE];
        self.rt.reset_slot(1)?;
        let mut cold = vec![self.extend(1, &ids[..e], 2048)?];
        cold.extend(self.decode_rows(1, dec)?);
        self.rt.reset_slot(0)?;
        self.extend(0, &ids[..p], 2048)?;
        let mut lg = Vec::new();
        let last = match how {
            Small::Alone => self.extend(0, &ids[p..e], 2048)?,
            Small::Resumed => {
                self.checkpoint(0)?;
                self.write_tail(0, junk, Tail::Prefill(100))?;
                self.rollback(0, p)?;
                self.extend(0, &ids[p..e], 2048)?
            }
            Small::Mixed => {
                for slot in [3, 4] {
                    self.rt.reset_slot(slot)?;
                    self.extend(slot, &junk[..200], 2048)?;
                }
                let n = self.rt.forward(&[
                    SegIn {
                        slot: 3,
                        ids: &junk[200..201],
                        logits: 1,
                    },
                    SegIn {
                        slot: 0,
                        ids: &ids[p..e],
                        logits: 1,
                    },
                    SegIn {
                        slot: 4,
                        ids: &junk[200..208],
                        logits: 8,
                    },
                ])?;
                self.rt.logits_host(n, &mut lg)?;
                self.rt.commit(&[0, u32::try_from(s)?, 0])?;
                row(&lg, 1)[..VOCAB].to_vec()
            }
        };
        let mut warm = vec![last];
        warm.extend(self.decode_rows(0, dec)?);
        let name = match how {
            Small::Alone => "alone",
            Small::Mixed => "mixed",
            Small::Resumed => "resumed",
        };
        self.report_rows(&format!("small_chunk_{name}_{s}"), &cold, &warm)
    }

    /// `reply_reuse_vs_cold`: `ids[..e]` cold in slot 1 against slot 0 with `ids[p..p + k]` committed as one-row
    /// decode rows (a parked sequence's own reply, resumed whole) and the rest prefilled; the last row and the decode
    /// rows after it, each within [`REPLY_KL_MAX`] of the cold row and with its argmax.
    fn decode_prefix_vs_prefill(&mut self, ids: &[i32], p: usize, k: usize, e: usize) -> Result<()> {
        let dec = &ids[e..e + RESUME_DECODE];
        self.rt.reset_slot(1)?;
        let mut cold = vec![self.extend(1, &ids[..e], 2048)?];
        cold.extend(self.decode_rows(1, dec)?);
        self.rt.reset_slot(0)?;
        self.extend(0, &ids[..p], 2048)?;
        self.decode_rows(0, &ids[p..p + k])?;
        let mut warm = vec![self.extend(0, &ids[p + k..e], 2048)?];
        warm.extend(self.decode_rows(0, dec)?);
        for (r, (c, w)) in cold.iter().zip(&warm).enumerate() {
            self.report_r(&format!("reply_reuse_vs_cold:P{p}:K{k}:E{e}:row{r}"), c, w)?;
        }
        Ok(())
    }

    /// `resume_vs_cold_panel`: per session, slot 0 runs the turns as the served engine does (prompt, checkpoint,
    /// the recorded reply written past it, rollback, the next prompt's new tokens), slot 1 prefills every prompt
    /// cold; last-row logits compared.
    fn resume_panel(&mut self, panel: &Panel) -> Result<()> {
        for (name, turns) in panel {
            for slot in [0, 2] {
                self.rt.reset_slot(slot)?;
                self.extend(slot, &turns[0].0, 2048)?;
                self.checkpoint(slot)?;
            }
            for i in 1..turns.len() {
                let (prev, reply) = (&turns[i - 1].0, &turns[i - 1].1);
                let prompt = &turns[i].0;
                let p = prev.len();
                ensure!(
                    prompt.len() > p && prompt[..p] == prev[..],
                    "{name} turn {i}: not append-only"
                );
                // the previous reply's rows (its last token is sampled, never fed)
                let reply = &reply[..reply.len().saturating_sub(1)];
                if !reply.is_empty() {
                    self.extend(0, reply, 2048)?;
                }
                self.rollback(0, p)?;
                let warm = self.extend(0, &prompt[p..], 2048)?;
                self.checkpoint(0)?;
                self.rt.reset_slot(1)?;
                let cold = self.extend(1, prompt, 2048)?;
                let test = format!(
                    "resume_vs_cold_panel:{name}:t{i}:P{p}:tail{}:E{}",
                    reply.len(),
                    prompt.len()
                );
                self.report(&test, 0, &cold, &warm)?;
                // reply reuse in slot 2: the reply as decode rows (8-row segments, all committed: what verify rows
                // write), then the whole sequence resumed when it is a prefix of the prompt
                let mut done = 0;
                while done < reply.len() {
                    let n = (reply.len() - done).min(8);
                    self.rt.forward(&[SegIn {
                        slot: 2,
                        ids: &reply[done..done + n],
                        logits: n,
                    }])?;
                    self.rt.commit(&[u32::try_from(n)?])?;
                    done += n;
                }
                let held = p + reply.len();
                let whole = prompt.len() > held && prompt[p..held] == reply[..];
                if !whole {
                    self.rollback(2, p)?;
                }
                let r = self.extend(2, &prompt[if whole { held } else { p }..], 2048)?;
                self.checkpoint(2)?;
                if whole {
                    self.report_r(
                        &format!("reply_reuse_vs_cold_panel:{name}:t{i}:L{held}:E{}", prompt.len()),
                        &cold,
                        &r,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// One reply-reuse line comparing next-token logits: differing values, max |difference|, KL(cold || other) over
    /// the softmax and whether the argmax differs; a row over [`REPLY_KL_MAX`] or with another argmax fails the gate.
    fn report_r(&mut self, test: &str, cold: &[f32], other: &[f32]) -> Result<()> {
        let ndiff = cold
            .iter()
            .zip(other)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let max_abs = cold
            .iter()
            .zip(other)
            .fold(0.0_f64, |m, (&x, &y)| m.max(f64::from((x - y).abs())));
        let lse = |v: &[f32]| {
            let m = v.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
            m + v.iter().map(|&x| (f64::from(x) - m).exp()).sum::<f64>().ln()
        };
        let (lp, lq) = (lse(cold), lse(other));
        let kl: f64 = cold
            .iter()
            .zip(other)
            .map(|(&x, &y)| {
                let a = f64::from(x) - lp;
                a.exp() * (a - (f64::from(y) - lq))
            })
            .sum();
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
                .0
        };
        let flip = argmax(cold) != argmax(other);
        let pass = kl <= REPLY_KL_MAX && !flip;
        self.reply_failures += usize::from(!pass);
        writeln!(
            self.out,
            "{{{},\"test\":\"{test}\",\"step\":0,\"n\":{},\"ndiff\":{ndiff},\"max_abs\":{},\"kl\":{},\"argmax_flip\":{flip},\"pass\":{pass}}}",
            self.identity,
            cold.len(),
            fmt::g(max_abs, 6),
            fmt::g(kl, 6)
        )?;
        self.out.flush()?;
        Ok(())
    }

    /// Last logit row of `ids` prefilled in `chunk`-row calls in `slot`.
    fn last_row(&mut self, slot: usize, ids: &[i32], chunk: usize) -> Result<Vec<f32>> {
        let p = prefill(self.rt, slot, ids, chunk, true)?;
        Ok(p.last[..VOCAB].to_vec())
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let (glue, graphs) = (cli.engine.glue_mode()?, cli.engine.graph_mode()?);
    ensure!(
        cli.engine.slots >= 8,
        "the batch tests use 8 slots, --slots is {}",
        cli.engine.slots
    );
    let cases = read_cases(&cli.cases)?;
    // prompt + golden continuation = a long token stream
    let by_id: HashMap<&str, Vec<i32>> = cases
        .iter()
        .map(|c| (c.id.as_str(), c.prompt.iter().chain(&c.tokens).copied().collect()))
        .collect();
    let streams: Vec<Vec<i32>> = NAMES
        .iter()
        .map(|n| by_id.get(n).cloned().with_context(|| format!("missing case {n}")))
        .collect::<Result<_>>()?;
    // the panel test keeps three slots of up to 101K tokens
    let pages = if cli.panel.is_some() { 320 } else { PAGES };
    let mut rt = cli.engine.open(pages, cli.max_chunk)?;
    let source = rt.identity();
    let mut record = cli.record.writer(
        "inv",
        &*rt,
        json!({"engine": cli.engine.json(pages, cli.max_chunk), "cases": cli.cases}),
        "none: every test compares the engine with itself under another batch, chunking or path",
    )?;
    let mut t = Inv {
        rt: &mut *rt,
        streams,
        out: BufWriter::new(std::io::stdout()),
        identity: identity_fields(&source),
        ndiff_total: 0,
        reply_failures: 0,
    };
    t.rt.set_glue(Glue::Reference)?;
    let alone_glue1 = t.run_a(0, &[], &[], None)?;
    t.rt.set_glue(glue)?;
    t.rt.set_graphs(Graphs::Off)?;
    let alone_eager = t.run_a(0, &[], &[], None)?;
    t.rt.set_graphs(graphs)?;
    let alone = t.run_a(0, &[], &[], None)?;
    // `ndiff_total` counts only the tests whose expected result is zero.
    let mut info_total = 0;
    t.report_rows("graph_vs_eager", &alone_eager, &alone)?;
    let before = t.ndiff_total;
    t.report_rows("info_glue1_vs_glue", &alone_glue1, &alone)?;
    info_total += t.ndiff_total - before;
    t.ndiff_total = before;
    let with4 = t.run_a(2, &[0, 1, 3], &[3, 8, 1], None)?;
    t.report_rows("batch4_vs_alone", &alone, &with4)?;
    let with8 = t.run_a(5, &[0, 1, 2, 3, 4, 6, 7], &[1, 8, 2, 8, 5, 1, 7], None)?;
    t.report_rows("batch8_vs_alone", &alone, &with8)?;
    // decode / verify rows riding in calls with a prefill chunk (prefill segment last, first, between decode
    // segments): bit-equal to the decode-only run (decode rows take the decode plans inside the call)
    for (at, test) in [
        (3, "mixed_prefill_last_vs_alone"),
        (0, "mixed_prefill_first_vs_alone"),
        (1, "mixed_prefill_between_vs_alone"),
    ] {
        let mixed = t.run_a(2, &[0, 1], &[8, 3], Some(Mixed { slot: 7, at }))?;
        t.report_rows(test, &alone, &mixed)?;
    }
    // chunk invariance: a 5,000-token stream prefilled in 2,048-row, 700-row and 1,111-row chunks (row 0's logits)
    let ids: Vec<i32> = match by_id.get("tf-raw-5000") {
        Some(p) => p[..5000.min(p.len())].to_vec(),
        None => t.streams[4][..t.streams[4].len().min(2100)].to_vec(),
    };
    let a = t.last_row(0, &ids, 2048)?;
    let b = t.last_row(0, &ids, 700)?;
    let c = t.last_row(1, &ids, 1111)?;
    t.report("chunk700_vs_2048", 0, &a, &b)?;
    t.report("chunk1111_slot1_vs_2048", 0, &a, &c)?;
    if cli.max_chunk >= 4096 {
        let d = t.last_row(2, &ids, 4096)?;
        t.report("chunk4096_slot2_vs_2048", 0, &a, &d)?;
    }
    if cli.max_chunk >= 8192 {
        // 12,000 tokens: chunks 8,192 + 3,808 vs 2,048 x 5 + 1,760
        let ids12 = by_id
            .get("tf-agent-12k")
            .filter(|p| p.len() >= 12_000)
            .context("tf-agent-12k for the 8,192-chunk test")?[..12_000]
            .to_vec();
        let a12 = t.last_row(0, &ids12, 2048)?;
        let b12 = t.last_row(3, &ids12, 8192)?;
        t.report("chunk8192_slot3_vs_2048", 0, &a12, &b12)?;
    }
    // verify rows == one-row decode rows (F1): page / window / ring boundaries, then the agent prompts
    let raw = by_id
        .get("tf-raw-5000")
        .context("tf-raw-5000 for verify_rows_vs_decode")?;
    for start in VERIFY_STARTS {
        t.verify_vs_decode("tf-raw-5000", raw, start)?;
        t.verify_vs_decode16("tf-raw-5000", raw, start)?;
    }
    let prompt_len: HashMap<&str, usize> = cases.iter().map(|c| (c.id.as_str(), c.prompt.len())).collect();
    for name in VERIFY_AGENT {
        let ids = by_id.get(name).with_context(|| format!("missing case {name}"))?;
        t.verify_vs_decode(name, ids, prompt_len[name] + 64)?;
        t.verify_vs_decode16(name, ids, prompt_len[name] + 64)?;
    }
    // verify calls of one graph key with other widths and row bounds: replayed == eager
    t.verify_layouts(graphs)?;
    // small prefill calls (row-class invariance): alone and mixed on every runtime, resumed with slot checkpoints
    let junk: Vec<i32> = raw.iter().chain(raw.iter()).rev().copied().collect();
    let resumable = t.rt.checkpoints().is_some();
    for s in SMALL_CHUNKS {
        t.small_chunk(raw, &junk, s, Small::Alone)?;
        t.small_chunk(raw, &junk, s, Small::Mixed)?;
        if resumable {
            t.small_chunk(raw, &junk, s, Small::Resumed)?;
        }
    }
    // resume_vs_cold (runtimes with slot checkpoints): page / window / ring boundaries, tails inside and past the
    // ring period, speculative tails; tokens written past the checkpoint are the stream reversed
    if resumable {
        for (p, tail, e) in RESUME_CASES {
            t.resume_vs_cold(raw, &junk, p, tail, e)?;
        }
        t.decode_prefix_vs_prefill(raw, 2000, 24, 3000)?;
        if let Some(path) = &cli.panel {
            let panel = read_panel(path)?;
            // two slots of up to 101K tokens: the other slots' pages back to the pool
            for slot in 0..cli.engine.slots as usize {
                t.rt.reset_slot(slot)?;
            }
            t.resume_panel(&panel)?;
        }
    } else {
        eprintln!("[inv] resume_vs_cold skipped: the runtime has no slot checkpoints");
    }
    let stats = t.rt.stats()?;
    let failing = t.ndiff_total;
    let reply = t.reply_failures;
    eprintln!(
        "[inv] done, gemm fallbacks {}, graph captures {}, differing bits in the gated tests {failing}, reply-reuse rows \
         failing (KL > {REPLY_KL_MAX:e} or argmax) {reply}",
        stats.gemm_fallbacks, stats.graph_captures
    );
    record.append(
        &json!({"done": true, "differing_bits_gated_tests": failing, "differing_bits_info_tests": info_total,
        "reply_reuse_failures": reply, "reply_reuse_kl_max": REPLY_KL_MAX,
        "stats": {"graph_captures": stats.graph_captures, "gemm_fallbacks": stats.gemm_fallbacks}}),
    )?;
    eprintln!("record {}", record.path().display());
    Ok(())
}
