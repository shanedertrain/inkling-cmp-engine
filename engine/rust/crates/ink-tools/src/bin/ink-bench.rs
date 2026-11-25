//! Timing tool (package T2): port of `engine/native/tools/bench.cpp` over the C ABI seam.
//!
//! ```text
//! ink-bench --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir DIR --model /model --cases /gin/cases.txt \
//!           --sessions 4 --ctx 45000 --rows 1|8 --secs 24 [--commit 1|4] [--ablate "a;b,c"] [--mode ...]
//! ```
//!
//! Modes (`--mode`): `decode` (default; N sessions of `--ctx` real tokens, then decode steps back to back for
//! `--secs`: one forward call of N segments of `--rows` rows, `--commit` rows kept, a stream sync; steps that start in
//! the first `--discard` s are dropped; every `--prof-every`-th step runs eager with marks for the per-component split;
//! `--ablate` cells), `prefill` (cold `--ctx`, then warm `--suffix` lists, `--chunk`, `--reps`, `--prof`), `pmix` (a
//! prompt on a cached base while the sessions decode), `mixed` (verify rows plus a prefill chunk, one call against
//! two). Output: the C++ tool's JSON lines on stdout (`summ_bench.py`, `summ_ablate.py` read them), the same lines in
//! the record (`--record-dir`). Differences from `bench.cpp`: `--slots` follows `--sessions`; the stream sync is a
//! 4-byte device read (the seam has no sync call); `_gpu_span` is the sum of the marks' intervals; `--fine 1` adds the
//! sub-component marks (`Runtime::set_profile_fine`: the Rust runtime; the C ABI has no such call); the graph-region
//! count on stderr needs an `ink_rt.h` call it does not have (see `bench/raw/2026-11-23-rust-T2-tools/`).
#![forbid(unsafe_code)]
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use ink_record::Source;
use ink_rt_api::{Runtime, SegIn};
use ink_tools::cases::read_cases;
use ink_tools::engine::{EngineArgs, RecordArgs};
use ink_tools::line::Line;
use ink_tools::tele::{Card, Tele};
use serde_json::{Value, json};

#[derive(Parser)]
#[command(name = "ink-bench", version = ink_record::BUILD_LINE, about = "Decode, prefill and mixed-call timing")]
struct Cli {
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    record: RecordArgs,
    /// `decode`, `prefill`, `pmix` or `mixed`.
    #[arg(long, default_value = "decode")]
    mode: String,
    /// Golden cases; the token stream comes from one of them (`--stream`).
    #[arg(long, default_value = "/gin/cases.txt")]
    cases: PathBuf,
    /// Case whose prompt and tokens are the token stream.
    #[arg(long, default_value = "tf-agent-120k")]
    stream: String,
    /// Flat JSON list of token ids (config B's prefill text); replaces `--cases`.
    #[arg(long)]
    pool: Option<PathBuf>,
    /// Sessions decoding.
    #[arg(long, default_value_t = 4)]
    sessions: usize,
    /// Context tokens per session.
    #[arg(long, default_value_t = 45_000)]
    ctx: usize,
    /// Rows per session per call: 1 plain decode, 8 verify of depth 7.
    #[arg(long, default_value_t = 1)]
    rows: usize,
    /// Rows kept per session per call (default 1 for 1 row, else 4).
    #[arg(long)]
    commit: Option<usize>,
    /// Seconds per cell.
    #[arg(long, default_value_t = 24.0)]
    secs: f64,
    /// Seconds at the start of a cell that are dropped.
    #[arg(long, default_value_t = 2.0)]
    discard: f64,
    /// Every n-th step records the per-component split (0: never).
    #[arg(long, default_value_t = 10)]
    prof_every: usize,
    /// KV pages.
    #[arg(long, default_value_t = 230)]
    pages: u32,
    /// Ablation cells: `;` between cells, `,` between the regions a cell disables (`full` = none).
    #[arg(long, default_value = "")]
    ablate: String,
    /// Rows per call of the prefill modes.
    #[arg(long, default_value_t = 2048)]
    chunk: usize,
    /// `prefill`: 0 runs without events (clean timing).
    #[arg(long, default_value_t = 1)]
    prof: u8,
    /// `prefill`: sub-component marks (Rust runtime; the C ABI has no such call).
    #[arg(long, default_value_t = 0)]
    fine: u8,
    /// Repetitions of `prefill` and `pmix`.
    #[arg(long)]
    reps: Option<usize>,
    /// Warm suffix lengths of `prefill` and `pmix` (comma list).
    #[arg(long, default_value = "1600,6000")]
    suffix: String,
    /// Cached tokens of the extra session of `pmix` and `mixed`.
    #[arg(long, default_value_t = 32_768)]
    pctx: usize,
    /// Prefill rows per call of `mixed` (comma list).
    #[arg(long, default_value = "256,512,1600")]
    plist: String,
}

fn secs(a: Instant, b: Instant) -> f64 {
    b.duration_since(a).as_secs_f64()
}

fn list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().with_context(|| format!("bad number {t:?} in a list")))
        .collect()
}

/// The pieces every mode shares: the runtime, the token stream, the record and the identity of what is measured.
struct Bench {
    rt: Box<dyn Runtime>,
    tok: Vec<i32>,
    card: Card,
    source: Source,
    record: ink_record::RecordWriter,
    n: usize,
    ctx: usize,
    off: usize,
    glue: u8,
    graphs: u8,
}

impl Bench {
    /// Waits for the queued device work.
    fn sync(&mut self) -> Result<()> {
        Ok(self.rt.sync()?)
    }

    /// The line's first fields as the C++ tool writes them (identity, card).
    fn head(&self, test: &str) -> Line {
        self.head_no_limit(test).num("power_limit_w", self.card.power_limit_w)
    }

    /// As [`Bench::head`] without the power limit (`bench.cpp`'s `mixed_call` line has none).
    fn head_no_limit(&self, test: &str) -> Line {
        Line::new("native_rt", test)
            .identity(&self.source)
            .text("uuid", &self.card.uuid)
            .text("vbios", &self.card.vbios)
    }

    fn emit(&mut self, line: Line) -> Result<()> {
        let value = line.emit()?;
        self.record.append(&json!({"line": value}))?;
        Ok(())
    }

    /// Prefills `slot` with `tokens[from..to]` in calls of `chunk` rows (no logits, every row committed).
    fn fill(&mut self, slot: usize, from: usize, to: usize, chunk: usize) -> Result<()> {
        let mut a = from;
        while a < to {
            let z = to.min(a + chunk);
            let seg = [SegIn {
                slot,
                ids: &self.tok[a..z],
                logits: 0,
            }];
            self.rt.forward(&seg)?;
            self.rt.commit(&[u32::try_from(z - a)?])?;
            a = z;
        }
        Ok(())
    }

    /// Adds the marks of the last call to `comp` (ms per region) and its GPU span to `_gpu_span`.
    fn add_marks(&mut self, comp: &mut BTreeMap<String, f64>) -> Result<()> {
        let marks = self.rt.marks()?;
        let mut span = 0.0;
        for (name, ms) in marks {
            *comp.entry(name).or_default() += f64::from(ms);
            span += f64::from(ms);
        }
        if span > 0.0 {
            *comp.entry("_gpu_span".to_owned()).or_default() += span;
        }
        Ok(())
    }

    /// One decode call over the sessions: N segments of `rows` rows at `pos`, `commit` rows kept.
    fn decode_call(&mut self, pos: &[usize], rows: usize, commit: usize) -> Result<()> {
        let segs: Vec<SegIn<'_>> = (0..self.n)
            .map(|s| {
                let b = self.off * s + pos[s];
                SegIn {
                    slot: s,
                    ids: &self.tok[b..b + rows],
                    logits: rows,
                }
            })
            .collect();
        self.rt.forward(&segs)?;
        self.rt.commit(&vec![u32::try_from(commit)?; self.n])?;
        Ok(())
    }

    fn engine_fields(&self, line: Line) -> Line {
        line.uint("glue", u64::from(self.glue))
    }
}

fn components(comp: &BTreeMap<String, f64>, scale: f64, decimals: usize) -> String {
    let body: Vec<String> = comp
        .iter()
        .map(|(k, v)| format!("\"{k}\":{:.decimals$}", v / scale))
        .collect();
    format!("{{{}}}", body.join(","))
}

fn percentile(sorted: &[f64], num: usize, den: usize) -> f64 {
    sorted.get(sorted.len() * num / den).copied().unwrap_or(0.0)
}

fn prefill_mode(b: &mut Bench, cli: &Cli) -> Result<()> {
    let chunk = cli.chunk;
    let prof = cli.prof != 0;
    let reps = cli.reps.unwrap_or(1);
    let suf = list(&cli.suffix)?;
    let ctx = cli.ctx;
    let phase = |b: &mut Bench, name: &str, from: usize, count: usize| -> Result<()> {
        let mut tele = Tele::start();
        let mut comp: BTreeMap<String, f64> = BTreeMap::new();
        b.rt.set_profile(prof)?;
        b.sync()?;
        let a = Instant::now();
        let mut calls = 0_u64;
        let mut x = from;
        while x < from + count {
            let z = (from + count).min(x + chunk);
            let seg = [SegIn {
                slot: 0,
                ids: &b.tok[x..z],
                logits: 1,
            }];
            b.rt.forward(&seg)?;
            b.rt.commit(&[u32::try_from(z - x)?])?;
            b.sync()?;
            if prof {
                b.add_marks(&mut comp)?;
            }
            calls += 1;
            x += chunk;
        }
        let e = Instant::now();
        tele.stop();
        b.rt.set_profile(false)?;
        let dt = secs(a, e);
        let line = b
            .head("prefill")
            .text("phase", name)
            .uint("past", from as u64)
            .uint("tokens", count as u64)
            .uint("chunk", chunk as u64)
            .uint("calls", calls)
            .num("seconds", dt)
            .num("tok_per_s", count as f64 / dt);
        let line = b.engine_fields(line).flag("profiled", prof).flag("fine", cli.fine != 0);
        let line = tele
            .window(line, a, e)
            .raw("component_ms_total", &components(&comp, 1.0, 2));
        b.emit(line)
    };
    for _ in 0..reps {
        b.rt.reset_slot(0)?;
        if ctx > 0 {
            phase(b, "cold", 0, ctx)?;
        }
        for &sfx in &suf {
            b.rt.reset_slot(0)?;
            b.fill(0, 0, ctx, chunk)?;
            phase(b, &format!("warm+{sfx}"), ctx, sfx)?;
        }
    }
    Ok(())
}

fn pmix_mode(b: &mut Bench, cli: &Cli, rows: usize, commit: usize) -> Result<()> {
    let (pctx, chunk) = (cli.pctx, cli.chunk);
    let reps = cli.reps.unwrap_or(3);
    let (n, ps) = (b.n, b.n);
    let suf = list(&cli.suffix)?;
    let mut pos = vec![b.ctx; n];
    for &sfx in &suf {
        for r in 0..reps {
            b.rt.reset_slot(ps)?;
            b.fill(ps, 0, pctx, 2048)?;
            b.sync()?;
            let mut tele = Tele::start();
            let a = Instant::now();
            let mut calls = 0_u64;
            let mut dec_rows = 0_u64;
            let mut x = pctx;
            while x < pctx + sfx {
                let take = (chunk - n * rows).min(pctx + sfx - x);
                let mut segs: Vec<SegIn<'_>> = (0..n)
                    .map(|s| {
                        let at = b.off * s + pos[s];
                        SegIn {
                            slot: s,
                            ids: &b.tok[at..at + rows],
                            logits: rows,
                        }
                    })
                    .collect();
                segs.push(SegIn {
                    slot: ps,
                    ids: &b.tok[x..x + take],
                    logits: 1,
                });
                b.rt.forward(&segs)?;
                let mut kept = vec![u32::try_from(commit)?; n];
                kept.push(u32::try_from(take)?);
                b.rt.commit(&kept)?;
                for p in &mut pos {
                    *p += commit;
                }
                dec_rows += (n * commit) as u64;
                x += take;
                calls += 1;
            }
            b.sync()?;
            let e = Instant::now();
            tele.stop();
            let dt = secs(a, e);
            let line = b
                .head("prefill_4session")
                .uint("decoders", n as u64)
                .uint("rows", rows as u64)
                .uint("commit_rows", commit as u64)
                .uint("ctx", b.ctx as u64)
                .uint("past", pctx as u64)
                .uint("tokens", sfx as u64)
                .uint("chunk", chunk as u64)
                .uint("calls", calls)
                .uint("rep", r as u64)
                .num("seconds", dt)
                .num("tok_per_s", sfx as f64 / dt)
                .num("decoder_tokens_per_s_per_stream", (dec_rows / n as u64) as f64 / dt);
            let line = tele.window(b.engine_fields(line), a, e);
            b.emit(line)?;
        }
    }
    Ok(())
}

fn mixed_mode(b: &mut Bench, cli: &Cli, rows: usize) -> Result<()> {
    let pctx = cli.pctx;
    let (n, ps) = (b.n, b.n);
    b.rt.reset_slot(ps)?;
    b.fill(ps, 0, pctx, 2048)?;
    let pos = vec![b.ctx; n];
    for p in list(&cli.plist)? {
        for together in [true, false] {
            let mut tele = Tele::start();
            let t0c = Instant::now();
            let (mut tw0, mut tw1) = (None, t0c);
            let mut it = 0_u64;
            loop {
                let a = Instant::now();
                if secs(t0c, a) > cli.secs {
                    break;
                }
                let verify: Vec<SegIn<'_>> = (0..n)
                    .map(|s| {
                        let at = b.off * s + pos[s];
                        SegIn {
                            slot: s,
                            ids: &b.tok[at..at + rows],
                            logits: rows,
                        }
                    })
                    .collect();
                let pf = SegIn {
                    slot: ps,
                    ids: &b.tok[pctx..pctx + p],
                    logits: 1,
                };
                if together {
                    let mut all = verify;
                    all.push(pf);
                    b.rt.forward(&all)?;
                    b.rt.commit(&vec![0; all.len()])?;
                } else {
                    b.rt.forward(&verify)?;
                    b.rt.commit(&vec![0; n])?;
                    b.rt.forward(&[pf])?;
                    b.rt.commit(&[0])?;
                }
                b.sync()?;
                let e = Instant::now();
                if secs(t0c, a) > cli.discard {
                    tw0.get_or_insert(a);
                    tw1 = e;
                    it += 1;
                }
            }
            tele.stop();
            let w0 = tw0.unwrap_or(t0c);
            let window = if tw0.is_some() { secs(w0, tw1) } else { 0.0 };
            let line = b
                .head_no_limit("mixed_call")
                .uint("sessions", n as u64)
                .uint("rows", rows as u64)
                .uint("ctx", b.ctx as u64)
                .uint("pctx", pctx as u64)
                .uint("prefill_rows", p as u64)
                .flag("one_call", together)
                .uint("iterations", it)
                .num("window_s", window)
                .num("ms_per_iteration", window / it.max(1) as f64 * 1e3);
            let line = tele.window(b.engine_fields(line), w0, tw1);
            b.emit(line)?;
        }
    }
    Ok(())
}

fn decode_mode(b: &mut Bench, cli: &Cli, rows: usize, commit: usize, prefill_s: f64) -> Result<()> {
    // cells: "full" or ';'-separated groups of ','-separated region names to disable
    let mut cells: Vec<(String, Vec<String>)> = Vec::new();
    if cli.ablate.is_empty() {
        cells.push(("full".to_owned(), Vec::new()));
    }
    for group in cli.ablate.split_terminator(';') {
        let regions = group
            .split(',')
            .filter(|r| *r != "full" && !r.is_empty())
            .map(str::to_owned)
            .collect();
        cells.push((group.to_owned(), regions));
    }
    let n = b.n;
    let mut pos = vec![b.ctx; n];
    if cells.iter().any(|c| !c.1.is_empty()) {
        // the seam names regions of captured graphs only: capture the decode graph with one step that keeps nothing
        b.decode_call(&pos, rows, 0)?;
        b.sync()?;
    }
    for (name, regions) in &cells {
        let refs: Vec<&str> = regions.iter().map(String::as_str).collect();
        b.rt.set_ablation(&refs)?;
        let mut tele = Tele::start();
        let mut step_s: Vec<f64> = Vec::new();
        let mut comp: BTreeMap<String, f64> = BTreeMap::new();
        let mut nprof = 0_u64;
        let t0 = Instant::now();
        let (mut tw0, mut tw1) = (None, t0);
        let mut rows_done = 0_u64;
        let mut profiling = false;
        for it in 0_usize.. {
            let a = Instant::now();
            if secs(t0, a) > cli.secs {
                break;
            }
            let prof = cli.prof_every > 0 && it % cli.prof_every == 0 && secs(t0, a) > cli.discard;
            if prof != profiling {
                b.rt.set_profile(prof)?;
                profiling = prof;
            }
            b.decode_call(&pos, rows, commit)?;
            b.sync()?;
            let e = Instant::now();
            for p in &mut pos {
                *p += commit;
            }
            if secs(t0, a) > cli.discard {
                tw0.get_or_insert(a);
                tw1 = e;
                if !prof {
                    step_s.push(secs(a, e));
                }
                rows_done += (n * rows) as u64;
            }
            if prof {
                b.add_marks(&mut comp)?;
                nprof += 1;
            }
        }
        if profiling {
            b.rt.set_profile(false)?;
        }
        tele.stop();
        step_s.sort_by(f64::total_cmp);
        let mean = step_s.iter().sum::<f64>() / step_s.len().max(1) as f64;
        let w0 = tw0.unwrap_or(t0);
        let window = if tw0.is_some() { secs(w0, tw1) } else { 0.0 };
        let stats = b.rt.stats()?;
        let line = b
            .head("decode_step")
            .uint("sessions", n as u64)
            .uint("ctx", b.ctx as u64)
            .uint("rows", rows as u64)
            .uint("commit_rows", commit as u64)
            .num("prefill_s", prefill_s)
            .uint("steps", step_s.len() as u64)
            .num("window_s", window)
            .num("step_ms_mean", mean * 1e3)
            .num("step_ms_median", percentile(&step_s, 1, 2) * 1e3)
            .num("step_ms_p10", percentile(&step_s, 1, 10) * 1e3)
            .num("step_ms_p90", percentile(&step_s, 9, 10) * 1e3)
            .num("rows_per_s", rows_done as f64 / window.max(1e-9))
            .uint("profiled_steps", nprof)
            .uint("graphs", u64::from(b.graphs))
            .uint("graph_captures", stats.graph_captures)
            .uint("graphs_live", stats.graphs_live)
            .uint("graph_bytes", stats.graph_bytes)
            .uint("graphs_refused", stats.graphs_refused)
            .uint("glue", u64::from(b.glue));
        let line = tele
            .window(line, w0, tw1)
            .raw("component_ms", &components(&comp, nprof.max(1) as f64, 4))
            .text("cell", name);
        b.emit(line)?;
    }
    b.rt.set_ablation(&[])?;
    Ok(())
}

fn main() -> Result<()> {
    let mut cli = Cli::parse();
    let n = cli.sessions;
    let rows = cli.rows;
    let commit = cli.commit.unwrap_or(if rows == 1 { 1 } else { 4 });
    ensure!(
        matches!(cli.mode.as_str(), "decode" | "prefill" | "pmix" | "mixed"),
        "unknown --mode {}",
        cli.mode
    );
    cli.engine.slots = u32::try_from((n + 1).max(4))?;
    let long = matches!(cli.mode.as_str(), "prefill" | "pmix");
    let max_chunk = if long {
        u32::try_from(cli.chunk.max(2048))?
    } else {
        ink_tools::engine::MAX_CHUNK
    };
    // the token stream: a pool of ids, or a golden case's prompt and tokens
    let tok: Vec<i32> = if let Some(pool) = &cli.pool {
        let text = fs::read_to_string(pool).with_context(|| pool.display().to_string())?;
        serde_json::from_str(&text).with_context(|| format!("{}: a flat JSON list of token ids", pool.display()))?
    } else {
        let case = read_cases(&cli.cases)?
            .into_iter()
            .find(|c| c.id == cli.stream)
            .with_context(|| format!("case not found: {}", cli.stream))?;
        case.prompt.into_iter().chain(case.tokens).collect()
    };
    let len = i64::try_from(tok.len())?;
    let (ctx_i, n_i) = (i64::try_from(cli.ctx)?, i64::try_from(n)?);
    let off = if n > 1 {
        20_000.min((len - ctx_i - 4000) / (n_i - 1))
    } else {
        0
    };
    ensure!(
        len >= off * (n_i - 1) + ctx_i + 4000 || cli.mode == "prefill",
        "token stream too short"
    );
    let off = usize::try_from(off.max(0))?;
    let card = Card::read();
    let mut rt = cli.engine.open(cli.pages, max_chunk)?;
    if cli.fine != 0 {
        rt.set_profile_fine(true)?;
    }
    let source = rt.identity();
    let record = cli.record.writer(
        "bench",
        &*rt,
        json!({"engine": cli.engine.json(cli.pages, max_chunk), "mode": cli.mode, "cases": cli.cases,
               "sessions": n, "ctx": cli.ctx, "rows": rows, "commit": commit, "secs": cli.secs,
               "discard": cli.discard, "prof_every": cli.prof_every, "ablate": cli.ablate, "chunk": cli.chunk,
               "reps": cli.reps, "suffix": cli.suffix, "pctx": cli.pctx, "plist": cli.plist, "prof": cli.prof,
               "stream": cli.stream, "pool": cli.pool}),
        "C++ ink_bench of the same build (same argv, same fields)",
    )?;
    let mut b = Bench {
        rt,
        tok,
        card,
        source,
        record,
        n,
        ctx: cli.ctx,
        off,
        glue: cli.engine.glue,
        graphs: cli.engine.graphs,
    };
    if cli.mode == "prefill" {
        prefill_mode(&mut b, &cli)?;
    } else {
        // contexts: N sessions of `ctx` real tokens, session s from token OFF * s
        let t0 = Instant::now();
        for s in 0..n {
            b.rt.reset_slot(s)?;
            let base = b.off * s;
            b.fill(s, base, base + cli.ctx, 2048)?;
        }
        b.sync()?;
        let prefill_s = secs(t0, Instant::now());
        match cli.mode.as_str() {
            "pmix" => {
                ensure!(b.n < cli.engine.slots as usize, "--mode pmix needs a spare slot");
                pmix_mode(&mut b, &cli, rows, commit)?;
            }
            "mixed" => mixed_mode(&mut b, &cli, rows)?,
            _ => decode_mode(&mut b, &cli, rows, commit, prefill_s)?,
        }
    }
    let stats = b.rt.stats()?;
    let done: Value = json!({"done": true, "stats": {"graph_captures": stats.graph_captures,
        "gemm_fallbacks": stats.gemm_fallbacks}});
    b.record.append(&done)?;
    eprintln!("record {}", b.record.path().display());
    Ok(())
}
