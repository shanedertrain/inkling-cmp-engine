//! Golden-set runner (package T1; port of `engine/native/tools/gold.cpp`, deleted at cut-over U3a).
//!
//! ```text
//! ink-gold --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir DIR --model /model --cases cases.txt \
//!          [--only id,id] [--skip tf|greedy] [--tf-batch 1] [--greedy-sessions 4] > gold.jsonl
//! ```
//!
//! `cases.txt` is written by `engine/native/tests/gold_prep.py` from a golden capture. Teacher-forced cases: chunked
//! prefill (2,048 rows per call, one session), position 0 = last prompt row, then config B's tokens fed back one row
//! per call (decode shape); with `--tf-batch N` the decode rows of N cases share each call. Per position: fp64
//! log-softmax at the golden top ids, argmax, top 20. Greedy cases: plain greedy decoding without speculation,
//! sessions decoding together, top 20 per emitted token. One JSON line per case on stdout, in the C++ tool's format;
//! `engine/native/tests/gold_post.py` turns them into `compare.py`'s format.
//!
//! Beside it, `--record-dir` receives the record `gold.jsonl` (identity of this tool and of the runtime library) and,
//! when the run is the oracle's shape (`--tf-batch 1`, greedy sessions 4), `e4-logits.jsonl`, `e4-logits-full.f32`
//! and `e4-greedy-tokens.json`: the E4 files of the oracle record set, for bit comparison with `ink-oracle record`.
#![forbid(unsafe_code)]
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use ink_rt_api::{LOGIT_ROW, Runtime, SegIn};
use ink_tools::cases::{Case, read_cases, select};
use ink_tools::e4::E4Rows;
use ink_tools::engine::{EngineArgs, MAX_CHUNK, RecordArgs, identity_fields};
use ink_tools::logits::{LogSoftmax, argmax_first, position_json, top_json};
use ink_tools::session::{prefill, row};
use serde_json::{Value, json};

#[derive(Parser)]
#[command(name = "ink-gold", version = ink_record::BUILD_LINE, about = "Golden-set runner")]
struct Cli {
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    record: RecordArgs,
    /// Global KV pages of 1,024 tokens.
    #[arg(long, default_value_t = 400)]
    pages: u32,
    /// Golden cases (`cases.txt` of `gold_prep.py`).
    #[arg(long, default_value = "/gin/cases.txt")]
    cases: PathBuf,
    /// Only these case ids (comma-separated).
    #[arg(long, default_value = "")]
    only: String,
    /// Skip groups whose name occurs in this text (`tf`, `greedy`).
    #[arg(long, default_value = "")]
    skip: String,
    /// Teacher-forced cases decoded together per call.
    #[arg(long, default_value_t = 1)]
    tf_batch: usize,
    /// Greedy sessions decoding together.
    #[arg(long, default_value_t = 4)]
    greedy_sessions: usize,
}

/// Header fields of an output line: the runtime library's identity, as the C++ tool prints it.
struct Header {
    identity: String,
}

impl Header {
    fn line(&self, id: &str, rest: &str) -> String {
        format!("{{{},\"id\":\"{id}\",{rest}}}", self.identity)
    }
}

/// E4 rows of the run, kept only when the order of the rows is the oracle's.
struct Oracle {
    rows: E4Rows,
    streams: Vec<Value>,
}

struct Run<'a> {
    rt: &'a mut dyn Runtime,
    header: Header,
    softmax: LogSoftmax,
    out: BufWriter<std::io::Stdout>,
    oracle: Option<Oracle>,
}

impl Run<'_> {
    fn emit(&mut self, line: &str) -> Result<()> {
        writeln!(self.out, "{line}")?;
        self.out.flush()?;
        Ok(())
    }

    fn e4(&mut self, id: &str, k: usize, pos: usize, row: &[f32]) -> Result<()> {
        if let Some(o) = &mut self.oracle {
            o.rows.row(id, k, pos, row)?;
        }
        Ok(())
    }

    fn position(&mut self, row: &[f32], gold_ids: &[i32]) -> String {
        let lp = self.softmax.compute(row);
        position_json(row, lp, gold_ids)
    }

    /// Teacher-forced cases `batch` (slot j = case j).
    fn teacher_forced(&mut self, batch: &[&Case]) -> Result<()> {
        let t0 = Instant::now();
        let mut pos: Vec<Vec<String>> = vec![Vec::new(); batch.len()];
        for (j, case) in batch.iter().enumerate() {
            let last = prefill(self.rt, j, &case.prompt, 2048, true)?.last;
            pos[j].push(self.position(&last, &case.top[0]));
            self.e4(&case.id, 0, case.prompt.len() - 1, &last)?;
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let nforced = batch.iter().map(|c| c.top.len().saturating_sub(1)).max().unwrap_or(0);
        let mut lg = Vec::new();
        for r in 0..nforced {
            let who: Vec<usize> = (0..batch.len())
                .filter(|&j| r < batch[j].top.len().saturating_sub(1))
                .collect();
            let segs: Vec<SegIn<'_>> = who
                .iter()
                .map(|&j| SegIn {
                    slot: j,
                    ids: &batch[j].tokens[r..=r],
                    logits: 1,
                })
                .collect();
            let n = self.rt.forward(&segs)?;
            self.rt.logits_host(n, &mut lg)?;
            for (k, &j) in who.iter().enumerate() {
                let line = self.position(row(&lg, k), &batch[j].top[r + 1]);
                pos[j].push(line);
                self.e4(&batch[j].id, r + 1, batch[j].prompt.len() + r, row(&lg, k))?;
            }
            self.rt.commit(&vec![1; segs.len()])?;
        }
        let decode_s = t1.elapsed().as_secs_f64();
        for (j, case) in batch.iter().enumerate() {
            let rest = format!(
                "\"group\":\"tf\",\"tf_batch\":{},\"prefill_s\":{prefill_s:.3},\"decode_s\":{decode_s:.3},\"positions\":[{}]",
                batch.len(),
                pos[j].join(",")
            );
            self.emit(&self.header.line(&case.id, &rest))?;
            eprintln!(
                "tf {}: {} positions, prefill {prefill_s:.2} s, decode {decode_s:.2} s",
                case.id,
                pos[j].len()
            );
        }
        Ok(())
    }

    /// Greedy cases `batch` (slot j = case j), decoded together without speculation.
    fn greedy(&mut self, batch: &[&Case]) -> Result<()> {
        let t0 = Instant::now();
        let mut toks: Vec<Vec<i32>> = Vec::new();
        let mut tops: Vec<Vec<String>> = Vec::new();
        for (j, case) in batch.iter().enumerate() {
            let last = prefill(self.rt, j, &case.prompt, 2048, true)?.last;
            let lp = self.softmax.compute(&last);
            let a = argmax_first(&last);
            tops.push(vec![top_json(lp, 20)]);
            toks.push(vec![i32::try_from(a)?]);
            self.e4(&case.id, 0, case.prompt.len() - 1, &last)?;
        }
        let mut lg = Vec::new();
        loop {
            let who: Vec<usize> = (0..batch.len())
                .filter(|&j| toks[j].len() < batch[j].tokens.len())
                .collect();
            if who.is_empty() {
                break;
            }
            let ids: Vec<i32> = who.iter().map(|&j| *toks[j].last().unwrap_or(&0)).collect();
            let segs: Vec<SegIn<'_>> = who
                .iter()
                .zip(&ids)
                .map(|(&slot, id)| SegIn {
                    slot,
                    ids: std::slice::from_ref(id),
                    logits: 1,
                })
                .collect();
            let n = self.rt.forward(&segs)?;
            self.rt.logits_host(n, &mut lg)?;
            for (k, &j) in who.iter().enumerate() {
                let r = row(&lg, k);
                let lp = self.softmax.compute(r);
                tops[j].push(top_json(lp, 20));
                let a = argmax_first(r);
                let pos = batch[j].prompt.len() + toks[j].len() - 1;
                self.e4(&batch[j].id, toks[j].len(), pos, r)?;
                toks[j].push(i32::try_from(a)?);
            }
            self.rt.commit(&vec![1; segs.len()])?;
        }
        let dt = t0.elapsed().as_secs_f64();
        for (j, case) in batch.iter().enumerate() {
            let ids: Vec<String> = toks[j].iter().map(i32::to_string).collect();
            let rest = format!(
                "\"group\":\"greedy\",\"sessions_together\":{},\"seconds\":{dt:.2},\"tokens\":[{}],\"top\":[{}]",
                batch.len(),
                ids.join(","),
                tops[j].join(",")
            );
            self.emit(&self.header.line(&case.id, &rest))?;
            eprintln!("greedy {}: {} tokens, {dt:.1} s", case.id, toks[j].len());
            if let Some(o) = &mut self.oracle {
                o.streams.push(json!({"case": case.id, "tokens": toks[j]}));
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(
        cli.tf_batch > 0 && cli.greedy_sessions > 0,
        "--tf-batch and --greedy-sessions must be positive"
    );
    let all = read_cases(&cli.cases)?;
    ensure!(
        all.iter()
            .all(|c| c.group != "tf" || c.top.len() >= 2 || cli.skip.contains("tf")),
        "a teacher-forced case has fewer than 2 positions"
    );
    let cases = select(all, &cli.only);
    let tf: Vec<&Case> = cases
        .iter()
        .filter(|c| c.group == "tf" && !cli.skip.contains("tf"))
        .collect();
    let gr: Vec<&Case> = cases
        .iter()
        .filter(|c| c.group == "greedy" && !cli.skip.contains("greedy"))
        .collect();
    let slots = usize::try_from(cli.engine.slots)?;
    if cli.tf_batch > slots || cli.greedy_sessions > slots {
        bail!("--tf-batch and --greedy-sessions must not exceed --slots {slots}");
    }
    let mut rt = cli.engine.open(cli.pages, MAX_CHUNK)?;
    let source = rt.identity();
    let mut record = cli.record.writer(
        "gold",
        &*rt,
        json!({"engine": cli.engine.json(cli.pages, MAX_CHUNK), "cases": cli.cases, "only": cli.only, "skip": cli.skip,
               "tf_batch": cli.tf_batch, "greedy_sessions": cli.greedy_sessions}),
        "none (this record binds the stdout of the run; compare it with the oracle E4 or with ink_gold)",
    )?;
    // The E4 files only mean something in the oracle's row order: one teacher-forced case at a time, greedy in 4s.
    let oracle_shape = cli.tf_batch == 1 && cli.greedy_sessions == 4;
    let oracle = if oracle_shape {
        Some(Oracle {
            rows: E4Rows::create(
                &cli.record.file("e4-logits.jsonl"),
                &cli.record.file("e4-logits-full.f32"),
            )?,
            streams: Vec::new(),
        })
    } else {
        None
    };
    let mut run = Run {
        rt: &mut *rt,
        header: Header {
            identity: identity_fields(&source),
        },
        softmax: LogSoftmax::default(),
        out: BufWriter::new(std::io::stdout()),
        oracle,
    };
    for batch in tf.chunks(cli.tf_batch) {
        run.teacher_forced(batch)?;
    }
    for batch in gr.chunks(cli.greedy_sessions) {
        run.greedy(batch)?;
    }
    let oracle = run.oracle.take();
    let mut files = Vec::new();
    if let Some(o) = oracle {
        let full_rows = o.rows.finish()?;
        let tokens = cli.record.file("e4-greedy-tokens.json");
        fs::write(&tokens, serde_json::to_vec(&o.streams)?).with_context(|| tokens.display().to_string())?;
        files.extend(["e4-logits.jsonl", "e4-logits-full.f32", "e4-greedy-tokens.json"]);
        eprintln!("e4: {full_rows} full rows");
    }
    let stats = rt.stats()?;
    record.append(
        &json!({"done": true, "tf_cases": tf.len(), "greedy_cases": gr.len(), "files": files,
        "row_floats": LOGIT_ROW,
        "stats": {"arena_used": stats.arena_used, "weight_bytes": stats.weight_bytes,
                  "graph_captures": stats.graph_captures, "gemm_fallbacks": stats.gemm_fallbacks}}),
    )?;
    eprintln!("done; record {}", record.path().display());
    Ok(())
}
