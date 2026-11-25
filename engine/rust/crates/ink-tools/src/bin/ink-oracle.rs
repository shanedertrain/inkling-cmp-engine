//! Oracle record set of the runtime (docs/RUST-PLAN.md 7.3, package A1): drives a [`Runtime`] and records what the
//! Rust runtime port is compared against bit for bit. Today the runtime is the C++ engine through the interim C ABI
//! (`CapiRuntime` over `libinkrt_c.so`); the recording code is written over `dyn Runtime`, so the same tool records the
//! Rust engine once it implements the trait. Formats: `engine/native/capi/README.md` "Oracle record set".
//!
//! * `ink-oracle smoke`: slot 0, the first n - 3 ids of `--ids` as prefill (chunks of 2,048), then three one-row
//!   decode calls; one JSON line per logit row and the raw rows. Byte-identical output to `ink_ref` (the same
//!   sequence on the C++ engine class without the ABI) when the bits agree.
//! * `ink-oracle record`: E1 weight sums and arena map, E2 launch traces of captured decode calls, E3 every tap of
//!   three cases, E4 logits of the S0 golden set (per-row FNV-1a 64, 16 full rows, greedy token streams); an
//!   `oracle.jsonl` record (identity, context, one line per file) binds them.
//! * `ink-oracle mixed`, `ink-oracle ablate`: calls outside the record set (mixed prefill + decode calls; graph
//!   ablation per region), one line per logit row, for a direct C++ / Rust comparison.
#![forbid(unsafe_code)]
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand};
use ink_record::{Component, ComponentName, DirtyPolicy, Gpu, Identity, RecordWriter, Source};
use ink_rt_api::{Dump, EngineConfig, Graphs, LOGIT_ROW, Runtime, SegIn, TapType};
use ink_tools::cases::{Case, read_cases};
use ink_tools::e4::{E4Rows, FULL_ROWS};
use ink_tools::engine::open_runtime;
use ink_tools::logits::{FNV_OFFSET, argmax_first as argmax, fnv1a64, row_hash};
use ink_tools::session::{self, CHUNK};
use serde_json::{Value, json};

#[derive(Parser)]
#[command(name = "ink-oracle", version = ink_record::BUILD_LINE, about = "Oracle record set of the runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct EngineArgs {
    /// Model directory.
    #[arg(long, default_value = "/model")]
    model: PathBuf,
    /// cuBLASLt pin file: loaded when it exists, written after init otherwise.
    #[arg(long)]
    gemm_algos: PathBuf,
    /// Sessions with state on the card.
    #[arg(long, default_value_t = 8)]
    slots: u32,
    /// Global KV pages of 1,024 tokens.
    #[arg(long, default_value_t = 256)]
    pages: u32,
    /// Rows per prefill segment (`t_max` = this + 64; 2,048 is the engine default).
    #[arg(long, default_value_t = 2048)]
    max_chunk: u32,
    /// Decode calls as captured CUDA graphs (the launch traces are the graphs'); `false`: eager, and the launch
    /// traces are of armed eager calls (Rust engine only).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    graphs: bool,
    /// Device memory the captured graphs may hold, MiB (Rust engine).
    #[arg(long, default_value_t = ink_rt_api::DEFAULT_GRAPH_MEM >> 20)]
    graph_mem_mib: u64,
}

#[derive(Subcommand)]
enum Command {
    /// Prefill plus three decode calls on slot 0; per-row hashes and raw rows (compare with `ink_ref`).
    Smoke {
        #[command(flatten)]
        engine: EngineArgs,
        /// Token ids, whitespace-separated, at least 4.
        #[arg(long)]
        ids: PathBuf,
        /// Output: one JSON line per logit row.
        #[arg(long)]
        out: PathBuf,
        /// Output: the logit rows as raw fp32 `[rows][201024]`.
        #[arg(long)]
        full: PathBuf,
    },
    /// Calls the record set does not hold, for a direct C++ / Rust comparison: mixed prefill + decode calls (decode
    /// rows, ragged verify rows and prefill chunks in one call, in both orders) beside decode-only calls, and with
    /// `--max-chunk 4096` 4,096-row chunks (5-page rings) with decode rows riding along. One line per logit row.
    Mixed {
        #[command(flatten)]
        engine: EngineArgs,
        /// Output: one JSON line per logit row (`call`, `seg`, `slot`, `row`, `argmax`, `fnv1a64`).
        #[arg(long)]
        out: PathBuf,
    },
    /// Graph ablation (C++ / Rust comparison of the region map): a captured decode call with each region disabled
    /// in turn. One line per logit row (`cell`, `row`, `argmax`, `fnv1a64`).
    Ablate {
        #[command(flatten)]
        engine: EngineArgs,
        /// Output: one JSON line per logit row.
        #[arg(long)]
        out: PathBuf,
    },
    /// Host-side split of decode steps (forward, commit, sync wait), medians in microseconds, one JSON line.
    Hostprof {
        #[command(flatten)]
        engine: EngineArgs,
        /// S0 golden cases (the token stream is `tf-agent-120k`, as ink-bench).
        #[arg(long)]
        cases: PathBuf,
        /// Sessions decoding.
        #[arg(long, default_value_t = 4)]
        sessions: usize,
        /// Context tokens per session.
        #[arg(long, default_value_t = 45_000)]
        ctx: usize,
        /// Rows per session per call.
        #[arg(long, default_value_t = 8)]
        rows: usize,
        /// Rows kept per session per call.
        #[arg(long, default_value_t = 4)]
        commit: usize,
        /// Timed steps.
        #[arg(long, default_value_t = 400)]
        steps: usize,
    },
    /// Write the oracle record set (E1-E4) into a new directory.
    Record {
        #[command(flatten)]
        engine: EngineArgs,
        /// S0 golden cases (`cases.txt` of `engine/native/tests/gold_prep.py`).
        #[arg(long)]
        cases: PathBuf,
        /// Output directory; created, must not exist.
        #[arg(long)]
        out: PathBuf,
        /// Only these golden case ids (comma-separated).
        #[arg(long, default_value = "")]
        only: String,
        /// Context of the long-context launch traces (0 skips them; > 98,304 is the second chunk regime).
        #[arg(long, default_value_t = 100_000)]
        long_ctx: usize,
        /// Prefill rows of the long tap case (> 3,072 wraps the sliding-window rings).
        #[arg(long, default_value_t = 3_500)]
        long_taps: usize,
        /// GPU index, UUID and vBIOS type of the card, for the record context.
        #[arg(long)]
        gpu: Option<u32>,
        /// Card UUID.
        #[arg(long, default_value = "")]
        uuid: String,
        /// vBIOS type (`6D` or `67`).
        #[arg(long, default_value = "")]
        vbios: String,
        /// Power cap in watts.
        #[arg(long)]
        power_cap_w: Option<u32>,
        /// Write a `.dirty.jsonl` record when a component is built from a dirty tree (debugging only).
        #[arg(long)]
        allow_dirty: bool,
    },
}

// ------------------------------------------------------------------------------------------------ helpers

/// Deterministic synthetic token ids (traces and taps whose contents do not matter, only their shapes and bits).
fn synthetic(n: usize, seed: u64) -> Vec<i32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            i32::try_from(x % 200_000).unwrap_or(0)
        })
        .collect()
}

fn create(args: &EngineArgs) -> Result<Box<dyn Runtime>> {
    let config = EngineConfig {
        model_dir: args.model.clone(),
        gemm_algos: args.gemm_algos.clone(),
        max_slots: args.slots,
        num_pages: args.pages,
        max_chunk: args.max_chunk,
        t_max: args.max_chunk + 64,
        graphs: if args.graphs { Graphs::On } else { Graphs::Off },
        graph_mem: args.graph_mem_mib << 20,
        ..EngineConfig::default()
    };
    open_runtime(config)
}

/// Whether `rt` is the Rust engine (config `M4`; the oracle is the C++ engine, `M2`).
fn rust(rt: &dyn Runtime) -> bool {
    rt.identity().implementation == ink_record::Engine::Rust
}

/// Prefill `ids` into a fresh `slot` in chunks of 2,048 rows; returns the logits of the last row when `logits`.
fn prefill(rt: &mut dyn Runtime, slot: usize, ids: &[i32], logits: bool, calls: &mut usize) -> Result<Vec<f32>> {
    let p = session::prefill(rt, slot, ids, CHUNK, logits)?;
    *calls += p.calls;
    Ok(p.last)
}

fn read_ids(path: &Path) -> Result<Vec<i32>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.split_whitespace()
        .map(|t| {
            t.parse::<i32>()
                .with_context(|| format!("bad id {t:?} in {}", path.display()))
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------ smoke

fn smoke(args: &EngineArgs, ids_path: &Path, out_path: &Path, full_path: &Path) -> Result<()> {
    let ids = read_ids(ids_path)?;
    ensure!(ids.len() >= 4, "{} needs at least 4 ids", ids_path.display());
    let mut rt_box = create(args)?;
    let rt: &mut dyn Runtime = &mut *rt_box;
    let mut out = BufWriter::new(File::create(out_path)?);
    let mut full = BufWriter::new(File::create(full_path)?);
    let mut emit = |call: usize, pos: usize, row: &[f32]| -> Result<()> {
        writeln!(
            out,
            "{{\"call\":{call},\"row\":0,\"pos\":{pos},\"argmax\":{},\"fnv1a64\":\"{:016x}\"}}",
            argmax(row),
            row_hash(row)
        )?;
        for x in row {
            full.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    };
    let p = ids.len() - 3;
    let mut calls = 0;
    let last = prefill(rt, 0, &ids[..p], true, &mut calls)?;
    emit(calls - 1, p - 1, &last)?;
    let mut lg = Vec::new();
    for (j, id) in ids.iter().enumerate().skip(p) {
        let n = rt.forward(&[SegIn {
            slot: 0,
            ids: std::slice::from_ref(id),
            logits: 1,
        }])?;
        rt.logits_host(n, &mut lg)?;
        emit(calls, j, &lg)?;
        rt.commit(&[1])?;
        calls += 1;
    }
    out.flush()?;
    full.flush()?;
    eprintln!("smoke: {} ids, {calls} calls, stats {:?}", ids.len(), rt.stats()?);
    Ok(())
}

// ------------------------------------------------------------------------------------------------ mixed

/// One forward call over `segs` (slot, token ids, logit rows); every logit row hashed into `out`; all rows committed.
fn mixed_call(
    rt: &mut dyn Runtime,
    out: &mut impl Write,
    call: usize,
    segs: &[(usize, Vec<i32>, usize)],
) -> Result<()> {
    let ins: Vec<SegIn<'_>> = segs
        .iter()
        .map(|(slot, ids, logits)| SegIn {
            slot: *slot,
            ids,
            logits: *logits,
        })
        .collect();
    let n = rt.forward(&ins)?;
    let mut lg = Vec::new();
    rt.logits_host(n, &mut lg)?;
    let mut r = 0;
    for (k, (slot, _, logits)) in segs.iter().enumerate() {
        for j in 0..*logits {
            let row = &lg[r * LOGIT_ROW..(r + 1) * LOGIT_ROW];
            writeln!(
                out,
                "{{\"call\":{call},\"seg\":{k},\"slot\":{slot},\"row\":{j},\"argmax\":{},\"fnv1a64\":\"{:016x}\"}}",
                argmax(row),
                row_hash(row)
            )?;
            r += 1;
        }
    }
    ensure!(r == n, "call {call}: {n} logit rows, {r} expected");
    let kept: Vec<u32> = segs
        .iter()
        .map(|(_, ids, _)| u32::try_from(ids.len()).unwrap_or(0))
        .collect();
    rt.commit(&kept)?;
    Ok(())
}

fn mixed(args: &EngineArgs, out_path: &Path) -> Result<()> {
    let mut rt_box = create(args)?;
    let rt: &mut dyn Runtime = &mut *rt_box;
    let mut out = BufWriter::new(File::create(out_path)?);
    let mut calls = 0;
    let ids = |n: usize, seed: u64| synthetic(n, seed);
    if args.max_chunk >= 4096 {
        // 4,096-row chunks: two chunks into slot 0 (keys 0..8,192: the 5-page rings wrap), a decode slot beside them
        prefill(rt, 1, &ids(300, 60), false, &mut calls)?;
        mixed_call(rt, &mut out, 0, &[(0, ids(4096, 61), 1), (1, ids(1, 62), 1)])?;
        mixed_call(rt, &mut out, 1, &[(1, ids(3, 63), 3), (0, ids(4096, 64), 4)])?;
        mixed_call(rt, &mut out, 2, &[(0, ids(1, 65), 1), (1, ids(1, 66), 1)])?;
        mixed_call(rt, &mut out, 3, &[(0, ids(8, 67), 8)])?;
    } else {
        for (s, n) in [300usize, 1200, 2500, 40].iter().enumerate() {
            prefill(rt, s, &ids(*n, 70 + s as u64), false, &mut calls)?;
        }
        // decode-only, then mixed calls with decode / ragged verify rows before and after prefill chunks
        let dec4 = [
            (0, ids(1, 80), 1),
            (1, ids(1, 81), 1),
            (2, ids(1, 82), 1),
            (3, ids(1, 83), 1),
        ];
        mixed_call(rt, &mut out, 0, &dec4)?;
        let c1 = [
            (0, ids(1, 84), 1),
            (1, ids(8, 85), 8),
            (4, ids(500, 86), 1),
            (2, ids(3, 87), 3),
        ];
        mixed_call(rt, &mut out, 1, &c1)?;
        mixed_call(
            rt,
            &mut out,
            2,
            &[(5, ids(2048, 88), 2), (3, ids(1, 89), 1), (0, ids(2, 90), 2)],
        )?;
        mixed_call(rt, &mut out, 3, &[(4, ids(700, 91), 1), (5, ids(1300, 92), 1)])?;
        let c4 = [
            (0, ids(1, 93), 1),
            (1, ids(1, 94), 1),
            (2, ids(8, 95), 8),
            (3, ids(1, 96), 1),
            (4, ids(1, 97), 1),
            (5, ids(5, 98), 5),
        ];
        mixed_call(rt, &mut out, 4, &c4)?;
    }
    out.flush()?;
    eprintln!("mixed: max_chunk {}, stats {:?}", args.max_chunk, rt.stats()?);
    Ok(())
}

// ------------------------------------------------------------------------------------------------ ablate

/// The regions of a captured decode graph (the engines' mark names that own nodes; `end` owns none).
const REGIONS: [&str; 12] = [
    "embed", "attn_in", "proj", "qkv_post", "attn", "wo", "glue_mid", "router", "experts", "glue_end", "msgs", "head",
];

/// One decode call over `segs` with every logit row hashed into `out` under `cell`; nothing committed.
fn ablate_call(rt: &mut dyn Runtime, out: &mut impl Write, cell: &str, segs: &[(usize, Vec<i32>)]) -> Result<()> {
    let ins: Vec<SegIn<'_>> = segs
        .iter()
        .map(|(slot, ids)| SegIn {
            slot: *slot,
            ids,
            logits: ids.len(),
        })
        .collect();
    let n = rt.forward(&ins)?;
    let mut lg = Vec::new();
    rt.logits_host(n, &mut lg)?;
    for r in 0..n {
        let row = &lg[r * LOGIT_ROW..(r + 1) * LOGIT_ROW];
        writeln!(
            out,
            "{{\"cell\":\"{cell}\",\"row\":{r},\"argmax\":{},\"fnv1a64\":\"{:016x}\"}}",
            argmax(row),
            row_hash(row)
        )?;
    }
    rt.commit(&vec![0; segs.len()])?;
    Ok(())
}

/// Graph ablation: one captured decode call (4 sessions, ragged 1/8/3/8), then the same call with each region
/// disabled in turn, two regions together, and none again. A disabled node leaves its outputs of the call before, so
/// equal logits in two engines mean the same nodes were disabled (the same region map).
fn ablate(args: &EngineArgs, out_path: &Path) -> Result<()> {
    ensure!(args.graphs, "ablate needs --graphs true");
    let mut rt_box = create(args)?;
    let rt: &mut dyn Runtime = &mut *rt_box;
    let mut out = BufWriter::new(File::create(out_path)?);
    let mut calls = 0;
    ensure!(
        rt.set_ablation(&["head"]).is_err(),
        "ablation accepted before any capture"
    );
    for s in 0..4 {
        prefill(rt, s, &synthetic(300 + 211 * s, 120 + s as u64), false, &mut calls)?;
    }
    let segs: Vec<(usize, Vec<i32>)> = [1usize, 8, 3, 8]
        .iter()
        .enumerate()
        .map(|(s, &w)| (s, synthetic(w, 130 + s as u64)))
        .collect();
    ablate_call(rt, &mut out, "full", &segs)?;
    for r in REGIONS {
        rt.set_ablation(&[r])?;
        ablate_call(rt, &mut out, r, &segs)?;
    }
    rt.set_ablation(&["router", "attn"])?;
    ablate_call(rt, &mut out, "router,attn", &segs)?;
    rt.set_ablation(&[])?;
    ablate_call(rt, &mut out, "full-again", &segs)?;
    ensure!(rt.set_ablation(&["nosuch"]).is_err(), "an unknown region was accepted");
    out.flush()?;
    eprintln!("ablate: stats {:?}", rt.stats()?);
    Ok(())
}

// ------------------------------------------------------------------------------------------------ hostprof

/// Host-side split of decode steps (the `ink-bench` decode loop: forward, commit, sync), for E6's explanation of a
/// step-time difference: per step the host time inside `forward` (metadata, uploads, graph launch or eager
/// launches), inside `commit`, and the wait in `sync`. The GPU is idle while the host is in `forward` before the
/// launch, so a host-time difference shows up one for one in the step time. One JSON line with medians (us).
fn hostprof(
    args: &EngineArgs,
    cases: &Path,
    sessions: usize,
    ctx: usize,
    rows: usize,
    commit: usize,
    steps: usize,
) -> Result<()> {
    let case = read_cases(cases)?
        .into_iter()
        .find(|c| c.id == "tf-agent-120k")
        .context("case tf-agent-120k")?;
    let tok: Vec<i32> = case.prompt.iter().chain(&case.tokens).copied().collect();
    // session s from token off * s, as bench.cpp
    let off = if sessions > 1 {
        20_000.min((tok.len() - ctx - 4000) / (sessions - 1))
    } else {
        0
    };
    let mut rt_box = create(args)?;
    let rt: &mut dyn Runtime = &mut *rt_box;
    for s in 0..sessions {
        session::prefill(rt, s, &tok[off * s..off * s + ctx], CHUNK, false)?;
    }
    rt.sync()?;
    let mut pos = vec![ctx; sessions];
    let (mut fw, mut cm, mut sy, mut tot) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..steps {
        let ids: Vec<&[i32]> = (0..sessions)
            .map(|s| &tok[off * s + pos[s]..off * s + pos[s] + rows])
            .collect();
        let segs: Vec<SegIn<'_>> = ids
            .iter()
            .enumerate()
            .map(|(slot, ids)| SegIn {
                slot,
                ids,
                logits: rows,
            })
            .collect();
        let t0 = std::time::Instant::now();
        rt.forward(&segs)?;
        let t1 = std::time::Instant::now();
        rt.commit(&vec![u32::try_from(commit)?; sessions])?;
        let t2 = std::time::Instant::now();
        rt.sync()?;
        let t3 = std::time::Instant::now();
        for p in &mut pos {
            *p += commit;
        }
        let us = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e6;
        fw.push(us(t0, t1));
        cm.push(us(t1, t2));
        sy.push(us(t2, t3));
        tot.push(us(t0, t3));
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let id = rt.identity();
    println!(
        "{}",
        json!({"engine": format!("{:?}", id.implementation), "version": id.version, "commit": id.commit, "dirty": id.dirty,
               "sessions": sessions, "ctx": ctx, "rows": rows, "commit_rows": commit, "steps": steps,
               "graph_captures": rt.stats()?.graph_captures,
               "forward_us_median": med(&mut fw), "commit_us_median": med(&mut cm), "sync_us_median": med(&mut sy),
               "step_us_median": med(&mut tot)})
    );
    Ok(())
}

// ------------------------------------------------------------------------------------------------ record

/// Files of the record set and the `oracle.jsonl` record that binds them to the identity.
struct Recorder {
    dir: PathBuf,
    record: RecordWriter,
}

impl Recorder {
    fn file(&mut self, name: &str, what: &str, detail: Value) -> Result<PathBuf> {
        self.record
            .append(&json!({"file": name, "what": what, "detail": detail}))?;
        Ok(self.dir.join(name))
    }
}

/// Collected taps of one forward call.
#[derive(Default)]
struct TapLog {
    lines: Vec<String>,
    keep_full: bool,
    full: Vec<u8>,
}

fn tap_type(t: TapType) -> &'static str {
    match t {
        TapType::Bf16 => "bf16",
        TapType::F32 => "f32",
        TapType::I32 => "i32",
    }
}

fn install_taps(rt: &mut dyn Runtime, log: &Arc<Mutex<TapLog>>) -> Result<()> {
    let log = Arc::clone(log);
    rt.set_tap(Some(Box::new(move |name: &str, ty: TapType, shape: &[i64], data: &[u8], _: &mut [u8]| {
        let Ok(mut l) = log.lock() else { return 0 };
        let i = l.lines.len();
        let off = l.full.len();
        let full = if l.keep_full {
            l.full.extend_from_slice(data);
            format!(",\"full_off\":{off}")
        } else {
            String::new()
        };
        let shape: Vec<String> = shape.iter().map(i64::to_string).collect();
        let line = format!(
            "{{\"i\":{i},\"name\":\"{name}\",\"type\":\"{}\",\"shape\":[{}],\"bytes\":{},\"fnv1a64\":\"{:016x}\"{full}}}",
            tap_type(ty),
            shape.join(","),
            data.len(),
            fnv1a64(data, FNV_OFFSET)
        );
        l.lines.push(line);
        0
    })))?;
    Ok(())
}

/// Writes the taps collected so far, prefixed by `call`, and clears them.
fn flush_taps(log: &Arc<Mutex<TapLog>>, call: &str, out: &mut impl Write, full: Option<&mut File>) -> Result<()> {
    let mut l = log.lock().map_err(|_| anyhow::anyhow!("tap log poisoned"))?;
    for line in l.lines.drain(..) {
        writeln!(out, "{{\"call\":\"{call}\",{}", &line[1..])?;
    }
    if let Some(f) = full {
        f.write_all(&l.full)?;
    }
    l.full.clear();
    Ok(())
}

/// E2: one decode call of the given widths on slots 0..widths.len() (contexts already prefilled), its graph traced
/// (graphs on: the call must capture a new graph), or with `eager` the armed eager call's launches.
fn trace(rt: &mut dyn Runtime, rec: &mut Recorder, name: &str, widths: &[usize], seed: u64, eager: bool) -> Result<()> {
    let ids: Vec<Vec<i32>> = widths
        .iter()
        .enumerate()
        .map(|(s, &w)| synthetic(w, seed + s as u64))
        .collect();
    let segs: Vec<SegIn<'_>> = ids
        .iter()
        .enumerate()
        .map(|(slot, ids)| SegIn {
            slot,
            ids,
            logits: ids.len(),
        })
        .collect();
    let before = rt.stats()?.graph_captures;
    if eager {
        rt.set_launch_trace(true)?;
        rt.forward(&segs)?;
        rt.set_launch_trace(false)?;
    } else {
        rt.forward(&segs)?;
        ensure!(
            rt.stats()?.graph_captures == before + 1,
            "{name}: the call did not capture a new graph"
        );
    }
    rt.commit(&vec![0; segs.len()])?;
    let mut contexts = Vec::new();
    for s in 0..widths.len() {
        contexts.push(rt.length(s)?);
    }
    let path = rec.file(
        &format!("e2-trace-{name}.jsonl"),
        "launch-trace",
        json!({"widths": widths, "contexts": contexts}),
    )?;
    rt.dump(Dump::LaunchTrace, &path)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record(
    args: &EngineArgs,
    cases_path: &Path,
    dir: &Path,
    only: &str,
    long_ctx: usize,
    long_taps: usize,
    gpu: Option<Gpu>,
    power_cap_w: Option<u32>,
    allow_dirty: bool,
) -> Result<()> {
    let mut cases = read_cases(cases_path)?;
    if !only.is_empty() {
        let keep: Vec<&str> = only.split(',').collect();
        cases.retain(|c| keep.contains(&c.id.as_str()));
    }
    ensure!(!cases.is_empty(), "no golden case selected");
    fs::create_dir(dir).with_context(|| format!("creating {} (must not exist)", dir.display()))?;
    let mut rt_box = create(args)?;
    let rt: &mut dyn Runtime = &mut *rt_box;
    let (is_rust, eager) = (rust(&*rt), !args.graphs);
    let identity = Identity::new(
        rt.identity(),
        if is_rust { "M4" } else { "M2" },
        [(ComponentName::Tools, Component::Source(Source::this_build()))],
    );
    let context = ink_record::Context {
        argv: std::env::args().collect(),
        config: json!({"model": args.model, "gemm_algos": args.gemm_algos, "slots": args.slots, "pages": args.pages,
                       "graphs": i32::from(args.graphs), "runtime": "Rust", "cases": cases_path, "long_ctx": long_ctx, "long_taps": long_taps}),
        gpu,
        power_cap_w,
        baseline: if is_rust {
            "the C++ oracle record set bench/raw/2026-11-23-rust-R5-forward/oracle/ (native 0.2.0 @ 1c7ac46)".to_owned()
        } else {
            "none: this is the oracle (C++ engine through ink_rt.h)".to_owned()
        },
        label: if is_rust {
            format!("rust-runtime-graphs{}", i32::from(args.graphs))
        } else {
            "oracle-native".to_owned()
        },
    };
    let policy = if allow_dirty {
        DirtyPolicy::Allow
    } else {
        DirtyPolicy::Refuse
    };
    let record = RecordWriter::create(dir, "oracle", identity, context, policy)?;
    let mut rec = Recorder {
        dir: dir.to_owned(),
        record,
    };
    let slots = usize::try_from(args.slots)?;

    // E1
    let p = rec.file("e1-weight-sums.jsonl", "weight-sums", json!({}))?;
    rt.dump(Dump::WeightSums, &p)?;
    let p = rec.file("e1-arena-map.jsonl", "arena-map", json!({}))?;
    rt.dump(Dump::ArenaMap, &p)?;
    eprintln!("E1 done");

    // E2: decode widths 1, 8 and ragged at 4 and 8 sessions, short contexts; then the long regime
    let mut calls = 0;
    for sessions in [4_usize, 8] {
        ensure!(sessions <= slots, "E2 needs {sessions} slots");
        for s in 0..sessions {
            prefill(rt, s, &synthetic(200 + 37 * s, 1000 + s as u64), false, &mut calls)?;
        }
        let ragged: Vec<usize> = [8, 3, 8, 1].iter().cycle().take(sessions).copied().collect();
        trace(rt, &mut rec, &format!("s{sessions}-w1"), &vec![1; sessions], 10, eager)?;
        trace(rt, &mut rec, &format!("s{sessions}-w8"), &vec![8; sessions], 20, eager)?;
        trace(rt, &mut rec, &format!("s{sessions}-ragged"), &ragged, 30, eager)?;
    }
    if long_ctx > 0 {
        prefill(rt, 0, &synthetic(long_ctx, 7), false, &mut calls)?;
        trace(rt, &mut rec, "long-w1", &[1], 40, eager)?;
        trace(rt, &mut rec, "long-w8", &[8], 50, eager)?;
    }
    eprintln!("E2 done");

    // E3: every tap (taps run the per-op reference glue, eager) of three cases
    let log = Arc::new(Mutex::new(TapLog::default()));
    install_taps(rt, &log)?;
    let tf = cases
        .iter()
        .find(|c| c.group == "tf")
        .context("no teacher-forced case for E3 short")?;
    let short: Vec<i32> = tf.prompt.iter().copied().take(160).collect();
    let mut taps = BufWriter::new(File::create(rec.file(
        "e3-taps.jsonl",
        "taps",
        json!({"cases": {"short": format!("{} prompt[:{}] + 1 decode row", tf.id, short.len()),
                         "batch": "4 slots x 64 synthetic rows, then 1 decode row each",
                         "long": format!("{long_taps} synthetic rows + 1 decode row")},
               "full": "e3-taps-short-decode.bin holds the tensors of call short.decode (full_off)"}),
    )?)?);
    let mut full = File::create(dir.join("e3-taps-short-decode.bin"))?;
    let mut c = 0;
    prefill(rt, 0, &short, true, &mut c)?;
    flush_taps(&log, "short.prefill", &mut taps, None)?;
    log.lock().map_err(|_| anyhow::anyhow!("tap log poisoned"))?.keep_full = true;
    rt.forward(&[SegIn {
        slot: 0,
        ids: &tf.tokens[..1],
        logits: 1,
    }])?;
    rt.commit(&[1])?;
    log.lock().map_err(|_| anyhow::anyhow!("tap log poisoned"))?.keep_full = false;
    flush_taps(&log, "short.decode", &mut taps, Some(&mut full))?;
    let batch: Vec<Vec<i32>> = (0..4).map(|s| synthetic(64, 2000 + s)).collect();
    for (s, ids) in batch.iter().enumerate() {
        prefill(rt, s, ids, false, &mut c)?;
        flush_taps(&log, &format!("batch.prefill{s}"), &mut taps, None)?;
    }
    let one: Vec<Vec<i32>> = (0..4).map(|s| synthetic(1, 3000 + s)).collect();
    let segs: Vec<SegIn<'_>> = one
        .iter()
        .enumerate()
        .map(|(slot, ids)| SegIn { slot, ids, logits: 1 })
        .collect();
    rt.forward(&segs)?;
    rt.commit(&[1, 1, 1, 1])?;
    flush_taps(&log, "batch.decode", &mut taps, None)?;
    prefill(rt, 0, &synthetic(long_taps, 4000), false, &mut c)?;
    flush_taps(&log, "long.prefill", &mut taps, None)?;
    rt.forward(&[SegIn {
        slot: 0,
        ids: &synthetic(1, 4001),
        logits: 1,
    }])?;
    rt.commit(&[1])?;
    flush_taps(&log, "long.decode", &mut taps, None)?;
    rt.set_tap(None)?;
    taps.flush()?;
    eprintln!("E3 done");

    // E4: the S0 golden set as ink_gold runs it (teacher-forced one case at a time; greedy 4 sessions together)
    let logits_path = rec.file(
        "e4-logits.jsonl",
        "logits",
        json!({"hash": "fnv1a64 over the 200058 valid fp32 of each row, little-endian",
               "full": format!("e4-logits-full.f32 holds the first {FULL_ROWS} rows (full index)")}),
    )?;
    let mut rows = E4Rows::create(&logits_path, &dir.join("e4-logits-full.f32"))?;
    let mut lg = Vec::new();
    for case in cases.iter().filter(|c| c.group == "tf") {
        let last = prefill(rt, 0, &case.prompt, true, &mut c)?;
        rows.row(&case.id, 0, case.prompt.len() - 1, &last)?;
        for r in 0..case.top.len().saturating_sub(1) {
            let n = rt.forward(&[SegIn {
                slot: 0,
                ids: &case.tokens[r..=r],
                logits: 1,
            }])?;
            rt.logits_host(n, &mut lg)?;
            rows.row(&case.id, r + 1, case.prompt.len() + r, &lg)?;
            rt.commit(&[1])?;
        }
    }
    let mut streams = Vec::new();
    let greedy: Vec<&Case> = cases.iter().filter(|c| c.group == "greedy").collect();
    for group in greedy.chunks(4) {
        let mut toks: Vec<Vec<i32>> = Vec::new();
        for (s, case) in group.iter().enumerate() {
            let last = prefill(rt, s, &case.prompt, true, &mut c)?;
            let a = rows.row(&case.id, 0, case.prompt.len() - 1, &last)?;
            toks.push(vec![i32::try_from(a)?]);
        }
        loop {
            let who: Vec<usize> = (0..group.len())
                .filter(|&s| toks[s].len() < group[s].tokens.len())
                .collect();
            if who.is_empty() {
                break;
            }
            let ids: Vec<i32> = who.iter().map(|&s| *toks[s].last().unwrap_or(&0)).collect();
            let segs: Vec<SegIn<'_>> = who
                .iter()
                .zip(&ids)
                .map(|(&slot, id)| SegIn {
                    slot,
                    ids: std::slice::from_ref(id),
                    logits: 1,
                })
                .collect();
            let n = rt.forward(&segs)?;
            rt.logits_host(n, &mut lg)?;
            for (k, &s) in who.iter().enumerate() {
                let row = &lg[k * LOGIT_ROW..(k + 1) * LOGIT_ROW];
                let pos = group[s].prompt.len() + toks[s].len() - 1;
                let a = rows.row(&group[s].id, toks[s].len(), pos, row)?;
                toks[s].push(i32::try_from(a)?);
            }
            rt.commit(&vec![1; segs.len()])?;
        }
        for (s, case) in group.iter().enumerate() {
            streams.push(json!({"case": case.id, "tokens": toks[s]}));
        }
    }
    let nfull = rows.finish()?;
    let p = rec.file(
        "e4-greedy-tokens.json",
        "greedy token streams",
        json!({"sessions_together": 4}),
    )?;
    fs::write(&p, serde_json::to_vec(&streams)?)?;
    let stats = rt.stats()?;
    rec.record
        .append(&json!({"done": true, "stats": {"arena_used": stats.arena_used,
        "weight_bytes": stats.weight_bytes, "graph_captures": stats.graph_captures}, "e4_full_rows": nfull}))?;
    eprintln!("E4 done; record {}", rec.record.path().display());
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Smoke { engine, ids, out, full } => smoke(&engine, &ids, &out, &full),
        Command::Mixed { engine, out } => mixed(&engine, &out),
        Command::Ablate { engine, out } => ablate(&engine, &out),
        Command::Hostprof {
            engine,
            cases,
            sessions,
            ctx,
            rows,
            commit,
            steps,
        } => hostprof(&engine, &cases, sessions, ctx, rows, commit, steps),
        Command::Record {
            engine,
            cases,
            out,
            only,
            long_ctx,
            long_taps,
            gpu,
            uuid,
            vbios,
            power_cap_w,
            allow_dirty,
        } => {
            let gpu = match gpu {
                Some(index) => Some(Gpu { index, uuid, vbios }),
                None if uuid.is_empty() && vbios.is_empty() => None,
                None => bail!("--uuid and --vbios need --gpu"),
            };
            record(
                &engine,
                &cases,
                &out,
                &only,
                long_ctx,
                long_taps,
                gpu,
                power_cap_w,
                allow_dirty,
            )
        }
    }
}
