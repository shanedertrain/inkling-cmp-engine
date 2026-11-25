//! Gate E1 of the Rust loader (docs/RUST-PLAN.md 7.3, package R4): loads the checkpoint into the arena with
//! `ink-model`, takes the memory plan of `Engine::alloc`, and writes the two E1 dumps in the C++ record formats:
//! `e1-weight-sums.jsonl` (FNV-1a 64 of every weight region read back from the device) and `e1-arena-map.jsonl`
//! (every take in order). With `--oracle` the weight sums are compared line by line with the C++ oracle's file and
//! the verdict goes to `gate.json`. Load time and peak host memory (`VmHWM`) are recorded as side facts.
//!
//! `ink-load --model /model --out DIR [--oracle e1-weight-sums.jsonl] [--gpu N --uuid U --vbios T --power-cap-w W]`
#![forbid(unsafe_code)]
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, bail, ensure};
use clap::Parser;
use ink_cuda::{Arena, Context};
use ink_model::dump::{arena_map_text, weight_sums, weight_sums_text};
use ink_model::{Activations, GateTailW, MemConfig, Weights};
use ink_record::{Component, ComponentName, DirtyPolicy, Gpu, Identity, RecordWriter, Source};
use serde_json::json;

#[derive(Parser)]
#[command(name = "ink-load", version = ink_record::BUILD_LINE, about = "E1: Rust loader weight sums and arena map")]
struct Cli {
    /// Model directory.
    #[arg(long, default_value = "/model")]
    model: PathBuf,
    /// Output directory; created, must not exist.
    #[arg(long)]
    out: PathBuf,
    /// The C++ oracle's `e1-weight-sums.jsonl`, compared line by line.
    #[arg(long)]
    oracle: Option<PathBuf>,
    /// Sessions with state on the card.
    #[arg(long, default_value_t = 8)]
    slots: usize,
    /// Global KV pages of 1,024 tokens.
    #[arg(long, default_value_t = 256)]
    pages: usize,
    /// Rows per forward call.
    #[arg(long, default_value_t = 2112)]
    t_max: usize,
    /// Rows per prefill segment (sets the sliding-window ring).
    #[arg(long, default_value_t = 2048)]
    max_chunk: usize,
    /// Logit rows per call.
    #[arg(long, default_value_t = 64)]
    logit_rows: usize,
    /// Expert repack threads (the C++ `load_threads`).
    #[arg(long, default_value_t = 16)]
    threads: usize,
    /// GPU index on the host, for the record context.
    #[arg(long)]
    gpu: Option<u32>,
    /// Card UUID: checked against the device.
    #[arg(long, default_value = "")]
    uuid: String,
    /// vBIOS type (`6D` or `67`).
    #[arg(long, default_value = "")]
    vbios: String,
    /// Power cap in watts.
    #[arg(long)]
    power_cap_w: Option<u32>,
    /// Write a `.dirty.jsonl` record when built from a dirty tree (debugging only).
    #[arg(long)]
    allow_dirty: bool,
}

/// `VmHWM` and `VmRSS` of this process in kB (`/proc/self/status`).
fn host_memory_kb() -> Result<(u64, u64)> {
    let s = fs::read_to_string("/proc/self/status")?;
    let field = |k: &str| -> Result<u64> {
        let line = s
            .lines()
            .find(|l| l.starts_with(k))
            .with_context(|| format!("{k} not in /proc/self/status"))?;
        Ok(line[k.len()..].trim().trim_end_matches("kB").trim().parse()?)
    };
    Ok((field("VmHWM:")?, field("VmRSS:")?))
}

/// Line-by-line comparison of two `weight-sums` files: (identical rows, differing rows, first differences).
fn compare(ours: &str, oracle: &Path) -> Result<serde_json::Value> {
    let theirs = fs::read_to_string(oracle).with_context(|| format!("reading {}", oracle.display()))?;
    let (a, b): (Vec<&str>, Vec<&str>) = (ours.lines().collect(), theirs.lines().collect());
    ensure!(!a.is_empty() && !b.is_empty(), "empty weight-sums file");
    let header_equal = a[0] == b[0];
    let (ra, rb) = (&a[1..], &b[1..]);
    let mut identical = 0usize;
    let mut differing = Vec::new();
    for i in 0..ra.len().max(rb.len()) {
        match (ra.get(i), rb.get(i)) {
            (Some(x), Some(y)) if x == y => identical += 1,
            (x, y) => differing.push(json!({"row": i, "rust": x, "oracle": y})),
        }
    }
    let n_diff = differing.len();
    differing.truncate(20);
    Ok(json!({
        "header_equal": header_equal, "header_rust": a[0], "header_oracle": b[0],
        "rows_rust": ra.len(), "rows_oracle": rb.len(), "identical": identical, "differing": n_diff,
        "first_differences": differing, "pass": header_equal && n_diff == 0 && ra.len() == rb.len(),
    }))
}

fn main() -> Result<()> {
    let a = Cli::parse();
    let gpu = match a.gpu {
        Some(index) => Some(Gpu {
            index,
            uuid: a.uuid.clone(),
            vbios: a.vbios.clone(),
        }),
        None if a.uuid.is_empty() && a.vbios.is_empty() => None,
        None => bail!("--uuid and --vbios need --gpu"),
    };
    let cfg = MemConfig {
        max_slots: a.slots,
        num_pages: a.pages,
        t_max: a.t_max,
        max_chunk: a.max_chunk,
        logit_rows_max: a.logit_rows,
        kv_fp8: false,
    };
    fs::create_dir(&a.out).with_context(|| format!("creating {} (must not exist)", a.out.display()))?;
    let identity = Identity::new(
        Source::this_build(),
        "M4",
        [(ComponentName::Tools, Component::Source(Source::this_build()))],
    );
    let context = ink_record::Context {
        argv: std::env::args().collect(),
        config: json!({"model": a.model, "slots": a.slots, "pages": a.pages, "t_max": a.t_max,
                       "logit_rows_max": a.logit_rows, "load_threads": a.threads,
                       "oracle": a.oracle}),
        gpu,
        power_cap_w: a.power_cap_w,
        baseline:
            "C++ native 0.2.0 @ 66fe353 weight sums (bench/raw/2026-11-23-a1-oracle-native/oracle/e1-weight-sums.jsonl)"
                .to_owned(),
        label: "rust-R4-loader".to_owned(),
    };
    let policy = if a.allow_dirty {
        DirtyPolicy::Allow
    } else {
        DirtyPolicy::Refuse
    };
    let mut rec = RecordWriter::create(&a.out, "r4", identity, context, policy)?;

    let ctx = Context::new(0)?;
    let dev = ctx.device()?;
    eprintln!("device {} {} free/total {:?}", dev.name, dev.uuid, dev.mem_free_total);
    ensure!(
        a.uuid.is_empty() || dev.uuid == a.uuid,
        "device {} is not --uuid {}",
        dev.uuid,
        a.uuid
    );
    let st = ctx.new_stream()?;
    let mut arena = Arena::new(&ctx, &st, cfg.arena_bytes())?;
    eprintln!("[r4] arena {} bytes", cfg.arena_bytes());
    let t0 = Instant::now();
    let w = Weights::load(&mut arena, &st, &a.model, a.threads)?;
    st.synchronize()?;
    let load_s = t0.elapsed().as_secs_f64();
    let (hwm_kb, rss_kb) = host_memory_kb()?;
    eprintln!("[r4] weights {} bytes in {load_s:.2} s, VmHWM {hwm_kb} kB", w.bytes);
    let acts = Activations::alloc(&mut arena, &cfg)?;
    let looped: Vec<bool> = w.layers.iter().map(|l| l.controllers.is_some()).collect();
    let _tc = GateTailW::alloc(&mut arena, &looped)?; // the C++ arena order (G7 weights after Engine::alloc)
    let end = acts.kpool.offset();
    eprintln!("[r4] arena used {} of {}", arena.plan().used(), arena.plan().cap());

    let t1 = Instant::now();
    let sums = weight_sums(&arena, &st, &w, end)?;
    let sums_s = t1.elapsed().as_secs_f64();
    let sums_text = weight_sums_text(&sums, end);
    fs::write(a.out.join("e1-weight-sums.jsonl"), &sums_text)?;
    rec.append(&json!({"file": "e1-weight-sums.jsonl", "what": "weight-sums", "detail": {"regions": sums.len(), "bytes": end}}))?;
    fs::write(a.out.join("e1-arena-map.jsonl"), arena_map_text(arena.plan()))?;
    rec.append(&json!({"file": "e1-arena-map.jsonl", "what": "arena-map",
                       "detail": {"takes": arena.plan().regions().len(), "cap": arena.plan().cap(), "used": arena.plan().used()}}))?;
    let side = json!({"what": "side facts (not a gate)", "load_s": load_s, "weight_sums_s": sums_s,
                      "vm_hwm_kb_after_load": hwm_kb, "vm_rss_kb_after_load": rss_kb, "weights_bytes": w.bytes,
                      "mem_free_total_before": dev.mem_free_total});
    rec.append(&side)?;
    if let Some(o) = &a.oracle {
        let g = compare(&sums_text, o)?;
        fs::write(
            a.out.join("gate.json"),
            serde_json::to_string_pretty(&json!({"gate": "E1 weight-sums", "oracle": o, "result": g}))? + "\n",
        )?;
        rec.append(&json!({"file": "gate.json", "what": "E1 weight-sums vs oracle", "detail": g}))?;
        eprintln!(
            "[r4] E1 weight-sums: identical {} differing {} pass {}",
            g["identical"], g["differing"], g["pass"]
        );
    }
    Ok(())
}
