//! Fast target-tap extractor for DFlash2-Global distillation and evaluation.
//!
//! Runs `ink_runtime::Engine` in pure prefill mode (`logits: 0`, ~6-7K tok/s per GPU),
//! gathers the 4 native target feature taps `[5, 23, 41, 48]` (`4 x 2304` BF16 = 18,432 B/tok)
//! via `SpecRuntime::gather_feats`, and streams contiguous binary shards to disk:
//!   - `taps.bf16`: raw BF16 bytes `[total_rows, 9216]`
//!   - `tokens.i32`: little-endian `i32` `[total_rows]`
//!   - `loss_mask.u8`: `u8` `[total_rows]` (1 = on-policy assistant completion token, 0 = prompt/tool)
//!   - `index.jsonl`: `{"id", "offset", "len", "prompt_len", "completion_tokens"}` per sequence
//!   - optional `verify_argmax.i32` when `--verify-logits-rows N` > 0 (exact engine argmax on the
//!     first `N` rows of sequence 0, to prove `tap[3] @ unembed[:200058].T` matches `>= 99.9%`).
#![deny(unsafe_code)] // allowed only at the bounds-contract call `gather_feats`

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use ink_rt_api::{FEAT_ROW_BYTES, Runtime, SegIn, SpecRuntime, VOCAB};
use ink_tools::engine::{EngineArgs, MAX_CHUNK};
use serde::Deserialize;
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "ink-taps",
    version = ink_record::BUILD_LINE,
    about = "Fast target-tap ([5,23,41,48] BF16) extractor for DFlash2-Global"
)]
struct Cli {
    #[command(flatten)]
    engine: EngineArgs,
    /// Input JSONL file with lines `{"id": "...", "tokens": [int, ...], "loss_mask": [0|1, ...], "prompt_len": int}`.
    #[arg(long)]
    input: PathBuf,
    /// Output directory (`taps.bf16`, `tokens.i32`, `loss_mask.u8`, `index.jsonl`).
    #[arg(long)]
    out_dir: PathBuf,
    /// Global KV pages of 1,024 tokens (default 140 -> 143K context for 1 slot).
    #[arg(long, default_value_t = 140)]
    pages: u32,
    /// Prefill chunk size (<= 2048).
    #[arg(long, default_value_t = 2048)]
    chunk: usize,
    /// Optional cap on sequences extracted (0 = all).
    #[arg(long, default_value_t = 0)]
    max_seqs: usize,
    /// Optional truncation cap per sequence (0 = no truncation).
    #[arg(long, default_value_t = 0)]
    max_len: usize,
    /// Shard modulo: only process input sequence indices `idx % shard_count == shard_idx`.
    #[arg(long, default_value_t = 0)]
    shard_idx: usize,
    /// Total number of parallel shards (1 = process all sequences).
    #[arg(long, default_value_t = 1)]
    shard_count: usize,
    /// If > 0, compute exact engine argmax on the first N rows of sequence 0 and write `verify_argmax.i32`.
    #[arg(long, default_value_t = 0)]
    verify_logits_rows: usize,
}

#[derive(Deserialize)]
struct SeqRow {
    id: String,
    tokens: Vec<i32>,
    #[serde(default)]
    loss_mask: Option<Vec<u8>>,
    #[serde(default)]
    prompt_len: usize,
}

fn spec_of(rt: &mut dyn Runtime) -> Result<&mut dyn SpecRuntime> {
    rt.spec().context("runtime lacks SpecRuntime")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(
        cli.chunk > 0 && cli.chunk <= MAX_CHUNK as usize,
        "--chunk must be in 1..={MAX_CHUNK}"
    );
    ensure!(
        cli.shard_count >= 1 && cli.shard_idx < cli.shard_count,
        "--shard-idx {} must be < --shard-count {}",
        cli.shard_idx,
        cli.shard_count
    );
    fs::create_dir_all(&cli.out_dir).with_context(|| format!("mkdir {}", cli.out_dir.display()))?;

    let mut engine_args = cli.engine.clone();
    engine_args.slots = engine_args.slots.max(1);
    let mut rt = engine_args.open(cli.pages, cli.chunk as u32)?;
    let max_ctx = (cli.pages as usize) * 1024;

    let (iota_dev, feats_dev) = {
        let sp = spec_of(&mut *rt)?;
        let iota = sp.alloc(cli.chunk * 4)?;
        let ramp: Vec<i32> = (0..cli.chunk)
            .map(|i| i32::try_from(i).expect("chunk fits i32"))
            .collect();
        sp.upload(iota, &ramp)?;
        let feats = sp.alloc(cli.chunk * FEAT_ROW_BYTES)?;
        (iota, feats)
    };

    let in_file = File::open(&cli.input).with_context(|| cli.input.display().to_string())?;
    let reader = BufReader::new(in_file);

    let mut taps_out = BufWriter::with_capacity(8 << 20, File::create(cli.out_dir.join("taps.bf16"))?);
    let mut toks_out = BufWriter::with_capacity(1 << 20, File::create(cli.out_dir.join("tokens.i32"))?);
    let mut mask_out = BufWriter::with_capacity(1 << 20, File::create(cli.out_dir.join("loss_mask.u8"))?);
    let mut idx_out = BufWriter::new(File::create(cli.out_dir.join("index.jsonl"))?);

    let mut host_i32 = vec![0i32; (cli.chunk * FEAT_ROW_BYTES) / 4];
    let mut total_rows: u64 = 0;
    let mut seq_count: usize = 0;
    let mut input_seq_idx: usize = 0;
    let t0 = Instant::now();

    for (line_no, line_res) in reader.lines().enumerate() {
        if cli.max_seqs > 0 && seq_count >= cli.max_seqs {
            break;
        }
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        let cur_idx = input_seq_idx;
        input_seq_idx += 1;
        if cur_idx % cli.shard_count != cli.shard_idx {
            continue;
        }

        let mut row: SeqRow = serde_json::from_str(&line).with_context(|| format!("parse line {}", line_no + 1))?;
        if cli.max_len > 0 && row.tokens.len() > cli.max_len {
            row.tokens.truncate(cli.max_len);
            if let Some(ref mut lm) = row.loss_mask {
                lm.truncate(cli.max_len);
            }
        }
        if row.tokens.len() < 16 {
            continue;
        }
        ensure!(
            row.tokens.len() <= max_ctx,
            "seq {} len {} exceeds max_ctx {max_ctx} (increase --pages)",
            row.id,
            row.tokens.len()
        );
        ensure!(
            row.tokens.iter().all(|&t| t >= 0 && (t as usize) < VOCAB),
            "seq {} contains out-of-vocab token id",
            row.id
        );

        let seq_len = row.tokens.len();
        let prompt_len = row.prompt_len.min(seq_len);
        let mask_vec: Vec<u8> = match row.loss_mask {
            Some(m) => {
                ensure!(
                    m.len() == seq_len,
                    "seq {} loss_mask len {} != tokens len {seq_len}",
                    row.id,
                    m.len()
                );
                m
            }
            None => (0..seq_len).map(|i| u8::from(i >= prompt_len)).collect(),
        };
        let comp_toks: usize = mask_vec.iter().map(|&b| usize::from(b != 0)).sum();

        // Optional exact argmax verification pass on first N rows of sequence 0
        if seq_count == 0 && cli.verify_logits_rows > 0 {
            let n_ver = cli.verify_logits_rows.min(seq_len);
            rt.reset_slot(0)?;
            let mut exact_argmax: Vec<i32> = Vec::with_capacity(n_ver);
            let mut chunk_am: Vec<i32> = Vec::new();
            for piece in row.tokens[..n_ver].chunks(64) {
                let m = piece.len();
                let _ = rt.forward(&[SegIn {
                    slot: 0,
                    ids: piece,
                    logits: m,
                }])?;
                rt.argmax_host(m, &mut chunk_am)?;
                exact_argmax.extend_from_slice(&chunk_am);
                rt.commit(&[u32::try_from(m)?])?;
            }
            let am_bytes: &[u8] = bytemuck::cast_slice(&exact_argmax);
            fs::write(cli.out_dir.join("verify_argmax.i32"), am_bytes)?;
            eprintln!("[ink-taps] Wrote verify_argmax.i32 ({n_ver} rows from seq 0)");
        }

        let offset = total_rows;
        rt.reset_slot(0)?;

        for piece in row.tokens.chunks(cli.chunk) {
            let m = piece.len();
            let _ = rt.forward(&[SegIn {
                slot: 0,
                ids: piece,
                logits: 0,
            }])?;
            {
                let sp = spec_of(&mut *rt)?;
                #[allow(unsafe_code)] // bounds contract: iota[0..m] = 0..m < piece.len()
                // SAFETY: iota holds 0..cli.chunk and m <= cli.chunk == piece.len() of last forward call.
                unsafe {
                    sp.gather_feats(iota_dev, m, feats_dev)?;
                }
                let n_i32 = (m * FEAT_ROW_BYTES) / 4;
                sp.download(feats_dev, &mut host_i32[..n_i32])?;
            }
            rt.commit(&[u32::try_from(m)?])?;

            let byte_slice: &[u8] = bytemuck::cast_slice(&host_i32[..(m * FEAT_ROW_BYTES) / 4]);
            taps_out.write_all(byte_slice)?;
        }

        let tok_bytes: &[u8] = bytemuck::cast_slice(&row.tokens);
        toks_out.write_all(tok_bytes)?;
        mask_out.write_all(&mask_vec)?;

        writeln!(
            idx_out,
            "{}",
            json!({
                "id": row.id,
                "offset": offset,
                "len": seq_len,
                "prompt_len": prompt_len,
                "completion_tokens": comp_toks,
            })
        )?;

        total_rows += seq_len as u64;
        seq_count += 1;
        if seq_count.is_multiple_of(10) || seq_len >= 16384 {
            let dt = t0.elapsed().as_secs_f64();
            let tps = (total_rows as f64) / dt.max(1e-6);
            eprintln!(
                "[ink-taps s{}/{}] seqs={seq_count} last={} (len={seq_len}, comp={comp_toks}) total_rows={total_rows} elapsed={dt:.1}s ({tps:.0} tok/s)",
                cli.shard_idx, cli.shard_count, row.id
            );
        }
    }

    taps_out.flush()?;
    toks_out.flush()?;
    mask_out.flush()?;
    idx_out.flush()?;
    let dt = t0.elapsed().as_secs_f64();
    let tps = (total_rows as f64) / dt.max(1e-6);
    eprintln!(
        "[ink-taps s{}/{}] DONE seqs={seq_count} total_rows={total_rows} elapsed={dt:.2}s ({tps:.0} tok/s) out={}",
        cli.shard_idx,
        cli.shard_count,
        cli.out_dir.display()
    );
    Ok(())
}
