//! Op-by-op comparison against the reference's dumped tensors (package T1; port of
//! `engine/native/tools/optest.cpp`, deleted at cut-over U3a).
//!
//! ```text
//! ink-optest --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir DIR --dump /dumps/short --mode chain \
//!            [--save tap,tap] [--out-dir /out] > out.jsonl
//! ```
//!
//! Replays the forward calls of a dump (`engine/native/tests/dump_ref.py`), comparing every tap the dump has with the
//! engine's tensor after the same op. `chain`: the engine runs on its own values (end-to-end drift). `iso`: the
//! engine's tensor is then overwritten with the reference's (the tap sink's write-back of `ink_rt.h`), so the next op
//! sees the reference's inputs. Dumps with fewer rows than the call (the `long` scenario) compare the last rows. One JSON line per (call, tap) and per (call,
//! segment) for the logits, in the C++ tool's format; `engine/native/tests/summ_optest.py` turns them into tables.
#![forbid(unsafe_code)]
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::{Parser, ValueEnum};
use ink_rt_api::{LOGIT_ROW, SegIn, TapType, VOCAB};
use ink_tools::engine::{EngineArgs, MAX_CHUNK, RecordArgs};
use ink_tools::fmt::g;
use serde_json::{Value, json};

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    Iso,
    Chain,
}

#[derive(Parser)]
#[command(name = "ink-optest", version = ink_record::BUILD_LINE, about = "Per-op comparison against dumped tensors")]
struct Cli {
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    record: RecordArgs,
    /// Global KV pages of 1,024 tokens.
    #[arg(long, default_value_t = 64)]
    pages: u32,
    /// Dump directory (`index.json` and the tensor files).
    #[arg(long)]
    dump: PathBuf,
    /// `iso` (the reference's tensor replaces the engine's after each compared op) or `chain`.
    #[arg(long, value_enum, default_value = "iso")]
    mode: Mode,
    /// Taps whose engine bytes are written to `--out-dir/nat_c<call>_<tap>.bin` (comma-separated).
    #[arg(long, default_value = "")]
    save: String,
    /// Directory of the saved taps.
    #[arg(long, default_value = "/out")]
    out_dir: PathBuf,
}

/// What the tap sink needs about the call being replayed.
struct Call {
    id: i64,
    /// `tensors` object of the call in `index.json`.
    tensors: Value,
    /// Rows of the call.
    rows: usize,
}

struct TapState {
    mode: Mode,
    dump: PathBuf,
    save: Vec<String>,
    out_dir: PathBuf,
    call: Call,
    /// JSON lines of the taps of the current call.
    lines: Vec<String>,
    error: Option<String>,
}

/// Element of a tapped tensor, as a comparable value.
fn element(ty: TapType, bytes: &[u8], i: usize) -> (u32, f64) {
    match ty {
        TapType::Bf16 => {
            let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            (u32::from(bits), f64::from(f32::from_bits(u32::from(bits) << 16)))
        }
        TapType::F32 => {
            let bits = u32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]]);
            (bits, f64::from(f32::from_bits(bits)))
        }
        TapType::I32 => {
            let bits = u32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]]);
            (bits, f64::from(i32::from_le_bytes(bits.to_le_bytes())))
        }
    }
}

impl TapState {
    /// Compares one tap with the dump; returns the bytes of `scratch` to write back over the engine's tensor.
    fn tap(&mut self, name: &str, ty: TapType, shape: &[i64], data: &[u8], scratch: &mut [u8]) -> Result<usize> {
        let Some(tj) = self.call.tensors.get(name) else {
            return Ok(0);
        };
        let n = shape
            .iter()
            .try_fold(1_usize, |n, &d| usize::try_from(d).ok().and_then(|d| n.checked_mul(d)));
        let n = n.with_context(|| format!("tap {name}: bad shape {shape:?}"))?;
        let first = usize::try_from(*shape.first().context("tap without shape")?)?;
        ensure!(first > 0, "tap {name}: empty first dimension");
        let per_row = n / first;
        let ref_rows = usize::try_from(tj["shape"][0].as_i64().context("dump shape")?)?;
        let row_off = self
            .call
            .rows
            .checked_sub(ref_rows)
            .with_context(|| format!("dump has more rows than the call: {name}"))?;
        let m = ref_rows * per_row;
        let esz = if ty == TapType::Bf16 { 2 } else { 4 };
        let file = tj["file"].as_str().context("dump file name")?;
        let reference = fs::read(self.dump.join(file)).with_context(|| format!("reading dump {file}"))?;
        ensure!(reference.len() == m * esz, "dump size {name}");
        let nat = &data[row_off * per_row * esz..(row_off * per_row + m) * esz];
        let (mut ndiff, mut nonfinite) = (0_usize, 0_usize);
        let (mut se, mut sr, mut mx, mut mref) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..m {
            let (xb, a) = element(ty, nat, i);
            let (yb, b) = element(ty, &reference, i);
            ndiff += usize::from(xb != yb);
            if !a.is_finite() {
                nonfinite += 1;
                continue;
            }
            se += (a - b) * (a - b);
            sr += b * b;
            mx = mx.max((a - b).abs());
            mref = mref.max(b.abs());
        }
        let frac = ndiff as f64 / m.max(1) as f64;
        self.lines.push(format!(
            "{{\"call\":{},\"tap\":\"{name}\",\"rows\":{ref_rows},\"n\":{m},\"ndiff\":{ndiff},\"frac_diff\":{},\"max_abs\":{},\
             \"max_ref\":{},\"rel_rms\":{},\"nonfinite\":{nonfinite}}}",
            self.call.id,
            g(frac, 6),
            g(mx, 6),
            g(mref, 6),
            g((se / sr.max(1e-300)).sqrt(), 6)
        ));
        if self.save.iter().any(|s| s == name) {
            let path = self.out_dir.join(format!("nat_c{}_{name}.bin", self.call.id));
            fs::write(&path, nat).with_context(|| path.display().to_string())?;
        }
        // iso: the reference's rows replace the engine's when they cover the tensor from its first row
        if self.mode == Mode::Iso && row_off == 0 {
            scratch[..reference.len()].copy_from_slice(&reference);
            return Ok(reference.len());
        }
        Ok(0)
    }
}

/// KL(reference || engine), argmaxes and the largest absolute difference of one logit row.
fn compare_row(a: &[f32], b: &[f32]) -> (f64, usize, usize, f64) {
    let (mut ma, mut mb) = (-1e30_f64, -1e30_f64);
    let (mut aa, mut ab) = (0, 0);
    let mut mx = 0.0_f64;
    for k in 0..VOCAB {
        if f64::from(a[k]) > ma {
            ma = f64::from(a[k]);
            aa = k;
        }
        if f64::from(b[k]) > mb {
            mb = f64::from(b[k]);
            ab = k;
        }
        mx = mx.max(f64::from((a[k] - b[k]).abs()));
    }
    let (mut za, mut zb) = (0.0_f64, 0.0_f64);
    for k in 0..VOCAB {
        za += (f64::from(a[k]) - ma).exp();
        zb += (f64::from(b[k]) - mb).exp();
    }
    let (la, lb) = (za.ln() + ma, zb.ln() + mb);
    let mut kl = 0.0_f64;
    for k in 0..VOCAB {
        let pb = (f64::from(b[k]) - lb).exp();
        kl += pb * ((f64::from(b[k]) - lb) - (f64::from(a[k]) - la));
    }
    (kl, aa, ab, mx)
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn index_calls(dump: &Path) -> Result<Vec<Value>> {
    let path = dump.join("index.json");
    let text = fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    let mut idx: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    match idx["calls"].take() {
        Value::Array(calls) => Ok(calls),
        _ => bail!("{}: no calls array", path.display()),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let calls = index_calls(&cli.dump)?;
    let mut rt = cli.engine.open(cli.pages, MAX_CHUNK)?;
    let mode = match cli.mode {
        Mode::Iso => "iso",
        Mode::Chain => "chain",
    };
    let mut record = cli.record.writer(
        "optest",
        &*rt,
        json!({"engine": cli.engine.json(cli.pages, MAX_CHUNK), "dump": cli.dump, "mode": mode, "save": cli.save}),
        "the S1 reference's dumped tensors (engine/native/tests/dump_ref.py)",
    )?;
    let state = Arc::new(Mutex::new(TapState {
        mode: cli.mode,
        dump: cli.dump.clone(),
        save: cli
            .save
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect(),
        out_dir: cli.out_dir.clone(),
        call: Call {
            id: 0,
            tensors: Value::Null,
            rows: 0,
        },
        lines: Vec::new(),
        error: None,
    }));
    let sink = Arc::clone(&state);
    rt.set_tap(Some(Box::new(
        move |name: &str, ty: TapType, shape: &[i64], data: &[u8], scratch: &mut [u8]| {
            let Ok(mut s) = sink.lock() else { return 0 };
            if s.error.is_some() {
                return 0;
            }
            s.tap(name, ty, shape, data, scratch).unwrap_or_else(|e| {
                s.error = Some(format!("{e:#}"));
                0
            })
        },
    )))?;
    let lock = || state.lock().map_err(|_| anyhow!("tap state poisoned"));
    let mut out = BufWriter::new(std::io::stdout());
    let mut seen = std::collections::BTreeSet::new();
    for c in &calls {
        let id = c["id"].as_i64().context("call id")?;
        let call_segs = c["segs"].as_array().context("call segs")?;
        let mut ids: Vec<Vec<i32>> = Vec::new();
        let mut plan: Vec<(usize, usize, usize)> = Vec::new(); // slot, pos0, logit rows
        let mut rows = 0;
        for (i, s) in call_segs.iter().enumerate() {
            let slot = usize::try_from(s["slot"].as_i64().context("seg slot")?)?;
            let seg_ids: Vec<i32> = s["ids"]
                .as_array()
                .context("seg ids")?
                .iter()
                .map(|x| x.as_i64().and_then(|v| i32::try_from(v).ok()).context("seg id"))
                .collect::<Result<_>>()?;
            let pos0 = usize::try_from(s["pos0"].as_i64().context("seg pos0")?)?;
            if seen.insert(slot) {
                rt.reset_slot(slot)?;
            }
            ensure!(rt.length(slot)? == pos0, "session length differs from the dump");
            let logits = match c["tensors"].get(format!("logits.s{i}")) {
                Some(lj) => usize::try_from(lj["shape"][0].as_i64().context("logits shape")?)?,
                None => 0,
            };
            rows += seg_ids.len();
            ids.push(seg_ids);
            plan.push((slot, pos0, logits));
        }
        {
            let mut s = lock()?;
            s.call = Call {
                id,
                tensors: c["tensors"].clone(),
                rows,
            };
            s.lines.clear();
        }
        let segs: Vec<SegIn<'_>> = plan
            .iter()
            .zip(&ids)
            .map(|(&(slot, _, logits), ids)| SegIn { slot, ids, logits })
            .collect();
        let nl = rt.forward(&segs)?;
        {
            let mut s = lock()?;
            if let Some(e) = s.error.take() {
                bail!("call {id}: {e}");
            }
            for line in s.lines.drain(..) {
                writeln!(out, "{line}")?;
            }
        }
        let mut lg = Vec::new();
        rt.logits_host(nl, &mut lg)?;
        let mut r0 = 0;
        for (i, &(_, _, logits)) in plan.iter().enumerate() {
            let Some(lj) = c["tensors"].get(format!("logits.s{i}")) else {
                continue;
            };
            let file = lj["file"].as_str().context("logits file")?;
            let reference = floats(&fs::read(cli.dump.join(file)).with_context(|| format!("reading dump {file}"))?);
            for r in 0..logits {
                let a = &lg[(r0 + r) * LOGIT_ROW..(r0 + r + 1) * LOGIT_ROW];
                let b = &reference[r * VOCAB..(r + 1) * VOCAB];
                let (kl, aa, ab, mx) = compare_row(a, b);
                writeln!(
                    out,
                    "{{\"call\":{id},\"logits_seg\":{i},\"row\":{r},\"kl_ref_nat\":{},\"argmax_nat\":{aa},\"argmax_ref\":{ab},\
                     \"max_abs\":{}}}",
                    g(kl, 6),
                    g(mx, 6)
                )?;
            }
            r0 += logits;
        }
        let kept: Vec<u32> = ids.iter().map(|v| u32::try_from(v.len())).collect::<Result<_, _>>()?;
        rt.commit(&kept)?;
        out.flush()?;
    }
    out.flush()?;
    let stats = rt.stats()?;
    eprintln!(
        "[optest] done, arena {:.2} GiB",
        stats.arena_used as f64 / 1_073_741_824.0
    );
    record.append(&json!({"done": true, "calls": calls.len(), "stats": {"arena_used": stats.arena_used}}))?;
    eprintln!("record {}", record.path().display());
    Ok(())
}
