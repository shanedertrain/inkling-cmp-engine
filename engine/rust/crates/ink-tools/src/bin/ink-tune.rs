//! The pin-file tuner (review finding V11; port of the C++ engine constructor's GEMM planning, `engine.cu`
//! `Engine::Engine` "every dense shape planned (and tuned) now", and `gemm_lt.hpp` `LtGemm::tune` / `adopt`): loads the
//! checkpoint, times every split-K-1 heuristic candidate of each planned dense GEMM shape on the real weights and
//! writes the fastest as an `ink-gemm-algos/1` pin file, the file every engine start loads.
//!
//! ```text
//! ink-tune --model /model --out gemm-algos.json --record-dir DIR [--tune-m 32] [--prefill-m 2048] [--logit-rows 64]
//!          [--split-k N,K,CLASS,SPLITS,REDUCTION]... [--gpu N --uuid U --vbios T --power-cap-w W]
//! ```
//!
//! As the C++: decode-class shapes are timed at `--tune-m` rows, mixed-class shapes at `--prefill-m`, the unembedding
//! of both classes at min(`--tune-m`, `--logit-rows`); the four decode shapes of the fused input projections (N =
//! 8600, 6552, 1056, 1024; D1) take the untimed heuristic choice of N = 8600 at 64 rows (`adopt`). Operands: layer 0's
//! projections, layer 5's `wk_dyn`, layer 6's gate controllers, layer 0's router, the unembedding; x and out are
//! zeroed scratch. `--split-k` sets SPLITK_NUM and REDUCTION_SCHEME (2 compute type, 4 output type) of one tuned
//! entry afterwards (the C++ `LtGemm::split_k`); the pin format must accept it. The set of shapes is
//! `ink_runtime::planned_shapes()`, the set the engine refuses to start without.
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context as _, Result, bail, ensure};
use clap::Parser;
use ink_cuda::{
    AlgoPin, AlgoPins, Arena, Bf16, Context, DevBuf, Dtype, GemmElem, GemmShape, LtGemm, PlanClass, Stream,
};
use ink_model::dims::{H, VOCAB_PAD};
use ink_model::{WEIGHTS_BYTES_ESTIMATE, Weights};
use ink_record::{Component, ComponentName, DirtyPolicy, Gpu, Identity, RecordWriter, Source};
use serde_json::json;

#[derive(Parser)]
#[command(name = "ink-tune", version = ink_record::BUILD_LINE, about = "Tune the dense GEMM pin file")]
struct Cli {
    /// Model directory.
    #[arg(long, default_value = "/model")]
    model: PathBuf,
    /// Pin file to write; must not exist.
    #[arg(long)]
    out: PathBuf,
    /// Record directory (per-shape candidate timings); created, must not exist.
    #[arg(long)]
    record_dir: PathBuf,
    /// Rows the decode-class shapes are timed at (the C++ `gemm_tune_m`).
    #[arg(long, default_value_t = 32)]
    tune_m: u32,
    /// Rows the mixed-class (prefill) shapes are timed at.
    #[arg(long, default_value_t = 2048)]
    prefill_m: u32,
    /// Logit rows per call (bounds the unembedding's rows).
    #[arg(long, default_value_t = 64)]
    logit_rows: u32,
    /// Split-K override of one entry: `N,K,CLASS,SPLITS,REDUCTION` (CLASS decode | mixed, REDUCTION 2 | 4).
    #[arg(long)]
    split_k: Vec<String>,
    /// Expert repack threads of the loader.
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

/// Rows of the x scratch: the largest timed M times the largest K (4,608 = 2H).
const X_ELEMS: usize = 2048 * 2 * H;
/// Elements of the out scratch (fp32): the C++ `logits_` (64 logit rows x VOCAB_PAD).
const OUT_F32: usize = 64 * VOCAB_PAD;

/// One `--split-k` override.
struct SplitK {
    n: u32,
    k: u32,
    class: PlanClass,
    splits: i32,
    reduction: u32,
}

fn parse_split_k(s: &str) -> Result<SplitK> {
    let p: Vec<&str> = s.split(',').collect();
    ensure!(p.len() == 5, "--split-k {s}: want N,K,CLASS,SPLITS,REDUCTION");
    let class = match p[2] {
        "decode" => PlanClass::Decode,
        "mixed" => PlanClass::Mixed,
        c => bail!("--split-k {s}: class {c} (decode or mixed)"),
    };
    let r = SplitK {
        n: p[0].parse()?,
        k: p[1].parse()?,
        class,
        splits: p[3].parse()?,
        reduction: p[4].parse()?,
    };
    ensure!(r.splits > 1, "--split-k {s}: SPLITS must be > 1");
    ensure!(
        matches!(r.reduction, 2 | 4),
        "--split-k {s}: REDUCTION must be 2 (compute type) or 4 (output type)"
    );
    Ok(r)
}

/// The weight a shape is timed on (the C++ constructor's choice) and the rows.
fn operand<'w>(w: &'w Weights, s: &GemmShape, a: &Cli) -> Result<(Operand<'w>, u32)> {
    let (l0, l5, l6) = (&w.layers[0], &w.layers[5], &w.layers[6]);
    let ctl = l6.controllers.as_ref().context("layer 6 has no loop controllers")?;
    let m = match s.class {
        PlanClass::Decode => a.tune_m,
        PlanClass::Mixed => a.prefill_m,
    };
    let (n, k) = (s.n as usize, s.k as usize);
    let op = match (n, k, s.input) {
        (n, k, Dtype::Bf16) if k == H && n == l0.wq.len() / H => Operand::Bf16(&l0.wq),
        (n, k, Dtype::Bf16) if k == H && n == l0.wr.len() / H => Operand::Bf16(&l0.wr),
        (n, k, Dtype::Bf16) if k == H && n == l0.wk.len() / H => Operand::Bf16(&l0.wk),
        (n, k, Dtype::Bf16) if k == H && n == l0.wsink.len() / H => Operand::Bf16(&l0.wsink),
        (32, k, Dtype::Bf16) if k == H => Operand::Bf16(l5.wk_dyn.as_ref().context("layer 5 has no wk_dyn")?),
        (n, k, Dtype::Bf16) if n == H && k == l0.wo.len() / H => Operand::Bf16(&l0.wo),
        (n, k, Dtype::Bf16) if k == 2 * H && n == ctl[0].down.len() / k => Operand::Bf16(&ctl[0].down),
        (n, k, Dtype::F32) if n == H && k == ctl[0].forget.len() / H => Operand::F32(&ctl[0].forget),
        (VOCAB_PAD, k, Dtype::Bf16) if k == H => {
            return Ok((Operand::Bf16(&w.unembed), a.tune_m.min(a.logit_rows)));
        }
        (n, _, Dtype::Bf16) if s.ldw as usize == H && n == l0.router.len() / H => Operand::Bf16(&l0.router),
        _ => bail!("{s}: no operand for this shape (the engine plans a shape this tuner does not know)"),
    };
    Ok((op, m))
}

enum Operand<'w> {
    Bf16(&'w DevBuf<Bf16>),
    F32(&'w DevBuf<f32>),
}

/// Times `shape` on `w` (x and out from the scratch, typed as the shape needs).
#[expect(clippy::too_many_arguments, reason = "the GEMM and its timing context")]
fn tune_one<I: GemmElem>(
    lt: &mut LtGemm,
    st: &Stream,
    ev: (&ink_cuda::Event, &ink_cuda::Event),
    shape: GemmShape,
    m: u32,
    w: &DevBuf<I>,
    x: &DevBuf<u8>,
    out: &DevBuf<u8>,
) -> Result<ink_cuda::Tuned> {
    let x = x.cast::<I>()?;
    Ok(match shape.output {
        Dtype::Bf16 => lt.tune(st, ev.0, ev.1, shape, m, &x, w, &out.cast::<Bf16>()?)?,
        Dtype::F32 => lt.tune(st, ev.0, ev.1, shape, m, &x, w, &out.cast::<f32>()?)?,
    })
}

fn pin_json(p: &AlgoPin) -> serde_json::Value {
    json!({"ID": p.id, "TILE_ID": p.tile, "SPLITK_NUM": p.splitk, "REDUCTION_SCHEME": p.reduction,
           "CTA_SWIZZLING": p.swizzle, "CUSTOM_OPTION": p.custom, "STAGES_ID": p.stages,
           "INNER_SHAPE_ID": p.inner_shape, "CLUSTER_SHAPE_ID": p.cluster_shape})
}

fn fused_base_n(shapes: &[GemmShape]) -> Option<u32> {
    shapes
        .iter()
        .filter(|s| {
            s.class == PlanClass::Decode
                && s.k as usize == H
                && s.input == Dtype::Bf16
                && s.output == Dtype::Bf16
                && s.ldw as usize == H
                && s.n > 4096
        })
        .map(|s| s.n)
        .max()
}

fn is_fused(s: &GemmShape, fused_n: u32) -> bool {
    s.class == PlanClass::Decode
        && s.k as usize == H
        && s.input == Dtype::Bf16
        && s.output == Dtype::Bf16
        && s.ldw as usize == H
        && (s.n == fused_n || s.n == 6552 || s.n == 1056 || s.n == 1024)
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
    ensure!(
        !a.out.exists(),
        "{} exists (the tuner writes a new pin file)",
        a.out.display()
    );
    let splits = a.split_k.iter().map(|s| parse_split_k(s)).collect::<Result<Vec<_>>>()?;
    fs::create_dir(&a.record_dir).with_context(|| format!("creating {} (must not exist)", a.record_dir.display()))?;
    let identity = Identity::new(
        Source::this_build(),
        "tune",
        [(ComponentName::Tools, Component::Source(Source::this_build()))],
    );
    let context = ink_record::Context {
        argv: std::env::args().collect(),
        config: json!({"model": a.model, "tune_m": a.tune_m, "prefill_m": a.prefill_m, "logit_rows": a.logit_rows,
                       "split_k": a.split_k, "out": a.out}),
        gpu,
        power_cap_w: a.power_cap_w,
        baseline: "C++ native LtGemm::tune (engine/native/src/gemm_lt.hpp) and engine/pins/gemm-algos.json".to_owned(),
        label: "ink-tune".to_owned(),
    };
    let policy = if a.allow_dirty {
        DirtyPolicy::Allow
    } else {
        DirtyPolicy::Refuse
    };
    let mut rec = RecordWriter::create(&a.record_dir, "tune", identity, context, policy)?;

    let ctx = Context::new(0)?;
    let dev = ctx.device()?;
    ensure!(
        a.uuid.is_empty() || dev.uuid == a.uuid,
        "device {} is not --uuid {}",
        dev.uuid,
        a.uuid
    );
    let st = ctx.new_stream()?;
    let mut arena = Arena::new(
        &ctx,
        &st,
        WEIGHTS_BYTES_ESTIMATE + 2 * X_ELEMS + 4 * OUT_F32 + (64 << 20),
    )?;
    let w = Weights::load(&mut arena, &st, &a.model, a.threads)?;
    let x = arena.take::<u8>(2 * X_ELEMS, "tune x")?;
    let out = arena.take::<u8>(4 * OUT_F32, "tune out")?;
    st.memset_zero(&x)?;
    st.memset_zero(&out)?;
    st.synchronize()?;
    let mut lt = LtGemm::new(&st, AlgoPins::new(0))?;
    let version = ctx.startup().cublaslt_version;
    let (e0, e1) = (ctx.new_event()?, ctx.new_event()?);
    let shapes = ink_runtime::planned_shapes();
    let fused_n = fused_base_n(&shapes).context("no fused projection shape")?;
    let fused = |s: &GemmShape| is_fused(s, fused_n);
    let mut pins: BTreeMap<GemmShape, AlgoPin> = BTreeMap::new();
    // the four fused decode shapes: the untimed heuristic choice of N = 8600 at 64 rows (the C++ adopt, which runs
    // before the timed shapes and keeps decode N = 1024 out of tuning)
    let base = shapes
        .iter()
        .find(|s| fused(s) && s.n == fused_n)
        .copied()
        .context("no N = 8600 shape")?;
    let adopted = lt.select(base, 64)?;
    for s in shapes.iter().filter(|s| fused(s)) {
        pins.insert(*s, adopted.pin);
        eprintln!(
            "[gemm-tune] {s}: adopted N={fused_n} heuristic candidate {} of {}",
            adopted.candidate, adopted.candidates
        );
        rec.append(
            &json!({"what": "adopt", "shape": s.to_string(), "from": base.to_string(), "m": 64,
                            "candidate": adopted.candidate, "candidates": adopted.candidates,
                            "algo": pin_json(&adopted.pin)}),
        )?;
    }
    for s in shapes.iter().filter(|s| !fused(s)) {
        let (op, m) = operand(&w, s, &a)?;
        let t = match op {
            Operand::Bf16(wb) => tune_one(&mut lt, &st, (&e0, &e1), *s, m, wb, &x, &out)?,
            Operand::F32(wb) => tune_one(&mut lt, &st, (&e0, &e1), *s, m, wb, &x, &out)?,
        };
        let us = t.candidates[t.best].us.unwrap_or(f32::NAN);
        eprintln!(
            "[gemm-tune] {s} M={m}: candidate {} of {}, algo {} tile {}, {us:.2} us",
            t.best,
            t.candidates.len(),
            t.pin.id,
            t.pin.tile
        );
        let cands: Vec<_> = t
            .candidates
            .iter()
            .map(|c| json!({"index": c.index, "us": c.us, "algo": pin_json(&c.pin)}))
            .collect();
        rec.append(
            &json!({"what": "tune", "shape": s.to_string(), "m": m, "best": t.best, "us": us,
                            "algo": pin_json(&t.pin), "candidates": cands}),
        )?;
        pins.insert(*s, t.pin);
    }
    for o in &splits {
        let s = GemmShape {
            n: o.n,
            k: o.k,
            ldw: o.k,
            input: Dtype::Bf16,
            output: Dtype::Bf16,
            class: o.class,
        };
        let p = pins
            .get_mut(&s)
            .with_context(|| format!("--split-k: {s} is not a planned shape"))?;
        p.splitk = o.splits;
        p.reduction = o.reduction;
        rec.append(&json!({"what": "split_k", "shape": s.to_string(), "algo": pin_json(p)}))?;
    }
    ensure!(
        pins.len() == shapes.len(),
        "{} pins for {} planned shapes",
        pins.len(),
        shapes.len()
    );
    let mut file = AlgoPins::new(version);
    for (s, p) in &pins {
        file.insert(*s, *p)?;
    }
    let text = file.to_json();
    AlgoPins::from_json(&text).context("the written pin file does not read back")?;
    let tmp = a.out.with_extension(format!("tmp.{}", std::process::id()));
    fs::write(&tmp, &text).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, &a.out).with_context(|| format!("renaming to {}", a.out.display()))?;
    rec.append(&json!({"what": "pins", "file": a.out, "shapes": pins.len(), "cublaslt_version": version}))?;
    eprintln!("[gemm-tune] wrote {} shapes to {}", pins.len(), a.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_base_selects_8600_not_unembed() {
        let shapes = ink_runtime::planned_shapes();
        let base = fused_base_n(&shapes).expect("fused base");
        assert_eq!(base, 8600);
        let mut adopted: Vec<u32> = shapes.iter().filter(|s| is_fused(s, base)).map(|s| s.n).collect();
        adopted.sort_unstable();
        assert_eq!(adopted, vec![1024, 1056, 6552, 8600]);
    }

    #[test]
    fn parse_split_k_checks_format_and_bounds() {
        let o = parse_split_k("2304,3072,decode,4,2").expect("valid");
        assert_eq!(
            (o.n, o.k, o.class, o.splits, o.reduction),
            (2304, 3072, PlanClass::Decode, 4, 2)
        );
        assert!(parse_split_k("bad").is_err());
        assert!(parse_split_k("2304,3072,decode,1,2").is_err());
        assert!(parse_split_k("2304,3072,decode,4,3").is_err());
    }
}
