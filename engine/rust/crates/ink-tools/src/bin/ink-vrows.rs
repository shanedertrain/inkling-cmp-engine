//! Verify rows against one-row decode rows along a speculative path (finding F1, `bench/raw/2026-11-23-f1-verify-rows/`).
//!
//! ```text
//! ink-vrows --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir DIR --cases cases.txt --case ID \
//!           --path 400 --fold /s2f/fold.bin --out rows.jsonl [--tap-pos P --tap-out taps.jsonl]
//! ```
//!
//! One golden greedy case, one session. (1) Speculative greedy in slot 0 (drafter, 8-row verify, device accept), as
//! `ink-spec`; every round's ids, accept count and the logits of its kept rows are kept. (2) Slot 1: the plain
//! one-row pass teacher-forced along the speculative tokens (`ink-spec`'s path check). (3) Slot 2: the same plain
//! pass, but before the decode row at every round's first position the round's 8-row verify segment runs on the
//! decode-built history and is rolled back (commit 0). (4) Slot 3: the speculative rounds replayed (same ids, same
//! accept counts), with a one-row decode of the round's anchor on the verify-built history before each round (rolled
//! back). One JSON line per compared row: `cmp` (which two), absolute position `q` of the row, row index `j` in its
//! segment, differing bits, largest |difference|, KL(plain || other) in nats, argmax of each, and the plain row's
//! logit gap to the other row's argmax.
//!
//! `--tap-pos P` (P = a round's first position): per-op comparison of every verify row of that round
//! against the one-row decode of the same position, both on the decode-built history (glue 1, taps). The decode rows
//! of positions P..P+7 are recorded first; the verify call then compares every tapped op of its rows with them and,
//! with `--tap-iso`, writes the decode rows back (each op sees the decode path's inputs: isolated per-op difference).
#![deny(unsafe_code)] // allowed only at the calls of the runtime's bounds-contract methods
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use ink_rt_api::{
    AcceptBufs, DRAFTS, DevBuf, DevSeg, DraftConfig, Drafter, FEAT_ROW_BYTES, Glue, HeadPrecision, KvPrecision,
    Runtime, SegIn, SpecRuntime, TapType, VOCAB, VerifyWidth,
};
use ink_tools::cases::read_cases;
use ink_tools::engine::{EngineArgs, MAX_CHUNK, RecordArgs};
use ink_tools::session::row;
use serde_json::json;

#[derive(Parser)]
#[command(name = "ink-vrows", version = ink_record::BUILD_LINE, about = "Verify rows vs one-row decode rows")]
struct Cli {
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    record: RecordArgs,
    /// Golden cases (`cases.txt` of `gold_prep.py`).
    #[arg(long, default_value = "/gin/cases.txt")]
    cases: PathBuf,
    /// The greedy case.
    #[arg(long)]
    case: String,
    /// Speculative tokens.
    #[arg(long, default_value_t = 400)]
    path: usize,
    /// Folded drafter weights.
    #[arg(long, default_value = "/s2f/fold.bin")]
    fold: PathBuf,
    /// Drafter positions.
    #[arg(long, default_value_t = 65_536)]
    draft_ctx: u32,
    /// KV pages.
    #[arg(long, default_value_t = 64)]
    pages: u32,
    /// Row comparisons (JSON lines).
    #[arg(long)]
    out: PathBuf,
    /// Per-op comparison at the round starting at this position (0: none).
    #[arg(long, default_value_t = 0)]
    tap_pos: usize,
    /// Write the decode rows back after each tapped op of the verify call.
    #[arg(long)]
    tap_iso: bool,
    /// Also the plain pass on the reference glue (glue 1), compared with the plain pass.
    #[arg(long)]
    glue1_plain: bool,
    /// Per-op lines (JSON).
    #[arg(long, default_value = "taps.jsonl")]
    tap_out: PathBuf,
}

const CHUNK: usize = MAX_CHUNK as usize;

fn argmax_first(r: &[f32]) -> usize {
    (1..VOCAB).fold(0, |a, k| if r[k] > r[a] { k } else { a })
}

fn id(i: usize) -> i32 {
    i32::try_from(i).expect("small")
}

fn spec_of(rt: &mut dyn Runtime) -> Result<&mut dyn SpecRuntime> {
    rt.spec().context("the runtime has no device-side speculation")
}

/// log-sum-exp of the valid logits (f64).
fn lse(r: &[f32]) -> f64 {
    let m = r[..VOCAB].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    m + r[..VOCAB].iter().map(|&x| (f64::from(x) - m).exp()).sum::<f64>().ln()
}

/// One row comparison: `a` is the reference (plain one-row decode).
fn compare(cmp: &str, q: usize, j: usize, a: &[f32], b: &[f32]) -> serde_json::Value {
    let (a, b) = (&a[..VOCAB], &b[..VOCAB]);
    let ndiff = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let max_abs = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (&x, &y)| m.max(f64::from((x - y).abs())));
    let (la, lb) = (lse(a), lse(b));
    let kl: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let lp = f64::from(x) - la;
            lp.exp() * (lp - (f64::from(y) - lb))
        })
        .sum();
    let (ia, ib) = (argmax_first(a), argmax_first(b));
    json!({"cmp": cmp, "q": q, "j": j, "ndiff": ndiff, "max_abs": max_abs, "kl": kl.max(0.0), "argmax_ref": ia,
           "argmax_other": ib, "gap": f64::from(a[ia]) - f64::from(a[ib]),
           "margin_ref": f64::from(a[ia]) - a.iter().enumerate().filter(|&(k, _)| k != ia)
               .fold(f64::NEG_INFINITY, |m, (_, &x)| m.max(f64::from(x)))})
}

struct Round {
    pos0: usize,
    ids: Vec<i32>,
    accepted: usize,
    rows: Vec<Vec<f32>>,
}

fn decode_row(rt: &mut dyn Runtime, slot: usize, tok: i32, buf: &mut Vec<f32>) -> Result<Vec<f32>> {
    let n = rt.forward(&[SegIn {
        slot,
        ids: &[tok],
        logits: 1,
    }])?;
    rt.logits_host(n, buf)?;
    rt.commit(&[1])?;
    Ok(row(buf, 0).to_vec())
}

fn verify_rows(rt: &mut dyn Runtime, slot: usize, ids: &[i32], keep: u32, buf: &mut Vec<f32>) -> Result<Vec<Vec<f32>>> {
    let n = rt.forward(&[SegIn {
        slot,
        ids,
        logits: ids.len(),
    }])?;
    rt.logits_host(n, buf)?;
    rt.commit(&[keep])?;
    Ok((0..n).map(|r| row(buf, r).to_vec()).collect())
}

fn prefill(rt: &mut dyn Runtime, slot: usize, ids: &[i32]) -> Result<Vec<f32>> {
    Ok(ink_tools::session::prefill(rt, slot, ids, CHUNK, true)?.last)
}

/// Tensor rows recorded by the tap sink: op name -> (type, row bytes, bytes of every row of the call).
type OpRows = BTreeMap<String, (TapType, usize, Vec<u8>)>;

fn elem_bytes(t: TapType) -> usize {
    match t {
        TapType::Bf16 => 2,
        TapType::F32 | TapType::I32 => 4,
    }
}

fn val(t: TapType, b: &[u8], i: usize) -> f64 {
    match t {
        TapType::Bf16 => f64::from(f32::from_bits(
            u32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])) << 16,
        )),
        TapType::F32 => f64::from(f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().expect("4"))),
        TapType::I32 => f64::from(i32::from_le_bytes(b[4 * i..4 * i + 4].try_into().expect("4"))),
    }
}

/// Per-op difference of row `ra` of `a` and row `rb` of `b` (same op): differing elements, largest |difference|,
/// largest difference relative to max(|a|, |b|) (1 bf16 ulp is 2^-8 .. 2^-7 relative), rms of a.
fn op_diff(t: TapType, rowb: usize, a: &[u8], b: &[u8]) -> (usize, f64, f64, f64) {
    let n = rowb / elem_bytes(t);
    let (mut nd, mut mx, mut rel, mut ss) = (0, 0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        let (x, y) = (val(t, a, i), val(t, b, i));
        ss += x * x;
        if x.to_bits() != y.to_bits() {
            nd += 1;
            let d = (x - y).abs();
            mx = mx.max(d);
            let s = x.abs().max(y.abs());
            if s > 0.0 {
                rel = rel.max(d / s);
            }
        }
    }
    (nd, mx, rel, (ss / n.max(1) as f64).sqrt())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.engine.slots >= 4, "ink-vrows uses slots 0-3");
    let cases = read_cases(&cli.cases)?;
    let case = cases
        .iter()
        .find(|c| c.id == cli.case)
        .with_context(|| format!("no case {}", cli.case))?;
    let prompt = case.prompt.clone();
    let plen = prompt.len();
    let mut rt = cli.engine.open(cli.pages, MAX_CHUNK)?;
    let mut record = cli.record.writer(
        "vrows",
        &*rt,
        json!({"engine": cli.engine.json(cli.pages, MAX_CHUNK), "cases": cli.cases, "case": cli.case,
               "path": cli.path, "fold": cli.fold, "draft_ctx": cli.draft_ctx, "tap_pos": cli.tap_pos,
               "tap_iso": cli.tap_iso}),
        "plain one-row decode rows of the same build (slot 1)",
    )?;
    let rt: &mut dyn Runtime = &mut *rt;
    let mut out = BufWriter::new(File::create(&cli.out).with_context(|| cli.out.display().to_string())?);
    let mut buf = Vec::new();

    // (1) speculative greedy, slot 0, drafter of one session
    let mut dr = spec_of(rt)?.create_drafter(&DraftConfig {
        model_dir: cli.engine.model.clone(),
        fold_bin: cli.fold.clone(),
        dflash2_dir: None,
        sessions: 1,
        max_ctx: cli.draft_ctx,
        head: HeadPrecision::Int8,
        global_kv: KvPrecision::Bf16,
        verify: VerifyWidth::Full,
        lookup: 0,
        lookup_min: 8,
    })?;
    let rounds = speculate(rt, dr.as_mut(), &prompt, cli.path, cli.draft_ctx as usize, &mut buf)?;
    // round r emits its accepted drafts and a bonus token, the anchor of round r + 1
    let mut spec: Vec<i32> = Vec::new();
    for r in &rounds {
        spec.extend_from_slice(&r.ids[..=r.accepted]);
        ensure!(r.rows.len() == r.accepted + 1, "rows kept");
    }
    // the bonus token of round r is the anchor of round r + 1
    for w in rounds.windows(2) {
        ensure!(w[1].pos0 == w[0].pos0 + w[0].accepted + 1, "round positions");
    }
    spec.truncate(cli.path);
    let start: BTreeMap<usize, usize> = rounds.iter().enumerate().map(|(i, r)| (r.pos0, i)).collect();
    eprintln!(
        "[vrows] {}: prompt {plen}, {} rounds, {} tokens",
        cli.case,
        rounds.len(),
        spec.len()
    );

    // (2) plain pass, slot 1: plain[q - plen + 1] = logits of the row at position q (q = plen - 1: prefill)
    let mut plain = vec![prefill(rt, 1, &prompt)?];
    for k in 1..spec.len() {
        plain.push(decode_row(rt, 1, spec[k - 1], &mut buf)?);
    }
    let pl = |q: usize| -> Option<&Vec<f32>> { plain.get(q + 1 - plen) };
    let mut emit = |v: serde_json::Value| -> Result<()> {
        writeln!(out, "{v}")?;
        Ok(())
    };
    // speculative verify rows vs plain
    for r in &rounds {
        for (j, rw) in r.rows.iter().enumerate() {
            if let Some(a) = pl(r.pos0 + j) {
                emit(compare("spec_vs_plain", r.pos0 + j, j, a, rw))?;
            }
        }
    }

    // (3) slot 2: plain pass with each round's verify segment on the decode-built history (rolled back)
    prefill(rt, 2, &prompt)?;
    for k in 1..spec.len() {
        let q = plen + k - 1;
        if let Some(&ri) = start.get(&q) {
            let r = &rounds[ri];
            ensure!(r.ids[0] == spec[k - 1], "anchor");
            let rows = verify_rows(rt, 2, &r.ids, 0, &mut buf)?;
            for (j, rw) in rows.iter().enumerate().take(r.accepted + 1) {
                if let Some(a) = pl(q + j) {
                    emit(compare("verify_on_decode_hist_vs_plain", q + j, j, a, rw))?;
                }
                emit(compare("verify_on_decode_hist_vs_spec", q + j, j, &r.rows[j], rw))?;
            }
        }
        let d = decode_row(rt, 2, spec[k - 1], &mut buf)?;
        if let Some(a) = pl(q) {
            emit(compare("decode_after_rollback_vs_plain", q, 0, a, &d))?;
        }
    }

    // (4) slot 3: replay of the rounds; before each, a one-row decode of the anchor on the verify-built history
    prefill(rt, 3, &prompt)?;
    for r in &rounds {
        ensure!(rt.length(3)? == r.pos0, "replay position");
        let d = verify_rows(rt, 3, &r.ids[..1], 0, &mut buf)?;
        if let Some(a) = pl(r.pos0) {
            emit(compare("decode_on_spec_hist_vs_plain", r.pos0, 0, a, &d[0]))?;
        }
        emit(compare("decode_on_spec_hist_vs_spec", r.pos0, 0, &r.rows[0], &d[0]))?;
        let keep = u32::try_from(r.accepted + 1)?;
        let rows = verify_rows(rt, 3, &r.ids, keep, &mut buf)?;
        for (j, rw) in rows.iter().enumerate().take(r.accepted + 1) {
            emit(compare("replay_vs_spec", r.pos0 + j, j, &r.rows[j], rw))?;
        }
    }
    // (5) slot 0: the plain pass on the per-op reference glue (another fp32 order of the message gates and norms: an
    // arithmetic change of the same kind as the verify rows', for scale)
    if cli.glue1_plain {
        rt.set_glue(Glue::Reference)?;
        let last = prefill(rt, 0, &prompt)?;
        emit(compare("plain_glue1_vs_plain", plen - 1, 0, &plain[0], &last))?;
        for k in 1..spec.len() {
            let d = decode_row(rt, 0, spec[k - 1], &mut buf)?;
            emit(compare("plain_glue1_vs_plain", plen + k - 1, 0, &plain[k], &d))?;
        }
        rt.set_glue(cli.engine.glue_mode()?)?;
    }

    out.flush()?;

    if cli.tap_pos > 0 {
        let ri = *start
            .get(&cli.tap_pos)
            .with_context(|| format!("no round starts at {}", cli.tap_pos))?;
        let r = &rounds[ri];
        taps(rt, &prompt, &spec, r, &cli)?;
    }
    record.append(&json!({"done": true, "rounds": rounds.len(), "tokens": spec.len()}))?;
    eprintln!("record {}", record.path().display());
    Ok(())
}

/// Speculative greedy decoding of one session in slot 0 (`ink-spec`'s round), keeping every round's rows; `ctx` is the
/// drafter's positions per session.
fn speculate(
    rt: &mut dyn Runtime,
    dr: &mut dyn Drafter,
    prompt: &[i32],
    len: usize,
    ctx: usize,
    buf: &mut Vec<f32>,
) -> Result<Vec<Round>> {
    // the drafter's bounds contract for every call below: positions stay below prompt + len + 8 <= ctx
    ensure!(
        prompt.len() + len + DRAFTS < ctx,
        "the path ({} + {len} tokens) does not fit the drafter's {ctx} positions",
        prompt.len()
    );
    let sp = spec_of(rt)?;
    let bufs = AcceptBufs {
        drafts: dr.drafts(),
        anchor: sp.alloc(4)?,
        pos: sp.alloc(4)?,
        res: sp.alloc(4 * 10)?,
        crow: sp.alloc(4 * 8)?,
        cpos: sp.alloc(4 * 8)?,
        csess: sp.alloc(4 * 8)?,
        lastfeat: sp.alloc(FEAT_ROW_BYTES)?,
        cfeats: sp.alloc(CHUNK * FEAT_ROW_BYTES)?,
    };
    let vids: DevBuf = sp.alloc(4 * 8)?;
    let iota: DevBuf = sp.alloc(4 * CHUNK)?;
    let posb: DevBuf = sp.alloc(4 * CHUNK)?;
    let sess: DevBuf = sp.alloc(4 * CHUNK)?;
    sp.upload(iota, &(0..CHUNK).map(id).collect::<Vec<_>>())?;
    // prefill, every row committed to the drafter (ink-spec Spec::prefill)
    rt.reset_slot(0)?;
    let mut last = Vec::new();
    for (c, piece) in prompt.chunks(CHUNK).enumerate() {
        let a = c * CHUNK;
        let fin = a + CHUNK >= prompt.len();
        let m = piece.len();
        let nl = rt.forward(&[SegIn {
            slot: 0,
            ids: piece,
            logits: usize::from(fin),
        }])?;
        let sp = spec_of(rt)?;
        sp.upload(posb, &(a..a + m).map(id).collect::<Vec<_>>())?;
        sp.upload(sess, &vec![0; m])?;
        let last_row = iota.i32s(m - 1, 1).context("row")?;
        #[allow(unsafe_code)] // the runtime's and the drafter's bounds contracts
        // SAFETY: `iota[i] = i`: rows 0..m of the call of m rows; positions a..a + m < prompt length <= ctx (checked
        // at entry); session 0 of the drafter's sessions.
        unsafe {
            sp.gather_feats(iota, m, bufs.cfeats)?;
            dr.commit_rows(bufs.cfeats, posb, sess, m)?;
            sp.gather_feats(last_row, 1, bufs.lastfeat)?;
        }
        if fin {
            rt.logits_host(nl, &mut last)?;
        }
        rt.commit(&[u32::try_from(m)?])?;
    }
    let tok = id(argmax_first(&last));
    let sp = spec_of(rt)?;
    sp.upload(bufs.anchor, &[tok])?;
    sp.upload(bufs.pos, &[id(prompt.len())])?;
    let mut emitted = 1;
    let mut rounds = Vec::new();
    while emitted < len {
        let pos0 = rt.length(0)?;
        #[allow(unsafe_code)] // the drafter's bounds contract
        // SAFETY: the anchor is the prompt's argmax (< VOCAB) or an emitted token, the position the prompt's length or
        // the accept kernel's row after the committed ones: below prompt + len, so p + 8 <= ctx (checked at entry)
        let state = unsafe { dr.set_state(bufs.anchor, bufs.pos) };
        state?;
        dr.frnn(bufs.lastfeat)?;
        dr.draft()?;
        let sp = spec_of(rt)?;
        sp.build_verify_ids(bufs.anchor, dr.drafts(), vids, 1, DRAFTS)?;
        let mut ids = vec![0; 8];
        sp.download(vids, &mut ids)?;
        #[allow(unsafe_code)] // the runtime's bounds contract
        // SAFETY: the verify ids are the anchor (as above) and the drafter's drafts (vocabulary ids).
        let fwd = unsafe {
            sp.forward_dev(&[DevSeg {
                slot: 0,
                ids: vids,
                rows: 8,
                logits: 8,
            }])
        };
        fwd?;
        let mut res = Vec::new();
        sp.spec_accept(&bufs, DRAFTS, &mut res)?;
        let a = usize::try_from(res[0])?;
        rt.logits_host(8, buf)?;
        let rows = (0..=a).map(|j| row(buf, j).to_vec()).collect();
        // emitted tokens: drafts 0..a-1 equal ids 1..a, then the bonus res[1 + a]
        for j in 0..a {
            if res[1 + j] != ids[1 + j] {
                bail!("accepted token differs from its draft");
            }
        }
        rt.commit(&[u32::try_from(a + 1)?])?;
        let n = usize::try_from(res[9])?;
        #[allow(unsafe_code)] // the drafter's bounds contract
        // SAFETY: the accept kernel's packed rows: positions pos0 .. pos0 + a < prompt + len + 8 <= ctx, session 0.
        let commit = unsafe { dr.commit_rows(bufs.cfeats, bufs.cpos, bufs.csess, n) };
        commit?;
        emitted += a + 1;
        rounds.push(Round {
            pos0,
            ids,
            accepted: a,
            rows,
        });
    }
    Ok(rounds)
}

/// Per-op comparison of every verify row of round `r` against the one-row decode of the same position, both on
/// the decode-built history: slot 1 decodes positions r.pos0 .. r.pos0 + 7 with r.ids (tapped, rows recorded),
/// slot 2 runs the 8-row verify call (tapped, compared; with `--tap-iso` the decode rows are written back).
fn taps(rt: &mut dyn Runtime, prompt: &[i32], spec: &[i32], r: &Round, cli: &Cli) -> Result<()> {
    let plen = prompt.len();
    let mut buf = Vec::new();
    // both slots to the round start along the speculative tokens (no taps)
    for slot in [1, 2] {
        prefill(rt, slot, prompt)?;
        for k in 1..=(r.pos0 - plen) {
            decode_row(rt, slot, spec[k - 1], &mut buf)?;
        }
    }
    // decode rows of the 8 positions, tapped
    let dec: Arc<Mutex<Vec<OpRows>>> = Arc::new(Mutex::new(Vec::new()));
    let mut dec_logits = Vec::new();
    for j in 0..8 {
        let sink = Arc::clone(&dec);
        sink.lock().expect("lock").push(OpRows::new());
        rt.set_tap(Some(Box::new(move |name, t, shape, host, _scratch| {
            let rows = usize::try_from(shape[0]).expect("rows");
            let rowb = host.len() / rows.max(1);
            sink.lock()
                .expect("lock")
                .last_mut()
                .expect("row")
                .insert(name.to_owned(), (t, rowb, host[..rowb].to_vec()));
            0
        })))?;
        dec_logits.push(decode_row(rt, 1, r.ids[j], &mut buf)?);
    }
    rt.set_tap(None)?;
    let dec = Arc::new(dec.lock().expect("lock").clone());
    // verify call, tapped
    let lines: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let iso = cli.tap_iso;
    {
        let dec = Arc::clone(&dec);
        let lines = Arc::clone(&lines);
        let mut order = 0_usize;
        rt.set_tap(Some(Box::new(move |name, t, shape, host, scratch| {
            let rows = usize::try_from(shape[0]).expect("rows");
            let rowb = host.len() / rows.max(1);
            let mut wrote = 0;
            for (j, d) in dec.iter().enumerate().take(rows) {
                let Some((_, rb, a)) = d.get(name) else { continue };
                if *rb != rowb {
                    continue;
                }
                let b = &host[j * rowb..(j + 1) * rowb];
                let (nd, mx, rel, rms) = op_diff(t, rowb, a, b);
                lines
                    .lock()
                    .expect("lock")
                    .push(json!({"order": order, "op": name, "row": j, "ndiff": nd,
                    "n": rowb / elem_bytes(t), "max_abs": mx, "max_rel": rel, "rms_ref": rms}));
                if iso {
                    if wrote == 0 {
                        scratch.copy_from_slice(host);
                    }
                    scratch[j * rowb..(j + 1) * rowb].copy_from_slice(a);
                    wrote = host.len();
                }
            }
            order += 1;
            wrote
        })))?;
    }
    let vrows = verify_rows(rt, 2, &r.ids, 0, &mut buf)?;
    rt.set_tap(None)?;
    let mut f = BufWriter::new(File::create(&cli.tap_out)?);
    for l in lines.lock().expect("lock").iter() {
        writeln!(f, "{l}")?;
    }
    for j in 0..8 {
        writeln!(
            f,
            "{}",
            compare(
                if iso { "tap_iso_logits" } else { "tap_chain_logits" },
                r.pos0 + j,
                j,
                &dec_logits[j],
                &vrows[j]
            )
        )?;
    }
    f.flush()?;
    Ok(())
}
