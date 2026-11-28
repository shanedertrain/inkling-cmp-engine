//! GPU parity, rollback, and latency benchmark harness for hybrid `RustDFlash2` (`DFlash2-Global`).
//!
//! Run on GPU 1 inside the CUDA container:
//! ```text
//! INK_MODEL_DIR=/model INK_DF2_DIR=/df2 INK_PARITY_DIR=/parity \
//!   cargo test --release -p ink-runtime --test dflash2_global_gpu -- --ignored --nocapture
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ink_rt_api::{
    DraftConfig, Drafter, EngineConfig, FEAT_ROW_BYTES, Graphs, HeadPrecision, KvPrecision, Runtime, SegIn,
    SpecRuntime, VerifyWidth,
};
use ink_runtime::{Engine, RustDFlash2};

fn env_path(k: &str) -> PathBuf {
    PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set")))
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Commits pre-generated BF16 features from `INK_PARITY_DIR/feats.bin` (`[pos, 4*H]` bf16), runs one eager draft at
/// `INK_PARITY_POS` with `INK_PARITY_ANCHOR`, and dumps per-layer `attn_out`, `logits`, `cand_ids`, and `drafts` to
/// `INK_PARITY_OUT`. Also verifies rollback parity (`commit pos+16 -> rollback to pos -> identical drafts`).
#[test]
#[ignore = "needs GPU + INK_MODEL_DIR + INK_DF2_DIR + INK_PARITY_DIR"]
fn dflash2_global_parity_dump() {
    let model_dir = env_path("INK_MODEL_DIR");
    let df2_dir = env_path("INK_DF2_DIR");
    let parity_dir = env_path("INK_PARITY_DIR");
    let out_dir = env_path("INK_PARITY_OUT");
    fs::create_dir_all(&out_dir).expect("mkdir out_dir");

    let pos: usize = std::env::var("INK_PARITY_POS")
        .expect("INK_PARITY_POS")
        .parse()
        .expect("pos usize");
    let anchor: i32 = std::env::var("INK_PARITY_ANCHOR")
        .unwrap_or_else(|_| "1234".into())
        .parse()
        .expect("anchor i32");
    let max_ctx = (pos + 128).max(4096) as u32;

    let feats_all = fs::read(parity_dir.join("feats.bin")).expect("read feats.bin");
    let extra: usize = 16;
    assert!(
        feats_all.len() >= (pos + extra) * FEAT_ROW_BYTES,
        "feats.bin too short: {} bytes for {} rows",
        feats_all.len(),
        pos + extra
    );

    let cfg = DraftConfig {
        model_dir,
        fold_bin: PathBuf::new(),
        dflash2_dir: Some(df2_dir),
        sessions: 1,
        max_ctx,
        head: HeadPrecision::Int8,
        global_kv: KvPrecision::Bf16,
        verify: VerifyWidth::Full,
        lookup: 0,
        lookup_min: 8,
    };
    let mut d = RustDFlash2::new(&cfg).expect("RustDFlash2::new");

    let chunk = 2048usize;
    let b_feats = d.alloc(chunk * FEAT_ROW_BYTES).expect("alloc feats");
    let b_pos = d.alloc(chunk * 4).expect("alloc pos");
    let b_sess = d.alloc(chunk * 4).expect("alloc sess");
    let b_anchor = d.alloc(4).expect("alloc anchor");
    let b_apos = d.alloc(4).expect("alloc apos");

    let commit_range = |d: &mut RustDFlash2, start: usize, end: usize| {
        for r0 in (start..end).step_by(chunk) {
            let rc = (end - r0).min(chunk);
            let slice = &feats_all[r0 * FEAT_ROW_BYTES..(r0 + rc) * FEAT_ROW_BYTES];
            let ps: Vec<i32> = (r0..r0 + rc).map(|p| p as i32).collect();
            let ss = vec![0i32; rc];
            d.upload(b_feats, slice).expect("up feats");
            d.upload(b_pos, &i32_bytes(&ps)).expect("up pos");
            d.upload(b_sess, &i32_bytes(&ss)).expect("up sess");
            // SAFETY: buffers sized for chunk rows; ps < max_ctx; ss < sessions.
            #[allow(unsafe_code)]
            unsafe {
                d.commit_rows(b_feats, b_pos, b_sess, rc).expect("commit_rows");
            }
        }
    };

    commit_range(&mut d, 0, pos);
    d.upload(b_anchor, &i32_bytes(&[anchor])).expect("up anchor");
    d.upload(b_apos, &i32_bytes(&[pos as i32])).expect("up apos");
    // SAFETY: anchor < VALID_VOCAB, pos + BLK <= max_ctx.
    #[allow(unsafe_code)]
    unsafe {
        d.set_state(b_anchor, b_apos).expect("set_state");
    }
    d.draft_with_dump(&[0], &out_dir).expect("draft_with_dump");
    d.sync().expect("sync");

    let mut d_before = vec![0u8; 7 * 4];
    d.download(d.drafts(), &mut d_before).expect("dl drafts");

    // Verify rollback semantics: commit 16 extra rows past `pos`, roll back `pos` to `pos`, draft again -> bit-identical!
    commit_range(&mut d, pos, pos + extra);
    // SAFETY: anchor < VALID_VOCAB, pos + BLK <= max_ctx.
    #[allow(unsafe_code)]
    unsafe {
        d.set_state(b_anchor, b_apos).expect("set_state rollback");
    }
    d.draft_sessions(&[0]).expect("draft after rollback");
    d.sync().expect("sync");
    let mut d_after = vec![0u8; 7 * 4];
    d.download(d.drafts(), &mut d_after).expect("dl drafts after rollback");
    assert_eq!(d_before, d_after, "rollback draft mismatch at pos={pos}");
    eprintln!("[parity] pos={pos} dump complete & rollback bit-identical PASS");
}

/// Benchmarks graphed `RustDFlash2` latency (`draft_ms`) at configurable `(sessions, ctx, k, iters)`.
/// Set `INK_DF2_PROF=1` for per-layer kernel timing breakdown (`l0_attn` vs `l2_global`).
#[test]
#[ignore = "needs GPU + INK_MODEL_DIR + INK_DF2_DIR"]
fn dflash2_global_bench() {
    let model_dir = env_path("INK_MODEL_DIR");
    let df2_dir = env_path("INK_DF2_DIR");
    let sessions: u32 = std::env::var("INK_BENCH_SESSIONS")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .expect("sessions");
    let max_ctx: u32 = std::env::var("INK_BENCH_MAX_CTX")
        .unwrap_or_else(|_| "102400".into())
        .parse()
        .expect("max_ctx");
    let ctx: usize = std::env::var("INK_BENCH_CTX")
        .unwrap_or_else(|_| "45000".into())
        .parse()
        .expect("ctx");
    let k: usize = std::env::var("INK_BENCH_K")
        .unwrap_or_else(|_| "4".into())
        .parse()
        .expect("k");
    let iters: usize = std::env::var("INK_BENCH_ITERS")
        .unwrap_or_else(|_| "200".into())
        .parse()
        .expect("iters");

    let cfg = DraftConfig {
        model_dir,
        fold_bin: PathBuf::new(),
        dflash2_dir: Some(df2_dir),
        sessions,
        max_ctx,
        head: HeadPrecision::Int8,
        global_kv: KvPrecision::Bf16,
        verify: VerifyWidth::Full,
        lookup: 0,
        lookup_min: 8,
    };
    let mut d = RustDFlash2::new(&cfg).expect("RustDFlash2::new");

    // Populate KV cache up to `ctx` for active sessions 0..k with deterministic non-zero BF16 features
    let chunk = 2048usize;
    let b_feats = d.alloc(chunk * FEAT_ROW_BYTES).expect("alloc feats");
    let b_pos = d.alloc(chunk * 4).expect("alloc pos");
    let b_sess = d.alloc(chunk * 4).expect("alloc sess");
    let mut feat_buf = vec![0u8; chunk * FEAT_ROW_BYTES];
    for (i, b) in feat_buf.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        // Deterministic finite BF16 values in [-0.5, 0.5]
        let v = (((i.wrapping_mul(2654435761) >> 24) as i32 - 128) as f32) * (1.0 / 256.0);
        let bits = (v.to_bits() >> 16) as u16;
        b.copy_from_slice(&bits.to_le_bytes());
    }
    d.upload(b_feats, &feat_buf).expect("up feats");
    for s in 0..k {
        for r0 in (0..ctx).step_by(chunk) {
            let rc = (ctx - r0).min(chunk);
            let ps: Vec<i32> = (r0..r0 + rc).map(|p| p as i32).collect();
            let ss = vec![s as i32; rc];
            d.upload(b_pos, &i32_bytes(&ps)).expect("up pos");
            d.upload(b_sess, &i32_bytes(&ss)).expect("up sess");
            // SAFETY: buffers sized for chunk rows; ps < max_ctx; ss < sessions.
            #[allow(unsafe_code)]
            unsafe {
                d.commit_rows(b_feats, b_pos, b_sess, rc).expect("commit_rows");
            }
        }
    }
    let anchors = vec![1024i32; sessions as usize];
    let positions = vec![ctx as i32; sessions as usize];
    let b_anchor = d.alloc(sessions as usize * 4).expect("alloc anchor");
    let b_apos = d.alloc(sessions as usize * 4).expect("alloc apos");
    d.upload(b_anchor, &i32_bytes(&anchors)).expect("up anchor");
    d.upload(b_apos, &i32_bytes(&positions)).expect("up apos");
    // SAFETY: anchors < VALID_VOCAB, ctx + BLK <= max_ctx.
    #[allow(unsafe_code)]
    unsafe {
        d.set_state(b_anchor, b_apos).expect("set_state");
    }
    let act: Vec<usize> = (0..k).collect();
    // Warmup + graph capture (also prints INK_DF2_PROF breakdown if set)
    for _ in 0..20 {
        d.draft_sessions(&act).expect("warmup");
    }
    d.sync().expect("sync warmup");

    let t0 = Instant::now();
    for _ in 0..iters {
        d.draft_sessions(&act).expect("bench");
    }
    d.sync().expect("sync bench");
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    eprintln!("[df2 bench] k={k} ctx={ctx} sessions={sessions} max_ctx={max_ctx} iters={iters} draft_ms={ms:.3}");
}

/// Shared helper: builds an 8-slot, 504-page `Engine`, creates either `RustDFlash2` (`dflash2_dir = Some`) or pooled
/// E4M3 `RustDrafter` (`dflash2_dir = None`) at `max_ctx = 262_144`, exercises page recycling (`reset_slot`), prefills
/// slot 5 to `pos = 246_770` (14 tokens before a 1,024-token page boundary `241 * 1024 = 246_784`), drafts, crosses the
/// page boundary with +32 committed tokens, rolls back to `246_770`, and asserts bit-identical drafts after rollback.
fn run_246k_pooled_rollback(dflash2_dir: Option<PathBuf>, fold_bin: PathBuf, label: &str) {
    let model_dir = env_path("INK_MODEL_DIR");
    let gemm_algos = env_path("INK_GEMM_ALGOS");
    let target_pos: usize = std::env::var("INK_246K_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(246_770);
    let extra: usize = 32; // crosses 241 * 1024 = 246,784 page boundary

    let ecfg = EngineConfig {
        model_dir: model_dir.clone(),
        gemm_algos,
        max_slots: 8,
        num_pages: 504,
        t_max: 2048 + 128,
        max_chunk: 2048,
        graphs: Graphs::On,
        ..EngineConfig::default()
    };
    let mut eng = Engine::new(&ecfg).expect("Engine::new");
    let dcfg = DraftConfig {
        model_dir,
        fold_bin,
        dflash2_dir,
        sessions: 8,
        max_ctx: 262_144,
        head: HeadPrecision::Int8,
        global_kv: KvPrecision::E4m3,
        verify: VerifyWidth::Full,
        lookup: 0,
        lookup_min: 8,
    };
    let mut dr = eng.create_drafter(&dcfg).expect("create_drafter");

    let chunk = 2048usize;
    let b_feats = eng.alloc(chunk * FEAT_ROW_BYTES).expect("alloc feats");
    let b_rows = eng.alloc(chunk * 4).expect("alloc rows");
    let b_pos = eng.alloc(chunk * 4).expect("alloc pos");
    let b_sess = eng.alloc(chunk * 4).expect("alloc sess");
    let b_last = eng.alloc(8 * FEAT_ROW_BYTES).expect("alloc lastfeat");
    let b_anchor = eng.alloc(8 * 4).expect("alloc anchor");
    let b_apos = eng.alloc(8 * 4).expect("alloc apos");
    let ramp: Vec<i32> = (0..chunk as i32).collect();
    eng.upload(b_rows, &ramp).expect("up ramp");

    // 1. Exercise page allocation + recycling on slot 3 before slot 5 grows to 246K.
    let warmup_toks: Vec<i32> = (0..4096).map(|i| 100 + (i % 5000)).collect();
    for r0 in (0..4096).step_by(chunk) {
        eng.forward(&[SegIn {
            slot: 3,
            ids: &warmup_toks[r0..r0 + chunk],
            logits: 0,
        }])
        .expect("warmup fwd");
        eng.commit(&[chunk as u32]).expect("warmup commit");
    }
    eng.reset_slot(3).expect("reset slot 3");
    assert_eq!(eng.free_pages().expect("free_pages"), 504);

    // Helper: prefill `[start..end)` into slot 5 and commit rows to drafter.
    let prefill_and_commit =
        |eng: &mut Engine, dr: &mut Box<dyn Drafter>, start: usize, end: usize, save_last: bool| -> i32 {
            let mut last_tok = 1000i32;
            for r0 in (start..end).step_by(chunk) {
                let rc = (end - r0).min(chunk);
                let toks: Vec<i32> = (r0..r0 + rc).map(|p| 200 + ((p * 37) % 10_000) as i32).collect();
                let want_logits = usize::from(r0 + rc == end);
                eng.forward(&[SegIn {
                    slot: 5,
                    ids: &toks,
                    logits: want_logits,
                }])
                .expect("prefill fwd");
                if want_logits > 0 {
                    let mut am = Vec::new();
                    eng.argmax_host(1, &mut am).expect("argmax");
                    last_tok = am[0];
                }
                eng.commit(&[rc as u32]).expect("prefill commit");
                let ps: Vec<i32> = (r0..r0 + rc).map(|p| p as i32).collect();
                let ss = vec![5i32; rc];
                eng.upload(b_pos.i32s(0, rc).unwrap(), &ps).expect("up pos");
                eng.upload(b_sess.i32s(0, rc).unwrap(), &ss).expect("up sess");
                // SAFETY: b_rows holds 0..rc < call rows; ps < max_ctx; ss = 5 < sessions.
                #[allow(unsafe_code)]
                unsafe {
                    eng.gather_feats(b_rows.i32s(0, rc).unwrap(), rc, b_feats)
                        .expect("gather_feats");
                    dr.commit_rows(b_feats, b_pos.i32s(0, rc).unwrap(), b_sess.i32s(0, rc).unwrap(), rc)
                        .expect("dr.commit_rows");
                    if save_last && want_logits > 0 {
                        let dst = b_last.range(5 * FEAT_ROW_BYTES, FEAT_ROW_BYTES).unwrap();
                        eng.gather_feats(b_rows.i32s(rc - 1, 1).unwrap(), 1, dst)
                            .expect("gather lastfeat");
                    }
                }
            }
            last_tok
        };

    let t0 = Instant::now();
    let anchor_tok = prefill_and_commit(&mut eng, &mut dr, 0, target_pos, true);
    eng.checkpoints()
        .expect("checkpoints")
        .checkpoint(5)
        .expect("checkpoint");
    let pages_at_pos = 504 - eng.free_pages().expect("free_pages");
    assert_eq!(pages_at_pos, target_pos.div_ceil(1024));

    let mut anchors = vec![100i32; 8];
    let mut positions = vec![0i32; 8];
    anchors[5] = anchor_tok;
    positions[5] = target_pos as i32;
    eng.upload(b_anchor, &anchors).expect("up anchor");
    eng.upload(b_apos, &positions).expect("up apos");
    dr.frnn(b_last).expect("frnn");
    // SAFETY: anchors < VOCAB, positions <= max_ctx.
    #[allow(unsafe_code)]
    unsafe {
        dr.set_state(b_anchor, b_apos).expect("set_state");
    }
    // Draft once before crossing the page boundary (also verifies graph capture at 246K!).
    dr.draft_sessions(&[5]).expect("draft before");
    let mut drafts_before = vec![0i32; 8 * 7];
    eng.download(dr.drafts(), &mut drafts_before).expect("dl drafts before");
    let slot5_before = drafts_before[5 * 7..6 * 7].to_vec();
    assert!(slot5_before.iter().all(|&t| (0..200_058).contains(&t)));

    // Cross the 1,024-token page boundary by committing `extra` tokens past `target_pos`.
    let _ = prefill_and_commit(&mut eng, &mut dr, target_pos, target_pos + extra, false);
    let pages_after_extra = 504 - eng.free_pages().expect("free_pages");
    assert_eq!(pages_after_extra, (target_pos + extra).div_ceil(1024));
    assert!(
        pages_after_extra > pages_at_pos,
        "must cross a 1,024-token page boundary"
    );

    // Roll back to `target_pos` (frees the crossed page!), re-sync anchor/pos/f_rnn, and draft again.
    eng.checkpoints()
        .expect("checkpoints")
        .rollback(5, target_pos)
        .expect("rollback");
    assert_eq!(504 - eng.free_pages().expect("free_pages"), pages_at_pos);

    eng.upload(b_anchor, &anchors).expect("up anchor rollback");
    eng.upload(b_apos, &positions).expect("up apos rollback");
    dr.frnn(b_last).expect("frnn rollback");
    // SAFETY: anchors < VOCAB, positions <= max_ctx.
    #[allow(unsafe_code)]
    unsafe {
        dr.set_state(b_anchor, b_apos).expect("set_state rollback");
    }
    dr.draft_sessions(&[5]).expect("draft after rollback");
    let mut drafts_after = vec![0i32; 8 * 7];
    eng.download(dr.drafts(), &mut drafts_after).expect("dl drafts after");
    let slot5_after = drafts_after[5 * 7..6 * 7].to_vec();
    assert_eq!(
        slot5_before, slot5_after,
        "[{label}] rollback draft mismatch at pos={target_pos}: before={slot5_before:?} after={slot5_after:?}"
    );
    eng.reset_slot(5).expect("reset slot 5");
    assert_eq!(eng.free_pages().expect("free_pages"), 504);
    eprintln!(
        "[{label}] 246K pooled-paging + page-boundary rollback PASS in {:.2}s (pos={target_pos}, pages {pages_at_pos}->{pages_after_extra}->{pages_at_pos}, drafts={slot5_before:?})",
        t0.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "needs GPU + INK_MODEL_DIR + INK_GEMM_ALGOS + INK_DF2_DIR"]
fn dflash2_global_246k_pooled_rollback() {
    run_246k_pooled_rollback(Some(env_path("INK_DF2_DIR")), PathBuf::new(), "dflash2-global");
}

#[test]
#[ignore = "needs GPU + INK_MODEL_DIR + INK_GEMM_ALGOS + INK_FOLD_BIN"]
fn dspark_246k_pooled_rollback() {
    run_246k_pooled_rollback(None, env_path("INK_FOLD_BIN"), "dspark-pooled-kv8");
}
