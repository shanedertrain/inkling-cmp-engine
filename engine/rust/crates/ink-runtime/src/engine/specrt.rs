// Ported from engine/native/src/engine.cu @ 1c7ac46 (`spec_accept`, `build_verify_ids`, `gather_feats`, the
// `dev_ids` segments of `Engine::forward`) and engine/native/capi/ink_rt_capi.cpp (`ink_rt_dev_alloc`, `_h2d`, `_d2h`,
// the checks of `ink_rt_forward`, `ink_rt_spec_accept`, `ink_rt_build_verify_ids`, `ink_rt_gather_feats`,
// `ink_draft_create`). Own code of this project.
//! Device-side speculation of the Rust engine (package R9): [`SpecRuntime`] for [`Engine`].
//!
//! The drafter shares the engine's context and stream ([`SpecRuntime::create_drafter`] hands it `Rc` clones of
//! both, as the C++ `DraftRT` runs on `Engine::stream()`): every drafter call is ordered with the forward calls
//! without events or host waits. It keeps its own arena: the engine's arena is planned before any drafter exists and
//! its map is gate E1's, equal to the C++ engine's (`ink-model::plan`), and the drafter is optional and sized by its
//! own configuration (as the C++ allocates it separately). Device ranges ([`SpecRuntime::alloc`], the drafts) are
//! handles of one registry the engine and its drafter share (`devbufs.rs`).

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Instant;

use ink_cuda::{Arena, DevBuf, Stream};
use ink_kernels::ink::SampleParam;
use ink_model::dims::{VOCAB, VOCAB_PAD};
use ink_rt_api::{
    AcceptBufs, DRAFTS, DevBuf as ApiBuf, DevSeg, DeviceSampling, DraftConfig, Drafter, FEAT_ROW_BYTES, MAX_DRAFTS,
    ProposalBufs, Runtime, RuntimeError, SampleTestOut, SampleTestRows, SamplerProbe, SegIn, SpecRuntime,
};

use super::meta::CallMeta;
use super::ops::Dev;
use super::{Engine, Res, Rt};
use crate::devbufs::DevBufs;
use crate::dflash2::{RustDFlash2, SharedTargetBufs};
use crate::draft::RustDrafter;

/// Reusable device scratch of [`SamplerProbe::sample_test_rows`] (outside the planned engine arena).
#[derive(Debug)]
struct ProbeScratch {
    cap_rows: usize,
    cap_par: usize,
    cap_keys: usize,
    t_lrow: DevBuf<i32>,
    t_par: DevBuf<i32>,
    t_pos: DevBuf<i32>,
    t_out: DevBuf<i32>,
    t_par_tbl: DevBuf<SampleParam>,
    t_stats: DevBuf<u64>,
    t_keys: Option<DevBuf<u32>>,
    _arena: Arena,
}

/// The engine's speculation state.
#[derive(Debug, Default)]
pub(super) struct SpecState {
    /// Device buffers handed out as ranges, shared with the drafter.
    bufs: Rc<RefCell<DevBufs>>,
    /// Per segment of the call being made by `forward_dev`: its device token ids.
    dev_ids: Vec<Option<DevBuf<i32>>>,
    /// The drafter was created (one per engine, as the C ABI).
    drafter: bool,
    /// Device bytes the drafter's graphs took (one per batch size, 0 until captured): part of the engine's graph reserve.
    draft_graph: Rc<Cell<usize>>,
    /// The device sampler's controls of the last call's segments (`max_slots` entries, own allocation as the C ABI's
    /// `cudaMalloc`, outside the planned arena), allocated by the first sampled call.
    samp: Option<(Arena, DevBuf<SampleParam>)>,
    /// Lazily allocated scratch for [`SamplerProbe::sample_test_rows`].
    probe: Option<ProbeScratch>,
}

/// `ink::sample_param` (engine/native/src/sample.hpp), bit for bit: the kernel's controls from the request's.
fn sample_param(p: &DeviceSampling) -> SampleParam {
    let tp = f64::from(p.top_p);
    let p24 = if tp.is_nan() || tp >= 1.0 {
        1 << 24
    } else if tp <= 0.0 {
        0
    } else {
        // llround: half away from zero, as f64::round; at most 2^24
        (tp * 16_777_216.0).round() as u32
    };
    let mp = if p.min_p.is_nan() || p.min_p <= 0.0 {
        0.0
    } else if p.min_p >= 1.0 {
        1.0
    } else {
        f64::from(p.min_p)
    };
    SampleParam {
        seed: p.seed,
        wmin: ((mp * 1_099_511_627_776.0).ceil() as u64).max(1),
        temperature: p.temperature,
        p24,
        top_k: i32::try_from(p.top_k).unwrap_or(i32::MAX),
        greedy: i32::from(p.temperature.is_nan() || p.temperature < 1e-5),
    }
}

impl SpecState {
    /// Device bytes the drafter's captured graph holds.
    pub(super) fn draft_graph_bytes(&self) -> usize {
        self.draft_graph.get()
    }

    /// The `dev_ids` copies of `Engine::forward`: each device-id segment's ids into the call's id buffer at its first
    /// row, after the host ids (zeros for those rows) were uploaded.
    pub(super) fn copy_dev_ids(&self, st: &Stream, ids: &DevBuf<i32>, m: &CallMeta) -> Res<()> {
        for (c, d) in m.call.iter().zip(&self.dev_ids) {
            if let Some(d) = d {
                st.dtod(&ids.slice(c.row0..c.row0 + c.m).rt()?, d).rt()?;
            }
        }
        Ok(())
    }
}

impl Engine {
    fn spec_dev(&self) -> Dev<'_> {
        Dev {
            k: &self.k,
            st: &self.st,
        }
    }

    /// `Engine::sample(par)` behind `ink_rt_sample_host` / `ink_rt_spec_accept_sampled`: `params` (one per segment of
    /// the uncommitted last call) to the device, then the sampled token of every logit row into `amax`.
    pub(super) fn sample_rows(&mut self, params: &[DeviceSampling], what: &str) -> Res<()> {
        let Some(m) = self.last.as_ref().filter(|_| !self.committed) else {
            return Err(RuntimeError::request(format!("{what}: no uncommitted forward call")));
        };
        if params.len() != m.call.len() {
            return Err(RuntimeError::request(format!(
                "{what}: {} sampling controls, the last forward call had {} segments",
                params.len(),
                m.call.len()
            )));
        }
        let (t, nlog, nseg) = (m.t, m.lrows.len(), m.call.len());
        if self.spec.samp.is_none() {
            let n = self.mem.max_slots;
            let mut arena = Arena::new(&self.ctx, &self.st, n * size_of::<SampleParam>() + 256).rt()?;
            let buf = arena.take::<SampleParam>(n, "sampling controls").rt()?;
            self.spec.samp = Some((arena, buf));
        }
        let (_, par) = self.spec.samp.as_ref().expect("allocated above");
        let host: Vec<SampleParam> = params.iter().map(sample_param).collect();
        self.st.htod(&par.slice(0..nseg).rt()?, &host).rt()?;
        if nlog == 0 {
            return Ok(());
        }
        let a = &self.a;
        self.spec_dev()
            .sample(nlog, t, nseg, &a.logits, par, &a.logit_rows, (&a.pos, &a.seg), &a.amax)
            .rt()
    }

    /// `ink_rt_spec_accept` / `ink_rt_spec_accept_sampled` / `spec_accept_leviathan`: argmax, coupled-seed sampled, or
    /// exact Leviathan rejection+residual sampled rows, then the standard accept kernel over those tokens.
    fn accept_rows(
        &mut self,
        bufs: &AcceptBufs,
        prop: Option<&ProposalBufs>,
        k: usize,
        params: Option<&[DeviceSampling]>,
        res: &mut Vec<i32>,
    ) -> Res<()> {
        if !(1..=MAX_DRAFTS).contains(&k) {
            return Err(RuntimeError::request(format!("spec_accept: k must be 1..{MAX_DRAFTS}")));
        }
        let Some(m) = self.last.as_ref().filter(|_| !self.committed) else {
            return Err(RuntimeError::request("spec_accept: no uncommitted forward call"));
        };
        let mut top = 0;
        for c in &m.call {
            if c.m == 0 || c.m > k + 1 || c.nlog != c.m {
                return Err(RuntimeError::request(
                    "spec_accept: every segment of the last call must have 1..k + 1 rows, all with logits",
                ));
            }
            top = top.max(c.slot + 1);
        }
        let (s, t, nlog) = (m.call.len(), m.t, m.lrows.len());
        let rs = k + 2;
        // buffer extents of the widest call (the C ABI's check): every segment at most k + 1 rows
        let r = s * (k + 1);
        let b = self.spec.bufs.clone();
        let b = b.borrow();
        let drafts = b.typed::<i32>(bufs.drafts, top * k, "drafts")?;
        let anchor = b.typed::<i32>(bufs.anchor, top, "anchor")?;
        let p = b.typed::<i32>(bufs.pos, top, "P")?;
        let res_dev = b.typed::<i32>(bufs.res, rs * s + 1, "res_dev")?;
        let crow = b.typed::<i32>(bufs.crow, r, "crow")?;
        let cpos = b.typed::<i32>(bufs.cpos, r, "cpos")?;
        let csess = b.typed::<i32>(bufs.csess, r, "csess")?;
        let lastfeat = b.typed::<ink_cuda::Bf16>(bufs.lastfeat, top * FEAT_ROW_BYTES / 2, "lastfeat")?;
        let cfeats = b.typed::<ink_cuda::Bf16>(bufs.cfeats, r * FEAT_ROW_BYTES / 2, "cfeats")?;
        match (params, prop) {
            (Some(params), Some(pr)) => {
                let prop_drafts = b.typed::<i32>(pr.drafts, top * DRAFTS, "prop_drafts")?;
                let prop_ids = b.typed::<i32>(pr.ids, top * DRAFTS * 16, "prop_ids")?;
                let prop_w = b.typed::<u32>(pr.weights, top * DRAFTS * 16, "prop_w")?;
                drop(b);
                if params.len() != s {
                    return Err(RuntimeError::request(format!(
                        "spec_accept_leviathan: {} sampling controls, last call had {s} segments",
                        params.len()
                    )));
                }
                if self.spec.samp.is_none() {
                    let n = self.mem.max_slots;
                    let mut arena = Arena::new(&self.ctx, &self.st, n * size_of::<SampleParam>() + 256).rt()?;
                    let buf = arena.take::<SampleParam>(n, "sampling controls").rt()?;
                    self.spec.samp = Some((arena, buf));
                }
                let (_, par) = self.spec.samp.as_ref().expect("allocated above");
                let host: Vec<SampleParam> = params.iter().map(sample_param).collect();
                self.st.htod(&par.slice(0..s).rt()?, &host).rt()?;
                let a = &self.a;
                self.spec_dev()
                    .sample_leviathan(
                        nlog,
                        t,
                        s,
                        top,
                        &a.logits,
                        par,
                        &a.logit_rows,
                        (&a.pos, &a.seg, &a.row0, &a.slot),
                        &prop_drafts,
                        &prop_ids,
                        &prop_w,
                        &a.amax,
                    )
                    .rt()?;
            }
            (Some(params), None) => {
                drop(b);
                self.sample_rows(params, "ink_rt_spec_accept_sampled")?;
            }
            (None, _) => {
                drop(b);
                self.spec_dev().argmax(nlog, &self.a.logits, &self.a.amax).rt()?;
            }
        }
        let (a, dev) = (&self.a, self.spec_dev());
        let rows = (&a.row0, &a.slot, &a.pos);
        dev.accept(
            t,
            s,
            k,
            &a.amax,
            &drafts,
            rows,
            &anchor,
            &p,
            &res_dev,
            (&crow, &cpos, &csess),
        )
        .rt()?;
        dev.last_feat(s, t, rs, &a.feats, (&a.row0, &a.slot), &res_dev, &lastfeat)
            .rt()?;
        let count = res_dev.slice(rs * s..rs * s + 1).rt()?;
        dev.gather_feats(r, t, &a.feats, &crow, Some(&count), &cfeats).rt()?;
        res.clear();
        res.resize(rs * s + 1, 0);
        self.st.dtoh(res, &res_dev).rt()
    }
}

impl SpecRuntime for Engine {
    fn alloc(&mut self, bytes: usize) -> Res<ApiBuf> {
        self.spec.bufs.borrow_mut().alloc(&self.ctx, &self.st, bytes)
    }

    fn upload(&mut self, dst: ApiBuf, src: &[i32]) -> Res<()> {
        let d = self
            .spec
            .bufs
            .borrow()
            .typed::<i32>(dst, src.len(), "upload destination")?;
        self.st.htod(&d, src).rt()
    }

    fn download(&mut self, src: ApiBuf, dst: &mut [i32]) -> Res<()> {
        let s = self
            .spec
            .bufs
            .borrow()
            .typed::<i32>(src, dst.len(), "download source")?;
        self.st.dtoh(dst, &s).rt()
    }

    #[allow(unsafe_code)] // implements the trait's contract method (ink-rt-api bounds contract)
    unsafe fn forward_dev(&mut self, segs: &[DevSeg]) -> Res<usize> {
        let mut ids = Vec::with_capacity(segs.len());
        for s in segs {
            ids.push(Some(self.spec.bufs.borrow().typed::<i32>(
                s.ids,
                s.rows,
                "segment ids",
            )?));
        }
        let zeros: Vec<Vec<i32>> = segs.iter().map(|s| vec![0; s.rows]).collect();
        let host: Vec<SegIn<'_>> = segs
            .iter()
            .zip(&zeros)
            .map(|(s, z)| SegIn {
                slot: s.slot,
                ids: z,
                logits: s.logits,
            })
            .collect();
        self.spec.dev_ids = ids;
        let n = Runtime::forward(self, &host);
        self.spec.dev_ids.clear();
        n
    }

    fn build_verify_ids(&mut self, anchor: ApiBuf, drafts: ApiBuf, vids: ApiBuf, n: usize, k: usize) -> Res<()> {
        if n == 0 || n > self.mem.max_slots {
            return Err(RuntimeError::request(format!(
                "build_verify_ids: n must be 1..{}",
                self.mem.max_slots
            )));
        }
        if !(1..=MAX_DRAFTS).contains(&k) {
            return Err(RuntimeError::request(format!(
                "build_verify_ids: k must be 1..{MAX_DRAFTS}"
            )));
        }
        let b = self.spec.bufs.borrow();
        let a = b.typed::<i32>(anchor, n, "anchor")?;
        let d = b.typed::<i32>(drafts, n * k, "drafts")?;
        let v = b.typed::<i32>(vids, n * (k + 1), "vids")?;
        self.spec_dev().verify_ids(n, k, &a, &d, &v).rt()
    }

    fn spec_accept(&mut self, bufs: &AcceptBufs, k: usize, res: &mut Vec<i32>) -> Res<()> {
        self.accept_rows(bufs, None, k, None, res)
    }

    fn accepts_sampled(&self) -> bool {
        true
    }

    fn spec_accept_sampled(
        &mut self,
        bufs: &AcceptBufs,
        k: usize,
        params: &[DeviceSampling],
        res: &mut Vec<i32>,
    ) -> Res<()> {
        self.accept_rows(bufs, None, k, Some(params), res)
    }

    fn spec_accept_leviathan(
        &mut self,
        bufs: &AcceptBufs,
        prop: &ProposalBufs,
        k: usize,
        params: &[DeviceSampling],
        res: &mut Vec<i32>,
    ) -> Res<()> {
        self.accept_rows(bufs, Some(prop), k, Some(params), res)
    }

    fn verify_widths(&mut self, conf: ApiBuf, tau: f32, n: usize, widths: ApiBuf, out: &mut Vec<i32>) -> Res<()> {
        if n == 0 || n > self.mem.max_slots {
            return Err(RuntimeError::request(format!(
                "verify_widths: n must be 1..{}",
                self.mem.max_slots
            )));
        }
        let b = self.spec.bufs.borrow();
        let c = b.typed::<f32>(conf, n * DRAFTS, "conf")?;
        let w = b.typed::<i32>(widths, n, "widths")?;
        self.spec_dev().verify_widths(n, &c, tau, &w).rt()?;
        out.clear();
        out.resize(n, 0);
        self.st.dtoh(out, &w).rt()
    }

    #[allow(unsafe_code)] // implements the trait's contract method (ink-rt-api bounds contract)
    unsafe fn gather_feats(&mut self, rows: ApiBuf, n: usize, dst: ApiBuf) -> Res<()> {
        if n == 0 {
            return Ok(());
        }
        let b = self.spec.bufs.borrow();
        let r = b.typed::<i32>(rows, n, "rows")?;
        let d = b.typed::<ink_cuda::Bf16>(dst, n * FEAT_ROW_BYTES / 2, "features")?;
        // the feature rows of the last call: rows below its T (the caller's)
        self.spec_dev()
            .gather_feats(n, self.mem.t_max, &self.a.feats, &r, None, &d)
            .rt()
    }

    fn create_drafter(&mut self, config: &DraftConfig) -> Res<Box<dyn Drafter>> {
        if self.spec.drafter {
            return Err(RuntimeError::request("this engine has a drafter already"));
        }
        if config.sessions == 0 || config.sessions as usize > self.mem.max_slots {
            return Err(RuntimeError::request(format!(
                "drafter sessions must be 1..{}",
                self.mem.max_slots
            )));
        }
        let d: Box<dyn Drafter> = if config.dflash2_dir.is_some() {
            Box::new(RustDFlash2::on(
                self.ctx.clone(),
                self.st.clone(),
                self.spec.bufs.clone(),
                self.spec.draft_graph.clone(),
                config,
                Some(SharedTargetBufs {
                    embed: self.w.embed.clone(),
                    unembed: self.w.unembed.clone(),
                }),
            )?)
        } else {
            Box::new(RustDrafter::on(
                self.ctx.clone(),
                self.st.clone(),
                self.spec.bufs.clone(),
                self.spec.draft_graph.clone(),
                config,
            )?)
        };
        let unheld = self.graph_mem.saturating_sub(self.graphs.held()) as u64;
        super::check_free(
            &self.ctx,
            unheld + super::HEADROOM,
            self.mem.kv_fp8,
            "the engine with its drafter",
        )?;
        self.spec.drafter = true;
        Ok(d)
    }
}

impl SamplerProbe for Engine {
    fn copy_dev(&mut self, dst: ApiBuf, src: ApiBuf, bytes: usize) -> Res<()> {
        if bytes == 0 {
            return Ok(());
        }
        let b = self.spec.bufs.borrow();
        let d = b.typed::<u8>(dst, bytes, "copy_dev dst")?;
        let s = b.typed::<u8>(src, bytes, "copy_dev src")?;
        self.st.dtod(&d, &s).rt()
    }

    fn copy_logits(&mut self, dst: ApiBuf, dst_row: usize, rows: usize) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let nlog = self.last_rows();
        if rows > nlog {
            return Err(RuntimeError::request(format!(
                "copy_logits: {rows} rows requested, the last call returned {nlog}"
            )));
        }
        let need_end = dst_row
            .checked_add(rows)
            .and_then(|r| r.checked_mul(VOCAB_PAD))
            .ok_or_else(|| RuntimeError::request("copy_logits: destination range overflow"))?;
        let b = self.spec.bufs.borrow();
        let d = b.typed::<f32>(dst, need_end, "copy_logits dst")?;
        let d_slice = d.slice(dst_row * VOCAB_PAD..need_end).rt()?;
        let s_slice = self.a.logits.slice(0..rows * VOCAB_PAD).rt()?;
        self.st.dtod(&d_slice, &s_slice).rt()
    }

    fn sample_test_rows(&mut self, rows: &SampleTestRows<'_>, out: &mut SampleTestOut<'_>) -> Res<()> {
        let r = rows.lrow.len();
        if rows.par_idx.len() != r || rows.pos.len() != r {
            return Err(RuntimeError::request(
                "sample_test_rows: lrow, par_idx and pos must have the same length",
            ));
        }
        if r == 0 {
            return Ok(());
        }
        if rows.params.is_empty() {
            return Err(RuntimeError::request("sample_test_rows: params must not be empty"));
        }
        if out.stats.is_some() && r > 64 {
            return Err(RuntimeError::request(format!(
                "sample_test_rows: stats requested for {r} rows (max 64)"
            )));
        }
        let (lg_buf, lg_rows) = match rows.logits {
            Some(h) => {
                let b = self.spec.bufs.borrow();
                let raw = b.resolve(h, "sample_test_rows logits")?;
                let row_bytes = VOCAB_PAD * size_of::<f32>();
                let n = raw.len() / row_bytes;
                if n == 0 {
                    return Err(RuntimeError::request("sample_test_rows: logits buffer holds 0 rows"));
                }
                (Some(b.typed::<f32>(h, n * VOCAB_PAD, "sample_test_rows logits")?), n)
            }
            None => (None, self.last_rows()),
        };
        for &lr in rows.lrow {
            let Ok(u) = usize::try_from(lr) else {
                return Err(RuntimeError::request(format!("sample_test_rows: negative lrow {lr}")));
            };
            if u >= lg_rows {
                return Err(RuntimeError::request(format!(
                    "sample_test_rows: lrow {u} >= {lg_rows} logit rows"
                )));
            }
        }
        for &pi in rows.par_idx {
            let Ok(u) = usize::try_from(pi) else {
                return Err(RuntimeError::request(format!(
                    "sample_test_rows: negative par_idx {pi}"
                )));
            };
            if u >= rows.params.len() {
                return Err(RuntimeError::request(format!(
                    "sample_test_rows: par_idx {u} >= {} params",
                    rows.params.len()
                )));
            }
        }
        let need_keys = if out.keys.is_some() { r * VOCAB } else { 0 };
        let realloc = match &self.spec.probe {
            Some(s) => s.cap_rows < r || s.cap_par < rows.params.len() || s.cap_keys < need_keys,
            None => true,
        };
        if realloc {
            let cap_rows = r.max(self.spec.probe.as_ref().map_or(64, |s| s.cap_rows));
            let cap_par = rows
                .params
                .len()
                .max(self.spec.probe.as_ref().map_or(64, |s| s.cap_par));
            let cap_keys = need_keys.max(self.spec.probe.as_ref().map_or(0, |s| s.cap_keys));
            let bytes = 4 * cap_rows * size_of::<i32>()
                + cap_par * size_of::<SampleParam>()
                + 64 * 8 * size_of::<u64>()
                + cap_keys * size_of::<u32>()
                + 4096;
            // drop the old scratch first so its memory is freed before allocating a larger one
            self.spec.probe = None;
            let mut arena = Arena::new(&self.ctx, &self.st, bytes).rt()?;
            let t_lrow = arena.take::<i32>(cap_rows, "probe lrow").rt()?;
            let t_par = arena.take::<i32>(cap_rows, "probe par").rt()?;
            let t_pos = arena.take::<i32>(cap_rows, "probe pos").rt()?;
            let t_out = arena.take::<i32>(cap_rows, "probe out").rt()?;
            let t_par_tbl = arena.take::<SampleParam>(cap_par, "probe par_tbl").rt()?;
            let t_stats = arena.take::<u64>(64 * 8, "probe stats").rt()?;
            let t_keys = if cap_keys > 0 {
                Some(arena.take::<u32>(cap_keys, "probe keys").rt()?)
            } else {
                None
            };
            self.spec.probe = Some(ProbeScratch {
                cap_rows,
                cap_par,
                cap_keys,
                t_lrow,
                t_par,
                t_pos,
                t_out,
                t_par_tbl,
                t_stats,
                t_keys,
                _arena: arena,
            });
        }
        let sc = self.spec.probe.as_ref().expect("allocated above");
        let host_par: Vec<SampleParam> = rows.params.iter().map(sample_param).collect();
        self.st.htod(&sc.t_lrow.slice(0..r).rt()?, rows.lrow).rt()?;
        self.st.htod(&sc.t_par.slice(0..r).rt()?, rows.par_idx).rt()?;
        self.st.htod(&sc.t_pos.slice(0..r).rt()?, rows.pos).rt()?;
        self.st
            .htod(&sc.t_par_tbl.slice(0..host_par.len()).rt()?, &host_par)
            .rt()?;
        let lg = lg_buf.as_ref().unwrap_or(&self.a.logits);
        let st_buf = out.stats.is_some().then_some(&sc.t_stats);
        let ks_buf = if out.keys.is_some() { sc.t_keys.as_ref() } else { None };
        self.spec_dev()
            .sample_test(
                r,
                lg_rows,
                host_par.len(),
                lg,
                &sc.t_par_tbl,
                &sc.t_lrow,
                &sc.t_par,
                &sc.t_pos,
                &sc.t_out,
                st_buf,
                ks_buf,
            )
            .rt()?;
        if let Some(o) = out.out.as_deref_mut() {
            o.clear();
            o.resize(r, 0);
            self.st.dtoh(o, &sc.t_out.slice(0..r).rt()?).rt()?;
        }
        if let Some(s) = out.stats.as_deref_mut() {
            s.clear();
            s.resize(r * 8, 0);
            self.st.dtoh(s, &sc.t_stats.slice(0..r * 8).rt()?).rt()?;
        }
        if let Some(k) = out.keys.as_deref_mut() {
            let kb = sc.t_keys.as_ref().expect("keys allocated above");
            k.clear();
            k.resize(r * VOCAB, 0);
            self.st.dtoh(k, &kb.slice(0..r * VOCAB).rt()?).rt()?;
        }
        self.st.synchronize().rt()
    }

    fn time_select(
        &mut self,
        sample: Option<&[DeviceSampling]>,
        warm: usize,
        reps: usize,
    ) -> Res<(f64, Instant, Instant)> {
        if reps == 0 {
            return Err(RuntimeError::request("time_select: reps must be > 0"));
        }
        let Some(m) = self.last.as_ref().filter(|_| !self.committed) else {
            return Err(RuntimeError::request("time_select: no uncommitted forward call"));
        };
        let (t, nlog, nseg) = (m.t, m.lrows.len(), m.call.len());
        if let Some(params) = sample {
            if params.len() != nseg {
                return Err(RuntimeError::request(format!(
                    "time_select: {} sampling controls, the last forward call had {nseg} segments",
                    params.len()
                )));
            }
            if self.spec.samp.is_none() {
                let n = self.mem.max_slots;
                let mut arena = Arena::new(&self.ctx, &self.st, n * size_of::<SampleParam>() + 256).rt()?;
                let buf = arena.take::<SampleParam>(n, "sampling controls").rt()?;
                self.spec.samp = Some((arena, buf));
            }
            let (_, par) = self.spec.samp.as_ref().expect("allocated above");
            let host: Vec<SampleParam> = params.iter().map(sample_param).collect();
            self.st.htod(&par.slice(0..nseg).rt()?, &host).rt()?;
        }
        let a_ev = self.ctx.new_event().rt()?;
        let b_ev = self.ctx.new_event().rt()?;
        let launch = |eng: &Self| -> Res<()> {
            if nlog == 0 {
                return Ok(());
            }
            let a = &eng.a;
            if sample.is_some() {
                let (_, par) = eng.spec.samp.as_ref().expect("allocated above");
                eng.spec_dev()
                    .sample(nlog, t, nseg, &a.logits, par, &a.logit_rows, (&a.pos, &a.seg), &a.amax)
                    .rt()
            } else {
                eng.spec_dev().argmax(nlog, &a.logits, &a.amax).rt()
            }
        };
        for _ in 0..warm {
            launch(self)?;
        }
        self.st.synchronize().rt()?;
        let t0 = Instant::now();
        self.st.record(&a_ev).rt()?;
        for _ in 0..reps {
            launch(self)?;
        }
        self.st.record(&b_ev).rt()?;
        b_ev.synchronize().rt()?;
        let t1 = Instant::now();
        let ms = f64::from(a_ev.elapsed_ms(&b_ev).rt()?) / reps as f64;
        Ok((ms, t0, t1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(temperature: f32, top_p: f32, min_p: f32, top_k: u32) -> SampleParam {
        sample_param(&DeviceSampling {
            seed: 7,
            temperature,
            top_p,
            min_p,
            top_k,
        })
    }

    /// `ink::sample_param` (sample.hpp) evaluated by hand at its edges.
    #[test]
    fn sample_param_equals_the_cpp_rule() {
        // greedy: !(T >= 1e-5), NaN included
        assert_eq!(
            (
                p(0.0, 1.0, 0.0, 0).greedy,
                p(9e-6, 1.0, 0.0, 0).greedy,
                p(1e-5, 1.0, 0.0, 0).greedy
            ),
            (1, 1, 0)
        );
        assert_eq!(p(f32::NAN, 1.0, 0.0, 0).greedy, 1);
        // top-p: off (2^24) at >= 1 or NaN, 0 at <= 0, else llround(p 2^24) (half away from zero)
        assert_eq!(p(1.0, 1.0, 0.0, 0).p24, 1 << 24);
        assert_eq!(p(1.0, f32::NAN, 0.0, 0).p24, 1 << 24);
        assert_eq!(p(1.0, -0.5, 0.0, 0).p24, 0);
        assert_eq!(p(1.0, 0.95, 0.0, 0).p24, (0.95f32 as f64 * 16_777_216.0).round() as u32);
        assert_eq!(p(1.0, 0.5, 0.0, 0).p24, 1 << 23);
        // 2^-25 is exactly half a step: rounds away from zero to 1
        assert_eq!(p(1.0, 2f32.powi(-25), 0.0, 0).p24, 1);
        // min-p as the smallest kept weight: max(1, ceil(min_p 2^40)); off -> 1, >= 1 -> 2^40 (only the mode's weight)
        assert_eq!(p(1.0, 1.0, 0.0, 0).wmin, 1);
        assert_eq!(p(1.0, 1.0, f32::NAN, 0).wmin, 1);
        assert_eq!(p(1.0, 1.0, 2.0, 0).wmin, 1 << 40);
        assert_eq!(
            p(1.0, 1.0, 0.05, 0).wmin,
            (0.05f32 as f64 * 1_099_511_627_776.0).ceil() as u64
        );
        assert_eq!(p(1.0, 1.0, 2f32.powi(-41), 0).wmin, 1);
        // top-k passes through; above i32 it saturates (the kernel reads it as "more than the vocabulary": off)
        assert_eq!(
            (p(1.0, 1.0, 0.0, 50).top_k, p(1.0, 1.0, 0.0, u32::MAX).top_k),
            (50, i32::MAX)
        );
        assert_eq!(p(0.7, 1.0, 0.0, 0).temperature.to_bits(), 0.7f32.to_bits());
        assert_eq!(p(0.7, 1.0, 0.0, 0).seed, 7);
    }
}
