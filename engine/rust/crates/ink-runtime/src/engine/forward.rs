// Ported from engine/native/src/engine.cu @ 1c7ac46: Engine::run_model, visit, attention, moe, message_gates, norm3,
// attn_res, gemm / gemm_f32 / for_groups, tapv, mark. Own code of this project.
//! One forward call: the 48 visits and the head, launching the C++ engine's kernels with its arguments in its
//! order, eagerly or into an open stream capture (decode calls with graphs on, `super::graphs`). Every launch goes through [`super::ops`]; dense GEMMs through `LtGemm` with the pinned algorithm of the
//! row group's plan class (decode rows on decode-class plans, prefill rows on mixed-class plans, as the C++
//! `for_groups`).

use bytemuck::{Pod, Zeroable};
use ink_cuda::{Bf16, Capture, Context, DevBuf, Event, GemmElem, GemmShape, LtGemm, PlanClass, Stream};
use ink_kernels::{da, pfa, pmoe};
use ink_model::dims::{
    BLOCK, DREL, GATE_RANK, H, HD, INTER, KD, KVH, LOOP_END, LOOP_START, NH, NLAYERS, NVISITS, PASSES, QD, REL_EXT,
    TOPK, VOCAB_PAD, Visit,
};
use ink_model::plan::FUSED_ROWS;
use ink_model::plan::PAGE_ELEMS;
use ink_model::{Activations, ExpertFmt, GateTailW, LayerW, MemConfig, Weights};
use ink_rt_api::{RuntimeError, TapFn, TapType};

use super::meta::{CallMeta, KPT, MAX_DEC_ROWS, RowGroup, SMS, WINDOW, pick_workers, pick_workers_warp_n};
use super::ops::{ConvRows, Dev, GateInputs, MgDownGate, MgTcGate, MoeBuild, ProjOut, QkvPost, SLD, res_src, rows};
use super::{Res, Rt};

/// Everything one forward call reads; built per call by the engine.
pub(super) struct Fwd<'e> {
    pub dev: Dev<'e>,
    pub st: &'e Stream,
    pub ctx: &'e Context,
    pub lt: &'e mut LtGemm,
    pub a: &'e Activations<ink_cuda::Arena>,
    /// The tensor-core gate tail's weights (G7).
    pub tc: &'e GateTailW<ink_cuda::Arena>,
    /// Transposed per-layer relative projections (`[1024][16]` bf16) for in-kernel TC bias in `prefill_attn`.
    pub rel_proj_t: &'e [DevBuf<Bf16>],
    pub w: &'e Weights,
    pub m: &'e CallMeta,
    pub mem: &'e MemConfig,
    pub plan: &'e [Visit],
    pub full_index: &'e [Option<usize>],
    pub ring_index: &'e [Option<usize>],
    pub da_ring_slots: &'e [usize; 17],
    /// `xk` and `xv` as one buffer (adjacent takes, checked by `Engine::new`): the k / v batch of the D1 projections.
    pub xkv: &'e DevBuf<Bf16>,
    /// Per-op reference glue (glue 1, or a tap is set).
    pub reference: bool,
    pub tap: Option<&'e mut TapFn>,
    /// Profile marks of this call (`Some` when profiling).
    pub marks: Option<&'e mut Vec<(String, Event)>>,
    /// With `marks`: also the sub-component marks (`fmark`, the C++ `profile_fine`).
    pub fine: bool,
    /// The open capture when this call is being captured (its marks become the graph's regions).
    pub capture: Option<&'e Capture<'e>>,
}

/// Calls above this many rows take the bit-identical prefill glue kernels (`pglue`: warp-per-row AttnRes and norms,
/// row-tiled convs), as the C++ `T > 64` (same bits either way: ink-inv `small_chunk_*`). The bit-changing tensor-core
/// gate tail is chosen by row class instead (`message_gates`).
const PGLUE_ROWS: usize = 64;

fn class(dec: bool) -> PlanClass {
    if dec { PlanClass::Decode } else { PlanClass::Mixed }
}

/// `x[off..]`.
fn at<T: Pod>(x: &DevBuf<T>, off: usize) -> Res<DevBuf<T>> {
    x.slice(off..x.len()).rt()
}

/// `x[off..off + n]`.
fn span<T: Pod>(x: &DevBuf<T>, off: usize, n: usize) -> Res<DevBuf<T>> {
    x.slice(off..off + n).rt()
}

fn need_some<'a, T>(x: &'a Option<T>, what: &str) -> Res<&'a T> {
    x.as_ref()
        .ok_or_else(|| RuntimeError::request(format!("weights: {what} missing")))
}

/// Which buffer holds the visit input.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Hin {
    Hid(usize),
    Blk1,
}

impl Fwd<'_> {
    fn t(&self) -> usize {
        self.m.t
    }

    /// `Engine::decode_only`: every segment of the call is a decode segment (<= 8 rows).
    fn decode_only(&self) -> bool {
        self.m.call.iter().all(|c| c.dec())
    }

    fn hin_buf(&self, h: Hin) -> &DevBuf<Bf16> {
        match h {
            Hin::Hid(i) => &self.a.hid[i],
            Hin::Blk1 => &self.a.blk[1],
        }
    }

    /// `Engine::mark`: in a capture, the region boundary of the graph (the capture's current leaf); otherwise, when
    /// profiling, a timing event.
    fn mark(&mut self, name: &str) -> Res<()> {
        if let Some(c) = self.capture {
            return c.mark(name).rt();
        }
        if let Some(m) = self.marks.as_mut() {
            let e = self.ctx.new_event().rt()?;
            self.st.record(&e).rt()?;
            m.push((name.to_owned(), e));
        }
        Ok(())
    }

    /// `Engine::fmark`: a sub-component mark, only with fine profiling and outside a capture.
    fn fmark(&mut self, name: &str) -> Res<()> {
        if self.fine && self.capture.is_none() {
            return self.mark(name);
        }
        Ok(())
    }

    /// `tapv`: a host copy of the tensor to the sink, its write-back before the consumers run.
    fn tap<T: Pod>(
        &mut self,
        visit: Option<usize>,
        name: &str,
        buf: &DevBuf<T>,
        ty: TapType,
        shape: &[usize],
    ) -> Res<()> {
        let Some(sink) = self.tap.as_mut() else { return Ok(()) };
        let nm = match visit {
            Some(v) => format!("v{v:02}.{name}"),
            None => name.to_owned(),
        };
        let bytes = shape.iter().product::<usize>() * size_of::<T>();
        let dev = buf.cast::<u8>().rt()?.slice(0..bytes).rt()?;
        self.st.synchronize().rt()?;
        let mut host = vec![0u8; bytes];
        self.st.dtoh(&mut host, &dev).rt()?;
        let mut scratch = vec![0u8; bytes];
        let shape64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
        let n = sink(&nm, ty, &shape64, &host, &mut scratch);
        if n > bytes {
            return Err(RuntimeError::request(format!(
                "tap {nm}: sink returned {n} for a tensor of {bytes} bytes"
            )));
        }
        if n > 0 {
            self.st.htod(&dev.slice(0..n).rt()?, &scratch[..n]).rt()?;
            self.st.synchronize().rt()?;
        }
        Ok(())
    }

    /// One pinned cuBLASLt GEMM: `out[oo..] (ld ldo) = x[xo..] (ld ldx) @ W[wo..] (ld ldw)^T`, `m` rows.
    #[expect(clippy::too_many_arguments, reason = "mirrors LtGemm::run")]
    fn run<I: GemmElem, O: GemmElem>(
        &mut self,
        cls: PlanClass,
        (n, k, ldw): (usize, usize, usize),
        x: &DevBuf<I>,
        xo: usize,
        ldx: usize,
        w: &DevBuf<I>,
        wo: usize,
        out: &DevBuf<O>,
        oo: usize,
        ldo: usize,
        m: usize,
    ) -> Res<()> {
        let shape = GemmShape {
            n: n as u32,
            k: k as u32,
            ldw: ldw as u32,
            input: I::DTYPE,
            output: O::DTYPE,
            class: cls,
        };
        self.lt
            .run(
                self.st,
                shape,
                &at(x, xo)?,
                ldx as u32,
                &at(w, wo)?,
                &at(out, oo)?,
                ldo as u32,
                m as u32,
            )
            .rt()
    }

    /// `Engine::gemm`: bf16 out, every row group on its own plan class.
    #[expect(clippy::too_many_arguments, reason = "mirrors Engine::gemm")]
    fn gemm(
        &mut self,
        x: &DevBuf<Bf16>,
        ldx: usize,
        w: &DevBuf<Bf16>,
        n: usize,
        k: usize,
        out: &DevBuf<Bf16>,
        ldo: usize,
    ) -> Res<()> {
        for g in self.m.rgroups.clone() {
            self.run(
                class(g.dec),
                (n, k, k),
                x,
                g.r0 * ldx,
                ldx,
                w,
                0,
                out,
                g.r0 * ldo,
                ldo,
                g.n,
            )?;
        }
        Ok(())
    }

    /// `Engine::gemm_f32` (reference path only): fp32 operands, the call's plan class (not per row group).
    #[expect(clippy::too_many_arguments, reason = "mirrors Engine::gemm_f32")]
    fn gemm_f32(
        &mut self,
        x: &DevBuf<f32>,
        ldx: usize,
        w: &DevBuf<f32>,
        n: usize,
        k: usize,
        out: &DevBuf<f32>,
        ldo: usize,
    ) -> Res<()> {
        let cls = class(self.m.call.iter().all(|c| c.dec()));
        let t = self.t();
        self.run(cls, (n, k, k), x, 0, ldx, w, 0, out, 0, ldo, t)
    }

    // ------------------------------------------------------------------------------------------------ pieces

    /// `Engine::attn_res`; with `g` > 0 (production path only: the reference path sets G = 0) the same launch writes
    /// the `g` loop-gate inputs to gin (C3): gate y's hidden part is the output for query (`qsel` ? y : 0), its
    /// message `msg + y H` (row stride 4H).
    #[expect(clippy::too_many_arguments, reason = "mirrors the C++ signature")]
    fn attn_res(
        &self,
        q: usize,
        queries: &DevBuf<f32>,
        nb: usize,
        last16: Option<&DevBuf<Bf16>>,
        last32: Option<&DevBuf<f32>>,
        out: &DevBuf<f32>,
        g: usize,
        msg: Option<&DevBuf<f32>>,
        qsel: bool,
    ) -> Res<()> {
        let blk: Vec<&DevBuf<Bf16>> = self.a.blk[..nb].iter().collect();
        let (g, gates) = match msg {
            Some(msg) if g > 0 && !self.reference => (
                g,
                Some(GateInputs {
                    msg,
                    ldm: 4 * H,
                    gin: &self.a.gin,
                    qsel,
                }),
            ),
            _ => (0, None),
        };
        let (t, s) = (self.t(), res_src(&blk, last16, last32, H));
        if !self.reference {
            self.dev.ar288(q, g, t, &s, queries, out, gates).rt()
        } else {
            self.dev
                .attn_res(q, g, t, &s, queries, out, gates, self.decode_only())
                .rt()
        }
    }

    /// `Engine::message_gates`: `g` loop message gates (`g0..g0 + g`), either the three-stage reference path or the
    /// fused production path (with `gin_ready`, AttnRes already wrote `gin`).
    #[expect(clippy::too_many_arguments, reason = "mirrors the C++ signature")]
    fn message_gates(
        &mut self,
        l: &LayerW,
        g0: usize,
        g: usize,
        hid: &DevBuf<f32>,
        hs: usize,
        ldh: usize,
        msg: &DevBuf<f32>,
        ldm: usize,
        out: &DevBuf<f32>,
        ldo: usize,
        gin_ready: bool,
        vmsg: Option<(&DevBuf<Bf16>, &DevBuf<Bf16>, &DevBuf<f32>)>,
    ) -> Res<()> {
        let t = self.t();
        let gi = t * 2 * H;
        let go = t * GATE_RANK;
        let ctl = need_some(&l.controllers, "loop controllers")?;
        let a = self.a;
        if self.reference {
            for y in 0..g {
                let c = &ctl[g0 + y];
                self.dev
                    .mg_in_one(t, &at(hid, y * hs)?, ldh, &at(msg, y * H)?, ldm, &a.gin)
                    .rt()?;
                self.gemm(&a.gin, 2 * H, &c.down, GATE_RANK, 2 * H, &a.gd, GATE_RANK)?;
                self.dev.bf16_to_f32(t * GATE_RANK, &a.gd, &a.gd32).rt()?;
                self.gemm_f32(&a.gd32, GATE_RANK, &c.forget, H, GATE_RANK, &a.mg_tmp[0], H)?;
                self.gemm_f32(&a.gd32, GATE_RANK, &c.gain, H, GATE_RANK, &a.mg_tmp[1], H)?;
                self.dev
                    .mg_out(
                        t,
                        &a.mg_tmp[0],
                        &a.mg_tmp[1],
                        &at(hid, y * hs)?,
                        ldh,
                        &at(msg, y * H)?,
                        ldm,
                        &at(out, y * H)?,
                        ldo,
                    )
                    .rt()?;
            }
            return Ok(());
        }
        let suffix = if g == 3 { "a" } else { "m" };
        if !gin_ready {
            self.fmark(&format!("mg_in_{suffix}"))?;
            self.dev.mg_in(t, g, hid, hs, ldh, msg, ldm, &a.gin).rt()?;
        }
        self.fmark(&format!("mg_down_{suffix}"))?;
        let ws_f32 = a.lt_ws.cast::<f32>().rt()?;
        let cnt_all = a.m_items2.cast::<i32>().rt()?;
        let cnt_mg = at(&cnt_all, 16)?;
        for grp in self.m.rgroups.clone() {
            if grp.dec {
                let mut off = 0;
                while off < grp.n {
                    let chunk = (grp.n - off).min(64);
                    let r0 = grp.r0 + off;
                    let bufs: Vec<(DevBuf<Bf16>, DevBuf<Bf16>)> = (0..g)
                        .map(|y| Ok((at(&a.gin, y * gi + r0 * 2 * H)?, at(&a.gd, y * go + r0 * GATE_RANK)?)))
                        .collect::<Res<_>>()?;
                    let gates: Vec<MgDownGate<'_>> = bufs
                        .iter()
                        .enumerate()
                        .map(|(y, (x, out))| MgDownGate {
                            x,
                            w: &ctl[g0 + y].down,
                            out,
                        })
                        .collect();
                    self.dev.mg_down_dec(chunk, &gates, &ws_f32, &cnt_mg).rt()?;
                    off += chunk;
                }
            } else {
                for y in 0..g {
                    let down = &ctl[g0 + y].down;
                    self.run(
                        PlanClass::Mixed,
                        (GATE_RANK, 2 * H, 2 * H),
                        &a.gin,
                        y * gi + grp.r0 * 2 * H,
                        2 * H,
                        down,
                        0,
                        &a.gd,
                        y * go + grp.r0 * GATE_RANK,
                        GATE_RANK,
                        grp.n,
                    )?;
                }
            }
        }
        self.fmark(&format!("mg_tail_{suffix}"))?;
        type Views = Vec<(DevBuf<Bf16>, [DevBuf<f32>; 3])>;
        let views = |r0: usize| -> Res<Views> {
            (0..g)
                .map(|y| {
                    Ok((
                        at(&a.gd, y * go + r0 * GATE_RANK)?,
                        [
                            at(hid, y * hs + r0 * ldh)?,
                            at(msg, y * H + r0 * ldm)?,
                            at(out, y * H + r0 * ldo)?,
                        ],
                    ))
                })
                .collect()
        };
        let tw = self
            .tc
            .w
            .get(l.idx)
            .and_then(Option::as_ref)
            .ok_or_else(|| RuntimeError::request("tensor-core gate tail weights missing"))?;
        for grp in self.m.rgroups.clone() {
            if let Some((msrc0, msrc1, mw)) = vmsg {
                let v = views(grp.r0)?;
                let gates: Vec<MgTcGate<'_>> = v
                    .iter()
                    .enumerate()
                    .map(|(y, (gd, [hid, msg, out]))| MgTcGate {
                        gd,
                        w: &tw[g0 + y],
                        hid,
                        msg,
                        out,
                    })
                    .collect();
                let ms0 = at(msrc0, grp.r0 * H)?;
                let ms1 = at(msrc1, grp.r0 * H)?;
                let mwg = at(mw, grp.r0 * 16)?;
                self.dev
                    .mg_tail_tc_vmsg(grp.n, &gates, &ms0, &ms1, &mwg, ldh, ldo, g0)
                    .rt()?;
            } else if grp.dec {
                let mut off = 0;
                while off < grp.n {
                    let chunk = (grp.n - off).min(64);
                    let v = views(grp.r0 + off)?;
                    let gates: Vec<MgTcGate<'_>> = v
                        .iter()
                        .enumerate()
                        .map(|(y, (gd, [hid, msg, out]))| MgTcGate {
                            gd,
                            w: &tw[g0 + y],
                            hid,
                            msg,
                            out,
                        })
                        .collect();
                    self.dev.mg_tail_tc_dec(chunk, &gates, ldh, ldm, ldo).rt()?;
                    off += chunk;
                }
            } else {
                let v = views(grp.r0)?;
                let gates: Vec<MgTcGate<'_>> = v
                    .iter()
                    .enumerate()
                    .map(|(y, (gd, [hid, msg, out]))| MgTcGate {
                        gd,
                        w: &tw[g0 + y],
                        hid,
                        msg,
                        out,
                    })
                    .collect();
                self.dev.mg_tail_tc(grp.n, &gates, ldh, ldm, ldo).rt()?;
            }
        }
        Ok(())
    }

    /// `Engine::norm3`: three RMSNorms of the q / k / v sources (row stride `ldi`) into xq, xk, xv.
    fn norm3(&mut self, src: &DevBuf<f32>, ldi: usize, w: [&DevBuf<f32>; 3]) -> Res<()> {
        let t = self.t();
        let a = self.a;
        let s = [at(src, 0)?, at(src, H)?, at(src, 2 * H)?];
        if self.reference {
            self.dev.rmsnorm_f32(t, &s[0], ldi, w[0], &a.xq).rt()?;
            self.dev.rmsnorm_f32(t, &s[1], ldi, w[1], &a.xk).rt()?;
            return self.dev.rmsnorm_f32(t, &s[2], ldi, w[2], &a.xv).rt();
        }
        // local visits pass attn_norm three times
        self.fmark(if std::ptr::eq(w[1], w[0]) { "norm3_l" } else { "norm3_g" })?;
        if t > PGLUE_ROWS {
            return self
                .dev
                .norm_rows(t, &[&s[0], &s[1], &s[2]], ldi, &w, &[&a.xq, &a.xk, &a.xv])
                .rt();
        }
        self.dev
            .rmsnorm_g3(t, [&s[0], &s[1], &s[2]], ldi, w, [&a.xq, &a.xk, &a.xv])
            .rt()
    }

    /// The pool (global layers) or ring (sliding layers) of visit `v`: K, V (raw bytes: 1 B/elem for global E4M3, 2 B/elem for bf16).
    fn kv(&self, v: usize, full: bool) -> Res<(DevBuf<u8>, DevBuf<u8>)> {
        let a = self.a;
        if full {
            let fi = self.full_index[v].ok_or_else(|| RuntimeError::request("full index"))?;
            let elem = if self.mem.kv_fp8 { 1 } else { 2 };
            let n = self.mem.num_pages * PAGE_ELEMS * elem;
            Ok((span(&a.kpool, fi * n, n)?, span(&a.vpool, fi * n, n)?))
        } else {
            let ri = self.ring_index[v].ok_or_else(|| RuntimeError::request("ring index"))?;
            let n = self.mem.max_slots * self.mem.ring_pages() * PAGE_ELEMS;
            Ok((
                span(&a.kring, ri * n, n)?.cast::<u8>().rt()?,
                span(&a.vring, ri * n, n)?.cast::<u8>().rt()?,
            ))
        }
    }

    /// `Engine::attention`: decode segments on decode_attn, prefill segments on prefill_attn.
    fn attention(&self, v: usize, l: &LayerW) -> Res<()> {
        let full = l.full;
        let fp8 = full && self.mem.kv_fp8;
        let elem = if fp8 { 1usize } else { 2usize };
        let a = self.a;
        let (kb, vb) = self.kv(v, full)?;
        let pt_call = if full { &a.pt_g } else { &a.pt_r };
        let ms = self.mem.max_slots;
        let dec: Vec<_> = self.m.call.iter().filter(|c| c.dec()).collect();
        if !dec.is_empty() {
            let n = dec.len();
            let r = super::meta::decode_r(&self.m.call);
            let gx = super::meta::decode_gx(&self.m.call, full);
            if n * 8 * gx * 3 * r * 128 > a.da_opart_floats {
                return Err(RuntimeError::request("decode partials"));
            }
            let mut p = da::Params::zeroed();
            p.k = kb.ptr();
            p.v = vb.ptr();
            p.page_table = at(pt_call, ms * KPT)?.ptr();
            p.pt_stride = KPT as i32;
            p.page_shift = 10;
            p.page_stride = (PAGE_ELEMS * elem) as i64;
            p.head_stride = (1024 * HD * elem) as i64;
            p.row_stride = (HD * elem) as i32;
            p.window = if full { 1 << 30 } else { WINDOW as i32 };
            p.ctx = a.da_ctx.ptr();
            p.row0 = a.da_row0.ptr();
            p.chunk_shift = at(&a.da_cs, if full { 0 } else { ms })?.ptr();
            p.R = r as i32;
            p.nrows = a.da_nrows.ptr();
            p.q = a.q.cast::<u16>().rt()?.ptr();
            p.rel = a.rel_logits.cast::<u16>().rt()?.ptr();
            p.extent = REL_EXT as i32;
            p.gate = a.gate.cast::<u16>().rt()?.ptr();
            p.sink = a.sink.cast::<u16>().rt()?.ptr();
            p.o_part = a.da_opart.ptr();
            p.ml_part = a.da_ml.ptr();
            p.max_chunks = gx as i32;
            p.out = a.attn.cast::<u16>().rt()?.ptr();
            let l2s = std::f32::consts::LOG2_E / (128.0f32).sqrt(); // da::kLog2e / sqrtf(kD)
            p.sc = if fp8 { l2s * 256.0 } else { l2s };
            p.vscale = if fp8 { 256.0 } else { 1.0 };
            let gxw = if full {
                pick_workers_warp_n(n, gx, SMS, if r > 8 { 2.0 } else { 4.0 })
            } else {
                pick_workers(n, gx, self.da_ring_slots[r])
            };
            self.dev.decode_attn(&p, !full, fp8, n, gxw).rt()?;
        }
        if self.m.call.iter().any(|c| !c.dec()) {
            let kind = usize::from(!full);
            let mut p = pfa::Params::zeroed();
            p.q = a.q.ptr();
            p.k = kb.cast::<Bf16>().rt()?.ptr();
            p.v = vb.cast::<Bf16>().rt()?.ptr();
            p.s_page = PAGE_ELEMS as i64;
            p.s_slot = HD as i64;
            p.s_head = (1024 * HD) as i64;
            p.page_shift = 10;
            p.page_table = pt_call.ptr();
            p.rel = a.rel.ptr();
            p.rel_proj_t = self.rel_proj_t[l.idx].ptr();
            p.gate = a.gate.ptr();
            p.sink = a.sink.ptr();
            p.out = a.attn.ptr();
            p.segs = a.pa_segs.cast::<pfa::Seg>().rt()?.ptr();
            p.items = at(&a.pa_items.cast::<[i32; 4]>().rt()?, kind * a.pa_items_cap)?.ptr();
            p.items2 = at(&a.pa_items2.cast::<[i32; 4]>().rt()?, kind * a.pa_items_cap)?.ptr();
            p.ws = a.pa_ws.ptr();
            p.cnt = at(&a.pa_cnt, kind * a.pa_tiles_cap * 8)?.ptr();
            p.window = if full { 0 } else { WINDOW as i32 };
            // fp32(0.08838834764831845f) * fp32(log2e), as the C++ (the literal 128^-0.5 rounded to f32 is 0x3db504f3)
            let qk_base = f32::from_bits(0x3db5_04f3) * std::f32::consts::LOG2_E;
            p.qk_scale = if fp8 { qk_base * 256.0 } else { qk_base };
            self.dev.prefill_attn(&p, fp8, self.m.pa_items[kind].len()).rt()?;
        }
        Ok(())
    }

    /// `Engine::moe`: router (three K-group GEMMs + top-8, or the reference router), routing build, experts, combine.
    fn moe(&mut self, v: usize, l: &LayerW, mx: &DevBuf<Bf16>, combine_here: bool) -> Res<()> {
        let t = self.t();
        let a = self.a;
        let tm = self.mem.t_max;
        let ridx = span(&a.ridx_all, v * tm * TOPK, tm * TOPK)?;
        let rw = span(&a.rw_all, v * tm * TOPK, tm * TOPK)?;
        if self.reference {
            self.dev
                .router_ref(t, &a.xm, &l.router, &l.router_bias, &ridx, &rw)
                .rt()?;
        } else {
            let cnt_router = a.m_items2.cast::<i32>().rt()?;
            for grp in self.m.rgroups.clone() {
                if grp.dec {
                    let mut off = 0;
                    while off < grp.n {
                        let chunk = (grp.n - off).min(64);
                        let r0 = grp.r0 + off;
                        self.dev
                            .router_decode(
                                chunk,
                                &at(&a.xm, r0 * H)?,
                                &l.router,
                                &l.router_bias,
                                &a.rlog,
                                &cnt_router,
                                &at(&ridx, r0 * TOPK)?,
                                &at(&rw, r0 * TOPK)?,
                            )
                            .rt()?;
                        off += chunk;
                    }
                } else {
                    for g in 0..3 {
                        let k = if g < 2 { 1024 } else { H - 2048 };
                        self.run(
                            PlanClass::Mixed,
                            (ink_model::dims::NEXP, k, H),
                            &a.xm,
                            grp.r0 * H + g * 1024,
                            H,
                            &l.router,
                            g * 1024,
                            &a.rlog,
                            g * grp.n * ink_model::dims::NEXP,
                            ink_model::dims::NEXP,
                            grp.n,
                        )?;
                    }
                    self.fmark("route_select")?;
                    self.dev
                        .route_select(
                            grp.n,
                            &a.rlog,
                            &l.router_bias,
                            &at(&ridx, grp.r0 * TOPK)?,
                            &at(&rw, grp.r0 * TOPK)?,
                        )
                        .rt()?;
                }
            }
        }
        self.tap(Some(v), "ridx", &ridx, TapType::I32, &[t, TOPK])?;
        self.tap(Some(v), "rw", &rw, TapType::F32, &[t, TOPK])?;
        self.mark("experts")?;
        // prefill_moe is batch-invariant per row: one pass over every row of the call
        let items13 = a.m_items13.cast::<pmoe::Item>().rt()?;
        self.fmark("moe_build")?;
        self.dev
            .moe_build(
                t,
                &MoeBuild {
                    idx: &ridx,
                    wts: &rw,
                    a13: &a.m_arow13,
                    a2: &a.m_arow2,
                    orow: &a.m_orow,
                    gamma: &a.m_gamma,
                    items: &items13,
                    n_items: &a.m_nitems,
                    cntb: &a.m_cntb,
                    baseb: &a.m_baseb,
                },
            )
            .rt()?;
        let mut p13 = pmoe::Params::zeroed();
        p13.A = a.xm.ptr();
        p13.a_row = a.m_arow13.ptr();
        p13.W = l.experts[0].w.ptr();
        p13.S = l.experts[0].s.ptr();
        p13.F = l.experts[0].f.ptr();
        p13.items = items13.ptr();
        p13.n_items = 0;
        p13.K = H as i32;
        p13.NWT = (2 * INTER / 64) as i32;
        p13.out = a.mh.ptr();
        p13.out_row = a.m_orow.ptr();
        p13.gamma = a.m_gamma.ptr();
        p13.n_items_dev = a.m_nitems.ptr();
        let mut p2 = p13;
        p2.A = a.mh.ptr();
        p2.a_row = a.m_arow2.ptr();
        p2.W = l.experts[1].w.ptr();
        p2.S = l.experts[1].s.ptr();
        p2.F = l.experts[1].f.ptr();
        p2.K = INTER as i32;
        p2.NWT = (H / 64) as i32;
        p2.out = a.my.ptr();
        self.fmark("moe_gemm")?;
        self.dev.moe_gemm(l.fmt == ExpertFmt::Nvfp4, t * TOPK, &p13, &p2).rt()?;
        if combine_here {
            self.fmark("moe_combine")?;
            self.dev.moe_combine(t, &a.my, mx).rt()?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------ one visit

    /// `Engine::visit`.
    fn visit(
        &mut self,
        v: usize,
        hin: Hin,
        hout: usize,
        nb: &mut usize,
        has_msgs: bool,
        vmsg_i: Option<usize>,
    ) -> Res<()> {
        let vi = self.plan[v];
        let w = self.w;
        let l = &w.layers[vi.layer];
        let a = self.a;
        let t = self.t();
        let tm = self.mem.t_max;
        let xs = tm * H;
        let kx = span(&a.convx, (v * 4) * xs, xs)?;
        let vx = span(&a.convx, (v * 4 + 1) * xs, xs)?;
        let ax = span(&a.convx, (v * 4 + 2) * xs, xs)?;
        let mx = span(&a.convx, (v * 4 + 3) * xs, xs)?;
        let rws = rows(&a.pos, &a.seg, &a.row0, &a.slot);
        let cst = at(&a.conv, v * 3 * SLD)?;
        let (kb, vb) = self.kv(v, l.full)?;
        let pt = if l.full { &a.pt_g } else { &a.pt_r };
        let reference = self.reference;
        let hin_b = self.hin_buf(hin).clone();
        let hout_b = &a.hid[hout];
        let blk_now: Vec<&DevBuf<Bf16>> = a.blk[..*nb].iter().collect();
        self.mark("attn_in")?;
        self.fmark(match (l.full, has_msgs, vi.layer) {
            (true, true, _) => "ar_in_gm",
            (true, false, _) => "ar_in_g",
            (false, _, 0) => "norm_in0",
            (false, true, _) => "ar_in_lm",
            (false, false, _) => "ar_in_l",
        })?;
        // ---- attention inputs (AttnRes, loop message gate, RMSNorm)
        let (mut pk_is_xq, mut pv_is_xq) = (false, false);
        let aq = l.attn_query.as_ref();
        let rows_glue = !reference && t > PGLUE_ROWS; // the prefill glue kernels (pglue)
        let mw_buf = match vmsg_i {
            Some(mi) => Some(span(&a.msgs, mi * t * 16, t * 16)?),
            None => None,
        };
        let vmsg = mw_buf.as_ref().map(|mw| (&a.msrc[0], &a.msrc[1], mw));
        if l.full && !has_msgs && !reference {
            let (s, q) = (
                res_src(&blk_now, Some(&hin_b), None, H),
                need_some(&l.attn_query, "attn_query")?,
            );
            let nw = [l.q_norm.as_ref(), l.k_norm.as_ref(), l.v_norm.as_ref()];
            let no = [Some(&a.xq), Some(&a.xk), Some(&a.xv)];
            self.dev.ar_norm288(3, t, &s, q, nw, no).rt()?;
        } else if l.full {
            if let Some((ms0, ms1, mw)) = vmsg {
                let s = res_src(&blk_now, Some(&hin_b), None, H);
                let q = need_some(&l.attn_query, "attn_query")?;
                self.dev.ar288_vmsg(3, t, &s, q, &a.ar_out, ms0, ms1, mw, &a.gin).rt()?;
            } else {
                self.attn_res(
                    3,
                    need_some(&l.attn_query, "attn_query")?,
                    *nb,
                    Some(&hin_b),
                    None,
                    &a.ar_out,
                    if has_msgs { 3 } else { 0 },
                    Some(&a.msgs),
                    true,
                )?;
            }
            let qkv_w = [
                need_some(&l.q_norm, "q_norm")?,
                need_some(&l.k_norm, "k_norm")?,
                need_some(&l.v_norm, "v_norm")?,
            ];
            if has_msgs {
                self.message_gates(
                    l,
                    0,
                    3,
                    &a.ar_out,
                    H,
                    3 * H,
                    &a.msgs,
                    4 * H,
                    &a.mg_out,
                    3 * H,
                    !reference,
                    vmsg,
                )?;
                self.norm3(&a.mg_out, 3 * H, qkv_w)?;
            } else {
                self.norm3(&a.ar_out, 3 * H, qkv_w)?;
            }
        } else if vi.layer == 0 {
            self.dev
                .rmsnorm_bf16(t, &hin_b, H, need_some(&l.attn_norm, "attn_norm")?, &a.xq)
                .rt()?;
            (pk_is_xq, pv_is_xq) = (true, true);
        } else if !has_msgs {
            if !reference {
                let (s, q) = (
                    res_src(&blk_now, Some(&hin_b), None, H),
                    aq.ok_or_else(|| RuntimeError::request("attn_query"))?,
                );
                let (nw, no) = ([l.attn_norm.as_ref(), None, None], [Some(&a.xq), None, None]);
                self.dev.ar_norm288(1, t, &s, q, nw, no).rt()?;
            } else {
                self.attn_res(
                    1,
                    need_some(&l.attn_query, "attn_query")?,
                    *nb,
                    Some(&hin_b),
                    None,
                    &a.ar_out,
                    0,
                    None,
                    false,
                )?;
                self.dev
                    .rmsnorm_f32(t, &a.ar_out, H, need_some(&l.attn_norm, "attn_norm")?, &a.xq)
                    .rt()?;
            }
            (pk_is_xq, pv_is_xq) = (true, true);
        } else {
            if let Some((ms0, ms1, mw)) = vmsg {
                let s = res_src(&blk_now, Some(&hin_b), None, H);
                let q = need_some(&l.attn_query, "attn_query")?;
                self.dev.ar288_vmsg(1, t, &s, q, &a.ar_out, ms0, ms1, mw, &a.gin).rt()?;
            } else {
                self.attn_res(
                    1,
                    need_some(&l.attn_query, "attn_query")?,
                    *nb,
                    Some(&hin_b),
                    None,
                    &a.ar_out,
                    3,
                    Some(&a.msgs),
                    false,
                )?;
            }
            self.message_gates(
                l,
                0,
                3,
                &a.ar_out,
                0,
                H,
                &a.msgs,
                4 * H,
                &a.mg_out,
                3 * H,
                !reference,
                vmsg,
            )?;
            let n = need_some(&l.attn_norm, "attn_norm")?;
            self.norm3(&a.mg_out, 3 * H, [n, n, n])?;
        }
        let px = &a.xq;
        let pk = if pk_is_xq { &a.xq } else { &a.xk };
        let pv = if pv_is_xq { &a.xq } else { &a.xv };
        self.tap(Some(v), "xq", px, TapType::Bf16, &[t, H])?;
        if !pk_is_xq {
            self.tap(Some(v), "xk", pk, TapType::Bf16, &[t, H])?;
        }
        if !pv_is_xq {
            self.tap(Some(v), "xv", pv, TapType::Bf16, &[t, H])?;
        }
        // ---- projections: fused cuBLASLt over the contiguous [q | gate | r | sink | k (| k_dyn) | v (| v_dyn)]
        // weight. On prefill-carrying production calls (`fused_qkv_pf`), every row group writes into `a.relk` at
        // `o * nt` (skipping `proj_split` completely for prefill rows; decode rows in mixed calls split into
        // `gate`/`sink` for `decode_attn`); `qkv_post_fused_rows` and `prefill_attn` read directly from `a.relk`.
        self.mark("proj")?;
        let tapping = self.tap.is_some();
        let any_pre = self.m.call.iter().any(|c| c.m > MAX_DEC_ROWS);
        let any_dec = self.m.call.iter().any(|c| c.m <= MAX_DEC_ROWS);
        let fused_qkv_pf = !reference && !tapping && any_pre;
        let nq = 2 * QD + NH * DREL + NH;
        let nk = KD + if l.full { 32 } else { 0 };
        let nt = nq + 2 * nk;
        for g in self.m.rgroups.clone() {
            let RowGroup { r0: o, n, dec } = g;
            if fused_qkv_pf {
                let c = class(dec);
                if pk_is_xq && pv_is_xq {
                    self.run(c, (nt, H, H), px, o * H, H, &l.wqkv, 0, &a.relk, o * nt, nt, n)?;
                } else {
                    self.run(c, (nq, H, H), px, o * H, H, &l.wqkv, 0, &a.relk, o * nt, nt, n)?;
                    let shape = GemmShape {
                        n: nk as u32,
                        k: H as u32,
                        ldw: H as u32,
                        input: ink_cuda::Dtype::Bf16,
                        output: ink_cuda::Dtype::Bf16,
                        class: c,
                    };
                    self.lt
                        .run2(
                            self.st,
                            shape,
                            &at(self.xkv, o * H)?,
                            H as u32,
                            (tm * H) as u64,
                            &at(&l.wqkv, nq * H)?,
                            &at(&a.relk, o * nt + nq)?,
                            nt as u32,
                            n as u32,
                        )
                        .rt()?;
                }
                if dec {
                    let s = |x: &DevBuf<Bf16>, ld: usize| at(x, o * ld);
                    let (q, gt, r, sk, k, kd, vv, vd) = (
                        s(&a.qproj, QD)?,
                        s(&a.gate, QD)?,
                        s(&a.rproj, NH * DREL)?,
                        s(&a.sink, NH)?,
                        s(&kx, KD)?,
                        s(&a.kdyn, 32)?,
                        s(&vx, KD)?,
                        s(&a.vdyn, 32)?,
                    );
                    let out = ProjOut {
                        q: &q,
                        g: &gt,
                        r: &r,
                        s: &sk,
                        k: &k,
                        kd: &kd,
                        v: &vv,
                        vd: &vd,
                    };
                    self.dev.proj_split(n, &at(&a.relk, o * nt)?, nt, l.full, &out).rt()?;
                }
            } else if !tapping && (!dec || self.m.dec_rows <= FUSED_ROWS) {
                let c = class(dec);
                let qkvf = if dec { &a.qkvf } else { &a.relk };
                if pk_is_xq && pv_is_xq {
                    self.run(c, (nt, H, H), px, o * H, H, &l.wqkv, 0, qkvf, 0, nt, n)?;
                } else {
                    // xk, xv adjacent (Engine::new checks it: `xkv`); wv = wk + nk rows (the loader's contiguity)
                    self.run(c, (nq, H, H), px, o * H, H, &l.wqkv, 0, qkvf, 0, nt, n)?;
                    let shape = GemmShape {
                        n: nk as u32,
                        k: H as u32,
                        ldw: H as u32,
                        input: ink_cuda::Dtype::Bf16,
                        output: ink_cuda::Dtype::Bf16,
                        class: c,
                    };
                    self.lt
                        .run2(
                            self.st,
                            shape,
                            &at(self.xkv, o * H)?,
                            H as u32,
                            (tm * H) as u64,
                            &at(&l.wqkv, nq * H)?,
                            &at(qkvf, nq)?,
                            nt as u32,
                            n as u32,
                        )
                        .rt()?;
                }
                let s = |x: &DevBuf<Bf16>, ld: usize| at(x, o * ld);
                let (q, gt, r, sk, k, kd, vv, vd) = (
                    s(&a.qproj, QD)?,
                    s(&a.gate, QD)?,
                    s(&a.rproj, NH * DREL)?,
                    s(&a.sink, NH)?,
                    s(&kx, KD)?,
                    s(&a.kdyn, 32)?,
                    s(&vx, KD)?,
                    s(&a.vdyn, 32)?,
                );
                let out = ProjOut {
                    q: &q,
                    g: &gt,
                    r: &r,
                    s: &sk,
                    k: &k,
                    kd: &kd,
                    v: &vv,
                    vd: &vd,
                };
                self.dev.proj_split(n, qkvf, nt, l.full, &out).rt()?;
            } else {
                let c = class(dec);
                let mm = |f: &mut Self, x: &DevBuf<Bf16>, wt: &DevBuf<Bf16>, nn: usize, out: &DevBuf<Bf16>| {
                    f.run(c, (nn, H, H), x, o * H, H, wt, 0, out, o * nn, nn, n)
                };
                mm(self, px, &l.wq, QD, &a.qproj)?;
                mm(self, px, &l.wr, NH * DREL, &a.rproj)?;
                mm(self, pk, &l.wk, KD, &kx)?;
                mm(self, pv, &l.wv, KD, &vx)?;
                if l.full {
                    mm(self, pk, need_some(&l.wk_dyn, "wk_dyn")?, 32, &a.kdyn)?;
                    mm(self, pv, need_some(&l.wv_dyn, "wv_dyn")?, 32, &a.vdyn)?;
                }
                mm(self, px, &l.wsink, NH, &a.sink)?;
                mm(self, px, &l.wgate, QD, &a.gate)?;
            }
        }
        // ---- q / rel post, relative logits, key / value conv, KV write
        self.mark("qkv_post")?;
        let (kw, vw) = if l.full {
            (need_some(&l.k_static, "k_static")?, need_some(&l.v_static, "v_static")?)
        } else {
            (need_some(&l.k_conv, "k_conv")?, need_some(&l.v_conv, "v_conv")?)
        };
        let vst = at(&cst, 1024)?;
        let fp8 = l.full && self.mem.kv_fp8;
        let qp = QkvPost {
            qproj: &a.qproj,
            q_scale: &l.q_scale,
            rproj: &a.rproj,
            r_scale: &l.r_scale,
            pos: &a.pos,
            full: l.full,
            fp8,
            q: &a.q,
            rel: &a.rel,
            kx: &kx,
            vx: &vx,
            kst: &cst,
            vst: &vst,
            rows: rws,
            kw,
            vw,
            kdyn: &a.kdyn,
            vdyn: &a.vdyn,
            kpool: &kb,
            vpool: &vb,
            pt,
        };
        if fused_qkv_pf {
            self.dev
                .qkv_post_fused_rows(t, &a.relk, nt, &a.gate, &a.sink, &a.len, qp)
                .rt()?;
        } else if !reference && any_pre {
            self.dev.qkv_post_rows(t, qp).rt()?;
        } else if !reference {
            self.dev.qkv_post_rows(t, qp).rt()?;
            self.dev.rel_logits_decode(t, &a.rel, &l.rel_proj, &a.rel_logits).rt()?;
        } else {
            self.dev.qpost_ref(t, &qp, &l.rel_proj, &a.rel_logits).rt()?;
        }
        self.tap(Some(v), "q", &a.q, TapType::Bf16, &[t, NH, HD])?;
        self.tap(Some(v), "rel_logits", &a.rel_logits, TapType::Bf16, &[t, NH, REL_EXT])?;
        self.tap(Some(v), "sink", &a.sink, TapType::Bf16, &[t, NH])?;
        self.tap(Some(v), "gate", &a.gate, TapType::Bf16, &[t, QD])?;
        if reference {
            let qp = QkvPost {
                qproj: &a.qproj,
                q_scale: &l.q_scale,
                rproj: &a.rproj,
                r_scale: &l.r_scale,
                pos: &a.pos,
                full: l.full,
                fp8,
                q: &a.q,
                rel: &a.rel,
                kx: &kx,
                vx: &vx,
                kst: &cst,
                vst: &vst,
                rows: rws,
                kw,
                vw,
                kdyn: &a.kdyn,
                vdyn: &a.vdyn,
                kpool: &kb,
                vpool: &vb,
                pt,
            };
            self.dev.kv_conv_ref(t, &qp, &a.k, &a.v).rt()?;
            self.tap(Some(v), "k", &a.k, TapType::Bf16, &[t, KVH, HD])?;
            self.tap(Some(v), "v", &a.v, TapType::Bf16, &[t, KVH, HD])?;
            self.dev.kv_write_ref(t, &qp, &a.k, &a.v).rt()?;
        }
        self.fmark("relk")?;
        if any_pre && any_dec && !reference {
            for grp in self.m.rgroups.clone() {
                if grp.dec {
                    self.dev
                        .rel_logits(
                            grp.n,
                            &at(&a.rel, grp.r0 * NH * DREL)?,
                            &l.rel_proj,
                            &at(&a.rel_logits, grp.r0 * NH * REL_EXT)?,
                        )
                        .rt()?;
                }
            }
        }
        // ---- attention, o projection
        self.mark("attn")?;
        self.fmark(if l.full { "attn_global" } else { "attn_swa" })?;
        self.attention(v, l)?;
        self.tap(Some(v), "attn", &a.attn, TapType::Bf16, &[t, QD])?;
        self.mark("wo")?;
        self.gemm(&a.attn, QD, &l.wo, H, QD, &ax, H)?;
        self.tap(Some(v), "wo", &ax, TapType::Bf16, &[t, H])?;
        // ---- attention conv + residual, MLP input (AttnRes, loop message gate, mlp_norm)
        self.mark("glue_mid")?;
        let bstart = vi.layer.is_multiple_of(BLOCK);
        let blk_out = bstart.then(|| &a.blk[*nb]);
        let cst_attn = at(&cst, 2048)?;
        if !reference {
            self.fmark(if has_msgs { "ar_mlp_m" } else { "ar_mlp" })?;
            let blk_prev: Vec<&DevBuf<Bf16>> = a.blk[..*nb].iter().collect();
            let s_prev = res_src(&blk_prev, None, None, H);
            if !has_msgs {
                self.dev
                    .conv_ar_norm288(
                        t,
                        &ax,
                        &l.attn_conv,
                        &cst_attn,
                        rws,
                        &hin_b,
                        blk_out,
                        &a.partial,
                        &s_prev,
                        &l.mlp_query,
                        &l.mlp_norm,
                        &a.xm,
                    )
                    .rt()?;
                if bstart {
                    *nb += 1;
                }
            } else {
                let mlp_msg = at(&a.msgs, 3 * H)?;
                if let Some((ms0, ms1, mw)) = vmsg {
                    self.dev
                        .conv_ar_gate288_vmsg(
                            t,
                            &ax,
                            &l.attn_conv,
                            &cst_attn,
                            rws,
                            &hin_b,
                            blk_out,
                            &a.partial,
                            &s_prev,
                            &l.mlp_query,
                            &a.ar_out,
                            ms0,
                            ms1,
                            mw,
                            &a.gin,
                        )
                        .rt()?;
                } else {
                    self.dev
                        .conv_ar_gate288(
                            t,
                            &ax,
                            &l.attn_conv,
                            &cst_attn,
                            rws,
                            &hin_b,
                            blk_out,
                            &a.partial,
                            &s_prev,
                            &l.mlp_query,
                            &a.ar_out,
                            GateInputs {
                                msg: &mlp_msg,
                                ldm: 4 * H,
                                gin: &a.gin,
                                qsel: false,
                            },
                        )
                        .rt()?;
                }
                if bstart {
                    *nb += 1;
                }
                self.message_gates(l, 3, 1, &a.ar_out, 0, H, &mlp_msg, 4 * H, &a.mg_out, H, true, vmsg)?;
                self.fmark("mlp_norm")?;
                if rows_glue {
                    self.dev.norm_rows(t, &[&a.mg_out], H, &[&l.mlp_norm], &[&a.xm]).rt()?;
                } else {
                    self.dev.rmsnorm_f32(t, &a.mg_out, H, &l.mlp_norm, &a.xm).rt()?;
                }
            }
        } else {
            self.fmark("conv_partial")?;
            self.dev
                .hidden_conv(t, &ax, &l.attn_conv, &cst_attn, rws, 0, None, &a.y)
                .rt()?;
            self.tap(Some(v), "y", &a.y, TapType::Bf16, &[t, H])?;
            self.dev.partial_ref(t, &hin_b, &a.y, blk_out, &a.partial).rt()?;
            if bstart {
                *nb += 1;
            }
            self.fmark(if has_msgs { "ar_mlp_m" } else { "ar_mlp" })?;
            let mlp_msg = at(&a.msgs, 3 * H)?;
            self.attn_res(
                1,
                &l.mlp_query,
                *nb,
                None,
                Some(&a.partial),
                &a.ar_out,
                usize::from(has_msgs),
                Some(&mlp_msg),
                false,
            )?;
            if has_msgs {
                self.message_gates(l, 3, 1, &a.ar_out, 0, H, &mlp_msg, 4 * H, &a.mg_out, H, false, None)?;
            }
            self.fmark("mlp_norm")?;
            let src = if has_msgs { &a.mg_out } else { &a.ar_out };
            self.dev.rmsnorm_f32(t, src, H, &l.mlp_norm, &a.xm).rt()?;
        }
        self.tap(Some(v), "xm", &a.xm, TapType::Bf16, &[t, H])?;
        // ---- router, experts, MLP conv + residual
        self.mark("router")?;
        let fuse_combine_conv = !reference && !tapping && rows_glue;
        self.moe(v, l, &mx, !fuse_combine_conv)?;
        self.tap(Some(v), "moe", &mx, TapType::Bf16, &[t, H])?;
        self.mark("glue_end")?;
        let cst_mlp = at(&cst, 4352)?;
        if fuse_combine_conv {
            self.dev
                .moe_combine_conv(t, &a.my, &l.mlp_conv, &cst_mlp, rws, &a.len, &a.partial, &mx, hout_b)
                .rt()?;
        } else if rows_glue {
            let mode = ConvRows::Mlp { out: hout_b };
            self.dev
                .conv_rows(t, &mx, &l.mlp_conv, &cst_mlp, rws, &a.partial, mode)
                .rt()?;
        } else {
            self.dev
                .hidden_conv(t, &mx, &l.mlp_conv, &cst_mlp, rws, 1, Some(&a.partial), hout_b)
                .rt()?;
        }
        self.tap(Some(v), "out", hout_b, TapType::Bf16, &[t, H])?;
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------ the model

    /// `Engine::run_model`.
    pub fn run_model(&mut self) -> Res<()> {
        let t = self.t();
        let a = self.a;
        let w = self.w;
        self.mark("embed")?;
        self.dev
            .embed_norm(t, &a.ids, &w.embed, &w.embed_norm, &a.hid[0])
            .rt()?;
        self.tap(None, "emb", &a.hid[0], TapType::Bf16, &[t, H])?;
        let (mut cur, mut nb, mut v) = (0usize, 0usize, 0usize);
        let mut hin = Hin::Hid(0);
        let use_vmsgs = !self.reference && self.tap.is_none() && t > PGLUE_ROWS;
        let run_visit = |f: &mut Self,
                         msgs: bool,
                         vmsg_i: Option<usize>,
                         cur: &mut usize,
                         hin: &mut Hin,
                         nb: &mut usize,
                         v: &mut usize|
         -> Res<()> {
            if matches!(*v, 5 | 23 | 41) {
                let j = match *v {
                    5 => 0,
                    23 => 1,
                    _ => 2,
                };
                f.dev.feat_copy(t, Some(f.hin_buf(*hin)), None, &a.feats, j).rt()?;
            }
            f.visit(*v, *hin, *cur ^ 1, nb, msgs, vmsg_i)?;
            *cur ^= 1;
            *hin = Hin::Hid(*cur);
            *v += 1;
            Ok(())
        };
        for _ in 0..LOOP_START {
            run_visit(self, false, None, &mut cur, &mut hin, &mut nb, &mut v)?;
        }
        // blk[0] = embedding (layer 0), blk[1] = pre_hidden (appended at layer 6 of every pass)
        for p in 0..PASSES {
            if p > 0 {
                let n = t * H;
                self.st.dtod(&span(&a.msrc[0], 0, n)?, &span(&a.blk[2], 0, n)?).rt()?;
                self.st
                    .dtod(&span(&a.msrc[1], 0, n)?, &span(self.hin_buf(hin), 0, n)?)
                    .rt()?;
                hin = Hin::Blk1;
                nb = 1;
                if use_vmsgs {
                    self.mark("msgs")?;
                    self.dev
                        .vmsg_weights(t, &a.msrc[0], &a.msrc[1], &w.msg_queries, &a.msgs)
                        .rt()?;
                }
            }
            for i in 0..LOOP_END - LOOP_START {
                if p > 0 && !use_vmsgs {
                    self.mark("msgs")?;
                    let s = res_src(&[&a.msrc[0]], Some(&a.msrc[1]), None, H);
                    let mq = at(&w.msg_queries, i * 4 * H)?;
                    if !self.reference {
                        self.dev.ar288(4, 0, t, &s, &mq, &a.msgs, None).rt()?;
                    } else {
                        self.dev
                            .attn_res(4, 0, t, &s, &mq, &a.msgs, None, self.decode_only())
                            .rt()?;
                    }
                    self.tap(Some(v), "msgs", &a.msgs, TapType::F32, &[t, 4, H])?;
                }
                let vi = (p > 0 && use_vmsgs).then_some(i);
                run_visit(self, p > 0, vi, &mut cur, &mut hin, &mut nb, &mut v)?;
            }
        }
        for _ in LOOP_END..NLAYERS {
            run_visit(self, false, None, &mut cur, &mut hin, &mut nb, &mut v)?;
        }
        if v != NVISITS || nb != 4 {
            return Err(RuntimeError::request("visit count / blocks"));
        }
        self.mark("head")?;
        let hb = self.hin_buf(hin).clone();
        self.attn_res(1, &w.final_query, nb, Some(&hb), None, &a.ar_out, 0, None, false)?;
        self.dev.rmsnorm_f32out(t, &a.ar_out, &w.final_norm, &a.h32).rt()?;
        self.tap(None, "h", &a.h32, TapType::F32, &[t, H])?;
        self.dev.feat_copy(t, None, Some(&a.h32), &a.feats, 3).rt()?;
        let nlog = self.m.lrows.len();
        if nlog > 0 {
            self.dev.gather_rows(nlog, t, &a.h32, &a.logit_rows, &a.hrows).rt()?;
            for g in self.m.lgroups.clone() {
                self.run(
                    class(g.dec),
                    (VOCAB_PAD, H, H),
                    &a.hrows,
                    g.r0 * H,
                    H,
                    &w.unembed,
                    0,
                    &a.logits,
                    g.r0 * VOCAB_PAD,
                    VOCAB_PAD,
                    g.n,
                )?;
            }
            self.dev.softcap(nlog, &a.logits).rt()?;
        }
        self.mark("end")?;
        Ok(())
    }
}
