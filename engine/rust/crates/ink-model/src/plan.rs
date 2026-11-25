//! The device memory plan: the arena size of `Engine::Engine` and the activation, KV and workspace buffers of
//! `Engine::alloc` (`engine/native/src/engine.cu` @ b2dc42b), taken in the C++ order with the C++ names right after
//! the weights.
//!
//! [`Activations`] is generic over [`Alloc`] so that one function makes the plan both on the device ([`Arena`],
//! typed [`DevBuf`]s) and on the host ([`ArenaPlan`], offsets only: tests and the arena map without a GPU).

use bytemuck::Pod;
use ink_cuda::{Arena, ArenaPlan, Bf16, CudaError, DevBuf};

use crate::dims::{
    DREL, GATE_RANK, H, HD, INTER, KD, KVH, NEXP, NH, NLAYERS, NVISITS, QD, REL_EXT, TOPK, VOCAB_PAD, WINDOW,
};
use crate::weights::WEIGHTS_BYTES_ESTIMATE;

/// Page-table row stride (262,144 + 2,048 tokens = 258 pages of 1,024).
pub const KPT: usize = 264;
/// Conv-state row stride: k 1024 | v 1024 | attn 2304 | mlp 2304 (`glue::SLD`).
pub const SLD: usize = 6656;
/// Elements of one KV page: 8 heads x 1,024 tokens x 128.
pub const PAGE_ELEMS: usize = KVH * 1024 * HD;
/// Sliding-window visits (rings per slot).
pub const RING_VISITS: usize = 40;
/// Full-attention visits (global pool layers).
pub const FULL_VISITS: usize = 8;
/// Feature tiles per expert item (`moedev::FT`: 2,304 / 64 / 4).
pub const MOE_FT: usize = 9;
/// Token rows per prefill_moe CTA tile (`pmoe::BN`).
pub const MOE_BN: usize = 128;
/// Feature tiles per expert item for decode (`moedev::FT_DEC`: 2,304 / 64 / 2, WARPS = 4 decode).
pub const MOE_DEC_FT: usize = 18;
/// Token rows per decode MoE CTA tile (`pmoe::BN_DEC`).
pub const MOE_DEC_BN: usize = 16;
/// Decode rows per call projected through the fused weight (`kFusedRows`: rows of `qkvf`).
pub const FUSED_ROWS: usize = 128;
/// Columns of the fused projection output (`kFusedCols`): q | gate | r | sink | k + k_dyn | v + v_dyn.
pub const FUSED_COLS: usize = 2 * QD + NH * DREL + NH + 2 * (KD + 32);
/// Routed rows per block of the prefill routing build (`pglue::kBuildRows`).
pub const MOE_BUILD_ROWS: usize = 128;
/// `kLtWorkspace`: the cuBLASLt workspace of split-K GEMM plans (G1: the decode gate-down shape), 16 MB.
pub const LT_WORKSPACE: usize = 16 << 20;
/// Prefill-attention split-partial floats of the engine's variant: `pfa::ws_floats_per_partial<4, 2>()` = 8 KV
/// heads x 4 warps x 32 x (64 x 2 + 4 x 2).
pub const PA_WS_FLOATS_PER_PARTIAL: usize = 8 * 4 * 32 * (64 * 2 + 4 * 2);
/// `sizeof(pmoe::Item)`, `sizeof(pfa::Seg)`, `sizeof(int4)`: four ints each.
pub const INT4_BYTES: usize = 16;

/// Sizes that shape the plan (`EngineConfig`; defaults as in C++ and `ink_rt.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemConfig {
    /// Sessions with state on the card.
    pub max_slots: usize,
    /// Global KV pages of 1,024 tokens.
    pub num_pages: usize,
    /// Rows per forward call.
    pub t_max: usize,
    /// Rows per prefill segment; sets the sliding-window ring (`max_chunk`).
    pub max_chunk: usize,
    /// Rows whose logits one call can return.
    pub logit_rows_max: usize,
    /// Store global-layer K/V cache in FP8 E4M3 (1 byte/elem instead of 2; SWA rings stay bf16).
    pub kv_fp8: bool,
}

impl Default for MemConfig {
    fn default() -> Self {
        MemConfig {
            max_slots: 8,
            num_pages: 256,
            t_max: 2048 + 64,
            max_chunk: 2048,
            logit_rows_max: 128,
            kv_fp8: false,
        }
    }
}

/// `moedev::items_bound`.
pub fn moe_items_bound(rows: usize) -> usize {
    MOE_FT * (rows / MOE_BN + 1 + NEXP.min(rows))
}

/// `moedev::dec_items_bound`: tight upper bound on decode MoE items (`FT_DEC` x `ceil(cnt[e] / BN_DEC)`).
pub fn moe_dec_items_bound(rows: usize) -> usize {
    let k = NEXP.min(rows);
    MOE_DEC_FT * (k + (rows - k) / MOE_DEC_BN)
}

impl MemConfig {
    /// Arena bytes, the C++ constructor's sum term by term (weights estimate, KV, scratch with its margins).
    pub fn arena_bytes(&self) -> usize {
        let (t, s) = (self.t_max, self.max_slots);
        let mut b = WEIGHTS_BYTES_ESTIMATE;
        let kv_elem = if self.kv_fp8 { 1 } else { 2 };
        b += 2 * FULL_VISITS * self.num_pages * PAGE_ELEMS * kv_elem; // global pool K + V
        b += 2 * RING_VISITS * s * self.ring_pages() * PAGE_ELEMS * 2; // rings
        b += s * NVISITS * 3 * SLD * 2; // conv states
        b += NVISITS * 4 * t * H * 2; // convx
        b += t * H * 4 * 60 + t * NH * (REL_EXT + 64) * 2 * 2; // scratch (generous)
        b += 8 * t * (INTER + H) * 2 + 8 * t * 16 + NEXP * 64 * 2 * 16 * 2; // MoE
        b += self.pa_ws_bytes() + (40 << 20); // prefill attn workspace
        b += s * 8 * 100 * 48 * 128 * 4 + (16 << 20); // decode partials (up to 16 rows = 48 cols/session)
        b += self.logit_rows_max * (VOCAB_PAD * 4 + H * 2);
        b += 64 << 20;
        b += LT_WORKSPACE; // split-K GEMM workspace
        b += NLAYERS * 4 * 2 * H * GATE_RANK * 2 + (1 << 20); // tensor-core gate tail weights (G7)
        b
    }

    /// Pages of a sliding-window ring per slot and visit (`ring_`: window 1,024 + `max_chunk` - 1 keys): 3 at 2,048
    /// rows, 5 at 4,096.
    pub fn ring_pages(&self) -> usize {
        (WINDOW + self.max_chunk - 1).div_ceil(1024)
    }

    /// Prefill-attention split workspace (`pa_ws_bytes_for(t_max)`).
    pub fn pa_ws_bytes(&self) -> usize {
        (3 * self.t_max / 128 + 2) * 18 * PA_WS_FLOATS_PER_PARTIAL * 4
    }

    /// Prefill-attention tile bound (`pa_tiles_cap_`).
    pub fn pa_tiles_cap(&self) -> usize {
        self.max_slots * (3 * self.t_max / 128 + 2)
    }

    /// Prefill-attention item bound (`pa_items_cap_`).
    pub fn pa_items_cap(&self) -> usize {
        self.pa_tiles_cap() * 24
    }
}

/// Where the plan's buffers come from: the device arena or host bookkeeping.
pub trait Alloc {
    /// A region of `T` elements.
    type Buf<T: Pod>;
    /// Takes `n` elements of `T` at the next 256-byte boundary, named `name` in the arena map.
    ///
    /// # Errors
    ///
    /// [`CudaError::ArenaExhausted`] when it does not fit.
    fn take<T: Pod>(&mut self, n: usize, name: &str) -> Result<Self::Buf<T>, CudaError>;
}

impl Alloc for Arena {
    type Buf<T: Pod> = DevBuf<T>;
    fn take<T: Pod>(&mut self, n: usize, name: &str) -> Result<DevBuf<T>, CudaError> {
        Arena::take(self, n, name)
    }
}

impl Alloc for ArenaPlan {
    /// The byte offset.
    type Buf<T: Pod> = usize;
    fn take<T: Pod>(&mut self, n: usize, name: &str) -> Result<usize, CudaError> {
        let bytes = n.checked_mul(size_of::<T>()).ok_or_else(|| CudaError::ArenaExhausted {
            what: name.to_owned(),
            need: usize::MAX,
            cap: self.cap(),
        })?;
        ArenaPlan::take(self, bytes, name)
    }
}

/// Every buffer of `Engine::alloc`, in take order. Field names are the C++ members without the trailing `_`; the
/// arena names are the C++ `what` strings.
#[allow(missing_docs)] // one field per C++ member, named as there
pub struct Activations<A: Alloc> {
    pub kpool: A::Buf<u8>,
    pub vpool: A::Buf<u8>,
    pub kring: A::Buf<Bf16>,
    pub vring: A::Buf<Bf16>,
    pub conv: A::Buf<Bf16>,
    pub convx: A::Buf<Bf16>,
    pub ids: A::Buf<i32>,
    pub pos: A::Buf<i32>,
    pub seg: A::Buf<i32>,
    pub row0: A::Buf<i32>,
    pub slot: A::Buf<i32>,
    pub len: A::Buf<i32>,
    pub cnt: A::Buf<i32>,
    pub logit_rows: A::Buf<i32>,
    pub pt_g: A::Buf<i32>,
    pub pt_r: A::Buf<i32>,
    pub hid: [A::Buf<Bf16>; 2],
    pub blk: [A::Buf<Bf16>; 4],
    pub msrc: [A::Buf<Bf16>; 2],
    pub msgs: A::Buf<f32>,
    pub ar_out: A::Buf<f32>,
    pub mg_out: A::Buf<f32>,
    pub mg_tmp: [A::Buf<f32>; 2],
    pub gd32: A::Buf<f32>,
    pub gin: A::Buf<Bf16>,
    pub gd: A::Buf<Bf16>,
    pub lt_ws: A::Buf<u8>,
    pub xq: A::Buf<Bf16>,
    pub xk: A::Buf<Bf16>,
    pub xv: A::Buf<Bf16>,
    pub xm: A::Buf<Bf16>,
    pub qkvf: A::Buf<Bf16>,
    pub qproj: A::Buf<Bf16>,
    pub rproj: A::Buf<Bf16>,
    pub kdyn: A::Buf<Bf16>,
    pub vdyn: A::Buf<Bf16>,
    pub q: A::Buf<Bf16>,
    pub rel: A::Buf<Bf16>,
    pub rel_logits: A::Buf<Bf16>,
    pub relk: A::Buf<Bf16>,
    pub k: A::Buf<Bf16>,
    pub v: A::Buf<Bf16>,
    pub sink: A::Buf<Bf16>,
    pub gate: A::Buf<Bf16>,
    pub attn: A::Buf<Bf16>,
    pub y: A::Buf<Bf16>,
    pub partial: A::Buf<f32>,
    pub m32: A::Buf<f32>,
    pub h32: A::Buf<f32>,
    pub ridx_all: A::Buf<i32>,
    pub rw_all: A::Buf<f32>,
    pub mh: A::Buf<Bf16>,
    pub my: A::Buf<Bf16>,
    pub m_arow13: A::Buf<i32>,
    pub m_arow2: A::Buf<i32>,
    pub m_orow: A::Buf<i32>,
    pub m_gamma: A::Buf<f32>,
    pub m_items_cap: usize,
    pub m_nitems: A::Buf<i32>,
    pub m_cntb: A::Buf<i32>,
    pub m_baseb: A::Buf<i32>,
    /// `pmoe::Item` x `m_items_cap` (raw bytes, as the C++ `take`).
    pub m_items13: A::Buf<u8>,
    pub m_items2: A::Buf<u8>,
    pub pa_ws_floats: usize,
    pub pa_ws: A::Buf<f32>,
    pub pa_tiles_cap: usize,
    pub pa_items_cap: usize,
    pub pa_cnt: A::Buf<i32>,
    /// `pfa::Seg` x slots (raw bytes).
    pub pa_segs: A::Buf<u8>,
    /// `int4` x 2 x `pa_items_cap` (raw bytes).
    pub pa_items: A::Buf<u8>,
    pub pa_items2: A::Buf<u8>,
    pub da_opart_floats: usize,
    pub da_opart: A::Buf<f32>,
    /// `float2` pairs.
    pub da_ml: A::Buf<[f32; 2]>,
    pub da_ctx: A::Buf<i32>,
    pub da_row0: A::Buf<i32>,
    pub da_cs: A::Buf<i32>,
    pub da_nrows: A::Buf<i32>,
    pub hrows: A::Buf<Bf16>,
    pub feats: A::Buf<Bf16>,
    pub rlog: A::Buf<f32>,
    pub amax: A::Buf<i32>,
    pub logits: A::Buf<f32>,
}

impl<A: Alloc> Activations<A> {
    /// Takes every buffer of `Engine::alloc` from `ar`, in the C++ order.
    ///
    /// # Errors
    ///
    /// [`CudaError::ArenaExhausted`] naming the first buffer that does not fit.
    pub fn alloc(ar: &mut A, c: &MemConfig) -> Result<Self, CudaError> {
        let (t, s) = (c.t_max, c.max_slots);
        let kv_bytes = FULL_VISITS * c.num_pages * PAGE_ELEMS * if c.kv_fp8 { 1 } else { 2 };
        let kpool = ar.take::<u8>(kv_bytes, "kpool")?;
        let vpool = ar.take::<u8>(kv_bytes, "vpool")?;
        let kring = ar.take::<Bf16>(RING_VISITS * s * c.ring_pages() * PAGE_ELEMS, "kring")?;
        let vring = ar.take::<Bf16>(RING_VISITS * s * c.ring_pages() * PAGE_ELEMS, "vring")?;
        let conv = ar.take::<Bf16>(s * NVISITS * 3 * SLD, "conv")?;
        let convx = ar.take::<Bf16>(NVISITS * 4 * t * H, "convx")?;
        let ids = ar.take::<i32>(t, "ids")?;
        let pos = ar.take::<i32>(t, "pos")?;
        let seg = ar.take::<i32>(t, "seg")?;
        let row0 = ar.take::<i32>(s, "row0")?;
        let slot = ar.take::<i32>(s, "slot")?;
        let len = ar.take::<i32>(s, "len")?;
        let cnt = ar.take::<i32>(s, "cnt")?;
        let logit_rows = ar.take::<i32>(c.logit_rows_max, "logit rows")?;
        let pt_g = ar.take::<i32>(s * KPT * 2, "pt g")?;
        let pt_r = ar.take::<i32>(s * KPT * 2, "pt r")?;
        let hid = [ar.take::<Bf16>(t * H, "hid")?, ar.take::<Bf16>(t * H, "hid")?];
        let blk = [
            ar.take::<Bf16>(t * H, "blk")?,
            ar.take::<Bf16>(t * H, "blk")?,
            ar.take::<Bf16>(t * H, "blk")?,
            ar.take::<Bf16>(t * H, "blk")?,
        ];
        let msrc = [ar.take::<Bf16>(t * H, "msrc")?, ar.take::<Bf16>(t * H, "msrc")?];
        let msgs = ar.take::<f32>(t * 4 * H, "msgs")?;
        let ar_out = ar.take::<f32>(t * 4 * H, "ar_out")?;
        let mg_out = ar.take::<f32>(t * 3 * H, "mg_out")?;
        let mg_tmp = [ar.take::<f32>(t * H, "mg_tmp")?, ar.take::<f32>(t * H, "mg_tmp")?];
        let gd32 = ar.take::<f32>(t * GATE_RANK, "gd32")?;
        let gin = ar.take::<Bf16>(3 * t * 2 * H, "gin")?; // three gates' inputs
        let gd = ar.take::<Bf16>(3 * t * GATE_RANK, "gd")?;
        let lt_ws = ar.take::<u8>(LT_WORKSPACE, "lt workspace")?;
        let xq = ar.take::<Bf16>(t * H, "xq")?;
        let xk = ar.take::<Bf16>(t * H, "xk")?;
        let xv = ar.take::<Bf16>(t * H, "xv")?;
        let xm = ar.take::<Bf16>(t * H, "xm")?;
        let qkvf = ar.take::<Bf16>(FUSED_ROWS * FUSED_COLS, "qkv fused")?;
        let qproj = ar.take::<Bf16>(t * QD, "qproj")?;
        let rproj = ar.take::<Bf16>(t * NH * DREL, "rproj")?;
        let kdyn = ar.take::<Bf16>(t * 32, "kdyn")?;
        let vdyn = ar.take::<Bf16>(t * 32, "vdyn")?;
        let q = ar.take::<Bf16>(t * QD, "q")?;
        let rel = ar.take::<Bf16>(t * NH * DREL, "rel")?;
        let rel_logits = ar.take::<Bf16>(t * NH * REL_EXT, "rel_logits")?;
        let relk = ar.take::<Bf16>(t * NH * (REL_EXT + 64), "relk")?;
        let k = ar.take::<Bf16>(t * KD, "k")?;
        let v = ar.take::<Bf16>(t * KD, "v")?;
        let sink = ar.take::<Bf16>(t * NH + 64, "sink")?;
        let gate = ar.take::<Bf16>(t * QD, "gate")?;
        let attn = ar.take::<Bf16>(t * QD, "attn")?;
        let y = ar.take::<Bf16>(t * H, "y")?;
        let partial = ar.take::<f32>(t * H, "partial")?;
        let m32 = ar.take::<f32>(t * H, "m32")?;
        let h32 = ar.take::<f32>(t * H, "h32")?;
        // routing per visit (48 slices)
        let ridx_all = ar.take::<i32>(NVISITS * t * TOPK, "ridx")?;
        let rw_all = ar.take::<f32>(NVISITS * t * TOPK, "rw")?;
        let mh = ar.take::<Bf16>(t * TOPK * INTER, "moe h")?;
        let my = ar.take::<Bf16>(t * TOPK * H, "moe y")?;
        let m_arow13 = ar.take::<i32>(t * TOPK, "arow13")?;
        let m_arow2 = ar.take::<i32>(t * TOPK, "arow2")?;
        let m_orow = ar.take::<i32>(t * TOPK, "orow")?;
        let m_gamma = ar.take::<f32>(t * TOPK, "gamma")?;
        let m_items_cap = moe_items_bound(t * TOPK).max(moe_dec_items_bound(64 * TOPK));
        let m_nitems = ar.take::<i32>(4, "moe n_items")?;
        // prefill routing build: per-block expert counts and bases
        let build_blocks = (t * TOPK).div_ceil(MOE_BUILD_ROWS);
        let m_cntb = ar.take::<i32>(build_blocks * NEXP, "moe cntb")?;
        let m_baseb = ar.take::<i32>(build_blocks * NEXP, "moe baseb")?;
        let m_items13 = ar.take::<u8>(m_items_cap * INT4_BYTES, "items13")?;
        let m_items2 = ar.take::<u8>(m_items_cap * INT4_BYTES, "items2")?;
        let pa_ws_floats = c.pa_ws_bytes() / 4;
        let pa_ws = ar.take::<f32>(pa_ws_floats, "pa ws")?;
        let (pa_tiles_cap, pa_items_cap) = (c.pa_tiles_cap(), c.pa_items_cap());
        let pa_cnt = ar.take::<i32>(2 * pa_tiles_cap * 8, "pa cnt")?;
        let pa_segs = ar.take::<u8>(s * INT4_BYTES, "pa segs")?;
        let pa_items = ar.take::<u8>(2 * pa_items_cap * INT4_BYTES, "pa items")?;
        let pa_items2 = ar.take::<u8>(2 * pa_items_cap * INT4_BYTES, "pa items2")?;
        let da_opart_floats = s * 8 * 100 * 48 * 128;
        let da_opart = ar.take::<f32>(da_opart_floats, "da opart")?;
        let da_ml = ar.take::<[f32; 2]>(s * 8 * 100 * 48, "da ml")?;
        let da_ctx = ar.take::<i32>(s, "da ctx")?;
        let da_row0 = ar.take::<i32>(s, "da row0")?;
        let da_cs = ar.take::<i32>(2 * s, "da cs")?;
        let da_nrows = ar.take::<i32>(s, "da nrows")?;
        let hrows = ar.take::<Bf16>(c.logit_rows_max * H, "hrows")?;
        let feats = ar.take::<Bf16>(t * 4 * H, "dspark feats")?;
        let rlog = ar.take::<f32>(3 * t * NEXP, "router group logits")?;
        let amax = ar.take::<i32>(c.logit_rows_max, "argmax rows")?;
        let logits = ar.take::<f32>(c.logit_rows_max * VOCAB_PAD, "logits")?;
        Ok(Activations {
            kpool,
            vpool,
            kring,
            vring,
            conv,
            convx,
            ids,
            pos,
            seg,
            row0,
            slot,
            len,
            cnt,
            logit_rows,
            pt_g,
            pt_r,
            hid,
            blk,
            msrc,
            msgs,
            ar_out,
            mg_out,
            mg_tmp,
            gd32,
            gin,
            gd,
            lt_ws,
            xq,
            xk,
            xv,
            xm,
            qkvf,
            qproj,
            rproj,
            kdyn,
            vdyn,
            q,
            rel,
            rel_logits,
            relk,
            k,
            v,
            sink,
            gate,
            attn,
            y,
            partial,
            m32,
            h32,
            ridx_all,
            rw_all,
            mh,
            my,
            m_arow13,
            m_arow2,
            m_orow,
            m_gamma,
            m_items_cap,
            m_nitems,
            m_cntb,
            m_baseb,
            m_items13,
            m_items2,
            pa_ws_floats,
            pa_ws,
            pa_tiles_cap,
            pa_items_cap,
            pa_cnt,
            pa_segs,
            pa_items,
            pa_items2,
            da_opart_floats,
            da_opart,
            da_ml,
            da_ctx,
            da_row0,
            da_cs,
            da_nrows,
            hrows,
            feats,
            rlog,
            amax,
            logits,
        })
    }
}

/// The tensor-core message-gate tail's bf16 controller weights (G7, the C++ `tc_w_`): `[2][H][64]` per gate (forget
/// rows, then gain rows), four gates per layer with loop controllers, taken right after [`Activations`] in layer
/// order, gates 0..4, as "gate tail tc W". Filled on the device from the fp32 transposed controllers.
pub struct GateTailW<A: Alloc> {
    /// Indexed by layer; `None` for a layer without loop controllers.
    pub w: Vec<Option<[A::Buf<Bf16>; 4]>>,
}

impl<A: Alloc> GateTailW<A> {
    /// Elements of one gate's weights.
    pub const GATE_ELEMS: usize = 2 * H * GATE_RANK;

    /// Takes the weights of every layer whose entry of `looped` is true (one entry per layer).
    ///
    /// # Errors
    ///
    /// [`CudaError::ArenaExhausted`] when a take does not fit.
    pub fn alloc(ar: &mut A, looped: &[bool]) -> Result<Self, CudaError> {
        let mut w = Vec::with_capacity(looped.len());
        for &l in looped {
            w.push(if l {
                Some([
                    ar.take::<Bf16>(Self::GATE_ELEMS, "gate tail tc W")?,
                    ar.take::<Bf16>(Self::GATE_ELEMS, "gate tail tc W")?,
                    ar.take::<Bf16>(Self::GATE_ELEMS, "gate tail tc W")?,
                    ar.take::<Bf16>(Self::GATE_ELEMS, "gate tail tc W")?,
                ])
            } else {
                None
            });
        }
        Ok(GateTailW { w })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1 << 30;

    #[test]
    fn arena_bytes_default_by_hand() {
        // T = 2112, S = 8, 256 pages, 2,048-row chunks (3 ring pages), 128 logit rows; each term of the Engine
        // constructor evaluated by hand.
        let terms: [usize; 13] = [
            21_000_000_000,
            2 * 8 * 256 * 1_048_576 * 2,                                 // 8,589,934,592
            2 * 40 * 8 * 3 * 1_048_576 * 2,                              // 4,026,531,840
            8 * 48 * 3 * 6656 * 2,                                       // 15,335,424
            48 * 4 * 2112 * 2304 * 2,                                    // 1,868,562,432
            2112 * 2304 * 4 * 60 + 2112 * 24 * 1088 * 4,                 // 1,167,851,520 + 220,594,176
            8 * 2112 * 3456 * 2 + 8 * 2112 * 16 + 128 * 64 * 2 * 16 * 2, // 116,785,152 + 270,336 + 524,288
            51 * 18 * 139_264 * 4 + (40 << 20),                          // 511,377,408 + 41,943,040
            8 * 8 * 100 * 48 * 128 * 4 + (16 << 20),                     // 157,286,400 + 16,777,216
            128 * (201_024 * 4 + 2304 * 2),                              // 103,514,112
            64 << 20,                                                    // 67,108,864
            16 << 20,                                                    // 16,777,216 split-K GEMM workspace (G1)
            24 * 4 * 2 * 2304 * 64 * 2 + (1 << 20),                      // 56,623,104 + 1,048,576 (G7)
        ];
        let c = MemConfig::default();
        assert_eq!(c.arena_bytes(), terms.iter().sum::<usize>());
        assert_eq!(c.arena_bytes(), 37_978_845_696);
    }

    #[test]
    fn bounds_by_hand() {
        // items_bound(16896) = 9 * (16896 / 128 + 1 + min(128, 16896)) = 9 * (132 + 1 + 128)
        assert_eq!(moe_items_bound(2112 * 8), 2349);
        assert_eq!(moe_items_bound(5), 54); // 9 * (5 / 128 = 0, + 1, + min(128, 5) = 5)
        // decode items bound (FT_DEC=18, BN_DEC=16): exact tight bound k + (R - k) / 16
        assert_eq!(moe_dec_items_bound(32), 18 * 32);
        assert_eq!(moe_dec_items_bound(256), 18 * 136);
        assert_eq!(moe_dec_items_bound(512), 18 * 152);
        // 3 * 2112 / 128 = 49 (integer), + 2 = 51, x 8 slots
        let c = MemConfig::default();
        assert_eq!((c.pa_tiles_cap(), c.pa_items_cap()), (408, 9792));
        assert_eq!(c.ring_pages(), 3);
        assert_eq!(MemConfig { max_chunk: 4096, ..c }.ring_pages(), 5);
    }

    #[test]
    fn plan_offsets_by_hand() {
        let c = MemConfig::default();
        let mut p = ArenaPlan::new(c.arena_bytes());
        let a = Activations::alloc(&mut p, &c).unwrap();
        // kpool / vpool: 8 * 256 * 1,048,576 bf16 = 4 GiB each; rings 40 * 8 * 3 * 1,048,576 * 2 = 2,013,265,920
        assert_eq!((a.kpool, a.vpool, a.kring), (0, 4 * GIB, 8 * GIB));
        assert_eq!(a.vring, 8 * GIB + 2_013_265_920);
        assert_eq!(a.conv, 8 * GIB + 2 * 2_013_265_920);
        // conv 8 * 48 * 3 * 6656 * 2 = 15,335,424 (a multiple of 256)
        assert_eq!(a.convx, a.conv + 15_335_424);
        // ids: 2112 ints = 8448 bytes = 33 x 256 exactly; row0 after seg; 8-int buffers round up to 256
        assert_eq!(a.ids, a.convx + 48 * 4 * 2112 * 2304 * 2);
        assert_eq!(
            (a.pos, a.seg, a.row0),
            (a.ids + 8448, a.ids + 2 * 8448, a.ids + 3 * 8448)
        );
        assert_eq!(
            (a.slot, a.len, a.cnt, a.logit_rows),
            (a.row0 + 256, a.row0 + 512, a.row0 + 768, a.row0 + 1024)
        );
        // pt g: 8 * 264 * 2 ints = 16,896 bytes = 66 x 256 (logit_rows = 128 ints = 512 bytes)
        assert_eq!(
            (a.pt_g, a.pt_r, a.hid[0]),
            (a.row0 + 1536, a.row0 + 1536 + 16_896, a.row0 + 1536 + 2 * 16_896)
        );
        // sink: 2112 * 24 + 64 bf16 = 101,504 bytes -> next at a multiple of 256 (101,632)
        assert_eq!(a.gate, a.sink + 101_632);
        let names: Vec<&str> = p.regions().iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names.len(), 84);
        assert_eq!(&names[..3], ["kpool", "vpool", "kring"]);
        assert_eq!(
            &names[names.len() - 3..],
            ["router group logits", "argmax rows", "logits"]
        );
        assert_eq!(p.regions().last().unwrap().bytes, 128 * 201_024 * 4);
    }

    #[test]
    fn kv_fp8_halves_global_pool_bytes() {
        let bf = MemConfig::default();
        let fp8 = MemConfig { kv_fp8: true, ..bf };
        // Default 256 pages: bf16 = 8 GiB global KV, fp8 = 4 GiB global KV (4 GiB saved)
        assert_eq!(bf.arena_bytes() - fp8.arena_bytes(), 4 * GIB);
        let mut p = ArenaPlan::new(fp8.arena_bytes());
        let a = Activations::alloc(&mut p, &fp8).unwrap();
        assert_eq!((a.kpool, a.vpool, a.kring), (0, 2 * GIB, 4 * GIB));
    }
}
