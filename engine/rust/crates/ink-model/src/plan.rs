//! The device memory plan: exact arena sizing and the activation, KV and workspace buffers of [`Activations`].
//!
//! [`Activations`] is generic over [`Alloc`] so that one function makes the plan both on the device ([`Arena`],
//! typed [`DevBuf`]s) and on the host ([`ArenaPlan`], offsets only: exact `arena_bytes()`, tests and the arena map
//! without a GPU).
//!
//! Lifetime-disjoint per-visit scratch buffers share a single `scratch_pool` allocation (overlays A/B/C/D):
//! - Overlay A (Phase 1 `attn_in` / Phase 4 `glue_mid` / Head): `ar_out`, `mg_out`, `mg_tmp`, `gin`, `gd`, `gd32`,
//!   `y`, `h32`, `ax_work`.
//! - Overlay B (Phase 2 `proj`+`qkv_post`+`rel_logits`, then Phase 3a `decode_attn`): `rel_logits` at offset 0,
//!   followed by Phase 2 (`qkvf_pf`, `qproj`, `rproj`, `kdyn`, `vdyn`, `k`, `v`, `kx_work`, `vx_work`) aliased with
//!   Phase 3a (`da_opart`, `da_ml`).
//! - Overlay C (Phase 3b `prefill_attn`): `pa_ws`.
//! - Overlay D (Phase 5 `moe`+`glue_end`): `rlog`, `mh`, `my`, `mx_work`.

use bytemuck::Pod;
use ink_cuda::{Arena, ArenaPlan, Bf16, CudaError, DevBuf};

use crate::dims::{
    DREL, GATE_RANK, H, HD, INTER, KD, KVH, LOOP_END, LOOP_START, NEXP, NH, NLAYERS, NVISITS, QD, REL_EXT, TOPK,
    VOCAB_PAD,
};
use crate::weights::WEIGHTS_BYTES_ESTIMATE;

/// Alignment of every arena region and overlay sub-buffer, in bytes.
const ALIGN: usize = 256;
/// Page-table row stride (262,144 + 2,048 tokens = 258 pages of 1,024).
pub const KPT: usize = 264;
/// Conv-state row stride: k 1024 | v 1024 | attn 2304 | mlp 2304 (`glue::SLD`).
pub const SLD: usize = 6656;
/// Elements of one KV page: 8 heads x 1,024 tokens x 128.
pub const PAGE_ELEMS: usize = KVH * 1024 * HD;
/// Sliding-window visits (rings per slot).
pub const RING_VISITS: usize = 40;
/// Persistent sliding-window ring pages per slot and visit (2,048 tokens >= WINDOW 1,024 + MAX_DEC_ROWS 16).
pub const PERSIST_RING_PAGES: usize = 2;
/// Compact `convx` commit-tail rows reserved per slot (`MAX_DEC_ROWS` = 16; prefill segments only save their last 3).
pub const CONV_TAIL_ROWS_PER_SLOT: usize = 16;
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
/// Maximum global prefill-attention key splits per row tile up to `KPT * 1,024` = 270,336 tokens (8 + 4 + 5 = 17).
pub const PA_MAX_SPLITS: usize = 17;
/// Prefill-attention split-partial floats of the engine's variant: `pfa::ws_floats_per_partial<4, 2>()` = 8 KV
/// heads x 4 warps x 32 x (64 x 2 + 4 x 2).
pub const PA_WS_FLOATS_PER_PARTIAL: usize = 8 * 4 * 32 * (64 * 2 + 4 * 2);
/// `sizeof(pmoe::Item)`, `sizeof(pfa::Seg)`, `sizeof(int4)`: four ints each.
pub const INT4_BYTES: usize = 16;
/// Slack added to exact `ArenaPlan::used()` for weight alignment / rounding (actual alignment gaps < 256 KiB).
pub const ARENA_SLACK_BYTES: usize = 16 << 20;

/// Sizes that shape the plan (`EngineConfig`; defaults as in `ink_rt.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemConfig {
    /// Sessions with state on the card.
    pub max_slots: usize,
    /// Global KV pages of 1,024 tokens.
    pub num_pages: usize,
    /// Rows per forward call.
    pub t_max: usize,
    /// Rows per prefill segment (`max_chunk`).
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
    /// Exact arena bytes from a dry-run [`ArenaPlan`] over weights estimate, [`Activations`], [`GateTailW`],
    /// `rel_proj_t`, plus [`ARENA_SLACK_BYTES`].
    pub fn arena_bytes(&self) -> usize {
        let mut plan = ArenaPlan::new(usize::MAX);
        plan.take(WEIGHTS_BYTES_ESTIMATE, "weights")
            .expect("unbounded ArenaPlan cannot exhaust");
        Activations::alloc(&mut plan, self).expect("unbounded ArenaPlan cannot exhaust");
        let looped: Vec<bool> = (0..NLAYERS).map(|l| (LOOP_START..LOOP_END).contains(&l)).collect();
        GateTailW::alloc(&mut plan, &looped).expect("unbounded ArenaPlan cannot exhaust");
        for _ in 0..NLAYERS {
            plan.take(REL_EXT * DREL * 2, "rel_proj_t")
                .expect("unbounded ArenaPlan cannot exhaust");
        }
        plan.used() + ARENA_SLACK_BYTES
    }

    /// Persistent sliding-window ring pages per slot and visit (fixed at 2 pages = 2,048 tokens, independent of
    /// `max_chunk`).
    pub fn ring_pages(&self) -> usize {
        PERSIST_RING_PAGES
    }

    /// Shared per-visit sliding-window scratch pages for prefill chunks (reused across all 40 SWA visits).
    /// Each prefill segment of `m` rows needs at most `ceil(m / 1,024)` scratch pages; across up to `max_slots`
    /// segments totaling at most `t_max` rows, total scratch pages <= `t_max / 1,024 + max_slots`.
    pub fn swa_scratch_pages(&self) -> usize {
        self.t_max / 1024 + self.max_slots
    }

    /// Rows per visit stored in the compact `convx` commit-tail buffer (`max_slots * 16`).
    pub fn conv_tail_rows(&self) -> usize {
        self.max_slots * CONV_TAIL_ROWS_PER_SLOT
    }

    /// Prefill-attention split workspace (`pa_ws_bytes_for(t_max)`).
    pub fn pa_ws_bytes(&self) -> usize {
        (3 * self.t_max / 128 + 2) * PA_MAX_SPLITS * PA_WS_FLOATS_PER_PARTIAL * 4
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

    /// Typed sub-region of `n` elements of `T` at `byte_off` inside an already-allocated byte pool.
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] when the range overflows `base` or violates alignment.
    fn sub<T: Pod>(&self, base: &Self::Buf<u8>, byte_off: usize, n: usize) -> Result<Self::Buf<T>, CudaError>;
}

impl Alloc for Arena {
    type Buf<T: Pod> = DevBuf<T>;
    fn take<T: Pod>(&mut self, n: usize, name: &str) -> Result<DevBuf<T>, CudaError> {
        Arena::take(self, n, name)
    }
    fn sub<T: Pod>(&self, base: &DevBuf<u8>, byte_off: usize, n: usize) -> Result<DevBuf<T>, CudaError> {
        let bytes = n.checked_mul(size_of::<T>()).ok_or_else(|| CudaError::Bounds {
            what: "scratch sub",
            detail: "byte size overflow".into(),
        })?;
        let end = byte_off.checked_add(bytes).ok_or_else(|| CudaError::Bounds {
            what: "scratch sub",
            detail: "byte offset overflow".into(),
        })?;
        base.slice(byte_off..end)?.cast::<T>()
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
    fn sub<T: Pod>(&self, base: &usize, byte_off: usize, n: usize) -> Result<usize, CudaError> {
        let _bytes = n.checked_mul(size_of::<T>()).ok_or_else(|| CudaError::Bounds {
            what: "scratch sub",
            detail: "byte size overflow".into(),
        })?;
        base.checked_add(byte_off).ok_or_else(|| CudaError::Bounds {
            what: "scratch sub",
            detail: "offset overflow".into(),
        })
    }
}

/// Bump helper for laying out non-overlapping buffers within one lifetime overlay of `scratch_pool`.
#[derive(Clone, Copy)]
struct Overlay {
    off: usize,
}

impl Overlay {
    fn new() -> Self {
        Self { off: 0 }
    }

    fn alloc<T: Pod>(&mut self, n: usize) -> usize {
        let start = self.off.next_multiple_of(ALIGN);
        self.off = start + n * size_of::<T>();
        start
    }

    fn bytes(self) -> usize {
        self.off
    }
}

/// Every buffer of `Engine::alloc`. Persistent buffers are taken directly from `ar`; lifetime-disjoint per-visit
/// scratch buffers are typed views into `scratch_pool`.
#[allow(missing_docs)]
pub struct Activations<A: Alloc> {
    pub kpool: A::Buf<u8>,
    pub vpool: A::Buf<u8>,
    /// Persistent SWA rings `[RING_VISITS][max_slots][2]` followed contiguously by `[swa_scratch_pages]` K pages.
    pub kring: A::Buf<Bf16>,
    /// Persistent SWA rings `[RING_VISITS][max_slots][2]` followed contiguously by `[swa_scratch_pages]` V pages.
    pub vring: A::Buf<Bf16>,
    pub conv: A::Buf<Bf16>,
    /// Compact per-visit commit-tail buffer: `[NVISITS][conv_tail_rows * SLD]` bf16 (SLD-packed by kind).
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
    /// Per-visit SWA page table for prefill-carrying calls (`[RING_VISITS][nseg][KPT]`).
    pub pt_r_pf: A::Buf<i32>,
    pub hid: [A::Buf<Bf16>; 2],
    pub blk: [A::Buf<Bf16>; 4],
    pub msrc: [A::Buf<Bf16>; 2],
    pub msgs: A::Buf<f32>,
    pub lt_ws: A::Buf<u8>,
    pub xq: A::Buf<Bf16>,
    pub xk: A::Buf<Bf16>,
    pub xv: A::Buf<Bf16>,
    pub xm: A::Buf<Bf16>,
    pub qkvf: A::Buf<Bf16>,
    pub q: A::Buf<Bf16>,
    pub rel: A::Buf<Bf16>,
    pub sink: A::Buf<Bf16>,
    pub gate: A::Buf<Bf16>,
    pub attn: A::Buf<Bf16>,
    pub partial: A::Buf<f32>,
    /// Current-visit router indices and weights (`[t_max][TOPK]`).
    pub ridx: A::Buf<i32>,
    pub rw: A::Buf<f32>,
    pub m_arow13: A::Buf<i32>,
    pub m_arow2: A::Buf<i32>,
    pub m_orow: A::Buf<i32>,
    pub m_gamma: A::Buf<f32>,
    pub m_items_cap: usize,
    pub m_nitems: A::Buf<i32>,
    pub m_cntb: A::Buf<i32>,
    pub m_baseb: A::Buf<i32>,
    /// `pmoe::Item` x `m_items_cap` (raw bytes).
    pub m_items13: A::Buf<u8>,
    pub m_items2: A::Buf<u8>,
    pub pa_tiles_cap: usize,
    pub pa_items_cap: usize,
    pub pa_cnt: A::Buf<i32>,
    /// `pfa::Seg` x slots (raw bytes).
    pub pa_segs: A::Buf<u8>,
    /// `int4` x 2 x `pa_items_cap` (raw bytes).
    pub pa_items: A::Buf<u8>,
    pub pa_items2: A::Buf<u8>,
    pub da_ctx: A::Buf<i32>,
    pub da_row0: A::Buf<i32>,
    pub da_cs: A::Buf<i32>,
    pub da_nrows: A::Buf<i32>,
    pub hrows: A::Buf<Bf16>,
    pub feats: A::Buf<Bf16>,
    pub amax: A::Buf<i32>,
    pub logits: A::Buf<f32>,
    /// Shared pool backing the lifetime-disjoint per-visit scratch overlays below.
    pub scratch_pool: A::Buf<u8>,
    // --- Overlay A: attn_in / glue_mid / head ---
    pub ar_out: A::Buf<f32>,
    pub mg_out: A::Buf<f32>,
    pub mg_tmp: [A::Buf<f32>; 2],
    pub gin: A::Buf<Bf16>,
    pub gd: A::Buf<Bf16>,
    pub gd32: A::Buf<f32>,
    pub y: A::Buf<Bf16>,
    pub h32: A::Buf<f32>,
    pub ax_work: A::Buf<Bf16>,
    // --- Overlay B: proj + qkv_post + rel_logits (B1) / decode_attn (B2) ---
    pub rel_logits: A::Buf<Bf16>,
    /// Fused prefill QKV projection output (`[t_max][FUSED_COLS]` bf16).
    pub qkvf_pf: A::Buf<Bf16>,
    pub qproj: A::Buf<Bf16>,
    pub rproj: A::Buf<Bf16>,
    pub kdyn: A::Buf<Bf16>,
    pub vdyn: A::Buf<Bf16>,
    pub k: A::Buf<Bf16>,
    pub v: A::Buf<Bf16>,
    pub kx_work: A::Buf<Bf16>,
    pub vx_work: A::Buf<Bf16>,
    pub da_opart_floats: usize,
    pub da_opart: A::Buf<f32>,
    /// `float2` pairs.
    pub da_ml: A::Buf<[f32; 2]>,
    // --- Overlay C: prefill_attn ---
    pub pa_ws_floats: usize,
    pub pa_ws: A::Buf<f32>,
    // --- Overlay D: moe + glue_end ---
    pub rlog: A::Buf<f32>,
    pub mh: A::Buf<Bf16>,
    pub my: A::Buf<Bf16>,
    pub mx_work: A::Buf<Bf16>,
}

impl<A: Alloc> Activations<A> {
    /// Takes every buffer of `Engine::alloc` from `ar`.
    ///
    /// # Errors
    ///
    /// [`CudaError::ArenaExhausted`] naming the first buffer that does not fit.
    pub fn alloc(ar: &mut A, c: &MemConfig) -> Result<Self, CudaError> {
        let (t, s) = (c.t_max, c.max_slots);
        let kv_bytes = FULL_VISITS * c.num_pages * PAGE_ELEMS * if c.kv_fp8 { 1 } else { 2 };
        let kpool = ar.take::<u8>(kv_bytes, "kpool")?;
        let vpool = ar.take::<u8>(kv_bytes, "vpool")?;
        let ring_total_pages = RING_VISITS * s * c.ring_pages() + c.swa_scratch_pages();
        let kring = ar.take::<Bf16>(ring_total_pages * PAGE_ELEMS, "kring")?;
        let vring = ar.take::<Bf16>(ring_total_pages * PAGE_ELEMS, "vring")?;
        let conv = ar.take::<Bf16>(s * NVISITS * 3 * SLD, "conv")?;
        let convx = ar.take::<Bf16>(NVISITS * c.conv_tail_rows() * SLD, "convx")?;
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
        let pt_r_pf = ar.take::<i32>(RING_VISITS * s * KPT, "pt r pf")?;
        let hid = [ar.take::<Bf16>(t * H, "hid")?, ar.take::<Bf16>(t * H, "hid")?];
        let blk = [
            ar.take::<Bf16>(t * H, "blk")?,
            ar.take::<Bf16>(t * H, "blk")?,
            ar.take::<Bf16>(t * H, "blk")?,
            ar.take::<Bf16>(t * H, "blk")?,
        ];
        let msrc = [ar.take::<Bf16>(t * H, "msrc")?, ar.take::<Bf16>(t * H, "msrc")?];
        let msgs = ar.take::<f32>(t * 4 * H, "msgs")?;
        let lt_ws = ar.take::<u8>(LT_WORKSPACE, "lt workspace")?;
        let xq = ar.take::<Bf16>(t * H, "xq")?;
        let xk = ar.take::<Bf16>(t * H, "xk")?;
        let xv = ar.take::<Bf16>(t * H, "xv")?;
        let xm = ar.take::<Bf16>(t * H, "xm")?;
        let qkvf = ar.take::<Bf16>(FUSED_ROWS * FUSED_COLS, "qkv fused")?;
        let q = ar.take::<Bf16>(t * QD, "q")?;
        let rel = ar.take::<Bf16>(t * NH * DREL, "rel")?;
        let sink = ar.take::<Bf16>(t * NH + 64, "sink")?;
        let gate = ar.take::<Bf16>(t * QD, "gate")?;
        let attn = ar.take::<Bf16>(t * QD, "attn")?;
        let partial = ar.take::<f32>(t * H, "partial")?;
        let ridx = ar.take::<i32>(t * TOPK, "ridx")?;
        let rw = ar.take::<f32>(t * TOPK, "rw")?;
        let m_arow13 = ar.take::<i32>(t * TOPK, "arow13")?;
        let m_arow2 = ar.take::<i32>(t * TOPK, "arow2")?;
        let m_orow = ar.take::<i32>(t * TOPK, "orow")?;
        let m_gamma = ar.take::<f32>(t * TOPK, "gamma")?;
        let m_items_cap = moe_items_bound(t * TOPK).max(moe_dec_items_bound(64 * TOPK));
        let m_nitems = ar.take::<i32>(4, "moe n_items")?;
        let build_blocks = (t * TOPK).div_ceil(MOE_BUILD_ROWS);
        let m_cntb = ar.take::<i32>(build_blocks * NEXP, "moe cntb")?;
        let m_baseb = ar.take::<i32>(build_blocks * NEXP, "moe baseb")?;
        let m_items13 = ar.take::<u8>(m_items_cap * INT4_BYTES, "items13")?;
        let m_items2 = ar.take::<u8>(m_items_cap * INT4_BYTES, "items2")?;
        let (pa_tiles_cap, pa_items_cap) = (c.pa_tiles_cap(), c.pa_items_cap());
        let pa_cnt = ar.take::<i32>(2 * pa_tiles_cap * 8, "pa cnt")?;
        let pa_segs = ar.take::<u8>(s * INT4_BYTES, "pa segs")?;
        let pa_items = ar.take::<u8>(2 * pa_items_cap * INT4_BYTES, "pa items")?;
        let pa_items2 = ar.take::<u8>(2 * pa_items_cap * INT4_BYTES, "pa items2")?;
        let da_ctx = ar.take::<i32>(s, "da ctx")?;
        let da_row0 = ar.take::<i32>(s, "da row0")?;
        let da_cs = ar.take::<i32>(2 * s, "da cs")?;
        let da_nrows = ar.take::<i32>(s, "da nrows")?;
        let hrows = ar.take::<Bf16>(c.logit_rows_max * H, "hrows")?;
        let feats = ar.take::<Bf16>(t * 4 * H, "dspark feats")?;
        let amax = ar.take::<i32>(c.logit_rows_max, "argmax rows")?;
        let logits = ar.take::<f32>(c.logit_rows_max * VOCAB_PAD, "logits")?;

        // --- Shared lifetime-disjoint per-visit scratch overlays ---
        // Overlay A: attn_in / glue_mid / head
        let mut ov_a = Overlay::new();
        let off_ar_out = ov_a.alloc::<f32>(t * 3 * H);
        let off_mg_out = ov_a.alloc::<f32>(t * 3 * H);
        let off_mg_tmp0 = ov_a.alloc::<f32>(t * H);
        let off_mg_tmp1 = ov_a.alloc::<f32>(t * H);
        let off_gin = ov_a.alloc::<Bf16>(3 * t * 2 * H);
        let off_gd = ov_a.alloc::<Bf16>(3 * t * GATE_RANK);
        let off_gd32 = ov_a.alloc::<f32>(t * GATE_RANK);
        let off_y = ov_a.alloc::<Bf16>(t * H);
        let off_h32 = ov_a.alloc::<f32>(t * H);
        let off_ax_work = ov_a.alloc::<Bf16>(t * H);

        // Overlay B: rel_logits (live Phase 2 -> 3a) + disjoint B1 (Phase 2) / B2 (Phase 3a decode_attn)
        let mut ov_b = Overlay::new();
        let off_rel_logits = ov_b.alloc::<Bf16>(t * NH * REL_EXT);
        let mut ov_b1 = ov_b;
        let off_qkvf_pf = ov_b1.alloc::<Bf16>(t * FUSED_COLS);
        let off_qproj = ov_b1.alloc::<Bf16>(t * QD);
        let off_rproj = ov_b1.alloc::<Bf16>(t * NH * DREL);
        let off_kdyn = ov_b1.alloc::<Bf16>(t * 32);
        let off_vdyn = ov_b1.alloc::<Bf16>(t * 32);
        let off_k = ov_b1.alloc::<Bf16>(t * KD);
        let off_v = ov_b1.alloc::<Bf16>(t * KD);
        let off_kx_work = ov_b1.alloc::<Bf16>(t * KD);
        let off_vx_work = ov_b1.alloc::<Bf16>(t * KD);

        let da_opart_floats = s * 8 * 100 * 48 * 128;
        let mut ov_b2 = ov_b;
        let off_da_opart = ov_b2.alloc::<f32>(da_opart_floats);
        let off_da_ml = ov_b2.alloc::<[f32; 2]>(s * 8 * 100 * 48);

        // Overlay C: prefill_attn
        let pa_ws_floats = c.pa_ws_bytes() / 4;
        let mut ov_c = Overlay::new();
        let off_pa_ws = ov_c.alloc::<f32>(pa_ws_floats);

        // Overlay D: moe + glue_end
        let rlog_elems = (3 * t * NEXP).max(9 * 128 * NEXP);
        let mut ov_d = Overlay::new();
        let off_rlog = ov_d.alloc::<f32>(rlog_elems);
        let off_mh = ov_d.alloc::<Bf16>(t * TOPK * INTER);
        let off_my = ov_d.alloc::<Bf16>(t * TOPK * H);
        let off_mx_work = ov_d.alloc::<Bf16>(t * H);

        let scratch_bytes = ov_a
            .bytes()
            .max(ov_b1.bytes())
            .max(ov_b2.bytes())
            .max(ov_c.bytes())
            .max(ov_d.bytes());
        let scratch_pool = ar.take::<u8>(scratch_bytes, "scratch pool")?;

        let ar_out = ar.sub::<f32>(&scratch_pool, off_ar_out, t * 3 * H)?;
        let mg_out = ar.sub::<f32>(&scratch_pool, off_mg_out, t * 3 * H)?;
        let mg_tmp = [
            ar.sub::<f32>(&scratch_pool, off_mg_tmp0, t * H)?,
            ar.sub::<f32>(&scratch_pool, off_mg_tmp1, t * H)?,
        ];
        let gin = ar.sub::<Bf16>(&scratch_pool, off_gin, 3 * t * 2 * H)?;
        let gd = ar.sub::<Bf16>(&scratch_pool, off_gd, 3 * t * GATE_RANK)?;
        let gd32 = ar.sub::<f32>(&scratch_pool, off_gd32, t * GATE_RANK)?;
        let y = ar.sub::<Bf16>(&scratch_pool, off_y, t * H)?;
        let h32 = ar.sub::<f32>(&scratch_pool, off_h32, t * H)?;
        let ax_work = ar.sub::<Bf16>(&scratch_pool, off_ax_work, t * H)?;

        let rel_logits = ar.sub::<Bf16>(&scratch_pool, off_rel_logits, t * NH * REL_EXT)?;
        let qkvf_pf = ar.sub::<Bf16>(&scratch_pool, off_qkvf_pf, t * FUSED_COLS)?;
        let qproj = ar.sub::<Bf16>(&scratch_pool, off_qproj, t * QD)?;
        let rproj = ar.sub::<Bf16>(&scratch_pool, off_rproj, t * NH * DREL)?;
        let kdyn = ar.sub::<Bf16>(&scratch_pool, off_kdyn, t * 32)?;
        let vdyn = ar.sub::<Bf16>(&scratch_pool, off_vdyn, t * 32)?;
        let k = ar.sub::<Bf16>(&scratch_pool, off_k, t * KD)?;
        let v = ar.sub::<Bf16>(&scratch_pool, off_v, t * KD)?;
        let kx_work = ar.sub::<Bf16>(&scratch_pool, off_kx_work, t * KD)?;
        let vx_work = ar.sub::<Bf16>(&scratch_pool, off_vx_work, t * KD)?;
        let da_opart = ar.sub::<f32>(&scratch_pool, off_da_opart, da_opart_floats)?;
        let da_ml = ar.sub::<[f32; 2]>(&scratch_pool, off_da_ml, s * 8 * 100 * 48)?;

        let pa_ws = ar.sub::<f32>(&scratch_pool, off_pa_ws, pa_ws_floats)?;

        let rlog = ar.sub::<f32>(&scratch_pool, off_rlog, rlog_elems)?;
        let mh = ar.sub::<Bf16>(&scratch_pool, off_mh, t * TOPK * INTER)?;
        let my = ar.sub::<Bf16>(&scratch_pool, off_my, t * TOPK * H)?;
        let mx_work = ar.sub::<Bf16>(&scratch_pool, off_mx_work, t * H)?;

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
            pt_r_pf,
            hid,
            blk,
            msrc,
            msgs,
            lt_ws,
            xq,
            xk,
            xv,
            xm,
            qkvf,
            q,
            rel,
            sink,
            gate,
            attn,
            partial,
            ridx,
            rw,
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
            pa_tiles_cap,
            pa_items_cap,
            pa_cnt,
            pa_segs,
            pa_items,
            pa_items2,
            da_ctx,
            da_row0,
            da_cs,
            da_nrows,
            hrows,
            feats,
            amax,
            logits,
            scratch_pool,
            ar_out,
            mg_out,
            mg_tmp,
            gin,
            gd,
            gd32,
            y,
            h32,
            ax_work,
            rel_logits,
            qkvf_pf,
            qproj,
            rproj,
            kdyn,
            vdyn,
            k,
            v,
            kx_work,
            vx_work,
            da_opart_floats,
            da_opart,
            da_ml,
            pa_ws_floats,
            pa_ws,
            rlog,
            mh,
            my,
            mx_work,
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
    fn arena_bytes_matches_exact_plan_plus_slack() {
        for chunk in [2048, 3072, 4096, 6144, 8192, 12288] {
            let c = MemConfig {
                max_chunk: chunk,
                t_max: chunk + 128,
                ..MemConfig::default()
            };
            let cap = c.arena_bytes();
            let mut p = ArenaPlan::new(cap);
            p.take(WEIGHTS_BYTES_ESTIMATE, "weights").unwrap();
            let _a = Activations::alloc(&mut p, &c).unwrap();
            let looped: Vec<bool> = (0..NLAYERS).map(|l| (LOOP_START..LOOP_END).contains(&l)).collect();
            let _tc = GateTailW::alloc(&mut p, &looped).unwrap();
            for i in 0..NLAYERS {
                p.take(REL_EXT * DREL * 2, &format!("rel_proj_t[{i}]")).unwrap();
            }
            assert_eq!(cap - p.used(), ARENA_SLACK_BYTES);
        }
    }

    #[test]
    fn bounds_and_decoupled_rings() {
        assert_eq!(moe_items_bound(2112 * 8), 2349);
        assert_eq!(moe_items_bound(5), 54);
        assert_eq!(moe_dec_items_bound(32), 18 * 32);
        assert_eq!(moe_dec_items_bound(256), 18 * 136);
        assert_eq!(moe_dec_items_bound(512), 18 * 152);
        let c = MemConfig::default();
        assert_eq!((c.pa_tiles_cap(), c.pa_items_cap()), (408, 9792));
        // Persistent SWA ring is always 2 pages regardless of max_chunk.
        assert_eq!(c.ring_pages(), 2);
        assert_eq!(MemConfig { max_chunk: 4096, ..c }.ring_pages(), 2);
        assert_eq!(MemConfig { max_chunk: 12288, ..c }.ring_pages(), 2);
        // Shared SWA scratch scales with t_max / 1024 + max_slots.
        assert_eq!(c.swa_scratch_pages(), 2 + 8);
        assert_eq!(
            MemConfig {
                t_max: 12416,
                max_chunk: 12288,
                ..c
            }
            .swa_scratch_pages(),
            12 + 8
        );
        assert_eq!(c.conv_tail_rows(), 8 * 16);
    }

    #[test]
    fn plan_offsets_and_overlays() {
        let c = MemConfig::default();
        let mut p = ArenaPlan::new(c.arena_bytes());
        let a = Activations::alloc(&mut p, &c).unwrap();
        // kpool / vpool: 8 * 256 * 1,048,576 bf16 = 4 GiB each.
        assert_eq!((a.kpool, a.vpool, a.kring), (0, 4 * GIB, 8 * GIB));
        // kring / vring: (40 * 8 * 2 + 10) = 650 pages of 2 MiB each = 1,363,148,800 bytes.
        let ring_bytes = (40 * 8 * 2 + 10) * PAGE_ELEMS * 2;
        assert_eq!(a.vring, 8 * GIB + ring_bytes);
        assert_eq!(a.conv, 8 * GIB + 2 * ring_bytes);
        // conv: 8 * 48 * 3 * 6656 * 2 = 15,335,424
        assert_eq!(a.convx, a.conv + 15_335_424);
        // convx: 48 * 128 * 6656 * 2 = 81,788,928 (78.0 MiB, independent of t_max)
        assert_eq!(a.ids, a.convx + 48 * 128 * 6656 * 2);
        // xk and xv must remain byte-adjacent for batched k/v projections.
        assert_eq!(a.xv, a.xk + c.t_max * H * 2);
        // Overlay start offsets equal scratch_pool base.
        assert_eq!(a.ar_out, a.scratch_pool);
        assert_eq!(a.rel_logits, a.scratch_pool);
        assert_eq!(a.pa_ws, a.scratch_pool);
        assert_eq!(a.rlog, a.scratch_pool);
        // Within Overlay A, ax_work does not overlap ar_out..h32.
        assert!(a.ax_work >= a.h32 + c.t_max * H * 4);
        // Within Overlay B, B1 (qkvf_pf..vx_work) and B2 (da_opart..da_ml) start after rel_logits.
        let after_rl = (a.rel_logits + c.t_max * NH * REL_EXT * 2).next_multiple_of(ALIGN);
        assert_eq!(a.qkvf_pf, after_rl);
        assert_eq!(a.da_opart, after_rl);
        // Within Overlay D, mx_work does not overlap mh/my.
        assert!(a.mx_work >= a.my + c.t_max * TOPK * H * 2);
        // Verify compact arena size vs old 37.98 GB default:
        assert!(c.arena_bytes() < 32_000_000_000, "got {}", c.arena_bytes());
    }

    #[test]
    fn kv_fp8_halves_global_pool_bytes() {
        let bf = MemConfig::default();
        let fp8 = MemConfig { kv_fp8: true, ..bf };
        assert_eq!(bf.arena_bytes() - fp8.arena_bytes(), 4 * GIB);
        let mut p = ArenaPlan::new(fp8.arena_bytes());
        let a = Activations::alloc(&mut p, &fp8).unwrap();
        assert_eq!((a.kpool, a.vpool, a.kring), (0, 2 * GIB, 4 * GIB));
    }
}
