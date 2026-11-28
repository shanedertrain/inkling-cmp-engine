// Ported from engine/native/src/draft.cu and draft.hpp (`ink::DraftRT`) and the host code of
// engine/native/src/drafter_rt.cuh (`load_weights`, struct `Drafter`: constructor, `step`, `cache`, `markov`,
// `commit_rows`, `layers`, `head_gemm`, `head_rest`) @ 1c7ac46, with the round batch of `draft_batch`, `bstep` and
// `DraftRT::draft_sessions` (draft_sel.cuh, package G2), at the settings DraftRT runs them with (one stream,
// `g_overlap = 0`; `g_head_warps 8`, `g_sk_wo 6`, `g_sk_down 6`, `g_attn_ctas 384`, `g_gemm3 0`, `g_head_rt2 0`,
// `g_argmax2 3`, `g_coop 0`; `ksteps 7`, full vocabulary; no IMMA head). Own code of this project.
//! [`RustDrafter`] (package R8 of docs/RUST-PLAN.md): the C++ DSpark drafter (`ink::DraftRT`) over the kernel pack,
//! the same kernels with the same arguments in the same order, bit for bit. Implements [`ink_rt_api::Drafter`].
//!
//! Per-session state on the card: three layers of draft K/V (64-token pages, page table `1 + i`), anchor and anchor
//! position `P`, `f_rnn`. `commit_rows` writes committed target rows' K/V (eager, chunks of 64 rows), `frnn` and
//! `set_state` set the round's inputs, `draft_sessions` runs the three layers and the Markov head for the sessions
//! that draft in the round only (their state gathered into compact rows, the outputs scattered back:
//! `engine/native/src/draft_sel.cuh`): eagerly once per batch size k, then as that k's CUDA graph captured on the
//! drafter's stream and replayed (as `DraftRT::draft_sessions`); `draft` is the batch of every session. Contract:
//! `engine/kernels/drafter/README.md`. Port map: `PORT-MAP.md` ("Drafter").
//!
//! Device ranges of the [`ink_rt_api::Drafter`] calls are handles of a registry (`devbufs.rs`): its own (from
//! [`RustDrafter::new`]) or the one of the engine that created it, which shares its context and stream
//! (`engine/specrt.rs`); a range not in the registry is refused.
//!
//! Field order is drop order: the graph, the buffers, the kernels, the arena they point into, the stream, the context.

mod ops;

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;

use bytemuck::{Pod, Zeroable};
use ink_cuda::{
    AlgoPins, Arena, ArenaPlan, Bf16, Context, CudaError, DevBuf, Dtype, GemmShape, Graph, LtGemm, PlanClass, Stream,
};
use ink_kernels::{F16, KernelId, Kernels, s2f};
use ink_model::Alloc;
use ink_model::drafter::{
    self as dm, AT_TILE, ATTN_CTAS, ATTN_SMEM, BLK, CONF_NCH, COS_SIN_POS, DRAFTS, DraftCheckpoints, H, HD, INTER, KVH,
    LRK, MK_NCH, NH, PAGE, RANK, RNN_IN, RNN_OUT, RNN_STATE, VP, VU, WIN,
};
use ink_model::dump::{FNV_OFFSET, arena_map_text, fnv1a64};
use ink_rt_api::{
    DevBuf as ApiBuf, DraftConfig, Drafter, FEAT_ROW_BYTES, HeadPrecision, KvPrecision, ProposalBufs, RuntimeError,
    VerifyWidth,
};

/// Top-K sparse proposal width for exact DSpark Leviathan rejection sampling (`s2f::LEV_K`).
const LEV_K: usize = 16;
/// Chunks for fused DSpark Leviathan scan (`s2f::LEV_NCH = ceil(VU / 512) = 391`).
const LEV_NCH: usize = VU.div_ceil(512);
/// Max compacted candidates per session (`s2f::LEV_NCAND = LEV_NCH * LEV_K`).
const LEV_NCAND: usize = LEV_NCH * LEV_K;
/// Default proposal `top_p` cap for DSpark Leviathan sampling (`+4.0%` acceptance on 20-case `sampled` and `+3.4%`
/// `self-t1-n4` M1 over `0.95`, 100% lossless target distribution via exact Leviathan rejection sampling).
/// Overridable via `INK_DRAFT_TOPP`.
const DEFAULT_DRAFT_TOPP: f32 = 0.90;

/// Prefill chunk size for the fast cuBLASLt `commit_rows` path (`3 * 1024 * 2048 = 6M <= s.part 8M` f32).
const COMMIT_CHUNK: usize = 1024;

fn kvctx_shape() -> GemmShape {
    GemmShape {
        n: (2 * KVH * HD) as u32,
        k: H as u32,
        ldw: H as u32,
        input: Dtype::Bf16,
        output: Dtype::F32,
        class: PlanClass::Mixed,
    }
}

use self::ops::{Batch, Dev, Epi, Gemm, gemm_frag_ids, gemm_smem};
use crate::devbufs::DevBufs;
use crate::engine::SharedGlobalPages;

type Res<T> = Result<T, RuntimeError>;

use crate::error::rt;

/// Ring buffer blocks per session for DSpark's SWA-1024 layers (`L0`, `L2`): `64 * 64 = 4,096` tokens per session,
/// preserving `[P - 1024, P)` across rollbacks up to `3,072` tokens while shrinking `L0/L2` KV to `0.50 GiB` at `n=8`.
const SWA_RING_BLOCKS: usize = 64;

/// Shape of one drafter: sessions, positions, head and global-layer K/V precision, and what follows from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    /// Sessions (`n`).
    pub n: usize,
    /// Positions per session (`L`).
    pub max_ctx: usize,
    /// int8 head with the exact bound (else the bf16 head).
    pub int8: bool,
    /// E4M3 storage for the global layer (`L1`) draft K/V (no duplicate BF16 L1 buffer).
    pub kv8: bool,
    /// The drafter probability of each draft (`conf_on`, draft_conf.cuh) for the gated verify width.
    pub conf: bool,
    /// Pages per session: ceil((L + 8) / 64).
    pub max_blocks: usize,
    /// SWA ring blocks per session for `L0` and `L2` (`min(max_blocks, SWA_RING_BLOCKS)`).
    pub swa_blocks_per_sess: usize,
    /// Total 64-token physical blocks in the global `L1` pool (including block-0 sentinel).
    pub l1_blocks: usize,
    /// Attention splits of the three layers.
    pub split_attn: [usize; 3],
}

impl Shape {
    /// Standalone `Drafter` sizes (dedicated per-session `L1` blocks; ring `L0/L2`).
    #[cfg(test)]
    pub(crate) fn new(n: usize, max_ctx: usize, int8: bool, kv8: bool, conf: bool) -> Shape {
        Self::new_with_pool(n, max_ctx, int8, kv8, conf, None)
    }

    /// `Drafter` sizes with optional target-engine global KV pool size (`num_pages` of 1,024 tokens = `16` blocks each).
    pub(crate) fn new_with_pool(
        n: usize,
        max_ctx: usize,
        int8: bool,
        kv8: bool,
        conf: bool,
        pooled_global_pages: Option<usize>,
    ) -> Shape {
        let seqlen = max_ctx + BLK;
        let max_blocks = seqlen.div_ceil(PAGE);
        let swa_blocks_per_sess = max_blocks.min(SWA_RING_BLOCKS);
        let l1_blocks = match pooled_global_pages {
            // Each 1,024-token target page maps to 16 contiguous 64-token DSpark blocks, plus 1 scratch block per
            // session for draft block `[P..P+7]` crossing into an unallocated target page, plus block-0 sentinel.
            Some(pages) => pages * 16 + n + 1,
            None => n * max_blocks + 1,
        };
        let mut split_attn = [0; 3];
        for (l, s) in split_attn.iter_mut().enumerate() {
            let window = if l == 1 { 0 } else { WIN + BLK };
            let keys = if window > 0 { window.min(seqlen) } else { seqlen };
            let tiles = keys.div_ceil(AT_TILE);
            let mut sp = tiles.min(ATTN_CTAS.div_ceil(8 * n)).max(1);
            sp = sp.max((tiles + 1).div_ceil(256)); // attn_block holds <= 256 pages per CTA
            *s = sp;
        }
        Shape {
            n,
            max_ctx,
            int8,
            kv8,
            conf,
            max_blocks,
            swa_blocks_per_sess,
            l1_blocks,
            split_attn,
        }
    }

    /// Rows of the draft block over all sessions (`T` = 8 n).
    pub(crate) fn t(&self) -> usize {
        self.n * BLK
    }

    /// Elements of one SWA layer's K (or V) ring cache (`L0`, `L2`).
    pub(crate) fn swa_cache_elems(&self) -> usize {
        (self.n * self.swa_blocks_per_sess + 1) * KVH * PAGE * HD
    }

    /// Elements of the global layer's (`L1`) K (or V) cache.
    pub(crate) fn global_cache_elems(&self) -> usize {
        self.l1_blocks * KVH * PAGE * HD
    }
}

/// One layer's weights (`struct Layer`).
pub(crate) struct LayerW<A: Alloc> {
    in_norm: A::Buf<Bf16>,
    post_norm: A::Buf<Bf16>,
    ctx_norm: A::Buf<Bf16>,
    q_norm: A::Buf<Bf16>,
    k_norm: A::Buf<Bf16>,
    k_norm_ctx: A::Buf<Bf16>,
    qkv_f: A::Buf<Bf16>,
    kvctx_f: A::Buf<Bf16>,
    kvctx_w: A::Buf<Bf16>,
    wo_f: A::Buf<Bf16>,
    gu_f: A::Buf<Bf16>,
    down_f: A::Buf<Bf16>,
}

/// The drafter's weights (`struct Weights`), in the C++ load order. Buffers only the C++ harness's other variants
/// read (the IMMA head `head_q8k32`, `head_qn`, the outlier columns) are not taken; the bf16 head's fragments only
/// with the bf16 head, the int8 head's only with the int8 head.
pub(crate) struct DraftW<A: Alloc> {
    layers: [LayerW<A>; 3],
    final_norm: A::Buf<Bf16>,
    mask_embed: A::Buf<Bf16>,
    f_norm: A::Buf<Bf16>,
    w1: A::Buf<Bf16>,
    w_state: A::Buf<Bf16>,
    wctx_f: A::Buf<Bf16>,
    rnn_b: A::Buf<f32>,
    alpha: A::Buf<f32>,
    head: A::Buf<Bf16>,
    embed: A::Buf<Bf16>,
    embed_norm: A::Buf<Bf16>,
    head_f: Option<A::Buf<Bf16>>,
    head_rn: A::Buf<f32>,
    head_scale: Option<A::Buf<f32>>,
    head_enorm: Option<A::Buf<f32>>,
    head_q8f: Option<A::Buf<i8>>,
    fold_w: A::Buf<Bf16>,
    fold_b: A::Buf<Bf16>,
    rn: A::Buf<f32>,
    fold_q: A::Buf<i8>,
    fold_s: A::Buf<f32>,
    efold: A::Buf<f32>,
    qnorm: A::Buf<f32>,
    lr_v: A::Buf<f32>,
    lr_u: A::Buf<Bf16>,
    lr_p: A::Buf<f32>,
    lr_q: A::Buf<f32>,
    cos_sin: A::Buf<f32>,
}

impl<A: Alloc> DraftW<A> {
    fn take(ar: &mut A, s: &Shape) -> Result<Self, CudaError> {
        let mut layer = |l: usize| -> Result<LayerW<A>, CudaError> {
            let p = |what: &str| format!("layers.{l}.{what}");
            Ok(LayerW {
                in_norm: ar.take(H, &p("input_layernorm"))?,
                post_norm: ar.take(H, &p("post_attention_layernorm"))?,
                ctx_norm: ar.take(H, &p("dflare_norm"))?,
                q_norm: ar.take(HD, &p("q_norm"))?,
                k_norm: ar.take(HD, &p("k_norm"))?,
                k_norm_ctx: ar.take(HD, &p("k_norm_ctx"))?,
                qkv_f: ar.take((NH + 2 * KVH) * HD * H, &p("qkv_f"))?,
                kvctx_f: ar.take(2 * KVH * HD * H, &p("kvctx_f"))?,
                kvctx_w: ar.take(2 * KVH * HD * H, &p("kvctx_w"))?,
                wo_f: ar.take(H * NH * HD, &p("wo_f"))?,
                gu_f: ar.take(2 * INTER * H, &p("gu_f"))?,
                down_f: ar.take(H * INTER, &p("down_f"))?,
            })
        };
        let layers = [layer(0)?, layer(1)?, layer(2)?];
        Ok(DraftW {
            layers,
            final_norm: ar.take(H, "norm")?,
            mask_embed: ar.take(H, "mask_embed")?,
            f_norm: ar.take(H, "markov_head.dflare_norm")?,
            w1: ar.take(VP * RANK, "markov_w1")?,
            w_state: ar.take(RNN_OUT * RNN_STATE, "w_state")?,
            wctx_f: ar.take(RNN_OUT * (RNN_IN - RNN_STATE), "wctx_f")?,
            rnn_b: ar.take(RNN_OUT, "rnn_b")?,
            alpha: ar.take(16, "alpha")?,
            head: ar.take(VP * H, "head")?,
            embed: ar.take(VP * H, "embed")?,
            embed_norm: ar.take(H, "embed_norm")?,
            head_f: if s.int8 { None } else { Some(ar.take(VP * H, "head_f")?) },
            head_rn: ar.take(VP, "head_rn")?,
            head_scale: if s.int8 { Some(ar.take(VP, "head_scale")?) } else { None },
            head_enorm: if s.int8 { Some(ar.take(VP, "head_enorm")?) } else { None },
            head_q8f: if s.int8 {
                Some(ar.take(VP * H, "head_q8f")?)
            } else {
                None
            },
            fold_w: ar.take(VP * RANK, "fold_w")?,
            fold_b: ar.take(VP, "fold_b")?,
            rn: ar.take(VP, "rn")?,
            fold_q: ar.take(VP * RANK, "fold_q")?,
            fold_s: ar.take(VP, "fold_s")?,
            efold: ar.take(VP, "efold")?,
            qnorm: ar.take(VP, "qnorm")?,
            lr_v: ar.take(16 * 256, "lrV")?,
            lr_u: ar.take(VP * 16, "lrU")?,
            lr_p: ar.take(VP, "lrP")?,
            lr_q: ar.take(VP, "lrQ")?,
            cos_sin: ar.take(COS_SIN_POS * 128, "cos_sin")?,
        })
    }
}

/// The drafter's state and workspaces (`struct Drafter`'s allocations, in the C++ order). Not taken: `feats` (read
/// by no path DraftRT runs), `dbg`, and the IMMA head's `hq`, `hsx`, `hdn`, `hn_in`. The round batch's buffers
/// (`sel` .. `conf_c`, draft_sel.cuh) come last, as in C++.
pub(crate) struct DraftS<A: Alloc> {
    anchor: A::Buf<i32>,
    p: A::Buf<i32>,
    bt: A::Buf<i32>,
    bt_swa: A::Buf<i32>,
    kc: [Option<A::Buf<Bf16>>; 3],
    vc: [Option<A::Buf<Bf16>>; 3],
    k8: Option<A::Buf<u8>>,
    v8: Option<A::Buf<u8>>,
    cmix: A::Buf<Bf16>,
    f_rnn: A::Buf<Bf16>,
    residual: A::Buf<Bf16>,
    xn: A::Buf<Bf16>,
    qbuf: A::Buf<Bf16>,
    attn_out: A::Buf<Bf16>,
    inter: A::Buf<Bf16>,
    hout: A::Buf<Bf16>,
    hf16: A::Buf<F16>,
    x7: A::Buf<Bf16>,
    part: A::Buf<f32>,
    opart: A::Buf<f32>,
    ml: A::Buf<f32>,
    ctx: A::Buf<f32>,
    b: A::Buf<Bf16>,
    bq: Option<A::Buf<f32>>,
    hnorm: A::Buf<f32>,
    state: A::Buf<Bf16>,
    latent: A::Buf<Bf16>,
    lat2: A::Buf<f32>,
    keys: A::Buf<u64>,
    seedkeys: A::Buf<u64>,
    drafts: A::Buf<i32>,
    nsurv: A::Buf<i32>,
    conf: A::Buf<f32>,
    cb_pr: A::Buf<f32>,
    cb_zd: A::Buf<f32>,
    cb_vd: A::Buf<i32>,
    cb_part: A::Buf<f32>,
    proj: A::Buf<f32>,
    latq: A::Buf<i8>,
    sel: A::Buf<i32>,
    anchor_c: A::Buf<i32>,
    p_c: A::Buf<i32>,
    bt_c: A::Buf<i32>,
    bt_swa_c: A::Buf<i32>,
    f_rnn_c: A::Buf<Bf16>,
    drafts_c: A::Buf<i32>,
    nsurv_c: A::Buf<i32>,
    conf_c: A::Buf<f32>,
    lev_chunk_val: A::Buf<f32>,
    lev_chunk_id: A::Buf<i32>,
    lev_chunk_z: A::Buf<f32>,
    lev_chunk_cmax: A::Buf<f32>,
    lev_cand_cnt: A::Buf<i32>,
    lev_g_lb: A::Buf<u32>,
    lev_active_mask: A::Buf<u32>,
    samp_temp: A::Buf<f32>,
    samp_top_p: A::Buf<f32>,
    samp_seed: A::Buf<u64>,
    prop_ids: A::Buf<i32>,
    prop_w: A::Buf<u32>,
}

impl<A: Alloc> DraftS<A> {
    fn take(ar: &mut A, s: &Shape) -> Result<Self, CudaError> {
        let (n, t) = (s.n, s.t());
        let se = s.swa_cache_elems();
        let ge = s.global_cache_elems();
        let anchor = ar.take(n, "anchor")?;
        let p = ar.take(n, "P")?;
        let bt = ar.take(n * s.max_blocks, "block_table")?;
        let bt_swa = ar.take(n * s.max_blocks, "block_table_swa")?;
        let (k0, v0) = (Some(ar.take(se, "Kc0")?), Some(ar.take(se, "Vc0")?));
        let (k1, v1) = if s.kv8 {
            (None, None)
        } else {
            (Some(ar.take(ge, "Kc1")?), Some(ar.take(ge, "Vc1")?))
        };
        let (k2, v2) = (Some(ar.take(se, "Kc2")?), Some(ar.take(se, "Vc2")?));
        let (k8, v8) = if s.kv8 {
            (Some(ar.take(ge, "K8")?), Some(ar.take(ge, "V8")?))
        } else {
            (None, None)
        };
        let smax = s.split_attn.iter().copied().max().unwrap_or(1);
        Ok(DraftS {
            anchor,
            p,
            bt,
            bt_swa,
            kc: [k0, k1, k2],
            vc: [v0, v1, v2],
            k8,
            v8,
            cmix: ar.take(3 * t.max(COMMIT_CHUNK) * H, "cmix")?,
            f_rnn: ar.take(n * H, "f_rnn")?,
            residual: ar.take(t * H, "residual")?,
            xn: ar.take(t * H, "xn")?,
            qbuf: ar.take(t * NH * HD, "qbuf")?,
            attn_out: ar.take(t * NH * HD, "attn_out")?,
            inter: ar.take(t * INTER, "inter")?,
            hout: ar.take(n * 7 * H, "hout")?,
            hf16: ar.take(n * 7 * H, "hf16")?,
            x7: ar.take(n * 7 * 2 * H, "x7")?,
            part: ar.take(8 << 20, "part")?,
            opart: ar.take(n * KVH * smax * 32 * HD, "opart")?,
            ml: ar.take(n * KVH * smax * 32 * 2, "ml")?,
            ctx: ar.take(n * 7 * RNN_OUT, "ctx")?,
            b: ar.take(n * 7 * VP, "B")?,
            bq: if s.int8 { Some(ar.take(n * 7 * VP, "Bq")?) } else { None },
            hnorm: ar.take(n * 7, "hnorm")?,
            state: ar.take(2 * n * RANK, "state")?,
            latent: ar.take(n * RANK, "latent")?,
            lat2: ar.take(7 * n, "lat2")?,
            keys: ar.take(7 * n, "keys")?,
            seedkeys: ar.take(7 * n * MK_NCH, "seedkeys")?,
            drafts: ar.take(7 * n, "drafts")?,
            nsurv: ar.take(7 * n, "nsurv")?,
            conf: ar.take(7 * n, "conf")?,
            cb_pr: ar.take(7 * n * LRK, "cb.pr")?,
            cb_zd: ar.take(7 * n, "cb.zd")?,
            cb_vd: ar.take(7 * n, "cb.vd")?,
            cb_part: ar.take(7 * CONF_NCH * n, "cb.part")?,
            proj: ar.take(n * 20, "proj")?,
            latq: ar.take(n * RANK, "latq")?,
            sel: ar.take(n, "sel")?,
            anchor_c: ar.take(n, "anchor_c")?,
            p_c: ar.take(n, "P_c")?,
            bt_c: ar.take(n * s.max_blocks, "bt_c")?,
            bt_swa_c: ar.take(n * s.max_blocks, "bt_swa_c")?,
            f_rnn_c: ar.take(n * H, "f_rnn_c")?,
            drafts_c: ar.take(7 * n, "drafts_c")?,
            nsurv_c: ar.take(7 * n, "nsurv_c")?,
            conf_c: ar.take(7 * n, "conf_c")?,
            lev_chunk_val: ar.take(n * LEV_NCAND, "lev_chunk_val")?,
            lev_chunk_id: ar.take(n * LEV_NCAND, "lev_chunk_id")?,
            lev_chunk_z: ar.take(LEV_NCH * n, "lev_chunk_z")?,
            lev_chunk_cmax: ar.take(LEV_NCH * n, "lev_chunk_cmax")?,
            lev_cand_cnt: ar.take(n, "lev_cand_cnt")?,
            lev_g_lb: ar.take(n, "lev_g_lb")?,
            lev_active_mask: ar.take(1, "lev_active_mask")?,
            samp_temp: ar.take(n, "samp_temp")?,
            samp_top_p: ar.take(n, "samp_top_p")?,
            samp_seed: ar.take(n, "samp_seed")?,
            prop_ids: ar.take(n * DRAFTS * LEV_K, "prop_ids")?,
            prop_w: ar.take(n * DRAFTS * LEV_K, "prop_w")?,
        })
    }
}

/// Bytes of the drafter's arena for `s` (weights and state, in take order).
pub(crate) fn arena_bytes(s: &Shape) -> usize {
    let mut plan = ArenaPlan::new(usize::MAX);
    DraftW::take(&mut plan, s).expect("an unbounded plan cannot be exhausted");
    DraftS::take(&mut plan, s).expect("an unbounded plan cannot be exhausted");
    plan.used()
}

/// The DSpark drafter in Rust; see the module docs.
pub struct RustDrafter {
    /// `graphs[k - 1]`: the greedy draft of a round batch of k sessions, once captured.
    graphs: Vec<Option<Graph>>,
    /// `graphs_sampled[k - 1]`: the Leviathan-sampled draft of a round batch of k sessions, once captured.
    graphs_sampled: Vec<Option<Graph>>,
    /// cuBLASLt handle for large prefill `commit_rows` (`rows > 64`) context K/V GEMMs.
    lt: RefCell<LtGemm>,
    /// Use fast cuBLASLt `commit_rows` for prefill chunks (`rows > 64`); `INK_DSPARK_FAST_COMMIT=0` disables.
    fast_commit: bool,
    s: DraftS<Arena>,
    w: DraftW<Arena>,
    k: Kernels,
    arena: Arena,
    /// Device ranges handed out ([`RustDrafter::alloc`], the drafts), shared with the engine that created it.
    bufs: Rc<RefCell<DevBufs>>,
    /// Device bytes the captured graphs took (per graph: free memory before the capture minus after its first replay;
    /// summed over the batch sizes), read by the engine that created the drafter for its graph reserve.
    graph_bytes: Rc<Cell<usize>>,
    /// The drafts' range.
    drafts: ApiBuf,
    /// The drafter probability's range, with [`VerifyWidth::Gated`].
    conf: Option<ApiBuf>,
    /// Sparse top-16 proposal distribution buffers for exact Leviathan rejection sampling (int8 head).
    proposal: Option<ProposalBufs>,
    /// Whether any session in the upcoming draft round has `temperature >= 1e-5`.
    any_sampled: bool,
    /// When true (`INK_DSPARK_LEV_FULL=1`), sampled sessions emit `conf = 0.0` (verify width 8).
    lev_full_verify: bool,
    /// Proposal temperature multiplier for sampled sessions (`INK_DRAFT_TEMP`, default `1.0`).
    draft_temp_scale: f32,
    /// Optional proposal `top_p` override for sampled sessions (`INK_DRAFT_TOPP`).
    draft_top_p: Option<f32>,
    /// Confidence signal mode for sampled sessions (`INK_DRAFT_CONF_MODE`: 0=p_max, 1=p(c*), 2=q(c*)).
    conf_mode: i32,
    /// Log cumulative confidence threshold for lossless Markov early-exit on steps `0..=3`
    /// (`INK_DSPARK_EARLY_TAU`, default disabled `-INFINITY` because `CallRows::fill` widening gets free accepted
    /// tokens from drafts past `verify_tau` at N>=4; set e.g. `INK_DSPARK_EARLY_TAU=0.70` to enable for N=1/ragged).
    early_tau_ln: f32,
    /// Cached host copies of `sel`, `samp_temp`, `samp_top_p`, `samp_seed` to skip unchanged H2D uploads.
    last_sel: Vec<i32>,
    last_eff_temps: Vec<u32>,
    last_eff_top_ps: Vec<u32>,
    last_seeds: Vec<u64>,
    /// Optional shared global page allocator from `Engine` (pooled `L1` paging across all sessions).
    shared_pages: Option<SharedGlobalPages>,
    pt_gen: Cell<u64>,
    bt_host: RefCell<Vec<i32>>,
    st: Rc<Stream>,
    ctx: Rc<Context>,
    shape: Shape,
}

impl std::fmt::Debug for RustDrafter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RustDrafter")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

/// One tapped tensor: name, bytes, FNV-1a 64 of its contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftTap {
    /// Buffer name (the C++ member or weight field).
    pub name: String,
    /// Size in bytes.
    pub bytes: usize,
    /// FNV-1a 64 of the bytes as read back.
    pub fnv1a64: u64,
}

fn host_bytes<T: Pod>(st: &Stream, b: &DevBuf<T>) -> Res<Vec<u8>> {
    let mut v = vec![T::zeroed(); b.len()];
    rt(st.dtoh(&mut v, b))?;
    Ok(bytemuck::cast_slice(&v).to_vec())
}

impl RustDrafter {
    /// A drafter on its own: device 0's primary context, its own stream and range registry (the replay tool
    /// `ink-draft`). An engine creates its drafter on its own context and stream ([`ink_rt_api::SpecRuntime::
    /// create_drafter`]).
    ///
    /// # Errors
    ///
    /// As [`RustDrafter::on`], or the device error of the context.
    pub fn new(config: &DraftConfig) -> Res<RustDrafter> {
        let ctx = Rc::new(rt(Context::new(0))?);
        let st = Rc::new(rt(ctx.new_stream())?);
        Self::on(
            ctx,
            st,
            Rc::new(RefCell::new(DevBufs::default())),
            Rc::new(Cell::new(0)),
            config,
            None,
        )
    }

    /// Loads the drafter on context `ctx` and stream `st`, as `DraftRT::DraftRT` on the engine's stream: weights
    /// from `model_dir` and `fold_bin`, state for `sessions` sessions of `max_ctx` positions; its device ranges go
    /// into `bufs`, the device bytes of its graphs into `graph_bytes` as they are captured.
    ///
    /// # Errors
    ///
    /// [`RuntimeError`] for an invalid configuration or any file or device failure.
    pub(crate) fn on(
        ctx: Rc<Context>,
        st: Rc<Stream>,
        bufs: Rc<RefCell<DevBufs>>,
        graph_bytes: Rc<Cell<usize>>,
        config: &DraftConfig,
        shared_pages: Option<SharedGlobalPages>,
    ) -> Res<RustDrafter> {
        let n = config.sessions as usize;
        if n == 0 || n > 8 {
            return Err(RuntimeError::request(format!(
                "sessions must be 1..8 (GEMM M <= 64), got {n}"
            )));
        }
        if config.max_ctx == 0 {
            return Err(RuntimeError::request("max_ctx must be positive"));
        }
        let shape = Shape::new_with_pool(
            n,
            config.max_ctx as usize,
            config.head == HeadPrecision::Int8,
            config.global_kv == KvPrecision::E4m3,
            config.verify != VerifyWidth::Full,
            shared_pages.as_ref().map(|sp| sp.num_pages),
        );
        if shape.max_ctx + BLK > COS_SIN_POS {
            return Err(RuntimeError::request(format!(
                "max_ctx {} beyond the RoPE table ({COS_SIN_POS} positions)",
                shape.max_ctx
            )));
        }
        let mut arena = rt(Arena::new(&ctx, &st, arena_bytes(&shape)))?;
        let w = rt(DraftW::take(&mut arena, &shape))?;
        let s = rt(DraftS::take(&mut arena, &shape))?;
        eprintln!(
            "[dspark] arena used {:.2} GiB (sessions {n}, max_ctx {}, kv8 {}, l1_blocks {}, swa_ring_blocks {}, pooled {})",
            arena.plan().used() as f64 / 1_073_741_824.0,
            shape.max_ctx,
            shape.kv8,
            shape.l1_blocks,
            shape.swa_blocks_per_sess,
            shared_pages.is_some(),
        );
        let k = rt(Kernels::load(&ctx))?;
        // kernel attributes the C++ sets: attn_block / attn_block8 in DraftRT's constructor, each gemm_frag instance
        // at its first launch (gemm_w)
        for id in [KernelId::S2fAttnBlock, KernelId::S2fAttnBlock8] {
            rt(k.function(id).set_max_dynamic_smem(ATTN_SMEM))?;
        }
        for (id, mt) in gemm_frag_ids() {
            rt(k.function(id).set_max_dynamic_smem(gemm_smem(mt)))?;
        }
        let drafts = bufs.borrow_mut().register(rt(s.drafts.cast::<u8>())?);
        let conf = if shape.conf {
            Some(bufs.borrow_mut().register(rt(s.conf.cast::<u8>())?))
        } else {
            None
        };
        let lev_off = std::env::var("INK_DSPARK_NO_LEVIATHAN").is_ok_and(|v| v == "1");
        let proposal = if shape.int8 && !lev_off {
            let ids = bufs.borrow_mut().register(rt(s.prop_ids.cast::<u8>())?);
            let weights = bufs.borrow_mut().register(rt(s.prop_w.cast::<u8>())?);
            Some(ProposalBufs { drafts, ids, weights })
        } else {
            None
        };
        let lev_full_verify = std::env::var("INK_DSPARK_LEV_FULL").is_ok_and(|v| v == "1");
        let draft_temp_scale = std::env::var("INK_DRAFT_TEMP")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|&v| v > 0.0 && v.is_finite())
            .unwrap_or(1.0);
        let draft_top_p = std::env::var("INK_DRAFT_TOPP")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|&v| v > 0.0 && v <= 1.0);
        let conf_mode = std::env::var("INK_DRAFT_CONF_MODE")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|&v| (0..=2).contains(&v))
            .unwrap_or(0);
        let early_tau_ln = if !shape.conf || lev_full_verify {
            f32::NEG_INFINITY
        } else {
            match std::env::var("INK_DSPARK_EARLY_TAU")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
            {
                Some(v) if v > 0.0 && v.is_finite() => v.min(1.0).ln(),
                _ => f32::NEG_INFINITY,
            }
        };
        let fast_commit = !std::env::var("INK_DSPARK_FAST_COMMIT").is_ok_and(|v| v == "0");
        let mut selector = rt(LtGemm::new(&st, AlgoPins::new(0)))?;
        let mut pins = AlgoPins::new(ctx.startup().cublaslt_version);
        let kv_sh = kvctx_shape();
        let sel_pin = rt(selector.select(kv_sh, COMMIT_CHUNK as u32))?;
        rt(pins.insert(kv_sh, sel_pin.pin))?;
        let lt = rt(LtGemm::new(&st, pins))?;
        let d = RustDrafter {
            graphs: (0..n).map(|_| None).collect(),
            graphs_sampled: (0..n).map(|_| None).collect(),
            lt: RefCell::new(lt),
            fast_commit,
            bufs,
            graph_bytes,
            drafts,
            conf,
            proposal,
            any_sampled: false,
            lev_full_verify,
            draft_temp_scale,
            draft_top_p,
            conf_mode,
            early_tau_ln,
            last_sel: Vec::new(),
            last_eff_temps: Vec::new(),
            last_eff_top_ps: Vec::new(),
            last_seeds: Vec::new(),
            shared_pages,
            pt_gen: Cell::new(u64::MAX),
            bt_host: RefCell::new(vec![0i32; n * shape.max_blocks]),
            s,
            w,
            k,
            arena,
            st,
            ctx,
            shape,
        };
        d.load(&config.model_dir, &config.fold_bin)?;
        d.init_state()?;
        rt(d.st.synchronize())?;
        Ok(d)
    }

    fn dev(&self) -> Dev<'_> {
        Dev {
            k: &self.k,
            st: &self.st,
        }
    }

    /// A scratch buffer of `n` elements in its own allocation (freed when dropped; load time only).
    fn scratch<T: Pod>(&self, n: usize) -> Res<(Arena, DevBuf<T>)> {
        let mut a = rt(Arena::new(&self.ctx, &self.st, n * size_of::<T>()))?;
        let b = rt(a.take::<T>(n, "scratch"))?;
        Ok((a, b))
    }

    /// `load_weights`: uploads, fragment permutations, int8 quantisation, norms, the fold and the RoPE table.
    fn load(&self, model_dir: &Path, fold_bin: &Path) -> Res<()> {
        let ck = rt(DraftCheckpoints::open(model_dir))?;
        let (mtp, mdl) = (&ck.mtp, &ck.model);
        let st = &self.st;
        let up = |dst: &DevBuf<Bf16>, src: &[u8]| rt(st.htod_bytes(dst, src));
        let t = tensor;
        let w = &self.w;
        for (i, l) in w.layers.iter().enumerate() {
            let p = format!("layers.{i}.");
            let a = format!("{p}self_attn.");
            up(&l.in_norm, t(mtp, &format!("{p}input_layernorm.weight"), H)?)?;
            up(&l.post_norm, t(mtp, &format!("{p}post_attention_layernorm.weight"), H)?)?;
            up(&l.ctx_norm, t(mtp, &format!("{p}dflare_norm.weight"), H)?)?;
            up(&l.q_norm, t(mtp, &format!("{a}q_norm.weight"), HD)?)?;
            up(&l.k_norm, t(mtp, &format!("{a}k_norm.weight"), HD)?)?;
            up(&l.k_norm_ctx, t(mtp, &format!("{a}k_norm_ctx.weight"), HD)?)?;
            // cat_rows + to_frag: the concatenation is a copy, made on the host
            let cat = |parts: &[(&str, usize)]| -> Res<Vec<u8>> {
                let mut v = Vec::new();
                for &(name, rows) in parts {
                    v.extend_from_slice(t(mtp, name, rows * H)?);
                }
                Ok(v)
            };
            let frag = |dst: &DevBuf<Bf16>, src: &[u8], n: usize, k: usize| -> Res<()> {
                let (_a, tmp) = self.scratch::<Bf16>(n * k)?;
                up(&tmp, src)?;
                rt(self.dev().permute_frag_bf16(&tmp, dst, n, k))?;
                rt(st.synchronize())
            };
            let qkv = cat(&[
                (&format!("{a}q_proj.weight"), NH * HD),
                (&format!("{a}k_proj.weight"), KVH * HD),
                (&format!("{a}v_proj.weight"), KVH * HD),
            ])?;
            frag(&l.qkv_f, &qkv, (NH + 2 * KVH) * HD, H)?;
            let kv = cat(&[
                (&format!("{a}k_proj_ctx.weight"), KVH * HD),
                (&format!("{a}v_proj_ctx.weight"), KVH * HD),
            ])?;
            up(&l.kvctx_w, &kv)?;
            frag(&l.kvctx_f, &kv, 2 * KVH * HD, H)?;
            frag(&l.wo_f, t(mtp, &format!("{a}o_proj.weight"), H * NH * HD)?, H, NH * HD)?;
            let gu = cat(&[
                (&format!("{p}mlp.gate_proj.weight"), INTER),
                (&format!("{p}mlp.up_proj.weight"), INTER),
            ])?;
            frag(&l.gu_f, &gu, 2 * INTER, H)?;
            frag(
                &l.down_f,
                t(mtp, &format!("{p}mlp.down_proj.weight"), H * INTER)?,
                H,
                INTER,
            )?;
        }
        up(&w.final_norm, t(mtp, "norm.weight", H)?)?;
        up(&w.mask_embed, t(mtp, "mask_embed", H)?)?;
        up(&w.f_norm, t(mtp, "markov_head.dflare_norm.weight", H)?)?;
        up(&w.w1, t(mtp, "markov_head.markov_w1.weight", VP * RANK)?)?;
        let rnn_w = t(mtp, "markov_head.rnn_proj.weight", RNN_OUT * RNN_IN)?;
        up(&w.w_state, &rt(dm::bf16_cols(rnn_w, RNN_OUT, RNN_IN, 0, RNN_STATE))?)?;
        {
            let wctx = rt(dm::bf16_cols(rnn_w, RNN_OUT, RNN_IN, RNN_STATE, RNN_IN - RNN_STATE))?;
            let (_a, tmp) = self.scratch::<Bf16>(RNN_OUT * (RNN_IN - RNN_STATE))?;
            up(&tmp, &wctx)?;
            rt(self
                .dev()
                .permute_frag_bf16(&tmp, &w.wctx_f, RNN_OUT, RNN_IN - RNN_STATE))?;
            rt(st.synchronize())?;
        }
        rt(st.htod(&w.rnn_b, &dm::widen_bf16(t(mtp, "markov_head.rnn_proj.bias", RNN_OUT)?)))?;
        rt(st.htod(&w.alpha, &rt(dm::fuse_alpha(t(mtp, "fuse_weight", 16)?))?))?;
        up(&w.head, t(mdl, "model.llm.unembed.weight", VP * H)?)?;
        up(&w.embed, t(mdl, "model.llm.embed.weight", VP * H)?)?;
        up(&w.embed_norm, t(mdl, "model.llm.embed_norm.weight", H)?)?;
        let dev = self.dev();
        if let Some(hf) = &w.head_f {
            rt(dev.permute_frag_bf16(&w.head, hf, VP, H))?;
        }
        rt(dev.row_norms(&w.head, &w.head_rn, VP, H))?;
        if let (Some(scale), Some(enorm), Some(q8f)) = (&w.head_scale, &w.head_enorm, &w.head_q8f) {
            let (_a, q) = self.scratch::<i8>(VP * H)?;
            rt(dev.quant_rows_i8(&w.head, &q, scale, enorm, VP, H))?;
            rt(dev.permute_frag_i8(&q, q8f, VP, H))?;
            rt(st.synchronize())?;
        }
        let fold = rt(dm::read_fold(fold_bin))?;
        up(&w.fold_w, &fold.w)?;
        up(&w.fold_b, &fold.b)?;
        rt(dev.row_norms(&w.fold_w, &w.rn, VP, RANK))?;
        rt(dev.quant_rows_i8(&w.fold_w, &w.fold_q, &w.fold_s, &w.efold, VP, RANK))?;
        rt(dev.row_norms_i8(&w.fold_q, &w.qnorm, VP, RANK))?;
        let lr = rt(dm::read_lowrank(&dm::lowrank_path(fold_bin)))?;
        rt(st.htod(&w.lr_v, &lr.v))?;
        up(&w.lr_u, &lr.u)?;
        rt(st.htod(&w.lr_p, &lr.p))?;
        rt(st.htod(&w.lr_q, &lr.q))?;
        rt(dev.cos_sin(&w.cos_sin, COS_SIN_POS))?;
        rt(st.synchronize())
    }

    /// The `Drafter` constructor's initial state: SWA ring block table for `L0`/`L2`, `L1` page table (identity
    /// standalone, or per-session scratch block when pooled with `Engine`), `P = max_ctx`, anchors `1000 + 7919 r`,
    /// and the draft K/V filled with noise (seeds 11 + l, 21 + l; E4M3 copies 41, 42).
    fn init_state(&self) -> Res<()> {
        let sh = self.shape;
        let s = &self.s;
        let mut bt_swa = vec![0i32; sh.n * sh.max_blocks];
        for r in 0..sh.n {
            for b in 0..sh.max_blocks {
                bt_swa[r * sh.max_blocks + b] = i32_of(1 + r * sh.swa_blocks_per_sess + (b % sh.swa_blocks_per_sess));
            }
        }
        let bt: Vec<i32> = if let Some(ref sp) = self.shared_pages {
            let mut v = vec![0i32; sh.n * sh.max_blocks];
            for r in 0..sh.n {
                let scratch = i32_of(1 + sp.num_pages * 16 + r);
                v[r * sh.max_blocks..(r + 1) * sh.max_blocks].fill(scratch);
            }
            v
        } else {
            (0..sh.n * sh.max_blocks).map(|i| i32_of(1 + i)).collect()
        };
        let p = vec![i32_of(sh.max_ctx); sh.n];
        let a: Vec<i32> = (0..sh.n).map(|r| i32_of(1000 + 7919 * r)).collect();
        rt(self.st.htod(&s.anchor, &a))?;
        rt(self.st.htod(&s.p, &p))?;
        rt(self.st.htod(&s.bt, &bt))?;
        rt(self.st.htod(&s.bt_swa, &bt_swa))?;
        let dev = self.dev();
        for l in 0..3 {
            if let Some(kc) = &s.kc[l] {
                rt(dev.fill_bf16(kc, 11 + l as u64, 1.0))?;
            }
            if let Some(vc) = &s.vc[l] {
                rt(dev.fill_bf16(vc, 21 + l as u64, 1.0))?;
            }
        }
        if let (Some(k8), Some(v8)) = (&s.k8, &s.v8) {
            rt(dev.fill_e4m3(k8, 41))?;
            rt(dev.fill_e4m3(v8, 42))?;
        }
        Ok(())
    }

    /// Synchronises DSpark's global `L1` block table (`s.bt`) with `Engine`'s `Slots` whenever any slot's page list
    /// changed (`page_gen`). Each 1,024-token target page maps to 16 contiguous 64-token DSpark blocks, and the next
    /// logical block past each slot's allocated target pages maps to its dedicated scratch block (so draft block
    /// `[P..P+7]` can safely cross into an unallocated target page before `Engine::forward_dev` reserves it).
    fn sync_page_table(&self) -> Result<(), CudaError> {
        let Some(ref shared) = self.shared_pages else {
            return Ok(());
        };
        let slots = shared.slots.borrow();
        let page_gen = slots.page_gen();
        if page_gen == self.pt_gen.get() {
            return Ok(());
        }
        let sh = self.shape;
        let mut host = self.bt_host.borrow_mut();
        for r in 0..sh.n {
            let row = r * sh.max_blocks;
            let scratch_blk = i32_of(1 + shared.num_pages * 16 + r);
            host[row..row + sh.max_blocks].fill(scratch_blk);
            let pages = slots.pages(r);
            for (lp, &tp) in pages.iter().enumerate() {
                let b0 = lp * 16;
                let p0 = 1 + tp * 16;
                for sub in 0..16 {
                    if b0 + sub < sh.max_blocks {
                        host[row + b0 + sub] = p0 + sub as i32;
                    }
                }
            }
        }
        self.st.htod(&self.s.bt, &host)?;
        self.pt_gen.set(page_gen);
        Ok(())
    }

    /// `Drafter::step()`.
    fn step(&self) -> s2f::Step {
        s2f::Step {
            n: i32_of(self.shape.n),
            _pad0: [0; 4],
            anchor: self.s.anchor.ptr(),
            P: self.s.p.ptr(),
            block_table: self.s.bt.ptr(),
            max_blocks: i32_of(self.shape.max_blocks),
            _pad1: [0; 4],
        }
    }

    /// `Drafter::bstep(k)`: the round batch's k compact rows (global `L1` block table).
    fn bstep(&self, k: usize) -> s2f::Step {
        s2f::Step {
            n: i32_of(k),
            _pad0: [0; 4],
            anchor: self.s.anchor_c.ptr(),
            P: self.s.p_c.ptr(),
            block_table: self.s.bt_c.ptr(),
            max_blocks: i32_of(self.shape.max_blocks),
            _pad1: [0; 4],
        }
    }

    /// The round batch's k compact rows with the SWA ring block table (`L0` and `L2`).
    fn bstep_swa(&self, k: usize) -> s2f::Step {
        s2f::Step {
            block_table: self.s.bt_swa_c.ptr(),
            ..self.bstep(k)
        }
    }

    /// `Drafter::cache(l)`: the E4M3 copy on the global layer (1) only (`k`/`v` null on `L1` when `kv8` is enabled).
    fn cache(&self, l: usize) -> s2f::CacheL {
        let mut c = s2f::CacheL::zeroed();
        if let Some(kc) = &self.s.kc[l] {
            c.k = kc.ptr();
        }
        if let Some(vc) = &self.s.vc[l] {
            c.v = vc.ptr();
        }
        if l == 1
            && let (Some(k8), Some(v8)) = (&self.s.k8, &self.s.v8)
        {
            c.k8 = k8.ptr();
            c.v8 = v8.ptr();
        }
        c
    }

    /// `Drafter::markov(k)` over the round batch's k compact rows (no IMMA head: `hdn`, `qhn`, `hsc`, `hn_in` null;
    /// `dbg` null).
    fn markov(&self, k: usize) -> s2f::Markov {
        let (s, w) = (&self.s, &self.w);
        let mut m = s2f::Markov::zeroed();
        m.n = i32_of(k);
        m.anchor = s.anchor_c.ptr();
        m.ctx = s.ctx.ptr();
        m.w_state = w.w_state.ptr();
        m.w1 = w.w1.ptr();
        m.fold_w = w.fold_w.ptr();
        m.fold_b = w.fold_b.ptr();
        m.rn = w.rn.ptr();
        m.B = s.b.ptr();
        if let Some(bq) = &s.bq {
            m.Bq = bq.ptr();
        }
        if let Some(e) = &w.head_enorm {
            m.enorm = e.ptr();
        }
        m.hrn = w.head_rn.ptr();
        m.hnorm = s.hnorm.ptr();
        m.head = w.head.ptr();
        m.h = s.hout.ptr();
        m.state = s.state.ptr();
        m.latent = s.latent.ptr();
        m.lat2 = s.lat2.ptr();
        m.keys = s.keys.ptr();
        m.seedkeys = s.seedkeys.ptr();
        m.drafts = s.drafts_c.ptr();
        m.nsurv = s.nsurv_c.ptr();
        m.fold_q = w.fold_q.ptr();
        m.fold_s = w.fold_s.ptr();
        m.efold = w.efold.ptr();
        m.lrV = w.lr_v.ptr();
        m.lrU = w.lr_u.ptr();
        m.lrP = w.lr_p.ptr();
        m.lrQ = w.lr_q.ptr();
        m.vlim = i32_of(VU);
        m.ksteps = i32_of(DRAFTS);
        m.proj = s.proj.ptr();
        m.latq = s.latq.ptr();
        m.qnorm = w.qnorm.ptr();
        m
    }

    /// `Drafter::commit_rows`: `rows` committed rows' features `[rows][4][H]` at positions `pos` of sessions `sess`,
    /// in chunks of 64: mix, the three layers' context K/V GEMMs (split-K 9), K norm + RoPE into the draft K/V.
    fn commit(
        &self,
        feats: &DevBuf<Bf16>,
        pos: &DevBuf<i32>,
        sess: &DevBuf<i32>,
        rows: usize,
    ) -> Result<(), CudaError> {
        self.sync_page_table()?;
        let (st, dev, w, s) = (self.step(), self.dev(), &self.w, &self.s);
        let mut cc = s2f::Caches::zeroed();
        for (l, c) in cc.l.iter_mut().enumerate() {
            *c = self.cache(l);
        }
        let sk = 9;
        let ctx_norm = [&w.layers[0].ctx_norm, &w.layers[1].ctx_norm, &w.layers[2].ctx_norm];
        let k_norm_ctx = [
            &w.layers[0].k_norm_ctx,
            &w.layers[1].k_norm_ctx,
            &w.layers[2].k_norm_ctx,
        ];
        if self.fast_commit && rows > 64 {
            let sh = kvctx_shape();
            for r0 in (0..rows).step_by(COMMIT_CHUNK) {
                let rc = COMMIT_CHUNK.min(rows - r0);
                let f = feats.slice(r0 * 4 * H..(r0 + rc) * 4 * H)?;
                dev.commit_mix(&f, &w.alpha, ctx_norm, &w.f_norm, &s.cmix, &s.f_rnn, rc)?;
                for l in 0..3 {
                    let x = s.cmix.slice(l * rc * H..(l + 1) * rc * H)?;
                    let part = s.part.slice(l * rc * 2048..(l + 1) * rc * 2048)?;
                    self.lt.borrow_mut().run(
                        &self.st,
                        sh,
                        &x,
                        H as u32,
                        &w.layers[l].kvctx_w,
                        &part,
                        2048,
                        rc as u32,
                    )?;
                }
                let p = pos.slice(r0..r0 + rc)?;
                let q = sess.slice(r0..r0 + rc)?;
                dev.epi_kvctx(&s.part, 1, rc, st, k_norm_ctx, &w.cos_sin, cc, &p, &q, Some(&s.bt_swa))?;
            }
            return Ok(());
        }
        for r0 in (0..rows).step_by(64) {
            let rc = 64.min(rows - r0);
            let f = feats.slice(r0 * 4 * H..(r0 + rc) * 4 * H)?;
            dev.commit_mix(&f, &w.alpha, ctx_norm, &w.f_norm, &s.cmix, &s.f_rnn, rc)?;
            for l in 0..3 {
                let x = s.cmix.slice(l * rc * H..s.cmix.len())?;
                let part = s.part.slice(l * sk * rc * 2048..s.part.len())?;
                dev.gemm(
                    Epi::F32,
                    &Gemm {
                        w: &w.layers[l].kvctx_f,
                        x: &x,
                        ldx: H,
                        m: rc,
                        n: 2 * KVH * HD,
                        k: H,
                        part: Some(&part),
                        out: None,
                        scale: None,
                        ldo: 2 * KVH * HD,
                    },
                    sk,
                )?;
            }
            let p = pos.slice(r0..r0 + rc)?;
            let q = sess.slice(r0..r0 + rc)?;
            dev.epi_kvctx(&s.part, sk, rc, st, k_norm_ctx, &w.cos_sin, cc, &p, &q, Some(&s.bt_swa))?;
        }
        Ok(())
    }

    /// `Drafter::draft_batch(k)`: the state of the sessions `sel[0..k)` into compact rows, layers and head on them,
    /// their outputs back to the sessions' rows.
    fn draft_batch(&self, k: usize, sampled: bool, prof: bool) -> Result<(), CudaError> {
        let (s, sh) = (&self.s, &self.shape);
        let b = Batch {
            sel: &s.sel,
            anchor: &s.anchor,
            p: &s.p,
            bt: &s.bt,
            f_rnn: &s.f_rnn,
            anchor_c: &s.anchor_c,
            p_c: &s.p_c,
            bt_c: &s.bt_c,
            f_rnn_c: &s.f_rnn_c,
            drafts_c: &s.drafts_c,
            nsurv_c: &s.nsurv_c,
            conf_c: &s.conf_c,
            drafts: &s.drafts,
            nsurv: &s.nsurv,
            conf: &s.conf,
        };
        let b_swa = Batch {
            bt: &s.bt_swa,
            bt_c: &s.bt_swa_c,
            ..b
        };
        let mut evs: Vec<(&'static str, ink_cuda::Event)> = Vec::new();
        let mark = |evs: &mut Vec<(&'static str, ink_cuda::Event)>, name: &'static str| -> Result<(), CudaError> {
            if prof {
                let e = self.ctx.new_event()?;
                self.st.record(&e)?;
                evs.push((name, e));
            }
            Ok(())
        };
        mark(&mut evs, "start")?;
        self.dev().draft_gather(k, sh.n, sh.max_blocks, &b)?;
        self.dev().draft_gather(k, sh.n, sh.max_blocks, &b_swa)?;
        mark(&mut evs, "gather")?;
        self.layers(k, prof, &mut evs)?;
        if sampled {
            self.head_sampled(k, prof, &mut evs)?;
        } else {
            self.head(k, prof, &mut evs)?;
        }
        self.dev().draft_scatter(k, sh.n, &b)?;
        mark(&mut evs, "scatter")?;
        if prof {
            self.st.synchronize()?;
            eprint!("[dspark prof k={k} sampled={sampled}]");
            for pair in evs.windows(2) {
                let dt = pair[0].1.elapsed_ms(&pair[1].1)?;
                eprint!(" {}={:.3}", pair[1].0, dt);
            }
            let tot = evs.first().unwrap().1.elapsed_ms(&evs.last().unwrap().1)?;
            eprintln!(" TOTAL={tot:.3} ms");
        }
        Ok(())
    }

    /// `Drafter::layers(k)` on the round batch's k compact rows: the block's embedding, then per layer qkv GEMM
    /// (split-K 4), q/k norm + RoPE, attention (the drafter's splits, sized by n), o_proj (split-K 6), add + norm,
    /// gate_up (split-K 2), SiLU, down (split-K 6), add + norm (the last one writes the head inputs).
    fn layers(&self, k: usize, prof: bool, evs: &mut Vec<(&'static str, ink_cuda::Event)>) -> Result<(), CudaError> {
        let (st, dev, w, s, sh) = (self.bstep(k), self.dev(), &self.w, &self.s, &self.shape);
        let t = k * BLK;
        let mut mark = |name: &'static str| -> Result<(), CudaError> {
            if prof {
                let e = self.ctx.new_event()?;
                self.st.record(&e)?;
                evs.push((name, e));
            }
            Ok(())
        };
        dev.embed_block(
            t,
            st,
            &w.embed,
            &w.embed_norm,
            &w.mask_embed,
            &w.layers[0].in_norm,
            &s.residual,
            &s.xn,
        )?;
        mark("embed")?;
        let f32_gemm = |wt: &DevBuf<Bf16>, x: &DevBuf<Bf16>, ldx: usize, n: usize, k: usize, sk: usize| {
            dev.gemm(
                Epi::F32,
                &Gemm {
                    w: wt,
                    x,
                    ldx,
                    m: t,
                    n,
                    k,
                    part: Some(&s.part),
                    out: None,
                    scale: None,
                    ldo: n,
                },
                sk,
            )
        };
        for (l, lw) in w.layers.iter().enumerate() {
            let st_l = if l == 1 { st } else { self.bstep_swa(k) };
            f32_gemm(&lw.qkv_f, &s.xn, H, (NH + 2 * KVH) * HD, H, 4)?;
            dev.epi_qkv(
                &s.part,
                4,
                t,
                st_l,
                &lw.q_norm,
                &lw.k_norm,
                &w.cos_sin,
                self.cache(l),
                &s.qbuf,
            )?;
            if l == 0 {
                mark("l0_qkv")?;
            }
            let window = if l == 1 { 0 } else { WIN + BLK };
            let split = sh.split_attn[l];
            dev.attn_block(
                sh.kv8 && l == 1,
                k,
                &s.qbuf,
                st_l,
                self.cache(l),
                window,
                split,
                &s.opart,
                &s.ml,
            )?;
            dev.attn_combine(k, &s.opart, &s.ml, split, &s.attn_out)?;
            match l {
                0 => mark("l0_attn")?,
                1 => mark("l1_gattn")?,
                _ => mark("l2_attn")?,
            }
            f32_gemm(&lw.wo_f, &s.attn_out, NH * HD, H, NH * HD, 6)?;
            dev.epi_addnorm(&s.part, 6, t, &s.residual, &lw.post_norm, Some(&s.xn), None)?;
            if l == 0 {
                mark("l0_wo")?;
            }
            f32_gemm(&lw.gu_f, &s.xn, H, 2 * INTER, H, 2)?;
            dev.epi_silu(&s.part, 2, t, &s.inter)?;
            if l == 0 {
                mark("l0_gu")?;
            }
            f32_gemm(&lw.down_f, &s.inter, INTER, H, INTER, 6)?;
            if l < 2 {
                dev.epi_addnorm(&s.part, 6, t, &s.residual, &w.layers[l + 1].in_norm, Some(&s.xn), None)?;
            } else {
                let head = (&s.hout, &s.hf16, &s.x7, &s.f_rnn_c, &s.hnorm);
                dev.epi_addnorm(&s.part, 6, t, &s.residual, &w.final_norm, None, Some(head))?;
            }
            match l {
                0 => mark("l0_down")?,
                1 => mark("l1_mlp")?,
                _ => mark("l2_mlp")?,
            }
        }
        Ok(())
    }

    /// `Drafter::head(k)` on the round batch: `head_gemm` (four memsets of the batch's rows, the vocabulary GEMM) and
    /// `head_rest` (rnn context GEMM split-K 24 + bias, Markov seed, 7 x (recur, zref, argmax3, conf_save), finish,
    /// then with the probability on its sum over the vocabulary and `conf_finish`, draft_conf.cuh).
    fn head(&self, k: usize, prof: bool, evs: &mut Vec<(&'static str, ink_cuda::Event)>) -> Result<(), CudaError> {
        let (dev, w, s, sh) = (self.dev(), &self.w, &self.s, &self.shape);
        let m = k * 7;
        let mut mark = |name: &'static str| -> Result<(), CudaError> {
            if prof {
                let e = self.ctx.new_event()?;
                self.st.record(&e)?;
                evs.push((name, e));
            }
            Ok(())
        };
        self.st.memset_zero(&s.keys.slice(0..7 * k)?)?;
        self.st.memset_zero(&s.seedkeys.slice(0..7 * k * MK_NCH)?)?;
        self.st.memset_zero(&s.lat2.slice(0..7 * k)?)?;
        self.st.memset_zero(&s.nsurv_c.slice(0..7 * k)?)?;
        if sh.int8 {
            let (q8f, scale, bq) = match (&w.head_q8f, &w.head_scale, &s.bq) {
                (Some(a), Some(b_), Some(c)) => (a, b_, c),
                _ => return Err(CudaError::Graph("int8 head without its buffers".into())),
            };
            let x = s.hf16.cast::<Bf16>()?;
            dev.gemm(
                Epi::HeadQ,
                &Gemm {
                    w: q8f,
                    x: &x,
                    ldx: H,
                    m,
                    n: VP,
                    k: H,
                    part: Some(bq),
                    out: None,
                    scale: Some(scale),
                    ldo: VP,
                },
                1,
            )?;
        } else {
            let hf = w
                .head_f
                .as_ref()
                .ok_or_else(|| CudaError::Graph("bf16 head without its fragments".into()))?;
            dev.gemm(
                Epi::Bf16,
                &Gemm {
                    w: hf,
                    x: &s.hout,
                    ldx: H,
                    m,
                    n: VP,
                    k: H,
                    part: None,
                    out: Some(&s.b),
                    scale: None,
                    ldo: VP,
                },
                1,
            )?;
        }
        mark("head_gemm")?;
        dev.gemm(
            Epi::F32,
            &Gemm {
                w: &w.wctx_f,
                x: &s.x7,
                ldx: 2 * H,
                m,
                n: RNN_OUT,
                k: 2 * H,
                part: Some(&s.part),
                out: None,
                scale: None,
                ldo: RNN_OUT,
            },
            24,
        )?;
        dev.epi_ctx(&s.part, 24, m, &w.rnn_b, &s.ctx)?;
        mark("rnn_ctx")?;
        let mk = self.markov(k);
        dev.markov_seed(sh.int8, k, &mk)?;
        mark("mk_seed")?;
        for i in 0..DRAFTS {
            dev.markov_recur(&mk, i)?;
            dev.markov_zref(sh.int8, k, &mk, i)?;
            dev.markov_argmax3(sh.int8, k, &mk, i)?;
            if sh.conf {
                dev.conf_save(k, &mk, i, &self.conf_buf())?;
            }
        }
        mark("mk_loop7")?;
        dev.markov_finish(&mk)?;
        if sh.conf {
            if !sh.int8 {
                return Err(CudaError::Graph(
                    "the drafter probability needs the int8 head (Bq)".into(),
                ));
            }
            let cb = self.conf_buf();
            dev.conf_sum(k, &mk, &cb, DRAFTS)?;
            dev.conf_finish(&cb, k, DRAFTS, &s.conf_c)?;
            mark("conf_sum")?;
        }
        Ok(())
    }

    /// Sampled Markov head for exact Leviathan rejection sampling (int8 head): computes exact full-rank bf16
    /// logits on the top-16 candidates per step, writes exact integer proposal weights summing to `2^22`, draws
    /// `c* ~ q` via Philox4x32-10, and feeds the sampled token autoregressively into the next Markov step.
    fn head_sampled(
        &self,
        k: usize,
        prof: bool,
        evs: &mut Vec<(&'static str, ink_cuda::Event)>,
    ) -> Result<(), CudaError> {
        let (dev, w, s) = (self.dev(), &self.w, &self.s);
        let m = k * 7;
        let mut mark = |name: &'static str| -> Result<(), CudaError> {
            if prof {
                let e = self.ctx.new_event()?;
                self.st.record(&e)?;
                evs.push((name, e));
            }
            Ok(())
        };
        self.st.memset_zero(&s.proj.slice(0..k * 20)?)?;
        let (q8f, scale, bq) = match (&w.head_q8f, &w.head_scale, &s.bq) {
            (Some(a), Some(b_), Some(c)) => (a, b_, c),
            _ => return Err(CudaError::Graph("DSpark Leviathan requires int8 head".into())),
        };
        let x = s.hf16.cast::<Bf16>()?;
        dev.gemm(
            Epi::HeadQ,
            &Gemm {
                w: q8f,
                x: &x,
                ldx: H,
                m,
                n: VP,
                k: H,
                part: Some(bq),
                out: None,
                scale: Some(scale),
                ldo: VP,
            },
            1,
        )?;
        mark("head_gemm")?;
        dev.gemm(
            Epi::F32,
            &Gemm {
                w: &w.wctx_f,
                x: &s.x7,
                ldx: 2 * H,
                m,
                n: RNN_OUT,
                k: 2 * H,
                part: Some(&s.part),
                out: None,
                scale: None,
                ldo: RNN_OUT,
            },
            24,
        )?;
        dev.epi_ctx(&s.part, 24, m, &w.rnn_b, &s.ctx)?;
        mark("rnn_ctx")?;
        let mk = self.markov(k);
        let lb = s2f::LevBuf {
            chunk_val: s.lev_chunk_val.ptr(),
            chunk_id: s.lev_chunk_id.ptr(),
            chunk_z: s.lev_chunk_z.ptr(),
            chunk_cmax: s.lev_chunk_cmax.ptr(),
            cand_cnt: s.lev_cand_cnt.ptr(),
            g_lb: s.lev_g_lb.ptr(),
            samp_temp: s.samp_temp.ptr(),
            samp_top_p: s.samp_top_p.ptr(),
            samp_seed: s.samp_seed.ptr(),
            pos_c: s.p_c.ptr(),
            sel: s.sel.ptr(),
            prop_ids: s.prop_ids.ptr(),
            prop_w: s.prop_w.ptr(),
            conf_c: s.conf_c.ptr(),
            full_verify: i32::from(self.lev_full_verify),
            conf_mode: self.conf_mode,
            active_mask: s.lev_active_mask.ptr(),
            early_tau_ln: self.early_tau_ln,
            _pad0: [0; 4],
        };
        for i in 0..DRAFTS {
            dev.markov_recur_proj(&mk, i, &lb)?;
            if i == 0 {
                mark("s0_recur")?;
            }
            dev.markov_lev_scan_fused(k, &mk, i, &lb)?;
            if i == 0 {
                mark("s0_scan")?;
            }
            dev.markov_lev_finish(k, &mk, i, &lb)?;
            if i == 0 {
                mark("s0_fin")?;
            }
        }
        mark("mk_s1_6")?;
        Ok(())
    }

    /// `Drafter::cb`: the probability pass's buffers, over the round batch's compact rows.
    fn conf_buf(&self) -> s2f::ConfBuf {
        let s = &self.s;
        s2f::ConfBuf {
            pr: s.cb_pr.ptr(),
            zd: s.cb_zd.ptr(),
            vd: s.cb_vd.ptr(),
            part: s.cb_part.ptr(),
        }
    }

    /// Allocates `bytes` zeroed bytes of device memory for this drafter's call inputs and returns its range.
    ///
    /// # Errors
    ///
    /// [`RuntimeError`] when `bytes` is 0 or beyond 2^40, or the allocation fails.
    pub fn alloc(&mut self, bytes: usize) -> Res<ApiBuf> {
        self.bufs.borrow_mut().alloc(&self.ctx, &self.st, bytes)
    }

    fn typed<T: Pod>(&self, h: ApiBuf, elems: usize, what: &str) -> Res<DevBuf<T>> {
        self.bufs.borrow().typed(h, elems, what)
    }

    /// Copies `src` into the buffer `dst` (from the start), ordered on the drafter's stream.
    ///
    /// # Errors
    ///
    /// [`RuntimeError`] for a foreign range, a size beyond the buffer, or the copy's failure.
    pub fn upload(&mut self, dst: ApiBuf, src: &[u8]) -> Res<()> {
        let b = self.typed::<u8>(dst, src.len(), "upload")?;
        rt(self.st.htod(&b, src))
    }

    /// Copies the start of buffer `src` into `dst` after the drafter's queued work has finished.
    ///
    /// # Errors
    ///
    /// As [`RustDrafter::upload`].
    pub fn download(&mut self, src: ApiBuf, dst: &mut [u8]) -> Res<()> {
        let b = self.typed::<u8>(src, dst.len(), "download")?;
        rt(self.st.dtoh(dst, &b))
    }

    /// The launch trace of the captured draft graph of a batch of `k` sessions (`ink-launch-trace/1` node lines)
    /// and the drafter arena's map, or `None` before the first draft of `k` sessions.
    ///
    /// # Errors
    ///
    /// The driver error of reading a node.
    pub fn launch_trace(&self, k: usize) -> Res<Option<(Vec<String>, String)>> {
        let Some(Some(g)) = k.checked_sub(1).and_then(|i| self.graphs.get(i)) else {
            return Ok(None);
        };
        Ok(Some((
            rt(g.launch_trace(&self.arena))?.0,
            arena_map_text(self.arena.plan()),
        )))
    }

    /// Every weight and state buffer of the drafter with the FNV-1a 64 of its contents, in take order (after the
    /// queued work has finished).
    ///
    /// # Errors
    ///
    /// The device error of a copy.
    pub fn taps(&self) -> Res<Vec<DraftTap>> {
        let mut out = Vec::new();
        let st = &self.st;
        let mut tap = |name: &str, bytes: Vec<u8>| {
            out.push(DraftTap {
                name: name.to_owned(),
                bytes: bytes.len(),
                fnv1a64: fnv1a64(&bytes, FNV_OFFSET),
            });
        };
        let (w, s) = (&self.w, &self.s);
        for (i, l) in w.layers.iter().enumerate() {
            let p = |x: &str| format!("layers.{i}.{x}");
            for (name, b) in [
                ("in_norm", &l.in_norm),
                ("post_norm", &l.post_norm),
                ("ctx_norm", &l.ctx_norm),
                ("q_norm", &l.q_norm),
                ("k_norm", &l.k_norm),
                ("k_norm_ctx", &l.k_norm_ctx),
                ("qkv_f", &l.qkv_f),
                ("kvctx_f", &l.kvctx_f),
                ("wo_f", &l.wo_f),
                ("gu_f", &l.gu_f),
                ("down_f", &l.down_f),
            ] {
                tap(&p(name), host_bytes(st, b)?);
            }
        }
        let bf = [
            ("final_norm", &w.final_norm),
            ("mask_embed", &w.mask_embed),
            ("f_norm", &w.f_norm),
            ("w1", &w.w1),
            ("w_state", &w.w_state),
            ("wctx_f", &w.wctx_f),
            ("head", &w.head),
            ("embed", &w.embed),
            ("embed_norm", &w.embed_norm),
            ("fold_w", &w.fold_w),
            ("fold_b", &w.fold_b),
            ("lrU", &w.lr_u),
        ];
        for (name, b) in bf {
            tap(name, host_bytes(st, b)?);
        }
        if let Some(b) = &w.head_f {
            tap("head_f", host_bytes(st, b)?);
        }
        if let Some(b) = &w.head_q8f {
            tap("head_q8f", host_bytes(st, b)?);
        }
        tap("fold_q", host_bytes(st, &w.fold_q)?);
        let f = [
            ("rnn_b", &w.rnn_b),
            ("alpha", &w.alpha),
            ("head_rn", &w.head_rn),
            ("rn", &w.rn),
            ("fold_s", &w.fold_s),
            ("efold", &w.efold),
            ("qnorm", &w.qnorm),
            ("lrV", &w.lr_v),
            ("lrP", &w.lr_p),
            ("lrQ", &w.lr_q),
            ("cos_sin", &w.cos_sin),
        ];
        for (name, b) in f {
            tap(name, host_bytes(st, b)?);
        }
        for (name, b) in [("head_scale", &w.head_scale), ("head_enorm", &w.head_enorm)] {
            if let Some(b) = b {
                tap(name, host_bytes(st, b)?);
            }
        }
        for (name, b) in [
            ("anchor", &s.anchor),
            ("P", &s.p),
            ("block_table", &s.bt),
            ("drafts", &s.drafts),
            ("nsurv", &s.nsurv),
        ] {
            tap(name, host_bytes(st, b)?);
        }
        for l in 0..3 {
            if let Some(kc) = &s.kc[l] {
                tap(&format!("Kc{l}"), host_bytes(st, kc)?);
            }
            if let Some(vc) = &s.vc[l] {
                tap(&format!("Vc{l}"), host_bytes(st, vc)?);
            }
        }
        if let (Some(k8), Some(v8)) = (&s.k8, &s.v8) {
            tap("K8", host_bytes(st, k8)?);
            tap("V8", host_bytes(st, v8)?);
        }
        for (name, b) in [
            ("cmix", &s.cmix),
            ("f_rnn", &s.f_rnn),
            ("residual", &s.residual),
            ("xn", &s.xn),
            ("qbuf", &s.qbuf),
            ("attn_out", &s.attn_out),
            ("inter", &s.inter),
            ("hout", &s.hout),
            ("x7", &s.x7),
            ("B", &s.b),
            ("state", &s.state),
            ("latent", &s.latent),
        ] {
            tap(name, host_bytes(st, b)?);
        }
        tap("hf16", host_bytes(st, &s.hf16)?);
        tap("latq", host_bytes(st, &s.latq)?);
        tap("keys", host_bytes(st, &s.keys)?);
        tap("seedkeys", host_bytes(st, &s.seedkeys)?);
        for (name, b) in [
            ("part", &s.part),
            ("opart", &s.opart),
            ("ml", &s.ml),
            ("ctx", &s.ctx),
            ("hnorm", &s.hnorm),
            ("lat2", &s.lat2),
            ("conf", &s.conf),
            ("cb.pr", &s.cb_pr),
            ("cb.zd", &s.cb_zd),
            ("cb.part", &s.cb_part),
            ("proj", &s.proj),
        ] {
            tap(name, host_bytes(st, b)?);
        }
        if let Some(b) = &s.bq {
            tap("Bq", host_bytes(st, b)?);
        }
        for (name, b) in [
            ("sel", &s.sel),
            ("anchor_c", &s.anchor_c),
            ("P_c", &s.p_c),
            ("bt_c", &s.bt_c),
            ("drafts_c", &s.drafts_c),
            ("nsurv_c", &s.nsurv_c),
        ] {
            tap(name, host_bytes(st, b)?);
        }
        tap("f_rnn_c", host_bytes(st, &s.f_rnn_c)?);
        tap("conf_c", host_bytes(st, &s.conf_c)?);
        tap("cb.vd", host_bytes(st, &s.cb_vd)?);
        Ok(out)
    }

    /// Milliseconds per draft of a batch of `k` sessions (replays of its graph, the batch of the last
    /// [`Drafter::draft_sessions`] with `k` sessions) over `iters` calls between two events on the drafter's stream.
    ///
    /// # Errors
    ///
    /// [`RuntimeError`] before that graph exists, or a device error.
    pub fn draft_ms(&mut self, k: usize, iters: usize) -> Res<f64> {
        let g = k
            .checked_sub(1)
            .and_then(|i| self.graphs.get(i))
            .and_then(Option::as_ref)
            .ok_or_else(|| RuntimeError::request(format!("no draft graph of {k} sessions yet")))?;
        let (e0, e1) = (rt(self.ctx.new_event())?, rt(self.ctx.new_event())?);
        rt(self.st.record(&e0))?;
        for _ in 0..iters {
            rt(g.launch())?;
        }
        rt(self.st.record(&e1))?;
        rt(e1.synchronize())?;
        Ok(f64::from(rt(e0.elapsed_ms(&e1))?) / iters.max(1) as f64)
    }
}

/// BF16 tensor `name` of `c` with `elems` elements, as raw bytes.
fn tensor<'a>(c: &'a ink_model::Checkpoint, name: &str, elems: usize) -> Res<&'a [u8]> {
    rt(dm::bf16_bytes(c, name, elems))
}

fn i32_of(x: usize) -> i32 {
    i32::try_from(x).expect("drafter sizes fit i32")
}

impl Drafter for RustDrafter {
    #[allow(unsafe_code)] // implements the trait's contract method (ink-rt-api bounds contract)
    unsafe fn commit_rows(&mut self, feats: ApiBuf, pos: ApiBuf, sess: ApiBuf, n: usize) -> Res<()> {
        if n == 0 {
            return Ok(());
        }
        if feats.bytes() < n * FEAT_ROW_BYTES {
            return Err(RuntimeError::request(format!(
                "commit_rows: features of {} bytes for {n} rows",
                feats.bytes()
            )));
        }
        let f = self.typed::<Bf16>(feats, n * 4 * H, "commit_rows features")?;
        let p = self.typed::<i32>(pos, n, "commit_rows positions")?;
        let s = self.typed::<i32>(sess, n, "commit_rows sessions")?;
        rt(self.commit(&f, &p, &s, n))
    }

    fn frnn(&mut self, feats: ApiBuf) -> Res<()> {
        let n = self.shape.n;
        let f = self.typed::<Bf16>(feats, n * 4 * H, "frnn features")?;
        rt(self.dev().frnn(n, &f, &self.w.alpha, &self.w.f_norm, &self.s.f_rnn))
    }

    #[allow(unsafe_code)] // implements the trait's contract method (ink-rt-api bounds contract)
    unsafe fn set_state(&mut self, anchor: ApiBuf, pos: ApiBuf) -> Res<()> {
        let n = self.shape.n;
        let a = self.typed::<i32>(anchor, n, "set_state anchor")?;
        let p = self.typed::<i32>(pos, n, "set_state positions")?;
        rt(self.st.dtod(&self.s.anchor, &a))?;
        rt(self.st.dtod(&self.s.p, &p))
    }

    fn draft(&mut self) -> Res<()> {
        let all: Vec<usize> = (0..self.shape.n).collect();
        self.draft_sessions(&all)
    }

    fn draft_sessions(&mut self, sessions: &[usize]) -> Res<()> {
        let (n, k) = (self.shape.n, sessions.len());
        if k == 0 || k > n {
            return Err(RuntimeError::request(format!(
                "draft_sessions: 1..{n} sessions, got {k}"
            )));
        }
        let mut seen = [false; 8];
        for &r in sessions {
            if r >= n || std::mem::replace(&mut seen[r], true) {
                return Err(RuntimeError::request(format!(
                    "draft_sessions: sessions must be distinct, 0..{}",
                    n - 1
                )));
            }
        }
        rt(self.sync_page_table())?;
        let list: Vec<i32> = sessions.iter().map(|&r| i32_of(r)).collect();
        if self.last_sel != list {
            rt(self.st.htod(&rt(self.s.sel.slice(0..k))?, &list))?;
            self.last_sel = list;
        }
        let sampled = self.any_sampled && self.proposal.is_some();
        let slot_missing = if sampled {
            self.graphs_sampled[k - 1].is_none()
        } else {
            self.graphs[k - 1].is_none()
        };
        if slot_missing {
            let prof = std::env::var("INK_DSPARK_PROF").is_ok_and(|v| v != "0");
            // eager once (as DraftRT::draft_sessions: the C++ sets kernel attributes there), then capture
            rt(self.draft_batch(k, sampled, false))?;
            if prof {
                rt(self.draft_batch(k, sampled, true))?;
            }
            rt(self.st.synchronize())?;
            let free0 = rt(self.ctx.mem_info())?.0;
            // a failure inside the capture is a device error whatever its cause (review V8; as the engine's)
            let cap = rt(self.st.begin_capture())?;
            let g = rt(self.draft_batch(k, sampled, false))
                .and_then(|()| rt(cap.end()))
                .map_err(RuntimeError::into_device)?;
            rt(g.launch())?;
            rt(self.st.synchronize())?;
            let free1 = rt(self.ctx.mem_info())?.0;
            let bytes = free0.saturating_sub(free1);
            // every batch size's graph is charged to the engine's graph reserve
            self.graph_bytes.set(self.graph_bytes.get() + bytes);
            tracing::debug!(target: "ink_step", phase = "graph", event = "draft_capture", k, sampled, nodes = g.nodes().len(),
                bytes, free = free1);
            if sampled {
                self.graphs_sampled[k - 1] = Some(g);
            } else {
                self.graphs[k - 1] = Some(g);
            }
            return Ok(());
        }
        let g = if sampled {
            self.graphs_sampled[k - 1].as_ref()
        } else {
            self.graphs[k - 1].as_ref()
        }
        .ok_or_else(|| RuntimeError::request("draft graph"))?;
        rt(g.launch())
    }

    fn drafts(&self) -> ApiBuf {
        self.drafts
    }

    fn conf(&self) -> Option<ApiBuf> {
        self.conf
    }

    fn set_sampling(&mut self, temps: &[f32], top_ps: &[f32], seeds: &[u64]) -> Res<()> {
        let n = self.shape.n;
        if temps.len() != n || top_ps.len() != n || seeds.len() != n {
            return Err(RuntimeError::request(format!(
                "set_sampling: expected {n} entries, got {} temps / {} top_ps / {} seeds",
                temps.len(),
                top_ps.len(),
                seeds.len()
            )));
        }
        self.any_sampled = temps.iter().any(|&t| t >= 1e-5);
        let eff_temps: Vec<f32> = temps
            .iter()
            .map(|&t| if t >= 1e-5 { t * self.draft_temp_scale } else { t })
            .collect();
        let eff_top_ps: Vec<f32> = temps
            .iter()
            .zip(top_ps)
            .map(|(&t, &p)| {
                if t >= 1e-5 {
                    self.draft_top_p.unwrap_or_else(|| p.min(DEFAULT_DRAFT_TOPP))
                } else {
                    p
                }
            })
            .collect();
        let temp_bits: Vec<u32> = eff_temps.iter().map(|&x| x.to_bits()).collect();
        if self.last_eff_temps != temp_bits {
            rt(self.st.htod(&self.s.samp_temp, &eff_temps))?;
            self.last_eff_temps = temp_bits;
        }
        let topp_bits: Vec<u32> = eff_top_ps.iter().map(|&x| x.to_bits()).collect();
        if self.last_eff_top_ps != topp_bits {
            rt(self.st.htod(&self.s.samp_top_p, &eff_top_ps))?;
            self.last_eff_top_ps = topp_bits;
        }
        if self.last_seeds != seeds {
            rt(self.st.htod(&self.s.samp_seed, seeds))?;
            self.last_seeds.clear();
            self.last_seeds.extend_from_slice(seeds);
        }
        Ok(())
    }

    fn proposal(&self) -> Option<ProposalBufs> {
        self.proposal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_matches_the_cpp_constructor() {
        // n = 4, L = 53,192 (ink-spec's default draft context): pages ceil(53,200 / 64) = 832
        let s = Shape::new(4, 53_192, true, false, true);
        assert_eq!(s.max_blocks, 832);
        assert_eq!(s.t(), 32);
        // layers 0, 2: 1,032 keys = 17 tiles, min(17, ceil(384 / 32) = 12) = 12; layer 1: 832 tiles -> 12, and at
        // least ceil(834 / 256) = 4
        assert_eq!(s.split_attn, [12, 12, 12]);
        let one = Shape::new(1, 53_192, true, false, true);
        // n = 1: ceil(384 / 8) = 48; layers 0, 2 capped by their 17 tiles
        assert_eq!(one.split_attn, [17, 48, 17]);
        let short = Shape::new(8, 100, true, false, true);
        // 108 keys: 2 tiles, ceil(384 / 64) = 6
        assert_eq!(short.split_attn, [2, 2, 2]);
        assert_eq!(short.global_cache_elems(), (8 * 2 + 1) * 8 * 64 * 128);
        assert_eq!(short.swa_cache_elems(), (8 * 2 + 1) * 8 * 64 * 128);
        let long = Shape::new(1, 262_144, true, false, true);
        // 262,152 keys = 4,097 tiles: at least ceil(4,098 / 256) = 17 splits; 48 from the CTA target
        assert_eq!(long.split_attn[1], 48);
        let huge = Shape::new(8, 262_144, true, false, true);
        assert_eq!(huge.split_attn[1], 17);
        assert_eq!(huge.swa_blocks_per_sess, SWA_RING_BLOCKS);
        let pooled = Shape::new_with_pool(8, 262_144, true, true, true, Some(504));
        assert_eq!(pooled.l1_blocks, 504 * 16 + 8 + 1);
        // Pooled 262K DSpark with kv8=true + ring L0/L2 fits well under 5.5 GiB total (incl. weights + head + KV)
        assert!(arena_bytes(&pooled) < (5500 << 20));
    }

    #[test]
    fn arena_plan_takes_the_variant_buffers_only() {
        let i8 = arena_bytes(&Shape::new(4, 16_384, true, false, true));
        let bf = arena_bytes(&Shape::new(4, 16_384, false, false, true));
        let kv8 = arena_bytes(&Shape::new(4, 16_384, true, true, true));
        // the bf16 head adds head_f (VP x H bf16) and drops the int8 head (VP x H int8, two VP fp32), Bq (28 VP fp32)
        let al = |x: usize| x.next_multiple_of(ink_cuda::ALIGN);
        assert_eq!(
            bf + al(VP * H) + 2 * al(VP * 4) + al(4 * 7 * VP * 4),
            i8 + al(VP * H * 2)
        );
        // kv8 replaces L1 bf16 (2 bytes/elem) with L1 E4M3 (1 byte/elem), saving 2 * ge bytes
        let ge = Shape::new(4, 16_384, true, false, true).global_cache_elems();
        assert_eq!(kv8 + 2 * al(ge * 2), i8 + 2 * al(ge));
    }
}
