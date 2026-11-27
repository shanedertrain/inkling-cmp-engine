//! [`RustDFlash2`]: native Rust + CUDA implementation of `incoai/Inkling-2-Mini-DFlash2`.
//!
//! Implements [`ink_rt_api::Drafter`] as a drop-in replacement for DSpark:
//! - Same target taps `[5, 23, 41, 48]` (`4 x 2304` BF16 feature rows)
//! - All 5 draft layers use Tensor Core MMA SWA-2048 over a head-major 4,096-slot per-session ring buffer
//!   (`RING = 4096`), so speculation works across the full 262,144-token target context and rollback preserves
//!   `[P - 2048, P)` for replies up to 2,048 tokens.
//! - Shares the target engine's `embed`, `embed_norm`, and `unembed` buffers when created through
//!   [`Engine::create_drafter`](crate::Engine) (saving ~1.85 GB VRAM and load time), and quantizes the
//!   shared `lm_head` to W8A16 (`s2f::gemm_frag2`, 440 MiB, 0% acceptance loss; `INK_DF2_BF16_HEAD=1` forces BF16).
//! - Captures one CUDA graph per active batch size `k = 1..sessions` on the shared stream.
//!
//! Field order is drop order: graphs, cuBLASLt handle, buffers, kernels, arena, stream, context.

mod ops;

use std::cell::{Cell, RefCell};
use std::fs;
use std::path::Path;
use std::rc::Rc;

use ink_cuda::{AlgoPins, Arena, Bf16, Context, CudaError, DevBuf, Dtype, GemmShape, Graph, LtGemm, PlanClass, Stream};
use ink_kernels::{F16, KernelId, Kernels};
use ink_model::dflash2::{
    self as m, ATTN_SMEM, ATTN_SPLITS, BLK, CONV_PROJ_N, COS_SIN_POS, DFlash2Checkpoints, DFlash2Config, DRAFTS,
    GLOBAL_ATTN_SMEM, GLOBAL_ATTN_SPLITS, H, HD, HEAD_N, INTER, KVH, LayerType, MASK_TOKEN_ID, NH, NLAYERS, RING,
    SEL_RANK, TOP_K, TOPK_CHUNKS, VOCAB_PAD,
};
use ink_rt_api::{DevBuf as ApiBuf, DraftConfig, Drafter, FEAT_ROW_BYTES, ProposalBufs, RuntimeError, VerifyWidth};

use self::ops::{Dev, head_gemm_ids, head_gemm_smem};
use crate::devbufs::DevBufs;
use crate::error::rt;

type Res<T> = Result<T, RuntimeError>;

/// Maximum rows processed per chunk in `commit_rows`.
const COMMIT_CHUNK: usize = 2048;

/// Shared target embedding/unembedding device buffers (borrowed via `DevBuf` `Arc` clones from `Engine`).
#[derive(Clone)]
pub(crate) struct SharedTargetBufs {
    pub embed: DevBuf<Bf16>,
    pub unembed: DevBuf<Bf16>,
}

struct LayerW {
    in_norm: DevBuf<Bf16>,
    post_norm: DevBuf<Bf16>,
    attn_conv_base: DevBuf<Bf16>,
    attn_conv_proj: DevBuf<Bf16>,
    mlp_conv_base: DevBuf<Bf16>,
    mlp_conv_proj: DevBuf<Bf16>,
    /// Fused `[Q(4096), K(1024), V(1024)]` x H = `[6144, 2304]`. `kv_proj` slices rows `4096..6144`.
    qkv_proj: DevBuf<Bf16>,
    o_proj: DevBuf<Bf16>,
    q_norm: DevBuf<Bf16>,
    k_norm: DevBuf<Bf16>,
    /// Fused `[Gate(6912), Up(6912)]` x H = `[13824, 2304]`.
    gate_up_proj: DevBuf<Bf16>,
    down_proj: DevBuf<Bf16>,
}

struct Weights {
    fc: DevBuf<Bf16>,
    hidden_norm: DevBuf<Bf16>,
    norm: DevBuf<Bf16>,
    sel_proj: DevBuf<Bf16>,
    pred_cb: DevBuf<Bf16>,
    succ_cb: DevBuf<Bf16>,
    layers: Vec<LayerW>,
    embed: DevBuf<Bf16>,
    embed_norm: DevBuf<Bf16>,
    /// First `HEAD_N` (200,064) rows of `unembed` (`[200064, 2304]`).
    head: DevBuf<Bf16>,
    /// W8A16 per-row quantized fragment-permuted `head` (`[200064, 2304]` i8 + `[200064]` f32 scale), unless BF16 forced.
    head_i8: Option<(DevBuf<i8>, DevBuf<f32>)>,
    mask_emb: DevBuf<Bf16>,
    cos_sin: DevBuf<Bf16>,
    cos_sin_global: Option<DevBuf<Bf16>>,
}

enum LayerKv {
    Swa {
        k_ring: DevBuf<Bf16>,
        v_ring: DevBuf<Bf16>,
    },
    GlobalE4m3 {
        k_glob: DevBuf<u8>,
        v_glob: DevBuf<u8>,
        ks_glob: DevBuf<f32>,
        vs_glob: DevBuf<f32>,
    },
}

struct State {
    kv: Vec<LayerKv>,
    // Commit scratch (up to COMMIT_CHUNK rows)
    ctx_h: DevBuf<Bf16>,
    ctx_norm: DevBuf<Bf16>,
    kv_tmp: DevBuf<Bf16>,
    // Draft block scratch (up to n * BLK rows)
    x: DevBuf<Bf16>,
    h: DevBuf<Bf16>,
    h_c: DevBuf<Bf16>,
    sub: DevBuf<Bf16>,
    dyn_buf: DevBuf<Bf16>,
    qkv: DevBuf<Bf16>,
    q_blk: DevBuf<Bf16>,
    k_blk: DevBuf<Bf16>,
    v_blk: DevBuf<Bf16>,
    part_ml: DevBuf<f32>,
    part_o: DevBuf<f32>,
    attn_out: DevBuf<Bf16>,
    gate_up: DevBuf<Bf16>,
    mlp_act: DevBuf<Bf16>,
    pred_h: DevBuf<Bf16>,
    pred_h_f16: DevBuf<F16>,
    logits: DevBuf<f32>,
    hp: DevBuf<f32>,
    chunk_max: DevBuf<f32>,
    chunk_z: DevBuf<f32>,
    cand_count: DevBuf<i32>,
    s1_vals: DevBuf<f32>,
    s1_ids: DevBuf<i32>,
    unary: DevBuf<f32>,
    cand_ids: DevBuf<i32>,
    row_z: DevBuf<f32>,
    // Session state
    anchor: DevBuf<i32>,
    pos: DevBuf<i32>,
    sel: DevBuf<i32>,
    samp_temp: DevBuf<f32>,
    samp_top_p: DevBuf<f32>,
    samp_seed: DevBuf<u64>,
    drafts: DevBuf<i32>,
    prop_ids: DevBuf<i32>,
    prop_w: DevBuf<u32>,
    conf: DevBuf<f32>,
}

fn shape(n: usize, k: usize, output: Dtype, class: PlanClass) -> GemmShape {
    GemmShape {
        n: n as u32,
        k: k as u32,
        ldw: k as u32,
        input: Dtype::Bf16,
        output,
        class,
    }
}

fn df2_shapes() -> Vec<(GemmShape, u32)> {
    vec![
        // Commit GEMMs (both decode-size and prefill-chunk-size)
        (shape(H, 4 * H, Dtype::Bf16, PlanClass::Decode), 32),
        (shape(H, 4 * H, Dtype::Bf16, PlanClass::Mixed), COMMIT_CHUNK as u32),
        (shape(2 * KVH * HD, H, Dtype::Bf16, PlanClass::Decode), 32),
        (
            shape(2 * KVH * HD, H, Dtype::Bf16, PlanClass::Mixed),
            COMMIT_CHUNK as u32,
        ),
        // Draft block GEMMs (M = 8 * k, tuned around k = 4 -> M = 32)
        (shape(CONV_PROJ_N, H, Dtype::Bf16, PlanClass::Decode), 32),
        (shape((NH + 2 * KVH) * HD, H, Dtype::Bf16, PlanClass::Decode), 32),
        (shape(H, NH * HD, Dtype::Bf16, PlanClass::Decode), 32),
        (shape(2 * INTER, H, Dtype::Bf16, PlanClass::Decode), 32),
        (shape(H, INTER, Dtype::Bf16, PlanClass::Decode), 32),
        // Selector projection + BF16 LM head fallback (M = 7 * k, tuned around k = 4 -> M = 28)
        (shape(SEL_RANK, H, Dtype::F32, PlanClass::Decode), 28),
        (shape(HEAD_N, H, Dtype::F32, PlanClass::Decode), 28),
    ]
}

fn part_slots_for(sessions: usize, has_global: bool) -> usize {
    let base = sessions * ATTN_SPLITS;
    if has_global {
        base.max(72).max(sessions * 8)
    } else {
        base
    }
}

fn arena_bytes(
    sessions: usize,
    kv_stride: usize,
    cfg: &DFlash2Config,
    shared_target: bool,
    use_i8_head: bool,
) -> usize {
    let w_layers = NLAYERS
        * ((2 * H
            + 4 * H
            + CONV_PROJ_N * H
            + 4 * H
            + CONV_PROJ_N * H
            + (NH + 2 * KVH) * HD * H
            + H * NH * HD
            + 2 * HD
            + 2 * INTER * H
            + H * INTER)
            * 2);
    let cos_sin_tables = 1 + usize::from(cfg.has_global());
    let w_other =
        (H * 4 * H + 2 * H + SEL_RANK * H + 2 * VOCAB_PAD * SEL_RANK + H + cos_sin_tables * COS_SIN_POS * HD) * 2;
    let w_target = if shared_target {
        0
    } else {
        (VOCAB_PAD * H + H + HEAD_N * H) * 2
    };
    let w_head_i8 = if use_i8_head { HEAD_N * H + HEAD_N * 4 } else { 0 };
    let kv_swa = cfg.swa_count() * 2 * sessions * KVH * RING * HD * 2;
    let kv_global = cfg.global_count() * (2 * sessions * KVH * kv_stride * HD + 2 * sessions * KVH * kv_stride * 4);
    let commit_s = (COMMIT_CHUNK * (2 * H + 2 * KVH * HD)) * 2;
    let rows = sessions * BLK;
    let drows = sessions * DRAFTS;
    let part_slots = part_slots_for(sessions, cfg.has_global());
    let blk_s = rows * (4 * H + CONV_PROJ_N + (NH + 2 * KVH) * HD + (NH + 2 * KVH) * HD + NH * HD + 3 * INTER) * 2
        + part_slots * KVH * 32 * (2 + HD) * 4
        + drows * H * 4
        + drows * (HEAD_N + SEL_RANK + TOPK_CHUNKS + 1 + 2 * TOPK_CHUNKS * TOP_K + 2 * TOP_K) * 4
        + sessions * (3 + DRAFTS) * 4;
    // Generous alignment slack (64 MiB)
    w_layers + w_other + w_target + w_head_i8 + kv_swa + kv_global + commit_s + blk_s + (64 << 20)
}

/// Split-KV partitions for `k` active sessions on SWA-2048 layers (targets >= 192 CTAs across 96 SMs).
fn splits_for(k: usize) -> usize {
    (256 / (k * KVH)).clamp(4, ATTN_SPLITS)
}

/// Split-KV partitions for `k` active sessions on global E4M3 layers (Design A: 2 warps/CTA, targets 296 CTAs = 4 CTAs/SM x 74 SMs).
fn global_splits_for(k: usize) -> usize {
    let ctas = (296 / (k * KVH)).clamp(4, GLOBAL_ATTN_SPLITS / 2);
    ctas * 2
}

/// Native Rust + CUDA DFlash2 speculative drafter.
pub struct RustDFlash2 {
    graphs: Vec<Option<Graph>>,
    graph_bytes: Rc<Cell<usize>>,
    lt: RefCell<LtGemm>,
    cfg: DFlash2Config,
    w: Weights,
    s: State,
    sessions: usize,
    max_ctx: usize,
    kv_stride: usize,
    drafts_api: ApiBuf,
    conf_api: Option<ApiBuf>,
    prop_api: Option<ProposalBufs>,
    use_top_p: bool,
    draft_temp_scale: f32,
    draft_top_p: Option<f32>,
    conf_mode: i32,
    bufs: Rc<RefCell<DevBufs>>,
    k: Kernels,
    _arena: Arena,
    st: Rc<Stream>,
    ctx: Rc<Context>,
}

impl std::fmt::Debug for RustDFlash2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RustDFlash2")
            .field("sessions", &self.sessions)
            .field("max_ctx", &self.max_ctx)
            .finish_non_exhaustive()
    }
}

impl RustDFlash2 {
    /// Standalone constructor (loads target `embed`, `embed_norm`, `unembed` from `config.model_dir`).
    ///
    /// # Errors
    ///
    /// [`RuntimeError`] on invalid configuration, missing checkpoint, or CUDA failure.
    pub fn new(config: &DraftConfig) -> Res<Self> {
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

    /// Creates a DFlash2 drafter sharing an engine's context, stream, buffer registry, and (optionally) target
    /// embedding/unembedding weights.
    ///
    /// # Errors
    ///
    /// [`RuntimeError`] on invalid configuration, missing checkpoint, or CUDA failure.
    pub(crate) fn on(
        ctx: Rc<Context>,
        st: Rc<Stream>,
        bufs: Rc<RefCell<DevBufs>>,
        graph_bytes: Rc<Cell<usize>>,
        config: &DraftConfig,
        shared_target: Option<SharedTargetBufs>,
    ) -> Res<Self> {
        let dflash_dir = config
            .dflash2_dir
            .as_deref()
            .ok_or_else(|| RuntimeError::request("RustDFlash2 requires DraftConfig::dflash2_dir"))?;
        let cfg = rt(DFlash2Config::load(dflash_dir))?;
        let sessions = config.sessions as usize;
        let max_ctx = config.max_ctx as usize;
        if sessions == 0 || max_ctx == 0 {
            return Err(RuntimeError::request("dflash2: sessions and max_ctx must be > 0"));
        }
        if max_ctx + BLK > COS_SIN_POS {
            return Err(RuntimeError::request(format!(
                "dflash2: max_ctx {max_ctx} + {BLK} exceeds RoPE table {COS_SIN_POS}"
            )));
        }
        let kv_stride = max_ctx.next_multiple_of(16);
        let k = rt(Kernels::load(&ctx))?;
        rt(k.function(KernelId::Df2SwaAttnSplitKernel)
            .set_max_dynamic_smem(ATTN_SMEM as u32))?;
        if cfg.has_global() {
            rt(k.function(KernelId::Df2GlobalAttnE4m3SplitKernel)
                .set_max_dynamic_smem(GLOBAL_ATTN_SMEM as u32))?;
        }
        for (id, mt) in head_gemm_ids() {
            rt(k.function(id).set_max_dynamic_smem(head_gemm_smem(mt) as u32))?;
        }

        let is_shared = shared_target.is_some();
        let use_i8_head = !std::env::var("INK_DF2_BF16_HEAD").is_ok_and(|v| v != "0");
        let total = arena_bytes(sessions, kv_stride, &cfg, is_shared, use_i8_head);
        let mut arena = rt(Arena::new(&ctx, &st, total))?;

        let ck = rt(DFlash2Checkpoints::open(dflash_dir, Some(config.model_dir.as_path())))?;
        let df = &ck.dflash;
        let tgt = ck.target.as_ref().expect("target checkpoint opened");

        let up = |buf: &DevBuf<Bf16>, bytes: &[u8]| rt(st.htod_bytes(buf, bytes));
        let load = |arena: &mut Arena, name: &str, elems: usize| -> Res<DevBuf<Bf16>> {
            let b = rt(arena.take::<Bf16>(elems, "df2 weight"))?;
            up(&b, rt(m::bf16_bytes(df, name, elems))?)?;
            Ok(b)
        };

        let fc = load(&mut arena, "fc.weight", H * 4 * H)?;
        let hidden_norm = load(&mut arena, "hidden_norm.weight", H)?;
        let norm = load(&mut arena, "norm.weight", H)?;
        let sel_proj = load(&mut arena, "candidate_selector.hidden_projection.weight", SEL_RANK * H)?;
        let pred_cb = load(
            &mut arena,
            "candidate_selector.predecessor_codebook",
            VOCAB_PAD * SEL_RANK,
        )?;
        let succ_cb = load(
            &mut arena,
            "candidate_selector.successor_codebook",
            VOCAB_PAD * SEL_RANK,
        )?;

        let mut layers = Vec::with_capacity(NLAYERS);
        for i in 0..NLAYERS {
            let p = format!("layers.{i}.");
            let a = format!("{p}self_attn.");
            let in_norm = load(&mut arena, &format!("{p}input_layernorm.weight"), H)?;
            let post_norm = load(&mut arena, &format!("{p}post_attention_layernorm.weight"), H)?;
            let attn_conv_base = load(&mut arena, &format!("{p}attention_conv.base_kernel"), 4 * H)?;
            let attn_conv_proj = load(
                &mut arena,
                &format!("{p}attention_conv.kernel_projection.weight"),
                CONV_PROJ_N * H,
            )?;
            let mlp_conv_base = load(&mut arena, &format!("{p}mlp_conv.base_kernel"), 4 * H)?;
            let mlp_conv_proj = load(
                &mut arena,
                &format!("{p}mlp_conv.kernel_projection.weight"),
                CONV_PROJ_N * H,
            )?;
            let qkv_bytes = rt(m::cat_bf16(
                df,
                &[
                    (&format!("{a}q_proj.weight"), NH * HD * H),
                    (&format!("{a}k_proj.weight"), KVH * HD * H),
                    (&format!("{a}v_proj.weight"), KVH * HD * H),
                ],
            ))?;
            let qkv_proj = rt(arena.take::<Bf16>((NH + 2 * KVH) * HD * H, "df2 qkv_proj"))?;
            up(&qkv_proj, &qkv_bytes)?;
            let o_proj = load(&mut arena, &format!("{a}o_proj.weight"), H * NH * HD)?;
            let q_norm = load(&mut arena, &format!("{a}q_norm.weight"), HD)?;
            let k_norm = load(&mut arena, &format!("{a}k_norm.weight"), HD)?;
            let gu_bytes = rt(m::cat_bf16(
                df,
                &[
                    (&format!("{p}mlp.gate_proj.weight"), INTER * H),
                    (&format!("{p}mlp.up_proj.weight"), INTER * H),
                ],
            ))?;
            let gate_up_proj = rt(arena.take::<Bf16>(2 * INTER * H, "df2 gate_up_proj"))?;
            up(&gate_up_proj, &gu_bytes)?;
            let down_proj = load(&mut arena, &format!("{p}mlp.down_proj.weight"), H * INTER)?;
            layers.push(LayerW {
                in_norm,
                post_norm,
                attn_conv_base,
                attn_conv_proj,
                mlp_conv_base,
                mlp_conv_proj,
                qkv_proj,
                o_proj,
                q_norm,
                k_norm,
                gate_up_proj,
                down_proj,
            });
        }

        let embed_norm = rt(arena.take::<Bf16>(H, "df2 embed_norm"))?;
        up(&embed_norm, rt(m::bf16_bytes(tgt, "model.llm.embed_norm.weight", H))?)?;
        let (embed, head) = match shared_target {
            Some(sh) => {
                let head = rt(sh.unembed.slice(0..HEAD_N * H))?;
                (sh.embed, head)
            }
            None => {
                let embed = rt(arena.take::<Bf16>(VOCAB_PAD * H, "df2 embed"))?;
                up(&embed, rt(m::bf16_bytes(tgt, "model.llm.embed.weight", VOCAB_PAD * H))?)?;
                let head = rt(arena.take::<Bf16>(HEAD_N * H, "df2 head"))?;
                let unembed_all = rt(m::bf16_bytes(tgt, "model.llm.unembed.weight", VOCAB_PAD * H))?;
                up(&head, &unembed_all[..HEAD_N * H * 2])?;
                (embed, head)
            }
        };

        let dev = Dev { k: &k, st: &st };
        let head_i8 = if use_i8_head {
            let head_q8f = rt(arena.take::<i8>(HEAD_N * H, "df2 head_q8f"))?;
            let head_scale = rt(arena.take::<f32>(HEAD_N, "df2 head_scale"))?;
            // Temporary scratch for quantization + fragment permutation, freed immediately after sync.
            let mut tmp = rt(Arena::new(&ctx, &st, HEAD_N * H + HEAD_N * 4 + (8 << 20)))?;
            let q_tmp = rt(tmp.take::<i8>(HEAD_N * H, "df2 q_tmp"))?;
            let enorm_tmp = rt(tmp.take::<f32>(HEAD_N, "df2 enorm_tmp"))?;
            rt(dev.quant_rows_i8(&head, &q_tmp, &head_scale, &enorm_tmp, HEAD_N, H))?;
            rt(dev.permute_frag_i8(&q_tmp, &head_q8f, HEAD_N, H))?;
            rt(st.synchronize())?;
            Some((head_q8f, head_scale))
        } else {
            None
        };

        let mask_emb = rt(arena.take::<Bf16>(H, "df2 mask_emb"))?;
        let cos_sin = rt(arena.take::<Bf16>(COS_SIN_POS * HD, "df2 cos_sin"))?;
        let cos_sin_global = if cfg.has_global() {
            Some(rt(arena.take::<Bf16>(COS_SIN_POS * HD, "df2 cos_sin_global"))?)
        } else {
            None
        };

        // Allocate per-layer KV cache (SWA-2048 BF16 ring or Global E4M3 linear [sessions, KVH, kv_stride, HD])
        let mut kv = Vec::with_capacity(NLAYERS);
        for &lt in &cfg.layer_types {
            match lt {
                LayerType::SlidingAttention => {
                    let k_ring = rt(arena.take::<Bf16>(sessions * KVH * RING * HD, "df2 k_ring"))?;
                    let v_ring = rt(arena.take::<Bf16>(sessions * KVH * RING * HD, "df2 v_ring"))?;
                    kv.push(LayerKv::Swa { k_ring, v_ring });
                }
                LayerType::FullAttention => {
                    let k_glob = rt(arena.take::<u8>(sessions * KVH * kv_stride * HD, "df2 k_glob"))?;
                    let v_glob = rt(arena.take::<u8>(sessions * KVH * kv_stride * HD, "df2 v_glob"))?;
                    let ks_glob = rt(arena.take::<f32>(sessions * KVH * kv_stride, "df2 ks_glob"))?;
                    let vs_glob = rt(arena.take::<f32>(sessions * KVH * kv_stride, "df2 vs_glob"))?;
                    rt(st.memset_zero(&ks_glob))?;
                    rt(st.memset_zero(&vs_glob))?;
                    kv.push(LayerKv::GlobalE4m3 {
                        k_glob,
                        v_glob,
                        ks_glob,
                        vs_glob,
                    });
                }
            }
        }
        let rows = sessions * BLK;
        let drows = sessions * DRAFTS;
        let part_slots = part_slots_for(sessions, cfg.has_global());
        let s = State {
            kv,
            ctx_h: rt(arena.take::<Bf16>(COMMIT_CHUNK * H, "df2 ctx_h"))?,
            ctx_norm: rt(arena.take::<Bf16>(COMMIT_CHUNK * H, "df2 ctx_norm"))?,
            kv_tmp: rt(arena.take::<Bf16>(COMMIT_CHUNK * 2 * KVH * HD, "df2 kv_tmp"))?,
            x: rt(arena.take::<Bf16>(rows * H, "df2 x"))?,
            h: rt(arena.take::<Bf16>(rows * H, "df2 h"))?,
            h_c: rt(arena.take::<Bf16>(rows * H, "df2 h_c"))?,
            sub: rt(arena.take::<Bf16>(rows * H, "df2 sub"))?,
            dyn_buf: rt(arena.take::<Bf16>(rows * CONV_PROJ_N, "df2 dyn"))?,
            qkv: rt(arena.take::<Bf16>(rows * (NH + 2 * KVH) * HD, "df2 qkv"))?,
            q_blk: rt(arena.take::<Bf16>(rows * NH * HD, "df2 q_blk"))?,
            k_blk: rt(arena.take::<Bf16>(rows * KVH * HD, "df2 k_blk"))?,
            v_blk: rt(arena.take::<Bf16>(rows * KVH * HD, "df2 v_blk"))?,
            part_ml: rt(arena.take::<f32>(part_slots * KVH * 32 * 2, "df2 part_ml"))?,
            part_o: rt(arena.take::<f32>(part_slots * KVH * 32 * HD, "df2 part_o"))?,
            attn_out: rt(arena.take::<Bf16>(rows * NH * HD, "df2 attn_out"))?,
            gate_up: rt(arena.take::<Bf16>(rows * 2 * INTER, "df2 gate_up"))?,
            mlp_act: rt(arena.take::<Bf16>(rows * INTER, "df2 mlp_act"))?,
            pred_h: rt(arena.take::<Bf16>(drows * H, "df2 pred_h"))?,
            pred_h_f16: rt(arena.take::<F16>(drows * H, "df2 pred_h_f16"))?,
            logits: rt(arena.take::<f32>(drows * HEAD_N, "df2 logits"))?,
            hp: rt(arena.take::<f32>(drows * SEL_RANK, "df2 hp"))?,
            chunk_max: rt(arena.take::<f32>(drows * TOPK_CHUNKS, "df2 chunk_max"))?,
            chunk_z: rt(arena.take::<f32>(drows * TOPK_CHUNKS, "df2 chunk_z"))?,
            cand_count: rt(arena.take::<i32>(drows, "df2 cand_count"))?,
            s1_vals: rt(arena.take::<f32>(drows * TOPK_CHUNKS * TOP_K, "df2 s1_vals"))?,
            s1_ids: rt(arena.take::<i32>(drows * TOPK_CHUNKS * TOP_K, "df2 s1_ids"))?,
            unary: rt(arena.take::<f32>(drows * TOP_K, "df2 unary"))?,
            cand_ids: rt(arena.take::<i32>(drows * TOP_K, "df2 cand_ids"))?,
            row_z: rt(arena.take::<f32>(drows, "df2 row_z"))?,
            anchor: rt(arena.take::<i32>(sessions, "df2 anchor"))?,
            pos: rt(arena.take::<i32>(sessions, "df2 pos"))?,
            sel: rt(arena.take::<i32>(sessions, "df2 sel"))?,
            samp_temp: rt(arena.take::<f32>(sessions, "df2 samp_temp"))?,
            samp_top_p: rt(arena.take::<f32>(sessions, "df2 samp_top_p"))?,
            samp_seed: rt(arena.take::<u64>(sessions, "df2 samp_seed"))?,
            drafts: rt(arena.take::<i32>(sessions * DRAFTS, "df2 drafts"))?,
            prop_ids: rt(arena.take::<i32>(sessions * DRAFTS * TOP_K, "df2 prop_ids"))?,
            prop_w: rt(arena.take::<u32>(sessions * DRAFTS * TOP_K, "df2 prop_w"))?,
            conf: rt(arena.take::<f32>(sessions * DRAFTS, "df2 conf"))?,
        };
        rt(st.memset_zero(&s.anchor))?;
        rt(st.memset_zero(&s.pos))?;
        rt(st.memset_zero(&s.sel))?;
        rt(st.memset_zero(&s.samp_temp))?;
        rt(st.htod(&s.samp_top_p, &vec![1.0f32; sessions]))?;
        rt(st.memset_zero(&s.samp_seed))?;
        rt(st.memset_zero(&s.drafts))?;
        rt(st.memset_zero(&s.prop_ids))?;
        rt(st.memset_zero(&s.prop_w))?;
        rt(st.memset_zero(&s.conf))?;

        // Precompute RoPE tables and normalized mask embedding
        rt(dev.rope_table(&cos_sin, COS_SIN_POS, cfg.rope_theta))?;
        if let Some(ref csg) = cos_sin_global {
            rt(dev.rope_table(csg, COS_SIN_POS, cfg.global_rope_theta))?;
        }
        let mask_raw = rt(embed.slice(MASK_TOKEN_ID * H..(MASK_TOKEN_ID + 1) * H))?;
        rt(dev.rms_norm(&mask_raw, &embed_norm, &mask_emb, 1, false))?;
        rt(st.synchronize())?;

        // Select cuBLASLt algorithms for all DFlash2 GEMM shapes (workspace-free, split-K=1)
        let mut selector = rt(LtGemm::new(&st, AlgoPins::new(0)))?;
        let mut pins = AlgoPins::new(ctx.startup().cublaslt_version);
        for (sh, m) in df2_shapes() {
            let sel_pin = rt(selector.select(sh, m))?;
            rt(pins.insert(sh, sel_pin.pin))?;
        }
        let lt = rt(LtGemm::new(&st, pins))?;

        let drafts_api = bufs.borrow_mut().register(rt(s.drafts.cast::<u8>())?);
        let conf_api = if config.verify != VerifyWidth::Full {
            Some(bufs.borrow_mut().register(rt(s.conf.cast::<u8>())?))
        } else {
            None
        };
        let prop_ids_api = bufs.borrow_mut().register(rt(s.prop_ids.cast::<u8>())?);
        let prop_w_api = bufs.borrow_mut().register(rt(s.prop_w.cast::<u8>())?);
        let use_leviathan = std::env::var("INK_DF2_NO_LEVIATHAN").is_err_and(|_| true)
            || std::env::var("INK_DF2_NO_LEVIATHAN").is_ok_and(|v| v == "0");
        // DFlash2's bilinear selector scores are not T=1 calibrated; unfiltered top-16 beats top_p (3.898 vs 3.786).
        let use_top_p = std::env::var("INK_DF2_TOPP").is_ok_and(|v| v == "1");
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
        let prop_api = use_leviathan.then_some(ProposalBufs {
            drafts: drafts_api,
            ids: prop_ids_api,
            weights: prop_w_api,
        });
        eprintln!(
            "[dflash2] arena used {:.2} GiB (sessions {sessions}, max_ctx {max_ctx}, layers {} swa + {} global, shared_target {is_shared}, head {})",
            arena.plan().used() as f64 / 1_073_741_824.0,
            cfg.swa_count(),
            cfg.global_count(),
            if use_i8_head { "int8" } else { "bf16" },
        );

        let w = Weights {
            fc,
            hidden_norm,
            norm,
            sel_proj,
            pred_cb,
            succ_cb,
            layers,
            embed,
            embed_norm,
            head,
            head_i8,
            mask_emb,
            cos_sin,
            cos_sin_global,
        };
        Ok(Self {
            graphs: (0..sessions).map(|_| None).collect(),
            graph_bytes,
            lt: RefCell::new(lt),
            cfg,
            w,
            s,
            sessions,
            max_ctx,
            kv_stride,
            drafts_api,
            conf_api,
            prop_api,
            use_top_p,
            draft_temp_scale,
            draft_top_p,
            conf_mode,
            bufs,
            k,
            _arena: arena,
            st,
            ctx,
        })
    }

    /// Allocates a device buffer in this drafter's registry (standalone `ink-draft`).
    pub fn alloc(&self, bytes: usize) -> Res<ApiBuf> {
        self.bufs.borrow_mut().alloc(&self.ctx, &self.st, bytes)
    }

    /// Uploads raw bytes to a registered buffer (standalone `ink-draft`).
    pub fn upload(&self, dst: ApiBuf, src: &[u8]) -> Res<()> {
        let b = self.bufs.borrow();
        let d = b.typed::<u8>(dst, src.len(), "df2 upload")?;
        rt(self.st.htod_bytes(&d, src))
    }

    /// Downloads raw bytes from a registered buffer (standalone `ink-draft`).
    pub fn download(&self, src: ApiBuf, dst: &mut [u8]) -> Res<()> {
        let b = self.bufs.borrow();
        let s = b.typed::<u8>(src, dst.len(), "df2 download")?;
        rt(self.st.dtoh(dst, &s))
    }

    /// Waits for queued work on the drafter's stream.
    pub fn sync(&self) -> Res<()> {
        rt(self.st.synchronize())
    }

    fn dev(&self) -> Dev<'_> {
        Dev {
            k: &self.k,
            st: &self.st,
        }
    }

    #[expect(clippy::too_many_arguments, reason = "GEMM shape and operands")]
    fn gemm_bf16(
        &self,
        n: usize,
        k: usize,
        class: PlanClass,
        x: &DevBuf<Bf16>,
        w: &DevBuf<Bf16>,
        out: &DevBuf<Bf16>,
        m: usize,
    ) -> Result<(), CudaError> {
        self.lt.borrow_mut().run(
            &self.st,
            shape(n, k, Dtype::Bf16, class),
            x,
            k as u32,
            w,
            out,
            n as u32,
            m as u32,
        )
    }

    #[expect(clippy::too_many_arguments, reason = "GEMM shape and operands")]
    fn gemm_f32(
        &self,
        n: usize,
        k: usize,
        class: PlanClass,
        x: &DevBuf<Bf16>,
        w: &DevBuf<Bf16>,
        out: &DevBuf<f32>,
        m: usize,
    ) -> Result<(), CudaError> {
        self.lt.borrow_mut().run(
            &self.st,
            shape(n, k, Dtype::F32, class),
            x,
            k as u32,
            w,
            out,
            n as u32,
            m as u32,
        )
    }

    fn commit_impl(
        &self,
        feats: &DevBuf<Bf16>,
        pos: &DevBuf<i32>,
        sess: &DevBuf<i32>,
        rows: usize,
    ) -> Result<(), CudaError> {
        let dev = self.dev();
        let (w, s) = (&self.w, &self.s);
        for r0 in (0..rows).step_by(COMMIT_CHUNK) {
            let rc = COMMIT_CHUNK.min(rows - r0);
            let cls = if rc <= 64 { PlanClass::Decode } else { PlanClass::Mixed };
            let f = feats.slice(r0 * 4 * H..(r0 + rc) * 4 * H)?;
            let p = pos.slice(r0..r0 + rc)?;
            let q = sess.slice(r0..r0 + rc)?;
            let ch = s.ctx_h.slice(0..rc * H)?;
            let cn = s.ctx_norm.slice(0..rc * H)?;
            let kv = s.kv_tmp.slice(0..rc * 2 * KVH * HD)?;

            self.gemm_bf16(H, 4 * H, cls, &f, &w.fc, &ch, rc)?;
            dev.rms_norm(&ch, &w.hidden_norm, &cn, rc, false)?;

            let kv_off = NH * HD * H;
            let kv_elems = 2 * KVH * HD * H;
            for (li, ly) in w.layers.iter().enumerate() {
                let w_kv = ly.qkv_proj.slice(kv_off..kv_off + kv_elems)?;
                self.gemm_bf16(2 * KVH * HD, H, cls, &cn, &w_kv, &kv, rc)?;
                match &s.kv[li] {
                    LayerKv::Swa { k_ring, v_ring } => {
                        dev.kv_norm_rope_store(&kv, &ly.k_norm, &w.cos_sin, &p, &q, k_ring, v_ring, rc, self.sessions)?;
                    }
                    LayerKv::GlobalE4m3 {
                        k_glob,
                        v_glob,
                        ks_glob,
                        vs_glob,
                    } => {
                        let csg = w.cos_sin_global.as_ref().expect("global cos_sin table");
                        dev.kv_norm_rope_store_global_e4m3(
                            &kv,
                            &ly.k_norm,
                            csg,
                            &p,
                            &q,
                            k_glob,
                            v_glob,
                            ks_glob,
                            vs_glob,
                            rc,
                            self.sessions,
                            self.max_ctx,
                            self.kv_stride,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn draft_batch(&self, k: usize, prof: bool, dump_dir: Option<&Path>) -> Result<(), CudaError> {
        const PRE_GLOB_MARKS: [&str; NLAYERS] = ["l0_pre", "l1_pre", "l2_pre", "l3_pre", "l4_pre"];
        const GLOB_MARKS: [&str; NLAYERS] = ["l0_global", "l1_global", "l2_global", "l3_global", "l4_global"];
        let dev = self.dev();
        let (w, s) = (&self.w, &self.s);
        let rows = k * BLK;
        let drows = k * DRAFTS;
        let splits = splits_for(k);
        let g_splits = global_splits_for(k);
        let mut evs: Vec<(&'static str, ink_cuda::Event)> = Vec::new();
        let mut mark = |name: &'static str| -> Result<(), CudaError> {
            if prof {
                let e = self.ctx.new_event()?;
                self.st.record(&e)?;
                evs.push((name, e));
            }
            Ok(())
        };
        mark("start")?;

        // 1. Block embeddings [k, 8, H] and layer-0 input RMSNorm
        dev.embed_block(&w.embed, &w.embed_norm, &w.mask_emb, &s.anchor, &s.sel, &s.x, k)?;
        dev.rms_norm(&s.x, &w.layers[0].in_norm, &s.h, rows, false)?;
        mark("embed")?;

        // 2. Five Qwen3-DFlash2 decoder layers (hybrid SWA-2048 BF16 ring + optional Global E4M3)
        for (li, ly) in w.layers.iter().enumerate() {
            let is_glob = self.cfg.is_global(li);
            // Attention conv prepare
            self.gemm_bf16(
                CONV_PROJ_N,
                H,
                PlanClass::Decode,
                &s.h,
                &ly.attn_conv_proj,
                &s.dyn_buf,
                rows,
            )?;
            dev.grouped_conv_in(&s.h, &s.dyn_buf, &ly.attn_conv_base, &s.h_c, k)?;
            if li == 0 {
                mark("l0_aconv")?;
            }

            // QKV projection + QK norm + NeoX RoPE
            self.gemm_bf16(
                (NH + 2 * KVH) * HD,
                H,
                PlanClass::Decode,
                &s.h_c,
                &ly.qkv_proj,
                &s.qkv,
                rows,
            )?;
            if li == 0 {
                mark("l0_qkv")?;
            }
            let cos_sin = if is_glob {
                w.cos_sin_global.as_ref().expect("global cos_sin table")
            } else {
                &w.cos_sin
            };
            dev.qkv_norm_rope_blk(
                &s.qkv, &ly.q_norm, &ly.k_norm, cos_sin, &s.pos, &s.sel, &s.q_blk, &s.k_blk, &s.v_blk, k,
            )?;
            if li == 0 {
                mark("l0_rope")?;
            } else if is_glob {
                mark(PRE_GLOB_MARKS[li])?;
            }

            // Tensor Core MMA attention: SWA-2048 BF16 ring or Global E4M3 + exact BF16 block keys
            match &s.kv[li] {
                LayerKv::Swa { k_ring, v_ring } => {
                    dev.swa_attn(
                        &s.q_blk,
                        &s.k_blk,
                        &s.v_blk,
                        k_ring,
                        v_ring,
                        &s.pos,
                        &s.sel,
                        &s.part_ml,
                        &s.part_o,
                        &s.attn_out,
                        k,
                        self.sessions,
                        splits,
                    )?;
                }
                LayerKv::GlobalE4m3 {
                    k_glob,
                    v_glob,
                    ks_glob,
                    vs_glob,
                } => {
                    dev.global_attn_e4m3(
                        &s.q_blk,
                        &s.k_blk,
                        &s.v_blk,
                        k_glob,
                        v_glob,
                        ks_glob,
                        vs_glob,
                        &s.pos,
                        &s.sel,
                        &s.part_ml,
                        &s.part_o,
                        &s.attn_out,
                        k,
                        self.sessions,
                        self.max_ctx,
                        self.kv_stride,
                        g_splits,
                    )?;
                }
            }
            if li == 0 {
                mark(if is_glob { "l0_global" } else { "l0_attn" })?;
            } else if is_glob {
                mark(GLOB_MARKS[li])?;
            }
            if let Some(dir) = dump_dir {
                let slice = s.attn_out.slice(0..rows * NH * HD)?;
                let mut host = vec![0u8; rows * NH * HD * 2];
                self.st.dtoh(&mut host, &slice.cast::<u8>()?)?;
                let _ = fs::write(dir.join(format!("l{li}_attn_out.bin")), &host);
            }

            // Output projection + fused conv finish + residual + post_attention_layernorm
            self.gemm_bf16(H, NH * HD, PlanClass::Decode, &s.attn_out, &ly.o_proj, &s.sub, rows)?;
            if li == 0 {
                mark("l0_oproj")?;
            }
            dev.grouped_conv_out_res_norm(
                &s.sub,
                &s.dyn_buf,
                &ly.attn_conv_base,
                &s.x,
                Some(&ly.post_norm),
                Some(&s.h),
                None,
                k,
                false,
            )?;
            if li == 0 {
                mark("l0_aout")?;
            }

            // MLP conv prepare
            self.gemm_bf16(
                CONV_PROJ_N,
                H,
                PlanClass::Decode,
                &s.h,
                &ly.mlp_conv_proj,
                &s.dyn_buf,
                rows,
            )?;
            dev.grouped_conv_in(&s.h, &s.dyn_buf, &ly.mlp_conv_base, &s.h_c, k)?;
            if li == 0 {
                mark("l0_mconv")?;
            }

            // Gate+Up -> SiLU*mul -> Down
            self.gemm_bf16(
                2 * INTER,
                H,
                PlanClass::Decode,
                &s.h_c,
                &ly.gate_up_proj,
                &s.gate_up,
                rows,
            )?;
            if li == 0 {
                mark("l0_gu")?;
            }
            dev.silu_mul(&s.gate_up, &s.mlp_act, rows)?;
            self.gemm_bf16(H, INTER, PlanClass::Decode, &s.mlp_act, &ly.down_proj, &s.sub, rows)?;
            if li == 0 {
                mark("l0_down")?;
            }

            // Fused MLP conv finish + residual + next layer's input_layernorm (or final norm on rows 1..7)
            if li + 1 < NLAYERS {
                dev.grouped_conv_out_res_norm(
                    &s.sub,
                    &s.dyn_buf,
                    &ly.mlp_conv_base,
                    &s.x,
                    Some(&w.layers[li + 1].in_norm),
                    Some(&s.h),
                    None,
                    k,
                    false,
                )?;
            } else {
                let f16_dst = w.head_i8.is_some().then_some(&s.pred_h_f16);
                dev.grouped_conv_out_res_norm(
                    &s.sub,
                    &s.dyn_buf,
                    &ly.mlp_conv_base,
                    &s.x,
                    Some(&w.norm),
                    Some(&s.pred_h),
                    f16_dst,
                    k,
                    true,
                )?;
            }
            if li == 0 {
                mark("l0_mout")?;
            }
        }
        mark("layers_1_4")?;

        // 3. LM Head (W8A16 gemm_frag2 for M <= 64, else BF16 cuBLASLt) -> 2-pass exact top-16 -> 16-warp selector
        if let Some((ref w_q8f, ref scale)) = w.head_i8
            && drows <= 64
        {
            dev.head_gemm_i8(w_q8f, scale, &s.pred_h_f16, &s.logits, drows, HEAD_N)?;
        } else {
            self.gemm_f32(HEAD_N, H, PlanClass::Decode, &s.pred_h, &w.head, &s.logits, drows)?;
        }
        mark("lm_head")?;
        dev.topk16(
            &s.logits,
            HEAD_N,
            &s.chunk_max,
            &s.chunk_z,
            &s.cand_count,
            &s.samp_temp,
            &s.sel,
            &s.s1_vals,
            &s.s1_ids,
            &s.unary,
            &s.cand_ids,
            &s.row_z,
            drows,
        )?;
        mark("topk16")?;
        self.gemm_f32(SEL_RANK, H, PlanClass::Decode, &s.pred_h, &w.sel_proj, &s.hp, drows)?;
        mark("sel_proj")?;
        dev.selector_walk(
            &s.cand_ids,
            &s.unary,
            &s.row_z,
            &s.hp,
            &w.pred_cb,
            &w.succ_cb,
            &s.anchor,
            &s.pos,
            &s.sel,
            &s.samp_temp,
            &s.samp_top_p,
            &s.samp_seed,
            &s.drafts,
            &s.prop_ids,
            &s.prop_w,
            &s.conf,
            self.conf_mode,
            k,
            self.sessions,
        )?;
        mark("sel_walk")?;
        if let Some(dir) = dump_dir {
            let l_sl = s.logits.slice(0..drows * HEAD_N)?;
            let mut l_host = vec![0u8; drows * HEAD_N * 4];
            self.st.dtoh(&mut l_host, &l_sl.cast::<u8>()?)?;
            let _ = fs::write(dir.join("logits.bin"), &l_host);
            let c_sl = s.cand_ids.slice(0..drows * TOP_K)?;
            let mut c_host = vec![0u8; drows * TOP_K * 4];
            self.st.dtoh(&mut c_host, &c_sl.cast::<u8>()?)?;
            let _ = fs::write(dir.join("cand_ids.bin"), &c_host);
            let d_sl = s.drafts.slice(0..self.sessions * DRAFTS)?;
            let mut d_host = vec![0u8; self.sessions * DRAFTS * 4];
            self.st.dtoh(&mut d_host, &d_sl.cast::<u8>()?)?;
            let _ = fs::write(dir.join("drafts.bin"), &d_host);
        }
        if prof {
            self.st.synchronize()?;
            eprint!("[df2 prof k={k}]");
            for pair in evs.windows(2) {
                let dt = pair[0].1.elapsed_ms(&pair[1].1)?;
                eprint!(" {}={:.3}", pair[1].0, dt);
            }
            let tot = evs.first().unwrap().1.elapsed_ms(&evs.last().unwrap().1)?;
            eprintln!(" TOTAL={tot:.3} ms");
        }
        Ok(())
    }

    /// Runs one eager (non-graphed) draft batch and dumps per-layer `attn_out`, `logits`, `cand_ids`, and `drafts`
    /// to `dump_dir` for numerical oracle parity checks.
    ///
    /// # Errors
    ///
    /// [`RuntimeError`] on invalid session indices or CUDA failure.
    pub fn draft_with_dump(&mut self, sessions: &[usize], dump_dir: &Path) -> Res<()> {
        let n = self.sessions;
        let k = sessions.len();
        if k == 0 || k > n {
            return Err(RuntimeError::request(format!("draft_with_dump: 1..{n} sessions")));
        }
        let sel: Vec<i32> = sessions
            .iter()
            .map(|&r| i32::try_from(r).expect("session fits i32"))
            .collect();
        rt(self.st.htod(&rt(self.s.sel.slice(0..k))?, &sel))?;
        rt(self.draft_batch(k, false, Some(dump_dir)))
    }

    fn run_batch(&mut self, sessions: &[usize]) -> Res<()> {
        let n = self.sessions;
        let k = sessions.len();
        if k == 0 || k > n {
            return Err(RuntimeError::request(format!("draft_sessions: 1..{n} sessions")));
        }
        let mut seen = vec![false; n];
        let mut sel = Vec::with_capacity(k);
        for &r in sessions {
            if r >= n || seen[r] {
                return Err(RuntimeError::request(format!(
                    "draft_sessions: sessions must be distinct, 0..{n}"
                )));
            }
            seen[r] = true;
            sel.push(i32::try_from(r).expect("session fits i32"));
        }
        rt(self.st.htod(&rt(self.s.sel.slice(0..k))?, &sel))?;
        if self.graphs[k - 1].is_none() {
            // Eager warmup (populates cuBLASLt plan/checked caches before capture)
            let prof = std::env::var("INK_DF2_PROF").is_ok_and(|v| v != "0");
            rt(self.draft_batch(k, false, None))?;
            if prof {
                rt(self.draft_batch(k, true, None))?;
            }
            rt(self.st.synchronize())?;
            let free0 = rt(self.ctx.mem_info())?.0;
            let cap = rt(self.st.begin_capture())?;
            let g = rt(self.draft_batch(k, false, None))
                .and_then(|()| rt(cap.end()))
                .map_err(RuntimeError::into_device)?;
            rt(g.launch())?;
            rt(self.st.synchronize())?;
            let free1 = rt(self.ctx.mem_info())?.0;
            let took = free0.saturating_sub(free1);
            self.graph_bytes.set(self.graph_bytes.get().saturating_add(took));
            self.graphs[k - 1] = Some(g);
            return Ok(());
        }
        rt(self.graphs[k - 1].as_ref().expect("captured above").launch())
    }
}

impl Drafter for RustDFlash2 {
    #[allow(unsafe_code)] // implements the trait's bounds-contract method
    unsafe fn commit_rows(&mut self, feats: ApiBuf, pos: ApiBuf, sess: ApiBuf, n: usize) -> Res<()> {
        if n == 0 {
            return Ok(());
        }
        let b = self.bufs.borrow();
        let f = b.typed::<Bf16>(feats, n * FEAT_ROW_BYTES / 2, "df2 commit feats")?;
        let p = b.typed::<i32>(pos, n, "df2 commit pos")?;
        let q = b.typed::<i32>(sess, n, "df2 commit sess")?;
        rt(self.commit_impl(&f, &p, &q, n))
    }

    fn frnn(&mut self, feats: ApiBuf) -> Res<()> {
        // Validate buffer handle for contract parity; DFlash2 has no Markov RNN state.
        let b = self.bufs.borrow();
        let _ = b.typed::<Bf16>(feats, self.sessions * FEAT_ROW_BYTES / 2, "df2 frnn feats")?;
        Ok(())
    }

    #[allow(unsafe_code)] // implements the trait's bounds-contract method
    unsafe fn set_state(&mut self, anchor: ApiBuf, pos: ApiBuf) -> Res<()> {
        let b = self.bufs.borrow();
        let a = b.typed::<i32>(anchor, self.sessions, "df2 state anchor")?;
        let p = b.typed::<i32>(pos, self.sessions, "df2 state pos")?;
        rt(self.st.dtod(&self.s.anchor, &a))?;
        rt(self.st.dtod(&self.s.pos, &p))
    }

    fn draft(&mut self) -> Res<()> {
        let all: Vec<usize> = (0..self.sessions).collect();
        self.run_batch(&all)
    }

    fn draft_sessions(&mut self, sessions: &[usize]) -> Res<()> {
        self.run_batch(sessions)
    }

    fn drafts(&self) -> ApiBuf {
        self.drafts_api
    }

    fn conf(&self) -> Option<ApiBuf> {
        self.conf_api
    }

    fn default_verify_tau(&self) -> f32 {
        // Calibrated for parallel mask-slot confidence (M1 sweep: tau 0.25 peaks at +15 tok/s over w8).
        0.25
    }

    fn is_swa_only(&self) -> bool {
        !self.cfg.has_global()
    }

    fn set_sampling(&mut self, temps: &[f32], top_ps: &[f32], seeds: &[u64]) -> Res<()> {
        if temps.len() != self.sessions || top_ps.len() != self.sessions || seeds.len() != self.sessions {
            return Err(RuntimeError::request(format!(
                "df2 set_sampling: expected {} sessions, got {}/{}/{}",
                self.sessions,
                temps.len(),
                top_ps.len(),
                seeds.len()
            )));
        }
        let eff_temps: Vec<f32> = temps
            .iter()
            .map(|&t| if t >= 1e-5 { t * self.draft_temp_scale } else { t })
            .collect();
        rt(self.st.htod(&self.s.samp_temp, &eff_temps))?;
        if let Some(p) = self.draft_top_p {
            let eff_top_ps: Vec<f32> = temps.iter().map(|&t| if t >= 1e-5 { p } else { 1.0 }).collect();
            rt(self.st.htod(&self.s.samp_top_p, &eff_top_ps))?;
        } else if self.use_top_p {
            rt(self.st.htod(&self.s.samp_top_p, top_ps))?;
        }
        rt(self.st.htod(&self.s.samp_seed, seeds))
    }

    fn proposal(&self) -> Option<ProposalBufs> {
        self.prop_api
    }
}
