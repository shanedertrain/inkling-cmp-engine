// Ported from engine/native/src/engine.cu @ 1c7ac46 (Engine::Engine, forward, commit, reset_slot, argmax,
// logits_host) and engine/native/capi/dumps.cpp (launch-trace header). Own code of this project.
//! The Rust forward engine (packages R5, R6 of docs/RUST-PLAN.md): the C++ `ink::Engine` forward pass over the
//! kernel pack, bit for bit, with decode calls captured and replayed as CUDA graphs as the C++ engine does
//! (`graphs`); device-side speculation (`specrt`, package R9) with the Rust drafter on the engine's context and
//! stream.
//!
//! Captured graphs stay within `EngineConfig::graph_mem` (`graphs`); the engine refuses to start when its allocations
//! (and later its drafter's) leave less than that plus [`HEADROOM`] free, so the KV page count it was given cannot
//! run the card out of memory later.
//!
//! Field order is drop order: the captured graphs first, then the cuBLASLt handle, the activation and weight buffers
//! and the kernels drop before the arena they point into, the stream before its context.
//! Every owner that frees device resources synchronises the context first (ink-cuda, review R3 item 8).

#[cfg(feature = "fault-inject")]
mod fault;
mod forward;
mod graphs;
mod meta;
mod ops;
mod resume;
mod specrt;

use std::fs;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use ink_cuda::{AlgoPins, Arena, Bf16, Capture, Context, DevBuf, Dtype, Event, GemmShape, LtGemm, PlanClass, Stream};
use ink_kernels::{KernelId, Kernels, glue};
use ink_model::dims::{
    DREL, GATE_RANK, H, KD, NEXP, NH, NVISITS, QD, REL_EXT, VOCAB, VOCAB_PAD, Visit, is_full_layer, visit_plan,
};
use ink_model::dump::{arena_map_text, weight_sums, weight_sums_text};
use ink_model::plan::FUSED_COLS;
use ink_model::{Activations, GateTailW, MemConfig, Weights};
use ink_record::Source;
use ink_rt_api::{
    DeviceSampling, Dump, EngineConfig, Glue, Graphs, LOGIT_ROW, Runtime, RuntimeError, RuntimeStats, SamplerProbe,
    SegIn, SlotCheckpoints, SpecRuntime, TapFn,
};

use graphs::GraphCache;
use meta::{CallMeta, Limits, SegReq, Slots, build_call};

use self::forward::Fwd;
use self::ops::{
    DA_SMEM, DA_SMEM16, Dev, MOE_DEC_SMEM_MXFP8, MOE_DEC_SMEM_N1S1, MOE_SMEM_MXFP8, MOE_SMEM_NVFP4,
    MOE_SMEM_NVFP4_N1S1, PFA_FP8_SMEM, PFA_SMEM, SLD, da_ring_smem,
};
use crate::error::Classify;

type Res<T> = Result<T, RuntimeError>;

/// Device memory left free beyond the graph reserve at start-up, for what the driver allocates later outside the
/// arena and the graphs (measured: `bench/raw/2026-11-23-graph-cache/`).
pub const HEADROOM: u64 = 256 << 20;

/// Bytes of one global KV page (K and V of every full-attention layer visit) in bf16.
const PAGE_BYTES_BF16: u64 = (2 * ink_model::plan::FULL_VISITS * ink_model::plan::PAGE_ELEMS * 2) as u64;

fn page_bytes(kv_fp8: bool) -> u64 {
    if kv_fp8 { PAGE_BYTES_BF16 / 2 } else { PAGE_BYTES_BF16 }
}

/// Sets the decode-attention nodes of a held graph to key chunks `chunks` (global, sliding window) and row bound `r`
/// of a call of `n` decode segments, as `Fwd::attention` launches them: `max_chunks` and `R` in every attention and
/// combine node, on the attention kernels the workers per session and head (grid x), on the combine kernels the
/// `3 R` row blocks (grid x). `r` is of the graph's kernel variant (its key), so function and block stay. The graph
/// then replays what the eager call launches.
fn retarget(
    e: &mut graphs::Entry,
    chunks: (usize, usize),
    r: usize,
    n: usize,
    ring_slots: &[usize; 17],
    opart: usize,
) -> Res<()> {
    for &i in &e.da_nodes {
        let mut p: ink_kernels::da::Params = e.graph.kernel_param(i).rt()?;
        let full = p.window >= 1 << 30;
        let gx = if full { chunks.0 } else { chunks.1 };
        if n * 8 * gx * ops::DA_G * r * 128 > opart {
            return Err(RuntimeError::request("decode partials"));
        }
        if meta::da_variant(usize::try_from(p.R).rt()?) != meta::da_variant(r) {
            return Err(RuntimeError::request(format!(
                "graph node {i}: R {} and {r} are of different kernel variants",
                p.R
            )));
        }
        p.max_chunks = i32::try_from(gx).rt()?;
        p.R = i32::try_from(r).rt()?;
        let node = &e.graph.nodes()[i];
        let Some((mut grid, _, _)) = node.dims else {
            return Err(RuntimeError::request(format!(
                "graph node {i} has no launch dimensions"
            )));
        };
        if node
            .kernel
            .as_deref()
            .is_some_and(|k| k.starts_with("_ZN2da14combine_kernel"))
        {
            grid[0] = u32::try_from(ops::DA_G * r).rt()?;
        } else {
            let w = if full {
                meta::pick_workers_warp_n(n, gx, meta::SMS, if r > 8 { 2.0 } else { 4.0 })
            } else {
                meta::pick_workers(n, gx, ring_slots[r])
            };
            grid[0] = u32::try_from(w).rt()?;
        }
        e.graph.set_kernel_param(i, grid, &p).rt()?;
    }
    e.chunks = chunks;
    e.r = r;
    Ok(())
}

/// Refuses a configuration whose allocations left less than `need` bytes of device memory free.
fn check_free(ctx: &Context, need: u64, kv_fp8: bool, what: &str) -> Res<()> {
    let free = ctx.mem_info().rt()?.0 as u64;
    if free >= need {
        return Ok(());
    }
    Err(RuntimeError::request(format!(
        "{what} leaves {:.2} GiB of device memory free; the graph reserve and headroom need {:.2} GiB: use {} fewer KV \
         pages or a smaller graph reserve",
        free as f64 / 1_073_741_824.0,
        need as f64 / 1_073_741_824.0,
        (need - free).div_ceil(page_bytes(kv_fp8))
    )))
}

/// Error conversion into the runtime's error type, keeping the error's kind (`crate::error`).
trait Rt<T> {
    fn rt(self) -> Res<T>;
}

impl<T, E: Classify> Rt<T> for Result<T, E> {
    fn rt(self) -> Res<T> {
        crate::error::rt(self)
    }
}

/// Where the last launch trace comes from.
#[derive(Debug)]
enum TraceSrc {
    /// An armed eager call: its key and node lines.
    Eager(String, Vec<String>),
    /// The graph of `key`, newly captured (the C ABI's `last_graph_key`), and the segments (`m,nlog;...`) of the call
    /// it was captured from (the C++ key of its launch trace).
    Graph { key: String, segs: String },
}

/// The Rust forward engine; implements [`Runtime`].
pub struct Engine {
    graphs: GraphCache,
    /// Device bytes of the graph reserve (forward graphs and the drafter's).
    graph_mem: usize,
    lt: LtGemm,
    /// `xk` and `xv` as one buffer (D1's k / v batch).
    xkv: DevBuf<Bf16>,
    /// Resume checkpoints per slot, with their own arena (`resume`).
    ckpt: resume::Checkpoints,
    a: Activations<Arena>,
    /// The tensor-core gate tail's bf16 controller weights (G7), taken after the activations as the C++ `tc_w_`.
    tc: GateTailW<Arena>,
    /// Transposed per-layer relative projections (`[1024][16]` bf16) for in-kernel TC bias in `prefill_attn`.
    rel_proj_t: Vec<DevBuf<Bf16>>,
    w: Weights,
    k: Kernels,
    arena: Arena,
    st: Rc<Stream>,
    ctx: Rc<Context>,
    /// Device-side speculation (package R9): device buffers, pending device token ids, the drafter flag.
    spec: specrt::SpecState,
    mem: MemConfig,
    plan: Vec<Visit>,
    full_index: Vec<Option<usize>>,
    ring_index: Vec<Option<usize>>,
    da_ring_slots: [usize; 17],
    slots: Slots,
    last: Option<CallMeta>,
    committed: bool,
    glue: Glue,
    tap: Option<TapFn>,
    trace_armed: bool,
    last_trace: Option<TraceSrc>,
    profile: bool,
    /// With `profile`: the sub-component marks too (the C++ `profile_fine`).
    profile_fine: bool,
    marks: Vec<(String, Event)>,
    graphs_on: bool,
    /// The served fail-fast test's device fault (`fault.rs`).
    #[cfg(feature = "fault-inject")]
    fault: fault::Fault,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("mem", &self.mem)
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}

/// The 25 dense GEMM shapes the C++ constructor plans (and the pin file holds): 11 per plan class, and the three
/// decode-class shapes of the fused input projections (D1: N = 8600, 6552, 1056; the fourth, 1024, is planned
/// anyway). The C++ gives those four the algorithm of N = 8600 (`LtGemm::adopt`) before the pin file overrides them.
pub fn planned_shapes() -> Vec<GemmShape> {
    let mut out = Vec::new();
    for class in [PlanClass::Decode, PlanClass::Mixed] {
        let s = |n: usize, k: usize, ldw: usize, output: Dtype| GemmShape {
            n: n as u32,
            k: k as u32,
            ldw: ldw as u32,
            input: if output == Dtype::F32 && k == GATE_RANK {
                Dtype::F32
            } else {
                Dtype::Bf16
            },
            output,
            class,
        };
        out.extend([
            s(QD, H, H, Dtype::Bf16),
            s(NH * DREL, H, H, Dtype::Bf16),
            s(KD, H, H, Dtype::Bf16),
            s(NH, H, H, Dtype::Bf16),
            s(32, H, H, Dtype::Bf16),
            s(H, QD, QD, Dtype::Bf16),
            s(GATE_RANK, 2 * H, 2 * H, Dtype::Bf16),
            s(H, GATE_RANK, GATE_RANK, Dtype::F32), // fp32 in and out: the reference path's gate GEMMs
            s(NEXP, 1024, H, Dtype::F32),
            s(NEXP, H - 2048, H, Dtype::F32),
            s(VOCAB_PAD, H, H, Dtype::F32),
        ]);
    }
    for n in [FUSED_COLS - 64, 2 * QD + NH * DREL + NH, KD + 32] {
        out.push(GemmShape {
            n: n as u32,
            k: H as u32,
            ldw: H as u32,
            input: Dtype::Bf16,
            output: Dtype::Bf16,
            class: PlanClass::Decode,
        });
    }
    out
}

impl Engine {
    /// Loads the model and plans the memory, as `ink::Engine::Engine`. Refuses a missing pin file (the algorithms
    /// must be pinned).
    ///
    /// # Errors
    ///
    /// [`RuntimeError`] for a refused configuration or any load or device failure.
    pub fn new(config: &EngineConfig) -> Res<Engine> {
        if config.max_chunk == 0 || config.max_chunk > config.t_max {
            return Err(RuntimeError::request("max_chunk"));
        }
        let pins_text = fs::read_to_string(&config.gemm_algos).map_err(|e| {
            RuntimeError::request(format!(
                "pin file {}: {e} (the Rust engine needs pinned algorithms; ink-tune writes them)",
                config.gemm_algos.display()
            ))
        })?;
        let mut pins = AlgoPins::from_json(&pins_text).rt()?;
        let shapes = planned_shapes();
        if pins.len() != shapes.len() {
            return Err(RuntimeError::request(format!(
                "pin file has {} shapes, the engine plans {}",
                pins.len(),
                shapes.len()
            )));
        }
        // Fused prefill projections reuse the pinned Mixed QD x H (for N = 8600, 6552) and KD x H (for N = 1056)
        // bf16 algorithms, so gemm-algos.json stays unchanged and large-N shapes keep their tuned tile.
        let mixed_q_algo = *pins
            .get(&GemmShape {
                n: QD as u32,
                k: H as u32,
                ldw: H as u32,
                input: Dtype::Bf16,
                output: Dtype::Bf16,
                class: PlanClass::Mixed,
            })
            .ok_or_else(|| RuntimeError::request("pin file missing mixed QD x H bf16 shape"))?;
        let mixed_k_algo = *pins
            .get(&GemmShape {
                n: KD as u32,
                k: H as u32,
                ldw: H as u32,
                input: Dtype::Bf16,
                output: Dtype::Bf16,
                class: PlanClass::Mixed,
            })
            .ok_or_else(|| RuntimeError::request("pin file missing mixed KD x H bf16 shape"))?;
        let mut mixed_fused_shapes = Vec::with_capacity(3);
        for (n, algo) in [
            (FUSED_COLS - 64, mixed_q_algo),
            (2 * QD + NH * DREL + NH, mixed_q_algo),
            (KD + 32, mixed_k_algo),
        ] {
            let s = GemmShape {
                n: n as u32,
                k: H as u32,
                ldw: H as u32,
                input: Dtype::Bf16,
                output: Dtype::Bf16,
                class: PlanClass::Mixed,
            };
            pins.insert(s, algo).rt()?;
            mixed_fused_shapes.push(s);
        }
        let plan = visit_plan();
        let (mut nf, mut nr) = (0, 0);
        let mut full_index = Vec::new();
        let mut ring_index = Vec::new();
        for v in &plan {
            if is_full_layer(v.layer) {
                full_index.push(Some(nf));
                ring_index.push(None);
                nf += 1;
            } else {
                full_index.push(None);
                ring_index.push(Some(nr));
                nr += 1;
            }
        }
        let mem = MemConfig {
            max_slots: config.max_slots as usize,
            num_pages: config.num_pages as usize,
            t_max: config.t_max as usize,
            max_chunk: config.max_chunk as usize,
            logit_rows_max: config.logit_rows_max as usize,
            kv_fp8: config.kv_fp8,
        };
        // shared with the drafter this engine creates (one context, one stream: specrt.rs)
        let ctx = Rc::new(Context::new(0).rt()?);
        let st = Rc::new(ctx.new_stream().rt()?);
        let mut arena = Arena::new(&ctx, &st, mem.arena_bytes()).rt()?;
        eprintln!(
            "[engine] arena {:.2} GiB (global_kv={})",
            mem.arena_bytes() as f64 / 1_073_741_824.0,
            if mem.kv_fp8 { "e4m3" } else { "bf16" }
        );
        let w = Weights::load(&mut arena, &st, &config.model_dir, config.load_threads as usize).rt()?;
        let a = Activations::alloc(&mut arena, &mem).rt()?;
        let looped: Vec<bool> = w.layers.iter().map(|l| l.controllers.is_some()).collect();
        let tc = GateTailW::alloc(&mut arena, &looped).rt()?;
        let mut rel_proj_t = Vec::with_capacity(w.layers.len());
        for i in 0..w.layers.len() {
            rel_proj_t.push(arena.take::<Bf16>(REL_EXT * DREL, &format!("rel_proj_t[{i}]")).rt()?);
        }
        eprintln!(
            "[engine] arena used {:.2} GiB (weights {:.2} GB)",
            arena.plan().used() as f64 / 1_073_741_824.0,
            w.bytes as f64 / 1e9
        );
        let k = Kernels::load(&ctx).rt()?;
        // kernel attributes the C++ sets (engine.cu: moe_attr_dev, da::prepare, da::ring_prepare; pfa::launch's
        // static attribute)
        for (id, smem) in [
            (KernelId::PmoeMoeGemm220380, MOE_SMEM_NVFP4),
            (KernelId::PmoeMoeGemm222380, MOE_SMEM_NVFP4),
            (KernelId::PmoeMoeGemm230380, MOE_SMEM_NVFP4_N1S1),
            (KernelId::PmoeMoeGemm232380, MOE_SMEM_NVFP4_N1S1),
            (KernelId::PmoeMoeGemm210380, MOE_SMEM_MXFP8),
            (KernelId::PmoeMoeGemm212380, MOE_SMEM_MXFP8),
            (KernelId::PmoeMoeGemmDec3054, MOE_DEC_SMEM_N1S1),
            (KernelId::PmoeMoeGemmDec3254, MOE_DEC_SMEM_N1S1),
            (KernelId::PmoeMoeGemmDec1054, MOE_DEC_SMEM_MXFP8),
            (KernelId::PmoeMoeGemmDec1254, MOE_DEC_SMEM_MXFP8),
            (KernelId::DaDecodeAttnKernel0False, DA_SMEM),
            (KernelId::DaDecodeAttnKernel0True, DA_SMEM),
            (KernelId::DaDecodeAttnKernel160, DA_SMEM16),
            (KernelId::DaDecodeAttnKernel161, DA_SMEM16),
            (KernelId::DaDecodeAttnKernel1False, DA_SMEM),
            (KernelId::DaDecodeAttnKernel1True, DA_SMEM),
            (KernelId::DaDecodeAttnRingKernel013, da_ring_smem(1)),
            (KernelId::DaDecodeAttnRingKernel023, da_ring_smem(2)),
            (KernelId::DaDecodeAttnRingKernel033, da_ring_smem(3)),
            (KernelId::DaDecodeAttnRingKernel043, da_ring_smem(4)),
            (KernelId::DaDecodeAttnRingKernel053, da_ring_smem(5)),
            (KernelId::DaDecodeAttnRingKernel063, da_ring_smem(6)),
            (KernelId::PfaPrefillAttnKernel42112False, PFA_SMEM),
            (KernelId::PfaPrefillAttnFp8Kernel42112False, PFA_FP8_SMEM),
        ] {
            k.function(id).set_max_dynamic_smem(smem).rt()?;
        }
        // the tensor-core gate tail's weights from the fp32 transposed controllers (engine.cu: mg_tc_weights_kernel)
        let dev = Dev { k: &k, st: &st };
        for (l, tw) in w.layers.iter().zip(&tc.w) {
            if let (Some(cs), Some(tw)) = (&l.controllers, tw) {
                for (c, t) in cs.iter().zip(tw) {
                    dev.mg_tc_weights(&c.forget_t, &c.gain_t, t).rt()?;
                }
            }
        }
        for (l, rt) in w.layers.iter().zip(&rel_proj_t) {
            dev.rel_proj_t(&l.rel_proj, rt).rt()?;
        }
        let mut da_ring_slots = [0usize; 17];
        for (r, slot) in da_ring_slots.iter_mut().enumerate().skip(1) {
            let nw = (3 * r).div_ceil(8);
            let id = match nw {
                1 => KernelId::DaDecodeAttnRingKernel013,
                2 => KernelId::DaDecodeAttnRingKernel023,
                3 => KernelId::DaDecodeAttnRingKernel033,
                4 => KernelId::DaDecodeAttnRingKernel043,
                5 => KernelId::DaDecodeAttnRingKernel053,
                _ => KernelId::DaDecodeAttnRingKernel063,
            };
            let occ = k
                .function(id)
                .max_active_blocks_per_sm((nw * 32) as u32, da_ring_smem(nw))
                .rt()?;
            *slot = 74 * usize::try_from(occ.max(1)).unwrap_or(1);
        }
        let mut lt = LtGemm::new(&st, pins).rt()?;
        lt.set_workspace(a.lt_ws.clone());
        for s in shapes.into_iter().chain(mixed_fused_shapes) {
            lt.pinned_opaque(s).rt()?;
        }
        // xk, xv adjacent: the k / v batch of the decode projections reads them as one strided buffer (D1)
        let xs = mem.t_max * H * 2;
        if a.xv.offset() != a.xk.offset() + xs {
            return Err(RuntimeError::request("k / v inputs not adjacent"));
        }
        let xkv = arena
            .span(a.xk.offset()..a.xk.offset() + 2 * xs)
            .rt()?
            .cast::<Bf16>()
            .rt()?;
        let ckpt = resume::Checkpoints::new(&ctx, &st, mem.max_slots)?;
        eprintln!(
            "[engine] checkpoints {:.2} GiB",
            resume::Checkpoints::bytes(mem.max_slots) as f64 / 1_073_741_824.0
        );
        st.synchronize().rt()?;
        let graph_mem = usize::try_from(config.graph_mem).rt()?;
        check_free(&ctx, config.graph_mem + HEADROOM, config.kv_fp8, "the engine")?;
        Ok(Engine {
            graphs: GraphCache::new(graph_mem),
            graph_mem,
            lt,
            xkv,
            ckpt,
            a,
            tc,
            rel_proj_t,
            w,
            k,
            arena,
            st,
            ctx,
            spec: specrt::SpecState::default(),
            slots: Slots::new(mem.max_slots, mem.num_pages),
            mem,
            plan,
            full_index,
            ring_index,
            da_ring_slots,
            last: None,
            committed: true,
            glue: Glue::Production,
            tap: None,
            trace_armed: false,
            last_trace: None,
            profile: false,
            profile_fine: false,
            marks: Vec::new(),
            graphs_on: config.graphs == Graphs::On,
            #[cfg(feature = "fault-inject")]
            fault: fault::Fault::from_env(),
        })
    }

    /// Profile marks (a timing event at every C++ `mark`) for the following calls; [`Runtime::marks`] reads them.
    /// Profiled calls run eagerly, as in the C++ engine.
    pub fn set_profile(&mut self, on: bool) {
        self.profile = on;
    }

    /// Nodes per region over every captured graph (the C++ `graph_region_counts`).
    pub fn graph_region_counts(&self) -> std::collections::BTreeMap<String, usize> {
        self.graphs.region_counts()
    }

    fn limits(&self) -> Limits {
        let m = &self.mem;
        Limits {
            max_chunk: m.max_chunk,
            t_max: m.t_max,
            logit_rows_max: m.logit_rows_max,
            ring: m.ring_pages(),
            pa_items_cap: m.pa_items_cap(),
            pa_tiles_cap: m.pa_tiles_cap(),
            pa_partials_cap: self.a.pa_ws_floats / ink_model::plan::PA_WS_FLOATS_PER_PARTIAL,
        }
    }

    /// The host-to-device copies of `Engine::forward`.
    fn upload(&self, m: &CallMeta) -> Res<()> {
        let (a, st) = (&self.a, &self.st);
        let (t, s) = (m.t, m.call.len());
        st.htod(&a.ids.slice(0..t).rt()?, &m.ids).rt()?;
        self.spec.copy_dev_ids(st, &a.ids, m)?;
        st.htod(&a.pos.slice(0..t).rt()?, &m.pos).rt()?;
        st.htod(&a.seg.slice(0..t).rt()?, &m.seg).rt()?;
        st.htod(&a.row0.slice(0..s).rt()?, &m.row0).rt()?;
        st.htod(&a.slot.slice(0..s).rt()?, &m.slot).rt()?;
        st.htod(&a.len.slice(0..s).rt()?, &m.len).rt()?;
        if !m.lrows.is_empty() {
            st.htod(&a.logit_rows.slice(0..m.lrows.len()).rt()?, &m.lrows).rt()?;
        }
        st.htod(&a.pt_g, &m.pt_g).rt()?;
        st.htod(&a.pt_r, &m.pt_r).rt()?;
        let nd = m.dctx.len();
        if nd > 0 {
            let ms = self.mem.max_slots;
            st.htod(&a.da_ctx.slice(0..nd).rt()?, &m.dctx).rt()?;
            st.htod(&a.da_row0.slice(0..nd).rt()?, &m.drow0).rt()?;
            st.htod(&a.da_cs.slice(0..nd).rt()?, &m.dcs_g).rt()?;
            st.htod(&a.da_cs.slice(ms..ms + nd).rt()?, &m.dcs_r).rt()?;
            st.htod(&a.da_nrows.slice(0..nd).rt()?, &m.dnr).rt()?;
        }
        let segs = a.pa_segs.cast::<[i32; 4]>().rt()?;
        st.htod(&segs.slice(0..s).rt()?, &m.pa_segs).rt()?;
        let (i1, i2) = (
            a.pa_items.cast::<[i32; 4]>().rt()?,
            a.pa_items2.cast::<[i32; 4]>().rt()?,
        );
        for kind in 0..2 {
            let n = m.pa_items[kind].len();
            if n > 0 {
                let off = kind * a.pa_items_cap;
                st.htod(&i1.slice(off..off + n).rt()?, &m.pa_items[kind]).rt()?;
                st.htod(&i2.slice(off..off + n).rt()?, &m.pa_items2[kind]).rt()?;
            }
        }
        Ok(())
    }

    fn check_slot(&self, slot: usize) -> Res<()> {
        if slot < self.slots.count() {
            Ok(())
        } else {
            Err(RuntimeError::request(format!("slot {slot} of {}", self.slots.count())))
        }
    }

    fn last_rows(&self) -> usize {
        self.last.as_ref().map_or(0, |m| m.lrows.len())
    }
}

impl Runtime for Engine {
    fn identity(&self) -> Source {
        Source::this_build()
    }

    fn reset_slot(&mut self, slot: usize) -> Res<()> {
        self.check_slot(slot)?;
        // a commit after the reset would advance the emptied slot over rows whose K/V went with its pages
        if !self.committed
            && let Some(m) = &self.last
            && m.call.iter().any(|c| c.slot == slot)
        {
            return Err(RuntimeError::request(format!(
                "reset_slot: slot {slot} is in the uncommitted last forward call (commit it first; 0 rows drops it)"
            )));
        }
        self.slots.reset(slot);
        self.ckpt.clear(slot);
        let n = NVISITS * 3 * SLD;
        self.st
            .memset_zero(&self.a.conv.slice(slot * n..(slot + 1) * n).rt()?)
            .rt()
    }

    fn length(&self, slot: usize) -> Res<usize> {
        self.check_slot(slot)?;
        Ok(self.slots.length(slot))
    }

    fn free_pages(&self) -> Res<usize> {
        Ok(self.slots.free_pages())
    }

    fn forward(&mut self, segs: &[SegIn<'_>]) -> Res<usize> {
        // a call that fails leaves nothing to commit, sample or accept (not the call before it)
        self.last = None;
        // before any page is taken: an id outside the vocabulary would make the embedding read out of bounds
        for s in segs {
            meta::check_ids(s.ids, VOCAB).rt()?;
        }
        let reqs: Vec<SegReq> = segs
            .iter()
            .map(|s| SegReq {
                slot: s.slot,
                m: s.ids.len(),
                logits: s.logits,
            })
            .collect();
        let lim = self.limits();
        let mut m = build_call(&mut self.slots, &reqs, &lim).rt()?;
        for (s, c) in segs.iter().zip(&m.call) {
            m.ids[c.row0..c.row0 + c.m].copy_from_slice(s.ids);
        }
        self.save_windows(&m.call)?;
        Engine::upload(self, &m)?; // the inherent method (SpecRuntime has an `upload` too)
        #[cfg(feature = "fault-inject")]
        self.fault.after_upload(&self.st, &self.a.ids)?;
        self.marks.clear();
        let glue = if self.glue == Glue::Reference { 1 } else { 6 };
        let key = meta::cache_key(glue, &m.call);
        // the C++ `graphable`: graphs on, no tap, no profile, decode segments only; an armed launch trace runs the
        // call eagerly (the trace of an eager call)
        let graphable =
            self.graphs_on && self.tap.is_none() && !self.profile && !self.trace_armed && meta::graphable(&m.call);
        // a held graph is replayed; a new key is captured while the reserve has room, else it runs eagerly
        self.graphs
            .set_reserve(self.graph_mem.saturating_sub(self.spec.draft_graph_bytes()));
        let graphed = graphable && (self.graphs.get(&key).is_some() || self.graphs.admits(&key));
        let trace = self.trace_armed;
        if trace {
            self.st.start_trace(&self.arena).rt()?;
        }
        let reference = self.glue == Glue::Reference || self.tap.is_some();
        let mut run = |capture: Option<&Capture<'_>>| {
            Fwd {
                dev: Dev {
                    k: &self.k,
                    st: &self.st,
                },
                st: &self.st,
                ctx: &self.ctx,
                lt: &mut self.lt,
                a: &self.a,
                tc: &self.tc,
                rel_proj_t: &self.rel_proj_t,
                w: &self.w,
                m: &m,
                mem: &self.mem,
                plan: &self.plan,
                full_index: &self.full_index,
                ring_index: &self.ring_index,
                da_ring_slots: &self.da_ring_slots,
                xkv: &self.xkv,
                reference,
                tap: self.tap.as_mut(),
                marks: self.profile.then_some(&mut self.marks),
                fine: self.profile && self.profile_fine,
                capture,
            }
            .run_model()
        };
        let chunks = (meta::decode_gx(&m.call, true), meta::decode_gx(&m.call, false));
        let r = meta::decode_r(&m.call);
        if graphed {
            if let Some(e) = self.graphs.entry_mut(&key) {
                if (e.chunks, e.r) != (chunks, r) {
                    retarget(e, chunks, r, m.call.len(), &self.da_ring_slots, self.a.da_opart_floats)?;
                }
                e.graph.launch().rt()?;
            } else {
                // capture at first use; a failure inside drops the capture, which ends it. Sticky CUDA device errors
                // propagate immediately (review V1: fail fast on a poisoned context); non-sticky capture refusals mark
                // the key refused so later calls run eagerly instead of retrying (review V8), and this call runs
                // eagerly. A successful capture is charged the free memory it took up to its first replay.
                let t0 = Instant::now();
                self.st.synchronize().rt()?;
                let free0 = self.ctx.mem_info().rt()?.0;
                let captured = (|| {
                    let cap = self.st.begin_capture().rt()?;
                    run(Some(&cap))?;
                    let mut graph = cap.end().rt()?;
                    self.graphs.ablate(&mut graph)?;
                    Ok::<_, RuntimeError>(graph)
                })();
                match captured {
                    Ok(graph) => {
                        graph.launch().rt()?;
                        self.st.synchronize().rt()?;
                        let free1 = self.ctx.mem_info().rt()?.0;
                        let bytes = free0.saturating_sub(free1);
                        let nodes = graph.nodes().len();
                        self.graphs.insert(key.clone(), graph, bytes, chunks, r);
                        tracing::debug!(
                            target: "ink_step",
                            phase = "graph",
                            event = "capture",
                            key = %key,
                            nodes,
                            bytes,
                            free = free1,
                            held = self.graphs.held(),
                            live = self.graphs.live(),
                            captures = self.graphs.captures(),
                            refused = self.graphs.refused(),
                            capture_us = u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX),
                        );
                        self.last_trace = Some(TraceSrc::Graph {
                            key: key.clone(),
                            segs: meta::graph_key(glue, &m.call)
                                .split_once(':')
                                .map_or_else(String::new, |k| k.1.to_owned()),
                        });
                    }
                    Err(err) if err.kind() == ink_rt_api::ErrorKind::Device => return Err(err),
                    Err(err) => {
                        self.graphs.refuse(&key);
                        tracing::warn!(
                            key = %key,
                            error = %err,
                            refused = self.graphs.refused(),
                            "graph capture failed; running eagerly"
                        );
                        run(None)?;
                    }
                }
            }
        } else {
            let res = run(None);
            if trace {
                let lines = self.st.finish_trace().unwrap_or_default();
                self.last_trace = Some(TraceSrc::Eager(meta::graph_key(glue, &m.call), lines));
            }
            res?;
        }
        let n = m.lrows.len();
        self.last = Some(m);
        self.committed = false;
        Ok(n)
    }

    fn commit(&mut self, kept: &[u32]) -> Res<()> {
        let Some(m) = self.last.as_ref() else {
            return Err(RuntimeError::request("commit without a forward call"));
        };
        if self.committed {
            return Err(RuntimeError::request("the last forward call was committed already"));
        }
        let s = m.call.len();
        if kept.len() != s {
            return Err(RuntimeError::request("commit counts"));
        }
        let mut cnt = Vec::with_capacity(s);
        for (k, c) in kept.iter().zip(&m.call) {
            if *k as usize > c.m {
                return Err(RuntimeError::request("commit count range"));
            }
            cnt.push(*k as i32);
        }
        let a = &self.a;
        self.st.htod(&a.cnt.slice(0..s).rt()?, &cnt).rt()?;
        let tm = self.mem.t_max;
        let args = glue::CommitArgs {
            convx: a.convx.ptr(),
            convx_kstride: (tm * H) as i64,
            convx_vstride: (4 * tm * H) as i64,
            state: a.conv.ptr(),
            st_slot_stride: (NVISITS * 3 * SLD) as i64,
            st_visit_stride: (3 * SLD) as i64,
            row0: a.row0.ptr(),
            slot: a.slot.ptr(),
            cnt: a.cnt.ptr(),
            nseg: s as i32,
            nvisits: NVISITS as i32,
        };
        Dev {
            k: &self.k,
            st: &self.st,
        }
        .conv_commit(s, &args)
        .rt()?;
        let calls: Vec<(usize, usize)> = m.call.iter().zip(kept).map(|(c, &k)| (c.slot, k as usize)).collect();
        for (slot, k) in calls {
            self.slots.advance(slot, k);
        }
        self.committed = true;
        Ok(())
    }

    fn logits_host(&self, rows: usize, out: &mut Vec<f32>) -> Res<()> {
        if rows > self.last_rows() {
            return Err(RuntimeError::request(format!(
                "{rows} logit rows requested, the last call returned {}",
                self.last_rows()
            )));
        }
        out.clear();
        out.resize(rows * LOGIT_ROW, 0.0);
        self.st.dtoh(out, &self.a.logits.slice(0..rows * VOCAB_PAD).rt()?).rt()
    }

    fn samples_on_device(&self) -> bool {
        true
    }

    fn sample_host(&mut self, params: &[DeviceSampling], rows: usize, out: &mut Vec<i32>) -> Res<()> {
        let nlog = self.last_rows();
        if rows > nlog {
            return Err(RuntimeError::request(format!(
                "{rows} sampled rows requested, the last call returned {nlog}"
            )));
        }
        self.sample_rows(params, "ink_rt_sample_host")?;
        out.clear();
        out.resize(rows, 0);
        self.st.dtoh(out, &self.a.amax.slice(0..rows).rt()?).rt()
    }

    fn argmax_host(&self, rows: usize, out: &mut Vec<i32>) -> Res<()> {
        let nlog = self.last_rows();
        if rows > nlog {
            return Err(RuntimeError::request(format!(
                "{rows} argmax rows requested, the last call returned {nlog}"
            )));
        }
        if nlog > 0 {
            Dev {
                k: &self.k,
                st: &self.st,
            }
            .argmax(nlog, &self.a.logits, &self.a.amax)
            .rt()?;
        }
        out.clear();
        out.resize(rows, 0);
        self.st.dtoh(out, &self.a.amax.slice(0..rows).rt()?).rt()
    }

    fn mask_logits(&mut self, first: usize, rows: usize, ids: &[i32]) -> Res<()> {
        let nlog = self.last_rows();
        if first + rows > nlog {
            return Err(RuntimeError::request(format!(
                "mask of logit rows {first}..{} requested, the last call returned {nlog}",
                first + rows
            )));
        }
        for &id in ids {
            let id = usize::try_from(id)
                .ok()
                .filter(|&i| i < ink_rt_api::VOCAB)
                .ok_or_else(|| RuntimeError::request(format!("mask of token id {id} outside the vocabulary")))?;
            self.st
                .fill_strided(
                    &self.a.logits,
                    first * VOCAB_PAD + id,
                    VOCAB_PAD,
                    rows,
                    f32::NEG_INFINITY,
                )
                .rt()?;
        }
        Ok(())
    }

    fn set_graphs(&mut self, graphs: Graphs) -> Res<()> {
        // as the C++ `set_graphs`: the cache stays; graphs off runs every call eagerly
        self.graphs_on = graphs == Graphs::On;
        Ok(())
    }

    fn set_glue(&mut self, glue: Glue) -> Res<()> {
        self.glue = glue;
        Ok(())
    }

    fn set_tap(&mut self, tap: Option<TapFn>) -> Res<()> {
        self.tap = tap;
        Ok(())
    }

    fn stats(&self) -> Res<RuntimeStats> {
        Ok(RuntimeStats {
            arena_used: self.arena.plan().used() as u64,
            weight_bytes: self.w.bytes as u64,
            graph_captures: self.graphs.captures(),
            graphs_live: self.graphs.live() as u64,
            graph_bytes: self.graphs.held() as u64,
            graphs_refused: self.graphs.refused() as u64,
            gemm_fallbacks: 0,
        })
    }

    fn sync(&mut self) -> Res<()> {
        self.st.synchronize().rt()
    }

    fn set_profile_fine(&mut self, on: bool) -> Res<()> {
        self.profile_fine = on;
        Ok(())
    }

    fn set_profile(&mut self, on: bool) -> Res<()> {
        Engine::set_profile(self, on);
        Ok(())
    }

    fn marks(&mut self) -> Res<Vec<(String, f32)>> {
        let mut out: Vec<(String, f32)> = Vec::new();
        for pair in self.marks.windows(2) {
            let ms = pair[0].1.elapsed_ms(&pair[1].1).rt()?;
            match out.iter_mut().find(|(n, _)| *n == pair[0].0) {
                Some(e) => e.1 += ms,
                None => out.push((pair[0].0.clone(), ms)),
            }
        }
        Ok(out)
    }

    fn dump(&mut self, what: Dump, path: &Path) -> Res<()> {
        let text = match what {
            Dump::WeightSums => {
                let end = self.a.kpool.offset();
                let sums = weight_sums(&self.arena, &self.st, &self.w, end).rt()?;
                weight_sums_text(&sums, end)
            }
            Dump::ArenaMap => arena_map_text(self.arena.plan()),
            Dump::LaunchTrace => {
                let (key, lines, edges) = match &self.last_trace {
                    None => {
                        return Err(RuntimeError::request(
                            "no launch trace: no graph captured yet and no armed eager call",
                        ));
                    }
                    Some(TraceSrc::Eager(key, lines)) => (key, lines.clone(), lines.len().saturating_sub(1)),
                    Some(TraceSrc::Graph { key, segs }) => {
                        let e = self
                            .graphs
                            .entry(key)
                            .ok_or_else(|| RuntimeError::request(format!("graph {key} not in the cache")))?;
                        let (lines, edges) = e.graph.launch_trace(&self.arena).rt()?;
                        // the C++ key format: glue|gx_full/gx_swa:segments, with the graph's chunks now
                        let glue = key.split_once('|').map_or(key.as_str(), |k| k.0);
                        (&format!("{glue}|{}/{}:{segs}", e.chunks.0, e.chunks.1), lines, edges)
                    }
                };
                let mut s = format!(
                    "{{\"format\":\"ink-launch-trace/1\",\"key\":\"{}\",\"nodes\":{},\"edges\":{edges},\"arena_cap\":{}}}\n",
                    key.replace(['"', '\\'], "_"),
                    lines.len(),
                    self.arena.plan().cap()
                );
                for l in &lines {
                    s.push_str(l);
                    s.push('\n');
                }
                s
            }
        };
        fs::write(path, text).map_err(|e| RuntimeError::request(format!("writing {}: {e}", path.display())))
    }

    fn set_launch_trace(&mut self, on: bool) -> Res<()> {
        self.trace_armed = on;
        Ok(())
    }

    fn set_ablation(&mut self, regions: &[&str]) -> Res<()> {
        self.graphs.set_ablation(regions)
    }

    fn spec(&mut self) -> Option<&mut dyn SpecRuntime> {
        Some(self)
    }

    fn checkpoints(&mut self) -> Option<&mut dyn SlotCheckpoints> {
        Some(self)
    }

    fn sampler_probe(&mut self) -> Option<&mut dyn SamplerProbe> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_shapes_are_the_cpp_constructors_22() {
        let s = planned_shapes();
        assert_eq!(s.len(), 25);
        let uniq: std::collections::BTreeSet<_> = s.iter().collect();
        assert_eq!(uniq.len(), 25);
        // the four fused-projection shapes of D1 (pin entries of the production pin file)
        for n in [8600, 6552, 1056, 1024] {
            assert!(
                s.iter()
                    .any(|x| x.n == n && x.k == 2304 && x.class == PlanClass::Decode)
            );
        }
        // router K groups use the full row stride of the router weights
        assert!(
            s.iter()
                .any(|x| x.n == 128 && x.k == 256 && x.ldw == 2304 && x.output == Dtype::F32)
        );
    }
}
