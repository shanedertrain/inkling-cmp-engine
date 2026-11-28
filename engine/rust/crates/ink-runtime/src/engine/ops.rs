// Ported from engine/native/src/engine.cu @ 1c7ac46 (every `<<<` launch site of the forward pass, commit and argmax;
// launch geometry of da::launch / da::ring_launch, pfa::launch, moe_launch_dev); deltas to 25f1fcc (PORT-MAP.md). Own code.
//! Every kernel launch of the engine: the only place `unsafe` is allowed in this crate (docs/RUST-PLAN.md 3.2, review
//! R3 deviation 4). Each function is one C++ launch site: it takes the engine's buffers, checks every extent the
//! host can bound, computes the launch geometry exactly as the C++ does, and calls the generated `unsafe` launcher
//! with a `// SAFETY:` comment stating the kernel's contract (its comment in the `.cuh`, `engine/native/README.md`).
//! Extents that depend on device data (page tables, routing, item lists) are bounded by how this engine builds that
//! data (`meta.rs`, the routing kernels' contracts) and are stated, not checked.
#![allow(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use ink_cuda::{Bf16, CudaError, DevBuf, DevPtr, LaunchCfg, Result, Stream};
use ink_kernels::*;
use ink_model::dims::{DREL, EPS, GATE_RANK, H, HD, KD, KVH, NEXP, NH, NVISITS, QD, REL_EXT, TOPK, VOCAB, VOCAB_PAD};

/// `glue::SLD`: conv-state row stride.
pub const SLD: usize = 6656;
/// decode_attn design A dynamic shared memory (`da::smem_bytes()` = 8 x 32 x 24 + 4 x 24 x 128 x 4).
pub const DA_SMEM: u32 = 6144 + 4 * 24 * 128 * 4;
/// decode_attn design A 16-row single-pass dynamic shared memory (`da::smem_bytes16()` = 12,288 + 32,768 + 24,576).
pub const DA_SMEM16: u32 = 12_288 + 32_768 + 2 * 24 * 128 * 4;
/// decode_attn design B (ring) shared memory at 3 stages, bf16, `nw <= 3` (`da::ring_smem`).
#[cfg(test)]
pub const DA_RING_SMEM: u32 = 3 * 16 * 272 * 2 + 3 * 8 * 128 * 4;
/// decode_attn design B (ring) shared memory at 3 stages, bf16, for `nw` column-split warps (1..=6).
pub const fn da_ring_smem(nw: usize) -> u32 {
    let fw = if nw > 3 { nw } else { 3 };
    3 * 16 * 272 * 2 + (fw as u32) * 8 * 128 * 4
}
/// prefill_attn<4, 2, 1, 1, 2> dynamic shared memory (`Cfg::SMEM` = 2 x (2 x 64 x 128 + 112 x 16 + 128 x 128 + 4 x 16 x 74) + 16).
pub const PFA_SMEM: u32 = 2 * (2 * 8192 + 112 * 16 + 16384 + 4 * 16 * 74) + 16;
/// prefill_attn_far<4, 2, 1, 1, 2> dynamic shared memory (`CfgFar::SMEM` = 2 x (2 x 64 x 128 + 128 x 128) + 16).
pub const PFA_FAR_SMEM: u32 = 2 * (2 * 8192 + 16384) + 16;
/// prefill_attn_fp8<4, 2, 1, 1, 2> dynamic shared memory (`CfgFp8::SMEM` = 2 x 8192 + 112 x 16 x 2 + 32768 + 4 x 16 x 74 x 2 + 16).
pub const PFA_FP8_SMEM: u32 = 2 * 8192 + 112 * 16 * 2 + 32768 + 4 * 16 * 74 * 2 + 16;
/// prefill_attn_fp8_far<4, 2, 2> dynamic shared memory (`CfgFp8Far::SMEM` = 2 x 8192 + 32768 + 16).
pub const PFA_FP8_FAR_SMEM: u32 = 2 * 8192 + 32768 + 16;
/// moe_gemm2<NVFP4B, *, 3, 8> shared memory (`Cfg2::SMEM`: 3 x (16384 + 4 x 2048 + 4 x 512) + 128 x 8).
pub const MOE_SMEM_NVFP4: u32 = 3 * (16384 + 8192 + 2048) + 1024;
/// moe_gemm2<NVFP4_N1S1, *, 3, 8> shared memory (`Cfg2::SMEM`: 3 x (16384 + 4 x 2048 + 4 x 256) + 128 x 8).
pub const MOE_SMEM_NVFP4_N1S1: u32 = 3 * (16384 + 8192 + 1024) + 1024;
/// moe_gemm2<MXFP8, *, 3, 8> shared memory (3 x (16384 + 4 x 4096 + 4 x 128) + 128 x 8).
pub const MOE_SMEM_MXFP8: u32 = 3 * (16384 + 16384 + 512) + 1024;
/// moe_gemm2 threads (8 warps).
const MOE_THREADS: u32 = 256;
/// moe_gemm_dec<NVFP4_N1S1, *, 5, 4> shared memory (5 x (2048 + 4096 + 512) + 128).
pub const MOE_DEC_SMEM_N1S1: u32 = 5 * (2048 + 4096 + 512) + 128;
/// moe_gemm_dec<MXFP8, *, 5, 4> shared memory (5 x (2048 + 8192 + 256) + 128).
pub const MOE_DEC_SMEM_MXFP8: u32 = 5 * (2048 + 8192 + 256) + 128;
/// moe_gemm_dec threads (4 warps).
const MOE_DEC_THREADS: u32 = 128;
/// `pglue::kBuildRows`, `kBuildRowsFast`, `kMgRows`, `kRelRows`.
#[allow(dead_code)]
pub const BUILD_ROWS: usize = 128;
const BUILD_ROWS_FAST: usize = 512;
const MG_ROWS: usize = 32;
/// Rank of a message gate's down projection (`[T][64]`).
const MG_RANK: usize = ink_model::dims::GATE_RANK;
const REL_ROWS: usize = 32;
/// `pglue::kWarpRows` (warps per block of the warp-per-row kernels), `kConvRows`, `kConvThreads`, `kTcRows`.
const WARP_ROWS: usize = 4;
const CONV_ROWS: usize = 8;
const CONV_THREADS: usize = 96;
const TC_ROWS: usize = 32;
const CONV_AR_ROWS: usize = 4;
const COMBINE_CONV_ROWS: usize = 32;
/// decode_attn: `kG` query heads per KV head, `kD`.
pub(super) const DA_G: usize = 3;

/// The pointer argument of a buffer.
fn b<T>(x: &DevBuf<T>) -> Ptr<'_, T> {
    Ptr::Buf(x)
}

/// `Some` buffer or the null pointer.
fn opt<T>(x: Option<&DevBuf<T>>) -> Ptr<'_, T> {
    x.map_or(Ptr::Null, Ptr::Buf)
}

/// Fails unless `x` holds at least `n` elements.
fn need<T: Pod>(x: &DevBuf<T>, n: usize, what: &'static str) -> Result<()> {
    if x.len() >= n {
        Ok(())
    } else {
        Err(CudaError::Bounds {
            what,
            detail: format!("{} elements, the launch reads or writes {n}", x.len()),
        })
    }
}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("launch dimension fits u32")
}

fn i(x: usize) -> i32 {
    i32::try_from(x).expect("kernel int argument fits i32")
}

fn cfg(grid: [usize; 3], block: usize, smem: u32) -> LaunchCfg {
    LaunchCfg {
        grid: [u(grid[0]), u(grid[1]), u(grid[2])],
        block: [u(block), 1, 1],
        smem,
    }
}

/// `grid1`: blocks of 256 for `n` elements, at most 74 x 16 (grid-stride kernels).
fn grid1(n: usize) -> usize {
    n.div_ceil(256).min(74 * 16)
}

/// The kernel and engine handles every launch needs.
pub(super) struct Dev<'a> {
    pub k: &'a Kernels,
    pub st: &'a Stream,
}

/// `glue::Rows` of the call (row metadata buffers of `Engine::alloc`).
pub(super) fn rows(pos: &DevBuf<i32>, seg: &DevBuf<i32>, row0: &DevBuf<i32>, slot: &DevBuf<i32>) -> glue::Rows {
    glue::Rows {
        pos: pos.ptr(),
        seg: seg.ptr(),
        row0: row0.ptr(),
        slot: slot.ptr(),
    }
}

/// `glue::ResSrc`: the `nb` completed blocks, the partial block as bf16 or fp32 (row stride `ld_last`).
pub(super) fn res_src(
    blk: &[&DevBuf<Bf16>],
    last16: Option<&DevBuf<Bf16>>,
    last32: Option<&DevBuf<f32>>,
    ld_last: usize,
) -> glue::ResSrc {
    let mut s = glue::ResSrc::zeroed();
    for (d, b) in s.blk.iter_mut().zip(blk) {
        *d = b.ptr();
    }
    s.nb = i(blk.len());
    s.last16 = last16.map_or(DevPtr::zeroed(), DevBuf::ptr);
    s.last32 = last32.map_or(DevPtr::zeroed(), DevBuf::ptr);
    s.ld_last = i(ld_last);
    s
}

/// Checks the AttnRes extents (Q query rows, T x Q x H out) and builds the loop-gate inputs `GateIn` of an AttnRes
/// launch with G gates (C3); G = 0 is the C++ default argument (all zero).
fn gate_in(
    q: usize,
    g: usize,
    t: usize,
    queries: &DevBuf<f32>,
    out: &DevBuf<f32>,
    gates: Option<GateInputs<'_>>,
) -> Result<glue::GateIn> {
    need(queries, q * H, "attn_res queries")?;
    need(out, t * q * H, "attn_res out")?;
    let gi = match (&gates, g) {
        (None, 0) => glue::GateIn::zeroed(),
        (Some(x), 1..=3) => {
            // gate y reads msg + y H at row stride ldm, writes gin + y gs (gs = T 2H) rows of 2H
            need(x.msg, (g - 1) * H + (t - 1) * x.ldm + H, "attn_res gate msg")?;
            need(x.gin, g * t * 2 * H, "attn_res gate gin")?;
            glue::GateIn {
                msg: x.msg.ptr(),
                ldm: i(x.ldm),
                _pad0: [0; 4],
                g: x.gin.ptr(),
                gs: (t * 2 * H) as i64,
                qsel: i(usize::from(x.qsel)),
                _pad1: [0; 4],
            }
        }
        _ => {
            return Err(CudaError::Bounds {
                what: "attn_res G",
                detail: format!("G = {g} with gate inputs {}", gates.is_some()),
            });
        }
    };
    if let Some(x) = &gates
        && x.qsel
        && q < g
    {
        return Err(CudaError::Bounds {
            what: "attn_res qsel",
            detail: format!("Q = {q} < G = {g}"),
        });
    }
    Ok(gi)
}

impl Dev<'_> {
    // ------------------------------------------------------------------------------------------------ embed, head

    /// `glue::embed_norm_kernel<<<T, 256>>>(ids, embed, embed_norm, hid0, EPS)`.
    pub fn embed_norm(
        &self,
        t: usize,
        ids: &DevBuf<i32>,
        emb: &DevBuf<Bf16>,
        w: &DevBuf<f32>,
        out: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(ids, t, "embed ids")?;
        need(w, H, "embed norm")?;
        need(out, t * H, "embed out")?;
        // SAFETY: block t reads ids[t] (t < T, checked) and row ids[t] of the embedding (ids are token ids < 200,058
        // from the caller, below the 201,024 padded rows of `emb`), writes out row t (T x H checked).
        unsafe {
            self.k.glue_embed_norm_kernel(
                self.st,
                cfg([t, 1, 1], 256, 0),
                GlueEmbedNormKernelArgs {
                    ids: b(ids),
                    emb: b(emb),
                    w: b(w),
                    out: b(out),
                    eps: EPS,
                },
            )
        }
    }

    /// `glue::feat_copy_kernel<<<grid1(T H), 256>>>(src, src32, feats, j, T)`: DSpark feature row block `j`.
    pub fn feat_copy(
        &self,
        t: usize,
        src: Option<&DevBuf<Bf16>>,
        src32: Option<&DevBuf<f32>>,
        feats: &DevBuf<Bf16>,
        j: usize,
    ) -> Result<()> {
        need(feats, t * 4 * H, "feats")?;
        if let Some(s) = src {
            need(s, t * H, "feat src")?;
        }
        if let Some(s) = src32 {
            need(s, t * H, "feat src32")?;
        }
        // SAFETY: grid-stride over T x H: reads row t of the one source given (bf16 or fp32, the other null as the
        // kernel takes it), writes feats[t][j][:] (T x 4 x H checked, j < 4).
        unsafe {
            self.k.glue_feat_copy_kernel(
                self.st,
                cfg([grid1(t * H), 1, 1], 256, 0),
                GlueFeatCopyKernelArgs {
                    src: opt(src),
                    src32: opt(src32),
                    feats: b(feats),
                    j: i(j),
                    T: i(t),
                },
            )
        }
    }

    /// `glue::attn_res_kernel<Q, G><<<T, 256>>>(s, queries, out, EPS, gi)`: (Q, G) = (1, 0), (3, 0), (4, 0), and with
    /// the loop-gate inputs (C3) (1, 1), (1, 3), (3, 3). With G = 0, `gi` is the C++ default argument (all zero).
    /// With `qsplit` (decode-only calls) the 3- and 4-query launches run one block per (row, query): `<1, 0>` on grid
    /// (T, Q) for (3, 0) and (4, 0), `<1, 1>` on grid (T, 3) for (3, 3) with `qsel` (block y writes gate y's inputs);
    /// the same bits as one block per row.
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn attn_res(
        &self,
        q: usize,
        g: usize,
        t: usize,
        s: &glue::ResSrc,
        queries: &DevBuf<f32>,
        out: &DevBuf<f32>,
        gates: Option<GateInputs<'_>>,
        qsplit: bool,
    ) -> Result<()> {
        let qsel = gates.as_ref().is_some_and(|x| x.qsel);
        let gi = gate_in(q, g, t, queries, out, gates)?;
        // (instance Q, instance G, query blocks per row)
        let (kq, kg, qb) = match (q, g) {
            (3 | 4, 0) if qsplit => (1, 0, q),
            (3, 3) if qsplit && qsel => (1, 1, 3),
            _ => (q, g, 1),
        };
        let c = cfg([t, qb, 1], 256, 0);
        let (queries, out) = (b(queries), b(out));
        let s = *s;
        macro_rules! go {
            ($f:ident, $a:ident) => {
                self.k.$f(
                    self.st,
                    c,
                    $a {
                        s,
                        queries,
                        out,
                        eps: EPS,
                        gi,
                    },
                )
            };
        }
        // SAFETY: block (t, y) reads row t of each block source in `s` (engine buffers of t_max x H, row stride H) and
        // of the partial (row stride ld_last), query rows y kq .. (y + 1) kq of the Q checked, writes
        // out[t][y kq .. (y + 1) kq][:] of the T x Q x H checked (row stride gridDim.y kq = Q); with G > 0 it also
        // reads row t of gate y's message (msg + y H, stride ldm) and its own outputs, and writes gin[y][t][0..2H] for
        // its gates (qsel: gate y = query y < G; checked for all G).
        unsafe {
            match (kq, kg) {
                (1, 0) => go!(glue_attn_res_kernel_1_0, GlueAttnResKernel10Args),
                (3, 0) => go!(glue_attn_res_kernel_3_0, GlueAttnResKernel30Args),
                (4, 0) => go!(glue_attn_res_kernel_4_0, GlueAttnResKernel40Args),
                (1, 1) => go!(glue_attn_res_kernel_1_1, GlueAttnResKernel11Args),
                (1, 3) => go!(glue_attn_res_kernel_1_3, GlueAttnResKernel13Args),
                (3, 3) => go!(glue_attn_res_kernel_3_3, GlueAttnResKernel33Args),
                _ => Err(CudaError::Bounds {
                    what: "attn_res Q / G",
                    detail: format!("{q} / {g}"),
                }),
            }
        }
    }

    /// `glue::attn_res_norm_kernel<Q><<<T, 256>>>(s, queries, w0, w1, w2, o0, o1, o2, EPS)` (Q = 1 or 3). With
    /// `qsplit` (decode-only calls) Q = 3 runs as `<1>` on grid (T, 3), block y on query y with w[y] into o[y]: the
    /// same bits.
    #[allow(dead_code)]
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn attn_res_norm(
        &self,
        q: usize,
        t: usize,
        s: &glue::ResSrc,
        queries: &DevBuf<f32>,
        w: [Option<&DevBuf<f32>>; 3],
        o: [Option<&DevBuf<Bf16>>; 3],
        qsplit: bool,
    ) -> Result<()> {
        need(queries, q * H, "attn_res_norm queries")?;
        for k in 0..q {
            need(
                w[k].ok_or(CudaError::Bounds {
                    what: "attn_res_norm",
                    detail: "missing weight".into(),
                })?,
                H,
                "norm w",
            )?;
            need(
                o[k].ok_or(CudaError::Bounds {
                    what: "attn_res_norm",
                    detail: "missing out".into(),
                })?,
                t * H,
                "norm out",
            )?;
        }
        let (kq, qb) = if q == 3 && qsplit { (1, 3) } else { (q, 1) };
        let c = cfg([t, qb, 1], 256, 0);
        // SAFETY: as attn_res for the sources; then RMSNorm of each mixed row of block (t, y) (queries y kq ..
        // (y + 1) kq of the Q checked) with w[q] into o[q] row t (Q outputs of T x H checked; unused outputs null, as
        // the kernel's instance ignores them beyond its queries).
        unsafe {
            match kq {
                1 => self.k.glue_attn_res_norm_kernel_1(
                    self.st,
                    c,
                    GlueAttnResNormKernel1Args {
                        s: *s,
                        queries: b(queries),
                        w0: opt(w[0]),
                        w1: opt(w[1]),
                        w2: opt(w[2]),
                        o0: opt(o[0]),
                        o1: opt(o[1]),
                        o2: opt(o[2]),
                        eps: EPS,
                    },
                ),
                3 => self.k.glue_attn_res_norm_kernel_3(
                    self.st,
                    c,
                    GlueAttnResNormKernel3Args {
                        s: *s,
                        queries: b(queries),
                        w0: opt(w[0]),
                        w1: opt(w[1]),
                        w2: opt(w[2]),
                        o0: opt(o[0]),
                        o1: opt(o[1]),
                        o2: opt(o[2]),
                        eps: EPS,
                    },
                ),
                _ => Err(CudaError::Bounds {
                    what: "attn_res_norm Q",
                    detail: q.to_string(),
                }),
            }
        }
    }

    /// `pglue::ar_rows_kernel<Q, G><<<ceil(T / 4), 128>>>(s, queries, out, EPS, gi, T)`: [`Dev::attn_res`] with
    /// one warp per row (prefill row counts, T > 64; same bits).
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    #[allow(dead_code)]
    pub fn ar_rows(
        &self,
        q: usize,
        g: usize,
        t: usize,
        s: &glue::ResSrc,
        queries: &DevBuf<f32>,
        out: &DevBuf<f32>,
        gates: Option<GateInputs<'_>>,
    ) -> Result<()> {
        let gi = gate_in(q, g, t, queries, out, gates)?;
        let c = cfg([t.div_ceil(WARP_ROWS), 1, 1], 32 * WARP_ROWS, 0);
        let (queries, out) = (b(queries), b(out));
        let s = *s;
        let rows = i(t);
        macro_rules! go {
            ($f:ident, $a:ident) => {
                self.k.$f(
                    self.st,
                    c,
                    $a {
                        s,
                        queries,
                        out,
                        eps: EPS,
                        gi,
                        T: rows,
                    },
                )
            };
        }
        // SAFETY: as attn_res, one warp per row t < T (warps past T stop): all buffer extents checked above.
        unsafe {
            match (q, g) {
                (1, 0) => go!(pglue_ar_rows_kernel_1_0, PglueArRowsKernel10Args),
                (3, 0) => go!(pglue_ar_rows_kernel_3_0, PglueArRowsKernel30Args),
                (4, 0) => go!(pglue_ar_rows_kernel_4_0, PglueArRowsKernel40Args),
                (1, 1) => go!(pglue_ar_rows_kernel_1_1, PglueArRowsKernel11Args),
                (1, 3) => go!(pglue_ar_rows_kernel_1_3, PglueArRowsKernel13Args),
                (3, 3) => go!(pglue_ar_rows_kernel_3_3, PglueArRowsKernel33Args),
                _ => Err(CudaError::Bounds {
                    what: "ar_rows Q / G",
                    detail: format!("{q} / {g}"),
                }),
            }
        }
    }

    /// `pglue::ar_norm_rows_kernel<Q><<<ceil(T / 4), 128>>>(s, queries, w0, w1, w2, o0, o1, o2, EPS, T)`:
    /// [`Dev::attn_res_norm`] with one warp per row (T > 64; same bits).
    #[allow(dead_code)]
    pub fn ar_norm_rows(
        &self,
        q: usize,
        t: usize,
        s: &glue::ResSrc,
        queries: &DevBuf<f32>,
        w: [Option<&DevBuf<f32>>; 3],
        o: [Option<&DevBuf<Bf16>>; 3],
    ) -> Result<()> {
        need(queries, q * H, "ar_norm_rows queries")?;
        for k in 0..q {
            let (Some(wk), Some(ok)) = (w[k], o[k]) else {
                return Err(CudaError::Bounds {
                    what: "ar_norm_rows",
                    detail: format!("weight or output {k} missing"),
                });
            };
            need(wk, H, "norm w")?;
            need(ok, t * H, "norm out")?;
        }
        let c = cfg([t.div_ceil(WARP_ROWS), 1, 1], 32 * WARP_ROWS, 0);
        macro_rules! go {
            ($f:ident, $a:ident) => {
                self.k.$f(
                    self.st,
                    c,
                    $a {
                        s: *s,
                        queries: b(queries),
                        w0: opt(w[0]),
                        w1: opt(w[1]),
                        w2: opt(w[2]),
                        o0: opt(o[0]),
                        o1: opt(o[1]),
                        o2: opt(o[2]),
                        eps: EPS,
                        T: i(t),
                    },
                )
            };
        }
        // SAFETY: as attn_res_norm, one warp per row t < T: RMSNorm of each of the Q mixed rows with w[q] into o[q]
        // row t (Q outputs of T x H checked; unused outputs null, ignored by the Q < 3 instance).
        unsafe {
            match q {
                1 => go!(pglue_ar_norm_rows_kernel_1, PglueArNormRowsKernel1Args),
                3 => go!(pglue_ar_norm_rows_kernel_3, PglueArNormRowsKernel3Args),
                _ => Err(CudaError::Bounds {
                    what: "ar_norm_rows Q",
                    detail: q.to_string(),
                }),
            }
        }
    }

    /// `pglue::ar288_kernel<0 | 1><<<(T, Q or NG), 288>>>`: 288-thread (9 warps x 8 elems) vectorized AttnRes.
    /// For prefill row counts (`t > 64`), launches single-CTA multi-query/multi-gate row-coarsened variants so sources
    /// are read ONCE instead of Q/G times (100% bit-identical).
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn ar288(
        &self,
        q: usize,
        g: usize,
        t: usize,
        s: &glue::ResSrc,
        queries: &DevBuf<f32>,
        out: &DevBuf<f32>,
        gates: Option<GateInputs<'_>>,
    ) -> Result<()> {
        let qsel = gates.as_ref().is_some_and(|x| x.qsel);
        let gi = gate_in(q, g, t, queries, out, gates)?;
        let (s, queries, out) = (*s, b(queries), b(out));
        // SAFETY: block (t, y) runs 288 threads (8 consecutive elems/thread) over row(s) with 16-byte ld8/st8; extents checked by gate_in.
        unsafe {
            let has_last16 = bytemuck::bytes_of(&s.last16).iter().any(|&x| x != 0);
            if t > 128 && q == 4 && g == 0 && s.nb == 1 && has_last16 {
                self.k.pglue_ar288_msgs_rows_kernel(
                    self.st,
                    cfg([t, 1, 1], 288, 0),
                    PglueAr288MsgsRowsKernelArgs {
                        s,
                        queries,
                        out,
                        eps: EPS,
                        T: i(t),
                    },
                )
            } else if t > 128 && q == 1 && g == 3 && !qsel {
                self.k.pglue_ar288_g3_shared_rows_kernel(
                    self.st,
                    cfg([t, 1, 1], 288, 0),
                    PglueAr288G3SharedRowsKernelArgs {
                        s,
                        queries,
                        out,
                        eps: EPS,
                        gi,
                        T: i(t),
                    },
                )
            } else if t > 128 && q == 3 && g == 3 && qsel {
                self.k.pglue_ar288_q3g3_rows_kernel(
                    self.st,
                    cfg([t, 1, 1], 288, 0),
                    PglueAr288Q3g3RowsKernelArgs {
                        s,
                        queries,
                        out,
                        eps: EPS,
                        gi,
                        T: i(t),
                    },
                )
            } else if g == 0 {
                self.k.pglue_ar288_kernel_0(
                    self.st,
                    cfg([t, q, 1], 288, 0),
                    PglueAr288Kernel0Args {
                        s,
                        queries,
                        out,
                        eps: EPS,
                        gi,
                    },
                )
            } else {
                self.k.pglue_ar288_kernel_1(
                    self.st,
                    cfg([t, g, 1], 288, 0),
                    PglueAr288Kernel1Args {
                        s,
                        queries,
                        out,
                        eps: EPS,
                        gi,
                    },
                )
            }
        }
    }

    /// Track 3A-7: compute virtual-message weights `[12][T][4]` `float4(p0, p1, inv, 0)` ONCE per pass into `mw`
    /// (`a.msgs`), replacing 12 materializing `msgs` launches (~75.5 MB/visit HBM traffic -> 1.6 MB/pass).
    pub fn vmsg_weights(
        &self,
        t: usize,
        msrc0: &DevBuf<Bf16>,
        msrc1: &DevBuf<Bf16>,
        msg_queries: &DevBuf<f32>,
        mw: &DevBuf<f32>,
    ) -> Result<()> {
        need(msrc0, t * H, "vmsg msrc0")?;
        need(msrc1, t * H, "vmsg msrc1")?;
        need(msg_queries, 12 * 4 * H, "vmsg msg_queries")?;
        need(mw, 12 * t * 16, "vmsg mw")?;
        // SAFETY: block (t0/16, i) reads msrc0/1[t0..t0+16], msg_queries[i*4..], writes float4 weights into mw (all checked).
        unsafe {
            self.k.pglue_vmsg_weights_kernel(
                self.st,
                cfg([t.div_ceil(16), 12, 1], 288, 0),
                PglueVmsgWeightsKernelArgs {
                    msrc0: b(msrc0),
                    msrc1: b(msrc1),
                    msg_queries: b(msg_queries),
                    mw: b(mw),
                    eps: EPS,
                    T: i(t),
                },
            )
        }
    }

    /// Track 3A-7: `ar288_g3_vmsg_kernel` (`q == 1`, local) or `ar288_q3g3_vmsg_kernel` (`q == 3`, global) reading
    /// L2-hot bf16 `msrc0/1` + `mw` instead of fp32 `a.msgs` (100% bit-identical).
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn ar288_vmsg(
        &self,
        q: usize,
        t: usize,
        s: &glue::ResSrc,
        queries: &DevBuf<f32>,
        out: &DevBuf<f32>,
        msrc0: &DevBuf<Bf16>,
        msrc1: &DevBuf<Bf16>,
        mw: &DevBuf<f32>,
        gin: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(queries, q * H, "ar288_vmsg queries")?;
        need(out, t * q * H, "ar288_vmsg out")?;
        need(msrc0, t * H, "ar288_vmsg msrc0")?;
        need(msrc1, t * H, "ar288_vmsg msrc1")?;
        need(mw, t * 16, "ar288_vmsg mw")?;
        need(gin, 3 * t * 2 * H, "ar288_vmsg gin")?;
        let gs = (t * 2 * H) as i64;
        // SAFETY: block t < T reads s, queries, msrc0/1[t], mw[t*16..], writes out and 3 gates of gin (all checked).
        unsafe {
            if q == 1 {
                self.k.pglue_ar288_g3_vmsg_kernel(
                    self.st,
                    cfg([t, 1, 1], 288, 0),
                    PglueAr288G3VmsgKernelArgs {
                        s: *s,
                        queries: b(queries),
                        out: b(out),
                        eps: EPS,
                        msrc0: b(msrc0),
                        msrc1: b(msrc1),
                        mw: b(mw),
                        gin: b(gin),
                        gs,
                        T: i(t),
                    },
                )
            } else {
                self.k.pglue_ar288_q3g3_vmsg_kernel(
                    self.st,
                    cfg([t, 1, 1], 288, 0),
                    PglueAr288Q3g3VmsgKernelArgs {
                        s: *s,
                        queries: b(queries),
                        out: b(out),
                        eps: EPS,
                        msrc0: b(msrc0),
                        msrc1: b(msrc1),
                        mw: b(mw),
                        gin: b(gin),
                        gs,
                        T: i(t),
                    },
                )
            }
        }
    }

    /// `pglue::ar_norm288_kernel<<<(T, Q), 288>>>`: 288-thread vectorized AttnRes + RMSNorm (for `t > 64`, single-CTA
    /// multi-query row-coarsened `ar_norm288_rows_kernel<Q>`, 100% bit-identical).
    pub fn ar_norm288(
        &self,
        q: usize,
        t: usize,
        s: &glue::ResSrc,
        queries: &DevBuf<f32>,
        w: [Option<&DevBuf<f32>>; 3],
        o: [Option<&DevBuf<Bf16>>; 3],
    ) -> Result<()> {
        need(queries, q * H, "ar_norm288 queries")?;
        for k in 0..q {
            let (Some(wk), Some(ok)) = (w[k], o[k]) else {
                return Err(CudaError::Bounds {
                    what: "ar_norm288",
                    detail: format!("weight or output {k} missing"),
                });
            };
            need(wk, H, "norm w")?;
            need(ok, t * H, "norm out")?;
        }
        // SAFETY: block runs 288 threads over row(s), writes RMSNorm output o[q][t] (all extents checked).
        unsafe {
            if t > 128 && q == 3 {
                self.k.pglue_ar_norm288_rows_kernel_3(
                    self.st,
                    cfg([t, 1, 1], 288, 0),
                    PglueArNorm288RowsKernel3Args {
                        s: *s,
                        queries: b(queries),
                        w0: opt(w[0]),
                        w1: opt(w[1]),
                        w2: opt(w[2]),
                        o0: opt(o[0]),
                        o1: opt(o[1]),
                        o2: opt(o[2]),
                        eps: EPS,
                        T: i(t),
                    },
                )
            } else {
                self.k.pglue_ar_norm288_kernel(
                    self.st,
                    cfg([t, q, 1], 288, 0),
                    PglueArNorm288KernelArgs {
                        s: *s,
                        queries: b(queries),
                        w0: opt(w[0]),
                        w1: opt(w[1]),
                        w2: opt(w[2]),
                        o0: opt(o[0]),
                        o1: opt(o[1]),
                        o2: opt(o[2]),
                        eps: EPS,
                    },
                )
            }
        }
    }

    /// `pglue::conv_ar_norm288_kernel<<<T, 288>>>` (or row-coarsened `conv_ar_norm288_rows_kernel` for `t > 64`):
    /// fused conv_partial + ar_mlp + mlp_norm (plain visits).
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn conv_ar_norm288(
        &self,
        t: usize,
        ax: &DevBuf<Bf16>,
        cw: &DevBuf<f32>,
        state: &DevBuf<Bf16>,
        r: glue::Rows,
        hin: &DevBuf<Bf16>,
        blk_out: Option<&DevBuf<Bf16>>,
        partial: &DevBuf<f32>,
        s: &glue::ResSrc,
        query: &DevBuf<f32>,
        norm_w: &DevBuf<f32>,
        xm_out: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(ax, t * H, "conv_ar ax")?;
        need(cw, H * 4, "conv_ar cw")?;
        need(hin, t * H, "conv_ar hin")?;
        need(partial, t * H, "conv_ar partial")?;
        need(query, H, "conv_ar query")?;
        need(norm_w, H, "conv_ar norm_w")?;
        need(xm_out, t * H, "conv_ar xm_out")?;
        if let Some(bk) = blk_out {
            need(bk, t * H, "conv_ar blk_out")?;
        }
        // SAFETY: block runs 288 threads (8 channels/thread), computes conv(ax) + partial (+ blk_out if bstart),
        // runs AttnRes over s.blk[0..nb_prev] + hin(if bstart) + partial, and writes RMSNorm to xm_out[t] (all checked).
        unsafe {
            if t > 128 {
                self.k.pglue_conv_ar_norm288_rows_kernel(
                    self.st,
                    cfg([t.div_ceil(CONV_AR_ROWS), 1, 1], 288, 0),
                    PglueConvArNorm288RowsKernelArgs {
                        ax: b(ax),
                        cw: b(cw),
                        state: b(state),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: r,
                        hin: b(hin),
                        block_start: i32::from(blk_out.is_some()),
                        blk_out: opt(blk_out),
                        partial32: b(partial),
                        s: *s,
                        query: b(query),
                        norm_w: b(norm_w),
                        xm_out: b(xm_out),
                        eps: EPS,
                        T: i(t),
                    },
                )
            } else {
                self.k.pglue_conv_ar_norm288_kernel(
                    self.st,
                    cfg([t, 1, 1], 288, 0),
                    PglueConvArNorm288KernelArgs {
                        ax: b(ax),
                        cw: b(cw),
                        state: b(state),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: r,
                        hin: b(hin),
                        block_start: i32::from(blk_out.is_some()),
                        blk_out: opt(blk_out),
                        partial32: b(partial),
                        s: *s,
                        query: b(query),
                        norm_w: b(norm_w),
                        xm_out: b(xm_out),
                        eps: EPS,
                    },
                )
            }
        }
    }

    /// `pglue::conv_ar_gate288_kernel<<<T, 288>>>` (or row-coarsened `conv_ar_gate288_rows_kernel` for `t > 64`):
    /// fused conv_partial + ar_mlp_m + mg_in (loop-message visits).
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn conv_ar_gate288(
        &self,
        t: usize,
        ax: &DevBuf<Bf16>,
        cw: &DevBuf<f32>,
        state: &DevBuf<Bf16>,
        r: glue::Rows,
        hin: &DevBuf<Bf16>,
        blk_out: Option<&DevBuf<Bf16>>,
        partial: &DevBuf<f32>,
        s: &glue::ResSrc,
        query: &DevBuf<f32>,
        ar_out: &DevBuf<f32>,
        gates: GateInputs<'_>,
    ) -> Result<()> {
        need(ax, t * H, "conv_ar_gate ax")?;
        need(cw, H * 4, "conv_ar_gate cw")?;
        need(hin, t * H, "conv_ar_gate hin")?;
        need(partial, t * H, "conv_ar_gate partial")?;
        if let Some(bk) = blk_out {
            need(bk, t * H, "conv_ar_gate blk_out")?;
        }
        let gi = gate_in(1, 1, t, query, ar_out, Some(gates))?;
        // SAFETY: block runs 288 threads, writes partial (+ blk_out if bstart), ar_out[t], and gate 0's gin[t] (all checked).
        unsafe {
            if t > 128 {
                self.k.pglue_conv_ar_gate288_rows_kernel(
                    self.st,
                    cfg([t.div_ceil(CONV_AR_ROWS), 1, 1], 288, 0),
                    PglueConvArGate288RowsKernelArgs {
                        ax: b(ax),
                        cw: b(cw),
                        state: b(state),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: r,
                        hin: b(hin),
                        block_start: i32::from(blk_out.is_some()),
                        blk_out: opt(blk_out),
                        partial32: b(partial),
                        s: *s,
                        query: b(query),
                        ar_out: b(ar_out),
                        eps: EPS,
                        gi,
                        T: i(t),
                    },
                )
            } else {
                self.k.pglue_conv_ar_gate288_kernel(
                    self.st,
                    cfg([t, 1, 1], 288, 0),
                    PglueConvArGate288KernelArgs {
                        ax: b(ax),
                        cw: b(cw),
                        state: b(state),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: r,
                        hin: b(hin),
                        block_start: i32::from(blk_out.is_some()),
                        blk_out: opt(blk_out),
                        partial32: b(partial),
                        s: *s,
                        query: b(query),
                        ar_out: b(ar_out),
                        eps: EPS,
                        gi,
                    },
                )
            }
        }
    }

    /// Track 3A-7: `conv_ar_gate288_vmsg_kernel` (reconstructs exact `msgs[3]` from L2-hot `msrc0/1` + `mw`).
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn conv_ar_gate288_vmsg(
        &self,
        t: usize,
        ax: &DevBuf<Bf16>,
        cw: &DevBuf<f32>,
        state: &DevBuf<Bf16>,
        r: glue::Rows,
        hin: &DevBuf<Bf16>,
        blk_out: Option<&DevBuf<Bf16>>,
        partial: &DevBuf<f32>,
        s: &glue::ResSrc,
        query: &DevBuf<f32>,
        ar_out: &DevBuf<f32>,
        msrc0: &DevBuf<Bf16>,
        msrc1: &DevBuf<Bf16>,
        mw: &DevBuf<f32>,
        gin: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(ax, t * H, "conv_ar_gate_vmsg ax")?;
        need(cw, H * 4, "conv_ar_gate_vmsg cw")?;
        need(hin, t * H, "conv_ar_gate_vmsg hin")?;
        need(partial, t * H, "conv_ar_gate_vmsg partial")?;
        need(query, H, "conv_ar_gate_vmsg query")?;
        need(ar_out, t * H, "conv_ar_gate_vmsg ar_out")?;
        need(msrc0, t * H, "conv_ar_gate_vmsg msrc0")?;
        need(msrc1, t * H, "conv_ar_gate_vmsg msrc1")?;
        need(mw, t * 16, "conv_ar_gate_vmsg mw")?;
        need(gin, t * 2 * H, "conv_ar_gate_vmsg gin")?;
        if let Some(bk) = blk_out {
            need(bk, t * H, "conv_ar_gate_vmsg blk_out")?;
        }
        // SAFETY: block runs 288 threads x 4 rows, writes partial (+ blk_out if bstart), ar_out, and gin (all checked).
        unsafe {
            self.k.pglue_conv_ar_gate288_vmsg_kernel(
                self.st,
                cfg([t.div_ceil(CONV_AR_ROWS), 1, 1], 288, 0),
                PglueConvArGate288VmsgKernelArgs {
                    ax: b(ax),
                    cw: b(cw),
                    state: b(state),
                    st_slot_stride: (NVISITS * 3 * SLD) as i64,
                    R: r,
                    hin: b(hin),
                    block_start: i32::from(blk_out.is_some()),
                    blk_out: opt(blk_out),
                    partial32: b(partial),
                    s: *s,
                    query: b(query),
                    ar_out: b(ar_out),
                    eps: EPS,
                    msrc0: b(msrc0),
                    msrc1: b(msrc1),
                    mw: b(mw),
                    gin: b(gin),
                    T: i(t),
                },
            )
        }
    }

    /// `ink::rmsnorm_f32out_kernel<<<T, 256>>>(in, w, out, EPS)`.
    pub fn rmsnorm_f32out(&self, t: usize, x: &DevBuf<f32>, w: &DevBuf<f32>, out: &DevBuf<f32>) -> Result<()> {
        need(x, t * H, "norm in")?;
        need(w, H, "norm w")?;
        need(out, t * H, "norm out")?;
        // SAFETY: block t reads row t of `x` (stride H) and w, writes row t of out (checked).
        unsafe {
            self.k.ink_rmsnorm_f32out_kernel(
                self.st,
                cfg([t, 1, 1], 256, 0),
                InkRmsnormF32outKernelArgs {
                    r#in: b(x),
                    w: b(w),
                    out: b(out),
                    eps: EPS,
                },
            )
        }
    }

    /// `glue::gather_rows_bf16_kernel<<<nlog, 256>>>(h32, logit_rows, hrows)`.
    pub fn gather_rows(
        &self,
        nlog: usize,
        t: usize,
        h: &DevBuf<f32>,
        rows: &DevBuf<i32>,
        out: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(h, t * H, "gather h")?;
        need(rows, nlog, "gather rows")?;
        need(out, nlog * H, "gather out")?;
        // SAFETY: block r reads rows[r] (< T: built by meta::build_call from the call's rows) and that row of h,
        // writes out row r (nlog x H checked).
        unsafe {
            self.k.glue_gather_rows_bf16_kernel(
                self.st,
                cfg([nlog, 1, 1], 256, 0),
                GlueGatherRowsBf16KernelArgs {
                    h: b(h),
                    rows: b(rows),
                    out: b(out),
                },
            )
        }
    }

    /// `glue::softcap_kernel<<<(ceil(V / 1024), nlog), 256>>>(logits, nlog, VOCAB_PAD, VOCAB)`.
    pub fn softcap(&self, nlog: usize, lg: &DevBuf<f32>) -> Result<()> {
        need(lg, nlog * VOCAB_PAD, "logits")?;
        // SAFETY: 2D grid (float4 blocks along vocab, rows along y) over nlog x VOCAB valid entries (checked).
        unsafe {
            self.k.glue_softcap_kernel(
                self.st,
                cfg([VOCAB.div_ceil(4 * 256), nlog, 1], 256, 0),
                GlueSoftcapKernelArgs {
                    lg: b(lg),
                    rows: i(nlog),
                    ld: i(VOCAB_PAD),
                    vocab: i(VOCAB),
                },
            )
        }
    }

    /// `ink::argmax_rows_kernel<<<nlog, 1024>>>(logits, VOCAB_PAD, VOCAB, amax)`.
    pub fn argmax(&self, nlog: usize, lg: &DevBuf<f32>, out: &DevBuf<i32>) -> Result<()> {
        need(lg, nlog * VOCAB_PAD, "logits")?;
        need(out, nlog, "argmax out")?;
        // SAFETY: block r reads row r (VOCAB valid entries, stride VOCAB_PAD), writes out[r] (checked).
        unsafe {
            self.k.ink_argmax_rows_kernel(
                self.st,
                cfg([nlog, 1, 1], 1024, 0),
                InkArgmaxRowsKernelArgs {
                    lg: b(lg),
                    ld: i(VOCAB_PAD),
                    vocab: i(VOCAB),
                    out: b(out),
                },
            )
        }
    }

    /// `ink::sample_rows_kernel<<<nlog, 1024>>>(SampleArgs)` with the engine mapping (`Engine::sample`): row r = logit
    /// row r of the last call, call row `lrow[r]`, controls `par[seg[call row]]`, position `pos[call row] + 1`; the
    /// sampled tokens into `out` (`engine/native/src/sample_kernels.cuh`).
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        nlog: usize,
        t: usize,
        nseg: usize,
        lg: &DevBuf<f32>,
        par: &DevBuf<ink::SampleParam>,
        lrow: &DevBuf<i32>,
        rows: (&DevBuf<i32>, &DevBuf<i32>),
        out: &DevBuf<i32>,
    ) -> Result<()> {
        let (pos, seg) = rows;
        need(lg, nlog * VOCAB_PAD, "logits")?;
        need(par, nseg, "sampling controls")?;
        need(lrow, nlog, "logit rows")?;
        need(pos, t, "positions")?;
        need(seg, t, "segments")?;
        need(out, nlog, "sampled out")?;
        let a = ink::SampleArgs {
            lg: lg.ptr(),
            ld: i(VOCAB_PAD),
            vocab: i(VOCAB),
            par: par.ptr(),
            lrow: lrow.ptr(),
            call_pos: pos.ptr(),
            call_seg: seg.ptr(),
            t_lrow: DevPtr::zeroed(),
            t_par: DevPtr::zeroed(),
            t_pos: DevPtr::zeroed(),
            out: out.ptr(),
            stats: DevPtr::zeroed(),
            keys: DevPtr::zeroed(),
        };
        // SAFETY: block r < nlog reads lrow[r] (a call row below T: the last call's logit rows), pos / seg of that row
        // (T checked), par[seg] (seg < nseg: the call's segment index; nseg checked), logit row r (VOCAB valid entries,
        // stride VOCAB_PAD, checked), writes out[r] (checked); stats / keys are null. The buffers outlive the launch.
        unsafe { self.k.ink_sample_rows_kernel(self.st, cfg([nlog, 1, 1], 1024, 0), &a) }
    }

    /// `ink::sample_leviathan_kernel<<<nlog, 1024>>>(LeviathanArgs)`: exact Leviathan rejection + residual sampling
    /// over a sparse top-16 drafter proposal distribution (`[slots][7][16]`).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_leviathan(
        &self,
        nlog: usize,
        t: usize,
        nseg: usize,
        max_slots: usize,
        lg: &DevBuf<f32>,
        par: &DevBuf<ink::SampleParam>,
        lrow: &DevBuf<i32>,
        rows: (&DevBuf<i32>, &DevBuf<i32>, &DevBuf<i32>, &DevBuf<i32>),
        drafts: &DevBuf<i32>,
        prop_ids: &DevBuf<i32>,
        prop_w: &DevBuf<u32>,
        out: &DevBuf<i32>,
    ) -> Result<()> {
        let (pos, seg, row0, slot) = rows;
        need(lg, nlog * VOCAB_PAD, "logits")?;
        need(par, nseg, "sampling controls")?;
        need(lrow, nlog, "logit rows")?;
        need(pos, t, "positions")?;
        need(seg, t, "segments")?;
        need(row0, nseg, "row0")?;
        need(slot, nseg, "slot")?;
        need(drafts, max_slots * 7, "drafts")?;
        need(prop_ids, max_slots * 7 * 16, "proposal ids")?;
        need(prop_w, max_slots * 7 * 16, "proposal weights")?;
        need(out, nlog, "sampled out")?;
        let a = ink::LeviathanArgs {
            lg: lg.ptr(),
            ld: i(VOCAB_PAD),
            vocab: i(VOCAB),
            par: par.ptr(),
            lrow: lrow.ptr(),
            call_pos: pos.ptr(),
            call_seg: seg.ptr(),
            call_row0: row0.ptr(),
            call_slot: slot.ptr(),
            nseg: i(nseg),
            T: i(t),
            drafts: drafts.ptr(),
            prop_ids: prop_ids.ptr(),
            prop_w: prop_w.ptr(),
            out: out.ptr(),
        };
        // SAFETY: all buffer extents are checked above; block r < nlog reads lrow[r] < T, seg < nseg, slot < max_slots,
        // and drafts/prop_ids/prop_w indexed by slot * 7 + j (j < 7).
        unsafe {
            self.k
                .ink_sample_leviathan_kernel(self.st, cfg([nlog, 1, 1], 1024, 0), &a)
        }
    }

    /// `ink::sample_rows_kernel<<<rows, 1024>>>(SampleArgs)` with the explicit test mapping (`spec.cpp` `test_rows`):
    /// row r reads logit row `t_lrow[r]`, controls `par[t_par[r]]`, position `t_pos[r]`; writes `out[r]`, and
    /// optionally `stats[r * 8 .. (r + 1) * 8]` and `keys[r * VOCAB .. (r + 1) * VOCAB]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_test(
        &self,
        rows: usize,
        lg_rows: usize,
        npar: usize,
        lg: &DevBuf<f32>,
        par: &DevBuf<ink::SampleParam>,
        t_lrow: &DevBuf<i32>,
        t_par: &DevBuf<i32>,
        t_pos: &DevBuf<i32>,
        out: &DevBuf<i32>,
        stats: Option<&DevBuf<u64>>,
        keys: Option<&DevBuf<u32>>,
    ) -> Result<()> {
        need(lg, lg_rows * VOCAB_PAD, "test logits")?;
        need(par, npar, "test sampling controls")?;
        need(t_lrow, rows, "test lrow")?;
        need(t_par, rows, "test par")?;
        need(t_pos, rows, "test pos")?;
        need(out, rows, "test sampled out")?;
        if let Some(st) = stats {
            need(st, rows * 8, "test stats")?;
        }
        if let Some(ks) = keys {
            need(ks, rows * VOCAB, "test keys")?;
        }
        let a = ink::SampleArgs {
            lg: lg.ptr(),
            ld: i(VOCAB_PAD),
            vocab: i(VOCAB),
            par: par.ptr(),
            lrow: DevPtr::zeroed(),
            call_pos: DevPtr::zeroed(),
            call_seg: DevPtr::zeroed(),
            t_lrow: t_lrow.ptr(),
            t_par: t_par.ptr(),
            t_pos: t_pos.ptr(),
            out: out.ptr(),
            stats: stats.map_or_else(DevPtr::zeroed, DevBuf::ptr),
            keys: keys.map_or_else(DevPtr::zeroed, DevBuf::ptr),
        };
        // SAFETY: block r < rows reads t_lrow[r] (< lg_rows, checked by caller), t_par[r] (< npar, checked by caller),
        // t_pos[r], logit row t_lrow[r] (VOCAB valid entries, stride VOCAB_PAD, lg_rows checked), par[t_par[r]] (npar
        // checked), writes out[r] (checked), and stats[8 r..8 r+8] / keys[VOCAB r..VOCAB (r+1)] when present (checked).
        // The buffers outlive the launch.
        unsafe { self.k.ink_sample_rows_kernel(self.st, cfg([rows, 1, 1], 1024, 0), &a) }
    }

    // ------------------------------------------------------------------------------------------------ norms

    /// `glue::rmsnorm_kernel<bf16 | float><<<T, 256>>>(in, ldi, w, out, H, EPS)`.
    pub fn rmsnorm_bf16(
        &self,
        t: usize,
        x: &DevBuf<Bf16>,
        ldi: usize,
        w: &DevBuf<f32>,
        out: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(x, (t - 1) * ldi + H, "norm in")?;
        need(w, H, "norm w")?;
        need(out, t * H, "norm out")?;
        // SAFETY: block t reads H values at x + t ldi (extent checked), w, writes row t of out (stride H, checked).
        unsafe {
            self.k.glue_rmsnorm_kernel_bf16(
                self.st,
                cfg([t, 1, 1], 256, 0),
                GlueRmsnormKernelBf16Args {
                    r#in: b(x),
                    ldi: i(ldi),
                    w: b(w),
                    out: b(out),
                    ldo: i(H),
                    eps: EPS,
                },
            )
        }
    }

    /// See [`Dev::rmsnorm_bf16`]; fp32 input.
    pub fn rmsnorm_f32(
        &self,
        t: usize,
        x: &DevBuf<f32>,
        ldi: usize,
        w: &DevBuf<f32>,
        out: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(x, (t - 1) * ldi + H, "norm in")?;
        need(w, H, "norm w")?;
        need(out, t * H, "norm out")?;
        // SAFETY: as rmsnorm_bf16.
        unsafe {
            self.k.glue_rmsnorm_kernel_float(
                self.st,
                cfg([t, 1, 1], 256, 0),
                GlueRmsnormKernelFloatArgs {
                    r#in: b(x),
                    ldi: i(ldi),
                    w: b(w),
                    out: b(out),
                    ldo: i(H),
                    eps: EPS,
                },
            )
        }
    }

    /// `glue::rmsnorm_g_kernel<<<(T, 3), 256>>>(NormG, ldi, H, EPS)`: three norms of row-strided fp32 sources.
    pub fn rmsnorm_g3(
        &self,
        t: usize,
        src: [&DevBuf<f32>; 3],
        ldi: usize,
        w: [&DevBuf<f32>; 3],
        out: [&DevBuf<Bf16>; 3],
    ) -> Result<()> {
        let mut a = glue::NormG::zeroed();
        for y in 0..3 {
            need(src[y], (t - 1) * ldi + H, "norm3 in")?;
            need(w[y], H, "norm3 w")?;
            need(out[y], t * H, "norm3 out")?;
            a.r#in[y] = src[y].ptr();
            a.w[y] = w[y].ptr();
            a.out[y] = out[y].ptr();
        }
        // SAFETY: block (t, y) reads H values at in[y] + t ldi, w[y], writes row t of out[y] (all checked above; the
        // fourth entry is unused at gridDim.y = 3). The DevPtrs name engine buffers that outlive the launch.
        unsafe {
            self.k.glue_rmsnorm_g_kernel(
                self.st,
                cfg([t, 3, 1], 256, 0),
                GlueRmsnormGKernelArgs {
                    a,
                    ldi: i(ldi),
                    ldo: i(H),
                    eps: EPS,
                },
            )
        }
    }

    /// `pglue::norm_rows_kernel<<<(ceil(T / 4), G), 128>>>(NormG, ldi, H, EPS, T)`, one warp per row (T > 64; same
    /// bits): G = 3 for `norm3` ([`Dev::rmsnorm_g3`]), G = 1 for the message visits' `mlp_norm` ([`Dev::rmsnorm_f32`]).
    pub fn norm_rows(
        &self,
        t: usize,
        src: &[&DevBuf<f32>],
        ldi: usize,
        w: &[&DevBuf<f32>],
        out: &[&DevBuf<Bf16>],
    ) -> Result<()> {
        let g = src.len();
        if g == 0 || g > 3 || w.len() != g || out.len() != g {
            return Err(CudaError::Bounds {
                what: "norm_rows",
                detail: format!("{g} sources, {} weights, {} outputs", w.len(), out.len()),
            });
        }
        let mut a = glue::NormG::zeroed();
        for y in 0..g {
            need(src[y], (t - 1) * ldi + H, "norm_rows in")?;
            need(w[y], H, "norm_rows w")?;
            need(out[y], t * H, "norm_rows out")?;
            a.r#in[y] = src[y].ptr();
            a.w[y] = w[y].ptr();
            a.out[y] = out[y].ptr();
        }
        // SAFETY: warp (t, y), t < T, reads H values at in[y] + t ldi and w[y], writes row t of out[y] (stride H): all
        // checked above; entries at y >= gridDim.y = G are unused.
        unsafe {
            self.k.pglue_norm_rows_kernel(
                self.st,
                cfg([t.div_ceil(WARP_ROWS), g, 1], 32 * WARP_ROWS, 0),
                PglueNormRowsKernelArgs {
                    a,
                    ldi: i(ldi),
                    ldo: i(H),
                    eps: EPS,
                    T: i(t),
                },
            )
        }
    }

    // ------------------------------------------------------------------------------------------------ message gates

    /// Production `glue::mg_in_kernel<<<(T, G), 256>>>(hid, ldh, msg, ldm, gin, EPS, hs, H, gi)`.
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn mg_in(
        &self,
        t: usize,
        g: usize,
        hid: &DevBuf<f32>,
        hs: usize,
        ldh: usize,
        msg: &DevBuf<f32>,
        ldm: usize,
        gin: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(hid, (g - 1) * hs + (t - 1) * ldh + H, "mg hid")?;
        need(msg, (g - 1) * H + (t - 1) * ldm + H, "mg msg")?;
        need(gin, g * t * 2 * H, "mg gin")?;
        // SAFETY: block (t, y) reads H values at hid + y hs + t ldh and msg + y H + t ldm, writes gin[y][t][0..2H]
        // (extents checked).
        unsafe {
            self.k.glue_mg_in_kernel(
                self.st,
                cfg([t, g, 1], 256, 0),
                GlueMgInKernelArgs {
                    hid: b(hid),
                    ldh: i(ldh),
                    msg: b(msg),
                    ldm: i(ldm),
                    g: b(gin),
                    eps: EPS,
                    hs: hs as i64,
                    ms: H as i64,
                    gs: (t * 2 * H) as i64,
                },
            )
        }
    }

    /// Reference `glue::mg_in_kernel<<<T, 256>>>(hid, ldh, msg, ldm, gin, EPS)` for one gate.
    pub fn mg_in_one(
        &self,
        t: usize,
        hid: &DevBuf<f32>,
        ldh: usize,
        msg: &DevBuf<f32>,
        ldm: usize,
        gin: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(hid, (t - 1) * ldh + H, "mg hid")?;
        need(msg, (t - 1) * ldm + H, "mg msg")?;
        need(gin, t * 2 * H, "mg gin")?;
        // SAFETY: as mg_in with one gate (strides 0).
        unsafe {
            self.k.glue_mg_in_kernel(
                self.st,
                cfg([t, 1, 1], 256, 0),
                GlueMgInKernelArgs {
                    hid: b(hid),
                    ldh: i(ldh),
                    msg: b(msg),
                    ldm: i(ldm),
                    g: b(gin),
                    eps: EPS,
                    hs: 0,
                    ms: 0,
                    gs: 0,
                },
            )
        }
    }

    /// `glue::bf16_to_f32_kernel<<<(n + 255) / 256, 256>>>(a, b, n)`.
    pub fn bf16_to_f32(&self, n: usize, a: &DevBuf<Bf16>, out: &DevBuf<f32>) -> Result<()> {
        need(a, n, "bf16 in")?;
        need(out, n, "f32 out")?;
        // SAFETY: thread k < n reads a[k], writes out[k] (checked).
        unsafe {
            self.k.glue_bf16_to_f32_kernel(
                self.st,
                cfg([n.div_ceil(256), 1, 1], 256, 0),
                GlueBf16ToF32KernelArgs {
                    a: b(a),
                    b: b(out),
                    n: i(n),
                },
            )
        }
    }

    /// `glue::mg_out_kernel<<<grid1(T H), 256>>>(f, g, hid, ldh, msg, ldm, out, ldo, T)`.
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn mg_out(
        &self,
        t: usize,
        fp: &DevBuf<f32>,
        gp: &DevBuf<f32>,
        hid: &DevBuf<f32>,
        ldh: usize,
        msg: &DevBuf<f32>,
        ldm: usize,
        out: &DevBuf<f32>,
        ldo: usize,
    ) -> Result<()> {
        need(fp, t * H, "mg f")?;
        need(gp, t * H, "mg g")?;
        need(hid, (t - 1) * ldh + H, "mg hid")?;
        need(msg, (t - 1) * ldm + H, "mg msg")?;
        need(out, (t - 1) * ldo + H, "mg out")?;
        // SAFETY: grid-stride over T x H: reads f, g (stride H), hid, msg at their strides, writes out (stride ldo);
        // all extents checked.
        unsafe {
            self.k.glue_mg_out_kernel(
                self.st,
                cfg([grid1(t * H), 1, 1], 256, 0),
                GlueMgOutKernelArgs {
                    fp: b(fp),
                    gp: b(gp),
                    hid: b(hid),
                    ldh: i(ldh),
                    msg: b(msg),
                    ldm: i(ldm),
                    out: b(out),
                    ldo: i(ldo),
                    T: i(t),
                },
            )
        }
    }

    /// The message-gate tail of gates `gates` (1..4): `pglue::mg_tail_rows_kernel<<<(H / 128, ceil(T / 32), G), 128>>>`
    /// for every row count
    /// (delta 2, P3: decode rows too; `glue::mg_fused_kernel` deleted).
    #[allow(dead_code)]
    pub fn mg_tail(&self, t: usize, gates: &[MgGate<'_>], ldh: usize, ldm: usize, ldo: usize) -> Result<()> {
        let g = gates.len();
        if t == 0 || !(1..=4).contains(&g) {
            return Err(CudaError::Bounds {
                what: "mg_tail",
                detail: format!("{t} rows, {g} gates (1..4)"),
            });
        }
        let mut m = glue::MgTail::zeroed();
        for (y, x) in gates.iter().enumerate() {
            need(x.gd, t * MG_RANK, "mg_tail gd")?;
            need(x.forget_t, MG_RANK * H, "mg_tail FT")?;
            need(x.gain_t, MG_RANK * H, "mg_tail GT")?;
            need(x.hid, (t - 1) * ldh + H, "mg_tail hid")?;
            need(x.msg, (t - 1) * ldm + H, "mg_tail msg")?;
            need(x.out, (t - 1) * ldo + H, "mg_tail out")?;
            m.gd[y] = x.gd.ptr();
            m.FT[y] = x.forget_t.ptr();
            m.GT[y] = x.gain_t.ptr();
            m.hid[y] = x.hid.ptr();
            m.msg[y] = x.msg.ptr();
            m.out[y] = x.out.ptr();
        }
        // SAFETY: gate y (blockIdx.z < G) reads gd[y] [T][64] bf16, FT[y] / GT[y] [64][H] fp32, hid[y] / msg[y] rows
        // at ldh / ldm and writes out[y] rows at ldo, T rows (a block of 32 rows stops at T): every extent is checked
        // above; the DevPtrs name engine buffers that outlive the launch.
        unsafe {
            self.k.pglue_mg_tail_rows_kernel(
                self.st,
                cfg([H / 128, t.div_ceil(MG_ROWS), g], 128, 0),
                PglueMgTailRowsKernelArgs {
                    m,
                    T: i(t),
                    ldh: i(ldh),
                    ldm: i(ldm),
                    ldo: i(ldo),
                },
            )
        }
    }

    /// G7: the message-gate tail of a prefill row group on bf16 tensor cores,
    /// `pglue::mg_tail_tc_kernel<<<(H / 128, ceil(n / 32), G), 128>>>(MgTc, n, ldh, ldm, ldo)` (bit-changing against
    /// [`Dev::mg_tail`], accepted by the head; decode row groups stay on `mg_tail`).
    pub fn mg_tail_tc(&self, n: usize, gates: &[MgTcGate<'_>], ldh: usize, ldm: usize, ldo: usize) -> Result<()> {
        let g = gates.len();
        if n == 0 || g == 0 || g > 4 || !ldh.is_multiple_of(4) || !ldm.is_multiple_of(4) || !ldo.is_multiple_of(4) {
            return Err(CudaError::Bounds {
                what: "mg_tail_tc",
                detail: format!(
                    "{n} rows, G = {g}, ldh {ldh}, ldm {ldm}, ldo {ldo} (float2 reads / float4 writes need aligned strides)"
                ),
            });
        }
        let mut m = pglue::MgTc::zeroed();
        for (y, x) in gates.iter().enumerate() {
            need(x.gd, n * MG_RANK, "mg_tail_tc gd")?;
            need(x.w, 2 * H * MG_RANK, "mg_tail_tc W")?;
            need(x.hid, (n - 1) * ldh + H, "mg_tail_tc hid")?;
            need(x.msg, (n - 1) * ldm + H, "mg_tail_tc msg")?;
            need(x.out, (n - 1) * ldo + H, "mg_tail_tc out")?;
            m.gd[y] = x.gd.ptr();
            m.W[y] = x.w.ptr();
            m.hid[y] = x.hid.ptr();
            m.msg[y] = x.msg.ptr();
            m.out[y] = x.out.ptr();
        }
        // SAFETY: gate y (blockIdx.z < G) reads gd[y] [n][64] bf16 (rows past n read as zero), W[y] bf16 [2][H][64]
        // (16-byte loads), hid[y] / msg[y] rows at ldh / ldm as float2, and writes out[y] rows at ldo as float4:
        // every extent is checked above, strides are aligned, and every base is an arena take (256-byte aligned)
        // plus an aligned element offset.
        unsafe {
            self.k.pglue_mg_tail_tc_kernel(
                self.st,
                cfg([H / 128, n.div_ceil(TC_ROWS), g], 128, 0),
                PglueMgTailTcKernelArgs {
                    m,
                    T: i(n),
                    ldh: i(ldh),
                    ldm: i(ldm),
                    ldo: i(ldo),
                },
            )
        }
    }

    /// Track 3A-7: `pglue::mg_tail_tc_vmsg_kernel<<<(H / 128, ceil(n / 32), G), 128>>>`: reads L2-hot bf16
    /// `msrc0/1` + `mw` instead of fp32 `a.msgs`, bit-identical to `mg_tail_tc` and `mg_tail_tc_dec`.
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn mg_tail_tc_vmsg(
        &self,
        n: usize,
        gates: &[MgTcGate<'_>],
        msrc0: &DevBuf<Bf16>,
        msrc1: &DevBuf<Bf16>,
        mw: &DevBuf<f32>,
        ldh: usize,
        ldo: usize,
        q_off: usize,
    ) -> Result<()> {
        let g = gates.len();
        if n == 0 || g == 0 || g > 4 || !ldh.is_multiple_of(4) || !ldo.is_multiple_of(4) || q_off + g > 4 {
            return Err(CudaError::Bounds {
                what: "mg_tail_tc_vmsg",
                detail: format!("{n} rows, G = {g}, ldh {ldh}, ldo {ldo}, q_off {q_off}"),
            });
        }
        need(msrc0, n * H, "mg_tail_tc_vmsg msrc0")?;
        need(msrc1, n * H, "mg_tail_tc_vmsg msrc1")?;
        need(mw, n * 16, "mg_tail_tc_vmsg mw")?;
        let mut m = pglue::MgTcVmsg::zeroed();
        m.msrc0 = msrc0.ptr();
        m.msrc1 = msrc1.ptr();
        m.mw = mw.ptr();
        for (y, x) in gates.iter().enumerate() {
            need(x.gd, n * MG_RANK, "mg_tail_tc_vmsg gd")?;
            need(x.w, 2 * H * MG_RANK, "mg_tail_tc_vmsg W")?;
            need(x.hid, (n - 1) * ldh + H, "mg_tail_tc_vmsg hid")?;
            need(x.out, (n - 1) * ldo + H, "mg_tail_tc_vmsg out")?;
            m.gd[y] = x.gd.ptr();
            m.W[y] = x.w.ptr();
            m.hid[y] = x.hid.ptr();
            m.out[y] = x.out.ptr();
        }
        // SAFETY: gate y < G reads gd/W/hid (aligned float4), shared msrc0/1 (aligned uint4), mw[n*16], writes out (all checked).
        unsafe {
            self.k.pglue_mg_tail_tc_vmsg_kernel(
                self.st,
                cfg([H / 128, n.div_ceil(TC_ROWS), g], 128, 0),
                PglueMgTailTcVmsgKernelArgs {
                    m,
                    T: i(n),
                    ldh: i(ldh),
                    ldo: i(ldo),
                    q_off: i(q_off),
                },
            )
        }
    }

    /// `pglue::mg_down_dec_kernel<MT><<<(4, 18, G), 256>>>`: fused decode gate-down projection for all G gates (n <= 128).
    pub fn mg_down_dec(&self, n: usize, gates: &[MgDownGate<'_>], ws: &DevBuf<f32>, cnt: &DevBuf<i32>) -> Result<()> {
        let g = gates.len();
        if n == 0 || n > 128 || g == 0 || g > 4 {
            return Err(CudaError::Bounds {
                what: "mg_down_dec",
                detail: format!("{n} rows (1..=128), G = {g}"),
            });
        }
        need(ws, g * 18 * n * MG_RANK, "mg_down_dec ws")?;
        need(cnt, g * 4, "mg_down_dec cnt")?;
        let mut m = pglue::MgDown::zeroed();
        for (y, x) in gates.iter().enumerate() {
            need(x.x, n * 2 * H, "mg_down_dec x")?;
            need(x.w, MG_RANK * 2 * H, "mg_down_dec W")?;
            need(x.out, n * MG_RANK, "mg_down_dec out")?;
            m.x[y] = x.x.ptr();
            m.W[y] = x.w.ptr();
            m.out[y] = x.out.ptr();
        }
        m.ws = ws.ptr();
        m.cnt = cnt.ptr();
        let c = cfg([4, 18, g], 256, 0);
        // SAFETY: grid (4, 18, G) writes G * 18 * n * 64 floats into ws and increments/resets cnt[0..4*G], writing out[z][0..n*64] (all checked).
        unsafe {
            if n <= 16 {
                self.k
                    .pglue_mg_down_dec_kernel_1(self.st, c, PglueMgDownDecKernel1Args { m, M: i(n) })
            } else if n <= 32 {
                self.k
                    .pglue_mg_down_dec_kernel_2(self.st, c, PglueMgDownDecKernel2Args { m, M: i(n) })
            } else if n <= 64 {
                self.k
                    .pglue_mg_down_dec_kernel_4(self.st, c, PglueMgDownDecKernel4Args { m, M: i(n) })
            } else {
                self.k
                    .pglue_mg_down_dec_kernel_8(self.st, c, PglueMgDownDecKernel8Args { m, M: i(n) })
            }
        }
    }

    /// `pglue::mg_tail_tc_dec_kernel<MT><<<(H / 64, 1, G), 128>>>`: decode BF16 tensor-core message-gate tail (n <= 64).
    pub fn mg_tail_tc_dec(&self, n: usize, gates: &[MgTcGate<'_>], ldh: usize, ldm: usize, ldo: usize) -> Result<()> {
        let g = gates.len();
        if n == 0 || n > 64 || g == 0 || g > 4 || [ldh, ldm, ldo].iter().any(|l| !l.is_multiple_of(2)) {
            return Err(CudaError::Bounds {
                what: "mg_tail_tc_dec",
                detail: format!("{n} rows (1..=64), G = {g}, ldh {ldh}, ldm {ldm}, ldo {ldo}"),
            });
        }
        let mut m = pglue::MgTc::zeroed();
        for (y, x) in gates.iter().enumerate() {
            need(x.gd, n * MG_RANK, "mg_tail_tc_dec gd")?;
            need(x.w, 2 * H * MG_RANK, "mg_tail_tc_dec W")?;
            need(x.hid, (n - 1) * ldh + H, "mg_tail_tc_dec hid")?;
            need(x.msg, (n - 1) * ldm + H, "mg_tail_tc_dec msg")?;
            need(x.out, (n - 1) * ldo + H, "mg_tail_tc_dec out")?;
            m.gd[y] = x.gd.ptr();
            m.W[y] = x.w.ptr();
            m.hid[y] = x.hid.ptr();
            m.msg[y] = x.msg.ptr();
            m.out[y] = x.out.ptr();
        }
        let c = cfg([H / 64, 1, g], 128, 0);
        // SAFETY: identical contract to mg_tail_tc with N = 64 cols/CTA and n <= 64 rows (all extents & even strides checked).
        unsafe {
            if n <= 16 {
                self.k.pglue_mg_tail_tc_dec_kernel_1(
                    self.st,
                    c,
                    PglueMgTailTcDecKernel1Args {
                        m,
                        T: i(n),
                        ldh: i(ldh),
                        ldm: i(ldm),
                        ldo: i(ldo),
                    },
                )
            } else if n <= 32 {
                self.k.pglue_mg_tail_tc_dec_kernel_2(
                    self.st,
                    c,
                    PglueMgTailTcDecKernel2Args {
                        m,
                        T: i(n),
                        ldh: i(ldh),
                        ldm: i(ldm),
                        ldo: i(ldo),
                    },
                )
            } else {
                self.k.pglue_mg_tail_tc_dec_kernel_4(
                    self.st,
                    c,
                    PglueMgTailTcDecKernel4Args {
                        m,
                        T: i(n),
                        ldh: i(ldh),
                        ldm: i(ldm),
                        ldo: i(ldo),
                    },
                )
            }
        }
    }

    /// G7: one gate's tensor-core weights from its fp32 transposed controller weights,
    /// `pglue::mg_tc_weights_kernel<<<2 H, 64>>>(FT, GT, W)`: W = bf16 [forget; gain] [2][H][64].
    pub fn mg_tc_weights(&self, ft: &DevBuf<f32>, gt: &DevBuf<f32>, w: &DevBuf<Bf16>) -> Result<()> {
        need(ft, GATE_RANK * H, "tc FT")?;
        need(gt, GATE_RANK * H, "tc GT")?;
        need(w, 2 * H * GATE_RANK, "tc W")?;
        // SAFETY: block n < 2H, thread k < 64 reads FT or GT [64][H] at (k, n mod H), writes W[n][k] (all checked).
        unsafe {
            self.k.pglue_mg_tc_weights_kernel(
                self.st,
                cfg([2 * H, 1, 1], GATE_RANK, 0),
                PglueMgTcWeightsKernelArgs {
                    FT: b(ft),
                    GT: b(gt),
                    W: b(w),
                },
            )
        }
    }

    /// Transposed relative projection `pglue::rel_proj_t_kernel<<<1024, 16>>>(src, dst)`: `[16][1024]` -> `[1024][16]`.
    pub fn rel_proj_t(&self, src: &DevBuf<Bf16>, dst: &DevBuf<Bf16>) -> Result<()> {
        need(src, DREL * REL_EXT, "rel_proj")?;
        need(dst, REL_EXT * DREL, "rel_proj_t")?;
        // SAFETY: block d < 1024, thread j < 16 reads src[j * 1024 + d] and writes dst[d * 16 + j] (all checked).
        unsafe {
            self.k.pglue_rel_proj_t_kernel(
                self.st,
                cfg([REL_EXT, 1, 1], DREL, 0),
                PglueRelProjTKernelArgs {
                    rel_proj: b(src),
                    rel_proj_t: b(dst),
                },
            )
        }
    }

    // ------------------------------------------------------------------------------------------------ projections

    /// `ink::proj_split_kernel<<<n, 256>>>(f, ld, dyn, q, g, r, s, k, kd, v, vd)` (D1): rows of the fused decode
    /// projection output `[q | gate | r | sink | k (| k_dyn) | v (| v_dyn)]` (row stride `ld`) into the separate
    /// buffers, each already offset to the group's first row.
    pub fn proj_split(&self, n: usize, f: &DevBuf<Bf16>, ld: usize, full: bool, o: &ProjOut<'_>) -> Result<()> {
        let nd = if full { 32 } else { 0 };
        let width = 2 * QD + NH * DREL + NH + 2 * (KD + nd);
        if n == 0 || ld < width {
            return Err(CudaError::Bounds {
                what: "proj_split",
                detail: format!("{n} rows, ld {ld} < {width}"),
            });
        }
        need(f, (n - 1) * ld + width, "proj_split in")?;
        for (x, w, what) in [
            (o.q, QD, "proj_split q"),
            (o.g, QD, "proj_split gate"),
            (o.r, NH * DREL, "proj_split r"),
            (o.s, NH, "proj_split sink"),
            (o.k, KD, "proj_split k"),
            (o.kd, nd, "proj_split k_dyn"),
            (o.v, KD, "proj_split v"),
            (o.vd, nd, "proj_split v_dyn"),
        ] {
            need(x, n * w, what)?;
        }
        // SAFETY: block t < n copies row t of f (ld checked >= the row's parts) into row t of each output at its own
        // width (k_dyn / v_dyn width 0 unless dyn: not touched then); every extent checked. 16-byte copies: every
        // width and offset is a multiple of 8 elements and the buffers are 256-byte aligned arena takes plus row
        // offsets of those widths.
        unsafe {
            self.k.ink_proj_split_kernel(
                self.st,
                cfg([n, 1, 1], 256, 0),
                InkProjSplitKernelArgs {
                    f: b(f),
                    ld: i(ld),
                    r#dyn: i(usize::from(full)),
                    q: b(o.q),
                    g: b(o.g),
                    r: b(o.r),
                    s: b(o.s),
                    k: b(o.k),
                    kd: b(o.kd),
                    v: b(o.v),
                    vd: b(o.vd),
                },
            )
        }
    }

    // ------------------------------------------------------------------------------------------------ q / k / v post

    /// Track 3A-3: zero-split QKV post reading directly from fused `qkvf` (`a.relk`, row stride `nt`).
    /// Q warps copy `gate`/`sink` into contiguous buffers so `prefill_attn` stays untouched; KV warps run an 8-row
    /// sliding window in registers. Bit-identical to `proj_split + qkv_post_rows`.
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn qkv_post_fused_rows(
        &self,
        t: usize,
        qkvf: &DevBuf<Bf16>,
        nt: usize,
        gate: &DevBuf<Bf16>,
        sink: &DevBuf<Bf16>,
        seg_len: &DevBuf<i32>,
        a: QkvPost<'_>,
    ) -> Result<()> {
        need(qkvf, t * nt, "qkvf")?;
        need(a.q, t * QD, "q")?;
        need(a.rel, t * NH * 16, "rel")?;
        need(gate, t * QD, "gate")?;
        need(sink, t * NH, "sink")?;
        need(a.kx, t * KD, "kx")?;
        need(a.vx, t * KD, "vx")?;
        need(a.pos, t, "pos")?;
        let c = cfg([t, 4, 1], 256, 0);
        if a.fp8 {
            // SAFETY: reads rows [0..T) of qkvf (stride nt, checked), conv states, scales and conv weights; writes q,
            // rel, gate, sink, conv_commit tail rows of kx/vx, and E4M3 K/V pages (all checked).
            unsafe {
                self.k.pglue_qkv_post_fused_rows_fp8_kernel(
                    self.st,
                    c,
                    PglueQkvPostFusedRowsFp8KernelArgs {
                        qkvf: b(qkvf),
                        nt: i(nt),
                        q_scale: b(a.q_scale),
                        r_scale: b(a.r_scale),
                        row_pos: b(a.pos),
                        full: i32::from(a.full),
                        T: i(t),
                        q: b(a.q),
                        rel: b(a.rel),
                        gate: b(gate),
                        sink: b(sink),
                        kx: b(a.kx),
                        vx: b(a.vx),
                        kst: b(a.kst),
                        vst: b(a.vst),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: a.rows,
                        seg_len: b(seg_len),
                        kw: b(a.kw),
                        vw: b(a.vw),
                        eps: EPS,
                        kpool: b(a.kpool),
                        vpool: b(a.vpool),
                        pt: b(a.pt),
                        pt_stride: i(crate::engine::meta::KPT),
                    },
                )
            }
        } else {
            let kpool = a.kpool.cast::<Bf16>()?;
            let vpool = a.vpool.cast::<Bf16>()?;
            // SAFETY: as FP8 variant above, writing bf16 K/V pages (all checked).
            unsafe {
                self.k.pglue_qkv_post_fused_rows_kernel(
                    self.st,
                    c,
                    PglueQkvPostFusedRowsKernelArgs {
                        qkvf: b(qkvf),
                        nt: i(nt),
                        q_scale: b(a.q_scale),
                        r_scale: b(a.r_scale),
                        row_pos: b(a.pos),
                        full: i32::from(a.full),
                        T: i(t),
                        q: b(a.q),
                        rel: b(a.rel),
                        gate: b(gate),
                        sink: b(sink),
                        kx: b(a.kx),
                        vx: b(a.vx),
                        kst: b(a.kst),
                        vst: b(a.vst),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: a.rows,
                        seg_len: b(seg_len),
                        kw: b(a.kw),
                        vw: b(a.vw),
                        eps: EPS,
                        kpool: b(&kpool),
                        vpool: b(&vpool),
                        pt: b(a.pt),
                        pt_stride: i(crate::engine::meta::KPT),
                    },
                )
            }
        }
    }

    /// `pglue::qkv_post_rows_kernel` (or `_fp8_kernel`)`<<<(T, 4), 256>>>` (prefill-carrying calls, production glue).
    pub fn qkv_post_rows(&self, t: usize, a: QkvPost<'_>) -> Result<()> {
        a.check(t)?;
        if a.fp8 {
            // SAFETY: row t (< T, checked extents of every row buffer) reads its projections and conv weights; for its
            // key/value it reads the slot's conv state (cst + slot * st_slot, kept by the engine for max_slots slots)
            // and writes E4M3 K/V bytes at page pt[seg * KPT + pos / 1024] (pages owned by the row's slot).
            unsafe {
                self.k.pglue_qkv_post_rows_fp8_kernel(
                    self.st,
                    cfg([t, 4, 1], 256, 0),
                    PglueQkvPostRowsFp8KernelArgs {
                        qproj: b(a.qproj),
                        q_scale: b(a.q_scale),
                        rproj: b(a.rproj),
                        r_scale: b(a.r_scale),
                        row_pos: b(a.pos),
                        full: i32::from(a.full),
                        q: b(a.q),
                        rel: b(a.rel),
                        kx: b(a.kx),
                        vx: b(a.vx),
                        kst: b(a.kst),
                        vst: b(a.vst),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: a.rows,
                        kw: b(a.kw),
                        vw: b(a.vw),
                        kdyn: b(a.kdyn),
                        vdyn: b(a.vdyn),
                        eps: EPS,
                        kpool: b(a.kpool),
                        vpool: b(a.vpool),
                        pt: b(a.pt),
                        pt_stride: i(crate::engine::meta::KPT),
                    },
                )
            }
        } else {
            let kpool = a.kpool.cast::<Bf16>()?;
            let vpool = a.vpool.cast::<Bf16>()?;
            // SAFETY: row t (< T, checked extents of every row buffer) reads its projections and conv weights; for its
            // key/value it reads the slot's conv state (cst + slot * st_slot, kept by the engine for max_slots slots)
            // and writes bf16 K/V at page pt[seg * KPT + pos / 1024] (pages owned by the row's slot or its ring).
            unsafe {
                self.k.pglue_qkv_post_rows_kernel(
                    self.st,
                    cfg([t, 4, 1], 256, 0),
                    PglueQkvPostRowsKernelArgs {
                        qproj: b(a.qproj),
                        q_scale: b(a.q_scale),
                        rproj: b(a.rproj),
                        r_scale: b(a.r_scale),
                        row_pos: b(a.pos),
                        full: i32::from(a.full),
                        q: b(a.q),
                        rel: b(a.rel),
                        kx: b(a.kx),
                        vx: b(a.vx),
                        kst: b(a.kst),
                        vst: b(a.vst),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: a.rows,
                        kw: b(a.kw),
                        vw: b(a.vw),
                        kdyn: b(a.kdyn),
                        vdyn: b(a.vdyn),
                        eps: EPS,
                        kpool: b(&kpool),
                        vpool: b(&vpool),
                        pt: b(a.pt),
                        pt_stride: i(crate::engine::meta::KPT),
                    },
                )
            }
        }
    }

    /// `glue::qkv_post_kernel` (or `_fp8_kernel`)`<<<(T, NH + KVH), HD>>>` (decode-only calls, production glue).
    #[allow(dead_code)]
    pub fn qkv_post(
        &self,
        t: usize,
        a: QkvPost<'_>,
        proj: &DevBuf<Bf16>,
        rel_logits: &DevBuf<Bf16>,
        kout: &DevBuf<Bf16>,
        vout: &DevBuf<Bf16>,
    ) -> Result<()> {
        a.check(t)?;
        need(rel_logits, t * NH * REL_EXT, "rel logits")?;
        need(kout, t * KVH * HD, "k")?;
        need(vout, t * KVH * HD, "v")?;
        if a.fp8 {
            // SAFETY: as qkv_post_rows (E4M3 global pool); also writes rel_logits [T][NH][1024] and k / v [T][KVH][HD]
            // (checked) and reads the layer's rel projection [16][1024].
            unsafe {
                self.k.glue_qkv_post_fp8_kernel(
                    self.st,
                    cfg([t, NH + KVH, 1], HD, 0),
                    GlueQkvPostFp8KernelArgs {
                        qproj: b(a.qproj),
                        q_scale: b(a.q_scale),
                        rproj: b(a.rproj),
                        r_scale: b(a.r_scale),
                        row_pos: b(a.pos),
                        full: i32::from(a.full),
                        proj: b(proj),
                        q: b(a.q),
                        rel: b(a.rel),
                        rel_logits: b(rel_logits),
                        kx: b(a.kx),
                        vx: b(a.vx),
                        kst: b(a.kst),
                        vst: b(a.vst),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: a.rows,
                        kw: b(a.kw),
                        vw: b(a.vw),
                        kdyn: b(a.kdyn),
                        vdyn: b(a.vdyn),
                        kout: b(kout),
                        vout: b(vout),
                        eps: EPS,
                        kpool: b(a.kpool),
                        vpool: b(a.vpool),
                        pt: b(a.pt),
                        pt_stride: i(crate::engine::meta::KPT),
                    },
                )
            }
        } else {
            let kpool = a.kpool.cast::<Bf16>()?;
            let vpool = a.vpool.cast::<Bf16>()?;
            // SAFETY: as qkv_post_rows (bf16 pool/ring); also writes rel_logits [T][NH][1024] and k / v [T][KVH][HD]
            // (checked) and reads the layer's rel projection [16][1024].
            unsafe {
                self.k.glue_qkv_post_kernel(
                    self.st,
                    cfg([t, NH + KVH, 1], HD, 0),
                    GlueQkvPostKernelArgs {
                        qproj: b(a.qproj),
                        q_scale: b(a.q_scale),
                        rproj: b(a.rproj),
                        r_scale: b(a.r_scale),
                        row_pos: b(a.pos),
                        full: i32::from(a.full),
                        proj: b(proj),
                        q: b(a.q),
                        rel: b(a.rel),
                        rel_logits: b(rel_logits),
                        kx: b(a.kx),
                        vx: b(a.vx),
                        kst: b(a.kst),
                        vst: b(a.vst),
                        st_slot_stride: (NVISITS * 3 * SLD) as i64,
                        R: a.rows,
                        kw: b(a.kw),
                        vw: b(a.vw),
                        kdyn: b(a.kdyn),
                        vdyn: b(a.vdyn),
                        kout: b(kout),
                        vout: b(vout),
                        eps: EPS,
                        kpool: b(&kpool),
                        vpool: b(&vpool),
                        pt: b(a.pt),
                        pt_stride: i(crate::engine::meta::KPT),
                    },
                )
            }
        }
    }

    /// Reference `glue::qpost_kernel<<<(T, NH), HD>>>` and `glue::rel_logits_kernel<<<(T, NH), 256>>>`.
    pub fn qpost_ref(&self, t: usize, a: &QkvPost<'_>, proj: &DevBuf<Bf16>, rel_logits: &DevBuf<Bf16>) -> Result<()> {
        a.check(t)?;
        need(rel_logits, t * NH * REL_EXT, "rel logits")?;
        // SAFETY: row t reads its q / r projections and the layer's scales, writes q and rel row t (checked).
        unsafe {
            self.k.glue_qpost_kernel(
                self.st,
                cfg([t, NH, 1], HD, 0),
                GlueQpostKernelArgs {
                    qproj: b(a.qproj),
                    q_scale: b(a.q_scale),
                    rproj: b(a.rproj),
                    r_scale: b(a.r_scale),
                    row_pos: b(a.pos),
                    full: i32::from(a.full),
                    q: b(a.q),
                    rel: b(a.rel),
                },
            )?;
        }
        // SAFETY: row t reads rel row t and the [16][1024] projection, writes rel_logits row t (checked).
        unsafe {
            self.k.glue_rel_logits_kernel(
                self.st,
                cfg([t, NH, 1], 256, 0),
                GlueRelLogitsKernelArgs {
                    rel: b(a.rel),
                    proj: b(proj),
                    out: b(rel_logits),
                },
            )
        }
    }

    /// Reference `glue::kv_conv_kernel<<<(T, KVH), HD>>>` (key / value conv into k, v).
    pub fn kv_conv_ref(&self, t: usize, a: &QkvPost<'_>, kout: &DevBuf<Bf16>, vout: &DevBuf<Bf16>) -> Result<()> {
        a.check(t)?;
        need(kout, t * KVH * HD, "k")?;
        need(vout, t * KVH * HD, "v")?;
        // SAFETY: as qkv_post_rows for the conv state reads; writes k / v row t (checked).
        unsafe {
            self.k.glue_kv_conv_kernel(
                self.st,
                cfg([t, KVH, 1], HD, 0),
                GlueKvConvKernelArgs {
                    kx: b(a.kx),
                    vx: b(a.vx),
                    kst: b(a.kst),
                    vst: b(a.vst),
                    st_slot_stride: (NVISITS * 3 * SLD) as i64,
                    R: a.rows,
                    full: i32::from(a.full),
                    kw: b(a.kw),
                    vw: b(a.vw),
                    kdyn: b(a.kdyn),
                    vdyn: b(a.vdyn),
                    kout: b(kout),
                    vout: b(vout),
                    eps: EPS,
                },
            )
        }
    }

    /// Reference `glue::kv_write_kernel` (or `_fp8_kernel`)`<<<(T, KVH), HD>>>(k, v, kb, vb, pt, KPT, rows)`.
    pub fn kv_write_ref(&self, t: usize, a: &QkvPost<'_>, k: &DevBuf<Bf16>, v: &DevBuf<Bf16>) -> Result<()> {
        need(k, t * KVH * HD, "k")?;
        need(v, t * KVH * HD, "v")?;
        if a.fp8 {
            // SAFETY: row t writes its E4M3 K/V bytes at the page its segment's page-table row names (as qkv_post_rows).
            unsafe {
                self.k.glue_kv_write_fp8_kernel(
                    self.st,
                    cfg([t, KVH, 1], HD, 0),
                    GlueKvWriteFp8KernelArgs {
                        k: b(k),
                        v: b(v),
                        kpool: b(a.kpool),
                        vpool: b(a.vpool),
                        pt: b(a.pt),
                        pt_stride: i(crate::engine::meta::KPT),
                        R: a.rows,
                    },
                )
            }
        } else {
            let kpool = a.kpool.cast::<Bf16>()?;
            let vpool = a.vpool.cast::<Bf16>()?;
            // SAFETY: row t writes its bf16 K/V at the page its segment's page-table row names (as qkv_post_rows).
            unsafe {
                self.k.glue_kv_write_kernel(
                    self.st,
                    cfg([t, KVH, 1], HD, 0),
                    GlueKvWriteKernelArgs {
                        k: b(k),
                        v: b(v),
                        kpool: b(&kpool),
                        vpool: b(&vpool),
                        pt: b(a.pt),
                        pt_stride: i(crate::engine::meta::KPT),
                        R: a.rows,
                    },
                )
            }
        }
    }

    /// `glue::rel_logits_kernel<<<(T, NH), 256>>>(rel, proj, out)`: distance-indexed relative logits for decode rows
    /// in a mixed call (`prefill_attn` computes relative bias in-kernel directly from `rel` and `rel_proj_t`).
    pub fn rel_logits(&self, t: usize, rel: &DevBuf<Bf16>, proj: &DevBuf<Bf16>, out: &DevBuf<Bf16>) -> Result<()> {
        need(rel, t * NH * DREL, "rel")?;
        need(proj, DREL * REL_EXT, "rel proj")?;
        need(out, t * NH * REL_EXT, "rel logits")?;
        // SAFETY: block (t, h) reads rel[t][h][0..16] and proj[0..16][0..1024], writes out[t][h][0..1024] (checked).
        unsafe {
            self.k.glue_rel_logits_kernel(
                self.st,
                cfg([t, NH, 1], 256, 0),
                GlueRelLogitsKernelArgs {
                    rel: b(rel),
                    proj: b(proj),
                    out: b(out),
                },
            )
        }
    }

    /// `pglue::relk_direct_kernel<<<(ceil(T / 32), NH), 256>>>(rel, proj, pos, T, relk, rel_logits?)`.
    #[allow(dead_code)]
    pub fn relk_direct(
        &self,
        t: usize,
        rel: &DevBuf<Bf16>,
        proj: &DevBuf<Bf16>,
        pos: &DevBuf<i32>,
        relk: &DevBuf<Bf16>,
        rel_logits: Option<&DevBuf<Bf16>>,
    ) -> Result<()> {
        need(rel, t * NH * 16, "rel")?;
        need(pos, t, "pos")?;
        need(relk, t * NH * (REL_EXT + 64), "relk")?;
        if let Some(r) = rel_logits {
            need(r, t * NH * REL_EXT, "rel logits")?;
        }
        // SAFETY: rows [32 x, 32 x + 32) & < T read rel and the projection, write the key-aligned bias rows
        // [T][NH][1088] and, when given, the distance-indexed logits [T][NH][1024] (checked).
        unsafe {
            self.k.pglue_relk_direct_kernel(
                self.st,
                cfg([t.div_ceil(REL_ROWS), NH, 1], 256, 0),
                PglueRelkDirectKernelArgs {
                    rel: b(rel),
                    proj: b(proj),
                    row_pos: b(pos),
                    T: i(t),
                    relk: b(relk),
                    rel_logits: opt(rel_logits),
                },
            )
        }
    }

    /// `pglue::rel_logits_decode_kernel<<<(8, ceil(T*24 / 16)), 128>>>`: weight-stationary decode relative logits (bit-identical).
    pub fn rel_logits_decode(
        &self,
        t: usize,
        rel: &DevBuf<Bf16>,
        proj: &DevBuf<Bf16>,
        out: &DevBuf<Bf16>,
    ) -> Result<()> {
        let rows = t * NH;
        need(rel, rows * DREL, "rel_logits_decode rel")?;
        need(proj, DREL * REL_EXT, "rel_logits_decode proj")?;
        need(out, rows * REL_EXT, "rel_logits_decode out")?;
        // SAFETY: block (bx, by) computes 128 cols x up to 16 rows of rel [rows, 16] @ proj [16, 1024] (all extents checked).
        unsafe {
            self.k.pglue_rel_logits_decode_kernel(
                self.st,
                cfg([REL_EXT / 128, rows.div_ceil(16), 1], 128, 0),
                PglueRelLogitsDecodeKernelArgs {
                    rel: b(rel),
                    proj: b(proj),
                    R: i(rows),
                    out: b(out),
                },
            )
        }
    }

    // ------------------------------------------------------------------------------------------------ attention

    /// `da::launch` (design A, global layers: bf16 or E4M3 when `fp8`) or `da::ring_launch` (design B, sliding window),
    /// then the combine. `grid_x` = workers per (session, head); `n` decode segments. Supports `1 <= R <= 16` in one KV pass.
    pub fn decode_attn(&self, p: &da::Params, ring: bool, fp8: bool, n: usize, grid_x: usize) -> Result<()> {
        let r = usize::try_from(p.R).unwrap_or(0);
        if !(1..=16).contains(&r) || n == 0 {
            return Err(CudaError::Bounds {
                what: "decode_attn",
                detail: format!("R = {r}, n = {n}"),
            });
        }
        // the kernel variant of R (the graph-cache key's `v`): within one, R changes only `p` and the grids
        let nw = crate::engine::meta::da_variant(r);
        // SAFETY: the da::Params contract (decode_attn.cuh "Params", engine/native/README.md): session s < n reads the
        // pages its page-table row s names (decode-order rows built by meta::build_call, pages of its slot or its
        // ring, inside the pool / ring of this visit), its ctx / row0 / chunk_shift / nrows entries (n uploaded),
        // q / rel / gate / sink rows [row0, row0 + nrows) (call rows < T <= t_max), and writes o_part / ml_part
        // [n][8][max_chunks][3 R] (bounded by the caller against da_opart) and out rows; grid_x surplus workers exit.
        // Shared memory per kernel as set at engine start.
        unsafe {
            if ring {
                if p.row_stride != 256 {
                    return Err(CudaError::Bounds {
                        what: "ring_launch",
                        detail: "row stride".into(),
                    });
                }
                let c = cfg([grid_x, 8, n], nw * 32, da_ring_smem(nw));
                match nw {
                    1 => self.k.da_decode_attn_ring_kernel_0_1_3(self.st, c, p),
                    2 => self.k.da_decode_attn_ring_kernel_0_2_3(self.st, c, p),
                    3 => self.k.da_decode_attn_ring_kernel_0_3_3(self.st, c, p),
                    4 => self.k.da_decode_attn_ring_kernel_0_4_3(self.st, c, p),
                    5 => self.k.da_decode_attn_ring_kernel_0_5_3(self.st, c, p),
                    _ => self.k.da_decode_attn_ring_kernel_0_6_3(self.st, c, p),
                }?;
            } else if nw > 3 {
                let c = cfg([grid_x, 8, n], 4 * 32, DA_SMEM16);
                if fp8 {
                    self.k.da_decode_attn_kernel16_1(self.st, c, p)?;
                } else {
                    self.k.da_decode_attn_kernel16_0(self.st, c, p)?;
                }
            } else {
                let c = cfg([grid_x, 8, n], 4 * 32, DA_SMEM);
                match (fp8, nw > 1) {
                    (false, true) => self.k.da_decode_attn_kernel_0_true(self.st, c, p)?,
                    (false, false) => self.k.da_decode_attn_kernel_0_false(self.st, c, p)?,
                    (true, true) => self.k.da_decode_attn_kernel_1_true(self.st, c, p)?,
                    (true, false) => self.k.da_decode_attn_kernel_1_false(self.st, c, p)?,
                }
            }
            self.k.da_combine_kernel(self.st, cfg([DA_G * r, 8, n], 128, 0), p)
        }
    }

    /// `pfa::launch<4, 2, 1, 1, 2>(p, nitems)` (or `prefill_attn_fp8_kernel` when `fp8`): nothing when there are no items.
    pub fn prefill_attn(&self, p: &pfa::Params, fp8: bool, nitems: usize) -> Result<()> {
        if nitems == 0 {
            return Ok(());
        }
        // SAFETY: the pfa::Params contract (prefill_attn.cuh "Params"): item x < nitems (uploaded by the engine for
        // this kind) names a segment, a 128-row tile and a key-block range inside the segment's past + len keys;
        // the CTA reads those keys through the segment's page-table row (call-order rows built by meta::build_call),
        // q / gate / sink / relk rows of the segment (< T), writes its out rows, and split partials at the item's
        // workspace base (bounded against pa_ws by meta::pa_items) with counters per (tile, KV head) inside pa_cnt.
        unsafe {
            if fp8 {
                self.k
                    .pfa_prefill_attn_fp8_kernel_4_2_1_1_2_false(self.st, cfg([nitems, 8, 1], 128, PFA_FP8_SMEM), p)
            } else {
                self.k
                    .pfa_prefill_attn_kernel_4_2_1_1_2_false(self.st, cfg([nitems, 8, 1], 128, PFA_SMEM), p)
            }
        }
    }

    /// `pfa::prefill_attn_far_kernel<4, 2, 1, 1, 2>` (or `prefill_attn_fp8_far_kernel<4, 2, 2>` when `fp8`):
    /// pure-FAR global items (`qmin >= it.w * 64 + WINDOW - 1`, `window == 0`, no bias/mask); nothing when 0 items.
    pub fn prefill_attn_far(&self, p: &pfa::Params, fp8: bool, nitems: usize) -> Result<()> {
        if nitems == 0 {
            return Ok(());
        }
        // SAFETY: same pfa::Params contract as `prefill_attn`; additionally every item x < nitems is guaranteed by
        // `meta::pa_items` to have `window == 0` and all key positions at distance >= 1024 from every query row in the
        // tile (strictly causal, strictly outside the relative-bias window, and strictly inside klen).
        unsafe {
            if fp8 {
                self.k
                    .pfa_prefill_attn_fp8_far_kernel_4_2_2(self.st, cfg([nitems, 8, 1], 128, PFA_FP8_FAR_SMEM), p)
            } else {
                self.k
                    .pfa_prefill_attn_far_kernel_4_2_1_1_2(self.st, cfg([nitems, 8, 1], 128, PFA_FAR_SMEM), p)
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ convs

    /// `glue::attn_conv_partial_kernel<<<grid1(T H), 256>>>` (production glue).
    #[allow(dead_code)]
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn attn_conv_partial(
        &self,
        t: usize,
        x: &DevBuf<Bf16>,
        w: &DevBuf<f32>,
        state: &DevBuf<Bf16>,
        r: glue::Rows,
        hidden: &DevBuf<Bf16>,
        blk: Option<&DevBuf<Bf16>>,
        y: &DevBuf<Bf16>,
        partial: &DevBuf<f32>,
    ) -> Result<()> {
        for (n, what) in [(x.len(), "x"), (hidden.len(), "hidden"), (y.len(), "y")] {
            if n < t * H {
                return Err(CudaError::Bounds {
                    what: "attn conv",
                    detail: what.into(),
                });
            }
        }
        need(partial, t * H, "partial")?;
        need(w, H * 4, "conv w")?;
        if let Some(b) = blk {
            need(b, t * H, "block")?;
        }
        // SAFETY: grid-stride over T x H: row t reads x / hidden rows and the conv state of its segment's slot
        // (state + slot * st_slot, max_slots slots of 3 rows), writes y, partial and (block start) blk rows (checked).
        unsafe {
            self.k.glue_attn_conv_partial_kernel(
                self.st,
                cfg([grid1(t * H), 1, 1], 256, 0),
                GlueAttnConvPartialKernelArgs {
                    x: b(x),
                    w: b(w),
                    state: b(state),
                    st_slot_stride: (NVISITS * 3 * SLD) as i64,
                    R: r,
                    hidden: b(hidden),
                    block_start: i32::from(blk.is_some()),
                    blk: opt(blk),
                    y_out: b(y),
                    partial32: b(partial),
                    T: i(t),
                },
            )
        }
    }

    /// `glue::hidden_conv_kernel<<<grid1(T H), 256>>>(x, w, state, st_slot, rows, mode, partial, out, T)`.
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn hidden_conv(
        &self,
        t: usize,
        x: &DevBuf<Bf16>,
        w: &DevBuf<f32>,
        state: &DevBuf<Bf16>,
        r: glue::Rows,
        mode: i32,
        partial: Option<&DevBuf<f32>>,
        out: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(x, t * H, "conv x")?;
        need(out, t * H, "conv out")?;
        need(w, H * 4, "conv w")?;
        if let Some(p) = partial {
            need(p, t * H, "partial")?;
        }
        // SAFETY: as attn_conv_partial; 2D grid (H / 256, T), mode 1 reads partial (given), mode 0 none.
        unsafe {
            self.k.glue_hidden_conv_kernel(
                self.st,
                cfg([H / 256, t, 1], 256, 0),
                GlueHiddenConvKernelArgs {
                    x: b(x),
                    w: b(w),
                    state: b(state),
                    st_slot_stride: (NVISITS * 3 * SLD) as i64,
                    R: r,
                    mode,
                    partial32: opt(partial),
                    out: b(out),
                    T: i(t),
                },
            )
        }
    }

    /// `pglue::conv_rows_kernel<MODE><<<(3, ceil(T / 8)), 96>>>(x, w, state, st_slot, rows, T, hin, block_start, blk,
    /// partial32, out)` (T > 64; same bits): [`ConvRows::Attn`] = attention conv + residual (`attn_conv_partial`
    /// without the unused `y` store, MODE 0), [`ConvRows::Mlp`] = MLP conv + residual (`hidden_conv` mode 1, MODE 1).
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn conv_rows(
        &self,
        t: usize,
        x: &DevBuf<Bf16>,
        w: &DevBuf<f32>,
        state: &DevBuf<Bf16>,
        r: glue::Rows,
        partial: &DevBuf<f32>,
        mode: ConvRows<'_>,
    ) -> Result<()> {
        need(x, t * H, "conv x")?;
        need(w, H * 4, "conv w")?;
        need(partial, t * H, "partial")?;
        let c = cfg([H / 8 / CONV_THREADS, t.div_ceil(CONV_ROWS), 1], CONV_THREADS, 0);
        let st_slot_stride = (NVISITS * 3 * SLD) as i64;
        match mode {
            ConvRows::Attn { hidden, blk } => {
                need(hidden, t * H, "conv hidden")?;
                if let Some(bk) = blk {
                    need(bk, t * H, "block")?;
                }
                // SAFETY: thread = 8 channels x 8 rows (< T): reads x rows (and the 3 earlier rows of the conv window
                // or its segment's slot state: state + slot st_slot, max_slots slots of 3 rows), hidden rows, writes
                // partial and (block start) blk rows; 16-byte accesses at 8-channel offsets of 256-byte aligned
                // takes (extents checked).
                unsafe {
                    self.k.pglue_conv_rows_kernel_0(
                        self.st,
                        c,
                        PglueConvRowsKernel0Args {
                            x: b(x),
                            w: b(w),
                            state: b(state),
                            st_slot_stride,
                            R: r,
                            T: i(t),
                            hin: b(hidden),
                            block_start: i32::from(blk.is_some()),
                            blk: opt(blk),
                            partial32: b(partial),
                            out: Ptr::Null,
                        },
                    )
                }
            }
            ConvRows::Mlp { out } => {
                need(out, t * H, "conv out")?;
                // SAFETY: as the attention mode for x and the state; reads partial rows, writes out rows (checked).
                unsafe {
                    self.k.pglue_conv_rows_kernel_1(
                        self.st,
                        c,
                        PglueConvRowsKernel1Args {
                            x: b(x),
                            w: b(w),
                            state: b(state),
                            st_slot_stride,
                            R: r,
                            T: i(t),
                            hin: Ptr::Null,
                            block_start: 0,
                            blk: Ptr::Null,
                            partial32: b(partial),
                            out: b(out),
                        },
                    )
                }
            }
        }
    }

    /// Reference `glue::partial_kernel<<<grid1(T H), 256>>>(hidden, y, block_start, blk, partial, T)`.
    pub fn partial_ref(
        &self,
        t: usize,
        hidden: &DevBuf<Bf16>,
        y: &DevBuf<Bf16>,
        blk: Option<&DevBuf<Bf16>>,
        partial: &DevBuf<f32>,
    ) -> Result<()> {
        need(hidden, t * H, "hidden")?;
        need(y, t * H, "y")?;
        need(partial, t * H, "partial")?;
        if let Some(b) = blk {
            need(b, t * H, "block")?;
        }
        // SAFETY: grid-stride over T x H rows of the checked buffers.
        unsafe {
            self.k.glue_partial_kernel(
                self.st,
                cfg([grid1(t * H), 1, 1], 256, 0),
                GluePartialKernelArgs {
                    hidden: b(hidden),
                    y: b(y),
                    block_start: i32::from(blk.is_some()),
                    blk: opt(blk),
                    partial32: b(partial),
                    T: i(t),
                },
            )
        }
    }

    /// `glue::conv_commit_kernel<<<(36, NVISITS, S), 256>>>(CommitArgs)`.
    pub fn conv_commit(&self, s: usize, a: &glue::CommitArgs) -> Result<()> {
        // SAFETY: CommitArgs (glue.cuh): segment s < nseg = S (row0 / slot / cnt uploaded for S segments) moves the
        // last rows of its compact convx tail inputs into its slot's conv state.
        unsafe {
            self.k
                .glue_conv_commit_kernel(self.st, cfg([4 * 9, NVISITS, s], 256, 0), a)
        }
    }

    /// `pglue::swa_ring_copy_kernel<<<(ceil(ntok / 8), 8, 2), 128>>>`: copies trailing SWA window tokens
    /// `[tok_lo, tok_lo + ntok)` across all 8 KV heads for both K and V from a shared scratch page into a slot's
    /// persistent 2-page SWA ring.
    pub fn swa_ring_copy(
        &self,
        k_src_page: &DevBuf<Bf16>,
        v_src_page: &DevBuf<Bf16>,
        k_dst_page: &DevBuf<Bf16>,
        v_dst_page: &DevBuf<Bf16>,
        tok_lo: usize,
        ntok: usize,
    ) -> Result<()> {
        if ntok == 0 {
            return Ok(());
        }
        let page_elems = KVH * 1024 * HD;
        need(k_src_page, page_elems, "swa_ring_copy k_src")?;
        need(v_src_page, page_elems, "swa_ring_copy v_src")?;
        need(k_dst_page, page_elems, "swa_ring_copy k_dst")?;
        need(v_dst_page, page_elems, "swa_ring_copy v_dst")?;
        if tok_lo + ntok > 1024 {
            return Err(CudaError::Bounds {
                what: "swa_ring_copy range",
                detail: format!("tok_lo {tok_lo} + ntok {ntok} > 1024"),
            });
        }
        // SAFETY: grid (ceil(ntok / 8), 8, 2), block 128 copies [tok_lo, tok_lo + ntok) x 8 heads x 128 bf16 (uint4)
        // for both K and V; all four 1,024-token page buffers and the token range are checked above.
        unsafe {
            self.k.pglue_swa_ring_copy_kernel(
                self.st,
                cfg([ntok.div_ceil(8), KVH, 2], 128, 0),
                PglueSwaRingCopyKernelArgs {
                    k_src_page: b(k_src_page),
                    v_src_page: b(v_src_page),
                    k_dst_page: b(k_dst_page),
                    v_dst_page: b(v_dst_page),
                    tok_lo: i(tok_lo),
                    ntok: i(ntok),
                },
            )
        }
    }

    // ------------------------------------------------------------------------------------------------ router, experts

    /// Reference `glue::router_kernel<<<T, NEXP>>>(xm, router, bias, ridx, rw)`.
    pub fn router_ref(
        &self,
        t: usize,
        x: &DevBuf<Bf16>,
        wr: &DevBuf<Bf16>,
        bias: &DevBuf<f32>,
        idx: &DevBuf<i32>,
        wts: &DevBuf<f32>,
    ) -> Result<()> {
        need(x, t * H, "router x")?;
        need(wr, NEXP * H, "router w")?;
        need(idx, t * TOPK, "ridx")?;
        need(wts, t * TOPK, "rw")?;
        // SAFETY: block t reads row t of x and the router, writes TOPK entries of row t (checked).
        unsafe {
            self.k.glue_router_kernel(
                self.st,
                cfg([t, 1, 1], NEXP, 0),
                GlueRouterKernelArgs {
                    x: b(x),
                    wr: b(wr),
                    bias: b(bias),
                    idx: b(idx),
                    wts: b(wts),
                },
            )
        }
    }

    /// `pglue::router_decode_kernel<MT><<<(8, 9, 1), 256>>>`: fused 3-K-group decode router GEMM + top-8 (m <= 64).
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn router_decode(
        &self,
        m: usize,
        x: &DevBuf<Bf16>,
        wr: &DevBuf<Bf16>,
        bias: &DevBuf<f32>,
        rlog: &DevBuf<f32>,
        cnt: &DevBuf<i32>,
        idx: &DevBuf<i32>,
        wts: &DevBuf<f32>,
    ) -> Result<()> {
        if m == 0 || m > 128 {
            return Err(CudaError::Bounds {
                what: "router_decode",
                detail: format!("{m} rows (1..=128)"),
            });
        }
        need(x, m * H, "router_decode x")?;
        need(wr, NEXP * H, "router_decode wr")?;
        need(bias, NEXP, "router_decode bias")?;
        need(rlog, 9 * m * NEXP, "router_decode rlog")?;
        need(cnt, 1, "router_decode cnt")?;
        need(idx, m * TOPK, "router_decode idx")?;
        need(wts, m * TOPK, "router_decode wts")?;
        let c = cfg([8, 9, 1], 256, 0);
        // SAFETY: 72 CTAs compute 9 K-slices into rlog [9][m][128]. For m <= 8 the last CTA runs 8-warp route_select
        // (1 row/warp) in the same launch; for 9 <= m <= 64, bit-identical router_gemm_dec + route_select9<<<m, 32>>>
        // parallelizes route_select across all m SMs (all extents checked above).
        unsafe {
            if m <= 8 {
                self.k.pglue_router_decode_kernel_1(
                    self.st,
                    c,
                    PglueRouterDecodeKernel1Args {
                        x: b(x),
                        wr: b(wr),
                        bias: b(bias),
                        rlog: b(rlog),
                        cnt: b(cnt),
                        M: i(m),
                        idx: b(idx),
                        wts: b(wts),
                    },
                )
            } else {
                if m <= 16 {
                    self.k.pglue_router_gemm_dec_kernel_1(
                        self.st,
                        c,
                        PglueRouterGemmDecKernel1Args {
                            x: b(x),
                            wr: b(wr),
                            rlog: b(rlog),
                            M: i(m),
                        },
                    )?;
                } else if m <= 32 {
                    self.k.pglue_router_gemm_dec_kernel_2(
                        self.st,
                        c,
                        PglueRouterGemmDecKernel2Args {
                            x: b(x),
                            wr: b(wr),
                            rlog: b(rlog),
                            M: i(m),
                        },
                    )?;
                } else if m <= 64 {
                    self.k.pglue_router_gemm_dec_kernel_4(
                        self.st,
                        c,
                        PglueRouterGemmDecKernel4Args {
                            x: b(x),
                            wr: b(wr),
                            rlog: b(rlog),
                            M: i(m),
                        },
                    )?;
                } else {
                    self.k.pglue_router_gemm_dec_kernel_8(
                        self.st,
                        c,
                        PglueRouterGemmDecKernel8Args {
                            x: b(x),
                            wr: b(wr),
                            rlog: b(rlog),
                            M: i(m),
                        },
                    )?;
                }
                self.k.pglue_route_select9_kernel(
                    self.st,
                    cfg([m, 1, 1], 32, 0),
                    PglueRouteSelect9KernelArgs {
                        rlog: b(rlog),
                        M: i(m),
                        bias: b(bias),
                        idx: b(idx),
                        wts: b(wts),
                    },
                )
            }
        }
    }

    /// `glue::route_select_kernel<<<T, 32>>>` (or 4-warp `pglue::route_select_rows_kernel<<<ceil(T/4), 128>>>` for `t > 64`).
    pub fn route_select(
        &self,
        t: usize,
        g: &DevBuf<f32>,
        bias: &DevBuf<f32>,
        idx: &DevBuf<i32>,
        wts: &DevBuf<f32>,
    ) -> Result<()> {
        need(g, 3 * t * NEXP, "router logits")?;
        need(bias, NEXP, "router bias")?;
        need(idx, t * TOPK, "ridx")?;
        need(wts, t * TOPK, "rw")?;
        // SAFETY: sums the three K-group logits of row t ([3][T][128]), writes TOPK entries (checked).
        unsafe {
            if t > 128 {
                self.k.pglue_route_select_rows_kernel(
                    self.st,
                    cfg([t.div_ceil(4), 1, 1], 128, 0),
                    PglueRouteSelectRowsKernelArgs {
                        g: b(g),
                        T: i(t),
                        bias: b(bias),
                        idx: b(idx),
                        wts: b(wts),
                    },
                )
            } else {
                self.k.glue_route_select_kernel(
                    self.st,
                    cfg([t, 1, 1], 32, 0),
                    GlueRouteSelectKernelArgs {
                        g: b(g),
                        T: i(t),
                        bias: b(bias),
                        idx: b(idx),
                        wts: b(wts),
                    },
                )
            }
        }
    }

    /// The routing build of `pmoe_rows`: fast `__match_any_sync` 512-row parallel build for `n > 64`, else `moedev::moe_build_dec_kernel`.
    pub fn moe_build(&self, n: usize, r: &MoeBuild<'_>) -> Result<()> {
        let rg = n * TOPK;
        need(r.idx, rg, "ridx")?;
        need(r.wts, rg, "rw")?;
        for (x, what) in [(r.a13, "arow13"), (r.a2, "arow2"), (r.orow, "orow")] {
            need(x, rg, what)?;
        }
        need(r.gamma, rg, "gamma")?;
        let need_items = if n > 64 {
            ink_model::plan::moe_items_bound(rg)
        } else {
            ink_model::plan::moe_dec_items_bound(rg)
        };
        need(r.items, need_items, "items")?;
        need(r.n_items, 1, "n_items")?;
        if n > 64 {
            let nb = rg.div_ceil(BUILD_ROWS_FAST);
            need(r.cntb, nb * NEXP, "cntb")?;
            need(r.baseb, nb * NEXP, "baseb")?;
            // SAFETY: block x counts experts of routed rows [512 x, 512 x + 512) & < Rg into cntb[x][128]
            // (checked); moe_plan_fast turns nb x 128 counts into bases and at most items_bound(Rg) items (checked);
            // moe_scatter_fast writes Rg sorted positions in O(1) via __match_any_sync (checked). Expert ids < 128.
            unsafe {
                self.k.pglue_moe_count_fast_kernel(
                    self.st,
                    cfg([nb, 1, 1], 256, 0),
                    PglueMoeCountFastKernelArgs {
                        idx: b(r.idx),
                        R: i(rg),
                        cntb: b(r.cntb),
                    },
                )?;
                self.k.pglue_moe_plan_fast_kernel(
                    self.st,
                    cfg([1, 1, 1], 512, 0),
                    PglueMoePlanFastKernelArgs {
                        cntb: b(r.cntb),
                        nb: i(nb),
                        baseb: b(r.baseb),
                        items: b(r.items),
                        n_items: b(r.n_items),
                    },
                )?;
                self.k.pglue_moe_scatter_fast_kernel(
                    self.st,
                    cfg([nb, 1, 1], 256, 0),
                    PglueMoeScatterFastKernelArgs {
                        idx: b(r.idx),
                        wts: b(r.wts),
                        R: i(rg),
                        baseb: b(r.baseb),
                        a13: b(r.a13),
                        a2: b(r.a2),
                        orow: b(r.orow),
                        gamma: b(r.gamma),
                    },
                )
            }
        } else {
            // SAFETY: one block (512 threads) sorts Rg <= 512 routed rows by expert, writes Rg positions and at most
            // dec_items_bound(Rg) decode items (FT_DEC=18, BN_DEC=16) and their count (checked).
            unsafe {
                self.k.moedev_moe_build_dec_kernel(
                    self.st,
                    cfg([1, 1, 1], 512, 0),
                    MoedevMoeBuildDecKernelArgs {
                        idx: b(r.idx),
                        wts: b(r.wts),
                        R: i(rg),
                        a13: b(r.a13),
                        a2: b(r.a2),
                        orow: b(r.orow),
                        gamma: b(r.gamma),
                        items: b(r.items),
                        n_items: b(r.n_items),
                    },
                )
            }
        }
    }

    /// `moe_launch_dev<FMT, MODE>`: for decode calls (`rg <= 64 * TOPK`) launches `pmoe::moe_gemm_dec<FMT, MODE, 5, 4>`
    /// (BN_DEC=16, 4 warps, NT in {1, 2}); for prefill launches `pmoe::moe_gemm2<FMT, MODE, 3, 8>` (FMT=3 NVFP4_N1S1
    /// 1-byte scales, or FMT=1 MXFP8).
    pub fn moe_gemm(&self, nvfp4: bool, rg: usize, p13: &pmoe::Params, p2: &pmoe::Params) -> Result<()> {
        if rg <= 64 * TOPK {
            let grid = ink_model::plan::moe_dec_items_bound(rg);
            let smem = if nvfp4 { MOE_DEC_SMEM_N1S1 } else { MOE_DEC_SMEM_MXFP8 };
            let c = LaunchCfg {
                grid: [u(grid), 1, 1],
                block: [MOE_DEC_THREADS, 1, 1],
                smem,
            };
            // SAFETY: matches moedev::moe_build_dec_kernel (FT_DEC=18, BN_DEC=16, *n_items_dev <= dec_items_bound(Rg));
            // CTAs at or past *n_items_dev exit before reading an item; weight/scale/activation buffers satisfy pmoe::Params.
            unsafe {
                if nvfp4 {
                    self.k.pmoe_moe_gemm_dec_3_0_5_4(self.st, c, p13)?;
                    self.k.pmoe_moe_gemm_dec_3_2_5_4(self.st, c, p2)
                } else {
                    self.k.pmoe_moe_gemm_dec_1_0_5_4(self.st, c, p13)?;
                    self.k.pmoe_moe_gemm_dec_1_2_5_4(self.st, c, p2)
                }
            }
        } else {
            let grid = ink_model::plan::moe_items_bound(rg);
            let smem = if nvfp4 { MOE_SMEM_NVFP4_N1S1 } else { MOE_SMEM_MXFP8 };
            let c = LaunchCfg {
                grid: [u(grid), 1, 1],
                block: [MOE_THREADS, 1, 1],
                smem,
            };
            // SAFETY: the pmoe::Params contract (prefill_moe.cuh): CTA x < *n_items_dev (written by the routing build,
            // <= items_bound(Rg) = grid) reads its item, the expert's repacked weights / scales / factor of this layer,
            // A rows a_row[p] for its sorted positions p < Rg, and writes h [Rg][I] (w13, sorted) or y rows out_row[p]
            // (w2) inside mh / my (t_max x TOPK rows); CTAs at or past the count exit before reading an item.
            unsafe {
                if nvfp4 {
                    self.k.pmoe_moe_gemm2_3_0_3_8_0(self.st, c, p13)?;
                    self.k.pmoe_moe_gemm2_3_2_3_8_0(self.st, c, p2)
                } else {
                    self.k.pmoe_moe_gemm2_1_0_3_8_0(self.st, c, p13)?;
                    self.k.pmoe_moe_gemm2_1_2_3_8_0(self.st, c, p2)
                }
            }
        }
    }

    /// Track 3A-6: fused `moe_combine` + MLP conv + residual (`pglue::moe_combine_conv_kernel<<<(3, ceil(T/64)), 96>>>`).
    /// Bit-identical to `moe_combine` + `conv_rows<1>`, writing `mx` only for segment tail rows needed by `conv_commit`.
    #[expect(clippy::too_many_arguments, reason = "the kernel's parameters")]
    pub fn moe_combine_conv(
        &self,
        t: usize,
        my: &DevBuf<Bf16>,
        w: &DevBuf<f32>,
        state: &DevBuf<Bf16>,
        r: glue::Rows,
        seg_len: &DevBuf<i32>,
        partial: &DevBuf<f32>,
        mx: &DevBuf<Bf16>,
        out: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(my, t * TOPK * H, "moe_combine_conv my")?;
        need(w, H * 4, "moe_combine_conv w")?;
        need(partial, t * H, "moe_combine_conv partial")?;
        need(mx, t * H, "moe_combine_conv mx")?;
        need(out, t * H, "moe_combine_conv out")?;
        // SAFETY: thread = 4 channels x 32 rows (< T); combines my in slot order, runs static_conv + partial -> out,
        // writes conv_commit tail rows into mx (all extents checked).
        unsafe {
            self.k.pglue_moe_combine_conv_kernel(
                self.st,
                cfg(
                    [H / 4 / CONV_THREADS, t.div_ceil(COMBINE_CONV_ROWS), 1],
                    CONV_THREADS,
                    0,
                ),
                PglueMoeCombineConvKernelArgs {
                    my: b(my),
                    w: b(w),
                    state: b(state),
                    st_slot_stride: (NVISITS * 3 * SLD) as i64,
                    R: r,
                    seg_len: b(seg_len),
                    T: i(t),
                    partial32: b(partial),
                    mx: b(mx),
                    out: b(out),
                },
            )
        }
    }

    /// `pmoe::moe_combine<<<ceil(T H / 8 / 256), 256>>>(my, out, T)`.
    pub fn moe_combine(&self, t: usize, y: &DevBuf<Bf16>, out: &DevBuf<Bf16>) -> Result<()> {
        need(y, t * TOPK * H, "moe y")?;
        need(out, t * H, "moe out")?;
        // SAFETY: thread k < T x H / 8 sums the TOPK expert rows of token k / (H / 8) into out (checked).
        unsafe {
            self.k.pmoe_moe_combine(
                self.st,
                cfg([(t * (H / 8)).div_ceil(256), 1, 1], 256, 0),
                PmoeMoeCombineArgs {
                    y: b(y),
                    out: b(out),
                    tokens: i(t),
                },
            )
        }
    }
}

/// One gate's buffers of [`Dev::mg_down_dec`]: its input `x` (`[n][4608]`), down weight `w` (`[64][4608]`), and output `out` (`[n][64]`).
pub(super) struct MgDownGate<'a> {
    pub x: &'a DevBuf<Bf16>,
    pub w: &'a DevBuf<Bf16>,
    pub out: &'a DevBuf<Bf16>,
}

/// One gate's buffers of [`Dev::mg_tail`]: its down projection `gd` (`[T][64]`), the controller's transposed forget
/// and gain weights (`[64][H]`), and the hidden, message and output rows, each at the gate's first column.
#[allow(dead_code)]
pub(super) struct MgGate<'a> {
    pub gd: &'a DevBuf<Bf16>,
    pub forget_t: &'a DevBuf<f32>,
    pub gain_t: &'a DevBuf<f32>,
    pub hid: &'a DevBuf<f32>,
    pub msg: &'a DevBuf<f32>,
    pub out: &'a DevBuf<f32>,
}

/// One gate's buffers of [`Dev::mg_tail_tc`]: its down projection `gd` (`[n][64]`), its tensor-core weights `w`
/// (`[2][H][64]` bf16, `GateTailW`), and the hidden, message and output rows, each at the group's first row.
pub(super) struct MgTcGate<'a> {
    pub gd: &'a DevBuf<Bf16>,
    pub w: &'a DevBuf<Bf16>,
    pub hid: &'a DevBuf<f32>,
    pub msg: &'a DevBuf<f32>,
    pub out: &'a DevBuf<f32>,
}

/// The outputs of [`Dev::proj_split`], each at the row group's first row.
pub(super) struct ProjOut<'a> {
    pub q: &'a DevBuf<Bf16>,
    pub g: &'a DevBuf<Bf16>,
    pub r: &'a DevBuf<Bf16>,
    pub s: &'a DevBuf<Bf16>,
    pub k: &'a DevBuf<Bf16>,
    pub kd: &'a DevBuf<Bf16>,
    pub v: &'a DevBuf<Bf16>,
    pub vd: &'a DevBuf<Bf16>,
}

/// The loop-gate inputs an AttnRes launch writes (C3): gate y's message is `msg + y H` (row stride `ldm`), its hidden
/// part this launch's output for query `qsel ? y : 0`; the inputs go to `gin + y T 2H`.
#[derive(Clone, Copy)]
pub(super) struct GateInputs<'a> {
    pub msg: &'a DevBuf<f32>,
    pub ldm: usize,
    pub gin: &'a DevBuf<Bf16>,
    pub qsel: bool,
}

/// The two modes of [`Dev::conv_rows`].
#[derive(Clone, Copy)]
pub(super) enum ConvRows<'a> {
    /// Attention conv + residual: `partial = hidden + conv` (block start: `blk = hidden`, `partial = conv`).
    #[allow(dead_code)]
    Attn {
        hidden: &'a DevBuf<Bf16>,
        blk: Option<&'a DevBuf<Bf16>>,
    },
    /// MLP conv + residual: `out = bf16(partial + conv)`.
    Mlp { out: &'a DevBuf<Bf16> },
}

/// Arguments shared by the q / k / v post kernels of one visit.
pub(super) struct QkvPost<'a> {
    pub qproj: &'a DevBuf<Bf16>,
    pub q_scale: &'a DevBuf<f32>,
    pub rproj: &'a DevBuf<Bf16>,
    pub r_scale: &'a DevBuf<f32>,
    pub pos: &'a DevBuf<i32>,
    pub full: bool,
    pub fp8: bool,
    pub q: &'a DevBuf<Bf16>,
    pub rel: &'a DevBuf<Bf16>,
    pub kx: &'a DevBuf<Bf16>,
    pub vx: &'a DevBuf<Bf16>,
    /// Conv state of this visit for slot 0, k part (`cst`); the kernels add slot x st_slot.
    pub kst: &'a DevBuf<Bf16>,
    /// `cst + 1024`.
    pub vst: &'a DevBuf<Bf16>,
    pub rows: glue::Rows,
    pub kw: &'a DevBuf<f32>,
    pub vw: &'a DevBuf<f32>,
    pub kdyn: &'a DevBuf<Bf16>,
    pub vdyn: &'a DevBuf<Bf16>,
    pub kpool: &'a DevBuf<u8>,
    pub vpool: &'a DevBuf<u8>,
    pub pt: &'a DevBuf<i32>,
}

impl QkvPost<'_> {
    fn check(&self, t: usize) -> Result<()> {
        need(self.qproj, t * QD, "qproj")?;
        need(self.rproj, t * NH * 16, "rproj")?;
        need(self.q, t * QD, "q")?;
        need(self.rel, t * NH * 16, "rel")?;
        need(self.kx, t * KD, "kx")?;
        need(self.vx, t * KD, "vx")?;
        need(self.pos, t, "pos")?;
        need(self.kdyn, t * 32, "kdyn")?;
        need(self.vdyn, t * 32, "vdyn")?;
        Ok(())
    }
}

/// Buffers of the routing build of one expert pass.
pub(super) struct MoeBuild<'a> {
    pub idx: &'a DevBuf<i32>,
    pub wts: &'a DevBuf<f32>,
    pub a13: &'a DevBuf<i32>,
    pub a2: &'a DevBuf<i32>,
    pub orow: &'a DevBuf<i32>,
    pub gamma: &'a DevBuf<f32>,
    pub items: &'a DevBuf<pmoe::Item>,
    pub n_items: &'a DevBuf<i32>,
    pub cntb: &'a DevBuf<i32>,
    pub baseb: &'a DevBuf<i32>,
}

// ------------------------------------------------------------------------------------------------ speculation (R9)
// Ported from engine/native/src/engine.cu @ 1c7ac46: `spec_accept`, `build_verify_ids`, `gather_feats`.

impl Dev<'_> {
    /// `ink::accept_kernel<<<1, 32>>>(amax, drafts, T, S, row0, slot, pos, anchor, P, res, crow, cpos, csess,
    /// nullptr, K)`: greedy acceptance of the S verify segments of the last call (draft stride `k`, result stride
    /// `k + 2`).
    #[allow(clippy::too_many_arguments)]
    pub fn accept(
        &self,
        t: usize,
        s: usize,
        k: usize,
        amax: &DevBuf<i32>,
        drafts: &DevBuf<i32>,
        rows: (&DevBuf<i32>, &DevBuf<i32>, &DevBuf<i32>),
        anchor: &DevBuf<i32>,
        p: &DevBuf<i32>,
        res: &DevBuf<i32>,
        commit: (&DevBuf<i32>, &DevBuf<i32>, &DevBuf<i32>),
    ) -> Result<()> {
        let (row0, slot, pos) = rows;
        let (crow, cpos, csess) = commit;
        let rs = k + 2;
        if s == 0 || s > 32 {
            return Err(CudaError::Bounds {
                what: "accept",
                detail: format!("{s} segments (one warp: 1..32)"),
            });
        }
        need(amax, t, "accept amax")?;
        need(row0, s, "accept row0")?;
        need(slot, s, "accept slot")?;
        need(pos, t, "accept pos")?;
        need(res, rs * s + 1, "accept res")?;
        need(crow, t, "accept crow")?;
        need(cpos, t, "accept cpos")?;
        need(csess, t, "accept csess")?;
        // SAFETY: thread s < S reads row0/slot of segment s (S checked), amax and pos of its rows (T checked), the K
        // drafts and the anchor / P of its slot q (slots below the caller's `top`, whose drafts / anchor / P extents
        // the caller checked), writes res[RS * s ..= RS * s + K + 1] and, thread 0, res[RS * S] (checked), and the
        // packed commit rows of the accepted rows (at most T, checked); `forced` is null.
        unsafe {
            self.k.ink_accept_kernel(
                self.st,
                cfg([1, 1, 1], 32, 0),
                InkAcceptKernelArgs {
                    amax: b(amax),
                    drafts: b(drafts),
                    T: i(t),
                    S: i(s),
                    row0: b(row0),
                    slot: b(slot),
                    pos: b(pos),
                    anchor: b(anchor),
                    P: b(p),
                    res: b(res),
                    crow: b(crow),
                    cpos: b(cpos),
                    csess: b(csess),
                    forced: Ptr::Null,
                    K: i(k),
                },
            )
        }
    }

    /// `ink::last_feat_kernel<<<S, 256>>>(feats, row0, slot, res, lastfeat, RS)`: each slot's last accepted row's
    /// features (`rs = accept_kernel`'s result stride `k + 2`).
    #[allow(clippy::too_many_arguments)]
    pub fn last_feat(
        &self,
        s: usize,
        t: usize,
        rs: usize,
        feats: &DevBuf<Bf16>,
        rows: (&DevBuf<i32>, &DevBuf<i32>),
        res: &DevBuf<i32>,
        lastfeat: &DevBuf<Bf16>,
    ) -> Result<()> {
        let (row0, slot) = rows;
        need(feats, t * 4 * H, "feats")?;
        need(row0, s, "row0")?;
        need(slot, s, "slot")?;
        need(res, rs * s + 1, "res")?;
        // SAFETY: block s reads row0[s] + res[rs * s] (an accepted row of segment s, below T: accept_kernel's contract)
        // of feats (T x 4 x H checked) and writes the feature row of slot[s] into lastfeat (the caller checked it
        // holds `top` rows, every slot of the call below `top`).
        unsafe {
            self.k.ink_last_feat_kernel(
                self.st,
                cfg([s, 1, 1], 256, 0),
                InkLastFeatKernelArgs {
                    feats: b(feats),
                    row0: b(row0),
                    slot: b(slot),
                    res: b(res),
                    lastfeat: b(lastfeat),
                    RS: i(rs),
                },
            )
        }
    }

    /// `ink::gather_feats_kernel<<<R, 256>>>(feats, rows, n, dst)`: feature rows `rows[i]` (i < *n when `n` is
    /// given) into `dst` row i.
    pub fn gather_feats(
        &self,
        r: usize,
        t: usize,
        feats: &DevBuf<Bf16>,
        rows: &DevBuf<i32>,
        n: Option<&DevBuf<i32>>,
        dst: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(feats, t * 4 * H, "feats")?;
        need(rows, r, "gather rows")?;
        need(dst, r * 4 * H, "gather dst")?;
        // SAFETY: block i < R reads rows[i] (R checked), a call row below T (the caller's rows: accept_kernel's
        // packed rows or call rows the caller names), copies that feature row of feats (T x 4 x H checked) into dst
        // row i (R x 4 x H checked); with `n`, blocks at or past *n return first.
        unsafe {
            self.k.ink_gather_feats_kernel(
                self.st,
                cfg([r, 1, 1], 256, 0),
                InkGatherFeatsKernelArgs {
                    feats: b(feats),
                    rows: b(rows),
                    n: opt(n),
                    dst: b(dst),
                },
            )
        }
    }

    /// `ink::verify_ids_kernel<<<ceil((k + 1) n / 256), 256>>>(anchor, drafts, vids, n, k)`: `[anchor, drafts]` per
    /// session (`k` drafts per session).
    pub fn verify_ids(
        &self,
        n: usize,
        k: usize,
        anchor: &DevBuf<i32>,
        drafts: &DevBuf<i32>,
        vids: &DevBuf<i32>,
    ) -> Result<()> {
        need(anchor, n, "anchor")?;
        need(drafts, k * n, "drafts")?;
        need(vids, (k + 1) * n, "vids")?;
        // SAFETY: thread i < (k + 1) n reads anchor[i / (k + 1)] or drafts[(i / (k + 1)) k + i % (k + 1) - 1] and
        // writes vids[i] (all checked); the kernel returns beyond.
        unsafe {
            self.k.ink_verify_ids_kernel(
                self.st,
                cfg([(n * (k + 1)).div_ceil(256), 1, 1], 256, 0),
                InkVerifyIdsKernelArgs {
                    anchor: b(anchor),
                    drafts: b(drafts),
                    vids: b(vids),
                    n: i(n),
                    K: i(k),
                },
            )
        }
    }

    /// `verify_width_kernel<<<1, 32>>>(conf, tau, n, width)` (`Engine::verify_widths`): the gated verify width of
    /// slots `0..n`.
    pub fn verify_widths(&self, n: usize, conf: &DevBuf<f32>, tau: f32, width: &DevBuf<i32>) -> Result<()> {
        need(conf, 7 * n, "conf")?;
        need(width, n, "widths")?;
        // SAFETY: thread q < n <= 32 reads conf[q][0..7] and writes width[q] (both checked); the kernel returns beyond.
        unsafe {
            self.k.ink_verify_width_kernel(
                self.st,
                cfg([1, 1, 1], 32, 0),
                InkVerifyWidthKernelArgs {
                    margin: b(conf),
                    tau,
                    n: i(n),
                    width: b(width),
                },
            )
        }
    }

    /// `verify_width_pack_kernel<<<1, 64>>>(conf, drafts, tau, n, out)`: packs `[widths[0..n], drafts[0..7*n]]` into
    /// one contiguous device buffer for single-sync D2H download.
    pub fn verify_widths_pack(
        &self,
        n: usize,
        conf: Option<&DevBuf<f32>>,
        drafts: Option<&DevBuf<i32>>,
        tau: f32,
        out: &DevBuf<i32>,
    ) -> Result<()> {
        if let Some(c) = conf {
            need(c, 7 * n, "conf")?;
        }
        if let Some(d) = drafts {
            need(d, 7 * n, "drafts")?;
            need(out, 8 * n, "packed widths+drafts")?;
        } else {
            need(out, n, "widths")?;
        }
        // SAFETY: threads compute widths[0..n] (or 8 if conf is null) into out[0..n] and copy drafts[0..7*n] into
        // out[n..8*n] (all extents checked above).
        unsafe {
            self.k.ink_verify_width_pack_kernel(
                self.st,
                cfg([1, 1, 1], 64, 0),
                InkVerifyWidthPackKernelArgs {
                    margin: opt(conf),
                    drafts: opt(drafts),
                    tau,
                    n: i(n),
                    out: b(out),
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_memory_sizes_equal_the_cpp_constants() {
        // values the C++ oracle trace shows for these kernels (bench/raw/2026-11-23-oracle-native-v2 e2 traces)
        assert_eq!(DA_SMEM, 55_296);
        assert_eq!(DA_RING_SMEM, 38_400);
        assert_eq!(MOE_SMEM_NVFP4, 80_896);
        assert_eq!(MOE_SMEM_NVFP4_N1S1, 77_824);
        assert_eq!(MOE_SMEM_MXFP8, 100_864);
        assert_eq!(MOE_DEC_SMEM_N1S1, 33_408);
        assert_eq!(MOE_DEC_SMEM_MXFP8, 52_608);
        // Cfg<4, 2, 1, 1>: NT 128, BR 128, TILE 8192, PT 1792, Q 16384, BSCRATCH 4736, MAIN 34560, SMEM 78608
        assert_eq!(PFA_SMEM, 78_608);
        assert_eq!(PFA_FP8_SMEM, 62_224);
        assert_eq!(grid1(4 * 2304), 36);
        assert_eq!(grid1(10_000_000), 74 * 16);
    }
}
