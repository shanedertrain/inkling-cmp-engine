// Ported from engine/native/src/drafter_rt.cuh @ 1c7ac46 (every `<<<` launch site of `load_weights`, `Drafter`'s
// constructor, `commit_rows`, `layers`, `head_gemm`, `head_rest`; the GEMM launch geometry of `gemm`, `gemm_t`,
// `gemm_w` at the defaults the C++ drafter runs with) and engine/native/src/draft.cu (`DraftRT::frnn`). Own code.
//! Every kernel launch of the drafter, one safe function per C++ launch site: the drafter's second module with
//! `#[allow(unsafe_code)]` beside `engine::ops`, under the same rule (docs/RUST-PLAN.md 3.2, review R3 deviation
//! 4). Each function checks the extents the host can bound, computes the launch geometry as the C++ does and states
//! the kernel's contract (`engine/kernels/drafter/drafter_kernels.cuh`, its README) in a `// SAFETY:` comment.
//! Extents that depend on device data (positions, sessions and anchors the caller wrote, the page table this
//! drafter built) are stated, not checked: positions below `max_blocks * 64 - 8`, sessions below `n`, anchors below
//! the padded vocabulary.
#![allow(unsafe_code)]

use bytemuck::Pod;
use ink_cuda::{Bf16, CudaError, DevBuf, LaunchCfg, Result, Stream};
use ink_kernels::*;
use ink_model::drafter::{ATTN_SMEM, CONF_NCH, H, HD, INTER, KVH, MK_NCH, MK3_NCH, NH, VP, VU, XS};

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

fn b<T>(x: &DevBuf<T>) -> Ptr<'_, T> {
    Ptr::Buf(x)
}

fn opt<T>(x: Option<&DevBuf<T>>) -> Ptr<'_, T> {
    x.map_or(Ptr::Null, Ptr::Buf)
}

/// Dynamic shared memory of `gemm_frag<MT, ...>`: two staged x chunks (`gemm_smem`).
pub(super) fn gemm_smem(mt: usize) -> u32 {
    u(2 * mt * 8 * XS * 2)
}

/// Epilogue of the fragment GEMM (`enum Epi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Epi {
    /// fp32 split-K partials (`EPI_F32`, bf16 weights).
    F32,
    /// bf16 output (`EPI_BF16`, bf16 weights: the exact head).
    Bf16,
    /// fp32 output scaled per row (`EPI_HEADQ`, int8 weights: the int8 head).
    HeadQ,
}

/// The last layer's head outputs: `hout`, `hf16`, `x7` and the `f_rnn` it reads, `hnorm`.
pub(super) type HeadOut<'a> = (
    &'a DevBuf<Bf16>,
    &'a DevBuf<F16>,
    &'a DevBuf<Bf16>,
    &'a DevBuf<Bf16>,
    &'a DevBuf<f32>,
);

/// The kernel and stream every launch needs.
pub(super) struct Dev<'a> {
    pub k: &'a Kernels,
    pub st: &'a Stream,
}

/// Every `gemm_frag<MT, I8, EPI, 8>` instance the drafter launches, by M tile count and epilogue.
macro_rules! gemm_frag {
    ($k:expr, $st:expr, $c:expr, $a:expr, $mt:expr, $epi:expr) => {
        match ($mt, $epi) {
            (1, Epi::F32) => $k.s2f_gemm_frag_1_false_0_8($st, $c, $a),
            (2, Epi::F32) => $k.s2f_gemm_frag_2_false_0_8($st, $c, $a),
            (3, Epi::F32) => $k.s2f_gemm_frag_3_false_0_8($st, $c, $a),
            (4, Epi::F32) => $k.s2f_gemm_frag_4_false_0_8($st, $c, $a),
            (5, Epi::F32) => $k.s2f_gemm_frag_5_false_0_8($st, $c, $a),
            (6, Epi::F32) => $k.s2f_gemm_frag_6_false_0_8($st, $c, $a),
            (7, Epi::F32) => $k.s2f_gemm_frag_7_false_0_8($st, $c, $a),
            (8, Epi::F32) => $k.s2f_gemm_frag_8_false_0_8($st, $c, $a),
            (1, Epi::Bf16) => $k.s2f_gemm_frag_1_false_1_8($st, $c, $a),
            (2, Epi::Bf16) => $k.s2f_gemm_frag_2_false_1_8($st, $c, $a),
            (3, Epi::Bf16) => $k.s2f_gemm_frag_3_false_1_8($st, $c, $a),
            (4, Epi::Bf16) => $k.s2f_gemm_frag_4_false_1_8($st, $c, $a),
            (5, Epi::Bf16) => $k.s2f_gemm_frag_5_false_1_8($st, $c, $a),
            (6, Epi::Bf16) => $k.s2f_gemm_frag_6_false_1_8($st, $c, $a),
            (7, Epi::Bf16) => $k.s2f_gemm_frag_7_false_1_8($st, $c, $a),
            (8, Epi::Bf16) => $k.s2f_gemm_frag_8_false_1_8($st, $c, $a),
            (1, Epi::HeadQ) => $k.s2f_gemm_frag_1_true_2_8($st, $c, $a),
            (2, Epi::HeadQ) => $k.s2f_gemm_frag_2_true_2_8($st, $c, $a),
            (3, Epi::HeadQ) => $k.s2f_gemm_frag_3_true_2_8($st, $c, $a),
            (4, Epi::HeadQ) => $k.s2f_gemm_frag_4_true_2_8($st, $c, $a),
            (5, Epi::HeadQ) => $k.s2f_gemm_frag_5_true_2_8($st, $c, $a),
            (6, Epi::HeadQ) => $k.s2f_gemm_frag_6_true_2_8($st, $c, $a),
            (7, Epi::HeadQ) => $k.s2f_gemm_frag_7_true_2_8($st, $c, $a),
            (8, Epi::HeadQ) => $k.s2f_gemm_frag_8_true_2_8($st, $c, $a),
            _ => Err(CudaError::Bounds {
                what: "drafter gemm",
                detail: format!("M tile count {} (M > 64)", $mt),
            }),
        }
    };
}

/// The kernel ids of the `gemm_frag` instances above (their shared memory attribute is set at start).
pub(super) fn gemm_frag_ids() -> Vec<(KernelId, usize)> {
    use KernelId as K;
    let ids = [
        [K::S2fGemmFrag1False08, K::S2fGemmFrag1False18, K::S2fGemmFrag1True28],
        [K::S2fGemmFrag2False08, K::S2fGemmFrag2False18, K::S2fGemmFrag2True28],
        [K::S2fGemmFrag3False08, K::S2fGemmFrag3False18, K::S2fGemmFrag3True28],
        [K::S2fGemmFrag4False08, K::S2fGemmFrag4False18, K::S2fGemmFrag4True28],
        [K::S2fGemmFrag5False08, K::S2fGemmFrag5False18, K::S2fGemmFrag5True28],
        [K::S2fGemmFrag6False08, K::S2fGemmFrag6False18, K::S2fGemmFrag6True28],
        [K::S2fGemmFrag7False08, K::S2fGemmFrag7False18, K::S2fGemmFrag7True28],
        [K::S2fGemmFrag8False08, K::S2fGemmFrag8False18, K::S2fGemmFrag8True28],
    ];
    ids.iter()
        .enumerate()
        .flat_map(|(m, row)| row.iter().map(move |&id| (id, m + 1)))
        .collect()
}

/// One fragment GEMM's operands (`ga(W, x, ldx, M, N, K, part, out, scale)`, `ldo` = N unless set).
pub(super) struct Gemm<'a, W: Pod> {
    pub w: &'a DevBuf<W>,
    pub x: &'a DevBuf<Bf16>,
    pub ldx: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub part: Option<&'a DevBuf<f32>>,
    pub out: Option<&'a DevBuf<Bf16>>,
    pub scale: Option<&'a DevBuf<f32>>,
    pub ldo: usize,
}

impl Dev<'_> {
    // ------------------------------------------------------------------------------------------------ load

    /// `permute_frag_bf16<<<ceil(N K / 8 / 256), 256>>>(w, out, N, K)` (`to_frag`).
    pub fn permute_frag_bf16(&self, w: &DevBuf<Bf16>, out: &DevBuf<Bf16>, n: usize, k: usize) -> Result<()> {
        need(w, n * k, "frag source")?;
        need(out, n * k, "frag out")?;
        let th = n * k / 8;
        // SAFETY: thread t < N K / 8 reads 8 elements of the N x K row-major w and writes 8 of out (both N K,
        // checked); the grid covers N K / 8 threads, the kernel returns beyond.
        unsafe {
            self.k.s2f_permute_frag_bf16(
                self.st,
                cfg([th.div_ceil(256), 1, 1], 256, 0),
                S2fPermuteFragBf16Args {
                    w: b(w),
                    out: b(out),
                    N: i(n),
                    K: i(k),
                },
            )
        }
    }

    /// `permute_frag_i8<<<ceil(N K / 8 / 256), 256>>>(q, out, N, K)` (int8 head fragments).
    pub fn permute_frag_i8(&self, w: &DevBuf<i8>, out: &DevBuf<i8>, n: usize, k: usize) -> Result<()> {
        need(w, n * k, "frag source")?;
        need(out, n * k, "frag out")?;
        let th = n * k / 8;
        // SAFETY: as permute_frag_bf16, int8 elements.
        unsafe {
            self.k.s2f_permute_frag_i8(
                self.st,
                cfg([th.div_ceil(256), 1, 1], 256, 0),
                S2fPermuteFragI8Args {
                    w: b(w),
                    out: b(out),
                    N: i(n),
                    K: i(k),
                },
            )
        }
    }

    /// `row_norms<<<N / 8, 256>>>(w, out, N, K)`.
    pub fn row_norms(&self, w: &DevBuf<Bf16>, out: &DevBuf<f32>, n: usize, k: usize) -> Result<()> {
        need(w, n * k, "row_norms w")?;
        need(out, n, "row_norms out")?;
        // SAFETY: warp per row (8 per block, N / 8 blocks, N a multiple of 8): reads row r of the N x K w, writes
        // out[r] (both checked).
        unsafe {
            self.k.s2f_row_norms(
                self.st,
                cfg([n / 8, 1, 1], 256, 0),
                S2fRowNormsArgs {
                    w: b(w),
                    out: b(out),
                    N: i(n),
                    K: i(k),
                },
            )
        }
    }

    /// `row_norms_i8<<<N / 8, 256>>>(q, out, N, K)`.
    pub fn row_norms_i8(&self, q: &DevBuf<i8>, out: &DevBuf<f32>, n: usize, k: usize) -> Result<()> {
        need(q, n * k, "row_norms_i8 q")?;
        need(out, n, "row_norms_i8 out")?;
        // SAFETY: as row_norms, int8 rows.
        unsafe {
            self.k.s2f_row_norms_i8(
                self.st,
                cfg([n / 8, 1, 1], 256, 0),
                S2fRowNormsI8Args {
                    q: b(q),
                    out: b(out),
                    N: i(n),
                    K: i(k),
                },
            )
        }
    }

    /// `quant_rows_i8<<<N, 256>>>(w, q, scale, enorm, K)`: per-row int8 with its scale and exact error norm.
    pub fn quant_rows_i8(
        &self,
        w: &DevBuf<Bf16>,
        q: &DevBuf<i8>,
        scale: &DevBuf<f32>,
        enorm: &DevBuf<f32>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        need(w, n * k, "quant w")?;
        need(q, n * k, "quant q")?;
        need(scale, n, "quant scale")?;
        need(enorm, n, "quant enorm")?;
        // SAFETY: block r < N reads row r of w (K elements), writes row r of q and scale[r], enorm[r] (checked).
        unsafe {
            self.k.s2f_quant_rows_i8(
                self.st,
                cfg([n, 1, 1], 256, 0),
                S2fQuantRowsI8Args {
                    w: b(w),
                    q: b(q),
                    scale: b(scale),
                    enorm: b(enorm),
                    K: i(k),
                },
            )
        }
    }

    /// `k_cos_sin<<<maxpos x 64 / 256, 256>>>(cs, maxpos)`: the RoPE table (vLLM `_compute_cos_sin_cache`, fp32).
    pub fn cos_sin(&self, cs: &DevBuf<f32>, maxpos: usize) -> Result<()> {
        need(cs, maxpos * 128, "cos_sin")?;
        // SAFETY: thread t < maxpos x 64 writes cs[pos][d] and cs[pos][64 + d] (maxpos x 128 checked); the kernel
        // returns beyond.
        unsafe {
            self.k.k_cos_sin(
                self.st,
                cfg([maxpos * 64 / 256, 1, 1], 256, 0),
                KCosSinArgs {
                    cs: b(cs),
                    maxpos: i(maxpos),
                },
            )
        }
    }

    /// `k_fill_bf16<<<ceil(n / 256), 256>>>(p, n, seed, scale)`: normal noise (the initial draft K/V).
    pub fn fill_bf16(&self, p: &DevBuf<Bf16>, seed: u64, scale: f32) -> Result<()> {
        let n = p.len();
        // SAFETY: thread t < n writes p[t] (n = the buffer's length); the kernel returns beyond.
        unsafe {
            self.k.k_fill_bf16(
                self.st,
                cfg([n.div_ceil(256), 1, 1], 256, 0),
                KFillBf16Args {
                    p: b(p),
                    n: n as u64,
                    seed,
                    scale,
                },
            )
        }
    }

    /// `k_fill_e4m3<<<ceil(n / 256), 256>>>(p, n, seed)`: E4M3 noise (the initial global-layer K/V copy).
    pub fn fill_e4m3(&self, p: &DevBuf<u8>, seed: u64) -> Result<()> {
        let n = p.len();
        // SAFETY: thread t < n writes p[t] (n = the buffer's length); the kernel returns beyond.
        unsafe {
            self.k.k_fill_e4m3(
                self.st,
                cfg([n.div_ceil(256), 1, 1], 256, 0),
                KFillE4m3Args {
                    p: b(p),
                    n: n as u64,
                    seed,
                },
            )
        }
    }

    // ------------------------------------------------------------------------------------------------ GEMM

    /// `gemm<I8, EPI>(ga(...), sk)`: split-K `sk`, `gemm_frag<ceil(M / 8), I8, EPI, 8><<<(ceil(N / 16 / 8), sk),
    /// 256, gemm_smem(MT)>>>(a)`; `a.kslice = K / sk`, which must be a multiple of 16 and divide K exactly.
    pub fn gemm<W: Pod>(&self, epi: Epi, g: &Gemm<'_, W>, sk: usize) -> Result<()> {
        let kslice = g.k / sk;
        if !kslice.is_multiple_of(16) || kslice * sk != g.k {
            return Err(CudaError::Bounds {
                what: "drafter gemm",
                detail: format!("bad split: K {} into {sk}", g.k),
            });
        }
        let mt = g.m.div_ceil(8);
        need(g.x, (g.m - 1) * g.ldx + g.k, "gemm x")?;
        if g.w.bytes() < g.n * g.k * if epi == Epi::HeadQ { 1 } else { 2 } {
            return Err(CudaError::Bounds {
                what: "gemm W",
                detail: format!("{} bytes for {} x {}", g.w.bytes(), g.n, g.k),
            });
        }
        match epi {
            Epi::F32 => need(
                g.part
                    .ok_or_else(|| CudaError::Graph("F32 gemm without partials".into()))?,
                sk * g.m * g.n,
                "gemm part",
            )?,
            Epi::Bf16 => need(
                g.out.ok_or_else(|| CudaError::Graph("bf16 gemm without out".into()))?,
                (g.m - 1) * g.ldo + g.n,
                "gemm out",
            )?,
            Epi::HeadQ => {
                need(
                    g.part.ok_or_else(|| CudaError::Graph("int8 gemm without out".into()))?,
                    (g.m - 1) * g.ldo + g.n,
                    "gemm out",
                )?;
                need(
                    g.scale
                        .ok_or_else(|| CudaError::Graph("int8 gemm without scales".into()))?,
                    g.n,
                    "gemm scale",
                )?;
            }
        }
        let a = s2f::GemmArgs {
            W: g.w.cast::<u8>()?.ptr(),
            x: g.x.ptr(),
            ldx: i(g.ldx),
            M: i(g.m),
            N: i(g.n),
            K: i(g.k),
            kslice: i(kslice),
            ldo: i(g.ldo),
            part: g.part.map_or(bytemuck::Zeroable::zeroed(), DevBuf::ptr),
            out: g.out.map_or(bytemuck::Zeroable::zeroed(), DevBuf::ptr),
            scale: g.scale.map_or(bytemuck::Zeroable::zeroed(), DevBuf::ptr),
        };
        let c = cfg([(g.n / 16).div_ceil(8), sk, 1], 256, gemm_smem(mt));
        // SAFETY: the fragment GEMM reads W (N x K in fragment layout, checked), x rows 0..M with stride ldx (checked)
        // and writes, per epilogue: fp32 partials part[sk][M][N] (F32), bf16 out[M][ldo] (Bf16), or fp32
        // part-as-out[M][ldo] scaled by scale[N] (HeadQ) (checked); M <= 64 (MT <= 8, else refused below); the
        // shared memory attribute of every instance was set at start.
        unsafe { gemm_frag!(self.k, self.st, c, &a, mt, epi) }
    }

    // ------------------------------------------------------------------------------------------------ commit

    /// `k_commit_mix<<<(R, 3), 256>>>(feats, alpha, ctx_norm x 3, f_norm, cmix, f_rnn, R)`: the three layers' mixes of
    /// R committed rows (the f_rnn block, y = 3, is not launched).
    #[allow(clippy::too_many_arguments)]
    pub fn commit_mix(
        &self,
        feats: &DevBuf<Bf16>,
        alpha: &DevBuf<f32>,
        ctx_norm: [&DevBuf<Bf16>; 3],
        f_norm: &DevBuf<Bf16>,
        cmix: &DevBuf<Bf16>,
        f_rnn: &DevBuf<Bf16>,
        rows: usize,
    ) -> Result<()> {
        need(feats, rows * 4 * H, "commit feats")?;
        need(alpha, 16, "alpha")?;
        need(cmix, 3 * rows * H, "cmix")?;
        // SAFETY: block (row, l), l < 3, reads feats[row][0..4][:] (R x 4 x H checked), alpha row l and ctx_norm[l]
        // (H), writes cmix[l][row][:] (3 x R x H checked); f_rnn is written only by y = 3, not launched.
        unsafe {
            self.k.s2f_k_commit_mix(
                self.st,
                cfg([rows, 3, 1], 256, 0),
                S2fKCommitMixArgs {
                    feats: b(feats),
                    alpha: b(alpha),
                    w0: b(ctx_norm[0]),
                    w1: b(ctx_norm[1]),
                    w2: b(ctx_norm[2]),
                    wf: b(f_norm),
                    cmix: b(cmix),
                    f_rnn: b(f_rnn),
                    T: i(rows),
                },
            )
        }
    }

    /// `epi_kvctx<<<(R, 3), 512>>>(part, sk, R, st, k_norm_ctx x 3, cos_sin, cc, row_pos, row_sess)`: k_norm_ctx and
    /// RoPE on the context K, both written to each layer's draft K/V at the rows' positions.
    #[allow(clippy::too_many_arguments)]
    pub fn epi_kvctx(
        &self,
        part: &DevBuf<f32>,
        sk: usize,
        rows: usize,
        st: s2f::Step,
        kn: [&DevBuf<Bf16>; 3],
        cs: &DevBuf<f32>,
        cc: s2f::Caches,
        row_pos: &DevBuf<i32>,
        row_sess: &DevBuf<i32>,
    ) -> Result<()> {
        need(part, 3 * sk * rows * 2048, "kvctx part")?;
        need(row_pos, rows, "row positions")?;
        need(row_sess, rows, "row sessions")?;
        // SAFETY: block (row, l) reads part[l][0..sk][row][:2048] (checked), row_pos[row], row_sess[row] (checked),
        // the RoPE row of the position (positions below the table's 270,336) and writes one token of each KV head of
        // layer l's K and V (and E4M3 copies when set) at the page `block_table[sess][pos / 64]` of this drafter's
        // table: positions below max_blocks x 64 and sessions below n are the caller's (module docs).
        unsafe {
            self.k.s2f_epi_kvctx(
                self.st,
                cfg([rows, 3, 1], 512, 0),
                S2fEpiKvctxArgs {
                    part: b(part),
                    SK: i(sk),
                    T: i(rows),
                    st,
                    kn0: b(kn[0]),
                    kn1: b(kn[1]),
                    kn2: b(kn[2]),
                    cs: b(cs),
                    cc,
                    row_pos: b(row_pos),
                    row_sess: b(row_sess),
                },
            )
        }
    }

    /// `k_frnn<<<n, 256>>>(feats_rows, alpha, f_norm, f_rnn)` (`DraftRT::frnn`).
    pub fn frnn(
        &self,
        n: usize,
        feats: &DevBuf<Bf16>,
        alpha: &DevBuf<f32>,
        f_norm: &DevBuf<Bf16>,
        f_rnn: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(feats, n * 4 * H, "frnn feats")?;
        need(alpha, 16, "alpha")?;
        need(f_rnn, n * H, "f_rnn")?;
        // SAFETY: block s < n reads feats[s][0..4][:] (checked), alpha row 3, f_norm (H), writes f_rnn[s][:] (checked).
        unsafe {
            self.k.s2f_k_frnn(
                self.st,
                cfg([n, 1, 1], 256, 0),
                S2fKFrnnArgs {
                    feats_rows: b(feats),
                    alpha: b(alpha),
                    wf: b(f_norm),
                    f_rnn: b(f_rnn),
                },
            )
        }
    }

    // ------------------------------------------------------------------------------------------------ layers

    /// `k_embed_block<<<T, 256>>>(st, embed, embed_norm, mask_embed, in_norm0, residual, xn)`.
    #[allow(clippy::too_many_arguments)]
    pub fn embed_block(
        &self,
        t: usize,
        st: s2f::Step,
        embed: &DevBuf<Bf16>,
        embed_norm: &DevBuf<Bf16>,
        mask_embed: &DevBuf<Bf16>,
        in_norm0: &DevBuf<Bf16>,
        residual: &DevBuf<Bf16>,
        xn: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(embed, VP * H, "embed")?;
        need(residual, t * H, "residual")?;
        need(xn, t * H, "xn")?;
        // SAFETY: block row < T = 8 n reads the anchor of session row / 8 (st.anchor[n], this drafter's) and its
        // embedding row (anchors below the padded vocabulary: the caller's), the norms (H), writes residual and xn
        // row `row` (T x H checked).
        unsafe {
            self.k.s2f_k_embed_block(
                self.st,
                cfg([t, 1, 1], 256, 0),
                S2fKEmbedBlockArgs {
                    st,
                    embed: b(embed),
                    embed_norm_w: b(embed_norm),
                    mask_embed: b(mask_embed),
                    in_norm0: b(in_norm0),
                    residual: b(residual),
                    xn: b(xn),
                },
            )
        }
    }

    /// `epi_qkv<<<T, 1024>>>(part, sk, T, st, q_norm, k_norm, cos_sin, cache, qbuf)`: q/k norm and RoPE; the
    /// block's own K/V written at P..P+7 of each session.
    #[allow(clippy::too_many_arguments)]
    pub fn epi_qkv(
        &self,
        part: &DevBuf<f32>,
        sk: usize,
        t: usize,
        st: s2f::Step,
        q_norm: &DevBuf<Bf16>,
        k_norm: &DevBuf<Bf16>,
        cs: &DevBuf<f32>,
        cache: s2f::CacheL,
        qbuf: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(part, sk * t * (NH + 2 * KVH) * HD, "qkv part")?;
        need(qbuf, t * NH * HD, "qbuf")?;
        // SAFETY: block row < T reads part[0..sk][row][:5120] (checked), the norms (HD), the RoPE row of P + row % 8,
        // writes qbuf row `row` (checked) and one token of each KV head of `cache` at P[row / 8] + row % 8 through
        // this drafter's page table (P below max_blocks x 64 - 8: the caller's, module docs).
        unsafe {
            self.k.s2f_epi_qkv(
                self.st,
                cfg([t, 1, 1], 1024, 0),
                S2fEpiQkvArgs {
                    part: b(part),
                    SK: i(sk),
                    T: i(t),
                    st,
                    qn: b(q_norm),
                    kn: b(k_norm),
                    cs: b(cs),
                    cache,
                    qout: b(qbuf),
                },
            )
        }
    }

    /// `attn_block<<<(splits, KVH, n), 128, ATTN_SMEM>>>(q, st, cache, window, splits, o_part, ml)` (`attn_block8`
    /// for the E4M3 global layer).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_block(
        &self,
        e4m3: bool,
        n: usize,
        q: &DevBuf<Bf16>,
        st: s2f::Step,
        cache: s2f::CacheL,
        window: usize,
        splits: usize,
        o_part: &DevBuf<f32>,
        ml: &DevBuf<f32>,
    ) -> Result<()> {
        need(q, n * 8 * NH * HD, "attn q")?;
        need(o_part, n * KVH * splits * 32 * HD, "attn o_part")?;
        need(ml, n * KVH * splits * 32 * 2, "attn ml")?;
        let c = cfg([splits, KVH, n], 128, ATTN_SMEM);
        let (qp, op, mp, w, s) = (b(q), b(o_part), b(ml), i(window), i(splits));
        // SAFETY: CTA (split, kv head, session) reads the session's 8 query rows (checked) and its keys 0..P+8 (or the
        // last window + 8) through this drafter's page table (at most 256 pages per CTA: the split count makes it
        // so), writes its partials and (max, sum) pairs (n x KVH x splits x 32 rows, checked); ATTN_SMEM is set as
        // the kernel's dynamic shared memory at start.
        unsafe {
            if e4m3 {
                self.k.s2f_attn_block8(
                    self.st,
                    c,
                    S2fAttnBlock8Args {
                        q: qp,
                        st,
                        cache,
                        window: w,
                        splits: s,
                        o_part: op,
                        ml: mp,
                    },
                )
            } else {
                self.k.s2f_attn_block(
                    self.st,
                    c,
                    S2fAttnBlockArgs {
                        q: qp,
                        st,
                        cache,
                        window: w,
                        splits: s,
                        o_part: op,
                        ml: mp,
                    },
                )
            }
        }
    }

    /// `attn_combine<<<n x KVH x 24, 128>>>(o_part, ml, splits, out)`.
    pub fn attn_combine(
        &self,
        n: usize,
        o_part: &DevBuf<f32>,
        ml: &DevBuf<f32>,
        splits: usize,
        out: &DevBuf<Bf16>,
    ) -> Result<()> {
        need(o_part, n * KVH * splits * 32 * HD, "combine o_part")?;
        need(ml, n * KVH * splits * 32 * 2, "combine ml")?;
        need(out, n * 8 * NH * HD, "attn out")?;
        // SAFETY: block (session, kv head, row of 24) reads the splits' partials of its rows (checked) and writes its
        // output row slice of out (T x NH x HD checked).
        unsafe {
            self.k.s2f_attn_combine(
                self.st,
                cfg([n * KVH * 24, 1, 1], 128, 0),
                S2fAttnCombineArgs {
                    o_part: b(o_part),
                    ml: b(ml),
                    splits: i(splits),
                    out: b(out),
                },
            )
        }
    }

    /// `epi_addnorm<<<T, 768>>>(part, sk, T, residual, w, xn, hout, hf16, x7, f_rnn, hnorm)`: residual += sum of
    /// the partials, then the norm into `xn`, or (last layer) the head inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn epi_addnorm(
        &self,
        part: &DevBuf<f32>,
        sk: usize,
        t: usize,
        residual: &DevBuf<Bf16>,
        w: &DevBuf<Bf16>,
        xn: Option<&DevBuf<Bf16>>,
        head: Option<HeadOut<'_>>,
    ) -> Result<()> {
        need(part, sk * t * H, "addnorm part")?;
        need(residual, t * H, "residual")?;
        if let Some(x) = xn {
            need(x, t * H, "xn")?;
        }
        if let Some((hout, hf16, x7, f_rnn, hnorm)) = head {
            let n = t / 8;
            need(hout, n * 7 * H, "hout")?;
            need(hf16, n * 7 * H, "hf16")?;
            need(x7, n * 7 * 2 * H, "x7")?;
            need(f_rnn, n * H, "f_rnn")?;
            need(hnorm, n * 7, "hnorm")?;
        }
        let (hout, hf16, x7, f_rnn, hnorm) = match head {
            Some((a, b_, c, d, e)) => (b(a), b(b_), b(c), b(d), b(e)),
            None => (Ptr::Null, Ptr::Null, Ptr::Null, Ptr::Null, Ptr::Null),
        };
        // SAFETY: block row < T reads part[0..sk][row][:H] (checked), updates residual row `row`, writes xn row `row`
        // when given; with the head outputs (last layer), mask rows (row % 8 >= 1) write hout, hf16, x7 = [h, f_rnn]
        // and hnorm of head row (row / 8) x 7 + row % 8 - 1 (n x 7 rows checked), reading f_rnn row row / 8.
        unsafe {
            self.k.s2f_epi_addnorm(
                self.st,
                cfg([t, 1, 1], 768, 0),
                S2fEpiAddnormArgs {
                    part: b(part),
                    SK: i(sk),
                    T: i(t),
                    residual: b(residual),
                    w: b(w),
                    xn: opt(xn),
                    hout,
                    hf16,
                    x7,
                    f_rnn,
                    hnorm,
                },
            )
        }
    }

    /// `epi_silu<<<ceil(T INTER / 256), 256>>>(part, sk, T, out)`.
    pub fn epi_silu(&self, part: &DevBuf<f32>, sk: usize, t: usize, out: &DevBuf<Bf16>) -> Result<()> {
        need(part, sk * t * 2 * INTER, "silu part")?;
        need(out, t * INTER, "inter")?;
        // SAFETY: thread e < T x INTER reads gate and up partials of element e (sk x T x 2 INTER checked), writes
        // out[e] (checked); the kernel returns beyond.
        unsafe {
            self.k.s2f_epi_silu(
                self.st,
                cfg([(t * INTER).div_ceil(256), 1, 1], 256, 0),
                S2fEpiSiluArgs {
                    part: b(part),
                    SK: i(sk),
                    T: i(t),
                    out: b(out),
                },
            )
        }
    }

    // ------------------------------------------------------------------------------------------------ head

    /// `epi_ctx<<<ceil(M 768 / 256), 256>>>(part, sk, M, rnn_b, ctx)`.
    pub fn epi_ctx(
        &self,
        part: &DevBuf<f32>,
        sk: usize,
        m: usize,
        rnn_b: &DevBuf<f32>,
        ctx: &DevBuf<f32>,
    ) -> Result<()> {
        need(part, sk * m * 768, "ctx part")?;
        need(rnn_b, 768, "rnn_b")?;
        need(ctx, m * 768, "ctx")?;
        // SAFETY: thread e < M x 768 sums the partials of element e (checked), adds the bias, writes ctx[e]
        // (checked); the kernel returns beyond.
        unsafe {
            self.k.s2f_epi_ctx(
                self.st,
                cfg([(m * 768).div_ceil(256), 1, 1], 256, 0),
                S2fEpiCtxArgs {
                    part: b(part),
                    SK: i(sk),
                    M: i(m),
                    rnn_b: b(rnn_b),
                    ctx: b(ctx),
                },
            )
        }
    }

    // The Markov launches take the `Markov` struct the drafter builds from its own buffers (`Drafter::markov`):
    // every pointer in it names a buffer of the drafter sized for its n sessions (state `[2][n][256]`, keys `[7][n]`,
    // seed keys `[7 n][98]`, drafts / nsurv / conf `[n][7]`, head outputs `[n 7][201,024]`, the weights), and the
    // drafter keeps them alive longer than its graph.

    /// `markov_seed<I8><<<(ceil(VU / 2048), n 7), 256>>>(mk)`.
    pub fn markov_seed(&self, int8: bool, n: usize, mk: &s2f::Markov) -> Result<()> {
        let c = cfg([VU.div_ceil(2048), n * 7, 1], 256, 0);
        // SAFETY: block (chunk, draft row) reads its 2,048 head output rows of the draft row and the fold bias, writes
        // the row's seed key of the chunk (module comment above: buffers of n sessions).
        unsafe {
            if int8 {
                self.k.s2f_markov_seed_true(self.st, c, mk)
            } else {
                self.k.s2f_markov_seed_false(self.st, c, mk)
            }
        }
    }

    /// `markov_recur<<<32, 256>>>(mk, i)`.
    pub fn markov_recur(&self, mk: &s2f::Markov, step: usize) -> Result<()> {
        // SAFETY: step < 7; reads the context and the state of step i of every session, writes state slot (i + 1) % 2
        // and the latent (module comment: buffers of n sessions).
        unsafe {
            self.k.s2f_markov_recur(
                self.st,
                cfg([32, 1, 1], 256, 0),
                S2fMarkovRecurArgs { mk: *mk, i: i(step) },
            )
        }
    }

    /// `markov_zref<I8><<<(ceil(MK_NCH / 8), n), 256>>>(mk, i)`.
    pub fn markov_zref(&self, int8: bool, n: usize, mk: &s2f::Markov, step: usize) -> Result<()> {
        let c = cfg([MK_NCH.div_ceil(8), n, 1], 256, 0);
        let a = (*mk, i(step));
        // SAFETY: block (8 seeds, session) evaluates the exact z of its seeds of step i and writes the step's
        // reference key (module comment: buffers of n sessions).
        unsafe {
            if int8 {
                self.k
                    .s2f_markov_zref_true(self.st, c, S2fMarkovZrefTrueArgs { mk: a.0, i: a.1 })
            } else {
                self.k
                    .s2f_markov_zref_false(self.st, c, S2fMarkovZrefFalseArgs { mk: a.0, i: a.1 })
            }
        }
    }

    /// `markov_argmax3<I8><<<(MK3_NCH, n), 256>>>(mk, i)`.
    pub fn markov_argmax3(&self, int8: bool, n: usize, mk: &s2f::Markov, step: usize) -> Result<()> {
        let c = cfg([MK3_NCH, n, 1], 256, 0);
        let a = (*mk, i(step));
        // SAFETY: block (chunk, session) prunes and evaluates its chunk's rows of step i (rows below 200,058), takes
        // the step's key max (atomic) and counts survivors (module comment: buffers of n sessions).
        unsafe {
            if int8 {
                self.k
                    .s2f_markov_argmax3_true(self.st, c, S2fMarkovArgmax3TrueArgs { mk: a.0, i: a.1 })
            } else {
                self.k
                    .s2f_markov_argmax3_false(self.st, c, S2fMarkovArgmax3FalseArgs { mk: a.0, i: a.1 })
            }
        }
    }

    /// `conf_save_kernel<<<n, 32>>>(mk, i, cb)` (draft_conf.cuh): after step i's argmax, the rank-16 projection, the
    /// draft and its exact z.
    pub fn conf_save(&self, n: usize, mk: &s2f::Markov, step: usize, cb: &s2f::ConfBuf) -> Result<()> {
        // SAFETY: block r < n reads proj[r] and keys[i][r] (buffers of n sessions, module comment above) and writes
        // pr[i][r][0..16], zd[i][r], vd[i][r] of the drafter's ConfBuf (7 x n rows).
        unsafe {
            self.k.s2f_conf_save_kernel(
                self.st,
                cfg([n, 1, 1], 32, 0),
                S2fConfSaveKernelArgs {
                    mk: *mk,
                    i: i(step),
                    cb: *cb,
                },
            )
        }
    }

    /// `conf_sum_kernel<N><<<(CONF_NCH, ksteps), 256>>>(mk, cb)` (draft_conf.cuh, `conf_sum_launch`): per step and
    /// 1,024-row chunk the partial sums of exp(z - 30) of every row but the draft, for the n sessions (N = n).
    pub fn conf_sum(&self, n: usize, mk: &s2f::Markov, cb: &s2f::ConfBuf, ksteps: usize) -> Result<()> {
        let c = cfg([CONF_NCH, ksteps, 1], 256, 0);
        let (m, b_) = (*mk, *cb);
        // SAFETY: block (chunk, step) reads the step's projections and drafts (ConfBuf, 7 x n), the chunk's rows of
        // lrU, fold_b and Bq (int8 head; head rows of the n sessions) and writes part[step][chunk][0..n]; N = n.
        unsafe {
            match n {
                1 => self
                    .k
                    .s2f_conf_sum_kernel_1(self.st, c, S2fConfSumKernel1Args { mk: m, cb: b_ }),
                2 => self
                    .k
                    .s2f_conf_sum_kernel_2(self.st, c, S2fConfSumKernel2Args { mk: m, cb: b_ }),
                3 => self
                    .k
                    .s2f_conf_sum_kernel_3(self.st, c, S2fConfSumKernel3Args { mk: m, cb: b_ }),
                4 => self
                    .k
                    .s2f_conf_sum_kernel_4(self.st, c, S2fConfSumKernel4Args { mk: m, cb: b_ }),
                5 => self
                    .k
                    .s2f_conf_sum_kernel_5(self.st, c, S2fConfSumKernel5Args { mk: m, cb: b_ }),
                6 => self
                    .k
                    .s2f_conf_sum_kernel_6(self.st, c, S2fConfSumKernel6Args { mk: m, cb: b_ }),
                7 => self
                    .k
                    .s2f_conf_sum_kernel_7(self.st, c, S2fConfSumKernel7Args { mk: m, cb: b_ }),
                _ => self
                    .k
                    .s2f_conf_sum_kernel_8(self.st, c, S2fConfSumKernel8Args { mk: m, cb: b_ }),
            }
        }
    }

    /// `conf_finish_kernel<<<1, 32>>>(cb, n, ksteps, conf)` (draft_conf.cuh): the cumulative log probability.
    pub fn conf_finish(&self, cb: &s2f::ConfBuf, n: usize, ksteps: usize, conf: &DevBuf<f32>) -> Result<()> {
        need(conf, n * 7, "conf")?;
        // SAFETY: thread r < n sums part[0..7][..][r] (ConfBuf of n sessions) and writes conf[r][0..7] (checked).
        unsafe {
            self.k.s2f_conf_finish_kernel(
                self.st,
                cfg([1, 1, 1], 32, 0),
                S2fConfFinishKernelArgs {
                    cb: *cb,
                    n: i(n),
                    ksteps: i(ksteps),
                    conf: b(conf),
                },
            )
        }
    }

    /// `markov_finish<<<1, 64>>>(mk)`: drafts from the step keys.
    pub fn markov_finish(&self, mk: &s2f::Markov) -> Result<()> {
        // SAFETY: one block writes drafts[n][7] from keys[7][n] (module comment: buffers of n sessions).
        unsafe { self.k.s2f_markov_finish(self.st, cfg([1, 1, 1], 64, 0), mk) }
    }

    /// `markov_recur_proj<<<32, 256>>>(mk, i, lb)` (`draft_leviathan.cuh`): fused recurrence + rank-16 proj + init.
    pub fn markov_recur_proj(&self, mk: &s2f::Markov, step: usize, lb: &s2f::LevBuf) -> Result<()> {
        // SAFETY: step < 7; reads context/state/lrV, writes state/latent, atomically accumulates mk.proj, resets cand_cnt/g_lb.
        unsafe {
            self.k.s2f_markov_recur_proj(
                self.st,
                cfg([32, 1, 1], 256, 0),
                S2fMarkovRecurProjArgs {
                    mk: *mk,
                    i: i(step),
                    lb: *lb,
                },
            )
        }
    }

    /// `markov_lev_scan_fused<N><<<LEV_NCH, 256>>>(mk, i, lb)` (`draft_leviathan.cuh`): single-pass multi-session scan.
    pub fn markov_lev_scan_fused(&self, k: usize, mk: &s2f::Markov, step: usize, lb: &s2f::LevBuf) -> Result<()> {
        const LEV_ROWS: usize = 512;
        const LEV_NCH: usize = VU.div_ceil(LEV_ROWS);
        let c = cfg([LEV_NCH, 1, 1], 256, 0);
        // SAFETY: block `chunk` scans 512 rows of lrU/fold_b ONCE across all k sessions and writes exact Z + pruned top-16.
        unsafe {
            match k {
                1 => self.k.s2f_markov_lev_scan_fused_1(
                    self.st,
                    c,
                    S2fMarkovLevScanFused1Args {
                        mk: *mk,
                        i: i(step),
                        lb: *lb,
                    },
                ),
                2 => self.k.s2f_markov_lev_scan_fused_2(
                    self.st,
                    c,
                    S2fMarkovLevScanFused2Args {
                        mk: *mk,
                        i: i(step),
                        lb: *lb,
                    },
                ),
                3 => self.k.s2f_markov_lev_scan_fused_3(
                    self.st,
                    c,
                    S2fMarkovLevScanFused3Args {
                        mk: *mk,
                        i: i(step),
                        lb: *lb,
                    },
                ),
                4 => self.k.s2f_markov_lev_scan_fused_4(
                    self.st,
                    c,
                    S2fMarkovLevScanFused4Args {
                        mk: *mk,
                        i: i(step),
                        lb: *lb,
                    },
                ),
                5 => self.k.s2f_markov_lev_scan_fused_5(
                    self.st,
                    c,
                    S2fMarkovLevScanFused5Args {
                        mk: *mk,
                        i: i(step),
                        lb: *lb,
                    },
                ),
                6 => self.k.s2f_markov_lev_scan_fused_6(
                    self.st,
                    c,
                    S2fMarkovLevScanFused6Args {
                        mk: *mk,
                        i: i(step),
                        lb: *lb,
                    },
                ),
                7 => self.k.s2f_markov_lev_scan_fused_7(
                    self.st,
                    c,
                    S2fMarkovLevScanFused7Args {
                        mk: *mk,
                        i: i(step),
                        lb: *lb,
                    },
                ),
                _ => self.k.s2f_markov_lev_scan_fused_8(
                    self.st,
                    c,
                    S2fMarkovLevScanFused8Args {
                        mk: *mk,
                        i: i(step),
                        lb: *lb,
                    },
                ),
            }
        }
    }

    /// `markov_lev_finish<<<k, 512>>>(mk, i, lb)` (`draft_leviathan.cuh`): 16-warp parallel exact top-16 + draw.
    pub fn markov_lev_finish(&self, k: usize, mk: &s2f::Markov, step: usize, lb: &s2f::LevBuf) -> Result<()> {
        // SAFETY: block r < k merges compacted candidates, evaluates 16 exact_z_warp<true> in parallel, samples c* ~ q,
        // and writes mk.keys, mk.drafts, prop_ids[slot][i], prop_w[slot][i], conf_c[r][i], resetting mk.proj for step i+1.
        unsafe {
            self.k.s2f_markov_lev_finish(
                self.st,
                cfg([k, 1, 1], 512, 0),
                S2fMarkovLevFinishArgs {
                    mk: *mk,
                    i: i(step),
                    lb: *lb,
                },
            )
        }
    }

    // ------------------------------------------------------------------------------------------------ round batch

    /// `draft_gather<<<k, 256>>>(sel, anchor, P, bt, max_blocks, f_rnn, anchor_c, P_c, bt_c, f_rnn_c)`
    /// (`draft_sel.cuh`): the state of the k drafted sessions `sel[0..k)` into compact rows `0..k`.
    pub fn draft_gather(&self, k: usize, n: usize, max_blocks: usize, b_: &Batch<'_>) -> Result<()> {
        need(b_.sel, k, "sel")?;
        for (x, what) in [
            (b_.anchor, "anchor"),
            (b_.p, "P"),
            (b_.anchor_c, "anchor_c"),
            (b_.p_c, "P_c"),
        ] {
            need(x, n, what)?;
        }
        need(b_.bt, n * max_blocks, "block_table")?;
        need(b_.bt_c, n * max_blocks, "bt_c")?;
        need(b_.f_rnn, n * H, "f_rnn")?;
        need(b_.f_rnn_c, n * H, "f_rnn_c")?;
        // SAFETY: block i < k <= n reads session r = sel[i] (distinct, below n: draft_sessions checked the list it
        // uploaded) of anchor, P, the page table row and f_rnn (n rows checked), writes compact row i (n rows checked).
        unsafe {
            self.k.s2f_draft_gather(
                self.st,
                cfg([k, 1, 1], 256, 0),
                S2fDraftGatherArgs {
                    sel: b(b_.sel),
                    anchor: b(b_.anchor),
                    P: b(b_.p),
                    bt: b(b_.bt),
                    max_blocks: i(max_blocks),
                    f_rnn: b(b_.f_rnn),
                    anchor_c: b(b_.anchor_c),
                    P_c: b(b_.p_c),
                    bt_c: b(b_.bt_c),
                    f_rnn_c: b(b_.f_rnn_c),
                },
            )
        }
    }

    /// `draft_scatter<<<1, 64>>>(sel, k, n, drafts_c, nsurv_c, conf_c, drafts, nsurv, conf)` (`draft_sel.cuh`):
    /// the k compact rows' drafts, survivors and drafter probabilities back to their sessions' rows.
    pub fn draft_scatter(&self, k: usize, n: usize, b_: &Batch<'_>) -> Result<()> {
        need(b_.sel, k, "sel")?;
        for (x, what) in [
            (b_.drafts_c, "drafts_c"),
            (b_.nsurv_c, "nsurv_c"),
            (b_.drafts, "drafts"),
            (b_.nsurv, "nsurv"),
        ] {
            need(x, 7 * n, what)?;
        }
        need(b_.conf_c, 7 * n, "conf_c")?;
        need(b_.conf, 7 * n, "conf")?;
        // SAFETY: thread e < 7 k <= 56 (block 64) reads compact element e (7 n checked) and writes the element of
        // session sel[e / 7] (distinct, below n) of drafts and conf [n][7], nsurv [7][n] (7 n checked).
        unsafe {
            self.k.s2f_draft_scatter(
                self.st,
                cfg([1, 1, 1], 64, 0),
                S2fDraftScatterArgs {
                    sel: b(b_.sel),
                    k: i(k),
                    n: i(n),
                    drafts_c: b(b_.drafts_c),
                    nsurv_c: b(b_.nsurv_c),
                    conf_c: b(b_.conf_c),
                    drafts: b(b_.drafts),
                    nsurv: b(b_.nsurv),
                    conf: b(b_.conf),
                },
            )
        }
    }
}

/// The buffers of the round batch's gather and scatter (`draft_sel.cuh`): the session list, the sessions' state and
/// outputs (`[n]` rows) and their compact copies.
pub(super) struct Batch<'a> {
    pub sel: &'a DevBuf<i32>,
    pub anchor: &'a DevBuf<i32>,
    pub p: &'a DevBuf<i32>,
    pub bt: &'a DevBuf<i32>,
    pub f_rnn: &'a DevBuf<Bf16>,
    pub anchor_c: &'a DevBuf<i32>,
    pub p_c: &'a DevBuf<i32>,
    pub bt_c: &'a DevBuf<i32>,
    pub f_rnn_c: &'a DevBuf<Bf16>,
    pub drafts_c: &'a DevBuf<i32>,
    pub nsurv_c: &'a DevBuf<i32>,
    pub conf_c: &'a DevBuf<f32>,
    pub drafts: &'a DevBuf<i32>,
    pub nsurv: &'a DevBuf<i32>,
    pub conf: &'a DevBuf<f32>,
}
