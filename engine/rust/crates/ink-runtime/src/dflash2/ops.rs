//! Safe Rust wrappers for every DFlash2 CUDA kernel in `engine/kernels/dflash2/dflash2_kernels.cuh`
//! plus the shared `s2f` W8A16 vocabulary head GEMM kernels from `pack_drafter.cubin`.
#![allow(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use ink_cuda::{Bf16, CudaError, DevBuf, LaunchCfg, Result, Stream};
use ink_kernels::{
    Df2EmbedBlockKernelArgs, Df2GroupedConvInKernelArgs, Df2GroupedConvOutResNormKernelArgs,
    Df2KvNormRopeStoreKernelArgs, Df2QkvNormRopeBlkKernelArgs, Df2RmsNormKernelArgs, Df2RopeTableKernelArgs,
    Df2SelectorWalkKernelArgs, Df2SiluMulKernelArgs, Df2SwaAttnReduceKernelArgs, Df2SwaAttnSplitKernelArgs,
    Df2Topk16ChunkMaxKernelArgs, Df2Topk16FilterKernelArgs, Df2Topk16Stage2KernelArgs, F16, KernelId, Kernels, Ptr,
    S2fPermuteFragI8Args, S2fQuantRowsI8Args, s2f,
};
use ink_model::dflash2::{
    ATTN_SMEM, BLK, CONV_PROJ_N, DRAFTS, H, HD, INTER, KVH, NH, RING, SEL_RANK, TOP_K, TOPK_CHUNKS,
};
use ink_model::drafter::XS;

fn need<T: Pod>(x: &DevBuf<T>, n: usize, what: &'static str) -> Result<()> {
    if x.len() >= n {
        Ok(())
    } else {
        Err(CudaError::Bounds {
            what,
            detail: format!("{} elements, need {n}", x.len()),
        })
    }
}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("launch dim fits u32")
}

fn i(x: usize) -> i32 {
    i32::try_from(x).expect("kernel int fits i32")
}

fn cfg(grid: [usize; 3], block: usize) -> LaunchCfg {
    cfg_smem(grid, block, 0)
}

fn cfg_smem(grid: [usize; 3], block: usize, smem: usize) -> LaunchCfg {
    LaunchCfg {
        grid: [u(grid[0]), u(grid[1]), u(grid[2])],
        block: [u(block), 1, 1],
        smem: u(smem),
    }
}

/// Dynamic shared memory of `s2f::gemm_frag2<MT, true, 2, 8>`.
pub(super) fn head_gemm_smem(mt: usize) -> usize {
    2 * mt * 8 * XS * 2
}

/// All `s2f::gemm_frag2<MT, true, 2, 8>` kernel ids (for dynamic smem setup at startup).
pub(super) fn head_gemm_ids() -> [(KernelId, usize); 8] {
    use KernelId as K;
    [
        (K::S2fGemmFrag21True28, 1),
        (K::S2fGemmFrag22True28, 2),
        (K::S2fGemmFrag23True28, 3),
        (K::S2fGemmFrag24True28, 4),
        (K::S2fGemmFrag25True28, 5),
        (K::S2fGemmFrag26True28, 6),
        (K::S2fGemmFrag27True28, 7),
        (K::S2fGemmFrag28True28, 8),
    ]
}

pub(super) struct Dev<'a> {
    pub k: &'a Kernels,
    pub st: &'a Stream,
}

impl Dev<'_> {
    pub(super) fn quant_rows_i8(
        &self,
        w: &DevBuf<Bf16>,
        q: &DevBuf<i8>,
        scale: &DevBuf<f32>,
        enorm: &DevBuf<f32>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        need(w, n * k, "df2 head quant w")?;
        need(q, n * k, "df2 head quant q")?;
        need(scale, n, "df2 head quant scale")?;
        need(enorm, n, "df2 head quant enorm")?;
        // SAFETY: block r < n reads row r of w and writes row r of q, scale[r], enorm[r].
        unsafe {
            self.k.s2f_quant_rows_i8(
                self.st,
                cfg([n, 1, 1], 256),
                S2fQuantRowsI8Args {
                    w: Ptr::Buf(w),
                    q: Ptr::Buf(q),
                    scale: Ptr::Buf(scale),
                    enorm: Ptr::Buf(enorm),
                    K: i(k),
                },
            )
        }
    }

    pub(super) fn permute_frag_i8(&self, w: &DevBuf<i8>, out: &DevBuf<i8>, n: usize, k: usize) -> Result<()> {
        need(w, n * k, "df2 head permute w")?;
        need(out, n * k, "df2 head permute out")?;
        let th = n * k / 8;
        // SAFETY: n and k are multiples of 16; both buffers hold n * k i8.
        unsafe {
            self.k.s2f_permute_frag_i8(
                self.st,
                cfg([th.div_ceil(256), 1, 1], 256),
                S2fPermuteFragI8Args {
                    w: Ptr::Buf(w),
                    out: Ptr::Buf(out),
                    N: i(n),
                    K: i(k),
                },
            )
        }
    }

    /// W8A16 `s2f::gemm_frag2<MT, true, 2, 8>`: `logits[m, n] = pred_h_f16[m, H] @ head_q8f[n, H]^T * scale[n]`.
    pub(super) fn head_gemm_i8(
        &self,
        w_q8f: &DevBuf<i8>,
        scale: &DevBuf<f32>,
        x_f16: &DevBuf<F16>,
        logits: &DevBuf<f32>,
        m: usize,
        n: usize,
    ) -> Result<()> {
        let mt = m.div_ceil(8);
        if mt == 0 || mt > 8 {
            return Err(CudaError::Bounds {
                what: "df2 head_gemm_i8",
                detail: format!("M={m} (MT={mt}) outside 1..=64"),
            });
        }
        need(w_q8f, n * H, "df2 head_q8f")?;
        need(scale, n, "df2 head_scale")?;
        need(x_f16, m * H, "df2 pred_h_f16")?;
        need(logits, m * n, "df2 head logits")?;
        let x_raw = x_f16.cast::<Bf16>()?;
        let a = s2f::GemmArgs {
            W: w_q8f.cast::<u8>()?.ptr(),
            x: x_raw.ptr(),
            ldx: i(H),
            M: i(m),
            N: i(n),
            K: i(H),
            kslice: i(H),
            ldo: i(n),
            part: logits.ptr(),
            out: Zeroable::zeroed(),
            scale: scale.ptr(),
        };
        // Each CTA of gemm_frag2<MT, true, 2, 8> covers 8 warps * 2 tiles * 16 rows = 256 N rows.
        let c = cfg_smem([(n / 16).div_ceil(16), 1, 1], 256, head_gemm_smem(mt));
        // SAFETY: extents checked above; max dynamic smem configured at startup; n and H are multiples of 16.
        unsafe {
            match mt {
                1 => self.k.s2f_gemm_frag2_1_true_2_8(self.st, c, &a),
                2 => self.k.s2f_gemm_frag2_2_true_2_8(self.st, c, &a),
                3 => self.k.s2f_gemm_frag2_3_true_2_8(self.st, c, &a),
                4 => self.k.s2f_gemm_frag2_4_true_2_8(self.st, c, &a),
                5 => self.k.s2f_gemm_frag2_5_true_2_8(self.st, c, &a),
                6 => self.k.s2f_gemm_frag2_6_true_2_8(self.st, c, &a),
                7 => self.k.s2f_gemm_frag2_7_true_2_8(self.st, c, &a),
                8 => self.k.s2f_gemm_frag2_8_true_2_8(self.st, c, &a),
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn rope_table(&self, cos_sin: &DevBuf<Bf16>, max_pos: usize, theta: f32) -> Result<()> {
        need(cos_sin, max_pos * HD, "df2 cos_sin")?;
        let blocks = (max_pos * (HD / 2)).div_ceil(256);
        // SAFETY: cos_sin holds max_pos * HD bf16; each thread writes 2 elements in bounds.
        unsafe {
            self.k.df2_rope_table_kernel(
                self.st,
                cfg([blocks, 1, 1], 256),
                Df2RopeTableKernelArgs {
                    cos_sin: Ptr::Buf(cos_sin),
                    max_pos: i(max_pos),
                    theta,
                },
            )
        }
    }

    #[expect(clippy::too_many_arguments, reason = "kernel arguments")]
    pub(super) fn embed_block(
        &self,
        embed: &DevBuf<Bf16>,
        embed_norm: &DevBuf<Bf16>,
        mask_emb: &DevBuf<Bf16>,
        anchor: &DevBuf<i32>,
        sel: &DevBuf<i32>,
        out: &DevBuf<Bf16>,
        k: usize,
    ) -> Result<()> {
        need(embed_norm, H, "df2 embed_norm")?;
        need(mask_emb, H, "df2 mask_emb")?;
        need(sel, k, "df2 sel")?;
        need(out, k * BLK * H, "df2 embed out")?;
        // SAFETY: caller guarantees anchor[sel[0..k]] < VOCAB and sel[0..k] < n.
        unsafe {
            self.k.df2_embed_block_kernel(
                self.st,
                cfg([k * BLK, 1, 1], 256),
                Df2EmbedBlockKernelArgs {
                    embed: Ptr::Buf(embed),
                    embed_norm: Ptr::Buf(embed_norm),
                    mask_emb: Ptr::Buf(mask_emb),
                    anchor: Ptr::Buf(anchor),
                    sel: Ptr::Buf(sel),
                    out: Ptr::Buf(out),
                    k: i(k),
                },
            )
        }
    }

    pub(super) fn rms_norm(
        &self,
        x: &DevBuf<Bf16>,
        w: &DevBuf<Bf16>,
        out: &DevBuf<Bf16>,
        rows: usize,
        skip_anchor: bool,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let src_rows = if skip_anchor { (rows / DRAFTS) * BLK } else { rows };
        need(x, src_rows * H, "df2 rms_norm x")?;
        need(w, H, "df2 rms_norm w")?;
        need(out, rows * H, "df2 rms_norm out")?;
        // SAFETY: x, w, out hold the checked rows * H elements.
        unsafe {
            self.k.df2_rms_norm_kernel(
                self.st,
                cfg([rows, 1, 1], 256),
                Df2RmsNormKernelArgs {
                    x: Ptr::Buf(x),
                    w: Ptr::Buf(w),
                    out: Ptr::Buf(out),
                    rows: i(rows),
                    skip_anchor: i32::from(skip_anchor),
                },
            )
        }
    }

    pub(super) fn grouped_conv_in(
        &self,
        h: &DevBuf<Bf16>,
        dyn_buf: &DevBuf<Bf16>,
        base: &DevBuf<Bf16>,
        out: &DevBuf<Bf16>,
        k: usize,
    ) -> Result<()> {
        let rows = k * BLK;
        need(h, rows * H, "df2 conv_in h")?;
        need(dyn_buf, rows * CONV_PROJ_N, "df2 conv_in dyn")?;
        need(base, 4 * H, "df2 conv_in base")?;
        need(out, rows * H, "df2 conv_in out")?;
        // SAFETY: all buffers hold rows * stride elements; tap 1 reads row-1 only when j > 0.
        unsafe {
            self.k.df2_grouped_conv_in_kernel(
                self.st,
                cfg([rows, 1, 1], 256),
                Df2GroupedConvInKernelArgs {
                    h: Ptr::Buf(h),
                    r#dyn: Ptr::Buf(dyn_buf),
                    base: Ptr::Buf(base),
                    out: Ptr::Buf(out),
                    k: i(k),
                },
            )
        }
    }

    #[expect(clippy::too_many_arguments, reason = "fused conv finish + residual + rmsnorm")]
    pub(super) fn grouped_conv_out_res_norm(
        &self,
        sub: &DevBuf<Bf16>,
        dyn_buf: &DevBuf<Bf16>,
        base: &DevBuf<Bf16>,
        residual: &DevBuf<Bf16>,
        norm_w: Option<&DevBuf<Bf16>>,
        norm_out: Option<&DevBuf<Bf16>>,
        f16_out: Option<&DevBuf<F16>>,
        k: usize,
        skip_anchor_norm: bool,
    ) -> Result<()> {
        let rows = k * BLK;
        need(sub, rows * H, "df2 conv_out sub")?;
        need(dyn_buf, rows * CONV_PROJ_N, "df2 conv_out dyn")?;
        need(base, 4 * H, "df2 conv_out base")?;
        need(residual, rows * H, "df2 conv_out residual")?;
        let out_rows = if skip_anchor_norm { k * DRAFTS } else { rows };
        if let Some(w) = norm_w {
            need(w, H, "df2 conv_out norm_w")?;
            let out = norm_out.ok_or_else(|| CudaError::Bounds {
                what: "df2 conv_out norm_out",
                detail: "missing norm_out for non-null norm_w".into(),
            })?;
            need(out, out_rows * H, "df2 conv_out norm_out")?;
        }
        if let Some(f16) = f16_out {
            need(f16, out_rows * H, "df2 conv_out f16_out")?;
        }
        // SAFETY: all non-null buffers checked above.
        unsafe {
            self.k.df2_grouped_conv_out_res_norm_kernel(
                self.st,
                cfg([rows, 1, 1], 256),
                Df2GroupedConvOutResNormKernelArgs {
                    sub: Ptr::Buf(sub),
                    r#dyn: Ptr::Buf(dyn_buf),
                    base: Ptr::Buf(base),
                    residual: Ptr::Buf(residual),
                    norm_w: norm_w.map_or(Ptr::Null, Ptr::Buf),
                    norm_out: norm_out.map_or(Ptr::Null, Ptr::Buf),
                    f16_out: f16_out.map_or(Ptr::Null, Ptr::Buf),
                    k: i(k),
                    skip_anchor_norm: i32::from(skip_anchor_norm),
                },
            )
        }
    }

    #[expect(clippy::too_many_arguments, reason = "kernel arguments")]
    pub(super) fn kv_norm_rope_store(
        &self,
        kv: &DevBuf<Bf16>,
        k_norm: &DevBuf<Bf16>,
        cos_sin: &DevBuf<Bf16>,
        pos: &DevBuf<i32>,
        sess: &DevBuf<i32>,
        k_ring: &DevBuf<Bf16>,
        v_ring: &DevBuf<Bf16>,
        n: usize,
        sessions: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        need(kv, n * 2 * KVH * HD, "df2 commit kv")?;
        need(k_norm, HD, "df2 k_norm")?;
        need(pos, n, "df2 commit pos")?;
        need(sess, n, "df2 commit sess")?;
        need(k_ring, sessions * KVH * RING * HD, "df2 k_ring")?;
        need(v_ring, sessions * KVH * RING * HD, "df2 v_ring")?;
        // SAFETY: caller guarantees pos[0..n] < cos_sin_pos and sess[0..n] < sessions.
        unsafe {
            self.k.df2_kv_norm_rope_store_kernel(
                self.st,
                cfg([n, KVH, 1], HD),
                Df2KvNormRopeStoreKernelArgs {
                    kv: Ptr::Buf(kv),
                    k_norm: Ptr::Buf(k_norm),
                    cos_sin: Ptr::Buf(cos_sin),
                    pos: Ptr::Buf(pos),
                    sess: Ptr::Buf(sess),
                    k_ring: Ptr::Buf(k_ring),
                    v_ring: Ptr::Buf(v_ring),
                    n: i(n),
                },
            )
        }
    }

    #[expect(clippy::too_many_arguments, reason = "kernel arguments")]
    pub(super) fn qkv_norm_rope_blk(
        &self,
        qkv: &DevBuf<Bf16>,
        q_norm: &DevBuf<Bf16>,
        k_norm: &DevBuf<Bf16>,
        cos_sin: &DevBuf<Bf16>,
        pos: &DevBuf<i32>,
        sel: &DevBuf<i32>,
        q_blk: &DevBuf<Bf16>,
        k_blk: &DevBuf<Bf16>,
        v_blk: &DevBuf<Bf16>,
        k: usize,
    ) -> Result<()> {
        let rows = k * BLK;
        need(qkv, rows * (NH + 2 * KVH) * HD, "df2 qkv")?;
        need(q_norm, HD, "df2 q_norm")?;
        need(k_norm, HD, "df2 k_norm")?;
        need(sel, k, "df2 sel")?;
        need(q_blk, rows * NH * HD, "df2 q_blk")?;
        need(k_blk, rows * KVH * HD, "df2 k_blk")?;
        need(v_blk, rows * KVH * HD, "df2 v_blk")?;
        // SAFETY: caller guarantees pos[sel[0..k]] + 7 < cos_sin_pos.
        unsafe {
            self.k.df2_qkv_norm_rope_blk_kernel(
                self.st,
                cfg([rows, NH + KVH, 1], HD),
                Df2QkvNormRopeBlkKernelArgs {
                    qkv: Ptr::Buf(qkv),
                    q_norm: Ptr::Buf(q_norm),
                    k_norm: Ptr::Buf(k_norm),
                    cos_sin: Ptr::Buf(cos_sin),
                    pos: Ptr::Buf(pos),
                    sel: Ptr::Buf(sel),
                    q_blk: Ptr::Buf(q_blk),
                    k_blk: Ptr::Buf(k_blk),
                    v_blk: Ptr::Buf(v_blk),
                    k: i(k),
                },
            )
        }
    }

    #[expect(clippy::too_many_arguments, reason = "attention split + reduce")]
    pub(super) fn swa_attn(
        &self,
        q_blk: &DevBuf<Bf16>,
        k_blk: &DevBuf<Bf16>,
        v_blk: &DevBuf<Bf16>,
        k_ring: &DevBuf<Bf16>,
        v_ring: &DevBuf<Bf16>,
        pos: &DevBuf<i32>,
        sel: &DevBuf<i32>,
        part_ml: &DevBuf<f32>,
        part_o: &DevBuf<f32>,
        out: &DevBuf<Bf16>,
        k: usize,
        sessions: usize,
        splits: usize,
    ) -> Result<()> {
        let rows = k * BLK;
        need(q_blk, rows * NH * HD, "df2 attn q_blk")?;
        need(k_blk, rows * KVH * HD, "df2 attn k_blk")?;
        need(v_blk, rows * KVH * HD, "df2 attn v_blk")?;
        need(k_ring, sessions * KVH * RING * HD, "df2 attn k_ring")?;
        need(v_ring, sessions * KVH * RING * HD, "df2 attn v_ring")?;
        need(sel, k, "df2 attn sel")?;
        need(part_ml, k * KVH * splits * 32 * 2, "df2 attn part_ml")?;
        need(part_o, k * KVH * splits * 32 * HD, "df2 attn part_o")?;
        need(out, rows * NH * HD, "df2 attn out")?;
        // SAFETY: all extents checked above; ATTN_SMEM configured at startup; sel[0..k] < sessions.
        unsafe {
            self.k.df2_swa_attn_split_kernel(
                self.st,
                cfg_smem([splits, KVH, k], 128, ATTN_SMEM),
                Df2SwaAttnSplitKernelArgs {
                    q_blk: Ptr::Buf(q_blk),
                    k_blk: Ptr::Buf(k_blk),
                    v_blk: Ptr::Buf(v_blk),
                    k_ring: Ptr::Buf(k_ring),
                    v_ring: Ptr::Buf(v_ring),
                    pos: Ptr::Buf(pos),
                    sel: Ptr::Buf(sel),
                    part_ml: Ptr::Buf(part_ml),
                    part_o: Ptr::Buf(part_o),
                    k: i(k),
                    splits: i(splits),
                },
            )?;
            self.k.df2_swa_attn_reduce_kernel(
                self.st,
                cfg([k * KVH * 32, 1, 1], HD),
                Df2SwaAttnReduceKernelArgs {
                    part_ml: Ptr::Buf(part_ml),
                    part_o: Ptr::Buf(part_o),
                    out: Ptr::Buf(out),
                    k: i(k),
                    splits: i(splits),
                },
            )
        }
    }

    pub(super) fn silu_mul(&self, gate_up: &DevBuf<Bf16>, out: &DevBuf<Bf16>, rows: usize) -> Result<()> {
        need(gate_up, rows * 2 * INTER, "df2 gate_up")?;
        need(out, rows * INTER, "df2 silu_mul out")?;
        // SAFETY: INTER = 6912 = 27 * 256.
        unsafe {
            self.k.df2_silu_mul_kernel(
                self.st,
                cfg([rows, INTER / 256, 1], 256),
                Df2SiluMulKernelArgs {
                    gate_up: Ptr::Buf(gate_up),
                    out: Ptr::Buf(out),
                    rows: i(rows),
                },
            )
        }
    }

    #[expect(clippy::too_many_arguments, reason = "two-pass threshold topk16")]
    pub(super) fn topk16(
        &self,
        logits: &DevBuf<f32>,
        logit_stride: usize,
        chunk_max: &DevBuf<f32>,
        chunk_z: &DevBuf<f32>,
        cand_count: &DevBuf<i32>,
        samp_temp: &DevBuf<f32>,
        sel: &DevBuf<i32>,
        s1_vals: &DevBuf<f32>,
        s1_ids: &DevBuf<i32>,
        unary: &DevBuf<f32>,
        cand_ids: &DevBuf<i32>,
        row_z: &DevBuf<f32>,
        rows: usize,
    ) -> Result<()> {
        need(logits, rows * logit_stride, "df2 topk logits")?;
        need(chunk_max, rows * TOPK_CHUNKS, "df2 topk chunk_max")?;
        need(chunk_z, rows * TOPK_CHUNKS, "df2 topk chunk_z")?;
        need(cand_count, rows, "df2 topk cand_count")?;
        need(s1_vals, rows * TOPK_CHUNKS * TOP_K, "df2 topk s1_vals")?;
        need(s1_ids, rows * TOPK_CHUNKS * TOP_K, "df2 topk s1_ids")?;
        need(unary, rows * TOP_K, "df2 topk unary")?;
        need(cand_ids, rows * TOP_K, "df2 topk cand_ids")?;
        need(row_z, rows, "df2 topk row_z")?;
        // SAFETY: all extents checked above; VALID_VOCAB <= logit_stride and logit_stride % 4 == 0.
        unsafe {
            self.k.df2_topk16_chunk_max_kernel(
                self.st,
                cfg([rows, TOPK_CHUNKS, 1], 256),
                Df2Topk16ChunkMaxKernelArgs {
                    logits: Ptr::Buf(logits),
                    chunk_max: Ptr::Buf(chunk_max),
                    chunk_z: Ptr::Buf(chunk_z),
                    cand_count: Ptr::Buf(cand_count),
                    samp_temp: Ptr::Buf(samp_temp),
                    sel: Ptr::Buf(sel),
                    rows: i(rows),
                    logit_stride: i(logit_stride),
                },
            )?;
            self.k.df2_topk16_filter_kernel(
                self.st,
                cfg([rows, TOPK_CHUNKS, 1], 256),
                Df2Topk16FilterKernelArgs {
                    logits: Ptr::Buf(logits),
                    chunk_max: Ptr::Buf(chunk_max),
                    s1_vals: Ptr::Buf(s1_vals),
                    s1_ids: Ptr::Buf(s1_ids),
                    cand_count: Ptr::Buf(cand_count),
                    rows: i(rows),
                    logit_stride: i(logit_stride),
                },
            )?;
            self.k.df2_topk16_stage2_kernel(
                self.st,
                cfg([rows, 1, 1], 256),
                Df2Topk16Stage2KernelArgs {
                    s1_vals: Ptr::Buf(s1_vals),
                    s1_ids: Ptr::Buf(s1_ids),
                    cand_count: Ptr::Buf(cand_count),
                    chunk_z: Ptr::Buf(chunk_z),
                    unary_out: Ptr::Buf(unary),
                    cand_ids_out: Ptr::Buf(cand_ids),
                    row_z_out: Ptr::Buf(row_z),
                    rows: i(rows),
                },
            )
        }
    }

    #[expect(clippy::too_many_arguments, reason = "selector walk arguments")]
    pub(super) fn selector_walk(
        &self,
        cand_ids: &DevBuf<i32>,
        unary: &DevBuf<f32>,
        row_z: &DevBuf<f32>,
        hp: &DevBuf<f32>,
        pred_cb: &DevBuf<Bf16>,
        succ_cb: &DevBuf<Bf16>,
        anchor: &DevBuf<i32>,
        pos: &DevBuf<i32>,
        sel: &DevBuf<i32>,
        samp_temp: &DevBuf<f32>,
        samp_top_p: &DevBuf<f32>,
        samp_seed: &DevBuf<u64>,
        drafts: &DevBuf<i32>,
        prop_ids: &DevBuf<i32>,
        prop_w: &DevBuf<u32>,
        conf: &DevBuf<f32>,
        conf_mode: i32,
        k: usize,
        sessions: usize,
    ) -> Result<()> {
        need(cand_ids, k * DRAFTS * TOP_K, "df2 sel cand_ids")?;
        need(unary, k * DRAFTS * TOP_K, "df2 sel unary")?;
        need(row_z, k * DRAFTS, "df2 sel row_z")?;
        need(hp, k * DRAFTS * SEL_RANK, "df2 sel hp")?;
        need(pos, sessions, "df2 sel pos")?;
        need(sel, k, "df2 sel")?;
        need(samp_temp, sessions, "df2 samp_temp")?;
        need(samp_top_p, sessions, "df2 samp_top_p")?;
        need(samp_seed, sessions, "df2 samp_seed")?;
        need(drafts, sessions * DRAFTS, "df2 drafts")?;
        need(prop_ids, sessions * DRAFTS * TOP_K, "df2 prop_ids")?;
        need(prop_w, sessions * DRAFTS * TOP_K, "df2 prop_w")?;
        need(conf, sessions * DRAFTS, "df2 conf")?;
        // SAFETY: cand_ids < VALID_VOCAB < VOCAB_PAD, anchor[sel[0..k]] < VALID_VOCAB, sel[0..k] < sessions.
        unsafe {
            self.k.df2_selector_walk_kernel(
                self.st,
                cfg([k, 1, 1], 512),
                Df2SelectorWalkKernelArgs {
                    cand_ids: Ptr::Buf(cand_ids),
                    unary: Ptr::Buf(unary),
                    row_z: Ptr::Buf(row_z),
                    hp: Ptr::Buf(hp),
                    pred_cb: Ptr::Buf(pred_cb),
                    succ_cb: Ptr::Buf(succ_cb),
                    anchor: Ptr::Buf(anchor),
                    pos: Ptr::Buf(pos),
                    sel: Ptr::Buf(sel),
                    samp_temp: Ptr::Buf(samp_temp),
                    samp_top_p: Ptr::Buf(samp_top_p),
                    samp_seed: Ptr::Buf(samp_seed),
                    drafts: Ptr::Buf(drafts),
                    prop_ids: Ptr::Buf(prop_ids),
                    prop_w: Ptr::Buf(prop_w),
                    conf: Ptr::Buf(conf),
                    conf_mode,
                    k: i(k),
                },
            )
        }
    }
}
