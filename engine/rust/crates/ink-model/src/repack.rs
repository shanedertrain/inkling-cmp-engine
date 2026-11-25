//! Routed-expert repack into the prefill_moe kernel's fragment layout (port of `weights.cpp` @ 66fe353, whose loops
//! are copied from `engine/kernels/prefill_moe/bench.cu`: `enc_scale`, `orig_row`, the NVFP4 "n1" branch and the
//! MXFP8 branch).
//!
//! Layout, per expert `e`, weight tile `wt` (64 output rows) and K step `ks` (64 inputs), blocks in `[e][wt][ks]`
//! order:
//! - NVFP4 n1: weights 2,048 bytes (4 m-tiles x 32 lanes x 4 k-chunks x one u32 of eight 4-bit codes, bit positions
//!   `POS`), scales 512 bytes (4 x 8 groups x 4 k-chunks x one u32 = the bf16 bits of the two rows' scales);
//! - MXFP8: weights 4,096 bytes (4 x 32 lanes x 4 k-chunks x 8 bytes), scales 128 bytes (e8m0, copied).
//!
//! Experts are independent; the work is split over threads by expert (the C++ `par`), which cannot change a byte.

use std::thread;

use crate::error::{Result, invalid};

/// K step of one block.
pub const BK: usize = 64;
/// Output rows of one weight tile.
pub const WT: usize = 64;

/// The two expert projections, in checkpoint order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proj {
    /// Gate and up, interleaved (`w13`, N = 2 x 1152, K = 2304).
    W13,
    /// Down (`w2`, N = 2304, K = 1152).
    W2,
}

/// Shape of one projection of `experts` experts: `n` output rows, `k` inputs (multiples of 64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertShape {
    /// Experts.
    pub experts: usize,
    /// Output rows.
    pub n: usize,
    /// Inputs.
    pub k: usize,
}

impl ExpertShape {
    fn check(self) -> Result<(usize, usize)> {
        if !self.n.is_multiple_of(WT) || !self.k.is_multiple_of(BK) || self.n == 0 || self.k == 0 {
            return Err(invalid(
                "expert shape",
                format!("{self:?}: n and k must be positive multiples of 64"),
            ));
        }
        Ok((self.k / BK, self.n / WT))
    }
}

/// Re-encodes an E4M3 scale (7 bits, sign removed, non-zero) as the n1 layout's 8-bit code: biased exponent
/// `E + 9` in bits 3-6, mantissa in bits 0-2, subnormals normalised.
///
/// # Errors
///
/// [`crate::ModelError::Invalid`] for a zero scale (the caller encodes zero itself) or an exponent outside -9..8.
pub fn enc_scale(c: u8) -> Result<u8> {
    let e = i32::from((c >> 3) & 15);
    let mm0 = i32::from(c & 7);
    let (big_e, mm) = if e == 0 {
        match mm0 {
            0 => return Err(invalid("nvfp4 scale", "zero scale must be handled by the caller")),
            1 => (-9, 0),
            2 | 3 => (-8, (mm0 - 2) * 4),
            _ => (-7, (mm0 - 4) * 2),
        }
    } else {
        (e - 7, mm0)
    };
    if !(-9..=8).contains(&big_e) {
        return Err(invalid(
            "nvfp4 scale",
            format!("exponent {big_e} of code {c:#x} outside -9..8"),
        ));
    }
    // 0 <= big_e + 9 <= 17 and 0 <= mm <= 7: the code fits in 8 bits
    Ok((((big_e + 9) << 3) | mm) as u8)
}

/// Checkpoint row of fragment row (`wt`, `mt`, `g`, `hh`): w13 interleaves gate row i and up row i (rows 2i, 2i + 1),
/// w2 takes rows g and g + 8 of each 16-row m-tile.
pub fn orig_row(p: Proj, wt: usize, mt: usize, g: usize, hh: usize) -> usize {
    match p {
        Proj::W13 => 2 * (wt * 32 + mt * 8 + g) + hh,
        Proj::W2 => wt * 64 + mt * 16 + g + 8 * hh,
    }
}

/// Bit position of code bit `bt` of the `j`-th value of a lane's k-chunk in the n1 word.
const POS: [[u32; 4]; 4] = [[6, 7, 8, 15], [3, 4, 5, 12], [0, 1, 2, 9], [10, 11, 13, 14]];

/// Runs `f(first expert, its output chunks)` over `threads` threads, each on a contiguous range of experts (the C++
/// `par`: step = ceil(experts / threads)).
fn par<F>(threads: usize, experts: usize, w: &mut [u8], w_per: usize, s: &mut [u8], s_per: usize, f: F) -> Result<()>
where
    F: Fn(usize, &mut [u8], &mut [u8]) -> Result<()> + Sync,
{
    let step = experts.div_ceil(threads.max(1)).max(1);
    thread::scope(|sc| {
        let handles: Vec<_> = w
            .chunks_mut(step * w_per)
            .zip(s.chunks_mut(step * s_per))
            .enumerate()
            .map(|(i, (wc, scs))| {
                let f = &f;
                sc.spawn(move || f(i * step, wc, scs))
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
    })
}

/// NVFP4 to the n1 layout. `w`: `[E][N][K/2]` packed codes (low nibble first), `s`: `[E][N][K/16]` E4M3 scales
/// (bit 7 = sign of the block: flips the code's sign bit; zero magnitude: all codes of the block are 0). Returns
/// (weights, scales).
///
/// # Errors
///
/// [`crate::ModelError::Invalid`] for a bad shape, input sizes that do not match it, or a scale [`enc_scale`]
/// refuses.
pub fn repack_nvfp4_n1(w: &[u8], s: &[u8], shape: ExpertShape, p: Proj, threads: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let (ks_n, nwt) = shape.check()?;
    let ExpertShape {
        experts,
        n: nn,
        k: kk_n,
    } = shape;
    if w.len() != experts * nn * kk_n / 2 || s.len() != experts * nn * kk_n / 16 {
        return Err(invalid(
            "nvfp4 sizes",
            format!("weights {} / scales {} bytes for {shape:?}", w.len(), s.len()),
        ));
    }
    const WB: usize = WT * BK / 2;
    const SB: usize = WT * BK / 16 * 2;
    let (w_per, s_per) = (nwt * ks_n * WB, nwt * ks_n * SB);
    let mut wn = vec![0u8; experts * w_per];
    let mut sn = vec![0u8; experts * s_per];
    par(threads, experts, &mut wn, w_per, &mut sn, s_per, |e0, wout, sout| {
        for (de, (wexp, sexp)) in wout.chunks_mut(w_per).zip(sout.chunks_mut(s_per)).enumerate() {
            let e = e0 + de;
            let scale = |n: usize, kb: usize| s[(e * nn + n) * (kk_n / 16) + kb];
            for wt in 0..nwt {
                for ks in 0..ks_n {
                    let blk = wt * ks_n + ks;
                    let wb = &mut wexp[blk * WB..(blk + 1) * WB];
                    let sb = &mut sexp[blk * SB..(blk + 1) * SB];
                    for mt in 0..4 {
                        for g in 0..8 {
                            for kk in 0..4 {
                                let mut wd = 0u32;
                                for hh in 0..2 {
                                    let c = scale(orig_row(p, wt, mt, g, hh), ks * 4 + kk) & 0x7f;
                                    let code = if c != 0 { u32::from(enc_scale(c)?) } else { 0 };
                                    wd |= ((code << 4) + 0x7600) << (16 * hh);
                                }
                                let o = ((mt * 8 + g) * 4 + kk) * 4;
                                sb[o..o + 4].copy_from_slice(&wd.to_le_bytes());
                            }
                        }
                        for lane in 0..32 {
                            let (g, t) = (lane >> 2, lane & 3);
                            for kk in 0..4 {
                                let mut q = 0u32;
                                for (j, pos_j) in POS.iter().enumerate() {
                                    for h in 0..2 {
                                        let n = orig_row(p, wt, mt, g, j & 1);
                                        let k = ks * 64 + kk * 16 + 2 * t + h + 8 * (j >> 1);
                                        let byte = w[(e * nn + n) * (kk_n / 2) + k / 2];
                                        let mut code = if k & 1 == 1 { byte >> 4 } else { byte & 15 };
                                        let sc = scale(n, k / 16);
                                        if sc & 0x80 != 0 {
                                            code ^= 8;
                                        }
                                        if sc & 0x7f == 0 {
                                            code = 0;
                                        }
                                        for (bt, pos) in pos_j.iter().enumerate() {
                                            if (code >> bt) & 1 == 1 {
                                                q |= 1 << (pos + 16 * h as u32);
                                            }
                                        }
                                    }
                                }
                                let o = ((mt * 32 + lane) * 4 + kk) * 4;
                                wb[o..o + 4].copy_from_slice(&q.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok((wn, sn))
}

/// NVFP4 to the n1s1 layout: identical `n1` weight bit layout (`2,048 B` per 64x64 block) with 1-byte [`enc_scale`]
/// scales (`256 B` per block instead of `512 B` BF16 scales). Saves 1.27 GB VRAM and 10% HBM bandwidth on NVFP4
/// layers while producing bit-identical dequantized BF16 weights.
///
/// # Errors
///
/// [`crate::ModelError::Invalid`] for a bad shape, input sizes that do not match it, or a scale [`enc_scale`]
/// refuses.
pub fn repack_nvfp4_n1s1(
    w: &[u8],
    s: &[u8],
    shape: ExpertShape,
    p: Proj,
    threads: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let (ks_n, nwt) = shape.check()?;
    let ExpertShape {
        experts,
        n: nn,
        k: kk_n,
    } = shape;
    if w.len() != experts * nn * kk_n / 2 || s.len() != experts * nn * kk_n / 16 {
        return Err(invalid(
            "nvfp4 sizes",
            format!("weights {} / scales {} bytes for {shape:?}", w.len(), s.len()),
        ));
    }
    const WB: usize = WT * BK / 2;
    const SB: usize = WT * BK / 16;
    let (w_per, s_per) = (nwt * ks_n * WB, nwt * ks_n * SB);
    let mut wn = vec![0u8; experts * w_per];
    let mut sn = vec![0u8; experts * s_per];
    par(threads, experts, &mut wn, w_per, &mut sn, s_per, |e0, wout, sout| {
        for (de, (wexp, sexp)) in wout.chunks_mut(w_per).zip(sout.chunks_mut(s_per)).enumerate() {
            let e = e0 + de;
            let scale = |n: usize, kb: usize| s[(e * nn + n) * (kk_n / 16) + kb];
            for wt in 0..nwt {
                for ks in 0..ks_n {
                    let blk = wt * ks_n + ks;
                    let wb = &mut wexp[blk * WB..(blk + 1) * WB];
                    let sb = &mut sexp[blk * SB..(blk + 1) * SB];
                    for mt in 0..4 {
                        for g in 0..8 {
                            for kk in 0..4 {
                                for hh in 0..2 {
                                    let c = scale(orig_row(p, wt, mt, g, hh), ks * 4 + kk) & 0x7f;
                                    sb[(mt * 8 + g) * 8 + kk * 2 + hh] = if c != 0 { enc_scale(c)? } else { 0 };
                                }
                            }
                        }
                        for lane in 0..32 {
                            let (g, t) = (lane >> 2, lane & 3);
                            for kk in 0..4 {
                                let mut q = 0u32;
                                for (j, pos_j) in POS.iter().enumerate() {
                                    for h in 0..2 {
                                        let n = orig_row(p, wt, mt, g, j & 1);
                                        let k = ks * 64 + kk * 16 + 2 * t + h + 8 * (j >> 1);
                                        let byte = w[(e * nn + n) * (kk_n / 2) + k / 2];
                                        let mut code = if k & 1 == 1 { byte >> 4 } else { byte & 15 };
                                        let sc = scale(n, k / 16);
                                        if sc & 0x80 != 0 {
                                            code ^= 8;
                                        }
                                        if sc & 0x7f == 0 {
                                            code = 0;
                                        }
                                        for (bt, pos) in pos_j.iter().enumerate() {
                                            if (code >> bt) & 1 == 1 {
                                                q |= 1 << (pos + 16 * h as u32);
                                            }
                                        }
                                    }
                                }
                                let o = ((mt * 32 + lane) * 4 + kk) * 4;
                                wb[o..o + 4].copy_from_slice(&q.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok((wn, sn))
}

/// Smallest and largest e8m0 scale code the MXFP8 decode is exact for.
pub const E8M0_RANGE: (u8, u8) = (40, 142);

/// MXFP8 to the plain fragment layout. `w`: `[E][N][K]` E4M3 codes, `s`: `[E][N][K/32]` e8m0 scales (each in
/// [`E8M0_RANGE`]). Returns (weights, scales).
///
/// # Errors
///
/// [`crate::ModelError::Invalid`] for a bad shape, sizes that do not match it, or a scale outside the range.
pub fn repack_mxfp8(w: &[u8], s: &[u8], shape: ExpertShape, p: Proj, threads: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let (ks_n, nwt) = shape.check()?;
    let ExpertShape {
        experts,
        n: nn,
        k: kk_n,
    } = shape;
    if w.len() != experts * nn * kk_n || s.len() != experts * nn * kk_n / 32 {
        return Err(invalid(
            "mxfp8 sizes",
            format!("weights {} / scales {} bytes for {shape:?}", w.len(), s.len()),
        ));
    }
    let (lo, hi) = (s.iter().min().copied(), s.iter().max().copied());
    if lo < Some(E8M0_RANGE.0) || hi > Some(E8M0_RANGE.1) {
        return Err(invalid(
            "mxfp8 scale",
            format!("e8m0 codes {lo:?}..{hi:?} outside the exact range of the decode {E8M0_RANGE:?}"),
        ));
    }
    const WB: usize = WT * BK;
    const SB: usize = WT * BK / 32;
    let (w_per, s_per) = (nwt * ks_n * WB, nwt * ks_n * SB);
    let mut wn = vec![0u8; experts * w_per];
    let mut sn = vec![0u8; experts * s_per];
    par(threads, experts, &mut wn, w_per, &mut sn, s_per, |e0, wout, sout| {
        for (de, (wexp, sexp)) in wout.chunks_mut(w_per).zip(sout.chunks_mut(s_per)).enumerate() {
            let e = e0 + de;
            let at = |n: usize, kx: usize| w[(e * nn + n) * kk_n + kx];
            for wt in 0..nwt {
                for ks in 0..ks_n {
                    let blk = wt * ks_n + ks;
                    let wb = &mut wexp[blk * WB..(blk + 1) * WB];
                    let sb = &mut sexp[blk * SB..(blk + 1) * SB];
                    for mt in 0..4 {
                        for g in 0..8 {
                            for grp in 0..2 {
                                for hh in 0..2 {
                                    let n = orig_row(p, wt, mt, g, hh);
                                    sb[(mt * 8 + g) * 4 + grp * 2 + hh] = s[(e * nn + n) * (kk_n / 32) + ks * 2 + grp];
                                }
                            }
                        }
                        for lane in 0..32 {
                            let (g, t) = (lane >> 2, lane & 3);
                            let (ng, nh) = (orig_row(p, wt, mt, g, 0), orig_row(p, wt, mt, g, 1));
                            for kk in 0..4 {
                                for half in 0..2 {
                                    let k = ks * 64 + kk * 16 + 2 * t + 8 * half;
                                    let o = ((mt * 32 + lane) * 4 + kk) * 8 + half * 4;
                                    wb[o..o + 4].copy_from_slice(&[at(nh, k), at(ng, k), at(nh, k + 1), at(ng, k + 1)]);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok((wn, sn))
}

#[cfg(test)]
mod tests {
    //! Expected values derived by hand from the C++ loops (`weights.cpp` @ 66fe353), on one expert of one 64 x 64
    //! block (KS = NWT = 1): which input byte lands at which output byte, and with which bits.
    use super::*;

    const ONE: ExpertShape = ExpertShape {
        experts: 1,
        n: 64,
        k: 64,
    };

    fn word(b: &[u8], i: usize) -> u32 {
        u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())
    }

    #[test]
    fn enc_scale_by_hand() {
        // e = 7, m = 0 -> E = 0 -> (9 << 3) | 0
        assert_eq!(enc_scale(0x38).unwrap(), 72);
        // subnormals: m0 = 1 -> E -9, m 0; m0 = 3 -> E -8, m (3 - 2) * 4 = 4; m0 = 5 -> E -7, m (5 - 4) * 2 = 2
        assert_eq!(enc_scale(0x01).unwrap(), 0);
        assert_eq!(enc_scale(0x03).unwrap(), (1 << 3) | 4);
        assert_eq!(enc_scale(0x05).unwrap(), (2 << 3) | 2);
        // largest: e = 15, m = 7 -> E 8 -> (17 << 3) | 7 = 143
        assert_eq!(enc_scale(0x7f).unwrap(), 143);
        assert!(enc_scale(0).is_err());
    }

    #[test]
    fn orig_row_by_hand() {
        assert_eq!(orig_row(Proj::W13, 0, 0, 0, 1), 1);
        assert_eq!(orig_row(Proj::W13, 1, 3, 7, 1), 2 * (32 + 24 + 7) + 1);
        assert_eq!(orig_row(Proj::W2, 0, 0, 0, 1), 8);
        assert_eq!(orig_row(Proj::W2, 2, 3, 7, 1), 128 + 48 + 7 + 8);
    }

    #[test]
    fn nvfp4_scale_word() {
        let w = vec![0u8; 64 * 32];
        let mut s = vec![0u8; 64 * 4];
        // row 0 (w13: mt 0, g 0, hh 0), k block 0 (ks 0, kk 0): code 0x38 -> 72 -> (72 << 4) + 0x7600 = 0x7a80;
        // row 1 (hh 1) is zero -> 0x7600 in the high half.
        s[0] = 0x38;
        // row 1, k block 2 with the sign bit: magnitude 0x01 -> 0, so (0 << 4) + 0x7600 in the high half of word kk 2
        s[4 + 2] = 0x81;
        let (_, sn) = repack_nvfp4_n1(&w, &s, ONE, Proj::W13, 1).unwrap();
        assert_eq!(sn.len(), 512);
        assert_eq!(word(&sn, 0), 0x7600_7a80);
        assert_eq!(word(&sn, 2), 0x7600_7600);
        let others = (1..128).filter(|&i| word(&sn, i) != 0x7600_7600).count();
        assert_eq!(others, 0);
    }

    #[test]
    fn nvfp4_n1s1_matches_n1() {
        let mut w = vec![0u8; 64 * 32];
        let mut s = vec![0u8; 64 * 4];
        w[0] = 0x55;
        s[0] = 0x38;
        s[4 + 2] = 0x81;
        let (wn2, sn2) = repack_nvfp4_n1(&w, &s, ONE, Proj::W13, 1).unwrap();
        let (wn1, sn1) = repack_nvfp4_n1s1(&w, &s, ONE, Proj::W13, 1).unwrap();
        assert_eq!(wn1, wn2);
        assert_eq!(sn1.len(), 256);
        // Every 1-byte scale code expands via (code << 4) + 0x7600 to the exact 2-byte BF16 scale of n1
        for (i, pair) in sn1.as_chunks::<2>().0.iter().enumerate() {
            let lo = (u32::from(pair[0]) << 4) + 0x7600;
            let hi = (u32::from(pair[1]) << 4) + 0x7600;
            assert_eq!(lo | (hi << 16), word(&sn2, i));
        }
    }

    #[test]
    fn nvfp4_code_bits() {
        let mut w = vec![0u8; 64 * 32];
        let mut s = vec![0u8; 64 * 4];
        // w13 row 0, k 0 (low nibble) = 5 = bits 0, 2 -> j 0: POS 6, 8; lane 0, kk 0 -> word 0
        // w13 row 0, k 1 (high nibble) = 5 -> h 1: the same bits + 16
        w[0] = 0x55;
        s[0] = 0x38;
        // w13 row 2 (mt 0, g 1, hh 0) k 0: 3 with a negative block scale -> 3 ^ 8 = 11 = bits 0, 1, 3 -> POS 6, 7, 15;
        // lane g * 4 + t = 4, kk 0 -> word (0 * 32 + 4) * 4 = 16
        w[2 * 32] = 0x03;
        s[2 * 4] = 0xb8;
        // w13 row 1 (hh 1 -> j 1), k 8 (j >> 1 = 1 -> j 3): 1 -> POS[3][0] = 10, lane 0, kk 0, h 0 -> word 0
        w[32 + 4] = 0x01;
        s[4] = 0x38;
        let (wn, _) = repack_nvfp4_n1(&w, &s, ONE, Proj::W13, 1).unwrap();
        assert_eq!(wn.len(), 2048);
        let want0 = (1u32 << 6) | (1 << 8) | (1 << 22) | (1 << 24) | (1 << 10);
        assert_eq!(word(&wn, 0), want0);
        // The sign flip applies to every code of the negative block, zero codes too (0 ^ 8 = 8 = bit 3): row 2's
        // other values in lane 4 are k 1 (j 0, h 1), k 8 (j 2, h 0), k 9 (j 2, h 1) -> POS[0][3] + 16, POS[2][3],
        // POS[2][3] + 16 = 31, 9, 25.
        let sign = (1u32 << 15) | (1 << 31) | (1 << 9) | (1 << 25);
        assert_eq!(word(&wn, 16), (1 << 6) | (1 << 7) | sign);
        // lanes 5-7 (t 1-3) of row 2, kk 0: all zero codes of the negative block -> bit 3 of j 0 and j 2, h 0 and 1
        for lane in 5..8 {
            assert_eq!(word(&wn, lane * 4), sign, "lane {lane}");
        }
        let nonzero: Vec<usize> = (0..512).filter(|&i| word(&wn, i) != 0).collect();
        assert_eq!(nonzero, [0, 16, 20, 24, 28]);
    }

    #[test]
    fn nvfp4_zero_scale_clears_codes_and_w2_rows() {
        let mut w = vec![0u8; 64 * 32];
        let mut s = vec![0u8; 64 * 4];
        // w2 row 8 = (mt 0, g 0, hh 1) -> j 1; k 0: code 5 -> POS[1][0], POS[1][2] = 3, 5; lane 0, kk 0
        w[8 * 32] = 0x05;
        s[8 * 4] = 0x38;
        // w2 row 0, k 16 (kk 1): code 7 but scale 0x80 (zero magnitude) -> code 0
        w[8] = 0x07;
        s[1] = 0x80;
        let (wn, _) = repack_nvfp4_n1(&w, &s, ONE, Proj::W2, 1).unwrap();
        assert_eq!(word(&wn, 0), (1 << 3) | (1 << 5));
        assert_eq!((1..512).filter(|&i| word(&wn, i) != 0).count(), 0);
    }

    #[test]
    fn mxfp8_by_hand() {
        let mut w = vec![0u8; 64 * 64];
        let mut s = vec![127u8; 64 * 2];
        // w13 row 1 = nh of (mt 0, g 0); k 0 -> lane 0, kk 0, half 0: byte 0
        w[64] = 0xab;
        // row 0 = ng, k 1 -> byte 3
        w[1] = 0xcd;
        // row 0, k 8 -> half 1, d[1] -> byte 5
        w[8] = 0xef;
        // row 3 = nh of (mt 0, g 1), k 18 = kk 1 + 2 * t (t 1) -> lane 5, byte ((5 * 4 + 1) * 8) + 0 = 168
        w[3 * 64 + 18] = 0x11;
        // scale of row 1 (g 0, hh 1), group 1 -> sb[0 * 4 + 1 * 2 + 1] = sb[3]
        s[2 + 1] = 100;
        let (wn, sn) = repack_mxfp8(&w, &s, ONE, Proj::W13, 1).unwrap();
        assert_eq!((wn.len(), sn.len()), (4096, 128));
        assert_eq!(&wn[0..8], &[0xab, 0, 0, 0xcd, 0, 0xef, 0, 0]);
        assert_eq!(wn[168], 0x11);
        assert_eq!(wn.iter().filter(|&&b| b != 0).count(), 4);
        assert_eq!(sn[3], 100);
        assert_eq!(sn.iter().filter(|&&b| b != 127).count(), 1);
    }

    #[test]
    fn mxfp8_scale_range_is_checked() {
        let w = vec![0u8; 64 * 64];
        let mut s = vec![127u8; 64 * 2];
        s[5] = 39;
        assert!(repack_mxfp8(&w, &s, ONE, Proj::W13, 1).is_err());
        s[5] = 143;
        assert!(repack_mxfp8(&w, &s, ONE, Proj::W13, 1).is_err());
        s[5] = 142;
        assert!(repack_mxfp8(&w, &s, ONE, Proj::W13, 1).is_ok());
    }

    #[test]
    fn threads_do_not_change_bytes() {
        let shape = ExpertShape {
            experts: 5,
            n: 128,
            k: 128,
        };
        let mut x = 0x9e37_79b9u32;
        let mut rnd = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    (x >> 24) as u8
                })
                .collect()
        };
        let w4 = rnd(5 * 128 * 64);
        let s4 = rnd(5 * 128 * 8);
        let one = repack_nvfp4_n1(&w4, &s4, shape, Proj::W2, 1).unwrap();
        assert_eq!(one, repack_nvfp4_n1(&w4, &s4, shape, Proj::W2, 3).unwrap());
        assert_eq!(one, repack_nvfp4_n1(&w4, &s4, shape, Proj::W2, 16).unwrap());
        let w8 = rnd(5 * 128 * 128);
        let s8: Vec<u8> = rnd(5 * 128 * 4).iter().map(|b| 40 + b % 100).collect();
        let one = repack_mxfp8(&w8, &s8, shape, Proj::W13, 1).unwrap();
        assert_eq!(one, repack_mxfp8(&w8, &s8, shape, Proj::W13, 4).unwrap());
    }
}
