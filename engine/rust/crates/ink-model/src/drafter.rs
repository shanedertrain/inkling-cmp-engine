// Ported from engine/native/src/drafter_rt.cuh @ 1c7ac46 (`load_weights`: the host side of the DSpark drafter's
// weight load; the kernels of the load run in ink-runtime). Own code of this project.
//! The DSpark drafter's files on the host (package R8): its dimensions, the two checkpoints it reads
//! (`mtp.safetensors` and the model checkpoint's head, embedding and embedding norm), config B's fold (`fold.bin`)
//! and its rank-16 basis (`lowrank16.bin`), and the small host computations of `load_weights` (the fused alpha
//! softmax, the rnn bias widened to fp32, the column split of the rnn projection). Contract:
//! `engine/kernels/drafter/README.md`.

use std::fs;
use std::path::{Path, PathBuf};

use safetensors::Dtype;

use crate::checkpoint::{CHECKPOINT_FILE, Checkpoint};
use crate::error::{ModelError, Result, invalid};

/// Hidden size.
pub const H: usize = 2304;
/// Query heads.
pub const NH: usize = 24;
/// Key/value heads.
pub const KVH: usize = 8;
/// Head dimension.
pub const HD: usize = 128;
/// MLP width.
pub const INTER: usize = 9216;
/// Markov rank.
pub const RANK: usize = 256;
/// Rows of one draft block: the anchor and 7 masks.
pub const BLK: usize = 8;
/// Sliding window of layers 0 and 2 (keys before the block).
pub const WIN: usize = 1024;
/// Padded vocabulary (head rows).
pub const VP: usize = 201_024;
/// Valid vocabulary (argmax range).
pub const VU: usize = 200_058;
/// Tokens per draft K/V page.
pub const PAGE: usize = 64;
/// Drafts per session per round.
pub const DRAFTS: usize = 7;
/// K chunk the fragment GEMMs stage in shared memory (`KC`) and its shared row stride (`XS` = `KC` + 8).
pub const XS: usize = 256 + 8;
/// Keys per attention tile (`AT_TILE`).
pub const AT_TILE: usize = 64;
/// Dynamic shared memory of `attn_block` / `attn_block8`: (4 x 64 + 32) x (128 + 8) x 2.
pub const ATTN_SMEM: u32 = ((4 * AT_TILE + 32) * (HD + 8) * 2) as u32;
/// Rows of one Markov seed chunk (`MK_CHUNK`).
pub const MK_CHUNK: usize = 2048;
/// Markov seed chunks over the valid vocabulary (`MK_NCH` = 98).
pub const MK_NCH: usize = VU.div_ceil(MK_CHUNK);
/// Chunks of `markov_argmax3`: 32 hot chunks of 512 rows, then 2,048-row chunks (`MK3_NCH` = 122).
pub const MK3_NCH: usize = 16_384 / 512 + (VU - 16_384).div_ceil(MK_CHUNK);
/// Positions of the RoPE cos/sin table (`drafter_rt.cuh`: 270,336).
pub const COS_SIN_POS: usize = 270_336;
/// Output width of the rnn projection.
pub const RNN_OUT: usize = 768;
/// Input width of the rnn projection: the state (512) and the context [h, f_rnn] (4,608).
pub const RNN_IN: usize = 5120;
/// State columns of the rnn projection.
pub const RNN_STATE: usize = 512;
/// CTAs the attention split aims at (`g_attn_ctas`).
pub const ATTN_CTAS: usize = 384;
/// Rank of the fold basis (`lowrank16.bin`, `LRK`).
pub const LRK: usize = 16;
/// Vocabulary rows per chunk of the drafter probability's sum (`draft_conf.cuh`: `CONF_ROWS`).
pub const CONF_ROWS: usize = 1024;
/// Chunks of that sum (`CONF_NCH` = 196).
pub const CONF_NCH: usize = VU.div_ceil(CONF_ROWS);
/// The drafter checkpoint in the model directory.
pub const MTP_FILE: &str = "mtp.safetensors";
/// The rank-16 fold basis, next to `fold.bin`.
pub const LOWRANK_FILE: &str = "lowrank16.bin";

/// The drafter's two checkpoints.
#[derive(Debug)]
pub struct DraftCheckpoints {
    /// `mtp.safetensors`: the drafter's layers and Markov head.
    pub mtp: Checkpoint,
    /// The model checkpoint: head (`unembed`), embedding and embedding norm.
    pub model: Checkpoint,
}

impl DraftCheckpoints {
    /// Maps both checkpoints of `model_dir`.
    ///
    /// # Errors
    ///
    /// As [`Checkpoint::open`].
    pub fn open(model_dir: &Path) -> Result<DraftCheckpoints> {
        Ok(DraftCheckpoints {
            mtp: Checkpoint::open(&model_dir.join(MTP_FILE))?,
            model: Checkpoint::open(&model_dir.join(CHECKPOINT_FILE))?,
        })
    }
}

/// The raw bytes of BF16 tensor `name` of `ck`, which must hold exactly `elems` elements (`load_bf16`).
///
/// # Errors
///
/// [`ModelError::Missing`], [`ModelError::Dtype`], or [`ModelError::Invalid`] for another element count.
pub fn bf16_bytes<'a>(ck: &'a Checkpoint, name: &str, elems: usize) -> Result<&'a [u8]> {
    let t = ck.tensor(name)?;
    t.expect(Dtype::BF16)?;
    if t.bytes.len() != elems * 2 {
        return Err(invalid(
            "drafter tensor",
            format!("{name}: {} bytes, expected {}", t.bytes.len(), elems * 2),
        ));
    }
    Ok(t.bytes)
}

fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    half::bf16::from_bits(u16::from_le_bytes([lo, hi])).to_f32()
}

/// `fuse_alpha = softmax(fuse_weight.float(), -1)` over the 4 x 4 bf16 `fuse_weight`, row by row in fp32 with the
/// C++ operation order (max, then the sum of `exp(x - max)` in column order, then each divided by the sum).
///
/// # Errors
///
/// [`ModelError::Invalid`] unless `bytes` holds 16 bf16 values.
pub fn fuse_alpha(bytes: &[u8]) -> Result<[f32; 16]> {
    if bytes.len() != 32 {
        return Err(invalid("fuse_weight", format!("{} bytes, expected 32", bytes.len())));
    }
    let x: Vec<f32> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16_to_f32(c[0], c[1]))
        .collect();
    let mut al = [0f32; 16];
    for r in 0..4 {
        let mut mx = -1e30f32;
        for c in 0..4 {
            mx = mx.max(x[r * 4 + c]);
        }
        let mut s = 0f32;
        for c in 0..4 {
            al[r * 4 + c] = (x[r * 4 + c] - mx).exp();
            s += al[r * 4 + c];
        }
        for c in 0..4 {
            al[r * 4 + c] /= s;
        }
    }
    Ok(al)
}

/// The bf16 values of `bytes` widened to fp32 (exact), as the C++ `p6::bf16_to_f32` loop of the rnn bias.
pub fn widen_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16_to_f32(c[0], c[1]))
        .collect()
}

/// Columns `c0 .. c0 + ncols` of a row-major bf16 matrix of `rows` rows and row stride `lds` (`k_copy_cols`, done on
/// the host: a pure copy).
///
/// # Errors
///
/// [`ModelError::Invalid`] when the columns do not lie inside the rows or `src` is too short.
pub fn bf16_cols(src: &[u8], rows: usize, lds: usize, c0: usize, ncols: usize) -> Result<Vec<u8>> {
    if c0 + ncols > lds || src.len() < rows * lds * 2 {
        return Err(invalid(
            "column split",
            format!(
                "{rows} x {lds} source of {} bytes, columns {c0}..{}",
                src.len(),
                c0 + ncols
            ),
        ));
    }
    let mut out = Vec::with_capacity(rows * ncols * 2);
    for r in 0..rows {
        out.extend_from_slice(&src[(r * lds + c0) * 2..(r * lds + c0 + ncols) * 2]);
    }
    Ok(out)
}

/// Config B's fp64 fold, stored as bf16 (`fold.bin`, written by `probes/s2f/cfgb_drafter.py dumpw`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fold {
    /// `fold_w` `[VP][RANK]` bf16 (raw bytes).
    pub w: Vec<u8>,
    /// `fold_b` `[VP]` bf16 (raw bytes).
    pub b: Vec<u8>,
}

/// The rank-16 fold basis (`lowrank16.bin`).
#[derive(Debug, Clone, PartialEq)]
pub struct LowRank {
    /// `V` `[16][256]` fp32.
    pub v: Vec<f32>,
    /// `U` `[VP][16]` bf16 (raw bytes).
    pub u: Vec<u8>,
    /// `P` `[VP]` fp32.
    pub p: Vec<f32>,
    /// `Q` `[VP]` fp32.
    pub q: Vec<f32>,
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|source| ModelError::Io {
        path: path.to_owned(),
        source,
    })
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// Splits `data` into consecutive pieces of the given byte sizes; trailing bytes are ignored (the C++ reads with
/// `fread` and never checks for more).
fn pieces<'a>(data: &'a [u8], sizes: &[usize], what: &'static str) -> Result<Vec<&'a [u8]>> {
    let need: usize = sizes.iter().sum();
    if data.len() < need {
        return Err(invalid(what, format!("{} bytes, need at least {need}", data.len())));
    }
    let mut out = Vec::with_capacity(sizes.len());
    let mut at = 0;
    for &s in sizes {
        out.push(&data[at..at + s]);
        at += s;
    }
    Ok(out)
}

fn parse_fold(data: &[u8], vp: usize, rank: usize) -> Result<Fold> {
    let p = pieces(data, &[vp * rank * 2, vp * 2], "fold.bin")?;
    Ok(Fold {
        w: p[0].to_vec(),
        b: p[1].to_vec(),
    })
}

fn parse_lowrank(data: &[u8], vp: usize) -> Result<LowRank> {
    let p = pieces(data, &[16 * 256 * 4, vp * 16 * 2, vp * 4, vp * 4], "lowrank16.bin")?;
    Ok(LowRank {
        v: f32s(p[0]),
        u: p[1].to_vec(),
        p: f32s(p[2]),
        q: f32s(p[3]),
    })
}

/// Reads `fold.bin`.
///
/// # Errors
///
/// [`ModelError::Io`], or [`ModelError::Invalid`] for a file shorter than `[VP][RANK] + [VP]` bf16.
pub fn read_fold(path: &Path) -> Result<Fold> {
    parse_fold(&read_file(path)?, VP, RANK)
}

/// The path of `lowrank16.bin`: the directory of `fold_bin` (as the C++ cuts the path at its last `/`).
pub fn lowrank_path(fold_bin: &Path) -> PathBuf {
    fold_bin
        .parent()
        .map_or_else(|| PathBuf::from(LOWRANK_FILE), |d| d.join(LOWRANK_FILE))
}

/// Reads `lowrank16.bin`.
///
/// # Errors
///
/// [`ModelError::Io`], or [`ModelError::Invalid`] for a short file.
pub fn read_lowrank(path: &Path) -> Result<LowRank> {
    parse_lowrank(&read_file(path)?, VP)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn constants_match_the_kernel_header() {
        assert_eq!(MK_NCH, 98);
        assert_eq!(MK3_NCH, 122);
        assert_eq!(ATTN_SMEM, 78_336);
        assert_eq!(RNN_IN - RNN_STATE, 2 * H);
    }

    #[test]
    fn fuse_alpha_is_a_row_softmax() {
        let al = fuse_alpha(&bf(&[
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -2.0, 3.0, 0.5, 0.5, 9.0, 9.0, 9.0, 9.0,
        ]))
        .unwrap();
        for r in 0..4 {
            let s: f32 = al[r * 4..r * 4 + 4].iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "row {r} sums to {s}");
        }
        assert_eq!(al[0], 0.25);
        assert_eq!(al[12], 0.25);
        assert!(al[4] > al[5] && al[5] == al[6]);
        assert!(al[9] > al[10] && al[10] == al[11] && al[11] > al[8]);
        assert!(fuse_alpha(&[0; 30]).is_err());
    }

    #[test]
    fn column_split_copies_the_named_columns() {
        // 2 x 3 matrix [[1,2,3],[4,5,6]]: columns 1..3
        let src = bf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(bf16_cols(&src, 2, 3, 1, 2).unwrap(), bf(&[2.0, 3.0, 5.0, 6.0]));
        assert_eq!(bf16_cols(&src, 2, 3, 0, 1).unwrap(), bf(&[1.0, 4.0]));
        assert!(bf16_cols(&src, 2, 3, 2, 2).is_err());
        assert_eq!(widen_bf16(&bf(&[1.5, -2.0])), vec![1.5, -2.0]);
    }

    #[test]
    fn fold_and_lowrank_files_split_in_order_and_ignore_trailing_bytes() {
        let (vp, rank) = (3, 2);
        let mut data: Vec<u8> = (0..(vp * rank * 2 + vp * 2) as u8).collect();
        data.extend_from_slice(&[0xee; 5]);
        let f = parse_fold(&data, vp, rank).unwrap();
        assert_eq!(f.w, (0..12).collect::<Vec<u8>>());
        assert_eq!(f.b, (12..18).collect::<Vec<u8>>());
        assert!(parse_fold(&data[..17], vp, rank).is_err());
        let mut lr = Vec::new();
        lr.extend((0..16 * 256).flat_map(|i| (i as f32).to_le_bytes()));
        lr.extend(std::iter::repeat_n(7u8, vp * 32));
        lr.extend([1.0f32, 2.0, 3.0].iter().flat_map(|x| x.to_le_bytes()));
        lr.extend([4.0f32, 5.0, 6.0].iter().flat_map(|x| x.to_le_bytes()));
        let l = parse_lowrank(&lr, vp).unwrap();
        assert_eq!(l.v[4095], 4095.0);
        assert_eq!(l.u.len(), vp * 32);
        assert_eq!((l.p.clone(), l.q.clone()), (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]));
        assert!(parse_lowrank(&lr[..lr.len() - 1], vp).is_err());
        assert_eq!(
            lowrank_path(Path::new("/s2f/fold.bin")),
            PathBuf::from("/s2f/lowrank16.bin")
        );
        assert_eq!(lowrank_path(Path::new("fold.bin")), PathBuf::from("lowrank16.bin"));
    }
}
