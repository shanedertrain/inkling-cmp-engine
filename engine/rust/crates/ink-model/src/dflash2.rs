//! Inkling-2-Mini-DFlash2 checkpoint and model dimensions.
//!
//! Checkpoint layout (`incoai/Inkling-2-Mini-DFlash2`):
//! - `model.safetensors`: 81 BF16 tensors (fc, hidden_norm, 5 Qwen3-DFlash2 layers with 2-tap grouped dynamic convs,
//!   final norm, candidate_selector hidden_projection + predecessor/successor codebooks).
//! - Shared target tensors (from target checkpoint or engine weights): `model.llm.embed.weight`,
//!   `model.llm.embed_norm.weight`, `model.llm.unembed.weight`.
use std::path::Path;

use safetensors::Dtype;

use crate::checkpoint::{CHECKPOINT_FILE, Checkpoint};
use crate::error::{Result, invalid};

/// Hidden dimension.
pub const H: usize = 2304;
/// Query heads.
pub const NH: usize = 32;
/// Key/value heads.
pub const KVH: usize = 8;
/// Head dimension.
pub const HD: usize = 128;
/// MLP intermediate dimension.
pub const INTER: usize = 6912;
/// Number of draft decoder layers.
pub const NLAYERS: usize = 5;
/// Draft block size (1 anchor + 7 mask tokens).
pub const BLK: usize = 8;
/// Draft tokens proposed per round.
pub const DRAFTS: usize = 7;
/// Mask token id (`config.json` `dflash_config.mask_token_id`).
pub const MASK_TOKEN_ID: usize = 200_056;
/// Valid (unpadded) vocabulary size of Inkling-2-Mini.
pub const VALID_VOCAB: usize = 200_058;
/// Padded head projection rows (multiple of 64 for cuBLASLt Tensor Cores).
pub const HEAD_N: usize = 200_064;
/// Full checkpoint vocabulary rows (padded storage).
pub const VOCAB_PAD: usize = 201_024;
/// Sliding window size (all 5 layers).
pub const SWA: usize = 2048;
/// Per-session KV ring capacity (4096 = 2 x SWA, preserves [P - 2048, P) across rollbacks up to 2048 tokens).
pub const RING: usize = 4096;
/// Dynamic conv groups (`H / 16 = 144`).
pub const CONV_GROUPS: usize = 144;
/// Dynamic conv taps (`2`).
pub const CONV_TAPS: usize = 2;
/// Dynamic conv projection output width (`2 sides x 2 taps x 144 groups = 576`).
pub const CONV_PROJ_N: usize = 2 * CONV_TAPS * CONV_GROUPS;
/// Candidate selector top-k.
pub const TOP_K: usize = 16;
/// Candidate selector low-rank state dimension.
pub const SEL_RANK: usize = 256;
/// Top-k stage-1 chunks over `VALID_VOCAB`.
pub const TOPK_CHUNKS: usize = 64;
/// RoPE theta for DFlash2 (`10000.0`, distinct from target/DSpark `1000000.0`).
pub const ROPE_THETA: f32 = 10_000.0;
/// Positions precomputed in the RoPE cos/sin table (`262144 + 64`).
pub const COS_SIN_POS: usize = 262_208;
/// Maximum split-KV partitions for SWA-2048 block attention.
pub const ATTN_SPLITS: usize = 16;
/// Dynamic shared memory bytes for `df2::swa_attn_split_kernel`: `(4 * 64 + 32) * (HD + 8) * 2`.
pub const ATTN_SMEM: usize = (4 * 64 + 32) * (HD + 8) * 2;
/// Checkpoint filename inside the DFlash2 model directory.
pub const DFLASH2_FILE: &str = "model.safetensors";

/// Mapped checkpoints for DFlash2.
#[derive(Debug)]
pub struct DFlash2Checkpoints {
    /// DFlash2 `model.safetensors`.
    pub dflash: Checkpoint,
    /// Optional target checkpoint (when not sharing the target engine's embed/unembed buffers).
    pub target: Option<Checkpoint>,
}

impl DFlash2Checkpoints {
    /// Opens the DFlash2 checkpoint and (optionally) the target checkpoint.
    ///
    /// # Errors
    ///
    /// As [`Checkpoint::open`].
    pub fn open(dflash_dir: &Path, target_dir: Option<&Path>) -> Result<Self> {
        let dflash = Checkpoint::open(&dflash_dir.join(DFLASH2_FILE))?;
        let target = match target_dir {
            Some(dir) => Some(Checkpoint::open(&dir.join(CHECKPOINT_FILE))?),
            None => None,
        };
        Ok(Self { dflash, target })
    }
}

/// Returns the raw BF16 bytes of `name` in `ck`, verifying exact element count `elems`.
///
/// # Errors
///
/// [`ModelError::Missing`], [`ModelError::Dtype`], or [`ModelError::Invalid`].
pub fn bf16_bytes<'a>(ck: &'a Checkpoint, name: &str, elems: usize) -> Result<&'a [u8]> {
    let t = ck.tensor(name)?;
    t.expect(Dtype::BF16)?;
    if t.bytes.len() != elems * 2 {
        return Err(invalid(
            "dflash2 tensor",
            format!("{name}: {} bytes, expected {}", t.bytes.len(), elems * 2),
        ));
    }
    Ok(t.bytes)
}

/// Concatenates multiple BF16 tensors row-wise into one host byte vector.
///
/// # Errors
///
/// As [`bf16_bytes`].
pub fn cat_bf16(ck: &Checkpoint, parts: &[(&str, usize)]) -> Result<Vec<u8>> {
    let total: usize = parts.iter().map(|(_, e)| e * 2).sum();
    let mut out = Vec::with_capacity(total);
    for &(name, elems) in parts {
        out.extend_from_slice(bf16_bytes(ck, name, elems)?);
    }
    Ok(out)
}
