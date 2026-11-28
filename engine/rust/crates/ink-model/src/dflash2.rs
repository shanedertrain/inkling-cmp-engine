//! Inkling-2-Mini-DFlash2 checkpoint and model dimensions.
//!
//! Checkpoint layout (`incoai/Inkling-2-Mini-DFlash2`):
//! - `model.safetensors`: 81 BF16 tensors (fc, hidden_norm, 5 Qwen3-DFlash2 layers with 2-tap grouped dynamic convs,
//!   final norm, candidate_selector hidden_projection + predecessor/successor codebooks).
//! - Shared target tensors (from target checkpoint or engine weights): `model.llm.embed.weight`,
//!   `model.llm.embed_norm.weight`, `model.llm.unembed.weight`.
use std::fs;
use std::path::Path;

use safetensors::Dtype;
use serde::Deserialize;

use crate::checkpoint::{CHECKPOINT_FILE, Checkpoint};
use crate::error::{ModelError, Result, invalid};

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
/// RoPE theta for DFlash2 SWA layers (`10000.0`, distinct from target/DSpark `1000000.0`).
pub const ROPE_THETA: f32 = 10_000.0;
/// Default RoPE theta for DFlash2 global (`full_attention`) layers.
pub const GLOBAL_ROPE_THETA: f32 = 1_000_000.0;
/// Default `draft_ctx` when any DFlash2 layer uses `full_attention` (paged E4M3 global KV supports full 262K).
pub const GLOBAL_DEFAULT_CTX: u32 = 262_144;
/// Tokens per paged global E4M3 KV page (matches target engine `PAGE = 1024`).
pub const GLOBAL_PAGE: usize = 1024;
/// Global page-table row stride per session (`264 * 1024 = 270,336 >= 262,144 + BLK`).
pub const GLOBAL_KPT: usize = 264;
/// Positions precomputed in the RoPE cos/sin table (`262144 + 64`).
pub const COS_SIN_POS: usize = 262_208;
/// Maximum split-KV partitions for SWA-2048 block attention.
pub const ATTN_SPLITS: usize = 16;
/// Maximum split-KV partitions for global E4M3 block attention (37 CTAs x 2 warps = 296 CTAs = 4 CTAs/SM on 74 SMs).
pub const GLOBAL_ATTN_SPLITS: usize = 74;
/// Dynamic shared memory bytes for `df2::swa_attn_split_kernel`: `(4 * 64 + 32) * (HD + 8) * 2`.
pub const ATTN_SMEM: usize = (4 * 64 + 32) * (HD + 8) * 2;
/// Dynamic shared memory bytes for `df2::global_attn_e4m3_split_kernel` (Design A uses 8 KiB static smem).
pub const GLOBAL_ATTN_SMEM: usize = 0;
/// Checkpoint filename inside the DFlash2 model directory.
pub const DFLASH2_FILE: &str = "model.safetensors";
/// Config filename inside the DFlash2 model directory.
pub const CONFIG_FILE: &str = "config.json";

/// Attention type of a DFlash2 draft decoder layer (`config.json` `layer_types`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Sliding-window attention (`sliding_attention`, SWA-2048, BF16 ring buffer).
    SlidingAttention,
    /// Full-context global attention (`full_attention`, E4M3 paged KV cache).
    FullAttention,
}

/// Parsed DFlash2 model configuration (`config.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct DFlash2Config {
    /// Attention type for each of the 5 draft decoder layers.
    pub layer_types: [LayerType; NLAYERS],
    /// RoPE base frequency for `sliding_attention` layers.
    pub rope_theta: f32,
    /// RoPE base frequency for `full_attention` layers.
    pub global_rope_theta: f32,
}

impl Default for DFlash2Config {
    fn default() -> Self {
        Self {
            layer_types: [LayerType::SlidingAttention; NLAYERS],
            rope_theta: ROPE_THETA,
            global_rope_theta: GLOBAL_ROPE_THETA,
        }
    }
}

#[derive(Deserialize)]
struct RawRopeParams {
    rope_theta: Option<f32>,
    global_rope_theta: Option<f32>,
}

#[derive(Deserialize)]
struct RawDFlash2Config {
    layer_types: Option<Vec<String>>,
    rope_theta: Option<f32>,
    global_rope_theta: Option<f32>,
    rope_parameters: Option<RawRopeParams>,
}

impl DFlash2Config {
    /// Loads `config.json` from `dflash_dir`. If `config.json` does not exist, returns the stock
    /// 5x `sliding_attention` default.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::Invalid`] on malformed JSON or unsupported `layer_types`.
    pub fn load(dflash_dir: &Path) -> Result<Self> {
        let path = dflash_dir.join(CONFIG_FILE);
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(&path).map_err(|source| ModelError::Io { path, source })?;
        Self::from_json(&text)
    }

    /// Parses a DFlash2 `config.json` payload.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::Invalid`] on malformed JSON, invalid `layer_types`, or non-positive RoPE theta.
    pub fn from_json(text: &str) -> Result<Self> {
        let raw: RawDFlash2Config =
            serde_json::from_str(text).map_err(|e| invalid("dflash2 config.json", e.to_string()))?;
        let mut layer_types = [LayerType::SlidingAttention; NLAYERS];
        if let Some(types) = raw.layer_types {
            if types.len() != NLAYERS {
                return Err(invalid(
                    "dflash2 config.json",
                    format!("layer_types has {} entries, expected {NLAYERS}", types.len()),
                ));
            }
            for (i, t) in types.iter().enumerate() {
                layer_types[i] = match t.as_str() {
                    "sliding_attention" => LayerType::SlidingAttention,
                    "full_attention" => LayerType::FullAttention,
                    other => {
                        return Err(invalid(
                            "dflash2 config.json",
                            format!("layer_types[{i}] unsupported: {other:?}"),
                        ));
                    }
                };
            }
        }
        let rope_theta = raw
            .rope_parameters
            .as_ref()
            .and_then(|p| p.rope_theta)
            .or(raw.rope_theta)
            .unwrap_or(ROPE_THETA);
        let global_rope_theta = raw
            .global_rope_theta
            .or_else(|| raw.rope_parameters.as_ref().and_then(|p| p.global_rope_theta))
            .unwrap_or(GLOBAL_ROPE_THETA);
        if !(rope_theta > 0.0 && rope_theta.is_finite()) {
            return Err(invalid(
                "dflash2 config.json",
                format!("invalid rope_theta {rope_theta}"),
            ));
        }
        if !(global_rope_theta > 0.0 && global_rope_theta.is_finite()) {
            return Err(invalid(
                "dflash2 config.json",
                format!("invalid global_rope_theta {global_rope_theta}"),
            ));
        }
        Ok(Self {
            layer_types,
            rope_theta,
            global_rope_theta,
        })
    }

    /// Returns `true` if layer `li` uses `full_attention`.
    #[must_use]
    pub fn is_global(&self, li: usize) -> bool {
        self.layer_types.get(li) == Some(&LayerType::FullAttention)
    }

    /// Returns `true` if any layer uses `full_attention`.
    #[must_use]
    pub fn has_global(&self) -> bool {
        self.layer_types.contains(&LayerType::FullAttention)
    }

    /// Number of `full_attention` layers.
    #[must_use]
    pub fn global_count(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == LayerType::FullAttention)
            .count()
    }

    /// Number of `sliding_attention` layers.
    #[must_use]
    pub fn swa_count(&self) -> usize {
        NLAYERS - self.global_count()
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_stock_five_swa() {
        let j = r#"{
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
        }"#;
        let cfg = DFlash2Config::from_json(j).unwrap();
        assert!(!cfg.has_global());
        assert_eq!(cfg.global_count(), 0);
        assert_eq!(cfg.swa_count(), 5);
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert_eq!(cfg.global_rope_theta, 1_000_000.0);
    }

    #[test]
    fn config_hybrid_layer2_global() {
        let j = r#"{
            "layer_types": ["sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention"],
            "rope_parameters": {"rope_theta": 10000.0},
            "global_rope_theta": 1000000.0
        }"#;
        let cfg = DFlash2Config::from_json(j).unwrap();
        assert!(cfg.has_global());
        assert_eq!(cfg.global_count(), 1);
        assert_eq!(cfg.swa_count(), 4);
        assert!(!cfg.is_global(0));
        assert!(!cfg.is_global(1));
        assert!(cfg.is_global(2));
        assert!(!cfg.is_global(3));
        assert!(!cfg.is_global(4));
        assert_eq!(cfg.global_rope_theta, 1_000_000.0);
    }

    #[test]
    fn config_two_global_layers() {
        let j = r#"{
            "layer_types": ["sliding_attention","full_attention","sliding_attention","full_attention","sliding_attention"],
            "global_rope_theta": 500000.0
        }"#;
        let cfg = DFlash2Config::from_json(j).unwrap();
        assert_eq!(cfg.global_count(), 2);
        assert!(cfg.is_global(1) && cfg.is_global(3));
        assert_eq!(cfg.global_rope_theta, 500_000.0);
    }

    #[test]
    fn config_missing_fields_defaults_to_swa() {
        let cfg = DFlash2Config::from_json("{}").unwrap();
        assert_eq!(cfg, DFlash2Config::default());
    }

    #[test]
    fn config_rejects_invalid_layer_type() {
        let j = r#"{"layer_types": ["sliding_attention","linear_attention","sliding_attention","sliding_attention","sliding_attention"]}"#;
        assert!(DFlash2Config::from_json(j).is_err());
    }
}
