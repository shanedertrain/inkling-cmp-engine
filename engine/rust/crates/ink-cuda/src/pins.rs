//! The cuBLASLt algorithm pin file `gemm-algos.json` (docs/RUST-PLAN.md 3.2 "Dense GEMMs").
//!
//! One algorithm per GEMM shape, chosen once (by `ink-bench`'s tuning mode, or [`crate::LtGemm::select`]) and
//! read at start-up, so two process starts run the same kernels and produce the same bits. Format:
//!
//! ```json
//! {"format": "ink-gemm-algos/1", "cublaslt_version": 130600,
//!  "entries": [{"n": 2304, "k": 2304, "ldw": 2304, "in": "bf16", "out": "bf16", "class": "decode",
//!               "algo": {"ID": 21, "TILE_ID": 18, "SPLITK_NUM": 1, "REDUCTION_SCHEME": 0, "CTA_SWIZZLING": 0,
//!                        "CUSTOM_OPTION": 0, "STAGES_ID": 22, "INNER_SHAPE_ID": 0, "CLUSTER_SHAPE_ID": 0}}]}
//! ```
//!
//! The attribute names are cuBLASLt's `CUBLASLT_ALGO_CONFIG_*`. The plan lists eight; `INNER_SHAPE_ID` is pinned
//! too because cuBLASLt 13 exposes it and `cublasLtMatmulAlgoInit` would otherwise leave it at its default.
//!
//! `SPLITK_NUM` > 1 is accepted only with an out-of-place reduction (`REDUCTION_SCHEME` 2 COMPUTE_TYPE or 4
//! OUTPUT_TYPE): the partials are summed in a fixed order, so such a plan stays batch-invariant (G1, the decode
//! gate-down shape); INPLACE (1, atomics) and NONE are refused.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{CudaError, Result};

/// The value of the file's `format` field.
pub const FORMAT: &str = "ink-gemm-algos/1";

/// Element type of GEMM operands or output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    /// bfloat16.
    Bf16,
    /// IEEE fp32 (no TF32).
    F32,
}

/// Plan class of a call (the C++ `LtGemm::klass` @ a3209a0): decode-only calls and calls with a prefill chunk may
/// pin different algorithms for the same shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanClass {
    /// Every segment of the call is a decode row group.
    Decode,
    /// The call contains a prefill chunk.
    Mixed,
}

/// A GEMM shape `out[M, N] = x[M, K] @ W[N, K]^T` without its row count M: the key of a pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GemmShape {
    /// Output columns (rows of W).
    pub n: u32,
    /// Reduction length.
    pub k: u32,
    /// Leading dimension of W (>= k).
    pub ldw: u32,
    /// Operand type of x and W.
    pub input: Dtype,
    /// Output type.
    pub output: Dtype,
    /// Plan class.
    pub class: PlanClass,
}

impl fmt::Display for GemmShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "N={} K={} ldw={} {:?}->{:?} {:?}",
            self.n, self.k, self.ldw, self.input, self.output, self.class
        )
    }
}

/// The configuration attributes of one cuBLASLt algorithm (`CUBLASLT_ALGO_CONFIG_*`, cuBLASLt's own types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[expect(
    missing_docs,
    reason = "fields are cuBLASLt's CUBLASLT_ALGO_CONFIG_* attributes of the same name"
)]
pub struct AlgoPin {
    #[serde(rename = "ID")]
    pub id: i32,
    #[serde(rename = "TILE_ID")]
    pub tile: u32,
    #[serde(rename = "SPLITK_NUM")]
    pub splitk: i32,
    #[serde(rename = "REDUCTION_SCHEME")]
    pub reduction: u32,
    #[serde(rename = "CTA_SWIZZLING")]
    pub swizzle: u32,
    #[serde(rename = "CUSTOM_OPTION")]
    pub custom: u32,
    #[serde(rename = "STAGES_ID")]
    pub stages: u32,
    #[serde(rename = "INNER_SHAPE_ID")]
    pub inner_shape: u16,
    #[serde(rename = "CLUSTER_SHAPE_ID")]
    pub cluster_shape: u16,
}

/// `CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE`: split-K partials in the compute type (fp32), reduced out of place.
pub const REDUCTION_COMPUTE_TYPE: u32 = 2;
/// `CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE`: split-K partials in the output type, reduced out of place.
pub const REDUCTION_OUTPUT_TYPE: u32 = 4;

impl AlgoPin {
    /// Split-K 1, or split-K with an out-of-place, fixed-order reduction (the C++ `gemm_pins::deterministic`).
    pub fn deterministic(&self) -> bool {
        self.splitk <= 1 || self.reduction == REDUCTION_COMPUTE_TYPE || self.reduction == REDUCTION_OUTPUT_TYPE
    }
}

/// One line of the file (the shape's fields inline; serde cannot combine `flatten` with `deny_unknown_fields`).
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    n: u32,
    k: u32,
    ldw: u32,
    #[serde(rename = "in")]
    input: Dtype,
    #[serde(rename = "out")]
    output: Dtype,
    class: PlanClass,
    algo: AlgoPin,
}

impl Entry {
    fn new(s: GemmShape, algo: AlgoPin) -> Entry {
        Entry {
            n: s.n,
            k: s.k,
            ldw: s.ldw,
            input: s.input,
            output: s.output,
            class: s.class,
            algo,
        }
    }

    fn shape(&self) -> GemmShape {
        GemmShape {
            n: self.n,
            k: self.k,
            ldw: self.ldw,
            input: self.input,
            output: self.output,
            class: self.class,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    format: String,
    cublaslt_version: usize,
    entries: Vec<Entry>,
}

/// The pinned algorithm of every GEMM shape, for one cuBLASLt version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgoPins {
    cublaslt_version: usize,
    pins: BTreeMap<GemmShape, AlgoPin>,
}

impl AlgoPins {
    /// No pins yet, for `cublaslt_version` (`cublasLtGetVersion`).
    pub fn new(cublaslt_version: usize) -> AlgoPins {
        AlgoPins {
            cublaslt_version,
            pins: BTreeMap::new(),
        }
    }

    /// Parses a pin file.
    ///
    /// # Errors
    ///
    /// [`CudaError::Pins`] for malformed JSON, unknown fields, another format, a duplicate shape, split-K above 1
    /// (batch invariance needs one k-loop per output element), or `ldw < k`.
    pub fn from_json(text: &str) -> Result<AlgoPins> {
        let file: File = serde_json::from_str(text).map_err(|e| CudaError::Pins(e.to_string()))?;
        if file.format != FORMAT {
            return Err(CudaError::Pins(format!(
                "format {:?}, expected {FORMAT:?}",
                file.format
            )));
        }
        let mut pins = AlgoPins::new(file.cublaslt_version);
        for e in file.entries {
            pins.insert(e.shape(), e.algo)?;
        }
        Ok(pins)
    }

    /// The file text: entries sorted by shape, one per line.
    pub fn to_json(&self) -> String {
        let entries: Vec<Entry> = self.pins.iter().map(|(s, a)| Entry::new(*s, *a)).collect();
        let lines: Vec<String> = entries.iter().map(|e| format!("  {}", serde_json::json!(e))).collect();
        format!(
            "{{\"format\": \"{FORMAT}\", \"cublaslt_version\": {}, \"entries\": [\n{}\n]}}\n",
            self.cublaslt_version,
            lines.join(",\n")
        )
    }

    /// Adds the pin of `shape`.
    ///
    /// # Errors
    ///
    /// [`CudaError::Pins`] for a duplicate shape, split-K above 1 without an out-of-place reduction, or `ldw < k`.
    pub fn insert(&mut self, shape: GemmShape, algo: AlgoPin) -> Result<()> {
        if !algo.deterministic() {
            return Err(CudaError::Pins(format!(
                "{shape}: SPLITK_NUM {} with REDUCTION_SCHEME {} (split-K needs {REDUCTION_COMPUTE_TYPE} COMPUTE_TYPE or \
                 {REDUCTION_OUTPUT_TYPE} OUTPUT_TYPE)",
                algo.splitk, algo.reduction
            )));
        }
        if shape.ldw < shape.k || shape.n == 0 || shape.k == 0 {
            return Err(CudaError::Pins(format!("{shape}: bad dimensions")));
        }
        if self.pins.insert(shape, algo).is_some() {
            return Err(CudaError::Pins(format!("{shape}: pinned twice")));
        }
        Ok(())
    }

    /// The pin of `shape`.
    pub fn get(&self, shape: &GemmShape) -> Option<&AlgoPin> {
        self.pins.get(shape)
    }

    /// The cuBLASLt version the pins were chosen with.
    pub fn cublaslt_version(&self) -> usize {
        self.cublaslt_version
    }

    /// Number of pinned shapes.
    pub fn len(&self) -> usize {
        self.pins.len()
    }

    /// Whether no shape is pinned.
    pub fn is_empty(&self) -> bool {
        self.pins.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(n: u32) -> GemmShape {
        GemmShape {
            n,
            k: 2304,
            ldw: 2304,
            input: Dtype::Bf16,
            output: Dtype::Bf16,
            class: PlanClass::Decode,
        }
    }

    fn algo(id: i32) -> AlgoPin {
        AlgoPin {
            id,
            tile: 18,
            splitk: 1,
            reduction: 0,
            swizzle: 0,
            custom: 0,
            stages: 22,
            inner_shape: 0,
            cluster_shape: 0,
        }
    }

    #[test]
    fn round_trip() {
        let mut p = AlgoPins::new(130600);
        p.insert(shape(4608), algo(21)).unwrap();
        p.insert(shape(2304), algo(6)).unwrap();
        let mut f32out = shape(201024);
        f32out.output = Dtype::F32;
        f32out.class = PlanClass::Mixed;
        p.insert(f32out, algo(31)).unwrap();
        let text = p.to_json();
        let q = AlgoPins::from_json(&text).unwrap();
        assert_eq!(p, q);
        assert_eq!(q.get(&shape(2304)).unwrap().id, 6);
        assert_eq!(q.to_json(), text, "serialisation is stable");
        assert!(
            text.contains("\"SPLITK_NUM\":1")
                && text.contains("\"in\":\"bf16\"")
                && text.contains("\"class\":\"mixed\"")
        );
    }

    #[test]
    fn reads_the_documented_example() {
        let text = r#"{"format": "ink-gemm-algos/1", "cublaslt_version": 130600,
          "entries": [{"n": 2304, "k": 2304, "ldw": 2304, "in": "bf16", "out": "bf16", "class": "decode",
            "algo": {"ID": 21, "TILE_ID": 18, "SPLITK_NUM": 1, "REDUCTION_SCHEME": 0, "CTA_SWIZZLING": 0,
                     "CUSTOM_OPTION": 0, "STAGES_ID": 22, "INNER_SHAPE_ID": 0, "CLUSTER_SHAPE_ID": 0}}]}"#;
        let p = AlgoPins::from_json(text).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p.cublaslt_version(), 130600);
        assert_eq!(*p.get(&shape(2304)).unwrap(), algo(21));
    }

    #[test]
    fn accepts_split_k_with_an_out_of_place_reduction() {
        for reduction in [REDUCTION_COMPUTE_TYPE, REDUCTION_OUTPUT_TYPE] {
            let g1 = AlgoPin {
                splitk: 32,
                reduction,
                ..algo(31)
            };
            let mut p = AlgoPins::new(130600);
            p.insert(shape(64), g1).unwrap();
            let q = AlgoPins::from_json(&p.to_json()).unwrap();
            assert_eq!(*q.get(&shape(64)).unwrap(), g1, "reduction {reduction} round-trips");
        }
    }

    #[test]
    fn refuses_bad_files() {
        let good = {
            let mut p = AlgoPins::new(130600);
            p.insert(shape(2304), algo(21)).unwrap();
            p.to_json()
        };
        for (bad, why) in [
            (good.replace("ink-gemm-algos/1", "ink-gemm-algos/2"), "format"),
            (good.replace("\"SPLITK_NUM\":1", "\"SPLITK_NUM\":2"), "split-K"),
            (good.replace("\"ldw\":2304", "\"ldw\":100"), "ldw < k"),
            (
                good.replace("\"class\":\"decode\"", "\"class\":\"decode\",\"m\":8"),
                "unknown field",
            ),
            (good.replace("\"STAGES_ID\":22,", ""), "missing attribute"),
            (good.replace("\"in\":\"bf16\"", "\"in\":\"f16\""), "unknown dtype"),
            ("{".to_owned(), "malformed"),
        ] {
            assert!(
                matches!(AlgoPins::from_json(&bad), Err(CudaError::Pins(_))),
                "{why} must be refused"
            );
        }
        let inplace = good
            .replace("\"SPLITK_NUM\":1", "\"SPLITK_NUM\":32")
            .replace("\"REDUCTION_SCHEME\":0", "\"REDUCTION_SCHEME\":1");
        assert!(
            matches!(AlgoPins::from_json(&inplace), Err(CudaError::Pins(_))),
            "split-K with the in-place (atomic) reduction must be refused"
        );
        let mut dup = AlgoPins::new(1);
        dup.insert(shape(2304), algo(1)).unwrap();
        assert!(dup.insert(shape(2304), algo(2)).is_err());
    }
}
