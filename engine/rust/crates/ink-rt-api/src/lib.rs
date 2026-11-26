//! The [`Runtime`], [`SpecRuntime`] and [`Drafter`] traits and their types for the Rust CUDA runtime.
#![deny(unsafe_code)]

use std::path::{Path, PathBuf};

use ink_record::Source;

/// Logit row width: the padded vocabulary.
pub const LOGIT_ROW: usize = 201_024;

/// Valid logits in a row; the rest is padding.
pub const VOCAB: usize = 200_058;

/// Hidden size of the model: one DSpark feature row is bf16 `[4][HIDDEN]`.
pub const HIDDEN: usize = 2304;

/// Bytes of the DSpark features of one row (bf16 `[4][HIDDEN]`).
pub const FEAT_ROW_BYTES: usize = 4 * HIDDEN * 2;

/// Drafts the drafter proposes per session per round (`draft.hpp`: drafts `[n][7]`).
pub const DRAFTS: usize = 7;

/// Maximum drafts per session per round when verbatim lookup drafts are appended (7 drafter + 8 lookup).
pub const MAX_DRAFTS: usize = 15;

/// Maximum verify rows per session (anchor + [`MAX_DRAFTS`]).
pub const MAX_VERIFY_ROWS: usize = MAX_DRAFTS + 1;

/// Threshold of the probability-gated verify width on the cumulative drafter probability of the leading drafts
/// (`DraftRT::kVerifyTau`; study: `bench/raw/2026-11-23-ptok-gate/`): a session verifies 1 + the leading drafts whose
/// probabilities' product stays at or above it, rounded up to 1 / 2 / 4 / 8 rows.
pub const VERIFY_TAU: f32 = 0.7;

/// The verify call's rows per session with the gated width ([`CallRows::parse`]): `m6` extended with 16-row lookup
/// buckets (`10, 12, 14, 16`); rounds without lookup (`widths <= 8`) stay strictly within `1,2,3,4,6,8`.
pub const VERIFY_CALL_ROWS: &str = "1,2,3,4,6,8,10,12,14,16";

/// Default [`EngineConfig::graph_mem`]: 1,024 MiB (covers 1..=16 verify call-row buckets across 1..=8 slots).
pub const DEFAULT_GRAPH_MEM: u64 = 1024 << 20;

/// Engine start-up configuration. Defaults are the production configuration of `ink::EngineConfig`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    /// Model directory.
    pub model_dir: PathBuf,
    /// Pinned cuBLASLt algorithms: loaded when the file exists, written after init otherwise.
    pub gemm_algos: PathBuf,
    /// Sessions with state on the card.
    pub max_slots: u32,
    /// Global KV pages of 1024 tokens.
    pub num_pages: u32,
    /// Rows per forward call.
    pub t_max: u32,
    /// Rows per prefill segment (sets the SWA ring); at most `t_max`.
    pub max_chunk: u32,
    /// Rows whose logits one call can return.
    pub logit_rows_max: u32,
    /// Threads that load weights.
    pub load_threads: u32,
    /// M at which each dense GEMM shape's algorithm is timed at start-up; 0 takes the heuristic's first choice.
    pub gemm_tune_m: u32,
    /// Capture decode-only calls as CUDA graphs.
    pub graphs: Graphs,
    /// Device bytes the captured graphs may hold (the forward graphs and the drafter's): a reserve the engine refuses
    /// to start without; a graph key beyond it runs eagerly. The C ABI runtime ignores it.
    pub graph_mem: u64,
    /// Store global-layer K/V cache in FP8 E4M3 (lossy; halves global KV VRAM and bandwidth; SWA rings stay bf16).
    pub kv_fp8: bool,
}

/// Whether decode-only calls run as captured CUDA graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Graphs {
    /// Capture and replay.
    On,
    /// Launch eagerly.
    Off,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("/model"),
            gemm_algos: PathBuf::from("gemm-algos.json"),
            max_slots: 8,
            num_pages: 256,
            t_max: 2048 + 64,
            max_chunk: 2048,
            logit_rows_max: 128,
            load_threads: 16,
            gemm_tune_m: 32,
            graphs: Graphs::On,
            graph_mem: DEFAULT_GRAPH_MEM,
            kv_fp8: false,
        }
    }
}

/// One sequence's rows in a forward call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegIn<'a> {
    /// Slot of the sequence.
    pub slot: usize,
    /// Token ids at positions `length(slot)` onward.
    pub ids: &'a [i32],
    /// Logits of the last `logits` rows of the segment; 0 returns none.
    pub logits: usize,
}

/// Sampling controls of one segment for the device sampler (`ink_rt_sampling`): below a temperature of 1e-5 the
/// argmax; otherwise the softmax of `logits / temperature` cut by `min_p` (0: off), `top_k` (0: off) and `top_p`
/// (1: off), one token drawn with Philox4x32-10 keyed by `seed` at the token's position
/// (`engine/native/src/sample_kernels.cuh`). The token depends only on the row's logits, these controls and the position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceSampling {
    /// Philox key.
    pub seed: u64,
    /// Softmax temperature.
    pub temperature: f32,
    /// Nucleus mass.
    pub top_p: f32,
    /// Minimum probability relative to the most likely token.
    pub min_p: f32,
    /// Candidates kept.
    pub top_k: u32,
}

/// One sequence's rows in a forward call whose token ids are on the device (speculative verify rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevSeg {
    /// Slot of the sequence.
    pub slot: usize,
    /// `rows` token ids (i32) at positions `length(slot)` onward.
    pub ids: DevBuf,
    /// Rows of the segment.
    pub rows: usize,
    /// Logits of the last `logits` rows of the segment; 0 returns none.
    pub logits: usize,
}

/// Element type of a tapped tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapType {
    /// bfloat16.
    Bf16,
    /// float32.
    F32,
    /// int32.
    I32,
}

/// Called after each tapped op with the op's name, element type, shape, a host copy of the tensor and a scratch
/// buffer of the same size. Returns `n`: the first `n` bytes of the scratch buffer replace the start of the engine's
/// tensor before the op's consumers run (0 leaves it alone; `n` beyond the buffer fails the forward call).
pub type TapFn = Box<dyn FnMut(&str, TapType, &[i64], &[u8], &mut [u8]) -> usize + Send>;

/// Engine counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeStats {
    /// Bytes of the arena in use.
    pub arena_used: u64,
    /// Bytes of weights on the card.
    pub weight_bytes: u64,
    /// CUDA graphs captured so far.
    pub graph_captures: u64,
    /// Captured graphs held now.
    pub graphs_live: u64,
    /// Device bytes the held graphs take (measured at capture; at most [`EngineConfig::graph_mem`]).
    pub graph_bytes: u64,
    /// Graph keys that ran eagerly because [`EngineConfig::graph_mem`] was full.
    pub graphs_refused: u64,
    /// Dense GEMMs that fell back from the pinned cuBLASLt algorithm.
    pub gemm_fallbacks: u64,
}

/// Glue path of the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glue {
    /// Per-op reference glue (always used while a tap is set); the C ABI's value 1.
    Reference,
    /// Production glue (fused AttnRes and norm, fused message-gate tail); the C ABI's value 6.
    Production,
}

/// What [`Runtime::dump`] writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dump {
    /// One JSON line per node of the last captured graph.
    LaunchTrace,
    /// FNV-1a 64 of every weight region as uploaded.
    WeightSums,
    /// Name, offset and bytes of every arena allocation.
    ArenaMap,
}

/// What state a failed runtime call leaves the runtime in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The call was refused or failed on the host (a limit, a bad argument, a call out of order): the runtime is
    /// usable and later calls can succeed. What a refused call has already done is stated on its method; at most it
    /// wrote rows that are not committed.
    Request,
    /// The device failed (a CUDA driver or cuBLASLt error status, a failed graph capture): the context may be dead
    /// (sticky errors), so no later call can be trusted. The runtime must be dropped and the process restarted, as
    /// the C++ runtime's `ink::fatal` exits the process.
    Device,
}

/// Failure of a runtime call: its [`ErrorKind`] and a message naming the cause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeError {
    kind: ErrorKind,
    message: String,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            ErrorKind::Request => write!(f, "runtime error: {}", self.message),
            ErrorKind::Device => write!(f, "runtime device error: {}", self.message),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl RuntimeError {
    /// A refused or host-side failed call ([`ErrorKind::Request`]).
    #[must_use]
    pub fn request(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Request,
            message: message.into(),
        }
    }

    /// A device failure ([`ErrorKind::Device`]).
    #[must_use]
    pub fn device(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Device,
            message: message.into(),
        }
    }

    /// The same error as a device failure (an error inside a graph capture leaves the stream's capture state
    /// unknown, whatever its cause).
    #[must_use]
    pub fn into_device(self) -> Self {
        Self {
            kind: ErrorKind::Device,
            ..self
        }
    }

    /// Whether the runtime is still usable.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The cause, without the kind.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// A range of device memory handed out by a runtime ([`SpecRuntime::alloc`], [`Drafter::drafts`]).
///
/// Only an address and a length: Rust never dereferences it. The runtime checks every range it is given against the
/// memory it handed out, so a range made up by hand is refused, not used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevBuf {
    addr: u64,
    bytes: usize,
}

impl DevBuf {
    /// A range of `bytes` bytes at device address `addr`.
    #[must_use]
    pub const fn new(addr: u64, bytes: usize) -> Self {
        Self { addr, bytes }
    }

    /// Device address of the first byte.
    #[must_use]
    pub const fn addr(self) -> u64 {
        self.addr
    }

    /// Length in bytes.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self.bytes
    }

    /// The `len` bytes at `offset`, or `None` when they do not lie inside this range.
    #[must_use]
    pub fn range(self, offset: usize, len: usize) -> Option<Self> {
        let end = offset.checked_add(len)?;
        (end <= self.bytes).then(|| Self {
            addr: self.addr + offset as u64,
            bytes: len,
        })
    }

    /// The `len` i32 elements starting at element `start`, or `None` when they do not lie inside this range.
    #[must_use]
    pub fn i32s(self, start: usize, len: usize) -> Option<Self> {
        self.range(start.checked_mul(4)?, len.checked_mul(4)?)
    }
}

/// Device buffers of one sparse drafter proposal distribution (`[sessions][7][16]` token ids and exact integer weights
/// summing to `2^22` per slot, plus the drafter's 7-strided `drafts`) for exact Leviathan rejection + residual sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProposalBufs {
    /// Drafter's own proposed tokens (`i32 [sessions][7]`), independent of any appended lookup drafts.
    pub drafts: DevBuf,
    /// Proposed candidate token ids (`i32 [sessions][7][16]`).
    pub ids: DevBuf,
    /// Exact integer proposal weights (`u32 [sessions][7][16]`, summing to `1 << 22`).
    pub weights: DevBuf,
}

/// Device buffers of one [`SpecRuntime::spec_accept`] call (sizes in `ink_rt.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptBufs {
    /// Drafts `[slots][7]` (i32), usually [`Drafter::drafts`].
    pub drafts: DevBuf,
    /// Last emitted token per slot (i32), updated.
    pub anchor: DevBuf,
    /// Position of the anchor per slot (i32), updated.
    pub pos: DevBuf,
    /// Accept results `[9 S + 1]` (i32).
    pub res: DevBuf,
    /// Packed commit rows for the drafter: call row, position, session `[8 S]` each (i32).
    pub crow: DevBuf,
    /// See `crow`.
    pub cpos: DevBuf,
    /// See `crow`.
    pub csess: DevBuf,
    /// DSpark features of each slot's last committed row `[slots]` rows.
    pub lastfeat: DevBuf,
    /// DSpark features of the packed commit rows `[8 S]` rows.
    pub cfeats: DevBuf,
}

/// The drafter's construction parameters (`ink::DraftRT`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftConfig {
    /// Drafter model directory.
    pub model_dir: PathBuf,
    /// Folded drafter weights (DSpark).
    pub fold_bin: PathBuf,
    /// Optional DFlash2 checkpoint directory (`model.safetensors`); when set, creates the DFlash2 drafter instead of
    /// DSpark (`--runtime rust` only).
    pub dflash2_dir: Option<PathBuf>,
    /// Sessions; drafter rows are engine slots `0..sessions`.
    pub sessions: u32,
    /// Positions per session.
    pub max_ctx: u32,
    /// Head precision.
    pub head: HeadPrecision,
    /// Global-layer K/V storage.
    pub global_kv: KvPrecision,
    /// Verify width: the drafter's probability of each draft computed for the gated width, or none (8 rows).
    pub verify: VerifyWidth,
    /// Verbatim n-gram lookup drafts appended after the drafter's 7 drafts (`0` = off, `8` = up to 8 lookup tokens).
    pub lookup: u32,
    /// Minimum backward verbatim match length `m` required to trigger lookup drafts (default `8`).
    pub lookup_min: u32,
}

/// How many rows a speculating session verifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyWidth {
    /// 1 + the leading drafts whose cumulative drafter probability stays at or above [`VERIFY_TAU`], rounded up to
    /// 1 / 2 / 4 / 8 rows: every draft computes [`Drafter::conf`] (`draft_conf.cuh`, +0.1-0.45 ms per round). The
    /// verify call's rows are then rounded up as [`CallRows`] says.
    Gated(CallRows),
    /// Always 8 rows (the anchor and 7 drafts); no probability pass.
    Full,
}

/// Rows of a verify call with the gated width: its `S` sessions' gated widths sum to `T`, rounded up to `S m` for
/// the smallest allowed `m` (a set from 1..=16 that holds 8), the extra rows widening the widest sessions below 8
/// rows first (then below 16 if lookup rows pushed `T` past `8 S`) ([`CallRows::fill`]). The calls' shapes (the graph
/// keys, `meta::cache_key`) are then `S` x the set; [`CallRows::RAGGED`] leaves `T` as it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallRows(u16);

impl CallRows {
    /// No rounding: each session verifies its gated width.
    pub const RAGGED: CallRows = CallRows(0);

    /// The set `ms` (each in 1..=16, 8 among them).
    ///
    /// # Errors
    ///
    /// A value outside 1..=16, or no 8.
    pub fn new(ms: &[usize]) -> Result<CallRows, RuntimeError> {
        let mut bits = 0u16;
        for &m in ms {
            if !(1..=MAX_VERIFY_ROWS).contains(&m) {
                return Err(RuntimeError::request(format!(
                    "verify rows per session {m} outside 1..={MAX_VERIFY_ROWS}"
                )));
            }
            bits |= 1 << (m - 1);
        }
        if bits & (1 << 7) == 0 {
            return Err(RuntimeError::request(
                "the verify rows per session must include 8".to_owned(),
            ));
        }
        Ok(CallRows(bits))
    }

    /// `ragged` or a comma list such as `1,2,3,4,6,8,10,12,14,16`.
    ///
    /// # Errors
    ///
    /// As [`CallRows::new`]; a value that is not a number.
    pub fn parse(s: &str) -> Result<CallRows, RuntimeError> {
        if s == "ragged" {
            return Ok(CallRows::RAGGED);
        }
        let ms = s
            .split(',')
            .map(|x| x.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| RuntimeError::request(format!("verify rows per session {s:?}: {e}")))?;
        CallRows::new(&ms)
    }

    /// The text [`CallRows::parse`] reads.
    pub fn text(self) -> String {
        if self == CallRows::RAGGED {
            return "ragged".to_owned();
        }
        let ms: Vec<String> = (1..=MAX_VERIFY_ROWS)
            .filter(|m| self.0 & (1 << (m - 1)) != 0)
            .map(|m| m.to_string())
            .collect();
        ms.join(",")
    }

    /// Widens `widths` (the gated widths of a call's sessions, each in 1..=16, ascending) to the call's rows: `S m`
    /// for the smallest allowed `m` with `S m >= T`, one row at a time to the widest session below 8 rows (the last
    /// such in call order), falling back to the widest session below 16 if `T > 8 S` requires it. Ascending stays
    /// ascending.
    pub fn fill(self, widths: &mut [usize]) {
        if self == CallRows::RAGGED {
            return;
        }
        let s = widths.len();
        let t: usize = widths.iter().sum();
        let cap = if widths.iter().any(|&w| w > 8) {
            MAX_VERIFY_ROWS
        } else {
            8
        };
        let m = (1..=cap)
            .find(|m| self.0 & (1 << (m - 1)) != 0 && m * s >= t)
            .unwrap_or(cap);
        for _ in t..m * s {
            if let Some(i) = widths.iter().rposition(|&w| w < 8) {
                widths[i] += 1;
            } else if let Some(i) = widths.iter().rposition(|&w| w < cap) {
                widths[i] += 1;
            } else {
                break;
            }
        }
    }
}

/// Precision of the drafter's head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadPrecision {
    /// bfloat16 head.
    Bf16,
    /// int8 head.
    Int8,
}

/// Storage of the drafter's global-layer K/V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    /// bfloat16.
    Bf16,
    /// FP8 E4M3.
    E4m3,
}

/// Device-side speculation of a runtime: device buffers, verify ids, greedy accept, DSpark features, the drafter.
///
/// # Bounds contract
///
/// Every [`DevBuf`] argument is checked against the memory the runtime handed out (a range it did not hand out, or
/// one too short for the call, is refused). What the runtime cannot check is the *contents* of a device buffer that
/// it uses as indices into device memory: token ids, call rows, drafter positions and sessions. The methods that take
/// such contents are `unsafe fn`s, and their `# Safety` sections state the bounds the caller keeps (on the host
/// where the caller computed the values, or by the contract of the kernel that wrote them: the accept results and
/// the drafts are in bounds by construction). Values that only flow into outputs (anchors and positions copied by
/// [`SpecRuntime::build_verify_ids`] and [`SpecRuntime::spec_accept`]) carry no contract there; they meet the one of
/// the method that consumes them.
pub trait SpecRuntime {
    /// Allocates `bytes` bytes of zeroed device memory, owned by the runtime until it is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `bytes` is 0 or the allocation fails.
    fn alloc(&mut self, bytes: usize) -> Result<DevBuf, RuntimeError>;
    /// Copies `src` into the start of `dst`, ordered after the work queued before.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `src` does not fit `dst` or the copy fails.
    fn upload(&mut self, dst: DevBuf, src: &[i32]) -> Result<(), RuntimeError>;
    /// Copies the start of `src` into `dst` after the work queued before has finished.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `dst` does not fit in `src` or the copy fails.
    fn download(&mut self, src: DevBuf, dst: &mut [i32]) -> Result<(), RuntimeError>;
    /// One forward call over segments with device token ids; returns the number of logit rows.
    ///
    /// # Safety
    ///
    /// Every token id in `segs[i].ids[0..rows]` is in `0..VOCAB` when the call's device work runs (the embedding
    /// gather reads row `id` of the embedding with no bound). A [`SpecRuntime::build_verify_ids`] output meets it
    /// when its anchors do (the drafts are in range by the drafter's contract).
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the call exceeds the engine's limits or a buffer is not the runtime's.
    #[allow(unsafe_code)] // the declaration of a method with a caller contract (bounds contract above)
    unsafe fn forward_dev(&mut self, segs: &[DevSeg]) -> Result<usize, RuntimeError>;
    /// `vids[q] = [anchor[q], drafts[q][0..k]]` for `q < n` (stride `k` in `drafts`, `k + 1` in `vids`), on the device.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a buffer is too small or not the runtime's.
    fn build_verify_ids(
        &mut self,
        anchor: DevBuf,
        drafts: DevBuf,
        vids: DevBuf,
        n: usize,
        k: usize,
    ) -> Result<(), RuntimeError>;
    /// Greedy acceptance of the last forward call, whose segments are verify segments of 1..`k + 1` rows with logits of
    /// every row (a segment of m rows verifies its first m - 1 drafts); fills `res` (`9 S + 1` values, `ink_rt.h`)
    /// after the device work.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the last call is not a verify call or a buffer is too small.
    fn spec_accept(&mut self, bufs: &AcceptBufs, k: usize, res: &mut Vec<i32>) -> Result<(), RuntimeError>;
    /// Whether [`SpecRuntime::spec_accept_sampled`] is available.
    fn accepts_sampled(&self) -> bool {
        false
    }
    /// As [`SpecRuntime::spec_accept`], with each verify row's token sampled on the device under `params[s]` of its
    /// segment `s` (one per segment) instead of its argmax: a draft is accepted when it equals its row's token. The
    /// emitted tokens are those one-row sampling with the same controls emits at the same positions.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] as `spec_accept`, or when the runtime has no device sampler.
    fn spec_accept_sampled(
        &mut self,
        bufs: &AcceptBufs,
        k: usize,
        params: &[DeviceSampling],
        res: &mut Vec<i32>,
    ) -> Result<(), RuntimeError> {
        let _ = (bufs, k, params, res);
        Err(RuntimeError::request("this runtime has no device sampler".to_owned()))
    }
    /// Exact Leviathan rejection + residual sampling over a sparse drafter proposal distribution (`prop`), followed by
    /// the standard accept kernel. Preserves the exact target filtered distribution `p` for `temperature > 0`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a buffer is too small or not the runtime's, or the runtime lacks Leviathan support.
    fn spec_accept_leviathan(
        &mut self,
        bufs: &AcceptBufs,
        prop: &ProposalBufs,
        k: usize,
        params: &[DeviceSampling],
        res: &mut Vec<i32>,
    ) -> Result<(), RuntimeError> {
        let _ = (bufs, prop, k, params, res);
        Err(RuntimeError::request(
            "this runtime has no Leviathan rejection sampler".to_owned(),
        ))
    }
    /// The probability-gated verify width of slots `0..n` (`engine.hpp` `verify_widths`): 1 + the leading drafts whose
    /// `conf` (f32 `[n][7]`, [`Drafter::conf`]) is at or above `tau` (the log of a probability), rounded up to 1 / 2 / 4
    /// / 8 rows, into `widths` (i32 `[n]` on the device) and `out` (one host sync). A runtime without the width kernel
    /// refuses, which is the default.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a buffer is too small or not the runtime's, `n` is not 1..slots, or the runtime has
    /// no width kernel.
    fn verify_widths(
        &mut self,
        conf: DevBuf,
        tau: f32,
        n: usize,
        widths: DevBuf,
        out: &mut Vec<i32>,
    ) -> Result<(), RuntimeError> {
        let _ = (conf, tau, n, widths, out);
        Err(RuntimeError::request(
            "this runtime has no gated verify width".to_owned(),
        ))
    }
    /// Coalesced verify widths + drafter drafts download in a single D2H sync: packs `[widths[0..n], drafts[0..7*n]]`
    /// into `packed` (`(1 + DRAFTS) * n` i32s on device) when supported, falling back to `verify_widths` + `download`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a buffer is too small or not the runtime's, or the device call fails.
    #[allow(clippy::too_many_arguments)]
    fn verify_widths_and_drafts(
        &mut self,
        conf: Option<DevBuf>,
        drafts: DevBuf,
        tau: f32,
        n: usize,
        packed: DevBuf,
        widths_out: &mut Vec<i32>,
        drafts_out: &mut [i32],
    ) -> Result<(), RuntimeError> {
        if let Some(c) = conf {
            let wbuf = packed
                .i32s(0, n)
                .ok_or_else(|| RuntimeError::request("packed buffer too small for widths".to_owned()))?;
            self.verify_widths(c, tau, n, wbuf, widths_out)?;
        } else {
            widths_out.clear();
            widths_out.resize(n, (DRAFTS + 1) as i32);
        }
        self.download(drafts, drafts_out)
    }
    /// DSpark features of call rows `rows[0..n]` of the last forward call into `dst` (`n` feature rows).
    ///
    /// # Safety
    ///
    /// Every value of `rows[0..n]` is a row of the last forward call (`0 <= row < ` its rows) when the device work
    /// runs: the gather reads feature row `row` with no bound.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a buffer is too small or not the runtime's.
    #[allow(unsafe_code)] // as forward_dev
    unsafe fn gather_feats(&mut self, rows: DevBuf, n: usize, dst: DevBuf) -> Result<(), RuntimeError>;
    /// Creates the drafter on this runtime's stream; it keeps the runtime's device state alive.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the runtime has a drafter already or the drafter cannot be created.
    fn create_drafter(&mut self, config: &DraftConfig) -> Result<Box<dyn Drafter>, RuntimeError>;
}

/// The forward engine: tools and the scheduler adapter talk to the model through it.
pub trait Runtime {
    /// Implementation, version and commit of this runtime, for records.
    fn identity(&self) -> Source;
    /// Empties a slot.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `slot` does not exist, when it is in the last forward call and that call is not
    /// committed yet (a commit after the reset would advance the emptied slot; commit 0 rows to drop the call), or the
    /// device call fails.
    fn reset_slot(&mut self, slot: usize) -> Result<(), RuntimeError>;
    /// Tokens cached in `slot`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `slot` does not exist.
    fn length(&self, slot: usize) -> Result<usize, RuntimeError>;
    /// KV pages of the global pool that no slot holds.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the count cannot be read.
    fn free_pages(&self) -> Result<usize, RuntimeError>;
    /// Runs one forward call over `segs` and returns the number of logit rows produced.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the call exceeds the engine's limits or the device call fails.
    fn forward(&mut self, segs: &[SegIn<'_>]) -> Result<usize, RuntimeError>;
    /// Commits `kept[i]` rows of segment `i` of the last forward call and drops the rest.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `kept` does not match the last call or the device call fails.
    fn commit(&mut self, kept: &[u32]) -> Result<(), RuntimeError>;
    /// Copies `rows` logit rows (`rows * LOGIT_ROW` floats) of the last forward call into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `rows` exceeds the last call's logit rows or the copy fails.
    fn logits_host(&self, rows: usize, out: &mut Vec<f32>) -> Result<(), RuntimeError>;
    /// The argmax of each of the first `rows` logit rows of the last forward call, computed on the device (first
    /// index of the maximum over the [`VOCAB`] valid entries; NaN never wins).
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `rows` exceeds the last call's logit rows or the device call fails.
    fn argmax_host(&self, rows: usize, out: &mut Vec<i32>) -> Result<(), RuntimeError>;
    /// Sets the logits of token `ids` to -inf in `rows` logit rows of the last forward call from row `first`, on the
    /// device, before the rows are read (argmax, accept): vLLM's min-tokens logits processor, which masks the stop
    /// token ids while fewer than `min_tokens` tokens are out.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] for rows beyond the last call's logit rows, an id outside the vocabulary, a device
    /// failure, or a runtime without device masking (the default).
    fn mask_logits(&mut self, first: usize, rows: usize, ids: &[i32]) -> Result<(), RuntimeError> {
        let _ = (first, rows, ids);
        Err(RuntimeError::request(
            "this runtime cannot mask logit rows on the device (min_tokens needs the Rust runtime)",
        ))
    }
    /// Whether [`Runtime::sample_host`] is available.
    fn samples_on_device(&self) -> bool {
        false
    }
    /// The token sampled on the device from each of the first `rows` logit rows of the last forward call; `params[i]`
    /// applies to segment `i` of that call (one entry per segment).
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the runtime has no device sampler, `params` does not match the last call or the
    /// device call fails.
    fn sample_host(&mut self, params: &[DeviceSampling], rows: usize, out: &mut Vec<i32>) -> Result<(), RuntimeError> {
        let _ = (params, rows, out);
        Err(RuntimeError::request("this runtime has no device sampler".to_owned()))
    }
    /// Switches CUDA graphs on or off.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the engine refuses the switch.
    fn set_graphs(&mut self, graphs: Graphs) -> Result<(), RuntimeError>;
    /// Selects the glue path for the following calls.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the engine refuses the switch.
    fn set_glue(&mut self, glue: Glue) -> Result<(), RuntimeError>;
    /// Installs or removes the tap sink.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the engine refuses the sink.
    fn set_tap(&mut self, tap: Option<TapFn>) -> Result<(), RuntimeError>;
    /// Current counters.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a counter cannot be read.
    fn stats(&self) -> Result<RuntimeStats, RuntimeError>;
    /// Profile split of the last call: region name and milliseconds.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the marks cannot be read.
    fn marks(&mut self) -> Result<Vec<(String, f32)>, RuntimeError>;
    /// Writes an oracle dump to `path`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when nothing was captured yet or the file cannot be written.
    fn dump(&mut self, what: Dump, path: &Path) -> Result<(), RuntimeError>;
    /// Records the launch trace of the following forward calls while `on` ([`Dump::LaunchTrace`] then writes the last
    /// one). For a runtime that runs eagerly (the Rust engine of package R5); a runtime that traces the CUDA graphs
    /// it captures (the C++ engine) refuses, which is the default.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the runtime does not trace eager calls.
    fn set_launch_trace(&mut self, on: bool) -> Result<(), RuntimeError> {
        let _ = on;
        Err(RuntimeError::request(
            "this runtime traces captured CUDA graphs, not eager calls".to_owned(),
        ))
    }
    /// Waits until the device work queued by the earlier calls has finished (the tools time steps with it).
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the device call fails or the runtime cannot wait.
    fn sync(&mut self) -> Result<(), RuntimeError> {
        Err(RuntimeError::request(
            "this runtime cannot wait for the device".to_owned(),
        ))
    }
    /// Switches the per-region profile on or off: with it on, the following forward calls record GPU events at the
    /// engine's marks (read with [`Runtime::marks`]); the C++ engine then runs them eagerly.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the runtime has no profile marks.
    fn set_profile(&mut self, on: bool) -> Result<(), RuntimeError> {
        let _ = on;
        Err(RuntimeError::request("this runtime has no profile marks".to_owned()))
    }
    /// With [`Runtime::set_profile`] on, also records the sub-component marks of the forward pass (AttnRes, gate,
    /// norm, conv, attention and expert pieces: the C++ `profile_fine`); eager calls only.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the runtime has no sub-component marks.
    fn set_profile_fine(&mut self, on: bool) -> Result<(), RuntimeError> {
        let _ = on;
        Err(RuntimeError::request(
            "this runtime has no sub-component marks".to_owned(),
        ))
    }
    /// Disables the named graph regions in captured decode steps (the benchmarks' ablation); an empty list disables
    /// none. Only a runtime that captures CUDA graphs has regions.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the runtime has no captured graphs or does not know a region.
    fn set_ablation(&mut self, regions: &[&str]) -> Result<(), RuntimeError> {
        let _ = regions;
        Err(RuntimeError::request(
            "this runtime has no captured graph regions".to_owned(),
        ))
    }
    /// Device-side speculation, when the runtime has it.
    fn spec(&mut self) -> Option<&mut dyn SpecRuntime>;
    /// Prompt-end checkpoints and slot rollback (prefix reuse across turns), when the runtime has them; the default
    /// (the C++ runtime) has none.
    fn checkpoints(&mut self) -> Option<&mut dyn SlotCheckpoints> {
        None
    }
    /// Explicit-mapping device sampler probe (`ink-spec --mode sampled | sdist | stime`), when the runtime has it; the
    /// default (the C++ runtime) has none.
    fn sampler_probe(&mut self) -> Option<&mut dyn SamplerProbe> {
        None
    }
}

/// Inputs of one [`SamplerProbe::sample_test_rows`] call: explicit test mapping over logit rows.
#[derive(Debug, Clone, Copy)]
pub struct SampleTestRows<'a> {
    /// Logit buffer (`[rows][LOGIT_ROW]` f32), or `None` to read the last forward call's logits.
    pub logits: Option<DevBuf>,
    /// Logit row index for each sampled row `r`.
    pub lrow: &'a [i32],
    /// Parameter index into `params` for each sampled row `r`.
    pub par_idx: &'a [i32],
    /// Token position passed to Philox for each sampled row `r`.
    pub pos: &'a [i32],
    /// Sampling parameter table indexed by `par_idx`.
    pub params: &'a [DeviceSampling],
}

/// Optional host downloads of [`SamplerProbe::sample_test_rows`].
#[derive(Debug, Default)]
pub struct SampleTestOut<'a> {
    /// Sampled token ids (`lrow.len()` entries).
    pub out: Option<&'a mut Vec<i32>>,
    /// Per-row kernel stats (`8 * lrow.len()` u64 entries; `lrow.len() <= 64`).
    pub stats: Option<&'a mut Vec<u64>>,
    /// Per-row candidate keys (`VOCAB * lrow.len()` u32 entries).
    pub keys: Option<&'a mut Vec<u32>>,
}

/// Explicit-mapping device sampler probe (`engine/native/tools/spec.cpp`: `sampled`, `sdist`, `stime`).
pub trait SamplerProbe {
    /// Copies `bytes` bytes from `src` to `dst` on the runtime's stream.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when either buffer is too small, unknown, overlaps, or the copy fails.
    fn copy_dev(&mut self, dst: DevBuf, src: DevBuf, bytes: usize) -> Result<(), RuntimeError>;
    /// Copies the first `rows` logit rows (`rows * LOGIT_ROW` f32) of the last forward call into `dst` starting at
    /// row `dst_row`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `rows` exceeds the last call's logit rows, `dst` is too small, or the copy fails.
    fn copy_logits(&mut self, dst: DevBuf, dst_row: usize, rows: usize) -> Result<(), RuntimeError>;
    /// Runs `sample_rows_kernel` in explicit test-mapping mode (`SampleArgs` with `lrow = null`) and downloads the
    /// requested outputs.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when any index or buffer is out of bounds, or the kernel/copy fails.
    fn sample_test_rows(&mut self, rows: &SampleTestRows<'_>, out: &mut SampleTestOut<'_>) -> Result<(), RuntimeError>;
    /// Times `reps` back-to-back launches (after `warm` untimed warmups) of device `argmax` (`sample = None`) or
    /// device `sample` (`sample = Some(params)`) on the last forward call's logits using CUDA events; returns
    /// `(gpu_ms_per_rep, wall_start, wall_end)` where `wall_start..wall_end` brackets the timed loop.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when there is no uncommitted forward call, `params` does not match it, or a CUDA call
    /// fails.
    fn time_select(
        &mut self,
        sample: Option<&[DeviceSampling]>,
        warm: usize,
        reps: usize,
    ) -> Result<(f64, std::time::Instant, std::time::Instant), RuntimeError>;
}

/// One resume checkpoint per slot: the state a slot needs to continue at a position it has passed.
///
/// A slot's state at length `n` is its global K/V `[0, n)`, the sliding-window keys `[n - 1023, n)` and the conv
/// state at `n`. [`SlotCheckpoints::checkpoint`] keeps what later rows would overwrite (the conv state now, the window
/// before the first write that could overwrite it), so [`SlotCheckpoints::rollback`] can put the slot back at the
/// checkpoint's position bit for bit.
pub trait SlotCheckpoints {
    /// Makes the slot's current length its checkpoint (replacing an older one).
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the slot does not exist, is empty, or the device copy fails.
    fn checkpoint(&mut self, slot: usize) -> Result<(), RuntimeError>;
    /// Position of the slot's checkpoint; `None` without one (a reset removes it).
    fn checkpoint_pos(&self, slot: usize) -> Option<usize>;
    /// Rolls the slot back to its checkpoint at `pos`: state as right after the checkpoint was taken, length `pos`,
    /// the pages past `pos` back to the free pool. The checkpoint stays.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the slot has no checkpoint at `pos`, or the device copy fails.
    fn rollback(&mut self, slot: usize, pos: usize) -> Result<(), RuntimeError>;
}

/// The drafter: proposes [`DRAFTS`] tokens per session that the target verifies (`ink::DraftRT`).
///
/// Buffers are checked as [`SpecRuntime`]'s (its bounds contract); the methods whose device contents index the
/// drafter's memory are `unsafe fn`s.
pub trait Drafter: std::fmt::Debug {
    /// Commits `n` target rows: features (`n` feature rows), positions and sessions (`n` i32 each).
    ///
    /// # Safety
    ///
    /// When the device work runs, every position of `pos[0..n]` is in `0..max_ctx` and every session of `sess[0..n]`
    /// in `0..sessions` ([`DraftConfig`]): the commit writes K/V at that session's position with no bound.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a buffer is too small or not the runtime's.
    #[allow(unsafe_code)] // as SpecRuntime::forward_dev
    unsafe fn commit_rows(&mut self, feats: DevBuf, pos: DevBuf, sess: DevBuf, n: usize) -> Result<(), RuntimeError>;
    /// The recurrent input of each session: its last committed row's features (`sessions` feature rows).
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the buffer is too small or not the runtime's.
    fn frnn(&mut self, feats: DevBuf) -> Result<(), RuntimeError>;
    /// Sets each session's anchor token and its position from device arrays of `sessions` i32.
    ///
    /// # Safety
    ///
    /// Until the next `set_state`, every session that [`Drafter::draft`] or [`Drafter::draft_sessions`] drafts has,
    /// when its device work runs, an anchor in `0..VOCAB` (the embedding gather) and a position `p` with
    /// `p <= max_ctx` (the draft block writes K/V at `p .. p + 8` of the `max_ctx + 8` positions the drafter holds),
    /// both read with no bound.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a buffer is too small or not the runtime's.
    #[allow(unsafe_code)] // as SpecRuntime::forward_dev
    unsafe fn set_state(&mut self, anchor: DevBuf, pos: DevBuf) -> Result<(), RuntimeError>;
    /// Runs the drafter's layers and head for every session; the drafts are then in [`Drafter::drafts`].
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the device call fails.
    fn draft(&mut self) -> Result<(), RuntimeError>;
    /// Runs the drafter's layers and head for the distinct `sessions` only (any order): their drafts are then in
    /// [`Drafter::drafts`], the other sessions' drafts are left as they were. A session's drafts are the same
    /// whichever other sessions draft with it (`engine/native/src/draft_sel.cuh`). A drafter that can only draft every
    /// session refuses, which is the default.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `sessions` is empty, names a session twice or one outside the drafter's, or the
    /// device call fails.
    fn draft_sessions(&mut self, sessions: &[usize]) -> Result<(), RuntimeError> {
        let _ = sessions;
        Err(RuntimeError::request("this drafter drafts every session".to_owned()))
    }
    /// The drafts `[sessions][7]` (i32) on the device.
    fn drafts(&self) -> DevBuf;
    /// The cumulative log drafter probability of each session's drafts 0..i `[sessions][7]` (f32) on the device, of
    /// the session's last draft; `None` when the drafter was created with [`VerifyWidth::Full`] (no probability pass),
    /// which is the default.
    fn conf(&self) -> Option<DevBuf> {
        None
    }
    /// Default probability threshold (`tau` in `(0, 1]`) for [`VerifyWidth::Gated`] with this drafter (`VERIFY_TAU` =
    /// `0.7` for autoregressive DSpark; lower for parallel diffusion drafters like DFlash2).
    fn default_verify_tau(&self) -> f32 {
        VERIFY_TAU
    }
    /// Sets per-session sampling temperature (`< 1e-5` = greedy), nucleus mass `top_p` (`<= 0` or `>= 1` = off), and
    /// Philox seed for drafters that sample from a proposal distribution (`DSpark`, `DFlash2`).
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the slice lengths do not match `sessions` or the device copy fails.
    fn set_sampling(&mut self, temps: &[f32], top_ps: &[f32], seeds: &[u64]) -> Result<(), RuntimeError> {
        let _ = (temps, top_ps, seeds);
        Ok(())
    }
    /// Sparse proposal distribution buffers (`[sessions][7][16]`) written by the drafter for exact Leviathan rejection
    /// sampling; `None` when the drafter does not expose a proposal distribution.
    fn proposal(&self) -> Option<ProposalBufs> {
        None
    }
}

#[cfg(test)]
mod call_rows_tests {
    use super::CallRows;

    #[test]
    fn rounds_up_to_s_m_and_widens_the_widest_below_8() {
        let m5 = CallRows::parse("1,2,4,6,8").unwrap();
        assert_eq!(m5.text(), "1,2,4,6,8");
        assert_eq!(CallRows::parse("ragged").unwrap(), CallRows::RAGGED);
        assert!(CallRows::parse("1,2,4").is_err());
        assert!(CallRows::parse("0,8").is_err());
        let fill = |c: CallRows, w: &[usize]| {
            let mut w = w.to_vec();
            c.fill(&mut w);
            w
        };
        // T = 11 over 4 sessions: m = 4, 16 rows; 4 -> 8, then the other 4 -> 5
        assert_eq!(fill(m5, &[1, 2, 4, 4]), [1, 2, 5, 8]);
        // T = 8 = 4 x 2: unchanged
        assert_eq!(fill(m5, &[1, 1, 2, 4]), [1, 1, 2, 4]);
        // T = 2 over 1 session: m = 2
        assert_eq!(fill(m5, &[2]), [2]);
        assert_eq!(
            fill(CallRows::parse("1,2,3,4,6,8").unwrap(), &[1, 2, 2, 4]),
            [1, 2, 2, 7]
        );
        assert_eq!(fill(m5, &[8, 8]), [8, 8]);
        // T = 17 over 3: m = 6, 18 rows
        assert_eq!(fill(m5, &[1, 8, 8]), [2, 8, 8]);
        assert_eq!(fill(CallRows::RAGGED, &[1, 2, 4]), [1, 2, 4]);
    }
}
