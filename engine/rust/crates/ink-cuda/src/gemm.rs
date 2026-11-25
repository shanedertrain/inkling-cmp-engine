//! Batch-invariant dense GEMMs through cuBLASLt with pinned algorithms (port of `LtGemm`,
//! `engine/native/src/gemm_lt.hpp` @ a3209a0, minus its heuristic fallback; its timing-based tuning is the `tune` submodule).
//!
//! `out[M, N] = x[M, K] @ W[N, K]^T` (row-major view; column-major for cuBLASLt: `A = W` (K x N, transposed),
//! `B = x` (K x M), `C = D = out` (N x M)), fp32 accumulation and scale, bf16 or fp32 operands (fp32 = IEEE, no TF32),
//! bf16 or fp32 output. The algorithm of every shape comes from [`AlgoPins`]; it is checked with
//! `cublasLtMatmulAlgoCheck` for each call shape, and a refusal is an error. A pin with split-K > 1 (out-of-place,
//! fixed-order reduction: G1) runs on the workspace given to [`LtGemm::set_workspace`]; an algorithm needing more
//! workspace than that is refused.
//!
//! cudarc's safe `CudaBlasLT` picks an algorithm by heuristic on every call, so it cannot pin one: handle,
//! descriptors and layouts use `cudarc::cublaslt::result`; algorithm init, attributes, check and the heuristic query
//! of [`LtGemm::select`] use `cudarc::cublaslt::sys`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::Arc;

use bytemuck::Pod;
use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys::{
    self, cublasComputeType_t, cublasLtMatmulAlgoCapAttributes_t as Cap, cublasLtMatmulAlgoConfigAttributes_t as Cfg,
    cublasLtMatmulDescAttributes_t, cublasLtMatmulPreferenceAttributes_t, cudaDataType,
};
use cudarc::driver::CudaContext;

use crate::arena::DevBuf;
use crate::context::Stream;
use crate::error::{CudaError, Result, blas, check_blas};
use crate::pins::{AlgoPin, AlgoPins, Dtype, GemmShape};

mod tune;
pub use tune::{Candidate, Tuned};

/// `CUBLAS_OP_N` / `CUBLAS_OP_T` (`cublasOperation_t`, an `int` enum of `cublas_api.h`; values checked by the ABI
/// test).
pub(crate) const CUBLAS_OP_N: i32 = 0;
pub(crate) const CUBLAS_OP_T: i32 = 1;

/// A bfloat16 value as stored on the device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bf16(pub u16);

// SAFETY: repr(transparent) over u16: no padding, every bit pattern valid.
unsafe impl bytemuck::Zeroable for Bf16 {}
// SAFETY: as above; Copy and 'static.
unsafe impl Pod for Bf16 {}

impl Bf16 {
    /// Rounds an f32 to the nearest bf16, ties to even (the C++ `f2bf`; finite inputs, NaN to the canonical NaN).
    pub fn from_f32(f: f32) -> Bf16 {
        let u = f.to_bits();
        if u & 0x7fff_ffff > 0x7f80_0000 {
            return Bf16(0x7fc0);
        }
        Bf16((u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16)
    }

    /// The exact f32 value.
    pub fn to_f32(self) -> f32 {
        f32::from_bits(u32::from(self.0) << 16)
    }
}

/// Element types a GEMM operand or output can have. Sealed: [`Bf16`] and `f32`.
pub trait GemmElem: Pod + private::Sealed {
    /// The matching [`Dtype`].
    const DTYPE: Dtype;
}
impl GemmElem for Bf16 {
    const DTYPE: Dtype = Dtype::Bf16;
}
impl GemmElem for f32 {
    const DTYPE: Dtype = Dtype::F32;
}
mod private {
    pub trait Sealed {}
    impl Sealed for super::Bf16 {}
    impl Sealed for f32 {}
}

fn cuda_type(d: Dtype) -> cudaDataType {
    match d {
        Dtype::Bf16 => cudaDataType::CUDA_R_16BF,
        Dtype::F32 => cudaDataType::CUDA_R_32F,
    }
}

#[derive(Debug)]
struct Handle {
    raw: sys::cublasLtHandle_t,
    _ctx: Arc<CudaContext>,
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: raw came from cublasLtCreate and is destroyed once; graphs whose nodes use its kernels hold this
        // Handle. Errors cannot be reported from drop.
        let _ = unsafe { lt::destroy_handle(self.raw) };
    }
}

#[derive(Debug)]
struct Desc(sys::cublasLtMatmulDesc_t);
impl Drop for Desc {
    fn drop(&mut self) {
        // SAFETY: created by cublasLtMatmulDescCreate, destroyed once.
        let _ = unsafe { lt::destroy_matmul_desc(self.0) };
    }
}

#[derive(Debug)]
struct Layout(sys::cublasLtMatrixLayout_t);
impl Drop for Layout {
    fn drop(&mut self) {
        // SAFETY: created by cublasLtMatrixLayoutCreate, destroyed once.
        let _ = unsafe { lt::destroy_matrix_layout(self.0) };
    }
}

struct Pref(sys::cublasLtMatmulPreference_t);
impl Drop for Pref {
    fn drop(&mut self) {
        // SAFETY: created by cublasLtMatmulPreferenceCreate, destroyed once.
        let _ = unsafe { lt::destroy_matmul_pref(self.0) };
    }
}

#[derive(Debug)]
struct Plan {
    op: Desc,
    a: Layout,
    algo: sys::cublasLtMatmulAlgo_t,
    /// Minimum byte alignment of the A (W), B (x), C and D (out) pointers the algorithm requires
    /// (`CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_*_BYTES`; review R3 finding F10).
    align: [u32; 4],
}

/// A heuristic choice of [`LtGemm::select`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selected {
    /// Its attributes, to pin.
    pub pin: AlgoPin,
    /// The opaque algorithm bytes cuBLASLt returned (to compare with [`LtGemm::pinned_opaque`]).
    pub opaque: [u64; 8],
    /// Position among the heuristic's candidates.
    pub candidate: usize,
    /// Number of candidates returned.
    pub candidates: usize,
}

/// cuBLASLt GEMMs with one pinned algorithm per shape.
#[derive(Debug)]
pub struct LtGemm {
    plans: HashMap<GemmShape, Plan>,
    layouts: HashMap<(Dtype, u32, u32, u32), Layout>,
    batch_layouts: HashMap<(u32, u32, u32, u64, u32), Layout>,
    /// `cublasLtMatmulAlgoCheck` results per call shape: `None` refused, else the workspace bytes the algorithm needs.
    checked: HashMap<CallKey, Option<usize>>,
    workspace: Option<DevBuf<u8>>,
    pins: AlgoPins,
    handle: Rc<Handle>,
}

/// A call shape as checked: the GEMM shape, rows, strides of x and out, and for a strided batch its x / w / out
/// batch strides and count (all 0 for a plain call).
type CallKey = (GemmShape, u32, u32, u32, [u64; 3], u32);

/// The batch of `LtGemm::run_batched`: `count` GEMMs, operand `i` at `i` times the stride (elements) past the first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Batch {
    /// Number of GEMMs (>= 1).
    count: u32,
    /// Elements from one x operand to the next.
    x_stride: u64,
    /// Elements from one weight to the next.
    w_stride: u64,
    /// Elements from one output to the next.
    out_stride: u64,
}

impl LtGemm {
    /// Creates the cuBLASLt handle and takes the pins.
    ///
    /// # Errors
    ///
    /// [`CudaError::Pins`] when the pins were chosen with another cuBLASLt version (an algorithm id is only meaningful
    /// for the library that reported it), or the cuBLASLt error of `cublasLtCreate`.
    pub fn new(stream: &Stream, pins: AlgoPins) -> Result<LtGemm> {
        // SAFETY: no arguments; the library was checked by the startup checks of the stream's Context.
        let version = unsafe { sys::cublasLtGetVersion() };
        if !pins.is_empty() && pins.cublaslt_version() != version {
            return Err(CudaError::Pins(format!(
                "pins are for cuBLASLt {}, this is {version}",
                pins.cublaslt_version()
            )));
        }
        let raw = lt::create_handle().map_err(blas("cublasLtCreate"))?;
        let handle = Rc::new(Handle {
            raw,
            _ctx: stream.inner.context().clone(),
        });
        Ok(LtGemm {
            plans: HashMap::new(),
            layouts: HashMap::new(),
            batch_layouts: HashMap::new(),
            checked: HashMap::new(),
            workspace: None,
            pins,
            handle,
        })
    }

    /// The pins in use.
    pub fn pins(&self) -> &AlgoPins {
        &self.pins
    }

    /// The workspace of split-K plans (the C++ `LtGemm::set_workspace`); an algorithm that needs more than its size
    /// is refused. Clears the cached algorithm checks.
    pub fn set_workspace(&mut self, ws: DevBuf<u8>) {
        self.workspace = Some(ws);
        self.checked.clear();
    }

    /// Picks an algorithm for `shape` the way the C++ `LtGemm::plan` does: the heuristic at `m_rep` rows with no
    /// workspace, first candidate with split-K <= 1, else the best one forced to split-K 1. For the tuning mode;
    /// the result goes into the pin file, it is never used directly.
    ///
    /// # Errors
    ///
    /// The cuBLASLt error, or [`CudaError::NoPin`] when the heuristic returns no candidate.
    pub fn select(&mut self, shape: GemmShape, m_rep: u32) -> Result<Selected> {
        let op = op_desc()?;
        let a = new_layout(shape.input, shape.k, shape.n, shape.ldw)?;
        let b = new_layout(shape.input, shape.k, m_rep, shape.k)?;
        let c = new_layout(shape.output, shape.n, m_rep, shape.n)?;
        let pref = Pref(lt::create_matmul_pref().map_err(blas("cublasLtMatmulPreferenceCreate"))?);
        let ws: usize = 0;
        let attr = cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
        // SAFETY: pref is live; the attribute is a size_t read from a valid local of that size.
        unsafe { lt::set_matmul_pref_attribute(pref.0, attr, (&raw const ws).cast(), size_of::<usize>()) }
            .map_err(blas("cublasLtMatmulPreferenceSetAttribute"))?;
        // SAFETY: cublasLtMatmulHeuristicResult_t is plain data (an opaque u64 array, integers, a float); the library
        // overwrites the first `n` entries.
        let mut res: [sys::cublasLtMatmulHeuristicResult_t; 16] = unsafe { std::mem::zeroed() };
        let mut n = 0i32;
        // SAFETY: all descriptors are live; res has room for the 16 requested results; n is a valid out-pointer.
        let st = unsafe {
            sys::cublasLtMatmulAlgoGetHeuristic(
                self.handle.raw,
                op.0,
                a.0,
                b.0,
                c.0,
                c.0,
                pref.0,
                16,
                res.as_mut_ptr(),
                &mut n,
            )
        };
        check_blas("cublasLtMatmulAlgoGetHeuristic", st)?;
        let n = usize::try_from(n).unwrap_or(0).min(res.len());
        if n == 0 {
            return Err(CudaError::NoPin(format!(
                "{shape}: the heuristic returned no algorithm"
            )));
        }
        let mut pick = None;
        for (i, r) in res[..n].iter().enumerate() {
            if get_attr::<i32>(&r.algo, Cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)? <= 1 {
                pick = Some(i);
                break;
            }
        }
        let candidate = match pick {
            Some(i) => i,
            None => {
                set_attr(&mut res[0].algo, Cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1i32)?;
                0
            }
        };
        let algo = res[candidate].algo;
        Ok(Selected {
            pin: read_pin(&algo)?,
            opaque: algo.data,
            candidate,
            candidates: n,
        })
    }

    /// The opaque bytes of the algorithm built from the pin of `shape` (to compare with the heuristic's).
    ///
    /// # Errors
    ///
    /// As [`LtGemm::run`] for building the plan.
    pub fn pinned_opaque(&mut self, shape: GemmShape) -> Result<[u64; 8]> {
        Ok(self.plan(shape)?.algo.data)
    }

    /// `out[m, n] = x[m, k] @ w[n, k]^T` with the pinned algorithm of `shape`. Allowed in a capture.
    ///
    /// `ldx` and `ldo` are the row strides of x and out in elements; W's stride is `shape.ldw`.
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] when a buffer is too small or a stride too short, or the element types do not match the
    /// shape; [`CudaError::NoPin`]; [`CudaError::GemmRefused`]; the cuBLASLt error of the call.
    #[expect(clippy::too_many_arguments, reason = "mirrors the C++ LtGemm::run signature")]
    pub fn run<I: GemmElem, O: GemmElem>(
        &mut self,
        stream: &Stream,
        shape: GemmShape,
        x: &DevBuf<I>,
        ldx: u32,
        w: &DevBuf<I>,
        out: &DevBuf<O>,
        ldo: u32,
        m: u32,
    ) -> Result<()> {
        check_operands(shape, I::DTYPE, O::DTYPE, (x.len(), ldx), w.len(), (out.len(), ldo), m)?;
        let align = self.plan(shape)?.align;
        check_alignment(shape, align, [w.addr(), x.addr(), out.addr(), out.addr()])?;
        let b = self.layout(shape.input, shape.k, m, ldx)?;
        let c = self.layout(shape.output, shape.n, m, ldo)?;
        let a = self.plans[&shape].a.0;
        let ws = self.checked_workspace((shape, m, ldx, ldo, [0; 3], 0), a, b, c, "")?;
        self.matmul(stream, shape, (w, a), (x.addr(), b), (out.addr(), c), ws)?;
        if stream.is_capturing() {
            stream.keep(x.keep_alive());
            stream.keep(out.keep_alive());
        }
        Ok(())
    }

    /// The workspace bytes the pinned algorithm of `key.0` needs for these layouts (checked once per call shape with
    /// `cublasLtMatmulAlgoCheck`, cached); [`CudaError::GemmRefused`] when it refuses them or needs more workspace
    /// than [`LtGemm::set_workspace`] gave.
    fn checked_workspace(
        &mut self,
        key: CallKey,
        a: sys::cublasLtMatrixLayout_t,
        b: sys::cublasLtMatrixLayout_t,
        c: sys::cublasLtMatrixLayout_t,
        what: &str,
    ) -> Result<usize> {
        let have = self.workspace.as_ref().map_or(0, DevBuf::len);
        let plan = &self.plans[&key.0];
        let ws = match self.checked.get(&key) {
            Some(&ws) => ws,
            None => {
                // SAFETY: plain data, overwritten by the check.
                let mut r: sys::cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
                // SAFETY: handle, descriptor, layouts and algo are live; r is a valid out-pointer.
                let st =
                    unsafe { sys::cublasLtMatmulAlgoCheck(self.handle.raw, plan.op.0, a, b, c, c, &plan.algo, &mut r) };
                let ws = (st == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS && r.workspaceSize <= have)
                    .then_some(r.workspaceSize);
                self.checked.insert(key, ws);
                ws
            }
        };
        ws.ok_or_else(|| CudaError::GemmRefused {
            shape: format!("{}{what}", key.0),
            m: key.1,
            ldx: key.2,
            ldo: key.3,
        })
    }

    /// The cuBLASLt call of a checked shape: `D = C = out`, `A = w`, `B = x`, on the workspace when the algorithm
    /// needs `ws` > 0 bytes (checked to fit). Keeps the handle, w and the workspace alive in a capture.
    fn matmul(
        &self,
        stream: &Stream,
        shape: GemmShape,
        (w, a): (&DevBuf<impl Pod>, sys::cublasLtMatrixLayout_t),
        (x, b): (u64, sys::cublasLtMatrixLayout_t),
        (out, c): (u64, sys::cublasLtMatrixLayout_t),
        ws: usize,
    ) -> Result<()> {
        let (one, zero) = (1.0f32, 0.0f32);
        let plan = &self.plans[&shape];
        let handle = self.handle.raw;
        let (ws_ptr, ws_len) = match (&self.workspace, ws) {
            (Some(buf), 1..) => (buf.addr() as *mut c_void, buf.len()),
            _ => (std::ptr::null_mut(), 0),
        };
        // a launch trace records the kernels cuBLASLt launches (trace.rs); otherwise this is a plain eager call
        stream.traced_library_call(|| {
            // SAFETY: descriptors, layouts and algo are live; the callers checked the extents of x, w and out (and of
            // every batch entry) against the layouts, so cuBLASLt reads and writes inside the three buffers; alpha and
            // beta are host f32 (pointer mode host, scale type f32); the workspace is either none (the algorithm needs
            // none) or the whole workspace buffer, which the check found large enough. The stream handle is the same
            // CUstream_st under cuBLASLt's own type name.
            unsafe {
                lt::matmul(
                    handle,
                    plan.op.0,
                    (&raw const one).cast(),
                    (&raw const zero).cast(),
                    w.addr() as *const c_void,
                    a,
                    x as *const c_void,
                    b,
                    out as *const c_void,
                    c,
                    out as *mut c_void,
                    c,
                    &plan.algo,
                    ws_ptr,
                    ws_len,
                    stream.inner.cu_stream().cast(),
                )
            }
            .map_err(blas("cublasLtMatmul"))
        })?;
        if stream.is_capturing() {
            stream.keep(self.handle.clone());
            stream.keep(w.keep_alive());
            if let (Some(buf), 1..) = (&self.workspace, ws) {
                stream.keep(buf.keep_alive());
            }
        }
        Ok(())
    }

    /// `batch.count` GEMMs of one bf16 -> bf16 shape as one strided batch (port of the C++ `LtGemm::runb`): for
    /// i < count, `out[i out_stride ..] (ld ldo) = x[i x_stride ..] (ld ldx) @ w[i w_stride ..]^T`, `m` rows, on the
    /// pinned algorithm of `shape` (W layout `k x n`, ld `k`): the same bits as `count` [`LtGemm::run`] calls, one
    /// launch. Allowed in a capture.
    ///
    /// # Errors
    ///
    /// As [`LtGemm::run`] for every batch entry; a refused batch is [`CudaError::GemmRefused`].
    #[expect(clippy::too_many_arguments, reason = "mirrors the C++ LtGemm::runb signature")]
    fn run_batched(
        &mut self,
        stream: &Stream,
        shape: GemmShape,
        x: &DevBuf<Bf16>,
        ldx: u32,
        w: &DevBuf<Bf16>,
        out: &DevBuf<Bf16>,
        ldo: u32,
        m: u32,
        batch: Batch,
    ) -> Result<()> {
        let bad = |detail: String| {
            Err(CudaError::Bounds {
                what: "gemm batch",
                detail,
            })
        };
        if shape.input != Dtype::Bf16 || shape.output != Dtype::Bf16 || shape.ldw != shape.k {
            return bad(format!("{shape}: a batch takes bf16 -> bf16 with ldw = k"));
        }
        if batch.count == 0 {
            return bad(format!("{shape}: empty batch"));
        }
        // the last entry starts (count - 1) strides past the first: one call's extents from there
        let last = |stride: u64| -> Result<usize> {
            usize::try_from(stride * u64::from(batch.count - 1)).map_err(|_| CudaError::Bounds {
                what: "gemm batch",
                detail: "stride".into(),
            })
        };
        let (xs, ws, os) = (last(batch.x_stride)?, last(batch.w_stride)?, last(batch.out_stride)?);
        if x.len() < xs || w.len() < ws || out.len() < os {
            return bad(format!("{shape}: batch offsets exceed the buffers"));
        }
        check_operands(
            shape,
            Dtype::Bf16,
            Dtype::Bf16,
            (x.len() - xs, ldx),
            w.len() - ws,
            (out.len() - os, ldo),
            m,
        )?;
        let align = self.plan(shape)?.align;
        for addr in batch_addrs(batch, w.addr(), x.addr(), out.addr()) {
            check_alignment(shape, align, addr)?;
        }
        let a = self.batch_layout(shape.k, shape.n, shape.k, batch.w_stride, batch.count)?;
        let b = self.batch_layout(shape.k, m, ldx, batch.x_stride, batch.count)?;
        let c = self.batch_layout(shape.n, m, ldo, batch.out_stride, batch.count)?;
        let key = (
            shape,
            m,
            ldx,
            ldo,
            [batch.x_stride, batch.w_stride, batch.out_stride],
            batch.count,
        );
        let ws = self.checked_workspace(key, a, b, c, &format!(" (batch of {})", batch.count))?;
        self.matmul(stream, shape, (w, a), (x.addr(), b), (out.addr(), c), ws)?;
        if stream.is_capturing() {
            stream.keep(x.keep_alive());
            stream.keep(out.keep_alive());
        }
        Ok(())
    }

    /// Two GEMMs of one bf16 -> bf16 shape as one strided batch (the C++ `LtGemm::run2`, D1): `run_batched` with the
    /// second weight `n k` and the second output `n` elements after the first.
    ///
    /// # Errors
    ///
    /// As [`LtGemm::run`] for both entries; a batch the algorithm refuses is [`CudaError::GemmRefused`] (the C++
    /// falls back to two calls and counts it).
    #[expect(clippy::too_many_arguments, reason = "mirrors the C++ LtGemm::run2 signature")]
    pub fn run2(
        &mut self,
        stream: &Stream,
        shape: GemmShape,
        x: &DevBuf<Bf16>,
        ldx: u32,
        xstride: u64,
        w: &DevBuf<Bf16>,
        out: &DevBuf<Bf16>,
        ldo: u32,
        m: u32,
    ) -> Result<()> {
        let batch = Batch {
            count: 2,
            x_stride: xstride,
            w_stride: u64::from(shape.n) * u64::from(shape.k),
            out_stride: u64::from(shape.n),
        };
        self.run_batched(stream, shape, x, ldx, w, out, ldo, m, batch)
    }

    /// A bf16 layout of a strided batch (`blay` of the C++), cached.
    fn batch_layout(
        &mut self,
        rows: u32,
        cols: u32,
        ld: u32,
        stride: u64,
        count: u32,
    ) -> Result<sys::cublasLtMatrixLayout_t> {
        use std::collections::hash_map::Entry;
        match self.batch_layouts.entry((rows, cols, ld, stride, count)) {
            Entry::Occupied(e) => Ok(e.get().0),
            Entry::Vacant(e) => {
                let l = new_layout(Dtype::Bf16, rows, cols, ld)?;
                let count = i32::try_from(count).map_err(|_| CudaError::Bounds {
                    what: "gemm batch",
                    detail: "count".into(),
                })?;
                let off = i64::try_from(stride).map_err(|_| CudaError::Bounds {
                    what: "gemm batch",
                    detail: "stride".into(),
                })?;
                for (attr, p, n) in [
                    (
                        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                        (&raw const count).cast::<c_void>(),
                        size_of::<i32>(),
                    ),
                    (
                        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                        (&raw const off).cast::<c_void>(),
                        size_of::<i64>(),
                    ),
                ] {
                    // SAFETY: l is a live layout; the attribute is an int32 (batch count) or int64 (stride) read from
                    // a valid local of that size (cublasLt.h).
                    let st = unsafe { sys::cublasLtMatrixLayoutSetAttribute(l.0, attr, p, n) };
                    check_blas("cublasLtMatrixLayoutSetAttribute", st)?;
                }
                Ok(e.insert(l).0)
            }
        }
    }

    fn plan(&mut self, shape: GemmShape) -> Result<&Plan> {
        if !self.plans.contains_key(&shape) {
            let pin = *self
                .pins
                .get(&shape)
                .ok_or_else(|| CudaError::NoPin(shape.to_string()))?;
            let op = op_desc()?;
            let a = new_layout(shape.input, shape.k, shape.n, shape.ldw)?;
            // SAFETY: plain data, filled by cublasLtMatmulAlgoInit.
            let mut algo: sys::cublasLtMatmulAlgo_t = unsafe { std::mem::zeroed() };
            let (tin, tout) = (cuda_type(shape.input), cuda_type(shape.output));
            // SAFETY: handle is live; algo is a valid out-pointer; types match the descriptors built above.
            let st = unsafe {
                sys::cublasLtMatmulAlgoInit(
                    self.handle.raw,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cudaDataType::CUDA_R_32F,
                    tin,
                    tin,
                    tout,
                    tout,
                    pin.id,
                    &mut algo,
                )
            };
            check_blas("cublasLtMatmulAlgoInit", st)?;
            write_pin(&mut algo, &pin)?;
            let align = [
                get_cap(&algo, Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES)?,
                get_cap(&algo, Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES)?,
                get_cap(&algo, Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES)?,
                get_cap(&algo, Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES)?,
            ];
            self.plans.insert(shape, Plan { op, a, algo, align });
        }
        Ok(&self.plans[&shape])
    }

    fn layout(&mut self, t: Dtype, rows: u32, cols: u32, ld: u32) -> Result<sys::cublasLtMatrixLayout_t> {
        use std::collections::hash_map::Entry;
        match self.layouts.entry((t, rows, cols, ld)) {
            Entry::Occupied(e) => Ok(e.get().0),
            Entry::Vacant(e) => Ok(e.insert(new_layout(t, rows, cols, ld)?).0),
        }
    }
}

/// The operand addresses (A = W, B = x, C, D = out) of every entry of a bf16 [`LtGemm::run_batched`] (review V12).
fn batch_addrs(batch: Batch, w: u64, x: u64, out: u64) -> impl Iterator<Item = [u64; 4]> {
    (0..u64::from(batch.count)).map(move |i| {
        let wi = w + 2 * i * batch.w_stride;
        let xi = x + 2 * i * batch.x_stride;
        let oi = out + 2 * i * batch.out_stride;
        [wi, xi, oi, oi]
    })
}

/// Checks the operand addresses (A = W, B = x, C = D = out) against the algorithm's minimum alignments.
fn check_alignment(shape: GemmShape, align: [u32; 4], addr: [u64; 4]) -> Result<()> {
    for ((what, a), p) in ["w", "x", "out (C)", "out (D)"].iter().zip(align).zip(addr) {
        if a > 1 && !p.is_multiple_of(u64::from(a)) {
            return Err(CudaError::Bounds {
                what: "gemm alignment",
                detail: format!("{shape}: {what} at {p:#x} is not {a}-byte aligned, as the pinned algorithm requires"),
            });
        }
    }
    Ok(())
}

/// Checks types, strides and extents of one call (column-major extents: `ld * (cols - 1) + rows`).
fn check_operands(
    shape: GemmShape,
    tin: Dtype,
    tout: Dtype,
    (x_len, ldx): (usize, u32),
    w_len: usize,
    (out_len, ldo): (usize, u32),
    m: u32,
) -> Result<()> {
    let bad = |detail: String| Err(CudaError::Bounds { what: "gemm", detail });
    if tin != shape.input || tout != shape.output {
        return bad(format!("{shape}: buffers are {tin:?}->{tout:?}"));
    }
    // n, k and m >= 1: `need` below subtracts one from each column count
    if m == 0 || shape.n == 0 || shape.k == 0 || ldx < shape.k || ldo < shape.n || shape.ldw < shape.k {
        return bad(format!("{shape}: m={m} ldx={ldx} ldo={ldo}"));
    }
    let need = |ld: u32, cols: u32, rows: u32| ld as usize * (cols as usize - 1) + rows as usize;
    for (what, have, want) in [
        ("x", x_len, need(ldx, m, shape.k)),
        ("w", w_len, need(shape.ldw, shape.n, shape.k)),
        ("out", out_len, need(ldo, m, shape.n)),
    ] {
        if have < want {
            return bad(format!("{shape} m={m}: {what} has {have} elements, needs {want}"));
        }
    }
    Ok(())
}

fn op_desc() -> Result<Desc> {
    let op = Desc(
        lt::create_matmul_desc(cublasComputeType_t::CUBLAS_COMPUTE_32F, cudaDataType::CUDA_R_32F)
            .map_err(blas("cublasLtMatmulDescCreate"))?,
    );
    for (attr, v) in [
        (cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T),
        (cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N),
    ] {
        // SAFETY: op is live; the attribute is a cublasOperation_t (int32) read from a valid local.
        unsafe { lt::set_matmul_desc_attribute(op.0, attr, (&raw const v).cast(), size_of::<i32>()) }
            .map_err(blas("cublasLtMatmulDescSetAttribute"))?;
    }
    Ok(op)
}

fn new_layout(t: Dtype, rows: u32, cols: u32, ld: u32) -> Result<Layout> {
    let l = lt::create_matrix_layout(cuda_type(t), rows.into(), cols.into(), ld.into())
        .map_err(blas("cublasLtMatrixLayoutCreate"))?;
    Ok(Layout(l))
}

/// A `CUBLASLT_ALGO_CONFIG_*` attribute value type.
trait AttrValue: Pod + Default {}
impl AttrValue for i32 {}
impl AttrValue for u32 {}
impl AttrValue for u16 {}

fn get_attr<T: AttrValue>(algo: &sys::cublasLtMatmulAlgo_t, attr: Cfg) -> Result<T> {
    let mut v = T::default();
    let mut written = 0usize;
    // SAFETY: algo is a valid algorithm struct; v is a writable buffer of size_of::<T>() bytes, the attribute's type
    // per cublasLt.h (checked by the callers' choice of T).
    let st = unsafe {
        sys::cublasLtMatmulAlgoConfigGetAttribute(algo, attr, (&raw mut v).cast(), size_of::<T>(), &mut written)
    };
    check_blas("cublasLtMatmulAlgoConfigGetAttribute", st)?;
    Ok(v)
}

/// A `CUBLASLT_ALGO_CAP_*` value of type `uint32_t` (the alignment caps).
fn get_cap(algo: &sys::cublasLtMatmulAlgo_t, attr: Cap) -> Result<u32> {
    let mut v = 0u32;
    let mut written = 0usize;
    // SAFETY: algo is an initialised algorithm struct; v is a writable u32, the type of the MIN_ALIGNMENT caps per
    // cublasLt.h.
    let st = unsafe {
        sys::cublasLtMatmulAlgoCapGetAttribute(algo, attr, (&raw mut v).cast(), size_of::<u32>(), &mut written)
    };
    check_blas("cublasLtMatmulAlgoCapGetAttribute", st)?;
    Ok(v)
}

fn set_attr<T: AttrValue>(algo: &mut sys::cublasLtMatmulAlgo_t, attr: Cfg, v: T) -> Result<()> {
    // SAFETY: algo is a valid algorithm struct; v is size_of::<T>() bytes of the attribute's type per cublasLt.h.
    let st = unsafe { sys::cublasLtMatmulAlgoConfigSetAttribute(algo, attr, (&raw const v).cast(), size_of::<T>()) };
    check_blas("cublasLtMatmulAlgoConfigSetAttribute", st)
}

fn read_pin(a: &sys::cublasLtMatmulAlgo_t) -> Result<AlgoPin> {
    Ok(AlgoPin {
        id: get_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_ID)?,
        tile: get_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
        splitk: get_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
        reduction: get_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
        swizzle: get_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)?,
        custom: get_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?,
        stages: get_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_STAGES_ID)?,
        inner_shape: get_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)?,
        cluster_shape: get_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)?,
    })
}

fn write_pin(a: &mut sys::cublasLtMatmulAlgo_t, p: &AlgoPin) -> Result<()> {
    set_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_TILE_ID, p.tile)?;
    set_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, p.splitk)?;
    set_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, p.reduction)?;
    set_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, p.swizzle)?;
    set_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, p.custom)?;
    set_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_STAGES_ID, p.stages)?;
    set_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, p.inner_shape)?;
    set_attr(a, Cfg::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, p.cluster_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pins::PlanClass;

    fn shape() -> GemmShape {
        GemmShape {
            n: 64,
            k: 32,
            ldw: 32,
            input: Dtype::Bf16,
            output: Dtype::F32,
            class: PlanClass::Decode,
        }
    }

    #[test]
    fn bf16_rounding_matches_the_cpp_f2bf() {
        assert_eq!(Bf16::from_f32(1.0).0, 0x3f80);
        assert_eq!(Bf16::from_f32(-2.5).0, 0xc020);
        // ties to even: 1 + 2^-8 is halfway between 1 and 1 + 2^-7
        assert_eq!(Bf16::from_f32(f32::from_bits(0x3f80_8000)).0, 0x3f80);
        assert_eq!(Bf16::from_f32(f32::from_bits(0x3f81_8000)).0, 0x3f82);
        assert_eq!(Bf16::from_f32(f32::NAN).0, 0x7fc0);
        for b in [0x0000u16, 0x3f80, 0xc120, 0x7f7f, 0x0001] {
            assert_eq!(Bf16::from_f32(Bf16(b).to_f32()).0, b, "round trip {b:#06x}");
        }
    }

    #[test]
    fn the_second_batch_of_run2_is_alignment_checked() {
        let s = |n: u32| GemmShape {
            n,
            k: 2304,
            ldw: 2304,
            input: Dtype::Bf16,
            output: Dtype::Bf16,
            class: PlanClass::Decode,
        };
        let check = |shape: GemmShape, xstride: u64| {
            let batch = Batch {
                count: 2,
                x_stride: xstride,
                w_stride: u64::from(shape.n) * u64::from(shape.k),
                out_stride: u64::from(shape.n),
            };
            batch_addrs(batch, 0x1000_0000, 0x2000_0000, 0x3000_0000)
                .try_for_each(|a| check_alignment(shape, [16; 4], a))
        };
        // the D1 k / v batch: n = 1024 (or 1056) columns, x batches t_max x H apart: both batches 16-byte aligned
        assert!(check(s(1024), 2112 * 2304).is_ok());
        assert!(check(s(1056), 2112 * 2304).is_ok());
        // batch 0 aligned, batch 1 not: out + n (2 x 1028 bytes = 8 mod 16), x + xstride (2 x 4 bytes)
        let e = check(s(1028), 2112 * 2304).expect_err("out of batch 1");
        assert!(e.to_string().contains("out (C)"), "{e}");
        let e = check(s(1024), 4).expect_err("x of batch 1");
        assert!(e.to_string().contains("x at"), "{e}");
    }

    #[test]
    fn operand_extents_are_checked() {
        let s = shape();
        // m = 3 rows, tight strides: x 3*32, w 64*32, out 3*64
        assert!(check_operands(s, Dtype::Bf16, Dtype::F32, (96, 32), 2048, (192, 64), 3).is_ok());
        // strided x: ld 40 -> 40*2 + 32
        assert!(check_operands(s, Dtype::Bf16, Dtype::F32, (112, 40), 2048, (192, 64), 3).is_ok());
        assert!(check_operands(s, Dtype::Bf16, Dtype::F32, (111, 40), 2048, (192, 64), 3).is_err());
        assert!(check_operands(s, Dtype::Bf16, Dtype::F32, (96, 32), 2047, (192, 64), 3).is_err());
        assert!(check_operands(s, Dtype::Bf16, Dtype::F32, (96, 32), 2048, (191, 64), 3).is_err());
        assert!(
            check_operands(s, Dtype::Bf16, Dtype::F32, (96, 31), 2048, (192, 64), 3).is_err(),
            "ldx < k"
        );
        assert!(
            check_operands(s, Dtype::Bf16, Dtype::F32, (96, 32), 2048, (192, 63), 3).is_err(),
            "ldo < n"
        );
        assert!(
            check_operands(s, Dtype::Bf16, Dtype::F32, (96, 32), 2048, (192, 64), 0).is_err(),
            "m = 0"
        );
        // n = 0 with ldw = k made the w extent wrap to 0 (release) or panic (debug) before the check
        let empty_n = GemmShape { n: 0, ..s };
        assert!(check_operands(empty_n, Dtype::Bf16, Dtype::F32, (96, 32), 0, (192, 64), 3).is_err());
        let empty_k = GemmShape { k: 0, ldw: 0, ..s };
        assert!(check_operands(empty_k, Dtype::Bf16, Dtype::F32, (96, 32), 2048, (192, 64), 3).is_err());
        assert!(
            check_operands(s, Dtype::Bf16, Dtype::Bf16, (96, 32), 2048, (192, 64), 3).is_err(),
            "out type"
        );
    }
}
