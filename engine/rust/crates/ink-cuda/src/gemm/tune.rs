// Ported from engine/native/src/gemm_lt.hpp @ f5c9169 (`LtGemm::tune`: the timing-based choice of one shape's algorithm).
// Own code of this project.
//! The tuning mode of the pin file (review finding V11): the C++ `LtGemm::tune`, which picked the algorithm of every
//! timed shape before the engine wrote `gemm-algos.json`. Every split-K-1, workspace-free candidate of the heuristic
//! at `m` rows (at most 32) runs 3 times untimed and 20 times between two events on the caller's operands; the fastest
//! wins, the first of equal times. Not used by the engine (it runs pinned algorithms only); `ink-tune` writes the pins.

use std::ffi::c_void;

use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys::{self, cublasLtMatmulAlgoConfigAttributes_t as Cfg, cublasLtMatmulPreferenceAttributes_t};

use super::{GemmElem, LtGemm, Pref, check_operands, get_attr, new_layout, op_desc, read_pin};
use crate::arena::DevBuf;
use crate::context::Stream;
use crate::error::{CudaError, Result, blas, check_blas};
use crate::pins::{AlgoPin, GemmShape};

/// Candidates the heuristic is asked for (the C++ `res[32]`).
const CANDIDATES: usize = 32;
/// Untimed and timed calls per candidate (the C++ 3 and 20).
const WARM: usize = 3;
const TIMED: usize = 20;

/// One heuristic candidate of [`LtGemm::tune`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Position among the heuristic's results.
    pub index: usize,
    /// Its attributes.
    pub pin: AlgoPin,
    /// Microseconds per call (mean of the timed calls); `None` when skipped (split-K > 1, a workspace, or a failed call).
    pub us: Option<f32>,
}

/// The result of [`LtGemm::tune`].
#[derive(Debug, Clone, PartialEq)]
pub struct Tuned {
    /// The fastest candidate's attributes, to pin.
    pub pin: AlgoPin,
    /// Its position among the heuristic's results.
    pub best: usize,
    /// Every result of the heuristic, in its order.
    pub candidates: Vec<Candidate>,
}

impl LtGemm {
    /// The C++ `LtGemm::tune` for `shape` at `m` rows on the given operands (`x` m x k, ld k; `w` n x ldw; `out`
    /// m x n, ld n; their contents only matter for timing). Runs on `stream` eagerly (never in a capture).
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] for operands that do not hold the shape at `m` rows; [`CudaError::NoPin`] when no
    /// candidate could be timed; cuBLASLt and driver errors outside the timed calls.
    #[expect(clippy::too_many_arguments, reason = "the GEMM's operands and the timing context")]
    pub fn tune<I: GemmElem, O: GemmElem>(
        &mut self,
        stream: &Stream,
        start: &crate::context::Event,
        end: &crate::context::Event,
        shape: GemmShape,
        m: u32,
        x: &DevBuf<I>,
        w: &DevBuf<I>,
        out: &DevBuf<O>,
    ) -> Result<Tuned> {
        if stream.is_capturing() {
            return Err(CudaError::Bounds {
                what: "gemm tune",
                detail: "not inside a capture".into(),
            });
        }
        check_operands(
            shape,
            I::DTYPE,
            O::DTYPE,
            (x.len(), shape.k),
            w.len(),
            (out.len(), shape.n),
            m,
        )?;
        let op = op_desc()?;
        let a = new_layout(shape.input, shape.k, shape.n, shape.ldw)?;
        let b = new_layout(shape.input, shape.k, m, shape.k)?;
        let c = new_layout(shape.output, shape.n, m, shape.n)?;
        let pref = Pref(lt::create_matmul_pref().map_err(blas("cublasLtMatmulPreferenceCreate"))?);
        let ws: usize = 0;
        let attr = cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
        // SAFETY: pref is live; the attribute is a size_t read from a valid local of that size.
        unsafe { lt::set_matmul_pref_attribute(pref.0, attr, (&raw const ws).cast(), size_of::<usize>()) }
            .map_err(blas("cublasLtMatmulPreferenceSetAttribute"))?;
        // SAFETY: plain data (an opaque u64 array, integers, a float); the library overwrites the first `n` entries.
        let mut res: [sys::cublasLtMatmulHeuristicResult_t; CANDIDATES] = unsafe { std::mem::zeroed() };
        let mut n = 0i32;
        // SAFETY: all descriptors are live; res has room for the CANDIDATES requested results; n is a valid out-pointer.
        let st = unsafe {
            sys::cublasLtMatmulAlgoGetHeuristic(
                self.handle.raw,
                op.0,
                a.0,
                b.0,
                c.0,
                c.0,
                pref.0,
                CANDIDATES as i32,
                res.as_mut_ptr(),
                &mut n,
            )
        };
        check_blas("cublasLtMatmulAlgoGetHeuristic", st)?;
        let n = usize::try_from(n).unwrap_or(0).min(CANDIDATES);
        let (one, zero) = (1.0f32, 0.0f32);
        let handle = self.handle.raw;
        let call = |algo: &sys::cublasLtMatmulAlgo_t| {
            // SAFETY: descriptors and layouts are live; x, w and out hold the shape at m rows (check_operands above)
            // with ld k / ldw / n as the layouts say; alpha and beta are host f32; no workspace (only candidates
            // that need none are run).
            unsafe {
                lt::matmul(
                    handle,
                    op.0,
                    (&raw const one).cast(),
                    (&raw const zero).cast(),
                    w.addr() as *const c_void,
                    a.0,
                    x.addr() as *const c_void,
                    b.0,
                    out.addr() as *const c_void,
                    c.0,
                    out.addr() as *mut c_void,
                    c.0,
                    algo,
                    std::ptr::null_mut(),
                    0,
                    stream.inner.cu_stream().cast(),
                )
            }
        };
        let mut candidates = Vec::with_capacity(n);
        let mut best: Option<(usize, f32)> = None;
        for (index, r) in res[..n].iter().enumerate() {
            let pin = read_pin(&r.algo)?;
            let runnable = get_attr::<i32>(&r.algo, Cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)? <= 1 && r.workspaceSize == 0;
            let mut us = None;
            if runnable {
                let ok = (0..WARM).all(|_| call(&r.algo).is_ok());
                stream.record(start)?;
                let ok = ok && (0..TIMED).all(|_| call(&r.algo).is_ok());
                stream.record(end)?;
                end.synchronize()?;
                if ok {
                    let t = start.elapsed_ms(end)? * 1000.0 / TIMED as f32;
                    us = Some(t);
                    if best.is_none_or(|(_, b)| t < b) {
                        best = Some((index, t));
                    }
                }
            }
            candidates.push(Candidate { index, pin, us });
        }
        let (best, _) = best.ok_or_else(|| CudaError::NoPin(format!("{shape} m={m}: no candidate could be timed")))?;
        Ok(Tuned {
            pin: candidates[best].pin,
            best,
            candidates,
        })
    }
}
