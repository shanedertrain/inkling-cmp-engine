//! The crate's error type: every failure carries the CUDA, cuBLASLt or NVML status it came from.

use cudarc::cublaslt::result::CublasError;
use cudarc::driver::DriverError;

/// Result of every fallible call in this crate.
pub type Result<T> = std::result::Result<T, CudaError>;

/// What went wrong in the CUDA layer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CudaError {
    /// A driver API call returned an error status.
    #[error("{call}: CUDA driver error {code} ({name})")]
    Driver {
        /// The driver API function (unversioned name, as in the CUDA docs).
        call: &'static str,
        /// The numeric `CUresult`.
        code: i32,
        /// The `CUresult` enumerator name, e.g. `CUDA_ERROR_INVALID_VALUE`.
        name: String,
    },
    /// A cuBLASLt call returned an error status.
    #[error("{call}: cuBLASLt status {code} ({name})")]
    Blas {
        /// The cuBLASLt function.
        call: &'static str,
        /// The numeric `cublasStatus_t`.
        code: i32,
        /// The `cublasStatus_t` enumerator name.
        name: String,
    },
    /// No candidate file of a shared library could be loaded.
    #[error("cannot load the {lib} library; tried {tried:?}")]
    LibraryNotFound {
        /// Library base name (`cuda`, `cublasLt`).
        lib: &'static str,
        /// The file names tried, in cudarc's order.
        tried: Vec<String>,
    },
    /// A loaded library lacks a symbol this crate calls.
    #[error("{file} lacks the symbol {symbol}")]
    MissingSymbol {
        /// The library file that was loaded.
        file: String,
        /// The missing symbol.
        symbol: &'static str,
    },
    /// A library or the driver is older or of another major version than the bindings.
    #[error("{what} version {found} is not supported (need {need})")]
    Version {
        /// What was checked.
        what: &'static str,
        /// The version found.
        found: String,
        /// The requirement.
        need: &'static str,
    },
    /// The arena has no room for a request.
    #[error("arena exhausted at {what}: need {need} bytes of {cap}")]
    ArenaExhausted {
        /// Name of the region being taken.
        what: String,
        /// End offset the request would need.
        need: usize,
        /// Arena capacity.
        cap: usize,
    },
    /// A length or range does not fit the buffer it addresses.
    #[error("{what}: {detail}")]
    Bounds {
        /// The operation.
        what: &'static str,
        /// What did not fit.
        detail: String,
    },
    /// The operation cannot be recorded into a stream capture (host copies, event records).
    #[error("{0} is not allowed while the stream is capturing")]
    Capturing(&'static str),
    /// A capture operation was called on a stream that is not capturing.
    #[error("{0} needs a stream that is capturing")]
    NotCapturing(&'static str),
    /// A cubin image or a symbol name is not usable.
    #[error("module: {0}")]
    Module(String),
    /// A graph node index or region does not exist, or the node type cannot be enabled or disabled.
    #[error("graph: {0}")]
    Graph(String),
    /// No pinned cuBLASLt algorithm for a GEMM shape.
    #[error("no pinned cuBLASLt algorithm for {0}")]
    NoPin(String),
    /// cuBLASLt refuses the pinned algorithm for one call shape (no heuristic fallback exists).
    #[error("cuBLASLt refuses the pinned algorithm for {shape} at m={m} (ldx {ldx}, ldo {ldo})")]
    GemmRefused {
        /// The GEMM shape.
        shape: String,
        /// Rows of the call.
        m: u32,
        /// Leading dimension of the activations.
        ldx: u32,
        /// Leading dimension of the output.
        ldo: u32,
    },
    /// The algorithm pin file is malformed or does not match this library.
    #[error("gemm algorithm pins: {0}")]
    Pins(String),
    /// An NVML call failed.
    #[error("NVML {call}: {message}")]
    Nvml {
        /// The NVML operation.
        call: &'static str,
        /// NVML's message.
        message: String,
    },
}

/// Maps a cudarc driver error of `call` to [`CudaError::Driver`].
///
/// Uses the enumerator name (no driver call), so it works when the driver library is absent.
pub(crate) fn driver(call: &'static str) -> impl FnOnce(DriverError) -> CudaError {
    move |e| CudaError::Driver {
        call,
        code: e.0 as i32,
        name: format!("{:?}", e.0),
    }
}

/// Maps a cudarc cuBLASLt error of `call` to [`CudaError::Blas`].
pub(crate) fn blas(call: &'static str) -> impl FnOnce(CublasError) -> CudaError {
    move |e| CudaError::Blas {
        call,
        code: e.0 as i32,
        name: format!("{:?}", e.0),
    }
}

/// Checks a raw driver status returned by a `sys` call.
pub(crate) fn check_driver(call: &'static str, status: cudarc::driver::sys::CUresult) -> Result<()> {
    status.result().map_err(driver(call))
}

/// Checks a raw cuBLASLt status returned by a `sys` call.
pub(crate) fn check_blas(call: &'static str, status: cudarc::cublaslt::sys::cublasStatus_t) -> Result<()> {
    status.result().map_err(blas(call))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::cublaslt::sys::cublasStatus_t;
    use cudarc::driver::sys::CUresult;

    #[test]
    fn driver_status_keeps_code_and_name() {
        let e = check_driver("cuLaunchKernel", CUresult::CUDA_ERROR_INVALID_VALUE).unwrap_err();
        assert_eq!(
            e,
            CudaError::Driver {
                call: "cuLaunchKernel",
                code: 1,
                name: "CUDA_ERROR_INVALID_VALUE".into()
            }
        );
        assert_eq!(
            e.to_string(),
            "cuLaunchKernel: CUDA driver error 1 (CUDA_ERROR_INVALID_VALUE)"
        );
        assert!(check_driver("cuInit", CUresult::CUDA_SUCCESS).is_ok());
    }

    #[test]
    fn capture_invalidated_maps_to_its_code() {
        let e = check_driver("cuStreamEndCapture", CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED).unwrap_err();
        assert!(matches!(e, CudaError::Driver { code: 901, .. }), "{e}");
    }

    #[test]
    fn blas_status_keeps_code_and_name() {
        let e = check_blas("cublasLtMatmul", cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED).unwrap_err();
        assert_eq!(
            e,
            CudaError::Blas {
                call: "cublasLtMatmul",
                code: 15,
                name: "CUBLAS_STATUS_NOT_SUPPORTED".into()
            }
        );
        assert!(check_blas("cublasLtCreate", cublasStatus_t::CUBLAS_STATUS_SUCCESS).is_ok());
    }

    #[test]
    fn error_is_send_sync_static() {
        fn assert_bounds<T: std::error::Error + Send + Sync + 'static>() {}
        assert_bounds::<CudaError>();
    }
}
