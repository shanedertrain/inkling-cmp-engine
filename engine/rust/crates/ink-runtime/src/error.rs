//! The errors of the layers below (ink-cuda, ink-model, std) as [`RuntimeError`]s of the right [`ErrorKind`].
//!
//! A driver or cuBLASLt error status is [`ErrorKind::Device`]: the C++ runtime ends the process on every one of
//! them (`CK`, `LT_CK` -> `ink::fatal`), and sticky statuses (illegal address, launch failure, ECC) cannot be told
//! from the others without querying a context that may be dead. Everything the host refuses before a launch
//! (bounds, a missing pin, an algorithm cuBLASLt refuses for a shape, a call in capture) is [`ErrorKind::Request`].

use std::fmt::Display;

use ink_cuda::CudaError;
use ink_model::ModelError;
use ink_rt_api::{ErrorKind, RuntimeError};

/// An error with a [`ErrorKind`]; [`ErrorKind::Request`] unless it says otherwise.
pub(crate) trait Classify: Display {
    fn kind(&self) -> ErrorKind {
        ErrorKind::Request
    }
}

impl Classify for CudaError {
    fn kind(&self) -> ErrorKind {
        match self {
            CudaError::Driver { .. } | CudaError::Blas { .. } => ErrorKind::Device,
            _ => ErrorKind::Request,
        }
    }
}

impl Classify for ModelError {
    fn kind(&self) -> ErrorKind {
        match self {
            ModelError::Cuda(e) => e.kind(),
            _ => ErrorKind::Request,
        }
    }
}

impl Classify for std::num::TryFromIntError {}

/// `e` as a [`RuntimeError`] of its kind.
pub(crate) fn runtime_error<E: Classify>(e: &E) -> RuntimeError {
    match e.kind() {
        ErrorKind::Request => RuntimeError::request(e.to_string()),
        ErrorKind::Device => RuntimeError::device(e.to_string()),
    }
}

/// `r` with its error as a [`RuntimeError`] of its kind.
pub(crate) fn rt<T, E: Classify>(r: Result<T, E>) -> Result<T, RuntimeError> {
    r.map_err(|e| runtime_error(&e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn driver(code: i32, name: &str) -> CudaError {
        CudaError::Driver {
            call: "cuStreamSynchronize",
            code,
            name: name.into(),
        }
    }

    #[test]
    fn driver_and_cublas_statuses_are_device_errors() {
        for e in [
            driver(700, "CUDA_ERROR_ILLEGAL_ADDRESS"),
            driver(719, "CUDA_ERROR_LAUNCH_FAILED"),
            driver(214, "CUDA_ERROR_ECC_UNCORRECTABLE"),
            driver(1, "CUDA_ERROR_INVALID_VALUE"),
            driver(901, "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED"),
            CudaError::Blas {
                call: "cublasLtMatmul",
                code: 13,
                name: "CUBLAS_STATUS_EXECUTION_FAILED".into(),
            },
        ] {
            let r = runtime_error(&e);
            assert_eq!(r.kind(), ErrorKind::Device, "{e}");
            assert_eq!(r.message(), e.to_string());
            assert!(r.to_string().starts_with("runtime device error: "), "{r}");
        }
        let wrapped = runtime_error(&ModelError::Cuda(driver(700, "CUDA_ERROR_ILLEGAL_ADDRESS")));
        assert_eq!(wrapped.kind(), ErrorKind::Device);
    }

    #[test]
    fn host_refusals_are_request_errors() {
        for e in [
            CudaError::Bounds {
                what: "htod",
                detail: "16 bytes into 8".into(),
            },
            CudaError::Capturing("dtoh"),
            CudaError::NoPin("n=8 k=8".into()),
            CudaError::GemmRefused {
                shape: "n=8 k=8".into(),
                m: 3,
                ldx: 8,
                ldo: 8,
            },
            CudaError::ArenaExhausted {
                what: "x".into(),
                need: 2,
                cap: 1,
            },
        ] {
            let r = runtime_error(&e);
            assert_eq!(r.kind(), ErrorKind::Request, "{e}");
            assert!(r.to_string().starts_with("runtime error: "), "{r}");
        }
        assert_eq!(
            runtime_error(&u8::try_from(300_i32).unwrap_err()).kind(),
            ErrorKind::Request
        );
    }

    #[test]
    fn a_capture_failure_becomes_a_device_error() {
        let e = RuntimeError::request("graph capture: refused GEMM").into_device();
        assert_eq!(e.kind(), ErrorKind::Device);
        assert_eq!(e.message(), "graph capture: refused GEMM");
    }
}
