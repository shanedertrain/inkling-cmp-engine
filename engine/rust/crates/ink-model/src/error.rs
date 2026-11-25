//! Errors of the loader and the memory plan.

use std::path::PathBuf;

use ink_cuda::CudaError;
use safetensors::{Dtype, SafeTensorError};

/// Result of this crate.
pub type Result<T, E = ModelError> = std::result::Result<T, E>;

/// A checkpoint that does not match the model, or a device or file failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    /// The checkpoint (or a dump file) could not be opened, mapped or written.
    #[error("{path}: {source}")]
    Io {
        /// File.
        path: PathBuf,
        /// Cause.
        source: std::io::Error,
    },
    /// The safetensors header is malformed.
    #[error("safetensors: {0}")]
    Safetensors(#[from] SafeTensorError),
    /// A tensor the model needs is not in the checkpoint.
    #[error("tensor missing: {0}")]
    Missing(String),
    /// A tensor has another dtype than the loader expects.
    #[error("{name}: dtype {got:?}, expected {want}")]
    Dtype {
        /// Tensor.
        name: String,
        /// Dtype in the checkpoint.
        got: Dtype,
        /// What the loader accepts.
        want: String,
    },
    /// A size, shape or value outside what the model and its kernels accept.
    #[error("{what}: {detail}")]
    Invalid {
        /// What was checked.
        what: &'static str,
        /// What was found.
        detail: String,
    },
    /// A device operation failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),
}

/// Shorthand for [`ModelError::Invalid`].
pub(crate) fn invalid(what: &'static str, detail: impl Into<String>) -> ModelError {
    ModelError::Invalid {
        what,
        detail: detail.into(),
    }
}
