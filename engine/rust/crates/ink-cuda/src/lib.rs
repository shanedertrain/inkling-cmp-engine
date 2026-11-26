//! Safe CUDA layer of the engine over cudarc (docs/RUST-PLAN.md 3.2, package R1): startup checks, the primary
//! context and its stream, timing events, one device arena, cubin modules and kernel launches, CUDA graphs with region
//! marks and node disable, cuBLASLt GEMMs with pinned algorithms, and an NVML sampler.
//!
//! All `unsafe` of the runtime's CUDA use lives here; raw handles and device addresses never leave the crate.
//! Context-bound types are not `Send`: one thread owns the context. Every CUDA object is RAII and holds the context
//! (and, for buffers, modules and graphs, the objects they refer to), so the context is released last.
//!
//! What comes from cudarc's safe layer and what goes to its `result` / `sys` bindings is listed in the README.

mod arena;
mod check;
mod context;
mod error;
mod gemm;
mod graph;
mod module;
mod nvml;
mod pins;
mod trace;

#[cfg(test)]
mod abi;

pub use arena::{ALIGN, Arena, ArenaPlan, DevBuf, DevPtr, Region};
pub use check::{CUBLASLT_MAJOR, MIN_DRIVER_VERSION, StartupReport, startup_checks};
pub use context::{Context, DeviceInfo, Event, Stream, format_uuid};
pub use error::{CudaError, Result};
pub use gemm::{Bf16, Candidate, GemmElem, LtGemm, Selected, Tuned};
pub use graph::{CachedKernelNode, Capture, Graph, GraphNode, NodeKind, NodeState};
pub use module::{Function, KernelArgs, LaunchCfg, Module};
pub use nvml::{GpuInfo, PERIOD_10HZ, Sample, Sampler, Summary, summarize};
pub use pins::{AlgoPin, AlgoPins, Dtype, FORMAT as PINS_FORMAT, GemmShape, PlanClass};
