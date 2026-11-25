//! Inkling-2-Mini on the device (package R4 of `docs/RUST-PLAN.md`): dimensions and visit plan, the checkpoint by
//! mmap, small parameters to fp32, expert repack, upload into the arena, and the memory plan of `Engine::alloc`, all
//! in the C++ engine's order (native 0.2.0 @ 66fe353) so that gate E1 compares one dump line by line; and the DSpark
//! drafter's files on the host ([`drafter`], package R8).
//!
//! `unsafe` appears once, for the weight mmap ([`Checkpoint::open`]); the crate denies it everywhere else.
#![deny(unsafe_code)]

mod checkpoint;
pub mod dflash2;
pub mod dims;
pub mod drafter;
pub mod dump;
mod error;
pub mod plan;
pub mod repack;
mod weights;

pub use checkpoint::{CHECKPOINT_FILE, Checkpoint, Tensor};
pub use error::{ModelError, Result};
pub use plan::{Activations, Alloc, GateTailW, MemConfig};
pub use weights::{
    Controller, ExpertFmt, ExpertW, LayerW, WEIGHTS_BYTES_ESTIMATE, WeightRegion, Weights, interleave_msg_queries,
    nvfp4_factors, transpose_gate,
};
