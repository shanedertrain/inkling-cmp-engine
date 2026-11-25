//! The runtime side of serving: [`Engine`], the Rust forward pass (package R5: the C++ engine's eager forward over
//! the kernel pack, bit for bit); [`RuntimeAdapter`], the scheduler's `ModelRuntime` over any forward
//! [`ink_rt_api::Runtime`] (package I1) with speculative rounds over its DSpark drafter (package I2); host sampling in
//! [`sampler`]; [`RustDrafter`], the DSpark drafter (package R8: the C++ `ink::DraftRT` over the kernel pack, bit for
//! bit). Graphs (R6) and the speculation loop (R9) join later.
//!
//! `unsafe` is denied except in the two launch modules `engine::ops` and `draft::ops`, which call the generated
//! kernel launchers (review R3 deviation 4).
#![deny(unsafe_code)]

mod adapter;
mod devbufs;
mod dflash2;
mod draft;
mod engine;
mod error;
pub mod lookup;
pub mod sampler;
mod spec;

pub use adapter::RuntimeAdapter;
pub use dflash2::RustDFlash2;
pub use draft::{DraftTap, RustDrafter};
pub use engine::{Engine, planned_shapes};
