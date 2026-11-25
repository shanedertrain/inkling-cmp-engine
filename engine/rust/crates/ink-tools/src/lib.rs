//! Shared code of the tool binaries of `ink-tools`: the golden-set tools `ink-gold`, `ink-inv`, `ink-optest` and the
//! oracle recorder `ink-oracle` (packages T1 and A1) drive a [`Runtime`](ink_rt_api::Runtime) through these helpers.
//! The front-end tools (`ink-textcli`, `ink-replay`) do not use them. [`draftlog`]: the drafter log of `ink-spec
//! --draft-log` and `ink-draft` (package R8).
#![forbid(unsafe_code)]

pub mod cases;
pub mod draftlog;
pub mod e4;
pub mod engine;
pub mod fmt;
pub mod line;
pub mod logits;
pub mod session;
pub mod tele;
