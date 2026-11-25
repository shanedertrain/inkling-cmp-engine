//! Build identity and benchmark records (`ink-record/1`).
//!
//! Every record names the engine, its version, the git commit and whether the tree was dirty
//! (`docs/RUST-PLAN.md` section 6, schema in `RECORDS.md`). [`RecordWriter`] refuses to write a record whose
//! identity is dirty or missing unless the caller passes [`DirtyPolicy::Allow`], and such a record goes to a
//! `.dirty.jsonl` file.
#![forbid(unsafe_code)]

mod identity;
mod writer;

pub use identity::{Component, ComponentName, Engine, Identity, IdentityError, Source};
pub use writer::{Context, DirtyPolicy, Gpu, RecordError, RecordWriter, SCHEMA};

/// Identity of this build as one line: `rust version=X commit=<40 hex> dirty=<0|1> diff=<hex|->`.
///
/// Printed by every binary's `--version`, as the first log line of a run, and served at `/version`.
pub const BUILD_LINE: &str = concat!(
    "rust version=",
    env!("CARGO_PKG_VERSION"),
    " commit=",
    env!("INK_BUILD_COMMIT"),
    " dirty=",
    env!("INK_BUILD_DIRTY"),
    " diff=",
    env!("INK_BUILD_DIFF"),
);
