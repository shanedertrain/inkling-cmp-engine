//! A real device fault for the served fail-fast test (review V1), compiled only with the `fault-inject` feature (never
//! in a release build of the server or the tools).
//!
//! With `INK_FAULT_AT_FORWARD=N` in the environment, the `N`-th forward call (1-based) overwrites its first device
//! token id with `i32::MAX` after the upload, past the host's vocabulary check: the production embedding kernel
//! then reads about 9.9 TB past the embedding table, a sticky `CUDA_ERROR_ILLEGAL_ADDRESS` that poisons the context,
//! exactly the failure the runtime must report as [`ink_rt_api::ErrorKind::Device`].

use ink_cuda::{DevBuf, Stream};

use super::{Res, Rt};

/// The forward call to break, and the calls made so far.
#[derive(Debug, Default)]
pub(super) struct Fault {
    at: Option<u64>,
    calls: u64,
}

impl Fault {
    /// Reads `INK_FAULT_AT_FORWARD`; unset or unparsable: no fault.
    pub fn from_env() -> Self {
        let at = std::env::var("INK_FAULT_AT_FORWARD").ok().and_then(|v| v.parse().ok());
        if let Some(n) = at {
            eprintln!("[engine] fault-inject: forward call {n} will read past the embedding table");
        }
        Self { at, calls: 0 }
    }

    /// Called after a forward call's ids are on the device: on the chosen call, corrupts the first one.
    pub fn after_upload(&mut self, st: &Stream, ids: &DevBuf<i32>) -> Res<()> {
        self.calls += 1;
        if self.at == Some(self.calls) {
            eprintln!(
                "[engine] fault-inject: corrupting token id 0 of forward call {}",
                self.calls
            );
            st.htod(&ids.slice(0..1).rt()?, &[i32::MAX]).rt()?;
        }
        Ok(())
    }
}
