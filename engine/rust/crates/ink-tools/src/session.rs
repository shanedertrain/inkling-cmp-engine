//! Calls shared by the tools: chunked prefill of one slot, and logit-row access.

use anyhow::{Result, ensure};
use ink_rt_api::{LOGIT_ROW, Runtime, SegIn};

/// Rows per prefill call of the golden tools (the engine's segment limit).
pub const CHUNK: usize = 2048;

/// Result of [`prefill`].
#[derive(Debug, Default)]
pub struct Prefilled {
    /// Logit row of the last prompt row (`LOGIT_ROW` floats); empty when none was asked for.
    pub last: Vec<f32>,
    /// Forward calls made.
    pub calls: usize,
}

/// Resets `slot` and prefills `ids` in calls of `chunk` rows, committing each; the last call returns the logits of
/// the last row when `want_logits`.
///
/// # Errors
///
/// Fails when `chunk` is 0 or a runtime call fails.
pub fn prefill(rt: &mut dyn Runtime, slot: usize, ids: &[i32], chunk: usize, want_logits: bool) -> Result<Prefilled> {
    ensure!(chunk > 0, "prefill chunk must be positive");
    rt.reset_slot(slot)?;
    let mut out = Prefilled::default();
    for (i, piece) in ids.chunks(chunk).enumerate() {
        let want = usize::from((i + 1) * chunk >= ids.len() && want_logits);
        let n = rt.forward(&[SegIn {
            slot,
            ids: piece,
            logits: want,
        }])?;
        if want == 1 {
            rt.logits_host(n, &mut out.last)?;
        }
        rt.commit(&[u32::try_from(piece.len())?])?;
        out.calls += 1;
    }
    Ok(out)
}

/// Row `r` of a `logits_host` buffer.
///
/// # Panics
///
/// Panics when the buffer has fewer than `r + 1` rows.
#[must_use]
pub fn row(buf: &[f32], r: usize) -> &[f32] {
    &buf[r * LOGIT_ROW..(r + 1) * LOGIT_ROW]
}
