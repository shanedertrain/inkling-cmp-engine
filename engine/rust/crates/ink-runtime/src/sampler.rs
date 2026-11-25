//! Raw logprobs over one logit row (`logprobs_mode = raw_logprobs`, as vLLM v1): a log-softmax of the unprocessed
//! logits. Sampling itself runs on the device (`Runtime::sample_host`, `engine/native/src/sample_kernels.cuh`).
//!

use std::cmp::Ordering;

use ink_backend::TokenLogprobs;

/// Index of the largest logit; the first one on ties (as `torch.argmax`). NaN never wins.
fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in logits.iter().enumerate() {
        if x > logits[best] || logits[best].is_nan() {
            best = i;
        }
    }
    best
}

/// Descending by weight, then ascending by token id (a total order, so every sort and select agrees).
fn by_weight_desc(a: &(f64, u32), b: &(f64, u32)) -> Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// Raw logprobs of `token` in `logits`: its log-softmax value, its rank (1 = the largest logit; ties rank together)
/// and the `top` most likely ids with their logprobs (ties in id order).
///
/// # Panics
///
/// When `token` is not an index into `logits`.
#[must_use]
pub fn logprobs(logits: &[f32], token: i32, top: u32) -> TokenLogprobs {
    let t = usize::try_from(token)
        .ok()
        .filter(|&t| t < logits.len())
        .unwrap_or_else(|| panic!("token {token} outside a row of {} logits", logits.len()));
    let max = f64::from(logits[argmax(logits)]);
    let lse = max + logits.iter().map(|&x| (f64::from(x) - max).exp()).sum::<f64>().ln();
    let lp = |x: f32| (f64::from(x) - lse) as f32;
    let x = logits[t];
    let above = logits.iter().filter(|&&y| y > x).count();
    let mut ranked: Vec<(f64, u32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &y)| (f64::from(y), u32::try_from(i).expect("row index fits u32")))
        .collect();
    let k = (top as usize).min(ranked.len());
    if k > 0 && k < ranked.len() {
        ranked.select_nth_unstable_by(k - 1, by_weight_desc);
    }
    ranked.truncate(k);
    ranked.sort_unstable_by(by_weight_desc);
    TokenLogprobs {
        logprob: lp(x),
        rank: u32::try_from(above + 1).expect("rank fits u32"),
        top_ids: ranked
            .iter()
            .map(|&(_, i)| i32::try_from(i).expect("row index fits i32"))
            .collect(),
        top_logprobs: ranked.iter().map(|&(y, _)| (y - lse) as f32).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprobs_are_the_raw_log_softmax() {
        let row = [1.0f32, 3.0, 2.0, 3.0];
        let lp = logprobs(&row, 2, 3);
        let lse = (1f64.exp() + 2.0 * 3f64.exp() + 2f64.exp()).ln();
        assert!((f64::from(lp.logprob) - (2.0 - lse)).abs() < 1e-6, "{lp:?}");
        assert_eq!(lp.rank, 3);
        assert_eq!(lp.top_ids, [1, 3, 2]);
        assert!((f64::from(lp.top_logprobs[0]) - (3.0 - lse)).abs() < 1e-6);
        assert_eq!(logprobs(&row, 1, 0).rank, 1);
        assert!(logprobs(&row, 1, 0).top_ids.is_empty());
        assert_eq!(logprobs(&row, 0, 20).top_ids.len(), 4);
    }
}
