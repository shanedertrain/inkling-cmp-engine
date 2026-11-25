//! Row statistics over fp32 logit rows: hashes, argmax, fp64 log-softmax and the top-k lists of the golden tools.
//! All of them look at the first [`VOCAB`] entries of a row only (the rest of a [`LOGIT_ROW`] is padding).

use ink_rt_api::VOCAB;

use crate::fmt;

/// FNV-1a 64 offset basis.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Folds `bytes` into the FNV-1a 64 state `h`.
#[must_use]
pub fn fnv1a64(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// FNV-1a 64 over the little-endian bytes of the valid part of a logit row.
#[must_use]
pub fn row_hash(row: &[f32]) -> u64 {
    row[..VOCAB]
        .iter()
        .fold(FNV_OFFSET, |h, x| fnv1a64(&x.to_le_bytes(), h))
}

/// First index of the maximum over the valid part; NaN never wins (it counts as minus infinity).
#[must_use]
pub fn argmax_first(row: &[f32]) -> usize {
    let mut a = 0;
    for (k, &x) in row[..VOCAB].iter().enumerate().skip(1) {
        if x > row[a] {
            a = k;
        }
    }
    a
}

/// fp64 log-softmax of the valid part of a row, in a reused buffer.
#[derive(Debug, Default)]
pub struct LogSoftmax {
    lp: Vec<f64>,
}

impl LogSoftmax {
    /// Computes `row[k] - log(sum(exp(row)))` in fp64, summing in index order.
    pub fn compute(&mut self, row: &[f32]) -> &[f64] {
        let row = &row[..VOCAB];
        let mx = row.iter().fold(-1e300_f64, |m, &x| m.max(f64::from(x)));
        let z: f64 = row.iter().map(|&x| (f64::from(x) - mx).exp()).sum();
        let l = z.ln() + mx;
        self.lp.clear();
        self.lp.extend(row.iter().map(|&x| f64::from(x) - l));
        &self.lp
    }
}

/// The `k` largest entries of `lp` as `(index, value)`, largest first, ties by smaller index first.
#[must_use]
pub fn top_k(lp: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut best: Vec<(usize, f64)> = Vec::with_capacity(k + 1);
    for (i, &v) in lp.iter().enumerate() {
        // Indices arrive in ascending order, so an equal value never beats an entry already kept.
        if best.len() == k && !best.last().is_some_and(|&(_, w)| v > w) {
            continue;
        }
        let at = best.partition_point(|&(_, w)| w >= v);
        best.insert(at, (i, v));
        best.truncate(k);
    }
    best
}

/// `[[index,value],...]` of the top `k`, values printed as `%.9g`.
#[must_use]
pub fn top_json(lp: &[f64], k: usize) -> String {
    let items: Vec<String> = top_k(lp, k)
        .iter()
        .map(|&(i, v)| format!("[{i},{}]", fmt::g(v, 9)))
        .collect();
    format!("[{}]", items.join(","))
}

/// One teacher-forced position: log-probabilities at the golden top ids, the argmax of the logits and the top 20.
#[must_use]
pub fn position_json(row: &[f32], lp: &[f64], gold_ids: &[i32]) -> String {
    let at: Vec<String> = gold_ids
        .iter()
        .map(|&id| {
            let v = usize::try_from(id)
                .ok()
                .and_then(|i| lp.get(i))
                .copied()
                .unwrap_or(f64::NAN);
            format!("[{id},{}]", fmt::g(v, 9))
        })
        .collect();
    format!(
        "{{\"lq_at\":[{}],\"argmax\":{},\"top20\":{}}}",
        at.join(","),
        argmax_first(row),
        top_json(lp, 20)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(f: impl Fn(usize) -> f32) -> Vec<f32> {
        (0..ink_rt_api::LOGIT_ROW).map(f).collect()
    }

    #[test]
    fn argmax_takes_the_first_maximum_and_ignores_nan() {
        let mut r = row(|_| 0.0);
        r[7] = 3.0;
        r[9] = 3.0;
        r[1] = f32::NAN;
        assert_eq!(argmax_first(&r), 7);
        // the padding after VOCAB is never looked at
        r[VOCAB + 1] = 100.0;
        assert_eq!(argmax_first(&r), 7);
    }

    #[test]
    fn log_softmax_sums_to_one() {
        let r = row(|i| (i % 17) as f32 * 0.25);
        let mut ls = LogSoftmax::default();
        let lp = ls.compute(&r);
        let total: f64 = lp.iter().map(|v| v.exp()).sum();
        assert!((total - 1.0).abs() < 1e-9, "{total}");
    }

    #[test]
    fn top_k_is_sorted_with_index_tiebreak() {
        let lp = [0.5, 2.0, 2.0, -1.0, 2.0, 0.5];
        assert_eq!(top_k(&lp, 4), vec![(1, 2.0), (2, 2.0), (4, 2.0), (0, 0.5)]);
        assert_eq!(top_k(&lp, 10).len(), 6);
        assert_eq!(top_json(&lp, 2), "[[1,2],[2,2]]");
    }

    #[test]
    fn row_hash_is_fnv_over_le_bytes() {
        let r = row(|_| 1.0);
        let one = 1.0_f32.to_le_bytes();
        let mut h = FNV_OFFSET;
        for _ in 0..VOCAB {
            h = fnv1a64(&one, h);
        }
        assert_eq!(row_hash(&r), h);
        assert_eq!(fnv1a64(b"", FNV_OFFSET), FNV_OFFSET);
        assert_eq!(fnv1a64(b"a", FNV_OFFSET), 0xaf63_dc4c_8601_ec8c);
    }
}
