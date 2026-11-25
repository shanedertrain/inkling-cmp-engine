//! Host-side incremental n-gram lookup index for lossless verbatim copy drafts (`m = 8` rule,
//! `bench/raw/2026-11-23-lookup-drafts/DESIGN.md`).
//!
//! Per session: the committed token stream (`prompt + committed emitted tokens`) and maps `k-gram -> most recent end
//! position` for `k = 5, 4, 3` (splitmix64 hash, verified against the stream on a hit), plus a per-position previous-hit
//! chain so slot rollbacks (`truncate`) run in `O(rolled-back tokens)` without rebuilding the maps.

use std::collections::HashMap;

/// N-gram lengths checked in order (longest first, matching `lookup.hpp` and the desk study).
const KS: [usize; 3] = [5, 4, 3];

/// Maximum backward verbatim match length scanned when scoring a hit.
const BACK_CAP: usize = 64;

/// Maximum hash-collision / self-match chain steps walked per lookup.
const MAX_CHAIN: usize = 8;

/// Sentinel for "no earlier occurrence" in `prev`.
const NONE: usize = usize::MAX;

/// Splitmix64 hash over `toks` (`k` tokens), matching `ink::NgramIndex::hash`.
#[inline]
fn ngram_hash(toks: &[i32]) -> u64 {
    let mut h = 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(toks.len() as u64);
    for &t in toks {
        let term = u64::from(t as u32)
            .wrapping_add(0x9e37_79b9_7f4a_7c15_u64)
            .wrapping_add(h << 6)
            .wrapping_add(h >> 2);
        h ^= term;
        h ^= h >> 31;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9_u64);
        h ^= h >> 27;
    }
    h
}

/// Incremental per-slot n-gram index over committed tokens.
#[derive(Debug, Default, Clone)]
pub struct NgramIndex {
    tokens: Vec<i32>,
    /// Per position `i`: previous end index with the same `KS[q]`-gram hash, or `NONE`.
    prev: Vec<[usize; 3]>,
    /// For each `q` (`KS[q]`): hash -> most recent end index in `tokens`.
    last: [HashMap<u64, usize>; 3],
}

impl NgramIndex {
    /// Empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of committed tokens currently indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the index has no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Clears the index (slot reset).
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev.clear();
        for m in &mut self.last {
            m.clear();
        }
    }

    /// Appends one committed token (`O(1)` amortized).
    pub fn push(&mut self, tok: i32) {
        let i = self.tokens.len();
        self.tokens.push(tok);
        let mut p = [NONE; 3];
        for (q, &k) in KS.iter().enumerate() {
            if i + 1 < k {
                continue;
            }
            let h = ngram_hash(&self.tokens[i + 1 - k..=i]);
            if let Some(old) = self.last[q].insert(h, i) {
                p[q] = old;
            }
        }
        self.prev.push(p);
    }

    /// Appends a slice of committed tokens.
    pub fn extend_from_slice(&mut self, toks: &[i32]) {
        self.tokens.reserve(toks.len());
        self.prev.reserve(toks.len());
        for &t in toks {
            self.push(t);
        }
    }

    /// Rolls the index back to `new_len` tokens (`O(old_len - new_len)`).
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.tokens.len() {
            return;
        }
        if new_len == 0 {
            self.clear();
            return;
        }
        while self.tokens.len() > new_len {
            let i = self.tokens.len() - 1;
            let p = self.prev.pop().expect("prev matches tokens");
            for (q, &k) in KS.iter().enumerate() {
                if i + 1 < k {
                    continue;
                }
                let h = ngram_hash(&self.tokens[i + 1 - k..=i]);
                if p[q] == NONE {
                    self.last[q].remove(&h);
                } else {
                    self.last[q].insert(h, p[q]);
                }
            }
            self.tokens.pop();
        }
    }

    /// Queries the virtual stream `self.tokens + extra` for the most recent earlier occurrence of its trailing
    /// `k`-gram (`k = 5, 4, 3`). Copies up to `out.len()` tokens that followed that occurrence into `out` and returns
    /// `(proposed_count, backward_match_len)`.
    pub fn propose(&self, extra: &[i32], out: &mut [i32]) -> (usize, usize) {
        let tlen = self.tokens.len();
        let n = tlen + extra.len();
        let at = |idx: usize| -> i32 {
            if idx < tlen {
                self.tokens[idx]
            } else {
                extra[idx - tlen]
            }
        };
        let mut key = [0i32; 5];
        for (q, &k) in KS.iter().enumerate() {
            if n < k + 1 {
                continue;
            }
            for (j, slot) in key[..k].iter_mut().enumerate() {
                *slot = at(n - k + j);
            }
            let h = ngram_hash(&key[..k]);
            let Some(&start_j) = self.last[q].get(&h) else {
                continue;
            };
            let mut j = start_j;
            let mut steps = 0;
            while j != NONE && steps < MAX_CHAIN {
                steps += 1;
                // Skip the suffix itself when extra is empty.
                if j + 1 < n && j + 1 >= k && self.tokens[j + 1 - k..=j] == key[..k] {
                    let mut back = 0;
                    let limit = BACK_CAP.min(j + 1);
                    while back < limit && at(j - back) == at(n - 1 - back) {
                        back += 1;
                    }
                    let count = (n - 1 - j).min(out.len());
                    for (c, dst) in out[..count].iter_mut().enumerate() {
                        *dst = at(j + 1 + c);
                    }
                    return (count, back);
                }
                j = self.prev[j][q];
            }
        }
        (0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_most_recent_verbatim_match_and_measures_back_length() {
        let mut idx = NgramIndex::new();
        // Repeated 10-token span [10..20] followed by [100..108], then some noise, then [10..12]
        let mut toks: Vec<i32> = (10..20).collect();
        toks.extend(100..108);
        toks.extend([999, 998, 997]);
        toks.extend(10..12);
        idx.extend_from_slice(&toks);

        // extra = [anchor=12, d0..d6 = 13..19] -> virtual stream ends with [10..19] (10-token verbatim match!)
        let extra = [12, 13, 14, 15, 16, 17, 18, 19];
        let mut prop = [-1i32; 8];
        let (cnt, back) = idx.propose(&extra, &mut prop);
        assert_eq!(cnt, 8);
        assert_eq!(back, 10);
        assert_eq!(prop, [100, 101, 102, 103, 104, 105, 106, 107]);
    }

    #[test]
    fn truncate_restores_index_exactly() {
        let mut a = NgramIndex::new();
        let mut b = NgramIndex::new();
        let base: Vec<i32> = (0..50).collect();
        a.extend_from_slice(&base);
        b.extend_from_slice(&base);
        // Append extra tokens to `a`, including repeats of base n-grams, then truncate back to 50.
        a.extend_from_slice(&[10, 11, 12, 13, 14, 777, 778, 779]);
        a.truncate(50);
        assert_eq!(a.len(), 50);
        let extra = [10, 11, 12, 13, 14];
        let (mut pa, mut pb) = ([-1; 8], [-1; 8]);
        assert_eq!(a.propose(&extra, &mut pa), b.propose(&extra, &mut pb));
        assert_eq!(pa, pb);
        assert_eq!(pa[0], 15);
    }

    #[test]
    fn short_match_below_gate_reports_exact_back_length() {
        let mut idx = NgramIndex::new();
        idx.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 52, 53]);
        // Only last 5 tokens [4,5,6,7,8] match -> back == 5 (< 8)
        let extra = [90, 91, 92, 4, 5, 6, 7, 8];
        let mut prop = [-1; 8];
        let (cnt, back) = idx.propose(&extra, &mut prop);
        assert_eq!(back, 5);
        assert_eq!(cnt, 8);
        assert_eq!(&prop[..4], &[50, 51, 52, 53]);
    }
}
