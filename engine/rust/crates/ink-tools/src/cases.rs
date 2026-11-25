//! `cases.txt`: the S0 golden set as the tools read it (written by `engine/native/tests/gold_prep.py`).
//!
//! Format, whitespace-separated: `CASE id group n_prompt n_tokens n_positions`, then the prompt ids, the golden token
//! ids, and `n_positions` lists `k id...` (the golden top ids of each position; empty for greedy cases).

use std::fs;
use std::path::Path;
use std::str::{FromStr, SplitWhitespace};

use anyhow::{Context, Result, anyhow, ensure};

/// One golden case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    /// Case id, such as `tf-chat-prose`.
    pub id: String,
    /// `tf` (teacher-forced) or `greedy`.
    pub group: String,
    /// Prompt token ids.
    pub prompt: Vec<i32>,
    /// Config B's continuation token ids.
    pub tokens: Vec<i32>,
    /// Golden top ids per position (teacher-forced cases).
    pub top: Vec<Vec<i32>>,
}

struct Words<'t> {
    it: SplitWhitespace<'t>,
    path: &'t Path,
}

impl Words<'_> {
    fn next(&mut self, what: &str) -> Result<&str> {
        self.it
            .next()
            .with_context(|| format!("{}: truncated at {what}", self.path.display()))
    }

    fn int<T: FromStr>(&mut self, what: &str) -> Result<T> {
        let path = self.path;
        let w = self.next(what)?;
        w.parse::<T>()
            .map_err(|_| anyhow!("{}: bad {what} {w:?}", path.display()))
    }

    fn ints(&mut self, n: usize, what: &str) -> Result<Vec<i32>> {
        (0..n).map(|_| self.int(what)).collect()
    }
}

/// Parses the text of a `cases.txt`; `path` is only used in error messages.
///
/// # Errors
///
/// Fails on a missing `CASE` tag, a truncated record or a token that is not an integer.
pub fn parse_cases(text: &str, path: &Path) -> Result<Vec<Case>> {
    let mut w = Words {
        it: text.split_whitespace(),
        path,
    };
    let mut cases = Vec::new();
    while let Some(tag) = w.it.next() {
        ensure!(tag == "CASE", "{}: expected CASE, got {tag:?}", path.display());
        let id = w.next("id")?.to_owned();
        let group = w.next("group")?.to_owned();
        let np: usize = w.int("np")?;
        let nt: usize = w.int("nt")?;
        let npos: usize = w.int("npos")?;
        let prompt = w.ints(np, "prompt")?;
        let tokens = w.ints(nt, "tokens")?;
        let mut top = Vec::with_capacity(npos);
        for _ in 0..npos {
            let k: usize = w.int("top count")?;
            top.push(w.ints(k, "top")?);
        }
        cases.push(Case {
            id,
            group,
            prompt,
            tokens,
            top,
        });
    }
    Ok(cases)
}

/// Reads and parses a `cases.txt`.
///
/// # Errors
///
/// Fails when the file cannot be read or parsed.
pub fn read_cases(path: &Path) -> Result<Vec<Case>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse_cases(&text, path)
}

/// Keeps the cases whose id is in the comma-separated list `only` (all when it is empty).
#[must_use]
pub fn select(cases: Vec<Case>, only: &str) -> Vec<Case> {
    if only.is_empty() {
        return cases;
    }
    let keep: Vec<&str> = only.split(',').collect();
    cases.into_iter().filter(|c| keep.contains(&c.id.as_str())).collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    const TEXT: &str = "CASE tf-a tf 3 2 2\n1 2 3\n4 5\n2 9 8\n1 7\nCASE gr-b greedy 1 1 0\n6\n7\n";

    #[test]
    fn parses_both_groups() {
        let cases = parse_cases(TEXT, Path::new("cases.txt")).expect("parse");
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].prompt, [1, 2, 3]);
        assert_eq!(cases[0].tokens, [4, 5]);
        assert_eq!(cases[0].top, vec![vec![9, 8], vec![7]]);
        assert_eq!(cases[1].group, "greedy");
        assert!(cases[1].top.is_empty());
        assert_eq!(select(cases, "gr-b").len(), 1);
    }

    #[test]
    fn rejects_truncated_input() {
        assert!(parse_cases("CASE x tf 3 0 0\n1 2", Path::new("c")).is_err());
        assert!(parse_cases("NOPE", Path::new("c")).is_err());
    }
}
