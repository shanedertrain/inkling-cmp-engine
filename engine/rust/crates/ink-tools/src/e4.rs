//! E4 of the oracle record set (`docs/RUST-PLAN.md` 7.3): one JSON line per logit row of the golden set with the
//! FNV-1a 64 of its 200,058 valid floats, the first rows in full, and the greedy token streams. Written by
//! `ink-oracle record` and by `ink-gold --record-dir`; the format is the oracle's
//! (`engine/native/capi/README.md`, "Oracle record set").

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Result;
use ink_rt_api::LOGIT_ROW;

use crate::logits::{argmax_first, row_hash};

/// Full rows kept beside the hashes.
pub const FULL_ROWS: usize = 16;

/// Writer of `e4-logits.jsonl` and `e4-logits-full.f32`.
#[derive(Debug)]
pub struct E4Rows {
    rows: BufWriter<File>,
    full: BufWriter<File>,
    nfull: usize,
}

impl E4Rows {
    /// Creates both files.
    ///
    /// # Errors
    ///
    /// Fails when a file cannot be created.
    pub fn create(logits: &Path, full: &Path) -> Result<Self> {
        Ok(Self {
            rows: BufWriter::new(File::create(logits)?),
            full: BufWriter::new(File::create(full)?),
            nfull: 0,
        })
    }

    /// Records one row (`LOGIT_ROW` floats): case id, index `k` within the case, position, argmax and hash. Returns
    /// the argmax.
    ///
    /// # Errors
    ///
    /// Fails when a write fails.
    pub fn row(&mut self, id: &str, k: usize, pos: usize, row: &[f32]) -> Result<usize> {
        let a = argmax_first(row);
        let full = if self.nfull < FULL_ROWS {
            for x in &row[..LOGIT_ROW] {
                self.full.write_all(&x.to_le_bytes())?;
            }
            self.nfull += 1;
            format!(",\"full\":{}", self.nfull - 1)
        } else {
            String::new()
        };
        writeln!(
            self.rows,
            "{{\"case\":\"{id}\",\"k\":{k},\"pos\":{pos},\"argmax\":{a},\"fnv1a64\":\"{:016x}\"{full}}}",
            row_hash(row)
        )?;
        Ok(a)
    }

    /// Flushes both files and returns the number of full rows written.
    ///
    /// # Errors
    ///
    /// Fails when a flush fails.
    pub fn finish(mut self) -> Result<usize> {
        self.rows.flush()?;
        self.full.flush()?;
        Ok(self.nfull)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_hashed_and_the_first_are_kept_in_full() {
        let dir = std::env::temp_dir().join(format!("ink-e4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let (l, f) = (dir.join("l.jsonl"), dir.join("f.f32"));
        let mut w = E4Rows::create(&l, &f).expect("create");
        let mut row = vec![0.0_f32; LOGIT_ROW];
        row[5] = 2.0;
        for k in 0..=FULL_ROWS {
            assert_eq!(w.row("c", k, 10 + k, &row).expect("row"), 5);
        }
        assert_eq!(w.finish().expect("finish"), FULL_ROWS);
        let text = std::fs::read_to_string(&l).expect("read");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), FULL_ROWS + 1);
        assert!(lines[0].starts_with("{\"case\":\"c\",\"k\":0,\"pos\":10,\"argmax\":5,\"fnv1a64\":\""));
        assert!(lines[0].ends_with(",\"full\":0}"));
        assert!(lines[FULL_ROWS].ends_with('}') && !lines[FULL_ROWS].contains("full"));
        assert_eq!(
            std::fs::metadata(&f).expect("meta").len(),
            (FULL_ROWS * LOGIT_ROW * 4) as u64
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
