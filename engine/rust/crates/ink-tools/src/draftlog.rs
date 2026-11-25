//! The drafter log (package R8, gate E8): every input a speculation loop hands its drafter, in call order, with the
//! drafts the drafter returned, so that another drafter can be fed the same committed rows, features and states and
//! its drafts compared round by round. Written by `ink-spec --draft-log` (the C++ drafter through the C ABI), read by
//! `ink-draft replay` (the Rust drafter) and the C++ replay tool of `bench/raw/2026-11-23-rust-R8-drafter/tools/`.
//!
//! Format (little-endian): the 8-byte magic `INKDLOG1`, then `u32` sessions, max context, int8 head (0/1), E4M3
//! global-layer K/V (0/1); then records, each one byte of kind and its payload:
//!
//! | Kind | Payload |
//! |---|---|
//! | `C` commit_rows | `u32` n, n feature rows (`[n][4][2304]` bf16), n `i32` positions, n `i32` sessions |
//! | `F` frnn | `sessions` feature rows |
//! | `S` set_state | `sessions` `i32` anchors, `sessions` `i32` positions |
//! | `D` draft | the drafts the recording drafter returned: `sessions x 7` `i32` |
//! | `B` draft_sessions | `u32` k, k `i32` sessions (distinct), then the drafts as `D` (only the k sessions' rows are new) |

use std::io::{self, Read, Write};

use anyhow::{Context, Result, bail, ensure};
use ink_rt_api::{DRAFTS, FEAT_ROW_BYTES};

/// The file magic.
pub const MAGIC: &[u8; 8] = b"INKDLOG1";

/// The drafter the log was recorded with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Sessions (drafter rows).
    pub sessions: u32,
    /// Positions per session.
    pub max_ctx: u32,
    /// int8 head.
    pub int8: bool,
    /// E4M3 global-layer K/V.
    pub kv8: bool,
}

/// One drafter call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `commit_rows`: features (`n` rows of [`FEAT_ROW_BYTES`]), positions and sessions (`n` each).
    Commit {
        /// Feature rows, raw bytes.
        feats: Vec<u8>,
        /// Positions.
        pos: Vec<i32>,
        /// Sessions.
        sess: Vec<i32>,
    },
    /// `frnn`: one feature row per session.
    Frnn(Vec<u8>),
    /// `set_state`: anchor tokens and positions.
    State {
        /// Anchor token per session.
        anchor: Vec<i32>,
        /// Anchor position per session.
        pos: Vec<i32>,
    },
    /// `draft`, with the recording drafter's drafts (`sessions x 7`).
    Draft(Vec<i32>),
    /// `draft_sessions`: the drafted sessions and the recording drafter's drafts (`sessions x 7`; the rows of the other
    /// sessions are whatever the drafter held).
    DraftSessions {
        /// The drafted sessions.
        sessions: Vec<i32>,
        /// Every session's drafts after the call.
        drafts: Vec<i32>,
    },
}

fn put_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn put_i32s(w: &mut impl Write, v: &[i32]) -> io::Result<()> {
    for x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn u32_of(n: usize) -> Result<u32> {
    u32::try_from(n).context("row count beyond u32")
}

/// Writes a drafter log.
#[derive(Debug)]
pub struct LogWriter<W: Write> {
    w: W,
    h: Header,
}

impl<W: Write> LogWriter<W> {
    /// Writes the magic and the header.
    ///
    /// # Errors
    ///
    /// The I/O error.
    pub fn new(mut w: W, h: Header) -> Result<Self> {
        w.write_all(MAGIC)?;
        for v in [h.sessions, h.max_ctx, u32::from(h.int8), u32::from(h.kv8)] {
            put_u32(&mut w, v)?;
        }
        Ok(Self { w, h })
    }

    /// Appends one call.
    ///
    /// # Errors
    ///
    /// Sizes that do not match the header or each other, or the I/O error.
    pub fn op(&mut self, op: &Op) -> Result<()> {
        let s = self.h.sessions as usize;
        match op {
            Op::Commit { feats, pos, sess } => {
                let n = pos.len();
                ensure!(
                    sess.len() == n && feats.len() == n * FEAT_ROW_BYTES,
                    "commit record sizes"
                );
                self.w.write_all(b"C")?;
                put_u32(&mut self.w, u32_of(n)?)?;
                self.w.write_all(feats)?;
                put_i32s(&mut self.w, pos)?;
                put_i32s(&mut self.w, sess)?;
            }
            Op::Frnn(feats) => {
                ensure!(feats.len() == s * FEAT_ROW_BYTES, "frnn record size");
                self.w.write_all(b"F")?;
                self.w.write_all(feats)?;
            }
            Op::State { anchor, pos } => {
                ensure!(anchor.len() == s && pos.len() == s, "state record sizes");
                self.w.write_all(b"S")?;
                put_i32s(&mut self.w, anchor)?;
                put_i32s(&mut self.w, pos)?;
            }
            Op::Draft(d) => {
                ensure!(d.len() == s * DRAFTS, "draft record size");
                self.w.write_all(b"D")?;
                put_i32s(&mut self.w, d)?;
            }
            Op::DraftSessions { sessions, drafts } => {
                ensure!(
                    !sessions.is_empty() && sessions.len() <= s && drafts.len() == s * DRAFTS,
                    "draft_sessions record sizes"
                );
                self.w.write_all(b"B")?;
                put_u32(&mut self.w, u32_of(sessions.len())?)?;
                put_i32s(&mut self.w, sessions)?;
                put_i32s(&mut self.w, drafts)?;
            }
        }
        Ok(())
    }

    /// Flushes and returns the writer.
    ///
    /// # Errors
    ///
    /// The I/O error.
    pub fn finish(mut self) -> Result<W> {
        self.w.flush()?;
        Ok(self.w)
    }
}

/// Reads a drafter log.
#[derive(Debug)]
pub struct LogReader<R: Read> {
    r: R,
    /// The header.
    pub header: Header,
}

fn get_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn get_bytes(r: &mut impl Read, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    r.read_exact(&mut v).context("truncated drafter log")?;
    Ok(v)
}

fn get_i32s(r: &mut impl Read, n: usize) -> Result<Vec<i32>> {
    Ok(get_bytes(r, n * 4)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect())
}

impl<R: Read> LogReader<R> {
    /// Reads and checks the magic and the header.
    ///
    /// # Errors
    ///
    /// A wrong magic, a flag other than 0 or 1, or the I/O error.
    pub fn new(mut r: R) -> Result<Self> {
        let mut m = [0u8; 8];
        r.read_exact(&mut m).context("drafter log header")?;
        ensure!(&m == MAGIC, "not a drafter log (magic {m:?})");
        let v = [get_u32(&mut r)?, get_u32(&mut r)?, get_u32(&mut r)?, get_u32(&mut r)?];
        ensure!(v[2] <= 1 && v[3] <= 1 && v[0] > 0, "drafter log header {v:?}");
        Ok(Self {
            r,
            header: Header {
                sessions: v[0],
                max_ctx: v[1],
                int8: v[2] == 1,
                kv8: v[3] == 1,
            },
        })
    }

    /// The next call, or `None` at the end of the log.
    ///
    /// # Errors
    ///
    /// An unknown record kind, a truncated record, or the I/O error.
    pub fn next_op(&mut self) -> Result<Option<Op>> {
        let mut k = [0u8; 1];
        match self.r.read(&mut k)? {
            0 => return Ok(None),
            1 => {}
            _ => unreachable!("read into one byte"),
        }
        let s = self.header.sessions as usize;
        let r = &mut self.r;
        Ok(Some(match k[0] {
            b'C' => {
                let n = get_u32(r)? as usize;
                let feats = get_bytes(r, n * FEAT_ROW_BYTES)?;
                let pos = get_i32s(r, n)?;
                let sess = get_i32s(r, n)?;
                Op::Commit { feats, pos, sess }
            }
            b'F' => Op::Frnn(get_bytes(r, s * FEAT_ROW_BYTES)?),
            b'S' => {
                let anchor = get_i32s(r, s)?;
                let pos = get_i32s(r, s)?;
                Op::State { anchor, pos }
            }
            b'D' => Op::Draft(get_i32s(r, s * DRAFTS)?),
            b'B' => {
                let k = get_u32(r)? as usize;
                ensure!(k > 0 && k <= s, "draft_sessions record of {k} sessions");
                let sessions = get_i32s(r, k)?;
                let drafts = get_i32s(r, s * DRAFTS)?;
                Op::DraftSessions { sessions, drafts }
            }
            other => bail!("unknown drafter log record {other:#x}"),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_reads_back_as_written() {
        let h = Header {
            sessions: 2,
            max_ctx: 1000,
            int8: true,
            kv8: false,
        };
        let ops = vec![
            Op::Commit {
                feats: (0..2 * FEAT_ROW_BYTES).map(|i| (i % 251) as u8).collect(),
                pos: vec![0, 1],
                sess: vec![1, 1],
            },
            Op::Commit {
                feats: Vec::new(),
                pos: Vec::new(),
                sess: Vec::new(),
            },
            Op::State {
                anchor: vec![7, -1],
                pos: vec![2, 0],
            },
            Op::Frnn(vec![3; 2 * FEAT_ROW_BYTES]),
            Op::Draft((0..14).collect()),
            Op::DraftSessions {
                sessions: vec![1],
                drafts: (20..34).collect(),
            },
        ];
        let mut w = LogWriter::new(Vec::new(), h).unwrap();
        for op in &ops {
            w.op(op).unwrap();
        }
        let bytes = w.finish().unwrap();
        let mut r = LogReader::new(bytes.as_slice()).unwrap();
        assert_eq!(r.header, h);
        let mut back = Vec::new();
        while let Some(op) = r.next_op().unwrap() {
            back.push(op);
        }
        assert_eq!(back, ops);
        // a truncated record is an error, not an end
        let mut r = LogReader::new(&bytes[..bytes.len() - 1]).unwrap();
        let mut err = false;
        for _ in 0..ops.len() {
            if r.next_op().is_err() {
                err = true;
                break;
            }
        }
        assert!(err);
        assert!(
            LogWriter::new(Vec::new(), h)
                .unwrap()
                .op(&Op::Draft(vec![0; 7]))
                .is_err()
        );
        assert!(LogReader::new(&b"INKDLOG2\0\0\0\0"[..]).is_err());
    }
}
