// Own code of this project (prefix reuse across turns, `bench/raw/2026-11-23-prefix-reuse/README.md`).
//! Resume checkpoints of the Rust engine: [`SlotCheckpoints`] for [`Engine`].
//!
//! A slot at length `n` needs its global K/V `[0, n)` (never overwritten by later positions), the sliding-window keys
//! `[n - 1023, n)` of its 40 rings and its conv state at `n`. A checkpoint copies the conv state at once (1.92 MB per
//! slot) and the window lazily: position `w` lands on ring entry `w mod (ring pages x 1,024)`, so the first write that
//! can overwrite a window key is at `window start + ring keys`. Before a call writes there, the window of every
//! sliding visit (K and V) is copied to the slot's save area (160 MiB per slot); a rollback copies it back. Tails
//! shorter than that never copy the window.
//!
//! The save areas live in their own arena: the engine's arena keeps the C++ plan (gate E1).

use ink_cuda::{Arena, Bf16, DevBuf, Stream};
use ink_model::dims::NVISITS;
use ink_model::plan::{PAGE_ELEMS, RING_VISITS, SLD};
use ink_rt_api::{RuntimeError, SlotCheckpoints};

use super::meta::{CallSeg, PAGE, WINDOW};
use super::{Engine, Res, Rt};

/// Elements of one head's 1,024 keys in a page (`[8 heads][1024][128]`).
const HEAD_ELEMS: usize = PAGE_ELEMS / 8;
/// Elements of one key of one head.
const KEY_ELEMS: usize = HEAD_ELEMS / PAGE;

/// Conv-state elements of one slot (`[48 visits][3][SLD]`).
pub(super) const CONV_ELEMS: usize = NVISITS * 3 * SLD;

/// Where a checkpoint's window is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Window {
    /// Still in the ring: no write since the checkpoint could overwrite it.
    InRing,
    /// Copied to the slot's save area.
    Saved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ckpt {
    pos: usize,
    window: Window,
}

/// Part of a window inside one virtual page: ring page `ring_page`, offsets `lo..hi` within the page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Piece {
    pub ring_page: usize,
    pub lo: usize,
    pub hi: usize,
}

/// First position of the sliding window of a row at `pos`: the keys it and every later row may read.
fn window_start(pos: usize) -> usize {
    pos.saturating_sub(WINDOW - 1)
}

/// The window `[window_start(pos), pos)` split at page boundaries, with the ring page each part lives in.
pub(super) fn window_pieces(pos: usize, ring: usize) -> Vec<Piece> {
    let w0 = window_start(pos);
    let mut out = Vec::new();
    let mut q = w0;
    while q < pos {
        let page = q / PAGE;
        let end = pos.min((page + 1) * PAGE);
        out.push(Piece {
            ring_page: page % ring,
            lo: q % PAGE,
            hi: end - page * PAGE,
        });
        q = end;
    }
    out
}

/// Whether rows ending before `end` (exclusive) can overwrite a window key of a checkpoint at `pos`.
pub(super) fn overwrites_window(pos: usize, ring: usize, end: usize) -> bool {
    pos > 0 && end > window_start(pos) + ring * PAGE
}

/// The per-slot checkpoints and their save areas.
pub(super) struct Checkpoints {
    conv: DevBuf<Bf16>,
    kwin: DevBuf<Bf16>,
    vwin: DevBuf<Bf16>,
    at: Vec<Option<Ckpt>>,
    /// Owns the buffers above; drops after them.
    _arena: Arena,
}

impl std::fmt::Debug for Checkpoints {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Checkpoints")
            .field("at", &self.at)
            .finish_non_exhaustive()
    }
}

impl Checkpoints {
    /// Bytes of the save areas of `slots` slots (conv state + K and V windows of every sliding visit).
    pub(super) fn bytes(slots: usize) -> usize {
        slots * (CONV_ELEMS + 2 * RING_VISITS * PAGE_ELEMS) * 2 + 3 * 256
    }

    pub(super) fn new(ctx: &ink_cuda::Context, st: &Stream, slots: usize) -> Res<Self> {
        let mut arena = Arena::new(ctx, st, Self::bytes(slots)).rt()?;
        let conv = arena.take::<Bf16>(slots * CONV_ELEMS, "ckpt conv").rt()?;
        let kwin = arena
            .take::<Bf16>(slots * RING_VISITS * PAGE_ELEMS, "ckpt k window")
            .rt()?;
        let vwin = arena
            .take::<Bf16>(slots * RING_VISITS * PAGE_ELEMS, "ckpt v window")
            .rt()?;
        Ok(Self {
            conv,
            kwin,
            vwin,
            at: vec![None; slots],
            _arena: arena,
        })
    }

    /// The slot was emptied.
    pub(super) fn clear(&mut self, slot: usize) {
        self.at[slot] = None;
    }
}

/// Direction of a window copy.
#[derive(Clone, Copy)]
enum Dir {
    Save,
    Restore,
}

impl Engine {
    /// Copies the window of `slot`'s checkpoint at `pos` between the rings and the save area.
    fn copy_window(&self, slot: usize, pos: usize, dir: Dir) -> Res<()> {
        let ring = self.mem.ring_pages();
        let visit_elems = self.mem.max_slots * ring * PAGE_ELEMS;
        let (a, ck) = (&self.a, &self.ckpt);
        for piece in window_pieces(pos, ring) {
            let n = (piece.hi - piece.lo) * KEY_ELEMS;
            for (ringbuf, save) in [(&a.kring, &ck.kwin), (&a.vring, &ck.vwin)] {
                for v in 0..RING_VISITS {
                    let src0 = v * visit_elems + (slot * ring + piece.ring_page) * PAGE_ELEMS;
                    let dst0 = (slot * RING_VISITS + v) * PAGE_ELEMS;
                    for h in 0..8 {
                        let off = h * HEAD_ELEMS + piece.lo * KEY_ELEMS;
                        let r = ringbuf.slice(src0 + off..src0 + off + n).rt()?;
                        let s = save.slice(dst0 + off..dst0 + off + n).rt()?;
                        match dir {
                            Dir::Save => self.st.dtod(&s, &r).rt()?,
                            Dir::Restore => self.st.dtod(&r, &s).rt()?,
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Before a forward call: saves the window of every checkpoint the call's rows could overwrite.
    pub(super) fn save_windows(&mut self, call: &[CallSeg]) -> Res<()> {
        let ring = self.mem.ring_pages();
        for c in call {
            let Some(ck) = self.ckpt.at[c.slot] else { continue };
            if ck.window == Window::InRing && overwrites_window(ck.pos, ring, c.pos0 + c.m) {
                self.copy_window(c.slot, ck.pos, Dir::Save)?;
                self.ckpt.at[c.slot] = Some(Ckpt {
                    window: Window::Saved,
                    ..ck
                });
            }
        }
        Ok(())
    }

    fn conv_of(&self, slot: usize) -> Res<(DevBuf<Bf16>, DevBuf<Bf16>)> {
        let r = slot * CONV_ELEMS..(slot + 1) * CONV_ELEMS;
        Ok((self.a.conv.slice(r.clone()).rt()?, self.ckpt.conv.slice(r).rt()?))
    }

    fn between_calls(&self, what: &str) -> Res<()> {
        if self.committed {
            Ok(())
        } else {
            Err(RuntimeError::request(format!(
                "{what}: the last forward call is not committed"
            )))
        }
    }
}

impl SlotCheckpoints for Engine {
    fn checkpoint(&mut self, slot: usize) -> Res<()> {
        self.check_slot(slot)?;
        self.between_calls("checkpoint")?;
        let pos = self.slots.borrow().length(slot);
        if pos == 0 {
            return Err(RuntimeError::request(format!("checkpoint of empty slot {slot}")));
        }
        let (live, saved) = self.conv_of(slot)?;
        self.st.dtod(&saved, &live).rt()?;
        self.ckpt.at[slot] = Some(Ckpt {
            pos,
            window: Window::InRing,
        });
        Ok(())
    }

    fn checkpoint_pos(&self, slot: usize) -> Option<usize> {
        self.ckpt.at.get(slot).copied().flatten().map(|c| c.pos)
    }

    fn rollback(&mut self, slot: usize, pos: usize) -> Res<()> {
        self.check_slot(slot)?;
        self.between_calls("rollback")?;
        let ck = self.ckpt.at[slot]
            .filter(|c| c.pos == pos)
            .ok_or_else(|| RuntimeError::request(format!("slot {slot} has no checkpoint at position {pos}")))?;
        let (live, saved) = self.conv_of(slot)?;
        self.st.dtod(&live, &saved).rt()?;
        if ck.window == Window::Saved {
            self.copy_window(slot, pos, Dir::Restore)?;
        }
        self.slots.borrow_mut().truncate(slot, pos).rt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_pieces_follow_pages_and_ring() {
        // inside one page: [2000 - 1023, 2000) = [977, 2000) spans pages 0 and 1
        assert_eq!(
            window_pieces(2000, 3),
            [
                Piece {
                    ring_page: 0,
                    lo: 977,
                    hi: 1024
                },
                Piece {
                    ring_page: 1,
                    lo: 0,
                    hi: 976
                }
            ]
        );
        // page-aligned checkpoint: [3072 - 1023, 3072) lies in page 2 only
        assert_eq!(
            window_pieces(3072, 3),
            [Piece {
                ring_page: 2,
                lo: 1,
                hi: 1024
            }]
        );
        // ring wrap: page 3 -> ring page 0
        assert_eq!(window_pieces(4000, 3)[1].ring_page, 0);
        // short prompt: [0, 700)
        assert_eq!(
            window_pieces(700, 3),
            [Piece {
                ring_page: 0,
                lo: 0,
                hi: 700
            }]
        );
        let total: usize = window_pieces(10_001, 3).iter().map(|p| p.hi - p.lo).sum();
        assert_eq!(total, 1023);
    }

    #[test]
    fn overwrite_rule_is_the_ring_period() {
        // checkpoint 5000: window [3977, 5000); position 3977 + 3072 = 7049 is the first that lands on a window key
        assert!(!overwrites_window(5000, 3, 7049));
        assert!(overwrites_window(5000, 3, 7050));
        // brute force over positions: a write at w overwrites q iff w = q (mod ring * 1024) and w > q
        for ring in [2usize, 3] {
            let period = ring * 1024;
            for pos in [1, 700, 1023, 1024, 1025, 2048, 3072, 5000] {
                let mut hit = false;
                for w in pos..pos + 8000 {
                    // the write at w lands on q = w - period * k, k >= 1
                    hit |= (1..=w / period).any(|k| (window_start(pos)..pos).contains(&(w - period * k)));
                    assert_eq!(
                        overwrites_window(pos, ring, w + 1),
                        hit,
                        "ring {ring} pos {pos} end {}",
                        w + 1
                    );
                }
            }
        }
    }
}
