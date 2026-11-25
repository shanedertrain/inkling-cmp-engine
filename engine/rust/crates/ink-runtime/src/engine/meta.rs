// Ported from engine/native/src/engine.cu @ 1c7ac46 (Engine::forward host part, reset_slot, decode_gx, attention's
// decode bounds, pick_workers / pick_workers_warp / chunk_shift_for). Own code of this project.
//! Host-side state and per-call metadata: slots and the KV page free list, the segments of one forward call, row
//! groups, page tables (global pages and the 3-page sliding-window rings), decode-attention metadata and the
//! prefill-attention work items with their key splits. No device access: everything here is tested without a GPU.

use std::fmt;

/// Page-table row stride (262,144 + 2,048 tokens = 258 pages of 1,024).
pub const KPT: usize = 264;
/// Segments of at most this many rows use decode attention, longer ones prefill attention.
pub const MAX_DEC_ROWS: usize = 16;
/// Tokens per KV page.
pub const PAGE: usize = 1024;
/// Sliding window (keys visible to a query: distance < WINDOW).
pub const WINDOW: usize = 1024;
/// decode_attn K/V heads (workers per session are counted per head).
const HKV: f64 = 8.0;
/// decode_attn warps per CTA of design A.
#[cfg(test)]
const DA_NW: f64 = 4.0;
/// SMs of the card (`pick_workers_warp`'s slot count for design A).
pub const SMS: usize = 148;

/// Refusal of a call or a slot operation (the C++ `INK_REQUIRE` messages).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaError(pub String);

impl fmt::Display for MetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A host-side refusal: the runtime stays usable.
impl crate::error::Classify for MetaError {}

fn require(ok: bool, what: &str) -> Result<(), MetaError> {
    if ok { Ok(()) } else { Err(MetaError(what.to_owned())) }
}

/// Sessions with state on the card and the global KV page pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slots {
    length: Vec<usize>,
    pages: Vec<Vec<i32>>,
    /// Free pages; taken from the back (`num_pages - 1` pushed first, so page 0 is taken first).
    free: Vec<i32>,
}

impl Slots {
    /// `max_slots` empty slots over `num_pages` pages.
    pub fn new(max_slots: usize, num_pages: usize) -> Slots {
        Slots {
            length: vec![0; max_slots],
            pages: vec![Vec::new(); max_slots],
            free: (0..num_pages as i32).rev().collect(),
        }
    }

    /// Number of slots.
    pub fn count(&self) -> usize {
        self.length.len()
    }

    /// Committed tokens of `slot`.
    pub fn length(&self, slot: usize) -> usize {
        self.length[slot]
    }

    /// Pages of `slot`, in logical order.
    pub fn pages(&self, slot: usize) -> &[i32] {
        &self.pages[slot]
    }

    /// Free pages.
    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    /// Empties `slot`: its pages go back to the free list in their order, length 0.
    pub fn reset(&mut self, slot: usize) {
        let pages = std::mem::take(&mut self.pages[slot]);
        self.free.extend(pages);
        self.length[slot] = 0;
    }

    /// Shortens `slot` to `len` committed tokens (at most its length): the pages past `ceil(len / PAGE)` go back to the
    /// free list in their order.
    pub fn truncate(&mut self, slot: usize, len: usize) -> Result<(), MetaError> {
        require(len <= self.length[slot], "truncate past the slot's length")?;
        let keep = len.div_ceil(PAGE).min(self.pages[slot].len());
        let extra = self.pages[slot].split_off(keep);
        self.free.extend(extra);
        self.length[slot] = len;
        Ok(())
    }

    /// Adds `n` committed tokens to `slot`.
    pub fn advance(&mut self, slot: usize, n: usize) {
        self.length[slot] += n;
    }

    /// Takes pages until `slot` holds `need`.
    fn reserve(&mut self, slot: usize, need: usize) -> Result<(), MetaError> {
        while self.pages[slot].len() < need {
            let p = self
                .free
                .pop()
                .ok_or_else(|| MetaError("KV page pool exhausted".into()))?;
            self.pages[slot].push(p);
        }
        Ok(())
    }
}

/// One segment of a forward call (the C++ `CallSeg`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallSeg {
    /// Slot.
    pub slot: usize,
    /// First row of the segment in the call.
    pub row0: usize,
    /// Rows.
    pub m: usize,
    /// Position of the first row (the slot's length before the call).
    pub pos0: usize,
    /// Logit rows (the last `nlog` rows).
    pub nlog: usize,
}

impl CallSeg {
    /// Decode segment (decode attention).
    pub fn dec(&self) -> bool {
        self.m <= MAX_DEC_ROWS
    }
}

/// Consecutive call rows (or logit rows) of one segment kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowGroup {
    /// First row.
    pub r0: usize,
    /// Rows.
    pub n: usize,
    /// Decode rows.
    pub dec: bool,
}

/// One segment as the caller gives it: slot, rows, logit rows wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegReq {
    /// Slot.
    pub slot: usize,
    /// Rows.
    pub m: usize,
    /// Logit rows wanted (the last ones; capped at `m`).
    pub logits: usize,
}

/// Engine limits a call is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Rows per prefill segment.
    pub max_chunk: usize,
    /// Rows per call.
    pub t_max: usize,
    /// Logit rows per call.
    pub logit_rows_max: usize,
    /// Pages of a sliding-window ring per slot and visit.
    pub ring: usize,
    /// Capacity of the prefill-attention item arrays per kind.
    pub pa_items_cap: usize,
    /// Capacity of the prefill-attention tile counters per kind.
    pub pa_tiles_cap: usize,
    /// Split partials the prefill-attention workspace holds.
    pub pa_partials_cap: usize,
}

/// Everything the host builds for one forward call, in the order the C++ copies it to the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallMeta {
    /// Segments.
    pub call: Vec<CallSeg>,
    /// Rows.
    pub t: usize,
    /// Token id per row (0 for rows whose ids come from the device).
    pub ids: Vec<i32>,
    /// Position per row.
    pub pos: Vec<i32>,
    /// Segment per row.
    pub seg: Vec<i32>,
    /// First row per segment.
    pub row0: Vec<i32>,
    /// Slot per segment.
    pub slot: Vec<i32>,
    /// Rows per segment.
    pub len: Vec<i32>,
    /// Call row of every logit row.
    pub lrows: Vec<i32>,
    /// Call rows grouped by kind.
    pub rgroups: Vec<RowGroup>,
    /// Logit rows grouped by kind (`r0` is the first logit row).
    pub lgroups: Vec<RowGroup>,
    /// Rows of the decode segments.
    pub dec_rows: usize,
    /// Global page table: call-order rows `[0, S)`, decode-order rows `[max_slots, max_slots + n_dec)`, `[2 max_slots][KPT]`.
    pub pt_g: Vec<i32>,
    /// Ring page table, same rows.
    pub pt_r: Vec<i32>,
    /// Decode segments: context, first row, chunk shift (global, sliding), rows.
    pub dctx: Vec<i32>,
    /// See `dctx`.
    pub drow0: Vec<i32>,
    /// See `dctx`.
    pub dcs_g: Vec<i32>,
    /// See `dctx`.
    pub dcs_r: Vec<i32>,
    /// See `dctx`.
    pub dnr: Vec<i32>,
    /// `pfa::Seg` per segment: q0, len, past, pt_off.
    pub pa_segs: Vec<[i32; 4]>,
    /// Prefill-attention items per kind (0 global, 1 sliding): `(segment, row, first key block, end key block)`.
    pub pa_items: [Vec<[i32; 4]>; 2],
    /// `(split index, splits of the tile, first workspace partial, tile id)`.
    pub pa_items2: [Vec<[i32; 4]>; 2],
}

/// Key chunk shift of decode attention: own context only (batch-invariant); sliding window 128 keys.
pub fn chunk_shift_for(global: bool, ctx: usize) -> u32 {
    if !global {
        7
    } else if ctx < 98_304 {
        10
    } else {
        12
    }
}

/// Key chunks of decode attention over the call's decode segments (`decode_gx`).
pub fn decode_gx(call: &[CallSeg], global: bool) -> usize {
    let mut gx = 0;
    for c in call.iter().filter(|c| c.dec()) {
        let cs = chunk_shift_for(global, c.pos0);
        let qlast = c.pos0 + c.m - 1;
        let klo = if global { 0 } else { c.pos0.saturating_sub(WINDOW - 1) };
        gx = gx.max((qlast >> cs) - (klo >> cs) + 1);
    }
    gx
}

/// Design B (sliding window) workers per (session, head), `pick_workers` (copied from the decode_attn bench).
pub fn pick_workers(n: usize, nch: usize, slots: usize) -> usize {
    let mut best = 1;
    let mut bt = 1e30f64;
    for w in 1..=nch {
        let t = ((n as f64) * HKV * (w as f64) / (slots as f64)).ceil() * ((nch as f64) / (w as f64)).ceil();
        if t < bt - 1e-9 {
            bt = t;
            best = w;
        }
    }
    best
}

/// Design A (global) workers per (session, head), `pick_workers_warp` (`da_nw` = concurrent chunks per CTA: 4 for
/// `R <= 8`, 2 for `9 <= R <= 16`).
pub fn pick_workers_warp_n(n: usize, nch: usize, slots: usize, da_nw: f64) -> usize {
    let mut best = 1;
    let mut bt = 1e30f64;
    for w in 1..=nch {
        let t = ((n as f64) * HKV * (w as f64) / (slots as f64)).ceil()
            * (((nch as f64) / (w as f64)).ceil() / da_nw).ceil();
        if t < bt - 1e-9 {
            bt = t;
            best = w;
        }
    }
    best
}

/// Design A (global) workers per (session, head), `pick_workers_warp` (`DA_NW = 4` chunks per CTA).
#[cfg(test)]
pub fn pick_workers_warp(n: usize, nch: usize, slots: usize) -> usize {
    pick_workers_warp_n(n, nch, slots, DA_NW)
}

/// The C++ engine's graph key of a call (`glue|gx_full/gx_swa:m,nlog;...`), written into launch traces.
pub fn graph_key(glue: i32, call: &[CallSeg]) -> String {
    let mut k = format!("{glue}|{}/{}:", decode_gx(call, true), decode_gx(call, false));
    for c in call {
        k.push_str(&format!("{},{};", c.m, c.nlog));
    }
    k
}

/// Rows of the widest decode segment of a call: decode attention's row bound `Params::R` (1 without decode rows).
pub fn decode_r(call: &[CallSeg]) -> usize {
    call.iter().filter(|c| c.dec()).map(|c| c.m).max().unwrap_or(1)
}

/// The decode-attention kernel variant of row bound `r`: warps per block of the sliding-window kernel
/// (`ceil(3 r / 8)`: 1 for R 1-2, 2 for R 3-5, 3 for R 6-8), which also picks the global kernel (Q staged in shared
/// memory when `3 r > 8`). Within one variant `R` changes only the launch parameter and grids.
pub fn da_variant(r: usize) -> usize {
    (3 * r).div_ceil(8)
}

/// The graph-cache key of a call: the glue version, the segment count, the call's rows and the decode-attention
/// kernel variant of its widest segment (`glue|S,T,v`). Everything else a decode or verify call launches depends on
/// these alone (one row group of `T` rows; per-segment rows, positions and pages are device data written before the
/// replay), except decode attention's row bound `R` and key chunks: a held graph's decode-attention nodes are set to
/// the call's before a replay (`Engine::forward`), so calls of one key with different per-segment widths share a
/// graph.
pub fn cache_key(glue: i32, call: &[CallSeg]) -> String {
    let t: usize = call.iter().map(|c| c.m).sum();
    format!("{glue}|{},{t},{}", call.len(), da_variant(decode_r(call)))
}

/// Whether a call runs as a captured graph (when graphs are on): decode segments that return logits for every row
/// (one-row decode, verify rows). Calls with a prefill segment, including a short prompt tail of up to 8 rows
/// (logits for its last row only), run eagerly and capture nothing, so the graph set stays the segment counts of
/// decode and verify calls (prompt tails made 2 of 479 captures of the M1 replay with context-length keys, but
/// with each width and position among the decode rows they would make about 150 more keys).
pub fn graphable(call: &[CallSeg]) -> bool {
    call.iter().all(|c| c.dec() && c.nlog == c.m)
}

/// Refuses a host token id outside `[0, vocab)`: the embedding kernel reads row `ids[t]` of the embedding without a
/// bound (`glue::embed_norm_kernel`), so an id from a caller must be checked before it reaches the device.
pub fn check_ids(ids: &[i32], vocab: usize) -> Result<(), MetaError> {
    match ids.iter().find(|&&id| usize::try_from(id).map_or(true, |i| i >= vocab)) {
        None => Ok(()),
        Some(id) => Err(MetaError(format!("token id {id} outside the vocabulary of {vocab}"))),
    }
}

/// Builds the call: validates the segments, takes the KV pages each slot needs (segment order), and builds every
/// host array of `Engine::forward`. On error nothing is taken.
pub fn build_call(slots: &mut Slots, segs: &[SegReq], lim: &Limits) -> Result<CallMeta, MetaError> {
    let max_slots = slots.count();
    require(!segs.is_empty() && segs.len() <= max_slots, "segments")?;
    let mut need_pages = 0usize;
    let mut t = 0usize;
    let mut nlog = 0usize;
    for (i, s) in segs.iter().enumerate() {
        require(s.slot < max_slots, "slot")?;
        require(s.m > 0 && s.m <= lim.max_chunk, "segment rows")?;
        require(segs[..i].iter().all(|o| o.slot != s.slot), "one segment per slot")?;
        let need = (slots.length(s.slot) + s.m).div_ceil(PAGE);
        require(need <= KPT, "context too long")?;
        need_pages += need.saturating_sub(slots.pages(s.slot).len());
        t += s.m;
        nlog += s.logits.min(s.m);
    }
    require(need_pages <= slots.free_pages(), "KV page pool exhausted")?;
    require(t <= lim.t_max, "rows per call")?;
    require(nlog <= lim.logit_rows_max, "logit rows")?;
    // the segments as the call will hold them: the prefill-attention work items depend on rows and positions only,
    // so they are built (and their capacity checked) before any page is taken
    let mut call = Vec::with_capacity(segs.len());
    let mut row = 0usize;
    for s in segs {
        call.push(CallSeg {
            slot: s.slot,
            row0: row,
            m: s.m,
            pos0: slots.length(s.slot),
            nlog: s.logits.min(s.m),
        });
        row += s.m;
    }
    let (pa_g, pa_g2) = pa_items(&call, false, lim)?;
    let (pa_s, pa_s2) = pa_items(&call, true, lim)?;

    let mut m = CallMeta {
        call: Vec::new(),
        t,
        ids: Vec::with_capacity(t),
        pos: Vec::with_capacity(t),
        seg: Vec::with_capacity(t),
        row0: Vec::new(),
        slot: Vec::new(),
        len: Vec::new(),
        lrows: Vec::new(),
        rgroups: Vec::new(),
        lgroups: Vec::new(),
        dec_rows: 0,
        pt_g: vec![0; 2 * max_slots * KPT],
        pt_r: vec![0; 2 * max_slots * KPT],
        dctx: Vec::new(),
        drow0: Vec::new(),
        dcs_g: Vec::new(),
        dcs_r: Vec::new(),
        dnr: Vec::new(),
        pa_segs: Vec::new(),
        pa_items: [Vec::new(), Vec::new()],
        pa_items2: [Vec::new(), Vec::new()],
    };
    for (i, (s, &c)) in segs.iter().zip(&call).enumerate() {
        let (len0, row) = (c.pos0, c.row0);
        m.call.push(c);
        for j in 0..s.m {
            m.ids.push(0);
            m.pos.push((len0 + j) as i32);
            m.seg.push(i as i32);
        }
        for j in s.m - c.nlog..s.m {
            m.lrows.push((row + j) as i32);
        }
        m.row0.push(row as i32);
        m.slot.push(s.slot as i32);
        m.len.push(s.m as i32);
        slots.reserve(s.slot, (len0 + s.m).div_ceil(PAGE))?;
    }
    // row groups by segment kind (consecutive segments of one kind merged): call rows and logit rows
    let mut l = 0;
    for c in &m.call {
        let dec = c.dec();
        if dec {
            m.dec_rows += c.m;
        }
        match m.rgroups.last_mut() {
            Some(g) if g.dec == dec => g.n += c.m,
            _ => m.rgroups.push(RowGroup {
                r0: c.row0,
                n: c.m,
                dec,
            }),
        }
        if c.nlog > 0 {
            match m.lgroups.last_mut() {
                Some(g) if g.dec == dec => g.n += c.nlog,
                _ => m.lgroups.push(RowGroup { r0: l, n: c.nlog, dec }),
            }
        }
        l += c.nlog;
    }
    // page tables: call order rows [0, S), decode order rows [max_slots, max_slots + n_dec)
    let mut nd = 0;
    for (s, c) in m.call.iter().enumerate() {
        let pages = slots.pages(c.slot);
        let mut fill = |r: usize| {
            for (j, &p) in pages.iter().enumerate() {
                m.pt_g[r * KPT + j] = p;
            }
            for j in 0..KPT {
                m.pt_r[r * KPT + j] = (c.slot * lim.ring + j % lim.ring) as i32;
            }
        };
        fill(s);
        if c.dec() {
            fill(max_slots + nd);
            nd += 1;
            m.dctx.push(c.pos0 as i32);
            m.drow0.push(c.row0 as i32);
            m.dcs_g.push(chunk_shift_for(true, c.pos0) as i32);
            m.dcs_r.push(chunk_shift_for(false, c.pos0) as i32);
            m.dnr.push(c.m as i32);
        }
    }
    // prefill attention: one Seg per segment; work items per kind
    for (s, c) in m.call.iter().enumerate() {
        m.pa_segs
            .push([c.row0 as i32, c.m as i32, c.pos0 as i32, (s * KPT) as i32]);
    }
    m.pa_items = [pa_g, pa_s];
    m.pa_items2 = [pa_g2, pa_s2];
    Ok(m)
}

/// Prefill-attention work items of one kind (`engine/kernels/prefill_attn/bench.cu` build_tiles rule): row tiles of
/// 128 packed rows; global: key splits at 8,192 below 64K, 16,384 below 128K, 32,768 beyond, biggest ranges first
/// (stable); sliding window unsplit, heaviest first (stable).
type Items = (Vec<[i32; 4]>, Vec<[i32; 4]>);
fn pa_items(call: &[CallSeg], swa: bool, lim: &Limits) -> Result<Items, MetaError> {
    struct It {
        a: [i32; 4],
        b: [i32; 4],
        w: i64,
    }
    let mut it: Vec<It> = Vec::new();
    let (mut tile, mut parts) = (0usize, 0usize);
    for (s, c) in call.iter().enumerate() {
        if c.dec() {
            continue;
        }
        let mut r0 = 0;
        while r0 < 3 * c.m {
            let qmax = c.pos0 + (r0 + 127).min(3 * c.m - 1) / 3;
            let qmin = c.pos0 + r0 / 3;
            let b0 = (if swa { qmin.saturating_sub(1023) } else { 0 }) / 64;
            let b1 = qmax / 64 + 1;
            let mut cuts = vec![b0];
            if !swa {
                let sb = 8192 / 64;
                let step = |x: usize| {
                    if x < 8 * sb {
                        sb
                    } else if x < 16 * sb {
                        2 * sb
                    } else {
                        4 * sb
                    }
                };
                let mut x = 0;
                while x < b1 {
                    if x > b0 {
                        cuts.push(x);
                    }
                    x += step(x);
                }
            }
            cuts.push(b1);
            let n = cuts.len() - 1;
            for k in 0..n {
                it.push(It {
                    a: [s as i32, r0 as i32, cuts[k] as i32, cuts[k + 1] as i32],
                    b: [k as i32, n as i32, parts as i32, tile as i32],
                    w: (cuts[k + 1] - cuts[k]) as i64,
                });
            }
            if n > 1 {
                parts += n;
            }
            r0 += 128;
            tile += 1;
        }
    }
    if swa {
        it.sort_by_key(|x| std::cmp::Reverse(x.w));
    } else {
        // a.z descending (first key block), then weight descending; stable
        it.sort_by(|x, y| y.a[2].cmp(&x.a[2]).then(y.w.cmp(&x.w)));
    }
    require(
        it.len() <= lim.pa_items_cap && tile <= lim.pa_tiles_cap,
        "prefill attention items",
    )?;
    require(parts.max(1) <= lim.pa_partials_cap, "prefill ws")?;
    Ok((it.iter().map(|x| x.a).collect(), it.iter().map(|x| x.b).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lim() -> Limits {
        Limits {
            max_chunk: 2048,
            t_max: 2112,
            logit_rows_max: 64,
            ring: 3,
            pa_items_cap: 9792,
            pa_tiles_cap: 408,
            pa_partials_cap: 51 * 18,
        }
    }

    #[test]
    fn free_list_takes_page_zero_first_and_returns_in_order() {
        let mut s = Slots::new(2, 4);
        let m = build_call(
            &mut s,
            &[SegReq {
                slot: 1,
                m: 1500,
                logits: 1,
            }],
            &lim(),
        )
        .unwrap();
        assert_eq!(s.pages(1), [0, 1]);
        assert_eq!(s.free_pages(), 2);
        assert_eq!(&m.pt_g[..2], [0, 1]);
        s.reset(1);
        // pages pushed back in order: [3, 2, 0, 1] -> next take is 1, then 0
        let _ = build_call(
            &mut s,
            &[SegReq {
                slot: 0,
                m: 2048,
                logits: 0,
            }],
            &lim(),
        )
        .unwrap();
        assert_eq!(s.pages(0), [1, 0]);
    }

    #[test]
    fn truncate_returns_the_pages_past_the_new_length() {
        let mut s = Slots::new(1, 8);
        let _ = build_call(
            &mut s,
            &[SegReq {
                slot: 0,
                m: 2048,
                logits: 0,
            }],
            &lim(),
        )
        .unwrap();
        s.advance(0, 2048);
        let _ = build_call(
            &mut s,
            &[SegReq {
                slot: 0,
                m: 1100,
                logits: 0,
            }],
            &lim(),
        )
        .unwrap();
        s.advance(0, 1000); // 100 rows not committed: page 3 holds no committed row
        assert_eq!((s.pages(0), s.free_pages()), (&[0, 1, 2, 3][..], 4));
        s.truncate(0, 1025).unwrap();
        assert_eq!((s.length(0), s.pages(0), s.free_pages()), (1025, &[0, 1][..], 6));
        s.truncate(0, 1024).unwrap();
        assert_eq!(s.pages(0), [0]);
        assert!(s.truncate(0, 1025).is_err());
        s.truncate(0, 0).unwrap();
        assert_eq!(s.free_pages(), 8);
    }

    #[test]
    fn exhausted_pool_takes_nothing() {
        let mut s = Slots::new(2, 2);
        let e = build_call(
            &mut s,
            &[
                SegReq {
                    slot: 0,
                    m: 1025,
                    logits: 0,
                },
                SegReq {
                    slot: 1,
                    m: 1,
                    logits: 0,
                },
            ],
            &lim(),
        );
        assert!(e.is_err());
        assert_eq!(s.free_pages(), 2);
        assert!(s.pages(0).is_empty());
    }

    #[test]
    fn refused_prefill_items_take_no_page() {
        // the item checks run after the page check passed (review finding): a workspace of one partial refuses a
        // prefill segment whose key range splits in two (10,000 + 100 keys: key blocks [0, 128), [128, 157))
        let small = Limits {
            pa_partials_cap: 1,
            ..lim()
        };
        let mut s = Slots::new(2, 16);
        s.advance(1, 10_000);
        let seg = [SegReq {
            slot: 1,
            m: 100,
            logits: 1,
        }];
        assert_eq!(build_call(&mut s, &seg, &small), Err(MetaError("prefill ws".into())));
        assert_eq!(s.free_pages(), 16);
        assert!(s.pages(1).is_empty());
        // with the engine's workspace the same call takes its 10 pages
        assert!(build_call(&mut s, &seg, &lim()).is_ok());
        assert_eq!(s.pages(1).len(), 10);
        assert_eq!(s.free_pages(), 6);
    }

    #[test]
    fn host_token_ids_must_be_in_the_vocabulary() {
        assert!(check_ids(&[0, 5, 200_057], 200_058).is_ok());
        assert!(check_ids(&[], 200_058).is_ok());
        assert!(check_ids(&[200_058], 200_058).is_err());
        assert!(check_ids(&[3, -1], 200_058).is_err());
        assert!(check_ids(&[i32::MAX], 200_058).is_err());
    }

    #[test]
    fn rows_groups_and_logit_rows_of_a_mixed_call() {
        // slot 2 decode (3 rows, 3 logits), slot 0 prefill 200 rows (1 logit), slot 1 decode 1 row (1 logit)
        let mut s = Slots::new(4, 16);
        s.advance(2, 500);
        s.advance(1, 70);
        let segs = [
            SegReq {
                slot: 2,
                m: 3,
                logits: 5,
            },
            SegReq {
                slot: 0,
                m: 200,
                logits: 1,
            },
            SegReq {
                slot: 1,
                m: 1,
                logits: 1,
            },
        ];
        let m = build_call(&mut s, &segs, &lim()).unwrap();
        assert_eq!(m.t, 204);
        assert_eq!(m.lrows, [0, 1, 2, 202, 203]);
        assert_eq!(
            m.rgroups,
            [
                RowGroup { r0: 0, n: 3, dec: true },
                RowGroup {
                    r0: 3,
                    n: 200,
                    dec: false
                },
                RowGroup {
                    r0: 203,
                    n: 1,
                    dec: true
                }
            ]
        );
        assert_eq!(
            m.lgroups,
            [
                RowGroup { r0: 0, n: 3, dec: true },
                RowGroup {
                    r0: 3,
                    n: 1,
                    dec: false
                },
                RowGroup { r0: 4, n: 1, dec: true }
            ]
        );
        assert_eq!(m.dec_rows, 4);
        assert_eq!(&m.pos[..4], [500, 501, 502, 0]);
        // pages: slot 2 first (1 page: 0), then slot 0 (page 1), slot 1 (page 2)
        assert_eq!((s.pages(2), s.pages(0), s.pages(1)), (&[0][..], &[1][..], &[2][..]));
        // ring table of segment 0 (slot 2): slot * 3 + j % 3
        assert_eq!(&m.pt_r[..4], [6, 7, 8, 6]);
        // decode order rows start at max_slots = 4: segment 0 then segment 2 (slot 1)
        assert_eq!(m.pt_g[4 * KPT], 0);
        assert_eq!(m.pt_g[5 * KPT], 2);
        assert_eq!(m.pt_r[5 * KPT + 1], 4);
        assert_eq!(
            (m.dctx.clone(), m.drow0.clone(), m.dnr.clone()),
            (vec![500, 70], vec![0, 203], vec![3, 1])
        );
        assert_eq!((m.dcs_g.clone(), m.dcs_r.clone()), (vec![10, 10], vec![7, 7]));
        assert_eq!(m.pa_segs[1], [3, 200, 0, KPT as i32]);
    }

    #[test]
    fn decode_bounds_and_workers() {
        let c = |pos0, m| CallSeg {
            slot: 0,
            row0: 0,
            m,
            pos0,
            nlog: 0,
        };
        // global chunk 1,024 keys below 98,304: qlast 2000 -> chunks 0..=1
        assert_eq!(decode_gx(&[c(2000, 1)], true), 2);
        // sliding: klo = 2000 - 1023 = 977 -> 977 >> 7 = 7, 2007 >> 7 = 15 -> 9 chunks
        assert_eq!(decode_gx(&[c(2000, 8)], false), 9);
        // 100,000 context: shift 12 -> 100,000 >> 12 = 24 -> 25 chunks
        assert_eq!(decode_gx(&[c(100_000, 1)], true), 25);
        // prefill segments are ignored
        assert_eq!(decode_gx(&[c(5, 100)], true), 0);
        // n = 4, 9 chunks, 74 x 2 = 148 slots: W=1: ceil(32/148)=1 * 9 = 9; W=9: ceil(288/148)=2 * 1 = 2;
        // W=5: 1 * 2 = 2 (first minimum below 2 - 1e-9 wins: W=5 gives ceil(160/148)=2 * 2 = 4) -> W=4: ceil(128/148)=1*3=3,
        // W=9 -> 2 * 1 = 2; W=8: ceil(256/148)=2 * 2 = 4; so best is 9 (t = 2)
        assert_eq!(pick_workers(4, 9, 148), 9);
        // design A: n = 4, 2 chunks: W=1: 1 * ceil(2/4)=1 -> 1; W=2: 1 * ceil(1/4) = 1, not better -> 1
        assert_eq!(pick_workers_warp(4, 2, 148), 1);
        assert_eq!(graph_key(6, &[c(2000, 1), c(10, 8)]), "6|2/9:1,0;8,0;");
        assert_eq!(cache_key(6, &[c(2000, 1), c(10, 8)]), "6|2,9,3");
        assert_eq!(cache_key(6, &[c(90_000, 1), c(10, 8)]), "6|2,9,3");
        // the same rows over the sessions in another layout, and the widest segment's variant
        assert_eq!(cache_key(6, &[c(10, 4), c(90_000, 5)]), "6|2,9,2");
        assert_eq!(cache_key(6, &[c(10, 3), c(90_000, 5)]), "6|2,8,2");
        assert_eq!(cache_key(1, &[c(10, 1), c(10, 1)]), "1|2,2,1");
        assert_eq!(
            [1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16].map(da_variant),
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        );
        assert_eq!(decode_r(&[c(10, 3), c(90_000, 5)]), 5);
    }

    #[test]
    fn only_decode_only_calls_are_graphed() {
        let c = |m, nlog| CallSeg {
            slot: 0,
            row0: 0,
            m,
            pos0: 100,
            nlog,
        };
        assert!(graphable(&[c(1, 1), c(8, 8), c(MAX_DEC_ROWS, MAX_DEC_ROWS)]));
        // a prefill chunk alone, or riding along with decode rows (mixed call), runs eagerly
        assert!(!graphable(&[c(MAX_DEC_ROWS + 1, 1)]));
        assert!(!graphable(&[c(1, 1), c(8, 8), c(2048, 1)]));
        // a prompt tail of a few rows (logits of its last row) or without logits, too
        assert!(!graphable(&[c(1, 1), c(6, 1)]));
        assert!(!graphable(&[c(3, 0)]));
    }

    #[test]
    fn prefill_items_split_global_keys_and_sort_like_the_cpp() {
        // one prefill segment of 200 rows at position 70,000: 3 * 200 = 600 packed rows -> tiles at 0,128,..,512 (5)
        let call = [CallSeg {
            slot: 0,
            row0: 0,
            m: 200,
            pos0: 70_000,
            nlog: 0,
        }];
        let (a, b) = pa_items(&call, false, &lim()).unwrap();
        // tile 0: qmax = 70,000 + 127 / 3 = 70,042, b1 = 70,042 / 64 + 1 = 1095; cuts 0,128,..,1024 (step 128 below
        // 1024), then 1024 + 256 = 1280 > b1 -> pieces: 8 of 128 blocks, then [1024, 1095)
        let t0: Vec<_> = a.iter().zip(&b).filter(|(_, y)| y[3] == 0).collect();
        assert_eq!(t0.len(), 9);
        // sorted by first key block descending: the [1024, 1095) piece of every tile comes first
        assert_eq!(a[0][2], 1024);
        assert_eq!(b.iter().filter(|y| y[3] == 4).count(), 9);
        // workspace partial bases: every tile has 9 pieces -> tile k starts at 9 k
        assert!(b.iter().all(|y| y[2] == 9 * y[3] && y[1] == 9));
        // sliding window: unsplit, one item per tile, heaviest first; b0 = (70,000 - 1023) / 64 = 1077
        let (a, b) = pa_items(&call, true, &lim()).unwrap();
        assert_eq!(a.len(), 5);
        assert!(b.iter().all(|y| y[0] == 0 && y[1] == 1));
        assert_eq!(a.iter().find(|x| x[1] == 0).unwrap()[2], 1077);
    }
}
