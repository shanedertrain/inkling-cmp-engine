//! The E1 oracle dumps `weight-sums` and `arena-map` in the record formats of the C++ side
//! (`engine/native/capi/dumps.cpp` @ 66fe353, `README.md` "Oracle record set"), byte for byte, so each comparison is
//! one diff.

use std::fmt::Write as _;

use ink_cuda::{Arena, ArenaPlan, Stream};

use crate::error::{Result, invalid};
use crate::weights::Weights;

/// FNV-1a 64 offset basis.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64 prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 of `bytes`, continuing from `h`.
pub fn fnv1a64(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(FNV_PRIME);
    }
    h
}

/// One line of `weight-sums`: a weight region from its offset to the next region's (alignment gap included).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSum {
    /// Dump name.
    pub name: String,
    /// Arena offset.
    pub off: usize,
    /// Extent in bytes.
    pub bytes: usize,
    /// FNV-1a 64 of the extent as read back from the device.
    pub fnv1a64: u64,
}

/// Bytes read back per copy (the C++ buffer size).
const CHUNK: usize = 64 << 20;

/// Reads back every weight region of `w` from `ar` and hashes it, each to the next region's offset and the last to
/// `end` (the first activation buffer, `kpool`), with the C++ checks: the weights are the arena's first allocations,
/// no two share an address, the span ends inside the used part of the arena.
///
/// # Errors
///
/// [`crate::ModelError::Invalid`] when a check fails, or the device error of a copy.
pub fn weight_sums(ar: &Arena, st: &Stream, w: &Weights, end: usize) -> Result<Vec<WeightSum>> {
    let r = w.regions();
    let (Some(first), Some(last)) = (r.first(), r.last()) else {
        return Err(invalid("weight-sums", "no weight regions"));
    };
    if first.offset != 0 {
        return Err(invalid(
            "weight-sums",
            "the weights are not the arena's first allocations",
        ));
    }
    if let Some(p) = r.windows(2).find(|p| p[0].offset == p[1].offset) {
        return Err(invalid(
            "weight-sums",
            format!("{} and {} share an address", p[0].name, p[1].name),
        ));
    }
    if end <= last.offset || end > ar.plan().used() {
        return Err(invalid("weight-sums", "weight span does not end before kpool"));
    }
    st.synchronize()?;
    let mut buf = vec![0u8; CHUNK];
    let mut out = Vec::with_capacity(r.len());
    for (i, reg) in r.iter().enumerate() {
        let (lo, hi) = (reg.offset, r.get(i + 1).map_or(end, |n| n.offset));
        let mut h = FNV_OFFSET;
        let mut x = lo;
        while x < hi {
            let n = CHUNK.min(hi - x);
            st.dtoh(&mut buf[..n], &ar.span(x..x + n)?)?;
            h = fnv1a64(&buf[..n], h);
            x += n;
        }
        out.push(WeightSum {
            name: reg.name.clone(),
            off: lo,
            bytes: hi - lo,
            fnv1a64: h,
        });
    }
    Ok(out)
}

/// The `weight-sums` file (`ink-weight-sums/1`): header line, then one line per region. `end` is the span's end
/// offset (the arena base is offset 0).
pub fn weight_sums_text(rows: &[WeightSum], end: usize) -> String {
    let mut s = format!(
        "{{\"format\":\"ink-weight-sums/1\",\"hash\":\"fnv1a64\",\"extent\":\"to the next region\",\"regions\":{},\"bytes\":{end}}}\n",
        rows.len()
    );
    for r in rows {
        // infallible: writing to a String
        let _ = writeln!(
            s,
            "{{\"name\":\"{}\",\"off\":{},\"bytes\":{},\"fnv1a64\":\"{:016x}\"}}",
            r.name, r.off, r.bytes, r.fnv1a64
        );
    }
    s
}

/// The `arena-map` file (`ink-arena-map/1`, the format hook H1 gives the C++ side): header with alignment, capacity,
/// used bytes and take count, then name, offset and requested bytes of every take in take order.
pub fn arena_map_text(plan: &ArenaPlan) -> String {
    let mut s = format!(
        "{{\"format\":\"ink-arena-map/1\",\"align\":{},\"cap\":{},\"used\":{},\"takes\":{}}}\n",
        ink_cuda::ALIGN,
        plan.cap(),
        plan.used(),
        plan.regions().len()
    );
    for r in plan.regions() {
        let _ = writeln!(
            s,
            "{{\"name\":\"{}\",\"off\":{},\"bytes\":{}}}",
            r.name, r.offset, r.bytes
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_reference_values() {
        // FNV-1a 64 test vectors (Fowler, Noll, Vo): "" and "a", "foobar"
        assert_eq!(fnv1a64(b"", FNV_OFFSET), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a", FNV_OFFSET), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar", FNV_OFFSET), 0x8594_4171_f739_67e8);
        // chunked = whole (the 64 MiB copies)
        assert_eq!(
            fnv1a64(b"bar", fnv1a64(b"foo", FNV_OFFSET)),
            fnv1a64(b"foobar", FNV_OFFSET)
        );
    }

    #[test]
    fn text_formats_match_dumps_cpp() {
        let rows = [WeightSum {
            name: "layers.0.attn_norm".into(),
            off: 0,
            bytes: 9216,
            fnv1a64: 0x65d9_0bc5_7af0_287b,
        }];
        // the first two lines of bench/raw/2026-11-23-a1-oracle-native/oracle/e1-weight-sums.jsonl, with its count
        assert_eq!(
            weight_sums_text(&rows, 20_250_028_032),
            "{\"format\":\"ink-weight-sums/1\",\"hash\":\"fnv1a64\",\"extent\":\"to the next region\",\"regions\":1,\"bytes\":20250028032}\n\
             {\"name\":\"layers.0.attn_norm\",\"off\":0,\"bytes\":9216,\"fnv1a64\":\"65d90bc57af0287b\"}\n"
        );
        let mut p = ArenaPlan::new(1000);
        p.take(10, "kpool").unwrap();
        p.take(3, "pt g").unwrap();
        assert_eq!(
            arena_map_text(&p),
            "{\"format\":\"ink-arena-map/1\",\"align\":256,\"cap\":1000,\"used\":259,\"takes\":2}\n\
             {\"name\":\"kpool\",\"off\":0,\"bytes\":10}\n{\"name\":\"pt g\",\"off\":256,\"bytes\":3}\n"
        );
    }
}
