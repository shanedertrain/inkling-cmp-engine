//! Device buffers handed out as [`ink_rt_api::DevBuf`] ranges (package R9): the Rust counterpart of the C ABI's
//! `ink_rt_dev_alloc` and its range checks. One registry is shared by an engine and the drafter it creates, so
//! either can be given a range the other handed out (verify ids, accept buffers, features, the drafts).
//!
//! A range's address is a handle, not a device address (ink-cuda keeps those inside): buffer number + 1 in the bits
//! from [`HANDLE_SHIFT`] up, byte offset below. A range that is not inside one registered buffer is refused.

use bytemuck::Pod;
use ink_cuda::{Arena, Context, DevBuf, Stream};
use ink_rt_api::{DevBuf as ApiBuf, RuntimeError};

use crate::error::rt;

type Res<T> = Result<T, RuntimeError>;

/// Bit at which the buffer number starts in a handle.
pub(crate) const HANDLE_SHIFT: u32 = 40;

/// The registry; see the module docs.
#[derive(Debug, Default)]
pub(crate) struct DevBufs {
    bufs: Vec<DevBuf<u8>>,
    /// Allocations made by [`DevBufs::alloc`] (each its own, zeroed), kept alive with the registry.
    arenas: Vec<Arena>,
}

impl DevBufs {
    /// Allocates `bytes` zeroed bytes in their own device allocation.
    pub(crate) fn alloc(&mut self, ctx: &Context, st: &Stream, bytes: usize) -> Res<ApiBuf> {
        if bytes == 0 || bytes >= 1 << HANDLE_SHIFT {
            return Err(RuntimeError::request(format!("device allocation of {bytes} bytes")));
        }
        let mut a = rt(Arena::new(ctx, st, bytes))?;
        let b = rt(a.take::<u8>(bytes, "dev alloc"))?;
        self.arenas.push(a);
        Ok(self.register(b))
    }

    /// Hands out `b` (a buffer its owner keeps, such as the drafts) as a range.
    pub(crate) fn register(&mut self, b: DevBuf<u8>) -> ApiBuf {
        let bytes = b.len();
        self.bufs.push(b);
        ApiBuf::new((self.bufs.len() as u64) << HANDLE_SHIFT, bytes)
    }

    /// The bytes a range names.
    pub(crate) fn resolve(&self, h: ApiBuf, what: &str) -> Res<DevBuf<u8>> {
        let idx = (h.addr() >> HANDLE_SHIFT).checked_sub(1);
        let off = (h.addr() & ((1 << HANDLE_SHIFT) - 1)) as usize;
        let buf = idx
            .and_then(|i| self.bufs.get(usize::try_from(i).ok()?))
            .ok_or_else(|| {
                RuntimeError::request(format!(
                    "{what}: {:#x} is not a device buffer of this runtime",
                    h.addr()
                ))
            })?;
        buf.slice(off..off.saturating_add(h.bytes()))
            .map_err(|e| RuntimeError::request(format!("{what}: {e}")))
    }

    /// The first `elems` elements of type `T` of a range (refused when the range is shorter or misaligned).
    pub(crate) fn typed<T: Pod>(&self, h: ApiBuf, elems: usize, what: &str) -> Res<DevBuf<T>> {
        let b = self.resolve(h, what)?;
        let need = elems * size_of::<T>();
        if b.len() < need {
            return Err(RuntimeError::request(format!("{what}: {} bytes, need {need}", b.len())));
        }
        rt(rt(b.slice(0..need))?.cast::<T>()).map_err(|e| RuntimeError::request(format!("{what}: {}", e.message())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_carry_the_buffer_number_and_offset() {
        let h = ApiBuf::new((3 << HANDLE_SHIFT) + 128, 64);
        assert_eq!(h.addr() >> HANDLE_SHIFT, 3);
        assert_eq!(h.addr() & ((1 << HANDLE_SHIFT) - 1), 128);
        let r = DevBufs::default();
        assert!(r.resolve(h, "x").is_err());
        assert!(r.resolve(ApiBuf::new(5, 4), "x").is_err());
    }
}
