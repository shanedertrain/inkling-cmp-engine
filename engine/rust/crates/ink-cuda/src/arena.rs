//! One device allocation at start, handed out by a bump allocator (port of `ink::Arena`, `engine/native/src/
//! common.hpp` @ a3209a0: 256-byte alignment, zeroed, never freed at run time; PLAN 2.1).
//!
//! [`ArenaPlan`] is the bookkeeping (offsets, names, the arena map) and needs no GPU. [`Arena`] owns the allocation;
//! each [`DevBuf`] holds a reference to it, so device memory outlives every buffer, launch argument and graph that
//! names it.

use std::any::Any;
use std::marker::PhantomData;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use bytemuck::Pod;
use cudarc::driver::{CudaContext, result, sys};

use crate::context::{Context, Stream};
use crate::error::{CudaError, Result, driver};

/// Alignment of every region, in bytes (as the C++ arena).
pub const ALIGN: usize = 256;

/// One named region of the arena map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    /// Name given at [`ArenaPlan::take`].
    pub name: String,
    /// Byte offset from the arena base.
    pub offset: usize,
    /// Requested size in bytes.
    pub bytes: usize,
}

/// Bump-allocator bookkeeping: offsets in take order, 256-byte aligned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaPlan {
    cap: usize,
    off: usize,
    regions: Vec<Region>,
}

impl ArenaPlan {
    /// An empty plan over `cap` bytes.
    pub fn new(cap: usize) -> ArenaPlan {
        ArenaPlan {
            cap,
            off: 0,
            regions: Vec::new(),
        }
    }

    /// Reserves `bytes` at the next 256-byte boundary and returns the offset.
    ///
    /// # Errors
    ///
    /// [`CudaError::ArenaExhausted`] when the region does not fit.
    pub fn take(&mut self, bytes: usize, name: &str) -> Result<usize> {
        let exhausted = |need| CudaError::ArenaExhausted {
            what: name.to_owned(),
            need,
            cap: self.cap,
        };
        let start = self
            .off
            .checked_next_multiple_of(ALIGN)
            .ok_or_else(|| exhausted(usize::MAX))?;
        let end = start.checked_add(bytes).ok_or_else(|| exhausted(usize::MAX))?;
        if end > self.cap {
            return Err(exhausted(end));
        }
        self.off = end;
        self.regions.push(Region {
            name: name.to_owned(),
            offset: start,
            bytes,
        });
        Ok(start)
    }

    /// Bytes used so far (end of the last region).
    pub fn used(&self) -> usize {
        self.off
    }

    /// Capacity in bytes.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Every region in take order (the `arena-map` dump).
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }
}

/// The device memory of an arena.
#[derive(Debug)]
pub(crate) struct Allocation {
    ptr: sys::CUdeviceptr,
    bytes: usize,
    ctx: Arc<CudaContext>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        // Eager work is asynchronous and holds no reference: wait for everything queued in the context (launches,
        // copies, replays of graphs already dropped) before the memory goes.
        self.ctx.record_err(self.ctx.synchronize());
        // SAFETY: ptr came from cuMemAlloc in Arena::new and is freed exactly once, here; every DevBuf and every
        // Graph that names it holds this Allocation, and the context was synchronised above, so no queued work can
        // use the memory afterwards.
        self.ctx.record_err(unsafe { result::free_sync(self.ptr) });
    }
}

/// One device allocation, zeroed, with its [`ArenaPlan`].
#[derive(Debug)]
pub struct Arena {
    alloc: Rc<Allocation>,
    plan: ArenaPlan,
}

impl Arena {
    /// Allocates `bytes` with `cuMemAlloc` and zeroes them on `stream` (waits for the memset).
    ///
    /// # Errors
    ///
    /// The driver error of the allocation or the memset.
    pub fn new(ctx: &Context, stream: &Stream, bytes: usize) -> Result<Arena> {
        stream.not_capturing("arena allocation")?;
        // cuMemAlloc allocates in the thread's current context, which is the last one bound on this thread (one
        // thread may hold the contexts of several devices): make it this one, as Allocation::drop does.
        ctx.raw().bind_to_thread().map_err(driver("cuCtxSetCurrent"))?;
        // SAFETY: plain allocation in ctx (bound above); the pointer is owned by Allocation from here.
        let ptr = unsafe { result::malloc_sync(bytes) }.map_err(driver("cuMemAlloc"))?;
        let alloc = Rc::new(Allocation {
            ptr,
            bytes,
            ctx: ctx.raw().clone(),
        });
        // SAFETY: [ptr, ptr + bytes) is the allocation just made; ordered on our stream, then waited for.
        unsafe { result::memset_d8_async(ptr, 0, bytes, stream.inner.cu_stream()) }
            .map_err(driver("cuMemsetD8Async"))?;
        stream.synchronize()?;
        Ok(Arena {
            alloc,
            plan: ArenaPlan::new(bytes),
        })
    }

    /// Takes `n` elements of `T` as the region `name`.
    ///
    /// # Errors
    ///
    /// [`CudaError::ArenaExhausted`] when it does not fit.
    pub fn take<T: Pod>(&mut self, n: usize, name: &str) -> Result<DevBuf<T>> {
        let bytes = n.checked_mul(size_of::<T>()).ok_or_else(|| CudaError::ArenaExhausted {
            what: name.to_owned(),
            need: usize::MAX,
            cap: self.plan.cap(),
        })?;
        let offset = self.plan.take(bytes, name)?;
        Ok(DevBuf {
            alloc: self.alloc.clone(),
            offset,
            len: n,
            _t: PhantomData,
        })
    }

    /// The bookkeeping (used bytes, arena map).
    pub fn plan(&self) -> &ArenaPlan {
        &self.plan
    }

    /// Device address and size of the allocation (crate-private: the launch trace relocates pointers against it).
    pub(crate) fn base_and_cap(&self) -> (u64, usize) {
        (self.alloc.ptr, self.alloc.bytes)
    }

    /// The bytes `r` of the arena, regardless of regions (alignment gaps included), for the oracle dumps that hash
    /// a span "to the next region" (`weight-sums`).
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] when `r` is not inside the allocation.
    pub fn span(&self, r: Range<usize>) -> Result<DevBuf<u8>> {
        if r.start > r.end || r.end > self.alloc.bytes {
            return Err(CudaError::Bounds {
                what: "arena span",
                detail: format!("{r:?} of {} bytes", self.alloc.bytes),
            });
        }
        Ok(DevBuf {
            alloc: self.alloc.clone(),
            offset: r.start,
            len: r.len(),
            _t: PhantomData,
        })
    }
}

/// A typed range of the arena: `len` elements of `T` at a byte offset.
///
/// Cloning is cheap (a reference count). The buffer keeps the arena's device memory alive.
#[derive(Debug)]
pub struct DevBuf<T> {
    alloc: Rc<Allocation>,
    offset: usize,
    len: usize,
    _t: PhantomData<T>,
}

impl<T> Clone for DevBuf<T> {
    fn clone(&self) -> Self {
        DevBuf {
            alloc: self.alloc.clone(),
            offset: self.offset,
            len: self.len,
            _t: PhantomData,
        }
    }
}

impl<T: Pod> DevBuf<T> {
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    pub fn bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Byte offset from the arena base (stable across runs with the same take order: the trace's `arena+offset`).
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The sub-range `r` of elements.
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] when `r` is not inside the buffer.
    pub fn slice(&self, r: Range<usize>) -> Result<DevBuf<T>> {
        if r.start > r.end || r.end > self.len {
            return Err(CudaError::Bounds {
                what: "slice",
                detail: format!("{r:?} of {} elements", self.len),
            });
        }
        Ok(DevBuf {
            alloc: self.alloc.clone(),
            offset: self.offset + r.start * size_of::<T>(),
            len: r.len(),
            _t: PhantomData,
        })
    }

    /// The same bytes as elements of `U` (a kernel that declares `bf16` storage as `const uint16_t*`, or KV bytes as
    /// `const uint8_t*`).
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] when the byte size is not a multiple of `size_of::<U>()` or the offset not a multiple of
    /// `align_of::<U>()`.
    pub fn cast<U: Pod>(&self) -> Result<DevBuf<U>> {
        let (bytes, size, align) = (self.bytes(), size_of::<U>(), align_of::<U>());
        let Some(len) = cast_len(bytes, self.offset, size, align) else {
            return Err(CudaError::Bounds {
                what: "cast",
                detail: format!(
                    "{bytes} bytes at offset {} as {} ({size} bytes, align {align})",
                    self.offset,
                    std::any::type_name::<U>()
                ),
            });
        };
        Ok(DevBuf {
            alloc: self.alloc.clone(),
            offset: self.offset,
            len,
            _t: PhantomData,
        })
    }

    /// The typed device address, for a pointer field of a by-value kernel parameter struct.
    ///
    /// A [`DevPtr`] does not keep the arena alive: pass the buffer to [`crate::KernelArgs::uses`] on the launch that
    /// carries the struct, so a capture keeps it.
    pub fn ptr(&self) -> DevPtr<T> {
        DevPtr {
            addr: self.addr(),
            _t: PhantomData,
        }
    }

    /// Device address (crate-private: raw addresses never leave the crate).
    pub(crate) fn addr(&self) -> sys::CUdeviceptr {
        debug_assert!(
            self.offset + self.bytes() <= self.alloc.bytes,
            "DevBuf outside its allocation"
        );
        self.alloc.ptr + self.offset as u64
    }

    /// Whether the two buffers share a byte (same allocation, intersecting non-empty ranges).
    pub(crate) fn overlaps<U: Pod>(&self, other: &DevBuf<U>) -> bool {
        Rc::ptr_eq(&self.alloc, &other.alloc) && ranges_overlap(self.offset, self.bytes(), other.offset, other.bytes())
    }

    /// The allocation, for a capture to keep alive.
    pub(crate) fn keep_alive(&self) -> Rc<dyn Any> {
        self.alloc.clone()
    }
}

/// Element count of `bytes` at `offset` reinterpreted as elements of `size` bytes and alignment `align`, if they fit
/// exactly.
fn cast_len(bytes: usize, offset: usize, size: usize, align: usize) -> Option<usize> {
    (size > 0 && bytes.is_multiple_of(size) && offset.is_multiple_of(align)).then(|| bytes / size)
}

/// Whether byte ranges `[a, a + na)` and `[b, b + nb)` intersect (empty ranges intersect nothing).
fn ranges_overlap(a: usize, na: usize, b: usize, nb: usize) -> bool {
    na > 0 && nb > 0 && a < b + nb && b < a + na
}

/// A device address of `T` elements inside a by-value kernel parameter struct (`T*` in the C++ definition).
///
/// Made by [`DevBuf::ptr`]; it has no accessor for the address (it is `Pod` only so that parameter structs can be
/// packed into launch arguments). The all-zero value is the null pointer, which kernels accept for unused slots.
#[repr(transparent)]
pub struct DevPtr<T> {
    addr: u64,
    _t: PhantomData<T>,
}

impl<T> Clone for DevPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for DevPtr<T> {}

impl<T> std::fmt::Debug for DevPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DevPtr({:#x})", self.addr)
    }
}

// SAFETY: repr(transparent) over u64 (PhantomData is zero-sized); all-zero is the null pointer.
unsafe impl<T: 'static> bytemuck::Zeroable for DevPtr<T> {}
// SAFETY: as above: no padding, every bit pattern is a valid (if possibly meaningless) address value, Copy, 'static.
unsafe impl<T: 'static> Pod for DevPtr<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_are_aligned_in_take_order() {
        let mut p = ArenaPlan::new(4096);
        assert_eq!(p.take(10, "a").unwrap(), 0);
        assert_eq!(p.take(1, "b").unwrap(), 256);
        assert_eq!(p.take(256, "c").unwrap(), 512);
        assert_eq!(p.take(0, "empty").unwrap(), 768);
        assert_eq!(p.used(), 768);
        let names: Vec<&str> = p.regions().iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c", "empty"]);
        assert_eq!(
            p.regions()[2],
            Region {
                name: "c".into(),
                offset: 512,
                bytes: 256
            }
        );
    }

    #[test]
    fn exhaustion_names_the_region_and_changes_nothing() {
        let mut p = ArenaPlan::new(1000);
        p.take(600, "kv").unwrap();
        let e = p.take(500, "logits").unwrap_err();
        assert_eq!(
            e,
            CudaError::ArenaExhausted {
                what: "logits".into(),
                need: 1268,
                cap: 1000
            }
        );
        assert_eq!(p.used(), 600);
        assert_eq!(p.regions().len(), 1);
        assert_eq!(p.take(1000 - 768, "tail").unwrap(), 768);
    }

    #[test]
    fn overflow_is_exhaustion_not_a_panic() {
        let mut p = ArenaPlan::new(usize::MAX);
        p.take(usize::MAX - 100, "big").unwrap();
        assert!(matches!(p.take(1, "x"), Err(CudaError::ArenaExhausted { .. })));
        assert!(matches!(
            ArenaPlan::new(100).take(usize::MAX, "y"),
            Err(CudaError::ArenaExhausted { .. })
        ));
    }

    #[test]
    fn casts_need_whole_elements_and_alignment() {
        assert_eq!(cast_len(512, 256, 2, 2), Some(256), "bf16 -> u16");
        assert_eq!(cast_len(512, 256, 1, 1), Some(512), "bf16 -> u8");
        assert_eq!(cast_len(6, 0, 4, 4), None, "3 bf16 are not whole f32");
        assert_eq!(cast_len(8, 2, 4, 4), None, "offset 2 is not f32-aligned");
        assert_eq!(cast_len(0, 256, 8, 8), Some(0));
        assert_eq!(cast_len(8, 0, 0, 1), None, "zero-sized element");
    }

    #[test]
    fn overlap_of_byte_ranges() {
        assert!(ranges_overlap(0, 8, 4, 8));
        assert!(ranges_overlap(4, 8, 0, 8));
        assert!(ranges_overlap(0, 16, 4, 2), "containment");
        assert!(!ranges_overlap(0, 8, 8, 8), "adjacent");
        assert!(!ranges_overlap(8, 8, 0, 8), "adjacent, reversed");
        assert!(!ranges_overlap(4, 0, 0, 8), "empty");
    }

    #[test]
    fn same_takes_give_same_offsets() {
        let run = || {
            let mut p = ArenaPlan::new(1 << 20);
            for (i, n) in [3usize, 4096, 17, 255, 256, 257].iter().enumerate() {
                p.take(*n, &format!("r{i}")).unwrap();
            }
            p
        };
        assert_eq!(run(), run());
    }
}
