//! The device's primary context, its one stream and timing events.
//!
//! [`Context`], [`Stream`] and [`Event`] wrap cudarc's safe `CudaContext`, `CudaStream` and `CudaEvent` (primary
//! context retained once, released when the last object holding it drops: the context goes last). They are neither
//! `Send` nor `Sync`: one engine thread owns the context (docs/RUST-PLAN.md 3.2), and every raw call of this crate
//! assumes the context is current on the calling thread, which cudarc made so when the context was created.
//! cudarc's per-buffer event tracking is off: this crate never uses `CudaSlice`, and all work is ordered on one
//! stream plus explicit events.

use std::any::Any;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use bytemuck::Pod;
use cudarc::driver::sys::{CUdevice_attribute, CUevent_flags};
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, result};

use crate::arena::DevBuf;
use crate::check::{StartupReport, startup_checks};
use crate::error::{CudaError, Result, driver};
use crate::graph::CaptureState;

/// Marker that removes `Send` and `Sync` from the context-bound types.
pub(crate) type NotSend = PhantomData<*const ()>;

/// The primary context of one device, created after the startup checks pass.
#[derive(Debug)]
pub struct Context {
    inner: Arc<CudaContext>,
    report: StartupReport,
    _not_send: NotSend,
}

/// Static facts about the device of a [`Context`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// `cuDeviceGetName`.
    pub name: String,
    /// `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`, as `nvidia-smi -L` prints it.
    pub uuid: String,
    /// Compute capability (8, 0 on GA100).
    pub compute_capability: (i32, i32),
    /// Number of SMs.
    pub sm_count: i32,
    /// Whether cooperative launches are supported.
    pub cooperative_launch: bool,
    /// Free and total device memory in bytes at the time of the query.
    pub mem_free_total: (usize, usize),
}

impl Context {
    /// Runs [`startup_checks`], then retains the primary context of device `ordinal` and makes it current on this
    /// thread.
    ///
    /// # Errors
    ///
    /// Any startup check failure, or the driver error of context creation.
    pub fn new(ordinal: usize) -> Result<Context> {
        let report = startup_checks()?;
        let inner = CudaContext::new(ordinal).map_err(driver("cuDevicePrimaryCtxRetain"))?;
        // SAFETY: event tracking only synchronises cudarc's own CudaSlice buffers across streams; this crate never
        // creates one (one arena of raw device memory), and orders all work on its own stream and events.
        unsafe { inner.disable_event_tracking() };
        Ok(Context {
            inner,
            report,
            _not_send: PhantomData,
        })
    }

    /// What the startup checks found.
    pub fn startup(&self) -> &StartupReport {
        &self.report
    }

    /// Queries name, UUID, compute capability, SM count, cooperative support and memory.
    ///
    /// # Errors
    ///
    /// The driver error of the failing query.
    pub fn device(&self) -> Result<DeviceInfo> {
        let attr = |a: CUdevice_attribute| self.inner.attribute(a).map_err(driver("cuDeviceGetAttribute"));
        let uuid = self.inner.uuid().map_err(driver("cuDeviceGetUuid"))?;
        Ok(DeviceInfo {
            name: self.inner.name().map_err(driver("cuDeviceGetName"))?,
            uuid: format_uuid(&uuid.bytes.map(|b| b as u8)),
            compute_capability: (
                attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            ),
            sm_count: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            cooperative_launch: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH)? != 0,
            mem_free_total: self.inner.mem_get_info().map_err(driver("cuMemGetInfo"))?,
        })
    }

    /// Free and total device memory in bytes now (`cuMemGetInfo`).
    ///
    /// # Errors
    ///
    /// The driver error of `cuMemGetInfo`.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.inner.mem_get_info().map_err(driver("cuMemGetInfo"))
    }

    /// Creates a non-blocking stream.
    ///
    /// # Errors
    ///
    /// The driver error of `cuStreamCreate`.
    pub fn new_stream(&self) -> Result<Stream> {
        let inner = self.inner.new_stream().map_err(driver("cuStreamCreate"))?;
        Ok(Stream {
            inner,
            capture: RefCell::new(None),
            trace: RefCell::new(None),
            _not_send: PhantomData,
        })
    }

    /// Creates an event with timing enabled (`CU_EVENT_DEFAULT`), for timestamps on a stream.
    ///
    /// # Errors
    ///
    /// The driver error of `cuEventCreate`.
    pub fn new_event(&self) -> Result<Event> {
        let inner = self
            .inner
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(driver("cuEventCreate"))?;
        Ok(Event {
            inner,
            _not_send: PhantomData,
        })
    }

    pub(crate) fn raw(&self) -> &Arc<CudaContext> {
        &self.inner
    }
}

/// A non-blocking stream of the context. All engine work goes on one of these.
///
/// Host-device copies take slices and check lengths. They are refused while the stream is capturing: a captured
/// host copy would keep a pointer to host memory the borrow checker no longer tracks.
#[derive(Debug)]
pub struct Stream {
    pub(crate) inner: Arc<CudaStream>,
    /// `Some` while a capture is open: its region marks and the objects its nodes refer to.
    pub(crate) capture: RefCell<Option<CaptureState>>,
    /// `Some` while an eager launch trace is being recorded ([`Stream::start_trace`]).
    pub(crate) trace: RefCell<Option<crate::trace::TraceState>>,
    _not_send: NotSend,
}

impl Stream {
    /// Blocks until all work queued on the stream has finished.
    ///
    /// # Errors
    ///
    /// The driver error of `cuStreamSynchronize` (also reports earlier asynchronous kernel faults).
    pub fn synchronize(&self) -> Result<()> {
        self.not_capturing("synchronize")?;
        self.inner.synchronize().map_err(driver("cuStreamSynchronize"))
    }

    /// Whether a capture is open on this stream.
    pub fn is_capturing(&self) -> bool {
        self.capture.borrow().is_some()
    }

    /// Copies `src` into `dst` (equal lengths). Returns once `src` may be reused (pageable source memory is staged
    /// before the call returns).
    ///
    /// Assumes `src` is pageable: nothing in the process page-locks it (`cuMemHostRegister`, `cuMemHostAlloc`). The
    /// driver copies from page-locked memory asynchronously, after this borrow has ended (review R3, F11).
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] on a length mismatch, [`CudaError::Capturing`], or the driver error.
    pub fn htod<T: Pod>(&self, dst: &DevBuf<T>, src: &[T]) -> Result<()> {
        self.not_capturing("host-to-device copy")?;
        same_len("htod", dst.len(), src.len())?;
        // SAFETY: dst is a live arena range of exactly size_of_val(src) bytes (lengths checked, the DevBuf keeps its
        // allocation alive). src is pageable host memory (this crate never pins or registers host memory), for which
        // cuMemcpyHtoDAsync copies the bytes to a staging buffer before returning, so the borrow may end afterwards.
        unsafe { result::memcpy_htod_async(dst.addr(), src, self.inner.cu_stream()) }
            .map_err(driver("cuMemcpyHtoDAsync"))
    }

    /// Copies the raw bytes `src` into `dst` (`src.len() == dst.bytes()`), for host data that is not aligned for `T`
    /// (a tensor inside a mapped checkpoint). Returns once `src` may be reused, as [`Stream::htod`].
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] on a size mismatch, [`CudaError::Capturing`], or the driver error.
    pub fn htod_bytes<T: Pod>(&self, dst: &DevBuf<T>, src: &[u8]) -> Result<()> {
        self.not_capturing("host-to-device copy")?;
        if src.len() != dst.bytes() {
            return Err(CudaError::Bounds {
                what: "htod_bytes",
                detail: format!("device buffer has {} bytes, host slice {}", dst.bytes(), src.len()),
            });
        }
        // SAFETY: dst is a live arena range of exactly src.len() bytes (checked above, the DevBuf keeps its
        // allocation alive); u8 has no alignment, and the bytes become T values on the device only, where T: Pod
        // accepts any bit pattern. src is pageable host memory, staged before the call returns (as `htod`).
        unsafe { result::memcpy_htod_async(dst.addr(), src, self.inner.cu_stream()) }
            .map_err(driver("cuMemcpyHtoDAsync"))
    }

    /// Copies `src` into `dst` (equal lengths) and waits for the copy and all earlier work on the stream.
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] on a length mismatch, [`CudaError::Capturing`], or the driver error.
    pub fn dtoh<T: Pod>(&self, dst: &mut [T], src: &DevBuf<T>) -> Result<()> {
        self.not_capturing("device-to-host copy")?;
        same_len("dtoh", src.len(), dst.len())?;
        // SAFETY: src is a live arena range of size_of_val(dst) bytes; dst is pageable host memory, for which
        // cuMemcpyDtoHAsync returns only after the copy has completed; the stream sync below makes that explicit.
        // T: Pod, so any bytes written are a valid T.
        unsafe { result::memcpy_dtoh_async(dst, src.addr(), self.inner.cu_stream()) }
            .map_err(driver("cuMemcpyDtoHAsync"))?;
        self.inner.synchronize().map_err(driver("cuStreamSynchronize"))
    }

    /// Copies `src` into `dst` on the device (equal lengths). Allowed in a capture (a memcpy node).
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] on a length mismatch, or the driver error.
    pub fn dtod<T: Pod>(&self, dst: &DevBuf<T>, src: &DevBuf<T>) -> Result<()> {
        same_len("dtod", dst.len(), src.len())?;
        if dst.overlaps(src) {
            // cuMemcpyDtoD's result is undefined for overlapping ranges
            return Err(CudaError::Bounds {
                what: "dtod",
                detail: format!(
                    "source (offset {}) and destination (offset {}) of {} bytes overlap",
                    src.offset(),
                    dst.offset(),
                    dst.bytes()
                ),
            });
        }
        // SAFETY: both are live arena ranges of dst.bytes() bytes that do not overlap; the copy is ordered on this
        // stream.
        unsafe { result::memcpy_dtod_async(dst.addr(), src.addr(), dst.bytes(), self.inner.cu_stream()) }
            .map_err(driver("cuMemcpyDtoDAsync"))?;
        self.keep(dst.keep_alive());
        self.keep(src.keep_alive());
        self.trace_dtod(dst.addr(), src.addr(), dst.bytes());
        Ok(())
    }

    /// Sets every byte of `dst` to zero. Allowed in a capture (a memset node).
    ///
    /// # Errors
    ///
    /// The driver error of `cuMemsetD8Async`.
    pub fn memset_zero<T: Pod>(&self, dst: &DevBuf<T>) -> Result<()> {
        // SAFETY: dst is a live arena range of dst.bytes() bytes; the memset is ordered on this stream.
        unsafe { result::memset_d8_async(dst.addr(), 0, dst.bytes(), self.inner.cu_stream()) }
            .map_err(driver("cuMemsetD8Async"))?;
        self.keep(dst.keep_alive());
        Ok(())
    }

    /// Sets `dst[first + i * stride]` to `value` for `i` in `0..count`: one column of a row-major matrix with rows of
    /// `stride` elements (`cuMemsetD2D32Async`, width 1). Not allowed in a capture.
    ///
    /// # Errors
    ///
    /// [`CudaError::Bounds`] when an element lies outside `dst`, [`CudaError::Capturing`], or the driver error.
    pub fn fill_strided(&self, dst: &DevBuf<f32>, first: usize, stride: usize, count: usize, value: f32) -> Result<()> {
        self.not_capturing("strided fill")?;
        if count == 0 {
            return Ok(());
        }
        let last = (count - 1)
            .checked_mul(stride)
            .and_then(|o| o.checked_add(first))
            .filter(|&l| l < dst.len() && stride > 0);
        if last.is_none() {
            return Err(CudaError::Bounds {
                what: "fill_strided",
                detail: format!(
                    "{count} elements from {first} every {stride} in a buffer of {}",
                    dst.len()
                ),
            });
        }
        let addr = dst.addr() + (first * 4) as u64;
        // SAFETY: every element written, first + i * stride for i < count, lies inside the live arena range dst
        // (checked above); f32 elements make address and pitch 4-byte aligned as the call requires; ordered on this
        // stream.
        let st = unsafe {
            cudarc::driver::sys::cuMemsetD2D32Async(addr, stride * 4, value.to_bits(), 1, count, self.inner.cu_stream())
        };
        crate::error::check_driver("cuMemsetD2D32Async", st)
    }

    /// Records `event` on this stream (a timestamp). Not allowed in a capture: graphs carry region marks instead.
    ///
    /// # Errors
    ///
    /// [`CudaError::Capturing`] or the driver error of `cuEventRecord`.
    pub fn record(&self, event: &Event) -> Result<()> {
        self.not_capturing("event record")?;
        event.inner.record(&self.inner).map_err(driver("cuEventRecord"))
    }

    pub(crate) fn not_capturing(&self, what: &'static str) -> Result<()> {
        if self.is_capturing() {
            Err(CudaError::Capturing(what))
        } else {
            Ok(())
        }
    }

    /// Hands an object a captured node refers to (allocation, module, cuBLASLt handle) to the open capture, so the
    /// graph keeps it alive. No-op outside a capture: the owners that free device resources (`Allocation`, the
    /// module) synchronise the context in their `Drop` before freeing, so queued eager work never outlives them.
    pub(crate) fn keep(&self, object: Rc<dyn Any>) {
        if let Some(c) = self.capture.borrow_mut().as_mut() {
            c.keep.push(object);
        }
    }
}

/// A timing event.
#[derive(Debug)]
pub struct Event {
    inner: CudaEvent,
    _not_send: NotSend,
}

impl Event {
    /// Blocks until the work before the event's last record has finished.
    ///
    /// # Errors
    ///
    /// The driver error of `cuEventSynchronize`.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize().map_err(driver("cuEventSynchronize"))
    }

    /// Milliseconds from this event to `end` (waits for both).
    ///
    /// # Errors
    ///
    /// The driver error of `cuEventElapsedTime` (e.g. an event that was never recorded).
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        self.inner.elapsed_ms(&end.inner).map_err(driver("cuEventElapsedTime"))
    }
}

fn same_len(what: &'static str, device: usize, host: usize) -> Result<()> {
    if device == host {
        Ok(())
    } else {
        Err(CudaError::Bounds {
            what,
            detail: format!("device buffer has {device} elements, host slice {host}"),
        })
    }
}

/// Formats a CUDA UUID the way `nvidia-smi -L` and NVML print it.
pub fn format_uuid(b: &[u8; 16]) -> String {
    let hex = |r: std::ops::Range<usize>| b[r].iter().map(|x| format!("{x:02x}")).collect::<String>();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        hex(0..4),
        hex(4..6),
        hex(6..8),
        hex(8..10),
        hex(10..16)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_formats_like_nvidia_smi() {
        let b = [
            0x5d, 0xd4, 0x49, 0xff, 0xe3, 0xec, 0x90, 0x34, 0x92, 0xb8, 0x8b, 0x70, 0x26, 0x50, 0x37, 0x6b,
        ];
        assert_eq!(format_uuid(&b), "GPU-5dd449ff-e3ec-9034-92b8-8b702650376b");
    }

    #[test]
    fn length_mismatch_is_an_error() {
        assert!(same_len("htod", 4, 4).is_ok());
        let e = same_len("htod", 4, 5).unwrap_err();
        assert!(matches!(e, CudaError::Bounds { what: "htod", .. }), "{e}");
    }

    #[test]
    fn context_bound_types_are_not_send() {
        // compile-time: a type that is Send would make this blanket impl apply twice and fail to compile
        trait AmbiguousIfSend<A> {
            fn some_item() {}
        }
        impl<T: ?Sized> AmbiguousIfSend<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
        let _ = <Context as AmbiguousIfSend<_>>::some_item;
        let _ = <Stream as AmbiguousIfSend<_>>::some_item;
        let _ = <Event as AmbiguousIfSend<_>>::some_item;
        let _ = <crate::DevBuf<f32> as AmbiguousIfSend<_>>::some_item;
        let _ = <crate::Module as AmbiguousIfSend<_>>::some_item;
        let _ = <crate::Function as AmbiguousIfSend<_>>::some_item;
        let _ = <crate::Graph as AmbiguousIfSend<_>>::some_item;
        let _ = <crate::LtGemm as AmbiguousIfSend<_>>::some_item;
        let _ = <crate::Arena as AmbiguousIfSend<_>>::some_item;
    }
}
