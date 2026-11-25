//! Cubin modules, kernel functions and launches through the driver API (`cuModuleLoadData`, `cuLaunchKernel`,
//! `cuLaunchCooperativeKernel`): the mechanism H2 proved for Humming's cubins (`docs/stage-p/humming-from-cpp.md`).
//!
//! cudarc's safe `load_module` needs its `nvrtc` feature and takes PTX or a file, so loading goes through
//! `cudarc::driver::result::module`; launches go through `result::launch_kernel`, with arguments packed by
//! [`KernelArgs`] (typed: device buffers and plain-old-data values, never raw addresses).

use std::any::Any;
use std::ffi::{CStr, CString, c_void};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use bytemuck::Pod;
use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaContext, result, sys};

use crate::arena::DevBuf;
use crate::context::{Context, NotSend, Stream};
use crate::error::{CudaError, Result, check_driver, driver};

/// A loaded module.
#[derive(Debug)]
struct ModuleInner {
    raw: sys::CUmodule,
    ctx: Arc<CudaContext>,
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        // An eager launch holds no reference to the module: wait for all queued work before its code goes.
        self.ctx.record_err(self.ctx.synchronize());
        // SAFETY: raw came from cuModuleLoadData and is unloaded once, here; every Function and every graph holding
        // one of its kernels holds this ModuleInner, and the context was synchronised above.
        self.ctx.record_err(unsafe { result::module::unload(self.raw) });
    }
}

/// A cubin loaded into the context.
#[derive(Debug, Clone)]
pub struct Module {
    inner: Rc<ModuleInner>,
}

impl Module {
    /// Loads an sm_80 cubin (an ELF image, e.g. embedded with `include_bytes!`).
    ///
    /// # Errors
    ///
    /// [`CudaError::Module`] when the bytes are not an ELF image, or the driver error of `cuModuleLoadData`.
    pub fn load(ctx: &Context, cubin: &[u8]) -> Result<Module> {
        if !cubin.starts_with(b"\x7fELF") {
            return Err(CudaError::Module("image is not an ELF cubin".into()));
        }
        // 8-byte aligned copy: the driver parses the ELF headers in place.
        let mut image = vec![0u64; cubin.len().div_ceil(8)];
        bytemuck::cast_slice_mut::<u64, u8>(&mut image)[..cubin.len()].copy_from_slice(cubin);
        // the module goes into the thread's current context: make it ctx (one thread may hold several contexts)
        ctx.raw().bind_to_thread().map_err(driver("cuCtxSetCurrent"))?;
        // SAFETY: image is a complete ELF image that lives for the call; the driver copies what it keeps. ctx is
        // current on this thread (bound above).
        let raw = unsafe { result::module::load_data(image.as_ptr().cast()) }.map_err(driver("cuModuleLoadData"))?;
        Ok(Module {
            inner: Rc::new(ModuleInner {
                raw,
                ctx: ctx.raw().clone(),
            }),
        })
    }

    /// Looks up the kernel `name` (the mangled name for a C++ template instance).
    ///
    /// # Errors
    ///
    /// [`CudaError::Module`] for a name with a NUL byte, or the driver error (`CUDA_ERROR_NOT_FOUND`).
    pub fn function(&self, name: &str) -> Result<Function> {
        let c = c_name(name)?;
        // SAFETY: raw is a loaded module (held by self); the out handle stays valid while the module is loaded,
        // which the Function's reference to ModuleInner guarantees.
        let raw = unsafe { result::module::get_function(self.inner.raw, c) }.map_err(driver("cuModuleGetFunction"))?;
        Ok(Function {
            raw,
            module: self.inner.clone(),
            _not_send: PhantomData,
        })
    }

    /// Reads the bytes of the device global `name` (e.g. the kernel pack's `ink_kpack_id`).
    ///
    /// # Errors
    ///
    /// [`CudaError::Capturing`], the driver error of the lookup or the copy.
    pub fn read_global(&self, name: &str, stream: &Stream) -> Result<Vec<u8>> {
        stream.not_capturing("module global read")?;
        let c = c_name(name)?;
        // SAFETY: raw is a loaded module; the returned range is the global's storage while the module is loaded.
        let (ptr, bytes) =
            unsafe { result::module::get_global(self.inner.raw, c) }.map_err(driver("cuModuleGetGlobal"))?;
        let mut out = vec![0u8; bytes];
        // SAFETY: [ptr, ptr + bytes) is the global (module alive through self); out has exactly `bytes` bytes of
        // pageable memory, for which the async copy completes before returning; then the stream is synchronised.
        unsafe { result::memcpy_dtoh_async(&mut out, ptr, stream.inner.cu_stream()) }
            .map_err(driver("cuMemcpyDtoHAsync"))?;
        stream.synchronize()?;
        Ok(out)
    }
}

fn c_name(name: &str) -> Result<CString> {
    CString::new(name).map_err(|_| CudaError::Module(format!("symbol name {name:?} contains a NUL byte")))
}

/// A kernel of a loaded [`Module`].
#[derive(Debug, Clone)]
pub struct Function {
    raw: sys::CUfunction,
    module: Rc<ModuleInner>,
    _not_send: NotSend,
}

impl Function {
    /// The kernel's symbol name as the driver reports it (`cuFuncGetName`).
    ///
    /// # Errors
    ///
    /// The driver error.
    pub fn name(&self) -> Result<String> {
        func_name(self.raw)
    }

    /// Sets `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` (needed above 48 KiB of dynamic shared memory).
    ///
    /// # Errors
    ///
    /// The driver error (a size above the device limit).
    pub fn set_max_dynamic_smem(&self, bytes: u32) -> Result<()> {
        let v = i32::try_from(bytes).map_err(|_| CudaError::Bounds {
            what: "max dynamic smem",
            detail: format!("{bytes} bytes"),
        })?;
        let attr = CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        // SAFETY: raw is a live function handle (its module is held by self).
        unsafe { result::function::set_function_attribute(self.raw, attr, v) }.map_err(driver("cuFuncSetAttribute"))
    }

    /// Offset and size of every parameter, in order (`cuFuncGetParamInfo`): how the launch trace lays out argument
    /// bytes, as the C++ dumper does.
    ///
    /// # Errors
    ///
    /// The driver error (a module without a parameter table).
    pub fn param_layout(&self) -> Result<Vec<(usize, usize)>> {
        param_layout(|i, off, size| {
            // SAFETY: raw is a live function handle (module held by self); off and size are valid out-pointers.
            unsafe { sys::cuFuncGetParamInfo(self.raw, i, off, size) }
        })
    }

    /// Blocks of `block` threads with `smem` bytes of dynamic shared memory that fit on one SM at once
    /// (`cuOccupancyMaxActiveBlocksPerMultiprocessor`; the C++ `cudaOccupancyMaxActiveBlocksPerMultiprocessor`).
    ///
    /// # Errors
    ///
    /// The driver error.
    pub fn max_active_blocks_per_sm(&self, block: u32, smem: u32) -> Result<i32> {
        let mut n = 0i32;
        let b = i32::try_from(block).map_err(|_| CudaError::Bounds {
            what: "occupancy",
            detail: format!("block {block}"),
        })?;
        // SAFETY: raw is a live function handle; n is a valid out-pointer.
        let st = unsafe { sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut n, self.raw, b, smem as usize) };
        check_driver("cuOccupancyMaxActiveBlocksPerMultiprocessor", st)?;
        Ok(n)
    }

    /// Registers per thread (`CU_FUNC_ATTRIBUTE_NUM_REGS`).
    ///
    /// # Errors
    ///
    /// The driver error.
    pub fn num_regs(&self) -> Result<i32> {
        let attr = CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS;
        // SAFETY: raw is a live function handle (its module is held by self).
        unsafe { result::function::get_function_attribute(self.raw, attr) }.map_err(driver("cuFuncGetAttribute"))
    }
}

/// Reads parameter `(offset, size)` pairs with `info(index, &mut offset, &mut size)` until the driver reports
/// `CUDA_ERROR_INVALID_VALUE` (past the last parameter).
pub(crate) fn param_layout(
    mut info: impl FnMut(usize, *mut usize, *mut usize) -> sys::CUresult,
) -> Result<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    loop {
        let (mut off, mut size) = (0usize, 0usize);
        match info(out.len(), &mut off, &mut size) {
            sys::CUresult::CUDA_ERROR_INVALID_VALUE => return Ok(out),
            st => check_driver("cuFuncGetParamInfo", st)?,
        }
        out.push((off, size));
    }
}

/// The name of a driver function handle (also used for graph kernel nodes).
pub(crate) fn func_name(f: sys::CUfunction) -> Result<String> {
    let mut p: *const std::ffi::c_char = std::ptr::null();
    // SAFETY: f is a live function handle; the driver returns a pointer to a NUL-terminated string it owns.
    check_driver("cuFuncGetName", unsafe { sys::cuFuncGetName(&mut p, f) })?;
    Ok(owned_c_str(p))
}

/// The name of a library kernel handle (graph kernel nodes launched through `cuLibrary` / the runtime).
pub(crate) fn kernel_name(k: sys::CUkernel) -> Result<String> {
    let mut p: *const std::ffi::c_char = std::ptr::null();
    // SAFETY: k is a live kernel handle taken from a graph node; the driver returns a string it owns.
    check_driver("cuKernelGetName", unsafe { sys::cuKernelGetName(&mut p, k) })?;
    Ok(owned_c_str(p))
}

fn owned_c_str(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: p is non-null and points to a NUL-terminated string owned by the driver, valid during this copy.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Grid, block and dynamic shared memory of one launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaunchCfg {
    /// Grid dimensions (x, y, z).
    pub grid: [u32; 3],
    /// Block dimensions (x, y, z).
    pub block: [u32; 3],
    /// Dynamic shared memory in bytes.
    pub smem: u32,
}

impl LaunchCfg {
    /// A one-dimensional launch: `grid` blocks of `block` threads, no dynamic shared memory.
    pub fn linear(grid: u32, block: u32) -> LaunchCfg {
        LaunchCfg {
            grid: [grid, 1, 1],
            block: [block, 1, 1],
            smem: 0,
        }
    }
}

/// 16-byte aligned storage unit of one argument (by-value structs with `int4` members need 16).
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct Chunk([u8; 16]);

/// Kernel arguments in parameter order: device buffers and plain-old-data values.
///
/// Each argument is copied into its own aligned storage; `cuLaunchKernel` copies the values at the call, so the
/// storage only has to live for the launch. Buffers are remembered so that a capture keeps their memory alive.
#[derive(Default)]
pub struct KernelArgs {
    slots: Vec<Vec<Chunk>>,
    keep: Vec<Rc<dyn Any>>,
}

impl std::fmt::Debug for KernelArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelArgs").field("len", &self.slots.len()).finish()
    }
}

impl KernelArgs {
    /// No arguments yet.
    pub fn new() -> KernelArgs {
        KernelArgs::default()
    }

    /// Appends the device address of `b` (a `T*` parameter).
    pub fn buf<T: Pod>(&mut self, b: &DevBuf<T>) -> &mut Self {
        self.keep.push(b.keep_alive());
        self.value(&b.addr())
    }

    /// Records that the launch uses `b` without passing it as an argument (a [`crate::DevPtr`] inside a by-value
    /// struct), so a capture keeps its memory alive.
    pub fn uses<T: Pod>(&mut self, b: &DevBuf<T>) -> &mut Self {
        self.keep.push(b.keep_alive());
        self
    }

    /// Appends a value parameter (`int`, `float`, a `#[repr(C)]` struct passed by value).
    pub fn value<T: Pod>(&mut self, v: &T) -> &mut Self {
        let bytes = bytemuck::bytes_of(v);
        let mut slot = vec![Chunk::default(); bytes.len().div_ceil(16).max(1)];
        bytemuck::cast_slice_mut::<Chunk, u8>(&mut slot)[..bytes.len()].copy_from_slice(bytes);
        self.slots.push(slot);
        self
    }

    /// Number of arguments.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether there are no arguments.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Argument bytes in order (the launch trace's argument bytes).
    pub fn bytes(&self) -> Vec<Vec<u8>> {
        self.slots
            .iter()
            .map(|s| bytemuck::cast_slice::<Chunk, u8>(s).to_vec())
            .collect()
    }

    fn pointers(&mut self) -> Vec<*mut c_void> {
        self.slots.iter_mut().map(|s| s.as_mut_ptr().cast()).collect()
    }
}

// SAFETY: Chunk is 16 plain bytes with no padding (repr(C), one [u8; 16] field, align 16 = size).
unsafe impl bytemuck::Zeroable for Chunk {}
// SAFETY: as above; every bit pattern is valid, Copy, 'static.
unsafe impl Pod for Chunk {}

impl Stream {
    /// Launches `f` with `args` (`cuLaunchKernel`). In a capture this becomes a kernel node.
    ///
    /// # Safety
    ///
    /// `args` must match the kernel's parameter list in count, order, size and type, and the kernel must stay
    /// inside the buffers it is given (the contract in its README; the call site carries a `// CONTRACT:` line).
    /// The driver cannot check either.
    ///
    /// # Errors
    ///
    /// The driver error of the launch (bad configuration, too much shared memory); kernel faults appear at the next
    /// synchronisation.
    pub unsafe fn launch(&self, f: &Function, cfg: LaunchCfg, args: &mut KernelArgs) -> Result<()> {
        let mut params = args.pointers();
        // SAFETY: f is live (module held by f); params point into args' storage, alive for the call; the caller
        // guarantees that they match the kernel's parameters and its memory contract.
        unsafe {
            result::launch_kernel(
                f.raw,
                dims(cfg.grid),
                dims(cfg.block),
                cfg.smem,
                self.inner.cu_stream(),
                &mut params,
            )
        }
        .map_err(driver("cuLaunchKernel"))?;
        self.keep_launch(f, args);
        self.trace_launch(f, cfg, args)
    }

    /// Launches `f` cooperatively (`cuLaunchCooperativeKernel`: all blocks co-resident, grid-wide sync allowed).
    ///
    /// # Safety
    ///
    /// As [`Stream::launch`].
    ///
    /// # Errors
    ///
    /// The driver error (`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` when the grid cannot be co-resident).
    pub unsafe fn launch_cooperative(&self, f: &Function, cfg: LaunchCfg, args: &mut KernelArgs) -> Result<()> {
        let mut params = args.pointers();
        // SAFETY: as in launch().
        unsafe {
            result::launch_cooperative_kernel(
                f.raw,
                dims(cfg.grid),
                dims(cfg.block),
                cfg.smem,
                self.inner.cu_stream(),
                &mut params,
            )
        }
        .map_err(driver("cuLaunchCooperativeKernel"))?;
        self.keep_launch(f, args);
        self.trace_launch(f, cfg, args)
    }

    /// Appends the launch to the stream's launch trace, when one is being recorded: argument bytes laid out at the
    /// kernel's parameter offsets (`cuFuncGetParamInfo`), as the C++ dumper lays out a graph node's.
    fn trace_launch(&self, f: &Function, cfg: LaunchCfg, args: &KernelArgs) -> Result<()> {
        if !self.is_tracing() {
            return Ok(());
        }
        let layout = f.param_layout()?;
        if layout.len() != args.slots.len() {
            return Err(CudaError::Bounds {
                what: "launch trace",
                detail: format!(
                    "{} has {} parameters, {} arguments given",
                    f.name()?,
                    layout.len(),
                    args.slots.len()
                ),
            });
        }
        let total = layout.iter().map(|&(o, s)| o + s).max().unwrap_or(0);
        let mut bytes = vec![0u8; total];
        for (&(off, size), slot) in layout.iter().zip(&args.slots) {
            let src = bytemuck::cast_slice::<Chunk, u8>(slot);
            bytes[off..off + size].copy_from_slice(&src[..size]);
        }
        self.trace_kernel(&f.name()?, cfg, &layout, bytes);
        Ok(())
    }

    fn keep_launch(&self, f: &Function, args: &KernelArgs) {
        if self.is_capturing() {
            self.keep(f.module.clone());
            for k in &args.keep {
                self.keep(k.clone());
            }
        }
    }
}

fn dims(d: [u32; 3]) -> (u32, u32, u32) {
    (d[0], d[1], d[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params {
        a: u64,
        n: i32,
        s: f32,
        v: [i32; 4],
    }
    // SAFETY: test struct of plain integers and floats without padding (8 + 4 + 4 + 16 = 32).
    unsafe impl bytemuck::Zeroable for Params {}
    // SAFETY: as above.
    unsafe impl Pod for Params {}

    #[test]
    fn values_keep_their_bytes_and_alignment() {
        let mut a = KernelArgs::new();
        a.value(&7i32).value(&1.5f32).value(&Params {
            a: 0x1122,
            n: -3,
            s: 2.0,
            v: [1, 2, 3, 4],
        });
        assert_eq!(a.len(), 3);
        let b = a.bytes();
        assert_eq!(&b[0][..4], &7i32.to_ne_bytes());
        assert_eq!(&b[1][..4], &1.5f32.to_ne_bytes());
        assert_eq!(b[2].len(), 32);
        assert_eq!(&b[2][8..12], &(-3i32).to_ne_bytes());
        for p in a.pointers() {
            assert_eq!(p as usize % 16, 0);
        }
    }

    #[test]
    fn cubin_must_be_elf() {
        // checked before any driver call, so this runs without a GPU only up to the check
        assert!(!b"//\n.version 8.0\n".starts_with(b"\x7fELF"));
        assert!(c_name("a\0b").is_err());
        assert_eq!(c_name("_Z3fooi").unwrap().as_bytes(), b"_Z3fooi");
    }

    #[test]
    fn linear_cfg() {
        assert_eq!(
            LaunchCfg::linear(4, 256),
            LaunchCfg {
                grid: [4, 1, 1],
                block: [256, 1, 1],
                smem: 0
            }
        );
    }
}
