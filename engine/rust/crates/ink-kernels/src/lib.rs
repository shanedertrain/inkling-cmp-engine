//! The kernel pack for the Rust runtime (docs/RUST-PLAN.md 3.2, package R2): embedded cubins, the kernel table,
//! `#[repr(C)]` mirrors of the launch-argument structs and one typed launcher per kernel.
//!
//! The CUDA kernels stay CUDA: `engine/kernels/pack/build.sh` compiles them from the same headers as the C++ engine
//! (gate E0: identical SASS), and `xtask gen-kernels` writes the generated module (`src/gen.rs`) from the cubins, `keys.toml` and `layout.json`.
//! A launcher is named after its key; its arguments are typed from the kernel's C++ signature, so a changed kernel
//! signature is a compile error at the call site.
//!
//! The cubins are embedded with `include_bytes!` when the crate is built with `INK_KPACK_DIR` pointing at a pack
//! (the CUDA image; build.rs refuses a pack whose id differs from [`KPACK_ID`]). Without it (builds without nvcc)
//! everything compiles and [`Kernels::load`] returns an error.
//!
//! # Safety of the launchers
//!
//! Every launcher is an `unsafe fn`, as cudarc's `launch` and cubecl's `launch_unchecked` are: the generator makes
//! argument count, order, size and type right (a C++ signature change is a compile error), but which addresses a
//! kernel touches depends on `cfg`, on its scalar arguments and on device data it reads (page tables, item lists,
//! row counts), none of which a host-side check can bound. The caller guarantees, for the kernel's contract (its
//! README and the comment above its definition):
//!
//! 1. every pointer argument, and every [`ink_cuda::DevPtr`] field of a struct argument, names device memory of this
//!    context large enough for every access the kernel makes with these arguments, `cfg` and the device data it
//!    reads; [`Ptr::Null`] and null `DevPtr`s only where the kernel documents the parameter as optional;
//! 2. buffers the kernel writes are not read or written by other work running concurrently with it;
//! 3. memory named by a `DevPtr` field stays allocated until the launch has finished, and, when the launch is
//!    captured, until the last replay of the graph (pointer arguments passed as [`Ptr::Buf`] are kept alive by the
//!    capture; `DevPtr` fields are not: hold the arena longer than the graphs, or pass the buffer to
//!    [`KernelArgs::uses`] when launching by hand);
//! 4. `cfg` is the geometry the kernel was written for (block size, dynamic shared memory, and for cooperative
//!    kernels a grid that fits co-resident).
//!
//! The safe layer is the runtime's: one function per call site that owns these facts (it built the buffers, the page
//! tables and the launch geometry) and states them in its `// SAFETY:` comment; E2 compares the argument bytes with
//! the C++ engine's.

#[path = "gen.rs"]
mod generated;

use bytemuck::Pod;
use ink_cuda::{Context, CudaError, DevBuf, Function, KernelArgs, Module, Result};

pub use generated::*;

/// IEEE half (`__half`), as raw bits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F16(pub u16);

// SAFETY: repr(transparent) over u16: no padding, every bit pattern valid.
unsafe impl bytemuck::Zeroable for F16 {}
// SAFETY: as above; Copy and 'static.
unsafe impl Pod for F16 {}

/// A pointer argument of a launcher: a device buffer, or the null pointer (for parameters the kernel documents as
/// optional). `&DevBuf<T>` converts into it.
#[derive(Debug)]
pub enum Ptr<'a, T> {
    /// The buffer's device address.
    Buf(&'a DevBuf<T>),
    /// `nullptr`.
    Null,
}

impl<'a, T> From<&'a DevBuf<T>> for Ptr<'a, T> {
    fn from(b: &'a DevBuf<T>) -> Self {
        Ptr::Buf(b)
    }
}

fn push_ptr<T: Pod>(args: &mut KernelArgs, p: Ptr<'_, T>) {
    match p {
        Ptr::Buf(b) => args.buf(b),
        Ptr::Null => args.value(&0u64),
    };
}

/// One cubin of the pack (`manifest.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubinInfo {
    /// Pack TU (`engine`, `drafter`).
    pub tu: &'static str,
    /// File name in the pack directory.
    pub file: &'static str,
    /// Size in bytes.
    pub bytes: usize,
    /// sha256 of the file.
    pub sha256: &'static str,
    /// FNV-1a 64 of the file (checked against the embedded bytes at load).
    pub fnv1a64: u64,
}

/// One kernel of the pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelInfo {
    /// keys.toml key (the launcher's name).
    pub key: &'static str,
    /// Index into [`CUBINS`].
    pub tu: usize,
    /// Mangled symbol (`cuModuleGetFunction`, graph node names).
    pub symbol: &'static str,
    /// C++ parameter types in order, as the demangler spells them (`float const*`, `da::Params`).
    pub params: &'static [&'static str],
    /// Parameter bytes (nvcc's `CBANK_PARAM_SIZE`).
    pub param_bytes: usize,
    /// Launched cooperatively (the kernel synchronises the grid).
    pub coop: bool,
}

impl KernelId {
    /// The kernel's table entry.
    pub fn info(self) -> &'static KernelInfo {
        &KERNELS[self as usize]
    }
}

#[cfg(ink_kpack)]
const EMBEDDED: Option<[&[u8]; 2]> = Some([
    include_bytes!(concat!(env!("INK_KPACK_DIR"), "/pack_engine.cubin")),
    include_bytes!(concat!(env!("INK_KPACK_DIR"), "/pack_drafter.cubin")),
]);
#[cfg(not(ink_kpack))]
const EMBEDDED: Option<[&[u8]; 2]> = None;

/// The loaded pack: both modules and every kernel's function handle.
#[derive(Debug)]
pub struct Kernels {
    functions: Vec<Function>,
}

impl Kernels {
    /// Loads the embedded cubins into the context and resolves every kernel of [`KERNELS`].
    ///
    /// # Errors
    ///
    /// [`CudaError::Module`] when the crate was built without a pack (`INK_KPACK_DIR`) or an embedded cubin's size
    /// differs from [`CUBINS`]; the driver error of a load or a lookup.
    pub fn load(ctx: &Context) -> Result<Kernels> {
        let Some(images) = EMBEDDED else {
            return Err(CudaError::Module(
                "ink-kernels was built without INK_KPACK_DIR: no cubins embedded".into(),
            ));
        };
        check_images(&images, &CUBINS)?;
        let modules = images
            .iter()
            .map(|image| Module::load(ctx, image))
            .collect::<Result<Vec<_>>>()?;
        let functions = KERNELS
            .iter()
            .map(|k| modules[k.tu].function(k.symbol))
            .collect::<Result<_>>()?;
        Ok(Kernels { functions })
    }

    /// The function handle of a kernel (attributes such as the dynamic shared memory limit, names).
    pub fn function(&self, k: KernelId) -> &Function {
        &self.functions[k as usize]
    }

    /// Every kernel's function handle, in [`KERNELS`] order (whole-pack tools: the launch smoke test, traces).
    pub fn functions(&self) -> &[Function] {
        &self.functions
    }
}

/// The embedded images are the cubins the launchers were generated from (size and FNV-1a 64). build.rs checks the
/// pack at build time; this also catches a stale build that embedded another pack at the same path (cargo's
/// change detection is by modification time).
fn check_images(images: &[&[u8]], cubins: &[CubinInfo]) -> Result<()> {
    for (image, info) in images.iter().zip(cubins) {
        let h = fnv1a64(image);
        if image.len() != info.bytes || h != info.fnv1a64 {
            return Err(CudaError::Module(format!(
                "{}: embedded {} bytes, FNV-1a {h:#018x}; generated for {} bytes, {:#018x}: rebuild with the pack \
                 src/gen.rs was generated from",
                info.file,
                image.len(),
                info.bytes,
                info.fnv1a64
            )));
        }
    }
    Ok(())
}

/// FNV-1a 64 (the hash of ink-model's weight sums and xtask gen-kernels).
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A struct mirror's layout as rustc computed it: name, size, align, (field, offset, size).
#[cfg(test)]
pub(crate) struct RustLayout {
    name: &'static str,
    size: usize,
    align: usize,
    fields: &'static [(&'static str, usize, usize)],
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed layout.json (C++ compiler, layout_emit.cpp) against rustc's layout of every mirror. Prints the
    /// table (`--nocapture`) for the record.
    #[test]
    fn struct_layouts_equal_the_cpp_compilers() {
        let cpp: serde_json::Value = serde_json::from_str(include_str!("../kpack/layout.json")).unwrap();
        let cpp = cpp["structs"].as_array().unwrap();
        assert_eq!(cpp.len(), generated::RUST_LAYOUTS.len());
        println!("| struct | size C++ / Rust | align C++ / Rust | fields | field offsets and sizes |");
        println!("|---|---|---|---|---|");
        for (c, r) in cpp.iter().zip(generated::RUST_LAYOUTS) {
            assert_eq!(c["name"].as_str().unwrap(), r.name);
            assert_eq!(c["size"].as_u64().unwrap() as usize, r.size, "{} size", r.name);
            assert_eq!(c["align"].as_u64().unwrap() as usize, r.align, "{} align", r.name);
            let cf = c["fields"].as_array().unwrap();
            assert_eq!(cf.len(), r.fields.len(), "{} fields", r.name);
            for (f, (name, off, size)) in cf.iter().zip(r.fields) {
                assert_eq!(f["name"].as_str().unwrap(), *name);
                assert_eq!(f["offset"].as_u64().unwrap() as usize, *off, "{}.{name} offset", r.name);
                assert_eq!(f["size"].as_u64().unwrap() as usize, *size, "{}.{name} size", r.name);
            }
            println!(
                "| `{}` | {} / {} | {} / {} | {} | equal |",
                r.name,
                c["size"],
                r.size,
                c["align"],
                r.align,
                r.fields.len()
            );
        }
    }

    #[test]
    fn table_is_consistent() {
        assert_eq!(CUBINS.len(), 2);
        let mut keys = std::collections::HashSet::new();
        for k in &KERNELS {
            assert!(k.tu < CUBINS.len());
            assert!(keys.insert(k.key), "duplicate key {}", k.key);
        }
        assert_eq!(KernelId::KBump.info().key, "k_bump");
        assert_eq!(KPACK_ID.len(), 64);
    }

    #[test]
    fn images_are_checked_by_size_and_hash() {
        // FNV-1a 64 reference values: "" and "a"
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        let info = CubinInfo {
            tu: "engine",
            file: "x.cubin",
            bytes: 1,
            sha256: "",
            fnv1a64: fnv1a64(b"a"),
        };
        assert!(check_images(&[b"a"], &[info]).is_ok());
        assert!(check_images(&[b"b"], &[info]).is_err());
        assert!(check_images(&[b"ab"], &[info]).is_err());
        // built with a pack (CUDA image): the embedded cubins are the ones gen.rs was generated from
        if let Some(images) = EMBEDDED {
            check_images(&images, &CUBINS).unwrap();
        }
    }
}
