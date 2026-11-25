//! Startup checks (review V2 R6): the right libraries, versions and symbols, as an error instead of a later panic.
//!
//! cudarc loads `libcuda` and `libcublasLt` on first use, takes the first of its candidate file names that loads
//! (including `.so.12`, `.so.11`, ...), and panics when a symbol is missing at its first call. [`startup_checks`] runs
//! before any cudarc call: it loads each library the way cudarc will (same candidate list, same order), resolves every
//! symbol this crate calls, and checks the driver (CUDA >= 13.3) and cuBLASLt (major 13) versions.

use libloading::Library;

use crate::error::{CudaError, Result};

/// Lowest driver API version the `cuda-13030` bindings are valid for (`cuDriverGetVersion`).
pub const MIN_DRIVER_VERSION: i32 = 13030;
/// cuBLASLt major version the bindings are generated for (`cublasLtGetVersion() / 10000`).
pub const CUBLASLT_MAJOR: usize = 13;

/// Every driver API function this crate calls, directly or through cudarc's `result` and `safe` layers:
/// `(name in the CUDA docs, symbol cudarc resolves)`. `abi_check.out` must bind each name to the same symbol.
pub(crate) const DRIVER_API: &[(&str, &str)] = &[
    ("cuInit", "cuInit"),
    ("cuDriverGetVersion", "cuDriverGetVersion"),
    ("cuGetErrorName", "cuGetErrorName"),
    ("cuGetErrorString", "cuGetErrorString"),
    ("cuDeviceGet", "cuDeviceGet"),
    ("cuDeviceGetAttribute", "cuDeviceGetAttribute"),
    ("cuDeviceGetName", "cuDeviceGetName"),
    ("cuDeviceGetUuid", "cuDeviceGetUuid_v2"),
    ("cuDevicePrimaryCtxRetain", "cuDevicePrimaryCtxRetain"),
    ("cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2"),
    ("cuCtxGetCurrent", "cuCtxGetCurrent"),
    ("cuCtxSetCurrent", "cuCtxSetCurrent"),
    ("cuMemGetInfo", "cuMemGetInfo_v2"),
    ("cuStreamCreate", "cuStreamCreate"),
    ("cuStreamDestroy", "cuStreamDestroy_v2"),
    ("cuStreamSynchronize", "cuStreamSynchronize"),
    ("cuEventCreate", "cuEventCreate"),
    ("cuEventRecord", "cuEventRecord"),
    ("cuEventSynchronize", "cuEventSynchronize"),
    ("cuEventElapsedTime", "cuEventElapsedTime_v2"),
    ("cuEventDestroy", "cuEventDestroy_v2"),
    ("cuMemAlloc", "cuMemAlloc_v2"),
    ("cuMemFree", "cuMemFree_v2"),
    ("cuMemsetD8", "cuMemsetD8_v2"),
    ("cuMemsetD8Async", "cuMemsetD8Async"),
    ("cuMemsetD2D32Async", "cuMemsetD2D32Async"),
    ("cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2"),
    ("cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2"),
    ("cuMemcpyDtoDAsync", "cuMemcpyDtoDAsync_v2"),
    ("cuModuleLoadData", "cuModuleLoadData"),
    ("cuModuleUnload", "cuModuleUnload"),
    ("cuModuleGetFunction", "cuModuleGetFunction"),
    ("cuModuleGetGlobal", "cuModuleGetGlobal_v2"),
    ("cuFuncGetAttribute", "cuFuncGetAttribute"),
    ("cuFuncSetAttribute", "cuFuncSetAttribute"),
    ("cuFuncGetName", "cuFuncGetName"),
    ("cuKernelGetName", "cuKernelGetName"),
    ("cuFuncGetParamInfo", "cuFuncGetParamInfo"),
    ("cuKernelGetParamInfo", "cuKernelGetParamInfo"),
    (
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    ),
    ("cuLaunchKernel", "cuLaunchKernel"),
    ("cuLaunchCooperativeKernel", "cuLaunchCooperativeKernel"),
    ("cuStreamBeginCapture", "cuStreamBeginCapture_v2"),
    ("cuStreamEndCapture", "cuStreamEndCapture"),
    ("cuStreamGetCaptureInfo", "cuStreamGetCaptureInfo_v3"),
    ("cuGraphInstantiateWithFlags", "cuGraphInstantiateWithFlags"),
    ("cuGraphLaunch", "cuGraphLaunch"),
    ("cuGraphExecDestroy", "cuGraphExecDestroy"),
    ("cuGraphDestroy", "cuGraphDestroy"),
    ("cuGraphGetNodes", "cuGraphGetNodes"),
    ("cuGraphGetEdges", "cuGraphGetEdges_v2"),
    ("cuGraphNodeGetType", "cuGraphNodeGetType"),
    ("cuGraphKernelNodeGetParams", "cuGraphKernelNodeGetParams_v2"),
    ("cuGraphKernelNodeSetParams", "cuGraphKernelNodeSetParams_v2"),
    ("cuGraphExecKernelNodeSetParams", "cuGraphExecKernelNodeSetParams_v2"),
    ("cuGraphMemsetNodeGetParams", "cuGraphMemsetNodeGetParams"),
    ("cuGraphNodeSetEnabled", "cuGraphNodeSetEnabled"),
    ("cuGraphMemcpyNodeGetParams", "cuGraphMemcpyNodeGetParams"),
];

/// Every cuBLASLt function this crate calls: `(name in the docs, symbol)`.
pub(crate) const CUBLASLT_API: &[(&str, &str)] = &[
    ("cublasLtGetVersion", "cublasLtGetVersion"),
    ("cublasLtCreate", "cublasLtCreate"),
    ("cublasLtDestroy", "cublasLtDestroy"),
    ("cublasLtMatmulDescCreate", "cublasLtMatmulDescCreate"),
    ("cublasLtMatmulDescSetAttribute", "cublasLtMatmulDescSetAttribute"),
    ("cublasLtMatmulDescDestroy", "cublasLtMatmulDescDestroy"),
    ("cublasLtMatrixLayoutCreate", "cublasLtMatrixLayoutCreate"),
    ("cublasLtMatrixLayoutSetAttribute", "cublasLtMatrixLayoutSetAttribute"),
    ("cublasLtMatrixLayoutDestroy", "cublasLtMatrixLayoutDestroy"),
    ("cublasLtMatmulPreferenceCreate", "cublasLtMatmulPreferenceCreate"),
    (
        "cublasLtMatmulPreferenceSetAttribute",
        "cublasLtMatmulPreferenceSetAttribute",
    ),
    ("cublasLtMatmulPreferenceDestroy", "cublasLtMatmulPreferenceDestroy"),
    ("cublasLtMatmulAlgoGetHeuristic", "cublasLtMatmulAlgoGetHeuristic"),
    ("cublasLtMatmulAlgoInit", "cublasLtMatmulAlgoInit"),
    (
        "cublasLtMatmulAlgoConfigSetAttribute",
        "cublasLtMatmulAlgoConfigSetAttribute",
    ),
    (
        "cublasLtMatmulAlgoConfigGetAttribute",
        "cublasLtMatmulAlgoConfigGetAttribute",
    ),
    ("cublasLtMatmulAlgoCheck", "cublasLtMatmulAlgoCheck"),
    ("cublasLtMatmulAlgoCapGetAttribute", "cublasLtMatmulAlgoCapGetAttribute"),
    ("cublasLtMatmul", "cublasLtMatmul"),
];

/// What [`startup_checks`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupReport {
    /// `cuDriverGetVersion`, e.g. 13030.
    pub driver_version: i32,
    /// The driver library file name that loaded.
    pub driver_file: String,
    /// `cublasLtGetVersion`, e.g. 130600.
    pub cublaslt_version: usize,
    /// The cuBLASLt library file name that loaded.
    pub cublaslt_file: String,
    /// Number of symbols resolved in both libraries.
    pub symbols: usize,
}

impl std::fmt::Display for StartupReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "driver {} ({}), cublasLt {} ({}), {} symbols resolved",
            self.driver_version, self.driver_file, self.cublaslt_version, self.cublaslt_file, self.symbols
        )
    }
}

/// Loads the driver and cuBLASLt as cudarc will, resolves every symbol of [`DRIVER_API`] and [`CUBLASLT_API`] and
/// checks both versions.
///
/// # Errors
///
/// [`CudaError::LibraryNotFound`], [`CudaError::MissingSymbol`], [`CudaError::Version`], or a driver error of
/// `cuDriverGetVersion`.
pub fn startup_checks() -> Result<StartupReport> {
    let (driver_file, n_driver) = probe("cuda", DRIVER_API)?;
    let mut driver_version = 0;
    // SAFETY: probe() resolved cuDriverGetVersion in the same library cudarc loads (same candidates, same order), so
    // the call cannot panic; the out-pointer is a valid local.
    let status = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) };
    crate::error::check_driver("cuDriverGetVersion", status)?;
    if driver_version < MIN_DRIVER_VERSION {
        return Err(CudaError::Version {
            what: "CUDA driver",
            found: driver_version.to_string(),
            need: ">= 13030",
        });
    }
    let (cublaslt_file, n_lt) = probe("cublasLt", CUBLASLT_API)?;
    // SAFETY: as above, cublasLtGetVersion was resolved in the library cudarc loads; it takes no arguments.
    let cublaslt_version = unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() };
    if cublaslt_version / 10000 != CUBLASLT_MAJOR {
        return Err(CudaError::Version {
            what: "cuBLASLt",
            found: cublaslt_version.to_string(),
            need: "major 13",
        });
    }
    Ok(StartupReport {
        driver_version,
        driver_file,
        cublaslt_version,
        cublaslt_file,
        symbols: n_driver + n_lt,
    })
}

/// Opens the first loadable candidate of `lib` in cudarc's order and resolves every symbol of `api` in it.
fn probe(lib: &'static str, api: &[(&str, &'static str)]) -> Result<(String, usize)> {
    let tried = cudarc::get_lib_name_candidates(lib);
    // SAFETY: loading the CUDA driver or cuBLASLt runs their initialisers; this is the load cudarc itself does
    // on first use, done earlier so that a failure is an error instead of a panic.
    let found = tried
        .iter()
        .find_map(|name| unsafe { Library::new(name) }.ok().map(|l| (name.clone(), l)));
    let Some((file, library)) = found else {
        return Err(CudaError::LibraryNotFound { lib, tried });
    };
    for &(_, symbol) in api {
        // SAFETY: only the presence of the symbol is checked; the returned pointer is never called or kept.
        let present = unsafe { library.get::<*const ()>(symbol) }.is_ok();
        if !present {
            return Err(CudaError::MissingSymbol { file, symbol });
        }
    }
    Ok((file, api.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_have_no_duplicates() {
        for table in [DRIVER_API, CUBLASLT_API] {
            let mut names: Vec<&str> = table.iter().map(|e| e.0).collect();
            names.sort_unstable();
            let n = names.len();
            names.dedup();
            assert_eq!(n, names.len(), "duplicate API name");
        }
    }

    /// Without a GPU (desktop container) the driver is absent: the check must report it, not panic.
    #[test]
    fn startup_checks_never_panic() {
        match startup_checks() {
            Ok(r) => assert!(r.driver_version >= MIN_DRIVER_VERSION && r.symbols > 0),
            Err(CudaError::LibraryNotFound { lib, tried }) => {
                assert_eq!(lib, "cuda");
                assert!(tried.iter().any(|t| t == "libcuda.so.1"), "{tried:?}");
            }
            Err(e) => panic!("unexpected startup error: {e}"),
        }
    }

    /// Every `sys::cu*` / `cublasLt*` function the sources call directly is in a table (and so is checked at startup
    /// and against the headers).
    #[test]
    fn direct_sys_calls_are_listed() {
        let sources = [
            include_str!("arena.rs"),
            include_str!("context.rs"),
            include_str!("gemm.rs"),
            include_str!("graph.rs"),
            include_str!("module.rs"),
            include_str!("trace.rs"),
            include_str!("check.rs"),
        ];
        let known: Vec<&str> = DRIVER_API.iter().chain(CUBLASLT_API).map(|e| e.1).collect();
        for src in sources {
            for (i, _) in src.match_indices("sys::") {
                let rest = &src[i + 5..];
                let ident: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                let is_call = rest[ident.len()..].starts_with('(');
                if is_call && (ident.starts_with("cu") || ident.starts_with("cublasLt")) {
                    assert!(
                        known.contains(&ident.as_str()),
                        "{ident} is called but not in DRIVER_API/CUBLASLT_API"
                    );
                }
            }
        }
    }
}
