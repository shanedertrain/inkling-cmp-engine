//! ABI test: `abi/abi_check.out` (what the CUDA 13.3.1 headers of `ink-ninfer-toolchain:cu1331` say, written by
//! `abi/abi_check.c`) against cudarc's `cuda-13030` bindings and this crate's symbol tables. No GPU needed.

use std::collections::BTreeMap;
use std::mem::{align_of, offset_of, size_of};

use cudarc::cublaslt::sys as lt;
use cudarc::driver::sys as drv;

use crate::check::{CUBLASLT_API, CUBLASLT_MAJOR, DRIVER_API, MIN_DRIVER_VERSION};
use crate::gemm::{CUBLAS_OP_N, CUBLAS_OP_T};

const HEADERS: &str = include_str!("../abi/abi_check.out");

/// Header lines that are informational only (not a property of the bindings).
const INFORMATIONAL: &[&str] = &["macro CUBLAS_VER_MINOR", "macro CUBLAS_VERSION"];

macro_rules! layout {
    ($m:ident, $t:ident) => {
        $m.insert(
            concat!("size ", stringify!($t)).to_owned(),
            size_of::<drv_or_lt::$t>() as i64,
        );
        $m.insert(
            concat!("align ", stringify!($t)).to_owned(),
            align_of::<drv_or_lt::$t>() as i64,
        );
    };
}

/// One module that names every struct the header lines mention (driver and cuBLASLt types do not collide).
mod drv_or_lt {
    pub use cudarc::cublaslt::sys::{cublasLtMatmulAlgo_t, cublasLtMatmulHeuristicResult_t};
    pub use cudarc::driver::sys::{
        CUDA_KERNEL_NODE_PARAMS, CUDA_MEMCPY3D, CUDA_MEMSET_NODE_PARAMS, CUgraphEdgeData, CUuuid,
    };
}

fn rust_side() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("macro CUDA_VERSION".to_owned(), i64::from(MIN_DRIVER_VERSION));
    m.insert("macro CUBLAS_VER_MAJOR".to_owned(), CUBLASLT_MAJOR as i64);
    layout!(m, CUuuid);
    layout!(m, CUDA_KERNEL_NODE_PARAMS);
    layout!(m, CUgraphEdgeData);
    layout!(m, CUDA_MEMCPY3D);
    layout!(m, CUDA_MEMSET_NODE_PARAMS);
    layout!(m, cublasLtMatmulAlgo_t);
    layout!(m, cublasLtMatmulHeuristicResult_t);
    macro_rules! off {
        ($t:ident, $($f:ident),+) => {
            $(m.insert(concat!("offset ", stringify!($t), ".", stringify!($f)).to_owned(),
                       offset_of!(drv_or_lt::$t, $f) as i64);)+
        };
    }
    off!(
        CUDA_KERNEL_NODE_PARAMS,
        func,
        gridDimX,
        blockDimX,
        sharedMemBytes,
        kernelParams,
        extra,
        kern,
        ctx
    );
    off!(
        CUDA_MEMCPY3D,
        srcXInBytes,
        srcMemoryType,
        srcHost,
        srcDevice,
        srcArray,
        srcPitch,
        srcHeight,
        dstXInBytes,
        dstMemoryType,
        dstHost,
        dstDevice,
        dstArray,
        dstPitch,
        dstHeight,
        WidthInBytes,
        Height,
        Depth
    );
    off!(CUDA_MEMSET_NODE_PARAMS, dst, pitch, value, elementSize, width, height);
    off!(cublasLtMatmulHeuristicResult_t, algo, workspaceSize, state, wavesCount);
    macro_rules! en {
        ($($name:ident = $v:expr),+ $(,)?) => {
            $(m.insert(concat!("enum ", stringify!($name)).to_owned(), $v as i64);)+
        };
    }
    en!(
        CUDA_SUCCESS = drv::CUresult::CUDA_SUCCESS,
        CUDA_ERROR_NOT_SUPPORTED = drv::CUresult::CUDA_ERROR_NOT_SUPPORTED,
        CUDA_ERROR_INVALID_VALUE = drv::CUresult::CUDA_ERROR_INVALID_VALUE,
        CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = drv::CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
        CU_STREAM_NON_BLOCKING = drv::CUstream_flags::CU_STREAM_NON_BLOCKING,
        CU_EVENT_DEFAULT = drv::CUevent_flags::CU_EVENT_DEFAULT,
        CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = drv::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        CU_STREAM_CAPTURE_STATUS_NONE = drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
        CU_STREAM_CAPTURE_STATUS_ACTIVE = drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE,
        CU_GRAPH_NODE_TYPE_KERNEL = drv::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL,
        CU_GRAPH_NODE_TYPE_MEMCPY = drv::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY,
        CU_GRAPH_NODE_TYPE_MEMSET = drv::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET,
        CU_MEMORYTYPE_HOST = drv::CUmemorytype::CU_MEMORYTYPE_HOST,
        CU_MEMORYTYPE_DEVICE = drv::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES =
            drv::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        CU_FUNC_ATTRIBUTE_NUM_REGS = drv::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR =
            drv::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR =
            drv::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = drv::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = drv::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
        CUBLAS_STATUS_SUCCESS = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        CUBLAS_STATUS_NOT_SUPPORTED = lt::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED,
        CUBLAS_STATUS_INVALID_VALUE = lt::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
        CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES =
            lt::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,
        CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES =
            lt::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES =
            lt::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,
        CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES =
            lt::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES,
        CUBLAS_OP_N = CUBLAS_OP_N,
        CUBLAS_OP_T = CUBLAS_OP_T,
        CUBLAS_COMPUTE_32F = lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        CUDA_R_32F = lt::cudaDataType::CUDA_R_32F,
        CUDA_R_16BF = lt::cudaDataType::CUDA_R_16BF,
        CUBLASLT_MATMUL_DESC_TRANSA = lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        CUBLASLT_MATMUL_DESC_TRANSB = lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES =
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        CUBLASLT_ALGO_CONFIG_ID = lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_ID,
        CUBLASLT_ALGO_CONFIG_TILE_ID = lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_TILE_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM = lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME =
            lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING =
            lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION =
            lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        CUBLASLT_ALGO_CONFIG_STAGES_ID = lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_STAGES_ID,
        CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID =
            lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
        CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID =
            lt::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = lt::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET =
            lt::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
    );
    m
}

#[test]
fn bindings_match_the_image_headers() {
    let rust = rust_side();
    let symbols: BTreeMap<&str, &str> = DRIVER_API.iter().chain(CUBLASLT_API).copied().collect();
    let mut seen_values = Vec::new();
    let mut seen_symbols = Vec::new();
    let mut bad = Vec::new();
    for line in HEADERS.lines().filter(|l| !l.trim().is_empty()) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let [kind, name, value] = parts[..] else {
            panic!("malformed line {line:?}")
        };
        if kind == "symbol" {
            seen_symbols.push(name);
            match symbols.get(name) {
                Some(&ours) if ours == value => {}
                Some(&ours) => bad.push(format!("{name}: header binds {value}, cudarc calls {ours}")),
                None => bad.push(format!("{name}: in abi_check.c but not in DRIVER_API/CUBLASLT_API")),
            }
            continue;
        }
        let key = format!("{kind} {name}");
        if INFORMATIONAL.contains(&key.as_str()) {
            continue;
        }
        let header: i64 = value.parse().unwrap_or_else(|_| panic!("bad value in {line:?}"));
        match rust.get(&key) {
            Some(&r) if r == header => {}
            Some(&r) => bad.push(format!("{key}: header {header}, bindings {r}")),
            None => bad.push(format!("{key}: no Rust-side value")),
        }
        seen_values.push(key);
    }
    for (name, _) in DRIVER_API.iter().chain(CUBLASLT_API) {
        if !seen_symbols.contains(name) {
            bad.push(format!("{name}: called but not checked by abi_check.c"));
        }
    }
    for key in rust.keys() {
        if !seen_values.contains(key) {
            bad.push(format!("{key}: not in abi_check.out"));
        }
    }
    assert!(bad.is_empty(), "ABI mismatches:\n{}", bad.join("\n"));
}
