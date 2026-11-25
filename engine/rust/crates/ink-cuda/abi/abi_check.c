/* abi_check.c: what the CUDA 13.3.1 headers of ink-ninfer-toolchain:cu1331 say about every driver / cuBLASLt item
 * ink-cuda uses (docs/RUST-PLAN.md 3.2 "ABI check", risk 9). Prints one line per item:
 *   symbol <api name> <symbol the header binds it to>    (versioned names: cuMemAlloc -> cuMemAlloc_v2)
 *   size   <type> <sizeof>
 *   align  <type> <alignof>
 *   offset <type>.<field> <offsetof>
 *   enum   <name> <value>
 *   macro  <name> <value>
 * The output is committed as abi_check.out; src/abi.rs compares it with cudarc's cuda-13030 bindings (no GPU needed).
 * Build and run (no GPU): spike/build.sh. Own code. */
#include <cublasLt.h>
#include <cuda.h>
#include <stddef.h>
#include <stdio.h>

#define STR_(x) #x
#define STR(x) STR_(x)
/* the header maps an API name to its versioned symbol with a macro; STR expands it once fully */
#define SYM(api) printf("symbol %s %s\n", #api, STR(api))
#define SIZE(t) printf("size %s %zu\nalign %s %zu\n", #t, sizeof(t), #t, _Alignof(t))
#define OFF(t, f) printf("offset %s.%s %zu\n", #t, #f, offsetof(t, f))
#define ENUM(e) printf("enum %s %lld\n", #e, (long long)(e))
#define MACRO(m) printf("macro %s %lld\n", #m, (long long)(m))

int main(void) {
  MACRO(CUDA_VERSION);
  MACRO(CUBLAS_VER_MAJOR);
  MACRO(CUBLAS_VER_MINOR);
  MACRO(CUBLAS_VERSION);

  /* driver API: every function ink-cuda calls, directly or through cudarc's result/safe layer */
  SYM(cuInit);
  SYM(cuDriverGetVersion);
  SYM(cuGetErrorName);
  SYM(cuGetErrorString);
  SYM(cuDeviceGet);
  SYM(cuDeviceGetAttribute);
  SYM(cuDeviceGetName);
  SYM(cuDeviceGetUuid);
  SYM(cuDevicePrimaryCtxRetain);
  SYM(cuDevicePrimaryCtxRelease);
  SYM(cuCtxGetCurrent);
  SYM(cuCtxSetCurrent);
  SYM(cuMemGetInfo);
  SYM(cuStreamCreate);
  SYM(cuStreamDestroy);
  SYM(cuStreamSynchronize);
  SYM(cuEventCreate);
  SYM(cuEventRecord);
  SYM(cuEventSynchronize);
  SYM(cuEventElapsedTime);
  SYM(cuEventDestroy);
  SYM(cuMemAlloc);
  SYM(cuMemFree);
  SYM(cuMemsetD8);
  SYM(cuMemsetD8Async);
  SYM(cuMemsetD2D32Async);
  SYM(cuMemcpyHtoDAsync);
  SYM(cuMemcpyDtoHAsync);
  SYM(cuMemcpyDtoDAsync);
  SYM(cuModuleLoadData);
  SYM(cuModuleUnload);
  SYM(cuModuleGetFunction);
  SYM(cuModuleGetGlobal);
  SYM(cuFuncGetAttribute);
  SYM(cuFuncSetAttribute);
  SYM(cuFuncGetName);
  SYM(cuKernelGetName);
  SYM(cuFuncGetParamInfo);
  SYM(cuKernelGetParamInfo);
  SYM(cuOccupancyMaxActiveBlocksPerMultiprocessor);
  SYM(cuLaunchKernel);
  SYM(cuLaunchCooperativeKernel);
  SYM(cuStreamBeginCapture);
  SYM(cuStreamEndCapture);
  SYM(cuStreamGetCaptureInfo);
  SYM(cuGraphInstantiateWithFlags);
  SYM(cuGraphLaunch);
  SYM(cuGraphExecDestroy);
  SYM(cuGraphDestroy);
  SYM(cuGraphGetNodes);
  SYM(cuGraphGetEdges);
  SYM(cuGraphNodeGetType);
  SYM(cuGraphKernelNodeGetParams);
  SYM(cuGraphKernelNodeSetParams);
  SYM(cuGraphExecKernelNodeSetParams);
  SYM(cuGraphMemsetNodeGetParams);
  SYM(cuGraphNodeSetEnabled);
  SYM(cuGraphMemcpyNodeGetParams);

  /* cuBLASLt */
  SYM(cublasLtGetVersion);
  SYM(cublasLtCreate);
  SYM(cublasLtDestroy);
  SYM(cublasLtMatmulDescCreate);
  SYM(cublasLtMatmulDescSetAttribute);
  SYM(cublasLtMatmulDescDestroy);
  SYM(cublasLtMatrixLayoutCreate);
  SYM(cublasLtMatrixLayoutSetAttribute);
  SYM(cublasLtMatrixLayoutDestroy);
  SYM(cublasLtMatmulPreferenceCreate);
  SYM(cublasLtMatmulPreferenceSetAttribute);
  SYM(cublasLtMatmulPreferenceDestroy);
  SYM(cublasLtMatmulAlgoGetHeuristic);
  SYM(cublasLtMatmulAlgoInit);
  SYM(cublasLtMatmulAlgoConfigSetAttribute);
  SYM(cublasLtMatmulAlgoConfigGetAttribute);
  SYM(cublasLtMatmulAlgoCheck);
  SYM(cublasLtMatmulAlgoCapGetAttribute);
  SYM(cublasLtMatmul);

  /* structs read or written across the boundary */
  SIZE(CUuuid);
  SIZE(CUDA_KERNEL_NODE_PARAMS);
  OFF(CUDA_KERNEL_NODE_PARAMS, func);
  OFF(CUDA_KERNEL_NODE_PARAMS, gridDimX);
  OFF(CUDA_KERNEL_NODE_PARAMS, blockDimX);
  OFF(CUDA_KERNEL_NODE_PARAMS, sharedMemBytes);
  OFF(CUDA_KERNEL_NODE_PARAMS, kernelParams);
  OFF(CUDA_KERNEL_NODE_PARAMS, extra);
  OFF(CUDA_KERNEL_NODE_PARAMS, kern);
  OFF(CUDA_KERNEL_NODE_PARAMS, ctx);
  SIZE(CUgraphEdgeData);
  SIZE(CUDA_MEMCPY3D);
  OFF(CUDA_MEMCPY3D, srcXInBytes);
  OFF(CUDA_MEMCPY3D, srcMemoryType);
  OFF(CUDA_MEMCPY3D, srcHost);
  OFF(CUDA_MEMCPY3D, srcDevice);
  OFF(CUDA_MEMCPY3D, srcArray);
  OFF(CUDA_MEMCPY3D, srcPitch);
  OFF(CUDA_MEMCPY3D, srcHeight);
  OFF(CUDA_MEMCPY3D, dstXInBytes);
  OFF(CUDA_MEMCPY3D, dstMemoryType);
  OFF(CUDA_MEMCPY3D, dstHost);
  OFF(CUDA_MEMCPY3D, dstDevice);
  OFF(CUDA_MEMCPY3D, dstArray);
  OFF(CUDA_MEMCPY3D, dstPitch);
  OFF(CUDA_MEMCPY3D, dstHeight);
  OFF(CUDA_MEMCPY3D, WidthInBytes);
  OFF(CUDA_MEMCPY3D, Height);
  OFF(CUDA_MEMCPY3D, Depth);
  SIZE(CUDA_MEMSET_NODE_PARAMS);
  OFF(CUDA_MEMSET_NODE_PARAMS, dst);
  OFF(CUDA_MEMSET_NODE_PARAMS, pitch);
  OFF(CUDA_MEMSET_NODE_PARAMS, value);
  OFF(CUDA_MEMSET_NODE_PARAMS, elementSize);
  OFF(CUDA_MEMSET_NODE_PARAMS, width);
  OFF(CUDA_MEMSET_NODE_PARAMS, height);
  SIZE(cublasLtMatmulAlgo_t);
  SIZE(cublasLtMatmulHeuristicResult_t);
  OFF(cublasLtMatmulHeuristicResult_t, algo);
  OFF(cublasLtMatmulHeuristicResult_t, workspaceSize);
  OFF(cublasLtMatmulHeuristicResult_t, state);
  OFF(cublasLtMatmulHeuristicResult_t, wavesCount);

  /* enum values passed or compared */
  ENUM(CUDA_SUCCESS);
  ENUM(CUDA_ERROR_NOT_SUPPORTED);
  ENUM(CUDA_ERROR_INVALID_VALUE);
  ENUM(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
  ENUM(CU_STREAM_NON_BLOCKING);
  ENUM(CU_EVENT_DEFAULT);
  ENUM(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
  ENUM(CU_STREAM_CAPTURE_STATUS_NONE);
  ENUM(CU_STREAM_CAPTURE_STATUS_ACTIVE);
  ENUM(CU_GRAPH_NODE_TYPE_KERNEL);
  ENUM(CU_GRAPH_NODE_TYPE_MEMCPY);
  ENUM(CU_GRAPH_NODE_TYPE_MEMSET);
  ENUM(CU_MEMORYTYPE_HOST);
  ENUM(CU_MEMORYTYPE_DEVICE);
  ENUM(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
  ENUM(CU_FUNC_ATTRIBUTE_NUM_REGS);
  ENUM(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
  ENUM(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
  ENUM(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  ENUM(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH);
  ENUM(CUBLAS_STATUS_SUCCESS);
  ENUM(CUBLAS_STATUS_NOT_SUPPORTED);
  ENUM(CUBLAS_STATUS_INVALID_VALUE);
  ENUM(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES);
  ENUM(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES);
  ENUM(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES);
  ENUM(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES);
  ENUM(CUBLAS_OP_N);
  ENUM(CUBLAS_OP_T);
  ENUM(CUBLAS_COMPUTE_32F);
  ENUM(CUDA_R_32F);
  ENUM(CUDA_R_16BF);
  ENUM(CUBLASLT_MATMUL_DESC_TRANSA);
  ENUM(CUBLASLT_MATMUL_DESC_TRANSB);
  ENUM(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES);
  ENUM(CUBLASLT_ALGO_CONFIG_ID);
  ENUM(CUBLASLT_ALGO_CONFIG_TILE_ID);
  ENUM(CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
  ENUM(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
  ENUM(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
  ENUM(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
  ENUM(CUBLASLT_ALGO_CONFIG_STAGES_ID);
  ENUM(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID);
  ENUM(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
  ENUM(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT);
  ENUM(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET);
  return 0;
}
