# ink-cuda: safe CUDA layer (package R1, docs/RUST-PLAN.md 3.2)

State (2026-11-23): phase 1 done (crate, no-GPU tests, ABI check, CUDA image, spike built on sm_80 GPU host). Phase 2 (the
spike on one GPU) not run yet.

| Module | What | Ported from |
|---|---|---|
| `check` | R6 startup checks: loads `libcuda` / `libcublasLt` with cudarc's candidate list, resolves every symbol of `DRIVER_API` / `CUBLASLT_API`, driver >= 13030, cuBLASLt major 13; an error, never cudarc's panic | review V2 R6 |
| `context` | `Context` (primary context, retained once), `Stream` (non-blocking), `Event` (timing); length-checked `htod` / `htod_bytes` (unaligned host bytes, R4) / `dtoh` / `dtod` / `memset_zero`; host copies refused during capture | `engine.cu` @ a3209a0 |
| `arena` | `ArenaPlan` (bump allocator, 256-byte aligned, arena map), `Arena` (one `cuMemAlloc`, zeroed; `span`: raw byte range for the oracle dumps, R4), `DevBuf<T>` (`cast::<U>()`: same bytes as another element type, R3), `DevPtr<T>` (pointer fields of by-value structs) | `common.hpp` `Arena` @ a3209a0 |
| `module` | `Module` (cubin via `cuModuleLoadData`, `read_global`), `Function` (`cuFuncGetName`, max dynamic smem, regs), `KernelArgs`, `LaunchCfg`, `Stream::launch` / `launch_cooperative` (unsafe: kernel contract) | `humming_launch.hpp`, H2 harness |
| `graph` | `Stream::begin_capture` (thread-local) -> `Capture::mark(region)` -> `end()` -> `Graph` (node walk: kind, kernel name, dims, region; `set_node` / `set_region` enable/disable; `launch`) | `engine.cu` 306-315, 846-902 @ a3209a0 |
| `gemm`, `pins` | `LtGemm` (pinned algorithm per `GemmShape`, `AlgoCheck` per call shape, refusal = error, no fallback), `select` (the C++ heuristic rule, for the tuning mode), `AlgoPins` (`gemm-algos.json`, format `ink-gemm-algos/1`), `Bf16` | `gemm_lt.hpp` @ a3209a0 |
| `nvml` | `Sampler` (power, SM clock, `SW_POWER_CAP` flag, energy counter; own thread; 10 Hz `PERIOD_10HZ`), `summarize` | `decode_attn/bench/common.hpp` `Tele` |
| `abi` (test) | `abi/abi_check.out` (headers of `ink-ninfer-toolchain:cu1331`, CUDA 13.3.1, cuBLAS 13.6) vs cudarc `cuda-13030`: versioned symbols, sizes, offsets, enum values | plan 3.2 "ABI check" |

## cudarc: safe layer vs `result` / `sys`

| Need | Used | Why not the safe layer |
|---|---|---|
| Primary context, bind, release | safe `CudaContext::new` (event tracking off) | — |
| Stream, sync | safe `CudaStream` | — |
| Events, elapsed time | safe `CudaEvent` | — |
| Capture begin/end, instantiate, launch, graph RAII | safe `begin_capture`, `end_capture`, `CudaGraph` | — |
| Device memory | `result::malloc_sync` / `free_sync`, `memset_d8_async`, `memcpy_*_async` | `CudaSlice` is one allocation per buffer with per-buffer event tracking; we need one arena |
| Cubin load, function, global | `result::module::*` | safe `load_module` needs cudarc's `nvrtc` feature and takes PTX / a file |
| Launch | `result::launch_kernel`, `launch_cooperative_kernel` | `LaunchArgs` takes `CudaSlice` / `DeviceRepr`; arguments here are arena buffers and `Pod` structs (`KernelArgs`) |
| Function attributes | `result::function::*` | — (no safe wrapper on a raw function) |
| Capture marks, node walk, node enable, kernel names | `sys::cuStreamGetCaptureInfo_v3`, `cuGraphGetNodes`, `cuGraphGetEdges_v2`, `cuGraphNodeGetType`, `cuGraphKernelNodeGetParams_v2`, `cuGraphNodeSetEnabled`, `cuFuncGetName`, `cuKernelGetName` | the safe layer has no node API |
| cuBLASLt handle, descriptors, layouts, matmul | `cublaslt::result::*` | safe `CudaBlasLT` runs the heuristic on every call: no pinning |
| Algorithm init/attributes/check, heuristic list | `cublaslt::sys::cublasLtMatmulAlgo{Init,ConfigSetAttribute,ConfigGetAttribute,Check,GetHeuristic}` | not wrapped (result's heuristic returns one candidate) |
| Startup probe | `libloading` + `cudarc::get_lib_name_candidates`, `sys::cuDriverGetVersion`, `sys::cublasLtGetVersion` | cudarc panics on a missing library or symbol |

## Unsafe inventory (`src/`; every block has a `// SAFETY:` comment)

| File | Blocks | What |
|---|---|---|
| `check.rs` | 4 | `Library::new`, symbol presence, `cuDriverGetVersion`, `cublasLtGetVersion` |
| `context.rs` | 6 | event tracking off, htod / htod_bytes / dtoh / dtod / memset (lengths checked first) |
| `arena.rs` | 3 + 2 impls | `cuMemAlloc`, memset, `cuMemFree` in `Drop`; `Pod`/`Zeroable` for `DevPtr` |
| `module.rs` | 10 + 2 impls + 2 `unsafe fn` | load, get function/global, global read, attributes, names, `CStr`, launches; `Pod` for the argument chunk; `Stream::launch` / `launch_cooperative` are `unsafe fn` (argument list and kernel extents cannot be checked) |
| `graph.rs` | 10 | capture info, nodes, edges, node type, kernel params, set enabled |
| `gemm.rs` | 16 + 2 impls | cuBLASLt descriptors' `Drop`, heuristic, algo init/attributes/check, matmul (extents checked by `check_operands`); `Pod` for `Bf16` |

Raw handles and device addresses stay in the crate (`DevBuf::addr` is `pub(crate)`; `DevPtr` has no accessor).
`DevBuf`, `Function`, `Graph` and `LtGemm` hold the allocation, module or cuBLASLt handle they use; a capture collects
them so a graph keeps alive everything its nodes refer to. Every object holds the context, so it is released last.
`Context`, `Stream`, `Event`, `Arena`, `DevBuf`, `Module`, `Function`, `Graph`, `LtGemm` are not `Send` (compile-time test).
Eager launches hold no references, so the two owners that free device resources (the arena's allocation, a module)
bind their context and synchronise it in `Drop` before freeing (R3); `Arena::new` and `Module::load` bind their context
first, so one thread holding several contexts allocates and loads in the right one.

## Commands

| Step | Command |
|---|---|
| Tests, lint (desktop, no GPU) | `scripts/build.sh cargo test -p ink-cuda --offline --locked`; `... cargo clippy -p ink-cuda --all-targets --offline --locked -- -D warnings` |
| Build on sm_80 host (no GPU): spike cubin, `lt_ref`, ABI check, release spike, tests in the CUDA image | `spike/build.sh` (run on an sm_80 GPU host) |
| Phase 2 (desktop, one free card N) | `crates/ink-cuda/spike/phase2.sh N` -> `bench/raw/<date>-rust-R1-spike/<tag>/` |
| Regenerate `abi/abi_check.out` | `spike/build.sh` writes `target/r1/abi_check.out` on an sm_80 host and fails if it differs from the committed file |
