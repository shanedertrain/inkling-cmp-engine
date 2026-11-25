# ink-model: loader and memory plan (package R4, docs/RUST-PLAN.md)

Inkling-2-Mini on the device: dimensions, visit plan, the checkpoint by mmap, small parameters to fp32, expert
repack, upload into the `ink-cuda` arena, and the buffers of `Engine::alloc`. Everything is taken from the arena in
the C++ engine's order with the C++ names (native 0.2.0 @ `66fe353`), so the same offsets come out and gate E1
compares one dump line by line. No C++ is replaced yet: the loader is cut over with the runtime (U4).

| Module | What | Ported from (@ 66fe353) |
|---|---|---|
| `dims` | constants, `is_full_layer`, `Visit`, `visit_plan` (48 visits) | `engine/native/src/model.hpp` |
| `checkpoint` | `Checkpoint` (memmap2 + `SafeTensors::read_metadata`, candle's pattern without the self-referential wrapper), `Tensor` views, `to_f32` (F32 copy, BF16 widened, exact) | `weights.cpp` `Mapped`, `host_f32` |
| `repack` | `enc_scale`, `orig_row`, `repack_nvfp4_n1`, `repack_mxfp8` (split by expert over threads, as `par`) | `weights.cpp` = `engine/kernels/prefill_moe/bench.cu` loops |
| `weights` | `Weights` / `LayerW` / `Controller` / `ExpertW`, `Weights::load` (upload order and arena names of `load_weights`), `regions()` (the dump names of `dumps.cpp` `weight_regions`), `transpose_gate`, `nvfp4_factors`, `interleave_msg_queries` | `model.hpp`, `weights.cpp`, `capi/dumps.cpp` |
| `plan` | `MemConfig` (`arena_bytes` = the constructor's sum), `Alloc` (implemented by `Arena` -> `DevBuf`, `ArenaPlan` -> offsets), `Activations::alloc` (every take of `Engine::alloc`, C++ order and names, Humming buffers when `humming`) | `engine.cu` `Engine::Engine`, `Engine::alloc`; `moe_dev.cuh` `items_bound`; `hum_dev.hpp` `sorted_len` |
| `dump` | `fnv1a64`, `weight_sums` (device read-back in 64 MiB copies, each region to the next one, the C++ checks), `weight_sums_text` (`ink-weight-sums/1`), `arena_map_text` (`ink-arena-map/1`, the format hook H1 defines) | `capi/dumps.cpp`, `dumps.hpp` |

The gate tool is `ink-tools` bin `ink-load` (load, plan, both dumps, line-by-line comparison with the C++ oracle's
`e1-weight-sums.jsonl`, `gate.json`, an `ink-record` record with load time and `VmHWM` as side facts).

## Unsafe

One block: `Mmap::map` in `Checkpoint::open` (the file must not change while mapped; the model directory is mounted
read-only in every container). Device copies go through `ink-cuda` (`Stream::htod_bytes`, added for checkpoint
tensors that are not aligned for their element type; `Arena::span` for the dump read-back).

## Tests (no GPU)

`scripts/build.sh cargo test -p ink-model --offline --locked`. Expected values are derived by hand from the C++
code, not from this port: repack bit and byte positions on one 64 x 64 block (code bits, sign flip of every code of a
negative block, zero scale, w13 and w2 rows, MXFP8 byte order and scales), `enc_scale` incl. subnormals, scale range
refusal, thread count invariance; the arena size term by term (= the oracle log's 35.29 GiB) and the first offsets of
the plan; visit plan; FNV-1a 64 reference vectors and the dump line formats against the oracle file's first line.
