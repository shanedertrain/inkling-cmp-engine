// Kernel pack TU "engine" (own code, package R2, docs/RUST-PLAN.md 3.2): the forward-pass kernels of the C++ engine's
// engine.cu, compiled from the same headers (included in place, no copies) into pack_engine.cubin for the Rust runtime.
// The instances come from keys.toml [engine]: build.sh writes keep_engine.inc, one INK_KEEP(instance) line per key;
// taking each kernel's address in host code makes the front end instantiate exactly those (gate E0 checks that every
// kernel's SASS equals libinkrt_c.so's).
#include "../glue/engine_kernels.cuh"
#include "../glue/glue.cuh"
#include "../glue/moe_dev.cuh"
#include "../glue/prefill_glue.cuh"
#include "../sample/sample_kernels.cuh"
#include "../decode_attn/decode_attn.cuh"
#include "../prefill_attn/prefill_attn.cuh"
#include "../prefill_moe/prefill_moe.cuh"

#define INK_KEEP(...) reinterpret_cast<const void*>(&__VA_ARGS__),
extern const void* const ink_pack_keep_engine[];
const void* const ink_pack_keep_engine[] = {
#include "keep_engine.inc"
};
