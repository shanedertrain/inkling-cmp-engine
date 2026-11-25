// Kernel pack TU "drafter" (own code, package R2, docs/RUST-PLAN.md 3.2): the DSpark drafter kernels of the C++
// engine's draft.cu, compiled from the same header (drafter_rt.cuh, included in place) into pack_drafter.cubin.
// Instances: keys.toml [drafter] -> keep_drafter.inc (see pack_engine.cu).
#include "../drafter/drafter_pack.cuh"
#include "../dflash2/dflash2_kernels.cuh"

#define INK_KEEP(...) reinterpret_cast<const void*>(&__VA_ARGS__),
extern const void* const ink_pack_keep_drafter[];
const void* const ink_pack_keep_drafter[] = {
#include "keep_drafter.inc"
};
