// inkling-cmp-engine native runtime: the engine's own small kernels (own code), moved verbatim out of engine.cu
// (package R2, docs/RUST-PLAN.md 3.2) so that engine.cu and the kernel pack (engine/kernels/pack/) compile the
// same text. They stay in ink's anonymous namespace, as in engine.cu: their mangled names carry a per-TU id.
#pragma once
#include "glue.cuh"
#include "../common/model.hpp"

namespace ink {
namespace {

__global__ void rmsnorm_f32out_kernel(const float* in, const float* w, float* out, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x;
  const float* x = in + (size_t)t * H;
  float v[1] = {0.f};
  for (int i = threadIdx.x; i < H; i += blockDim.x) v[0] = __fadd_rn(v[0], __fmul_rn(x[i], x[i]));
  glue::block_sum<1>(v, sh);
  const float r = glue::rstd_of(v[0], H, eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x) out[(size_t)t * H + i] = __fmul_rn(__fmul_rn(x[i], r), w[i]);
}

// first-index argmax of each logit row over the unpadded vocabulary (NaN never wins)
__global__ void argmax_rows_kernel(const float* lg, int ld, int vocab, int* out) {
  const int r = blockIdx.x;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int c = threadIdx.x; c < vocab; c += blockDim.x) {
    const float x = lg[(size_t)r * ld + c];
    if (x > bv || (x == bv && c < bi)) { bv = x; bi = c; }
  }
  __shared__ float sv[32];
  __shared__ int si[32];
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < (int)blockDim.x / 32; ++w)
      if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
    out[r] = bi == 0x7fffffff ? 0 : bi;
  }
}
// greedy accept (one thread per segment; thread 0 packs the commit rows afterwards)
// segment s (rows row0[s] .. next row0 or T) verifies its first rows - 1 drafts (ragged widths)
// K: drafts per slot in `drafts` ([slots][K]; 7 = the drafter's, 15 = the drafter's 7 + 8 lookup drafts); a draft < 0
// is never accepted. res[s * (K + 2)] = a, then the a + 1 emitted tokens; commit rows packed, count at res[S * (K + 2)].
// forced (optional, [slots][K + 1]): reference tokens of a teacher-forced run (drafter study) instead of the target argmax
__global__ void accept_kernel(const int* amax, const int* drafts, int T, int S, const int* row0, const int* slot,
                              const int* pos, int* anchor, int* P, int* res, int* crow, int* cpos, int* csess,
                              const int* forced, int K) {
  const int s = threadIdx.x, RS = K + 2;
  if (s < S) {
    const int r0 = row0[s], q = slot[s];
    const int k = (s + 1 < S ? row0[s + 1] : T) - r0 - 1;
    auto ref = [&](int j) { return forced ? forced[q * (K + 1) + j] : amax[r0 + j]; };
    int a = 0;
    while (a < k && drafts[q * K + a] >= 0 && ref(a) == drafts[q * K + a]) ++a;
    res[s * RS] = a;
    for (int j = 0; j < a; ++j) res[s * RS + 1 + j] = drafts[q * K + j];
    res[s * RS + 1 + a] = ref(a);
    anchor[q] = ref(a);
    P[q] = pos[r0] + a + 1;
  }
  __syncthreads();
  if (s == 0) {
    int n = 0;
    for (int x = 0; x < S; ++x)
      for (int j = 0; j <= res[x * RS]; ++j) { crow[n] = row0[x] + j; cpos[n] = pos[row0[x]] + j; csess[n] = slot[x]; ++n; }
    res[S * RS] = n;
  }
}
// copy DSpark feature rows: dst[i] = feats[rows[i]] ([4][H] bf16 each); count from *n (or fixed m when n == nullptr)
__global__ void gather_feats_kernel(const bf16* feats, const int* rows, const int* n, bf16* dst) {
  const int i = blockIdx.x;
  if (n && i >= *n) return;
  const bf16* src = feats + (size_t)rows[i] * 4 * H;
  for (int c = threadIdx.x; c < 4 * H; c += blockDim.x) dst[(size_t)i * 4 * H + c] = src[c];
}
// lastfeat[slot[s]] = feats[row0[s] + a_s], a_s = res[s * RS] (RS = accept_kernel's result stride K + 2)
__global__ void last_feat_kernel(const bf16* feats, const int* row0, const int* slot, const int* res, bf16* lastfeat,
                                 int RS) {
  const int s = blockIdx.x;
  const bf16* src = feats + (size_t)(row0[s] + res[s * RS]) * 4 * H;
  for (int c = threadIdx.x; c < 4 * H; c += blockDim.x) lastfeat[(size_t)slot[s] * 4 * H + c] = src[c];
}

// confidence-gated verify width: 1 + the leading drafts whose margin >= tau, rounded up to 1, 2, 4 or 8 rows
__global__ void verify_width_kernel(const float* margin, float tau, int n, int* width) {
  const int q = threadIdx.x;
  if (q >= n) return;
  int lead = 0;
  while (lead < 7 && margin[q * 7 + lead] >= tau) ++lead;
  width[q] = lead == 0 ? 1 : lead == 1 ? 2 : lead <= 3 ? 4 : 8;
}

// vids[q][0..K] = anchor[q], drafts[q][0..K-1] (K drafts per slot; a missing draft < 0 becomes token 0)
__global__ void verify_ids_kernel(const int* anchor, const int* drafts, int* vids, int n, int K) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * (K + 1)) return;
  const int q = i / (K + 1), j = i % (K + 1);
  vids[i] = j == 0 ? anchor[q] : max(0, drafts[q * K + j - 1]);
}

// decode-row projections: rows of the fused output [q 3072 | gate 3072 | r 384 | sink 24 | k 1024 (| k_dyn 32) | v 1024
// (| v_dyn 32)] (row stride ld) into the separate buffers the glue and attention read (16-byte copies; bits unchanged)
__global__ void __launch_bounds__(256) proj_split_kernel(const bf16* f, int ld, int dyn, bf16* q, bf16* g, bf16* r,
                                                         bf16* s, bf16* k, bf16* kd, bf16* v, bf16* vd) {
  const int t = blockIdx.x;
  bf16* out[8] = {q, g, r, s, k, kd, v, vd};
  const int w[8] = {QD, QD, NH * DREL, NH, KD, dyn ? 32 : 0, KD, dyn ? 32 : 0};
  const uint4* src = reinterpret_cast<const uint4*>(f + (size_t)t * ld);
  int off = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint4* dst = reinterpret_cast<uint4*>(out[j] + (size_t)t * w[j]);
    for (int i = threadIdx.x; i < w[j] / 8; i += blockDim.x) dst[i] = src[off / 8 + i];
    off += w[j];
  }
}

}  // namespace
}  // namespace ink
