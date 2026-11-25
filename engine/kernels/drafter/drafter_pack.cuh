// inkling-cmp-engine kernel pack: DSpark drafter device kernels and launch structs (own code).
// Split from engine/native/src/drafter_rt.cuh at Rust runtime cut-over (Step 3) so the kernel pack compiles
// without engine/native/ or probes/p6-attn/.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "drafter_kernels.cuh"
#include "draft_conf.cuh"
#include "draft_sel.cuh"
#include "draft_leviathan.cuh"

using namespace s2f;
using bf16 = __nv_bfloat16;

namespace ink_dr_rng {
__host__ __device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
__host__ __device__ inline float u01(uint64_t h) { return ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f); }
}  // namespace ink_dr_rng

__global__ void k_fill_bf16(bf16* p, size_t n, uint64_t seed, float scale) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint64_t h = ink_dr_rng::splitmix64(seed ^ (i * 0x9E3779B97F4A7C15ull));
    float u1 = ink_dr_rng::u01(h), u2 = ink_dr_rng::u01(ink_dr_rng::splitmix64(h));
    float g = sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);  // normal
    p[i] = __float2bfloat16_rn(g * scale);
  }
}
__global__ void k_fill_e4m3(uint8_t* p, size_t n, uint64_t seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint64_t h = ink_dr_rng::splitmix64(seed ^ (i * 0x9E3779B97F4A7C15ull));
    float u1 = ink_dr_rng::u01(h), u2 = ink_dr_rng::u01(ink_dr_rng::splitmix64(h));
    p[i] = f2e4m3(sqrtf(-2.f * logf(u1)) * cospif(2.f * u2));
  }
}
__global__ void k_gather_cols(const bf16* head, const int* ochan, bf16* Wo) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)VP * NOUT) Wo[i] = head[(i / NOUT) * H + ochan[i % NOUT]];
}
__global__ void k_copy_cols(const bf16* src, int lds, int c0, int ncols, bf16* dst, int rows) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)rows * ncols) dst[i] = src[(i / ncols) * lds + c0 + i % ncols];
}
__global__ void k_cos_sin(float* cs, int maxpos) {  // vLLM _compute_cos_sin_cache, fp32
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)maxpos * 64) return;
  int pos = i / 64, d = i % 64;
  float inv = 1.0f / powf(1e6f, (float)(2 * d) / 128.0f);
  float f = (float)pos * inv;
  cs[(size_t)pos * 128 + d] = cosf(f);
  cs[(size_t)pos * 128 + 64 + d] = sinf(f);
}
// head inputs from captured h rows (eqhead mode): hout copy, f16 copy, x7 = [h, f_rnn], ||h||
__global__ void k_prep_head(const bf16* h, const bf16* f_rnn, int n, bf16* hout, __half* hf16, bf16* x7, float* hnorm) {
  int hr = blockIdx.x, r = hr / 7;
  float ss = 0.f;
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    bf16 v = h[(size_t)hr * H + k];
    float f = __bfloat162float(v);
    hout[(size_t)hr * H + k] = v;
    hf16[(size_t)hr * H + k] = __float2half_rn(f);
    x7[(size_t)hr * 2 * H + k] = v;
    x7[(size_t)hr * 2 * H + H + k] = f_rnn[(size_t)r * H + k];
    ss += f * f;
  }
  __shared__ float red[32];
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0;
    for (int i = 0; i < (int)blockDim.x / 32; ++i) t += red[i];
    hnorm[hr] = sqrtf(t) * (1.0f + 1e-5f) + 1e-6f;
  }
}

// Captured config B head inputs (cap.bin) for time/kbench: NREC sets of n sessions (records concatenated row-wise).
struct RealHead {
  int nrec = 0;
  bf16 *h = nullptr, *f = nullptr;
  int* a = nullptr;
  int* counter = nullptr;
};
// overwrite the head inputs with captured record (counter % nrec): hout, hf16, x7, hnorm, f_rnn, anchor
__global__ void k_real_head(RealHead rh, int n, bf16* hout, __half* hf16, bf16* x7, float* hnorm, bf16* f_rnn,
                            int* anchor) {
  int rec = *rh.counter % rh.nrec;
  int hr = blockIdx.x, r = hr / 7;
  const bf16* h = rh.h + ((size_t)rec * n * 7 + hr) * H;
  const bf16* fr = rh.f + ((size_t)rec * n + r) * H;
  float ss = 0.f;
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    bf16 v = h[k];
    float fv = __bfloat162float(v);
    hout[(size_t)hr * H + k] = v;
    hf16[(size_t)hr * H + k] = __float2half_rn(fv);
    x7[(size_t)hr * 2 * H + k] = v;
    x7[(size_t)hr * 2 * H + H + k] = fr[k];
    if (hr % 7 == 0) f_rnn[(size_t)r * H + k] = fr[k];
    ss += fv * fv;
  }
  __shared__ float red[32];
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0;
    for (int i = 0; i < (int)blockDim.x / 32; ++i) t += red[i];
    hnorm[hr] = sqrtf(t) * (1.0f + 1e-5f) + 1e-6f;
    if (hr % 7 == 0) anchor[r] = rh.a[rec * n + r];
  }
}
__global__ void k_bump(int* c) { atomicAdd(c, 1); }
