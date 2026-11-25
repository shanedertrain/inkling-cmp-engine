// inkling-cmp-engine native runtime: the drafter's probability of each draft, the signal of the confidence-gated verify
// width (own code; study and choice: bench/raw/2026-11-23-ptok-gate/). The drafts are unchanged.
// conf[r][i] = sum_{j <= i} log p_j, p_j = config B's p_tok (softmax of the Markov logits z over the unpadded vocabulary)
// of draft j, so "leading drafts with conf >= log tau" is the cumulative-product rule of P10. z(draft) is exact (the
// argmax key); every other row v uses z_v = softcap(bf16(Bq_v + bf16(fold_b_v + U_v . (V latent)))): Bq the int8 head's
// base logit, the fold product through config B's rank-16 basis (lowrank16.bin, 32 B per row; the full bf16 fold row
// was measured: AUC 0.85-0.92 vs 0.84-0.89 per position, +0.8-1.4 ms per round against +0.1-0.45 ms, no better gate).
// Sum of exp(z - 30) without a running max: |z| < 30 (softcap). Per step: conf_save_kernel keeps the rank-16
// projection and the draft; after the loop one pass over the vocabulary (conf_sum_kernel, deterministic partials) and
// conf_finish_kernel.
#pragma once

namespace s2f {

struct ConfBuf {
  float* pr;     // [7][n][LRK] rank-16 projection V latent (markov_zref's proj)
  float* zd;     // [7][n]      exact z of the draft
  int* vd;       // [7][n]      draft row
  float* part;   // [7][CONF_NCH][n] partial sums of exp(z - 30) over the rows other than the draft
};
constexpr int CONF_ROWS = 1024;
constexpr int CONF_NCH = (VU + CONF_ROWS - 1) / CONF_ROWS;   // 196

// grid n, 256 threads; after markov_argmax of step i (proj and key of step i are final)
__global__ void conf_save_kernel(Markov mk, int i, ConfBuf cb) {
  const int r = blockIdx.x, u = i * mk.n + r;
  if (threadIdx.x < LRK) cb.pr[u * LRK + threadIdx.x] = mk.proj[r * (LRK + 4) + threadIdx.x];
  if (threadIdx.x == 0) {
    const unsigned long long k = mk.keys[u];
    cb.zd[u] = key_z(k);
    cb.vd[u] = key_index(k);
  }
}

__device__ __forceinline__ float tanh_fast(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// grid (CONF_NCH, ksteps), 256 threads; thread per row, all N sessions of step blockIdx.y (int8 head: Bq)
template <int N>
__global__ void __launch_bounds__(256) conf_sum_kernel(Markov mk, ConfBuf cb) {
  __shared__ float pr[N * LRK];
  __shared__ int dv[N];
  __shared__ float red[8][N];
  const int i = blockIdx.y;
  for (int e = threadIdx.x; e < N * LRK; e += blockDim.x) pr[e] = cb.pr[i * N * LRK + e];
  if (threadIdx.x < N) dv[threadIdx.x] = cb.vd[i * N + threadIdx.x];
  __syncthreads();
  float acc[N];
#pragma unroll
  for (int r = 0; r < N; ++r) acc[r] = 0.f;
  const int v1 = min(VU, (blockIdx.x + 1) * CONF_ROWS);
  for (int v = blockIdx.x * CONF_ROWS + threadIdx.x; v < v1; v += blockDim.x) {
    float dot[N];
#pragma unroll
    for (int r = 0; r < N; ++r) dot[r] = 0.f;
    const uint4* f = (const uint4*)(mk.lrU + (size_t)v * LRK);
#pragma unroll
    for (int c = 0; c < LRK / 8; ++c) {
      const uint4 q = f[c];
      const __nv_bfloat162* q2 = (const __nv_bfloat162*)&q;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 w = __bfloat1622float2(q2[e]);
#pragma unroll
        for (int r = 0; r < N; ++r) dot[r] += pr[r * LRK + c * 8 + 2 * e] * w.x + pr[r * LRK + c * 8 + 2 * e + 1] * w.y;
      }
    }
    const float fb = bf2f(mk.fold_b[v]);
#pragma unroll
    for (int r = 0; r < N; ++r) {
      if (v == dv[r]) continue;
      const float s = rb(mk.Bq[(size_t)(r * 7 + i) * VP + v] + rb(fb + dot[r]));
      acc[r] += __expf(SOFTCAP * tanh_fast(s * (1.0f / SOFTCAP)) - SOFTCAP);
    }
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
  for (int r = 0; r < N; ++r) {
    const float a = warp_sum(acc[r]);
    if (lane == 0) red[warp][r] = a;
  }
  __syncthreads();
  if (threadIdx.x < N) {
    float a = 0.f;
    for (int w = 0; w < 8; ++w) a += red[w][threadIdx.x];
    cb.part[((size_t)i * CONF_NCH + blockIdx.x) * N + threadIdx.x] = a;
  }
}

// 1 block, n threads: conf[r][i] = sum_{j <= i} (z_d - log(sum_{v != d} exp z_v + exp z_d)); -inf past ksteps
__global__ void conf_finish_kernel(ConfBuf cb, int n, int ksteps, float* conf) {
  const int r = threadIdx.x;
  if (r >= n) return;
  float cum = 0.f;
  for (int i = 0; i < 7; ++i) {
    if (i >= ksteps) { conf[r * 7 + i] = -INFINITY; continue; }
    float s = 0.f;
    for (int c = 0; c < CONF_NCH; ++c) s += cb.part[((size_t)i * CONF_NCH + c) * n + r];
    const float z = cb.zd[i * n + r];
    s += expf(z - SOFTCAP);
    cum += (z - SOFTCAP) - logf(s);
    conf[r * 7 + i] = cum;
  }
}

static void conf_sum_launch(const Markov& mk, const ConfBuf& cb, int ksteps, cudaStream_t s) {
  const dim3 g(CONF_NCH, ksteps);
  switch (mk.n) {
    case 1: conf_sum_kernel<1><<<g, 256, 0, s>>>(mk, cb); break;
    case 2: conf_sum_kernel<2><<<g, 256, 0, s>>>(mk, cb); break;
    case 3: conf_sum_kernel<3><<<g, 256, 0, s>>>(mk, cb); break;
    case 4: conf_sum_kernel<4><<<g, 256, 0, s>>>(mk, cb); break;
    case 5: conf_sum_kernel<5><<<g, 256, 0, s>>>(mk, cb); break;
    case 6: conf_sum_kernel<6><<<g, 256, 0, s>>>(mk, cb); break;
    case 7: conf_sum_kernel<7><<<g, 256, 0, s>>>(mk, cb); break;
    default: conf_sum_kernel<8><<<g, 256, 0, s>>>(mk, cb); break;
  }
}

}  // namespace s2f
