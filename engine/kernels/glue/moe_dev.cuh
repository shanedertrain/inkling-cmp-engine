// inkling-cmp-engine native runtime: device-side routing layout for prefill_moe (own code).
//
// moe_build_kernel (one CTA of 1024 threads) turns the router output (idx, wts [T * 8]) into exactly what the host
// path built (engine/kernels/prefill_moe/bench.cu build_chunk + make_items, v2 kernel with 8 warps): per expert the
// flat rows in ascending order (perm), a_row (w13: token, w2: sorted position), out_row, gamma, and the item list
// (experts by row count descending, ties to the lower expert; per expert FT = 9 feature tiles x 128-row tiles). w13
// and w2 have the same NWT (36) and so the same list. n_items goes to device memory (Params::n_items_dev): moe_gemm2's
// CTAs at or past it exit, so the launch grid can be a fixed upper bound (CUDA-graph friendly) and no host round trip
// is needed.
#pragma once
#include "../prefill_moe/prefill_moe.cuh"

namespace moedev {

constexpr int FT = 9;       // (2 * 1152 / 64) / 4 = 2304 / 64 / 4 (WARPS=8)
constexpr int FT_DEC = 18;  // (2 * 1152 / 64) / 2 = 2304 / 64 / 2 (WARPS=4 decode)

inline int items_bound(int R) { return FT * (R / pmoe::BN + 1 + std::min(pmoe::E, R)); }
inline int dec_items_bound(int R) {
  const int k = std::min(pmoe::E, R);
  return FT_DEC * (k + (R - k) / pmoe::BN_DEC);
}

// exclusive prefix sum of in[0..128) into out[0..129) by warp 0 (lane l: elements 4l .. 4l + 3); integers, so the
// result equals the serial loop's
__device__ __forceinline__ void scan128(const int* in, int* out) {
  static_assert(pmoe::E == 128, "scan128 assumes 128 experts");
  const int lane = threadIdx.x & 31;
  if (threadIdx.x >= 32) return;
  int a[4], s = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) { a[j] = in[4 * lane + j]; s += a[j]; }
  int inc = s;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += y;
  }
  int base = inc - s;
#pragma unroll
  for (int j = 0; j < 4; ++j) { out[4 * lane + j] = base; base += a[j]; }
  if (lane == 31) out[128] = inc;
}

// Fast single-CTA decode routing build (R <= 512, block = 256 = 8 warps):
// reads idx/wts ONCE into registers, uses hardware __match_any_sync per 32-row chunk for exact ascending
// per-expert ranks, overlaps scan128 (warp 0) with descending expert rank-sort (warps 4..7), and scatters
// all R entries in parallel. Bit-identical outputs to the original 1024-thread kernel.
__global__ void __launch_bounds__(256) moe_build_kernel(const int* __restrict__ idx, const float* __restrict__ wts,
                                                        int R, int* __restrict__ a13, int* __restrict__ a2,
                                                        int* __restrict__ orow, float* __restrict__ gamma,
                                                        pmoe::Item* __restrict__ items, int* __restrict__ n_items) {
  static_assert(pmoe::E == 128 && pmoe::TOPK == 8 && pmoe::BN == 128, "moe_build assumptions");
  __shared__ uint16_t wcnt[16][pmoe::E];
  __shared__ int cnt[pmoe::E], eoff[pmoe::E + 1], order[pmoe::E], ioff[pmoe::E + 1], tiles[pmoe::E];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nw = (R + 31) >> 5;  // 1..16 (R <= 512)

  // Zero only the nw active rows of wcnt (32-bit stores: 64 uint32 per chunk)
  uint32_t* wcnt32 = reinterpret_cast<uint32_t*>(wcnt);
  for (int i = tid; i < nw * (pmoe::E / 2); i += 256) wcnt32[i] = 0u;
  __syncthreads();

  int e_reg[2] = {-1, -1};
  float w_reg[2] = {0.f, 0.f};
  int rnk_reg[2] = {0, 0};
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int w = warp + pass * 8;
    if (w < nw) {
      const int r = (w << 5) + lane;
      const bool ok = r < R;
      const int e = ok ? idx[r] : -1;
      if (ok) {
        e_reg[pass] = e;
        w_reg[pass] = wts[r];
      }
      const unsigned active = __ballot_sync(0xffffffffu, ok);
      if (ok) {
        const unsigned m = __match_any_sync(active, e);
        const unsigned lower = m & ((1u << lane) - 1u);
        rnk_reg[pass] = __popc(lower);
        if (lower == 0u) wcnt[w][e] = static_cast<uint16_t>(__popc(m));
      }
    }
  }
  __syncthreads();

  if (tid < pmoe::E) {
    int s = 0;
    for (int w = 0; w < nw; ++w) {
      const int c = wcnt[w][tid];
      wcnt[w][tid] = static_cast<uint16_t>(s);
      s += c;
    }
    cnt[tid] = s;
  }
  __syncthreads();

  // Overlap eoff prefix sum (warp 0) with stable descending expert rank-sort (warps 4..7)
  if (tid < 32) {
    scan128(cnt, eoff);
  } else if (tid >= 128) {
    const int e = tid - 128;
    const int ce = cnt[e];
    int rk = 0;
#pragma unroll 8
    for (int f = 0; f < pmoe::E; ++f) {
      const int cf = cnt[f];
      rk += (cf > ce) || (cf == ce && f < e);
    }
    order[rk] = e;
  }
  __syncthreads();

  // Parallel scatter of all R rows in exact ascending order per expert
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int e = e_reg[pass];
    if (e >= 0) {
      const int w = warp + pass * 8;
      const int r = (w << 5) + lane;
      const int p = eoff[e] + static_cast<int>(wcnt[w][e]) + rnk_reg[pass];
      a13[p] = r >> 3;
      a2[p] = p;
      orow[p] = r;
      gamma[p] = w_reg[pass];
    }
  }
  if (tid < pmoe::E) tiles[tid] = FT * ((cnt[order[tid]] + pmoe::BN - 1) >> 7);
  __syncthreads();

  scan128(tiles, ioff);
  __syncthreads();

  if (tid == 0) *n_items = ioff[pmoe::E];
  if (tid < pmoe::E) {
    const int e = order[tid], c = cnt[e], nt = (c + pmoe::BN - 1) >> 7;
    int o = ioff[tid];
    for (int ft = 0; ft < FT; ++ft)
      for (int t = 0; t < nt; ++t) items[o++] = {e, ft, eoff[e] + t * pmoe::BN, min(pmoe::BN, c - t * pmoe::BN)};
  }
}

// Decode routing build (R <= 512, i.e., T <= 64): 512 threads (16 warps), parallel 4-thread-per-expert rank sort,
// caches idx/wts in shared memory, loops only over active experts in descending count order, and emits FT_DEC=18
// feature tiles x BN_DEC=16 row tiles.
__global__ void __launch_bounds__(512) moe_build_dec_kernel(const int* idx, const float* wts, int R, int* a13, int* a2,
                                                            int* orow, float* gamma, pmoe::Item* items, int* n_items) {
  __shared__ int s_idx[512];
  __shared__ float s_wts[512];
  __shared__ int cnt[pmoe::E], eoff[pmoe::E + 1], order[pmoe::E], ioff[pmoe::E + 1], tiles[pmoe::E];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (R > 512) __trap();
  if (tid < pmoe::E) cnt[tid] = 0;
  if (tid < R) {
    s_idx[tid] = idx[tid];
    s_wts[tid] = wts[tid];
  }
  __syncthreads();
  if (tid < R) atomicAdd(&cnt[s_idx[tid]], 1);
  __syncthreads();
  scan128(cnt, eoff);
  // Parallel rank sort: 4 threads per expert (tid = 4*e + q) -> 32 iterations each instead of 128!
  {
    const int e = tid >> 2, q = tid & 3;
    const int ce = cnt[e];
    int rk = 0;
    const int f0 = q * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int f = f0 + j;
      rk += cnt[f] > ce || (cnt[f] == ce && f < e);
    }
    rk += __shfl_xor_sync(0xffffffffu, rk, 1);
    rk += __shfl_xor_sync(0xffffffffu, rk, 2);
    if (q == 0) order[rk] = e;
  }
  __syncthreads();
  // Active experts only: order[] is sorted descending by count, so stop as soon as cnt == 0!
  for (int idx_o = warp; idx_o < pmoe::E; idx_o += 16) {
    const int e = order[idx_o];
    if (cnt[e] == 0) break;
    int base = eoff[e];
    for (int r0 = 0; r0 < R; r0 += 32) {
      const int r = r0 + lane;
      const bool hit = r < R && s_idx[r] == e;
      const unsigned m = __ballot_sync(0xffffffffu, hit);
      if (hit) {
        const int p = base + __popc(m & ((1u << lane) - 1u));
        a13[p] = r / pmoe::TOPK;
        a2[p] = p;
        orow[p] = r;
        gamma[p] = s_wts[r];
      }
      base += __popc(m);
    }
  }
  if (tid < pmoe::E) tiles[tid] = FT_DEC * ((cnt[order[tid]] + pmoe::BN_DEC - 1) / pmoe::BN_DEC);
  __syncthreads();
  scan128(tiles, ioff);
  __syncthreads();
  if (tid == 0) *n_items = ioff[pmoe::E];
  if (tid < pmoe::E) {
    const int e = order[tid], c = cnt[e], nt = (c + pmoe::BN_DEC - 1) / pmoe::BN_DEC;
    int o = ioff[tid];
    for (int ft = 0; ft < FT_DEC; ++ft)
      for (int t = 0; t < nt; ++t) items[o++] = {e, ft, eoff[e] + t * pmoe::BN_DEC, min(pmoe::BN_DEC, c - t * pmoe::BN_DEC)};
  }
}

}  // namespace moedev
