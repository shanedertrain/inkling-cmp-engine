// inkling-cmp-engine S2f: batched sm_80 DSpark drafter kernels (Inkling-2-Mini). Our own code, written 2026-11-23.
// Numerics follow config B's served drafter (vLLM models/inkling2/nvidia/dspark.py, nvidia/ops/dspark.py,
// models/inkling2/speculator.py @ 6e9d1cce, Apache-2.0; see docs/engine/configB-numerics.md) at the rounding points:
// every config B op that outputs bf16 is rounded to bf16 here at the same place; fp32 summation order differs.
//
// Kernels:
//   gemm_frag      skinny GEMM y[M<=64, N] = x[M, K] . W[N, K]^T, W pre-permuted into mma.m16n8k16 A-fragment order
//                  (bf16, 16 B per lane per 16x16 tile) or int8 (8 B per lane, per-row scale, f16 mma); x staged in
//                  shared memory per 256-wide K chunk; split-K over gridDim.y; epilogue: fp32 partials / bf16 / head.
//   epi_*          split-K reduction fused with the next config B op (qk-norm+rope+cache write, fused_add_rmsnorm,
//                  silu_and_mul, k_norm_ctx+rope+cache write, rnn context bias).
//   attn_block     DSpark block attention (8 rows x GQA 3 = 24 query rows per KV head, bidirectional, local window
//                  1,024 + 8 or all keys), paged bf16 K/V, split over keys, mma.sync, bf16 P, fp32 softmax (log2);
//                  attn_combine merges splits and rounds to bf16.
//   markov_*       recurrent Markov head over the full padded vocabulary, EXACT argmax with candidate pruning:
//                  per vocab row v an upper bound of the softcapped logit (Cauchy-Schwarz on the rank-256 fold term,
//                  plus a rigorous int8 head error bound in the int8 variant); only rows whose bound reaches the
//                  current best are evaluated exactly. Drafts equal a full-vocabulary evaluation (proof in README).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>
#include <cooperative_groups.h>

namespace s2f {

constexpr int H = 2304, NH = 24, KVH = 8, HD = 128, INTER = 9216, RANK = 256, BLK = 8, WIN = 1024;
constexpr int VP = 201024, VU = 200058;
constexpr float EPS = 1e-6f, SOFTCAP = 30.0f;
constexpr int PAGE = 64;

// ------------------------------------------------------------------------------------------------ small helpers
__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rb(float x) {  // round to bf16 (RNE) and back, as config B's _round_bf16
  uint32_t b = __float_as_uint(x);
  b = (b + 0x7FFFu + ((b >> 16) & 1u)) & 0xFFFF0000u;
  return __uint_as_float(b);
}
__device__ __forceinline__ float bf_round_up(float x) {  // smallest bf16 >= x (finite inputs)
  float r = rb(x);
  if (r < x) {
    uint32_t b = __float_as_uint(r);
    b = (r >= 0.f) ? b + 0x10000u : b - 0x10000u;
    if (r == 0.f) b = 0x00010000u;
    r = __uint_as_float(b);
  }
  return r;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// config B add_softcap: x = bf16(a + b); x = x / softcap (Triton '/', div.full.f32); tanh (libdevice); * softcap; bf16
__device__ __forceinline__ float softcap_z(float s_rounded) {
  float x = div_full(s_rounded, SOFTCAP);
  x = tanhf(x);
  return rb(x * SOFTCAP);
}
__device__ __forceinline__ uint32_t ord16(float zb) {  // order-preserving 16-bit code of a bf16 value
  uint32_t u = __float_as_uint(zb) >> 16;
  return (u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u);
}
__device__ __forceinline__ unsigned long long argkey(float zb, int v) {
  return ((unsigned long long)ord16(zb) << 32) | (unsigned long long)(0xFFFFFFFFu - (uint32_t)v);
}
__device__ __forceinline__ int key_index(unsigned long long k) { return (int)(0xFFFFFFFFu - (uint32_t)(k & 0xFFFFFFFFull)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void cp_async16(void* s, const void* g, bool pred) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
  int n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(g), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ldg_stream2(const uint2* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
// 4 int8 (bytes b0..b3 of w, stored as [e0, e2, e1, e3]) -> two f16x2 regs (e0,e1), (e2,e3), exact
__device__ __forceinline__ void i8x4_to_f16x2(uint32_t w, uint32_t& h01, uint32_t& h23) {
  uint32_t u = w ^ 0x80808080u;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(h01) : "r"(u), "r"(0x64646464u), "r"(0x5250u));
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(h23) : "r"(u), "r"(0x64646464u), "r"(0x5351u));
  const uint32_t magic = 0x64806480u;  // 1152 = 1024 + 128
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h01) : "r"(h01), "r"(magic));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h23) : "r"(h23), "r"(magic));
}

// ------------------------------------------------------------------------------------------------ skinny GEMM
// W fragment layout: tile (nt, kt) of 16x16 at ((nt * (K/16)) + kt) * 32 lanes * 8 elements; lane l = 4g + t holds
// W[g][2t], W[g][2t+1], W[g+8][2t], W[g+8][2t+1], W[g][2t+8], W[g][2t+9], W[g+8][2t+8], W[g+8][2t+9] (bf16).
// int8: the same 8 elements per lane as bytes [e0,e2,e1,e3, e4,e6,e5,e7] (i8x4_to_f16x2 order), per-row fp32 scale.
enum Epi { EPI_F32 = 0, EPI_BF16 = 1, EPI_HEADQ = 2 };
constexpr int KC = 256;            // K chunk staged in shared memory
constexpr int XS = KC + 8;         // smem row stride (halves) -> conflict-free ldmatrix
constexpr int GEMM_WARPS = 8;

struct GemmArgs {
  const void* W;          // fragment layout
  const __nv_bfloat16* x; // [M, ldx] (bf16; f16 for int8)
  int ldx;
  int M, N, K, kslice;    // kslice = K / gridDim.y (multiple of KC or the whole K)
  int ldo;                // row stride of out / Bq (EPI_BF16, EPI_HEADQ); EPI_F32 partials use N
  float* part;            // EPI_F32: [gridDim.y][M][N]
  __nv_bfloat16* out;     // EPI_BF16: [M][N]
  // EPI_HEADQ (int8 head): part = Bq fp32 [M][N] = acc * scale[n]
  const float* scale;
};

template <int MT, bool INT8, int EPI, int NW = GEMM_WARPS>
__global__ void __launch_bounds__(NW * 32) gemm_frag(GemmArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* xs = (__nv_bfloat16*)smem_raw;  // [2][MT*8][XS]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nt = blockIdx.x * NW + warp;
  const int ntiles = a.N / 16;
  const int k0 = blockIdx.y * a.kslice;
  const int KT = a.K / 16;
  const int nchunks = (a.kslice + KC - 1) / KC;
  const int MR = MT * 8;
  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  auto load_x = [&](int c, int buf) {
    const int kk = k0 + c * KC;
    const int klen = min(KC, a.kslice - c * KC);
    const int pieces = MR * (KC / 8);
    for (int p = threadIdx.x; p < pieces; p += NW * 32) {
      int m = p / (KC / 8), q = (p % (KC / 8)) * 8;
      bool ok = (m < a.M) && (q < klen);
      const __nv_bfloat16* g = a.x + (size_t)(ok ? m : 0) * a.ldx + kk + (ok ? q : 0);
      cp_async16(xs + ((size_t)buf * MR + m) * XS + q, g, ok);
    }
    cp_commit();
  };
  load_x(0, 0);
  for (int c = 0; c < nchunks; ++c) {
    if (c + 1 < nchunks) { load_x(c + 1, (c + 1) & 1); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    const int klen = min(KC, a.kslice - c * KC);
    const int ksteps = klen / 16;
    const __nv_bfloat16* xb = xs + (size_t)(c & 1) * MR * XS;
    if (nt < ntiles) {
      const int kt0 = (k0 + c * KC) / 16;
      if constexpr (!INT8) {
        // MT >= 5: two half-chunks of 8 loads (32 fewer registers -> 2 CTAs per SM at M = 64)
        constexpr int PF = MT >= 5 ? 8 : 16;
        const uint4* wp = (const uint4*)a.W + ((size_t)nt * KT + kt0) * 32 + lane;
#pragma unroll
        for (int hh = 0; hh < 16 / PF; ++hh) {
          uint4 wf[PF];
#pragma unroll
          for (int s = 0; s < PF; ++s)
            if (hh * PF + s < ksteps) wf[s] = ldg_stream(wp + (size_t)(hh * PF + s) * 32);
#pragma unroll
          for (int s = 0; s < PF; ++s) {
            if (hh * PF + s < ksteps) {
              uint32_t af[4] = {wf[s].x, wf[s].y, wf[s].z, wf[s].w};
#pragma unroll
              for (int j = 0; j < MT; ++j) {
                uint32_t b0, b1;
                ldsm_x2(b0, b1, xb + (size_t)(j * 8 + (lane & 7)) * XS + (hh * PF + s) * 16 + ((lane >> 3) & 1) * 8);
                mma_bf16(acc[j], af, b0, b1);
              }
            }
          }
        }
      } else {
        const uint2* wp = (const uint2*)a.W + ((size_t)nt * KT + kt0) * 32 + lane;
        uint2 wf[16];
#pragma unroll
        for (int s = 0; s < 16; ++s)
          if (s < ksteps) wf[s] = ldg_stream2(wp + (size_t)s * 32);
#pragma unroll
        for (int s = 0; s < 16; ++s) {
          if (s < ksteps) {
            uint32_t af[4];
            i8x4_to_f16x2(wf[s].x, af[0], af[1]);
            i8x4_to_f16x2(wf[s].y, af[2], af[3]);
#pragma unroll
            for (int j = 0; j < MT; ++j) {
              uint32_t b0, b1;
              ldsm_x2(b0, b1, xb + (size_t)(j * 8 + (lane & 7)) * XS + s * 16 + ((lane >> 3) & 1) * 8);
              mma_f16(acc[j], af, b0, b1);
            }
          }
        }
      }
    }
    __syncthreads();
  }
  if (nt >= ntiles) return;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = nt * 16 + g, n1 = n0 + 8;
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int m0 = j * 8 + 2 * t, m1 = m0 + 1;
    float v[4] = {acc[j][0], acc[j][1], acc[j][2], acc[j][3]};  // (n0,m0) (n0,m1) (n1,m0) (n1,m1)
    if constexpr (EPI == EPI_F32) {
      float* p = a.part + (size_t)blockIdx.y * a.M * a.N;
      if (m0 < a.M) { p[(size_t)m0 * a.N + n0] = v[0]; p[(size_t)m0 * a.N + n1] = v[2]; }
      if (m1 < a.M) { p[(size_t)m1 * a.N + n0] = v[1]; p[(size_t)m1 * a.N + n1] = v[3]; }
    } else if constexpr (EPI == EPI_BF16) {
      if (m0 < a.M) { a.out[(size_t)m0 * a.ldo + n0] = __float2bfloat16_rn(v[0]); a.out[(size_t)m0 * a.ldo + n1] = __float2bfloat16_rn(v[2]); }
      if (m1 < a.M) { a.out[(size_t)m1 * a.ldo + n0] = __float2bfloat16_rn(v[1]); a.out[(size_t)m1 * a.ldo + n1] = __float2bfloat16_rn(v[3]); }
    } else {
      const float s0 = a.scale[n0], s1 = a.scale[n1];
      if (m0 < a.M) { a.part[(size_t)m0 * a.ldo + n0] = v[0] * s0; a.part[(size_t)m0 * a.ldo + n1] = v[2] * s1; }
      if (m1 < a.M) { a.part[(size_t)m1 * a.ldo + n0] = v[1] * s0; a.part[(size_t)m1 * a.ldo + n1] = v[3] * s1; }
    }
  }
}

// Same contract as gemm_frag, but each warp owns TWO 16-row tiles and reuses every x fragment (ldmatrix) for
// both: half the shared-memory traffic per weight byte (vocabulary GEMM, M <= 64). W loads in half-chunks of 8
// k-steps (bf16: 16 x 16 B in flight per lane; int8: 16 x 8 B).
template <int MT, bool INT8, int EPI, int NW>
__global__ void __launch_bounds__(NW * 32) gemm_frag2(GemmArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* xs = (__nv_bfloat16*)smem_raw;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nt0 = (blockIdx.x * NW + warp) * 2;
  const int ntiles = a.N / 16;
  const int k0 = blockIdx.y * a.kslice;
  const int KT = a.K / 16;
  const int nchunks = (a.kslice + KC - 1) / KC;
  const int MR = MT * 8;
  float acc[2][MT][4];
#pragma unroll
  for (int q = 0; q < 2; ++q)
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[q][j][0] = acc[q][j][1] = acc[q][j][2] = acc[q][j][3] = 0.f;
  auto load_x = [&](int c, int buf) {
    const int kk = k0 + c * KC;
    const int klen = min(KC, a.kslice - c * KC);
    const int pieces = MR * (KC / 8);
    for (int p = threadIdx.x; p < pieces; p += NW * 32) {
      int m = p / (KC / 8), q = (p % (KC / 8)) * 8;
      bool ok = (m < a.M) && (q < klen);
      const __nv_bfloat16* g = a.x + (size_t)(ok ? m : 0) * a.ldx + kk + (ok ? q : 0);
      cp_async16(xs + ((size_t)buf * MR + m) * XS + q, g, ok);
    }
    cp_commit();
  };
  const bool have1 = nt0 + 1 < ntiles;
  load_x(0, 0);
  for (int c = 0; c < nchunks; ++c) {
    if (c + 1 < nchunks) { load_x(c + 1, (c + 1) & 1); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    const int klen = min(KC, a.kslice - c * KC);
    const int ksteps = klen / 16;
    const __nv_bfloat16* xb = xs + (size_t)(c & 1) * MR * XS;
    if (nt0 < ntiles) {
      const int kt0 = (k0 + c * KC) / 16;
#pragma unroll
      for (int hlf = 0; hlf < 2; ++hlf) {
        if constexpr (!INT8) {
          const uint4* wp0 = (const uint4*)a.W + ((size_t)nt0 * KT + kt0 + hlf * 8) * 32 + lane;
          const uint4* wp1 = wp0 + (size_t)KT * 32;
          uint4 w0[8], w1[8];
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            if (hlf * 8 + s < ksteps) {
              w0[s] = ldg_stream(wp0 + (size_t)s * 32);
              if (have1) w1[s] = ldg_stream(wp1 + (size_t)s * 32);
            }
          }
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            if (hlf * 8 + s < ksteps) {
              uint32_t a0[4] = {w0[s].x, w0[s].y, w0[s].z, w0[s].w};
              uint32_t a1[4] = {w1[s].x, w1[s].y, w1[s].z, w1[s].w};
#pragma unroll
              for (int j = 0; j < MT; ++j) {
                uint32_t b0, b1;
                ldsm_x2(b0, b1, xb + (size_t)(j * 8 + (lane & 7)) * XS + (hlf * 8 + s) * 16 + ((lane >> 3) & 1) * 8);
                mma_bf16(acc[0][j], a0, b0, b1);
                if (have1) mma_bf16(acc[1][j], a1, b0, b1);
              }
            }
          }
        } else {
          const uint2* wp0 = (const uint2*)a.W + ((size_t)nt0 * KT + kt0 + hlf * 8) * 32 + lane;
          const uint2* wp1 = wp0 + (size_t)KT * 32;
          uint2 w0[8], w1[8];
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            if (hlf * 8 + s < ksteps) {
              w0[s] = ldg_stream2(wp0 + (size_t)s * 32);
              if (have1) w1[s] = ldg_stream2(wp1 + (size_t)s * 32);
            }
          }
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            if (hlf * 8 + s < ksteps) {
              uint32_t a0[4], a1[4];
              i8x4_to_f16x2(w0[s].x, a0[0], a0[1]);
              i8x4_to_f16x2(w0[s].y, a0[2], a0[3]);
              i8x4_to_f16x2(w1[s].x, a1[0], a1[1]);
              i8x4_to_f16x2(w1[s].y, a1[2], a1[3]);
#pragma unroll
              for (int j = 0; j < MT; ++j) {
                uint32_t b0, b1;
                ldsm_x2(b0, b1, xb + (size_t)(j * 8 + (lane & 7)) * XS + (hlf * 8 + s) * 16 + ((lane >> 3) & 1) * 8);
                mma_f16(acc[0][j], a0, b0, b1);
                if (have1) mma_f16(acc[1][j], a1, b0, b1);
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int nt = nt0 + q;
    if (nt >= ntiles) continue;
    const int n0 = nt * 16 + g, n1 = n0 + 8;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      const int m0 = j * 8 + 2 * t, m1 = m0 + 1;
      float v[4] = {acc[q][j][0], acc[q][j][1], acc[q][j][2], acc[q][j][3]};
      if constexpr (EPI == EPI_BF16) {
        if (m0 < a.M) { a.out[(size_t)m0 * a.ldo + n0] = __float2bfloat16_rn(v[0]); a.out[(size_t)m0 * a.ldo + n1] = __float2bfloat16_rn(v[2]); }
        if (m1 < a.M) { a.out[(size_t)m1 * a.ldo + n0] = __float2bfloat16_rn(v[1]); a.out[(size_t)m1 * a.ldo + n1] = __float2bfloat16_rn(v[3]); }
      } else if constexpr (EPI == EPI_HEADQ) {
        const float s0 = a.scale[n0], s1 = a.scale[n1];
        if (m0 < a.M) { a.part[(size_t)m0 * a.ldo + n0] = v[0] * s0; a.part[(size_t)m0 * a.ldo + n1] = v[2] * s1; }
        if (m1 < a.M) { a.part[(size_t)m1 * a.ldo + n0] = v[1] * s0; a.part[(size_t)m1 * a.ldo + n1] = v[3] * s1; }
      } else {
        float* p = a.part + (size_t)blockIdx.y * a.M * a.N;
        if (m0 < a.M) { p[(size_t)m0 * a.N + n0] = v[0]; p[(size_t)m0 * a.N + n1] = v[2]; }
        if (m1 < a.M) { p[(size_t)m1 * a.N + n0] = v[1]; p[(size_t)m1 * a.N + n1] = v[3]; }
      }
    }
  }
}

// gemm_frag3: gemm_frag with the W stream software-pipelined across k-groups of 8 steps: the loads of group g+1 are
// issued before the mma of group g (register double buffer), so every warp keeps a group in flight while computing.
template <int MT, bool INT8, int EPI, int NW = GEMM_WARPS>
__global__ void __launch_bounds__(NW * 32) gemm_frag3(GemmArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* xs = (__nv_bfloat16*)smem_raw;
  constexpr int G = 8;
  using WT = typename std::conditional<INT8, uint2, uint4>::type;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nt = blockIdx.x * NW + warp;
  const int ntiles = a.N / 16;
  const int k0 = blockIdx.y * a.kslice;
  const int KT = a.K / 16;
  const int S = a.kslice / 16;
  const int nchunks = (a.kslice + KC - 1) / KC;
  const int ngroups = (S + G - 1) / G;
  const int MR = MT * 8;
  const bool act = nt < ntiles;
  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  auto load_x = [&](int c, int buf) {
    const int kk = k0 + c * KC;
    const int klen = min(KC, a.kslice - c * KC);
    const int pieces = MR * (KC / 8);
    for (int p = threadIdx.x; p < pieces; p += NW * 32) {
      int m = p / (KC / 8), q = (p % (KC / 8)) * 8;
      bool ok = (m < a.M) && (q < klen);
      const __nv_bfloat16* g = a.x + (size_t)(ok ? m : 0) * a.ldx + kk + (ok ? q : 0);
      cp_async16(xs + ((size_t)buf * MR + m) * XS + q, g, ok);
    }
    cp_commit();
  };
  const WT* wp = (const WT*)a.W + ((size_t)(act ? nt : 0) * KT + k0 / 16) * 32 + lane;
  auto load_w = [&](int g, WT* dst) {
#pragma unroll
    for (int s = 0; s < G; ++s) {
      const int st = g * G + s;
      if (act && st < S) {
        if constexpr (INT8) dst[s] = ldg_stream2(wp + (size_t)st * 32);
        else dst[s] = ldg_stream(wp + (size_t)st * 32);
      }
    }
  };
  WT cur[G], nxt[G];
  load_x(0, 0);
  load_w(0, cur);
  for (int g = 0; g < ngroups; ++g) {
    const int s0 = g * G;
    const int chunk = s0 / 16;
    if (s0 % 16 == 0) {
      if (chunk + 1 < nchunks) { load_x(chunk + 1, (chunk + 1) & 1); cp_wait<1>(); } else { cp_wait<0>(); }
      __syncthreads();
    }
    if (g + 1 < ngroups) load_w(g + 1, nxt);
    const __nv_bfloat16* xb = xs + (size_t)(chunk & 1) * MR * XS;
    if (act) {
#pragma unroll
      for (int s = 0; s < G; ++s) {
        const int st = s0 + s;
        if (st < S) {
          uint32_t af[4];
          if constexpr (INT8) {
            i8x4_to_f16x2(cur[s].x, af[0], af[1]);
            i8x4_to_f16x2(cur[s].y, af[2], af[3]);
          } else {
            af[0] = cur[s].x; af[1] = cur[s].y; af[2] = cur[s].z; af[3] = cur[s].w;
          }
          const int col = (st % 16) * 16;
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            uint32_t b0, b1;
            ldsm_x2(b0, b1, xb + (size_t)(j * 8 + (lane & 7)) * XS + col + ((lane >> 3) & 1) * 8);
            if constexpr (INT8) mma_f16(acc[j], af, b0, b1); else mma_bf16(acc[j], af, b0, b1);
          }
        }
      }
    }
#pragma unroll
    for (int s = 0; s < G; ++s) cur[s] = nxt[s];
    if ((s0 + G) % 16 == 0 || g == ngroups - 1) __syncthreads();
  }
  if (!act) return;
  const int gq = lane >> 2, t = lane & 3;
  const int n0 = nt * 16 + gq, n1 = n0 + 8;
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int m0 = j * 8 + 2 * t, m1 = m0 + 1;
    float v[4] = {acc[j][0], acc[j][1], acc[j][2], acc[j][3]};
    if constexpr (EPI == EPI_F32) {
      float* p = a.part + (size_t)blockIdx.y * a.M * a.N;
      if (m0 < a.M) { p[(size_t)m0 * a.N + n0] = v[0]; p[(size_t)m0 * a.N + n1] = v[2]; }
      if (m1 < a.M) { p[(size_t)m1 * a.N + n0] = v[1]; p[(size_t)m1 * a.N + n1] = v[3]; }
    } else if constexpr (EPI == EPI_BF16) {
      if (m0 < a.M) { a.out[(size_t)m0 * a.ldo + n0] = __float2bfloat16_rn(v[0]); a.out[(size_t)m0 * a.ldo + n1] = __float2bfloat16_rn(v[2]); }
      if (m1 < a.M) { a.out[(size_t)m1 * a.ldo + n0] = __float2bfloat16_rn(v[1]); a.out[(size_t)m1 * a.ldo + n1] = __float2bfloat16_rn(v[3]); }
    } else {
      const float s0v = a.scale[n0], s1v = a.scale[n1];
      if (m0 < a.M) { a.part[(size_t)m0 * a.ldo + n0] = v[0] * s0v; a.part[(size_t)m0 * a.ldo + n1] = v[2] * s1v; }
      if (m1 < a.M) { a.part[(size_t)m1 * a.ldo + n0] = v[1] * s0v; a.part[(size_t)m1 * a.ldo + n1] = v[3] * s1v; }
    }
  }
}

// gemm_frag4: ping-pong register sets (A/B) of G k-steps: the loads of group g+1 are in flight while group g computes,
// no register copies (the compiler waits on a set only when it computes it). x staged per 256-wide chunk as before.
template <int MT, bool INT8, int EPI, int NW = GEMM_WARPS>
__global__ void __launch_bounds__(NW * 32) gemm_frag4(GemmArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* xs = (__nv_bfloat16*)smem_raw;
  constexpr int G = 8;
  using WT = typename std::conditional<INT8, uint2, uint4>::type;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nt = blockIdx.x * NW + warp;
  const int ntiles = a.N / 16;
  const int k0 = blockIdx.y * a.kslice;
  const int KT = a.K / 16;
  const int S = a.kslice / 16;
  const int nchunks = (a.kslice + KC - 1) / KC;
  const int ngroups = (S + G - 1) / G;
  const int MR = MT * 8;
  const bool act = nt < ntiles;
  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  auto load_x = [&](int c, int buf) {
    const int kk = k0 + c * KC;
    const int klen = min(KC, a.kslice - c * KC);
    const int pieces = MR * (KC / 8);
    for (int p = threadIdx.x; p < pieces; p += NW * 32) {
      int m = p / (KC / 8), q = (p % (KC / 8)) * 8;
      bool ok = (m < a.M) && (q < klen);
      const __nv_bfloat16* g = a.x + (size_t)(ok ? m : 0) * a.ldx + kk + (ok ? q : 0);
      cp_async16(xs + ((size_t)buf * MR + m) * XS + q, g, ok);
    }
    cp_commit();
  };
  const WT* wp = (const WT*)a.W + ((size_t)(act ? nt : 0) * KT + k0 / 16) * 32 + lane;
  auto load_w = [&](int g, WT* dst) {
#pragma unroll
    for (int s = 0; s < G; ++s) {
      const int st = g * G + s;
      if (act && st < S) {
        if constexpr (INT8) dst[s] = ldg_stream2(wp + (size_t)st * 32);
        else dst[s] = ldg_stream(wp + (size_t)st * 32);
      }
    }
  };
  auto compute = [&](int g, const WT* w) {
    const int s0 = g * G;
    const __nv_bfloat16* xb = xs + (size_t)((s0 / 16) & 1) * MR * XS;
#pragma unroll
    for (int s = 0; s < G; ++s) {
      const int st = s0 + s;
      if (act && st < S) {
        uint32_t af[4];
        if constexpr (INT8) {
          i8x4_to_f16x2(w[s].x, af[0], af[1]);
          i8x4_to_f16x2(w[s].y, af[2], af[3]);
        } else {
          af[0] = w[s].x; af[1] = w[s].y; af[2] = w[s].z; af[3] = w[s].w;
        }
        const int col = (st % 16) * 16;
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          uint32_t b0, b1;
          ldsm_x2(b0, b1, xb + (size_t)(j * 8 + (lane & 7)) * XS + col + ((lane >> 3) & 1) * 8);
          if constexpr (INT8) mma_f16(acc[j], af, b0, b1); else mma_bf16(acc[j], af, b0, b1);
        }
      }
    }
  };
  // chunk bookkeeping: before computing the first group of a chunk, that chunk's x must be resident; after the last
  // group of a chunk, all warps must be done with it before the buffer is refilled (two chunks later).
  auto enter = [&](int g) {
    const int s0 = g * G;
    if (s0 % 16 == 0) {
      const int chunk = s0 / 16;
      if (chunk + 1 < nchunks) { load_x(chunk + 1, (chunk + 1) & 1); cp_wait<1>(); } else { cp_wait<0>(); }
      __syncthreads();
    }
  };
  auto leave = [&](int g) {
    if (((g * G + G) % 16 == 0) || g == ngroups - 1) __syncthreads();
  };
  WT wa[G], wb[G];
  load_x(0, 0);
  load_w(0, wa);
  for (int g = 0; g < ngroups; g += 2) {
    if (g + 1 < ngroups) load_w(g + 1, wb);
    enter(g);
    compute(g, wa);
    leave(g);
    if (g + 1 >= ngroups) break;
    if (g + 2 < ngroups) load_w(g + 2, wa);
    enter(g + 1);
    compute(g + 1, wb);
    leave(g + 1);
  }
  if (!act) return;
  const int gq = lane >> 2, t = lane & 3;
  const int n0 = nt * 16 + gq, n1 = n0 + 8;
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int m0 = j * 8 + 2 * t, m1 = m0 + 1;
    float v[4] = {acc[j][0], acc[j][1], acc[j][2], acc[j][3]};
    if constexpr (EPI == EPI_F32) {
      float* p = a.part + (size_t)blockIdx.y * a.M * a.N;
      if (m0 < a.M) { p[(size_t)m0 * a.N + n0] = v[0]; p[(size_t)m0 * a.N + n1] = v[2]; }
      if (m1 < a.M) { p[(size_t)m1 * a.N + n0] = v[1]; p[(size_t)m1 * a.N + n1] = v[3]; }
    } else if constexpr (EPI == EPI_BF16) {
      if (m0 < a.M) { a.out[(size_t)m0 * a.ldo + n0] = __float2bfloat16_rn(v[0]); a.out[(size_t)m0 * a.ldo + n1] = __float2bfloat16_rn(v[2]); }
      if (m1 < a.M) { a.out[(size_t)m1 * a.ldo + n0] = __float2bfloat16_rn(v[1]); a.out[(size_t)m1 * a.ldo + n1] = __float2bfloat16_rn(v[3]); }
    } else {
      const float s0v = a.scale[n0], s1v = a.scale[n1];
      if (m0 < a.M) { a.part[(size_t)m0 * a.ldo + n0] = v[0] * s0v; a.part[(size_t)m0 * a.ldo + n1] = v[2] * s1v; }
      if (m1 < a.M) { a.part[(size_t)m1 * a.ldo + n0] = v[1] * s0v; a.part[(size_t)m1 * a.ldo + n1] = v[3] * s1v; }
    }
  }
}


// ---- int8 x int8 head GEMM (IMMA m16n8k32, int32 accumulate, exact), x = per-row int8 activations.
// W fragment layout for 16x32 int8 tiles: lane 4g + t holds 16 bytes W[g][4t..4t+3], W[g+8][4t..4t+3],
// W[g][4t+16..+19], W[g+8][4t+16..+19] (= the m16n8k32 A registers a0..a3). Output Bq[m][n] = acc * s_h[m] * s_w[n].
__device__ __forceinline__ void mma_s8(int* c, const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
constexpr int XS8 = KC + 16;  // smem row stride (bytes) of the int8 x chunk
struct GemmI8Args {
  const uint4* W;          // k32 fragment layout
  const int8_t* x;         // [M][K] int8
  const float* xs;         // [M][K / KC] activation scales s_h per 256-wide group
  const float* ws;         // [N] weight scales s_w
  float* out;              // [M][ldo]
  int M, N, K, ldo;
  // optional exact outlier channels: out += sum_{k in O} h[m][O_k] * Wo[n][k] (fp32)
  const __nv_bfloat16* h = nullptr;
  const int* ochan = nullptr;
  const __nv_bfloat16* Wo = nullptr;
};
template <int MT, int NW = GEMM_WARPS>
__global__ void __launch_bounds__(NW * 32) gemm_imma(GemmI8Args a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  int8_t* xsm = (int8_t*)smem_raw;  // [2][MT*8][XS8], then hO [MT*8][32] fp32 when outliers are on
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nt = blockIdx.x * NW + warp;
  const int ntiles = a.N / 16;
  const int KT = a.K / 32;
  const int nchunks = a.K / KC;
  const int MR = MT * 8;
  float* hO = (float*)(smem_raw + (size_t)2 * MR * XS8);
  if (a.Wo)
    for (int i = threadIdx.x; i < MR * 32; i += NW * 32)
      hO[i] = (i / 32) < a.M ? bf2f(a.h[(size_t)(i / 32) * H + a.ochan[i % 32]]) : 0.f;
  int acc[MT][4];
  float facc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
    facc[j][0] = facc[j][1] = facc[j][2] = facc[j][3] = 0.f;
  }
  const int NG = a.K / KC;
  const int tq = lane & 3;
  auto load_x = [&](int c, int buf) {
    const int pieces = MR * (KC / 16);
    for (int p = threadIdx.x; p < pieces; p += NW * 32) {
      int m = p / (KC / 16), q = (p % (KC / 16)) * 16;
      bool ok = m < a.M;
      cp_async16(xsm + ((size_t)buf * MR + m) * XS8 + q, a.x + (size_t)(ok ? m : 0) * a.K + c * KC + q, ok);
    }
    cp_commit();
  };
  load_x(0, 0);
  for (int c = 0; c < nchunks; ++c) {
    if (c + 1 < nchunks) { load_x(c + 1, (c + 1) & 1); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    const int8_t* xb = xsm + (size_t)(c & 1) * MR * XS8;
    if (nt < ntiles) {
      const uint4* wp = a.W + ((size_t)nt * KT + c * (KC / 32)) * 32 + lane;
      uint4 wf[KC / 32];
#pragma unroll
      for (int s = 0; s < KC / 32; ++s) wf[s] = ldg_stream(wp + (size_t)s * 32);
#pragma unroll
      for (int s = 0; s < KC / 32; ++s) {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          uint32_t b0, b1;  // 8 tokens x 32 k: two 8x16-byte matrices
          ldsm_x2(b0, b1, xb + (size_t)(j * 8 + (lane & 7)) * XS8 + s * 32 + ((lane >> 3) & 1) * 16);
          mma_s8(acc[j], wf[s], b0, b1);
        }
      }
#pragma unroll
      for (int j = 0; j < MT; ++j) {  // close the 256-wide group: scale by its activation scale, reset
        const int m0 = j * 8 + 2 * tq, m1 = m0 + 1;
        const float s0 = m0 < a.M ? a.xs[m0 * NG + c] : 0.f, s1 = m1 < a.M ? a.xs[m1 * NG + c] : 0.f;
        facc[j][0] += (float)acc[j][0] * s0; facc[j][2] += (float)acc[j][2] * s0;
        facc[j][1] += (float)acc[j][1] * s1; facc[j][3] += (float)acc[j][3] * s1;
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
      }
    }
    __syncthreads();
  }
  if (nt >= ntiles) return;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = nt * 16 + g, n1 = n0 + 8;
  const float w0 = a.ws[n0], w1 = a.ws[n1];
  if (a.Wo) {  // exact outlier channels (bf16 weights, fp32), 8 channels at a time (register budget)
#pragma unroll
    for (int j = 0; j < MT; ++j) { facc[j][0] *= w0; facc[j][2] *= w1; facc[j][1] *= w0; facc[j][3] *= w1; }
#pragma unroll
    for (int c8 = 0; c8 < 4; ++c8) {
      uint4 u0 = ((const uint4*)(a.Wo + (size_t)n0 * 32))[c8], u1 = ((const uint4*)(a.Wo + (size_t)n1 * 32))[c8];
      const __nv_bfloat162* p0 = (const __nv_bfloat162*)&u0;
      const __nv_bfloat162* p1 = (const __nv_bfloat162*)&u1;
      float wo0[8], wo1[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float2 f0 = __bfloat1622float2(p0[e]), f1 = __bfloat1622float2(p1[e]);
        wo0[2 * e] = f0.x; wo0[2 * e + 1] = f0.y; wo1[2 * e] = f1.x; wo1[2 * e + 1] = f1.y;
      }
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        const int m0 = j * 8 + 2 * t, m1 = m0 + 1;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const float h0 = hO[m0 * 32 + c8 * 8 + k], h1 = hO[m1 * 32 + c8 * 8 + k];
          facc[j][0] += h0 * wo0[k]; facc[j][2] += h0 * wo1[k]; facc[j][1] += h1 * wo0[k]; facc[j][3] += h1 * wo1[k];
        }
      }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      const int m0 = j * 8 + 2 * t, m1 = m0 + 1;
      if (m0 < a.M) { a.out[(size_t)m0 * a.ldo + n0] = facc[j][0]; a.out[(size_t)m0 * a.ldo + n1] = facc[j][2]; }
      if (m1 < a.M) { a.out[(size_t)m1 * a.ldo + n0] = facc[j][1]; a.out[(size_t)m1 * a.ldo + n1] = facc[j][3]; }
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int m0 = j * 8 + 2 * t, m1 = m0 + 1;
    if (m0 < a.M) {
      a.out[(size_t)m0 * a.ldo + n0] = facc[j][0] * w0;
      a.out[(size_t)m0 * a.ldo + n1] = facc[j][2] * w1;
    }
    if (m1 < a.M) {
      a.out[(size_t)m1 * a.ldo + n0] = facc[j][1] * w0;
      a.out[(size_t)m1 * a.ldo + n1] = facc[j][3] * w1;
    }
  }
}
__global__ void permute_frag_i8k32(const int8_t* w, int8_t* out, int N, int K) {
  size_t tiles = (size_t)(N / 16) * (K / 32);
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tiles * 32) return;
  size_t tile = idx / 32;
  int lane = idx % 32, g = lane >> 2, t = lane & 3;
  int nt = tile / (K / 32), kt = tile % (K / 32);
  int r0 = nt * 16 + g, r1 = r0 + 8, c0 = kt * 32 + 4 * t;
  int8_t* o = out + idx * 16;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    o[e] = w[(size_t)r0 * K + c0 + e];
    o[4 + e] = w[(size_t)r1 * K + c0 + e];
    o[8 + e] = w[(size_t)r0 * K + c0 + 16 + e];
    o[12 + e] = w[(size_t)r1 * K + c0 + 16 + e];
  }
}
// int8 activations of the head input, one scale per 256-wide group: s_g = max|h_g| / 127, hq = rint(h / s_g),
// dh = ||h - s hq||_2 over the row (rounded up). One block of 288 threads (9 warps = 9 groups) per row.
__global__ void k_quant_h(const __nv_bfloat16* h, int8_t* hq, float* hs, float* hdn, int rows,
                          const uint8_t* omask = nullptr, float* hn_in = nullptr) {
  // omask (optional): channels handled exactly outside the int8 path are zeroed here; hn_in = ||h without them||
  const int r = blockIdx.x, g = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __shared__ float red[9];
  __shared__ float red2[9];
  const __nv_bfloat16* hp = h + (size_t)r * H + g * KC;
  float x[8];
  float mx = 0.f, ss = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    x[e] = bf2f(hp[lane + e * 32]);
    if (omask && omask[g * KC + lane + e * 32]) x[e] = 0.f;
    mx = fmaxf(mx, fabsf(x[e]));
    ss += x[e] * x[e];
  }
  ss = warp_sum(ss);
  if (lane == 0) red2[g] = ss;
  mx = warp_max(mx);
  const float s = mx > 0.f ? mx / 127.f : 1.f;
  float e2 = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int q = max(-127, min(127, __float2int_rn(x[e] / s)));
    hq[(size_t)r * H + g * KC + lane + e * 32] = (int8_t)q;
    float d = x[e] - s * (float)q;
    e2 += d * d;
  }
  e2 = warp_sum(e2);
  if (lane == 0) { red[g] = e2; hs[r * (H / KC) + g] = s; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float t2 = 0.f, s2 = 0.f;
    for (int i = 0; i < H / KC; ++i) { t2 += red[i]; s2 += red2[i]; }
    hdn[r] = sqrtf(t2) * (1.0f + 1e-4f) + 1e-6f;
    if (hn_in) hn_in[r] = sqrtf(s2) * (1.0f + 1e-5f) + 1e-6f;
  }
}
// exact part of the head logits on the outlier channels: Bq[m][v] += sum_{k in O} h[m][k] W[v][k] (fp32, bf16 W
// gathered as Wo [VP][NO]); one thread per (v, m)
constexpr int NOUT = 32;
__global__ void k_head_outliers(const __nv_bfloat16* h, const int* ochan, const __nv_bfloat16* Wo, float* Bq, int M,
                                int NV) {
  __shared__ float hsm[64][NOUT];
  for (int i = threadIdx.x; i < M * NOUT; i += blockDim.x) hsm[i / NOUT][i % NOUT] = bf2f(h[(size_t)(i / NOUT) * H + ochan[i % NOUT]]);
  __syncthreads();
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= NV) return;
  float w[NOUT];
  const uint4* wp = (const uint4*)(Wo + (size_t)v * NOUT);
#pragma unroll
  for (int c = 0; c < NOUT / 8; ++c) {
    uint4 u = wp[c];
    const __nv_bfloat162* b2 = (const __nv_bfloat162*)&u;
#pragma unroll
    for (int e = 0; e < 4; ++e) { float2 f = __bfloat1622float2(b2[e]); w[c * 8 + 2 * e] = f.x; w[c * 8 + 2 * e + 1] = f.y; }
  }
  for (int m = 0; m < M; ++m) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < NOUT; ++k) acc += hsm[m][k] * w[k];
    Bq[(size_t)m * VP + v] += acc;
  }
}

// permute row-major W [N][K] (bf16) into fragment layout
__global__ void permute_frag_bf16(const __nv_bfloat16* w, __nv_bfloat16* out, int N, int K) {
  size_t tiles = (size_t)(N / 16) * (K / 16);
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tiles * 32) return;
  size_t tile = idx / 32;
  int lane = idx % 32, g = lane >> 2, t = lane & 3;
  int nt = tile / (K / 16), kt = tile % (K / 16);
  int r0 = nt * 16 + g, r1 = r0 + 8, c0 = kt * 16 + 2 * t;
  __nv_bfloat16* o = out + idx * 8;
  o[0] = w[(size_t)r0 * K + c0];     o[1] = w[(size_t)r0 * K + c0 + 1];
  o[2] = w[(size_t)r1 * K + c0];     o[3] = w[(size_t)r1 * K + c0 + 1];
  o[4] = w[(size_t)r0 * K + c0 + 8]; o[5] = w[(size_t)r0 * K + c0 + 9];
  o[6] = w[(size_t)r1 * K + c0 + 8]; o[7] = w[(size_t)r1 * K + c0 + 9];
}
// int8 per-row symmetric quantisation of the head + exact error norms (fp64): q = rint(w / s), s = max|w| / 127.
// enorm[v] = ||w_v - s q_v||_2 rounded up (fp32), row by row (one block per row).
__global__ void quant_rows_i8(const __nv_bfloat16* w, int8_t* q, float* scale, float* enorm, int K) {
  int v = blockIdx.x;
  const __nv_bfloat16* r = w + (size_t)v * K;
  __shared__ float red[32];
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(bf2f(r[k])));
  mx = warp_max(mx);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float m = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    m = warp_max(m);
    if (threadIdx.x == 0) red[0] = m;
  }
  __syncthreads();
  float s = red[0] > 0.f ? red[0] / 127.f : 1.f;
  __syncthreads();
  double e2 = 0.0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float x = bf2f(r[k]);
    int qi = __float2int_rn(x / s);
    qi = max(-127, min(127, qi));
    q[(size_t)v * K + k] = (int8_t)qi;
    double d = (double)x - (double)s * (double)qi;
    e2 += d * d;
  }
  __shared__ double rd[32];
  for (int o = 16; o; o >>= 1) e2 += __shfl_xor_sync(0xffffffffu, e2, o);
  if ((threadIdx.x & 31) == 0) rd[threadIdx.x >> 5] = e2;
  __syncthreads();
  if (threadIdx.x == 0) {
    double tot = 0;
    for (int i = 0; i < (int)blockDim.x / 32; ++i) tot += rd[i];
    scale[v] = s;
    enorm[v] = __double2float_ru(sqrt(tot) * (1.0 + 1e-6));
  }
}
__global__ void permute_frag_i8(const int8_t* w, int8_t* out, int N, int K) {
  size_t tiles = (size_t)(N / 16) * (K / 16);
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tiles * 32) return;
  size_t tile = idx / 32;
  int lane = idx % 32, g = lane >> 2, t = lane & 3;
  int nt = tile / (K / 16), kt = tile % (K / 16);
  int r0 = nt * 16 + g, r1 = r0 + 8, c0 = kt * 16 + 2 * t;
  int8_t e[8] = {w[(size_t)r0 * K + c0], w[(size_t)r0 * K + c0 + 1], w[(size_t)r1 * K + c0], w[(size_t)r1 * K + c0 + 1],
                 w[(size_t)r0 * K + c0 + 8], w[(size_t)r0 * K + c0 + 9], w[(size_t)r1 * K + c0 + 8], w[(size_t)r1 * K + c0 + 9]};
  int8_t* o = out + idx * 8;
  o[0] = e[0]; o[1] = e[2]; o[2] = e[1]; o[3] = e[3];
  o[4] = e[4]; o[5] = e[6]; o[6] = e[5]; o[7] = e[7];
}
// fp32 row L2 norms (rounded up) of an int8 matrix [N][K]
__global__ void row_norms_i8(const int8_t* q, float* out, int N, int K) {
  int v = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (v >= N) return;
  double s = 0;
  for (int k = threadIdx.x & 31; k < K; k += 32) { double x = q[(size_t)v * K + k]; s += x * x; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) out[v] = __double2float_ru(sqrt(s) * (1.0 + 1e-9));
}
// fp32 row L2 norms (rounded up) of a bf16 matrix [N][K]
__global__ void row_norms(const __nv_bfloat16* w, float* out, int N, int K) {
  int v = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (v >= N) return;
  double s = 0;
  for (int k = threadIdx.x & 31; k < K; k += 32) { double x = bf2f(w[(size_t)v * K + k]); s += x * x; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) out[v] = __double2float_ru(sqrt(s) * (1.0 + 1e-6));
}

// ------------------------------------------------------------------------------------------------ per-step context
struct Step {
  int n;                       // sessions
  const int* anchor;           // [n] anchor token
  const int* P;                // [n] anchor position (= committed length)
  const int* block_table;      // [n][max_blocks]
  int max_blocks;
};

__device__ __forceinline__ size_t slot_of(const Step& st, int r, int pos) {
  return (size_t)st.block_table[r * st.max_blocks + pos / PAGE] * PAGE + pos % PAGE;
}

// K/V cache layout per layer: [blocks][KVH][PAGE][HD] for K and V separately.
__device__ __forceinline__ size_t kv_off(size_t slot, int h) {
  return ((slot / PAGE) * KVH + h) * (size_t)PAGE * HD + (slot % PAGE) * HD;
}

// rope (neox, full head dim) in fp32 on pair (x1 at d, x2 at d+64), cos/sin table [pos][128] = [cos(64) | sin(64)]
__device__ __forceinline__ void rope_pair(float& x1, float& x2, const float* cs, int pos, int d) {
  float c = cs[(size_t)pos * HD + d], s = cs[(size_t)pos * HD + 64 + d];
  float o1 = x1 * c - x2 * s, o2 = x2 * c + x1 * s;
  x1 = o1; x2 = o2;
}

// ---- commit: mixes of the verified rows' 4 taps (3 layers) + f_rnn of each session's last row
// feats [T][4][H] bf16; alpha [4][4] fp32 (row l = layer l, row 3 = f_rnn); out cmix [3][T][H]; f_rnn [n][H]
__global__ void k_commit_mix(const __nv_bfloat16* feats, const float* alpha, const __nv_bfloat16* w0,
                             const __nv_bfloat16* w1, const __nv_bfloat16* w2, const __nv_bfloat16* wf,
                             __nv_bfloat16* cmix, __nv_bfloat16* f_rnn, int T) {
  // block = (row, which) with which 0..2 = layer mix, 3 = f_rnn (only for the last row of a session)
  int row = blockIdx.x, which = blockIdx.y;
  if (which == 3 && (row % BLK) != BLK - 1) return;
  const __nv_bfloat16* w = which == 0 ? w0 : which == 1 ? w1 : which == 2 ? w2 : wf;
  const float* al = alpha + which * 4;
  constexpr int PER = H / 256;  // 9
  float m[PER];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    int h = threadIdx.x + i * 256;
    float acc = 0.f;
#pragma unroll
    for (int tp = 0; tp < 4; ++tp) acc += bf2f(feats[((size_t)row * 4 + tp) * H + h]) * al[tp];
    m[i] = acc;
    ss += acc * acc;
  }
  __shared__ float red[8];
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  float inv = rsqrtf(tot / H + EPS);
  __nv_bfloat16* o = which == 3 ? f_rnn + (size_t)(row / BLK) * H : cmix + ((size_t)which * T + row) * H;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    int h = threadIdx.x + i * 256;
    o[h] = __float2bfloat16_rn(m[i] * inv * bf2f(w[h]));
  }
}

// ---- f_rnn[s] = mix_rmsnorm(feats of session s's last committed row) (alpha row 3, markov_head.dflare_norm)
__global__ void k_frnn(const __nv_bfloat16* feats_rows, const float* alpha, const __nv_bfloat16* wf, __nv_bfloat16* f_rnn) {
  const int s = blockIdx.x;  // feats_rows [n][4][H]
  const float* al = alpha + 12;
  constexpr int PER = H / 256;
  float m[PER];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    int h = threadIdx.x + i * 256;
    float acc = 0.f;
#pragma unroll
    for (int tp = 0; tp < 4; ++tp) acc += bf2f(feats_rows[((size_t)s * 4 + tp) * H + h]) * al[tp];
    m[i] = acc;
    ss += acc * acc;
  }
  __shared__ float red[8];
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  float inv = rsqrtf(tot / H + EPS);
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    int h = threadIdx.x + i * 256;
    f_rnn[(size_t)s * H + h] = __float2bfloat16_rn(m[i] * inv * bf2f(wf[h]));
  }
}

// ---- commit epilogue: kv_ctx partials [3][SK][T][2048] -> k: k_norm_ctx + rope -> cache; v -> cache
// grid (T, 3), block 512 = 16 warps: warps 0-7 = K heads, 8-15 = V heads
struct CacheL {
  __nv_bfloat16 *k, *v;
  uint8_t *k8 = nullptr, *v8 = nullptr;  // lossy-draft option: E4M3 (unit scale, RNE, saturating) copy written instead
};
__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
// exact E4M3 -> fp32 (finite codes): place e/m in an fp32 with exponent field e, then scale by 2^120
__device__ __forceinline__ float e4m3f(uint32_t b) {
  uint32_t u = ((b & 0x80u) << 24) | ((b & 0x7Fu) << 20);
  return __fmul_rn(__uint_as_float(u), 1.329227995784916e36f);  // 2^120
}
struct Caches { CacheL l[3]; };
// rows: default = the step's n*8 verified rows (session row/8 at P-8+row%8); with row_pos (+ row_sess) any rows of
// any sessions at any positions (prefill chunks, ragged commits, rollback re-commits).
__global__ void epi_kvctx(const float* part, int SK, int T, Step st, const __nv_bfloat16* kn0,
                          const __nv_bfloat16* kn1, const __nv_bfloat16* kn2, const float* cs, Caches cc,
                          const int* row_pos = nullptr, const int* row_sess = nullptr,
                          const int* bt_swa = nullptr) {
  int row = blockIdx.x, l = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const __nv_bfloat16* kn = l == 0 ? kn0 : l == 1 ? kn1 : kn2;
  int r = row / BLK, j = row % BLK;
  int pos = st.P[r] - BLK + j;  // the step's verified rows sit at P-8 .. P-1
  if (row_pos) { pos = row_pos[row]; r = row_sess ? row_sess[row] : 0; }
  int head = warp & 7;
  bool isv = warp >= 8;
  int col = (isv ? KVH * HD : 0) + head * HD;
  const float* pb = part + (size_t)l * SK * T * 2048;
  float x[4];
  int d0 = 2 * lane, d1 = 64 + 2 * lane;
  int ds[4] = {d0, d0 + 1, d1, d1 + 1};
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    float s = 0.f;
    for (int k = 0; k < SK; ++k) s += pb[((size_t)k * T + row) * 2048 + col + ds[e]];
    x[e] = rb(s);  // linear output bf16
  }
  const int* bt = (l != 1 && bt_swa) ? bt_swa : st.block_table;
  size_t slot = (size_t)bt[r * st.max_blocks + pos / PAGE] * PAGE + pos % PAGE;
  size_t off = kv_off(slot, head);
  if (isv) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if (cc.l[l].v) cc.l[l].v[off + ds[e]] = __float2bfloat16_rn(x[e]);
      if (cc.l[l].v8) cc.l[l].v8[off + ds[e]] = f2e4m3(x[e]);
    }
    return;
  }
  float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  float inv = rsqrtf(ss / HD + EPS);  // flashinfer rmsnorm
#pragma unroll
  for (int e = 0; e < 4; ++e) x[e] = rb(x[e] * inv * bf2f(kn[ds[e]]));
  rope_pair(x[0], x[2], cs, pos, d0);
  rope_pair(x[1], x[3], cs, pos, d0 + 1);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    if (cc.l[l].k) cc.l[l].k[off + ds[e]] = __float2bfloat16_rn(x[e]);
    if (cc.l[l].k8) cc.l[l].k8[off + ds[e]] = f2e4m3(rb(x[e]));
  }
}

// ---- block input: residual = [embed_norm(anchor), mask_embed x 7] (bf16), xn = rmsnorm(residual, in_norm0)
__global__ void k_embed_block(Step st, const __nv_bfloat16* embed, const __nv_bfloat16* embed_norm_w,
                              const __nv_bfloat16* mask_embed, const __nv_bfloat16* in_norm0,
                              __nv_bfloat16* residual, __nv_bfloat16* xn) {
  int row = blockIdx.x, r = row / BLK, j = row % BLK;
  constexpr int PER = H / 256;
  __shared__ float red[8];
  float x[PER];
  float ss;
  if (j == 0) {
    const __nv_bfloat16* e = embed + (size_t)st.anchor[r] * H;
    ss = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) { x[i] = bf2f(e[threadIdx.x + i * 256]); ss += x[i] * x[i]; }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < 8; ++i) tot += red[i];
    float inv = rsqrtf(tot / H + EPS);
#pragma unroll
    for (int i = 0; i < PER; ++i) x[i] = rb(x[i] * inv * bf2f(embed_norm_w[threadIdx.x + i * 256]));
    __syncthreads();
  } else {
#pragma unroll
    for (int i = 0; i < PER; ++i) x[i] = bf2f(mask_embed[threadIdx.x + i * 256]);
  }
  ss = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) { residual[(size_t)row * H + threadIdx.x + i * 256] = __float2bfloat16_rn(x[i]); ss += x[i] * x[i]; }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < 8; ++i) tot += red[i];
  float inv = rsqrtf(tot / H + EPS);
#pragma unroll
  for (int i = 0; i < PER; ++i)
    xn[(size_t)row * H + threadIdx.x + i * 256] = __float2bfloat16_rn(x[i] * inv * bf2f(in_norm0[threadIdx.x + i * 256]));
}

// ---- qkv epilogue: partials [SK][T][5120] -> bf16 -> q/k per-head rmsnorm (rounded) + rope -> q buffer; k, v -> cache
// grid T, block 1024 = 32 warps (24 q heads + 8 k heads), then V by the k warps
__global__ void epi_qkv(const float* part, int SK, int T, Step st, const __nv_bfloat16* qn, const __nv_bfloat16* kn,
                        const float* cs, CacheL cache, __nv_bfloat16* qout) {
  int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int r = row / BLK, j = row % BLK;
  int pos = st.P[r] + j;
  constexpr int N = (NH + 2 * KVH) * HD;
  int d0 = 2 * lane, d1 = 64 + 2 * lane;
  int ds[4] = {d0, d0 + 1, d1, d1 + 1};
  auto get = [&](int col, float* x) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float s = 0.f;
      for (int k = 0; k < SK; ++k) s += part[((size_t)k * T + row) * N + col + ds[e]];
      x[e] = rb(s);
    }
  };
  float x[4];
  get(warp * HD, x);
  bool isq = warp < NH;
  const __nv_bfloat16* w = isq ? qn : kn;
  // config B _qk_norm_rope_kernel: inv = 1 / sqrt(ms + eps); x * inv * w -> bf16; rope fp32 -> bf16
  float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  float inv = 1.0f / sqrtf(ss / HD + EPS);
#pragma unroll
  for (int e = 0; e < 4; ++e) x[e] = rb(x[e] * inv * bf2f(w[ds[e]]));
  rope_pair(x[0], x[2], cs, pos, d0);
  rope_pair(x[1], x[3], cs, pos, d0 + 1);
  if (isq) {
#pragma unroll
    for (int e = 0; e < 4; ++e) qout[(size_t)row * NH * HD + warp * HD + ds[e]] = __float2bfloat16_rn(x[e]);
  } else {
    int h = warp - NH;
    size_t off = kv_off(slot_of(st, r, pos), h);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if (cache.k) cache.k[off + ds[e]] = __float2bfloat16_rn(x[e]);
      if (cache.k8) cache.k8[off + ds[e]] = f2e4m3(rb(x[e]));
    }
    float v[4];
    get((NH + KVH) * HD + h * HD, v);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if (cache.v) cache.v[off + ds[e]] = __float2bfloat16_rn(v[e]);
      if (cache.v8) cache.v8[off + ds[e]] = f2e4m3(v[e]);
    }
  }
}

// ---- partials [SK][T][H] -> bf16 x; residual += x (fp32 sum, stored bf16); xn = rmsnorm(unrounded sum) * w
// (flashinfer fused_add_rmsnorm). If hout != nullptr (final norm): rows j >= 1 of each block go to the head input
// hout[(r*7 + j-1)] (bf16) and hf16 (f16 copy) + x7 [7n][2H] = [h, f_rnn] for the rnn context GEMM, and
// hnorm[(r*7+j-1)] = ||h||_2 (fp32, rounded up).
__global__ void epi_addnorm(const float* part, int SK, int T, __nv_bfloat16* residual, const __nv_bfloat16* w,
                            __nv_bfloat16* xn, __nv_bfloat16* hout, __half* hf16, __nv_bfloat16* x7,
                            const __nv_bfloat16* f_rnn, float* hnorm) {
  int row = blockIdx.x;
  constexpr int PER = H / 768;
  float s[PER];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    int h = threadIdx.x + i * 768;
    float a = 0.f;
    for (int k = 0; k < SK; ++k) a += part[((size_t)k * T + row) * H + h];
    float sum = bf2f(residual[(size_t)row * H + h]) + rb(a);
    residual[(size_t)row * H + h] = __float2bfloat16_rn(sum);
    s[i] = sum;
    ss += sum * sum;
  }
  __shared__ float red[24];
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < 24; ++i) tot += red[i];
  float inv = rsqrtf(tot / H + EPS);
  int r = row / BLK, j = row % BLK;
  float hh = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    int h = threadIdx.x + i * 768;
    __nv_bfloat16 o = __float2bfloat16_rn(s[i] * inv * bf2f(w[h]));
    if (xn) xn[(size_t)row * H + h] = o;
    if (hout && j >= 1) {
      size_t hr = (size_t)r * 7 + (j - 1);
      hout[hr * H + h] = o;
      float of = bf2f(o);
      hf16[hr * H + h] = __float2half_rn(of);
      x7[hr * 2 * H + h] = o;
      x7[hr * 2 * H + H + h] = f_rnn[(size_t)r * H + h];
      hh += of * of;
    }
  }
  if (hout && j >= 1) {
    __syncthreads();
    hh = warp_sum(hh);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = hh;
    __syncthreads();
    if (threadIdx.x == 0) {
      float t2 = 0.f;
      for (int i = 0; i < 24; ++i) t2 += red[i];
      hnorm[(size_t)r * 7 + (j - 1)] = sqrtf(t2) * (1.0f + 1e-5f) + 1e-6f;
    }
  }
}

// ---- gate_up partials [SK][T][2*INTER] -> silu(bf16 g) * bf16 u (fp32, IEEE div, precise exp) -> bf16
__global__ void epi_silu(const float* part, int SK, int T, __nv_bfloat16* out) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)T * INTER) return;
  int row = idx / INTER, c = idx % INTER;
  float g = 0.f, u = 0.f;
  for (int k = 0; k < SK; ++k) {
    g += part[((size_t)k * T + row) * 2 * INTER + c];
    u += part[((size_t)k * T + row) * 2 * INTER + INTER + c];
  }
  g = rb(g); u = rb(u);
  float si = __fdiv_rn(g, 1.0f + expf(-g));
  out[idx] = __float2bfloat16_rn(si * u);
}

// ---- rnn context: ctx[(r*7+j)][768] = rnn_b + sum partials (fp32 out, addmm)
__global__ void epi_ctx(const float* part, int SK, int M, const float* rnn_b, float* ctx) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)M * 768) return;
  int c = idx % 768;
  float s = 0.f;
  for (int k = 0; k < SK; ++k) s += part[(size_t)k * M * 768 + idx];
  ctx[idx] = s + rnn_b[c];
}

// ------------------------------------------------------------------------------------------------ block attention
// grid (splits, KVH, n), block 128 (4 warps = 2 m-tiles x 2 key halves of each 64-key tile).
// q [T][NH][HD] bf16 (rows r*8 .. r*8+7), query row qi = j*3 + g3 (head h*3 + g3), 24 rows (+8 zero padding).
// Partial outputs: o_part [n][KVH][splits][32][HD] fp32 (normalised by l), ml [n][KVH][splits][32][2] (m log2, l).
constexpr int AT_TILE = 64, AT_KS = HD + 8;
constexpr int ATTN_SMEM = (4 * AT_TILE + 32) * AT_KS * 2;
__global__ void __launch_bounds__(128) attn_block(const __nv_bfloat16* q, Step st, CacheL cache, int window,
                                                  int splits, float* o_part, float* ml) {
  extern __shared__ __align__(16) unsigned char at_smem[];  // dynamic: ATTN_SMEM bytes
  auto ks = (__nv_bfloat16(*)[AT_TILE][AT_KS])at_smem;
  auto vs = (__nv_bfloat16(*)[AT_TILE][AT_KS])(at_smem + 2 * AT_TILE * AT_KS * 2);
  auto qsm = (__nv_bfloat16(*)[AT_KS])(at_smem + 4 * AT_TILE * AT_KS * 2);
  __shared__ float comb[2][32][2];
  const int sp = blockIdx.x, h = blockIdx.y, r = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int mt = warp & 1, kh = warp >> 1;
  const int seqlen = st.P[r] + BLK;
  // page-aligned key range: tile t = page (kstart / PAGE + t), contiguous 16 KB of K and of V in this layout
  const int kwin = window ? max(0, seqlen - window) : 0;   // first key inside the window
  const int kstart = (kwin / PAGE) * PAGE;
  const int ntile_total = (seqlen - kstart + AT_TILE - 1) / AT_TILE;
  const int per = (ntile_total + splits - 1) / splits;
  const int t0 = sp * per, t1 = min(ntile_total, t0 + per);
  const float sl2 = 0.08838834764831845f * 1.4426950408889634f;  // D^-0.5 * log2(e)
  __shared__ int pg[256];  // block-table entries of this CTA's tiles
  for (int i = threadIdx.x; i < t1 - t0 && i < 256; i += 128) pg[i] = st.block_table[r * st.max_blocks + kstart / PAGE + t0 + i];
  // load Q (24 rows; rows 24..31 zero)
  for (int i = threadIdx.x; i < 32 * (HD / 8); i += 128) {
    int row = i / (HD / 8), c = (i % (HD / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < 24) {
      int j = row / 3, g3 = row % 3;
      v = *(const uint4*)(q + ((size_t)(r * BLK + j) * NH + h * 3 + g3) * HD + c);
    }
    *(uint4*)&qsm[row][c] = v;
  }
  __syncthreads();
  auto load_tile = [&](int tt, int buf) {
    const size_t base = ((size_t)pg[tt - t0] * KVH + h) * (size_t)PAGE * HD;
    const __nv_bfloat16* kg = cache.k + base;
    const __nv_bfloat16* vg = cache.v + base;
#pragma unroll
    for (int it = 0; it < (AT_TILE * HD / 8) / 128; ++it) {  // 8 pieces of 16 B per thread for K and for V
      int ii = threadIdx.x + it * 128;
      int kk = ii / (HD / 8), c = (ii % (HD / 8)) * 8;
      cp_async16(&ks[buf][kk][c], kg + (size_t)kk * HD + c, true);
      cp_async16(&vs[buf][kk][c], vg + (size_t)kk * HD + c, true);
    }
    cp_commit();
  };
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  uint32_t qf[8][4];
  if (t0 < t1) load_tile(t0, 0);
  __syncthreads();
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) ldsm_x4(qf[kk], &qsm[mt * 16 + (lane & 15)][kk * 16 + (lane >> 4) * 8]);
  const int g = lane >> 2, t = lane & 3;
  for (int tt = t0; tt < t1; ++tt) {
    const int buf = (tt - t0) & 1;
    if (tt + 1 < t1) { load_tile(tt + 1, buf ^ 1); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncthreads();
    // S = Q K^T for 32 keys (this warp's half): 4 n-tiles of 8 keys
    float s[4][4];
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      s[nn][0] = s[nn][1] = s[nn][2] = s[nn][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint32_t b0, b1;
        ldsm_x2(b0, b1, &ks[buf][kh * 32 + nn * 8 + (lane & 7)][kk * 16 + ((lane >> 3) & 1) * 8]);
        mma_bf16(s[nn], qf[kk], b0, b1);
      }
    }
    // mask + scale (log2 domain)
    const int kbase = kstart + tt * AT_TILE + kh * 32;
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nn = 0; nn < 4; ++nn)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int key = kbase + nn * 8 + 2 * t + (e & 1);
        float v = (key < seqlen && key >= kwin) ? s[nn][e] * sl2 : -INFINITY;
        s[nn][e] = v;
        mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
    float scale_old[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 1));
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 2));
      float mnew = fmaxf(m_run[hh], mx[hh]);
      scale_old[hh] = (m_run[hh] == -INFINITY) ? 0.f : exp2f(m_run[hh] - mnew);
      m_run[hh] = mnew;
      l_run[hh] *= scale_old[hh];
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      o[i][0] *= scale_old[0]; o[i][1] *= scale_old[0];
      o[i][2] *= scale_old[1]; o[i][3] *= scale_old[1];
    }
    uint32_t pa[2][4];  // P as A fragments: 2 k16 steps (keys 0-15, 16-31 of this half)
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      float p0 = m_run[0] == -INFINITY ? 0.f : exp2f(s[nn][0] - m_run[0]);
      float p1 = m_run[0] == -INFINITY ? 0.f : exp2f(s[nn][1] - m_run[0]);
      float p2 = m_run[1] == -INFINITY ? 0.f : exp2f(s[nn][2] - m_run[1]);
      float p3 = m_run[1] == -INFINITY ? 0.f : exp2f(s[nn][3] - m_run[1]);
      l_run[0] += p0 + p1;
      l_run[1] += p2 + p3;
      __nv_bfloat162 lo = __floats2bfloat162_rn(p0, p1), hi = __floats2bfloat162_rn(p2, p3);
      int ks16 = nn >> 1, half = nn & 1;
      pa[ks16][half * 2 + 0] = *(uint32_t*)&lo;
      pa[ks16][half * 2 + 1] = *(uint32_t*)&hi;
    }
    // O += P V : V tile rows = keys (k dim), cols = d (n dim) -> ldmatrix.trans
#pragma unroll
    for (int ks16 = 0; ks16 < 2; ++ks16) {
#pragma unroll
      for (int dn = 0; dn < 16; dn += 2) {
        uint32_t bv[4];
        ldsm_x4_t(bv, &vs[buf][kh * 32 + ks16 * 16 + (lane & 15)][dn * 8 + (lane >> 4) * 8]);
        mma_bf16(o[dn], pa[ks16], bv[0], bv[1]);
        mma_bf16(o[dn + 1], pa[ks16], bv[2], bv[3]);
      }
    }
    __syncthreads();
  }
  // row sums across the 4 lanes of a row
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    l_run[hh] += __shfl_xor_sync(0xffffffffu, l_run[hh], 1);
    l_run[hh] += __shfl_xor_sync(0xffffffffu, l_run[hh], 2);
  }
  // combine the two key halves (kh 0, 1) through shared memory: kh 1 writes m, l, o; kh 0 merges
  float* osm = (float*)&ks[0][0][0];  // reuse: [2 mt][16 rows][128] fp32 = 16 KB (ks is 2*64*136*2 = 34 KB)
  if (kh == 1) {
    if (t == 0) {
      comb[mt][g][0] = m_run[0]; comb[mt][g][1] = l_run[0];
      comb[mt][g + 8][0] = m_run[1]; comb[mt][g + 8][1] = l_run[1];
    }
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
      int c = dn * 8 + 2 * t;
      osm[(mt * 16 + g) * HD + c] = o[dn][0]; osm[(mt * 16 + g) * HD + c + 1] = o[dn][1];
      osm[(mt * 16 + g + 8) * HD + c] = o[dn][2]; osm[(mt * 16 + g + 8) * HD + c + 1] = o[dn][3];
    }
  }
  __syncthreads();
  if (kh == 0) {
    size_t pbase = (((size_t)r * KVH + h) * splits + sp) * 32;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      int row = g + hh * 8;
      float m2 = comb[mt][row][0], l2 = comb[mt][row][1];
      float mn = fmaxf(m_run[hh], m2);
      float a1 = m_run[hh] == -INFINITY ? 0.f : exp2f(m_run[hh] - mn);
      float a2 = m2 == -INFINITY ? 0.f : exp2f(m2 - mn);
      float l = l_run[hh] * a1 + l2 * a2;
      float il = l > 0.f ? 1.f / l : 0.f;
      int qrow = mt * 16 + row;
#pragma unroll
      for (int dn = 0; dn < 16; ++dn) {
        int c = dn * 8 + 2 * t;
        float v0 = (o[dn][hh * 2] * a1 + osm[qrow * HD + c] * a2) * il;
        float v1 = (o[dn][hh * 2 + 1] * a1 + osm[qrow * HD + c + 1] * a2) * il;
        *(float2*)&o_part[(pbase + qrow) * HD + c] = make_float2(v0, v1);
      }
      if (t == 0) { ml[(pbase + qrow) * 2] = mn; ml[(pbase + qrow) * 2 + 1] = l; }
    }
  }
}
__global__ void __launch_bounds__(128) attn_block8(const __nv_bfloat16* q, Step st, CacheL cache, int window,
                                                  int splits, float* o_part, float* ml) {
  extern __shared__ __align__(16) unsigned char at_smem[];  // dynamic: ATTN_SMEM bytes
  auto ks = (__nv_bfloat16(*)[AT_TILE][AT_KS])at_smem;
  auto vs = (__nv_bfloat16(*)[AT_TILE][AT_KS])(at_smem + 2 * AT_TILE * AT_KS * 2);
  auto qsm = (__nv_bfloat16(*)[AT_KS])(at_smem + 4 * AT_TILE * AT_KS * 2);
  __shared__ float comb[2][32][2];
  const int sp = blockIdx.x, h = blockIdx.y, r = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int mt = warp & 1, kh = warp >> 1;
  const int seqlen = st.P[r] + BLK;
  // page-aligned key range: tile t = page (kstart / PAGE + t), contiguous 16 KB of K and of V in this layout
  const int kwin = window ? max(0, seqlen - window) : 0;   // first key inside the window
  const int kstart = (kwin / PAGE) * PAGE;
  const int ntile_total = (seqlen - kstart + AT_TILE - 1) / AT_TILE;
  const int per = (ntile_total + splits - 1) / splits;
  const int t0 = sp * per, t1 = min(ntile_total, t0 + per);
  const float sl2 = 0.08838834764831845f * 1.4426950408889634f;  // D^-0.5 * log2(e)
  __shared__ int pg[256];  // block-table entries of this CTA's tiles
  for (int i = threadIdx.x; i < t1 - t0 && i < 256; i += 128) pg[i] = st.block_table[r * st.max_blocks + kstart / PAGE + t0 + i];
  // load Q (24 rows; rows 24..31 zero)
  for (int i = threadIdx.x; i < 32 * (HD / 8); i += 128) {
    int row = i / (HD / 8), c = (i % (HD / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < 24) {
      int j = row / 3, g3 = row % 3;
      v = *(const uint4*)(q + ((size_t)(r * BLK + j) * NH + h * 3 + g3) * HD + c);
    }
    *(uint4*)&qsm[row][c] = v;
  }
  __syncthreads();
  // FP8 path: K/V pages are E4M3 (8 KB each per tile); each thread loads 4 x 16 B of K and of V into registers
  // (next tile in flight during the current tile's math) and converts them exactly to bf16 in shared memory.
  uint4 rk[4], rv[4];
  auto fetch = [&](int tt) {
    const size_t base = ((size_t)pg[tt - t0] * KVH + h) * (size_t)PAGE * HD;
    const uint4* kg = (const uint4*)(cache.k8 + base);
    const uint4* vg = (const uint4*)(cache.v8 + base);
#pragma unroll
    for (int it = 0; it < 4; ++it) { rk[it] = ldg_stream(kg + threadIdx.x + it * 128); rv[it] = ldg_stream(vg + threadIdx.x + it * 128); }
  };
  auto store = [&](int buf) {
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      int ii = threadIdx.x + it * 128;           // 16-byte piece index: 8 pieces per 128-wide row
      int kk = ii / (HD / 16), c = (ii % (HD / 16)) * 16;
      uint32_t wk[4] = {rk[it].x, rk[it].y, rk[it].z, rk[it].w}, wv[4] = {rv[it].x, rv[it].y, rv[it].z, rv[it].w};
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        __nv_bfloat162 k01 = __floats2bfloat162_rn(e4m3f(wk[q4] & 0xff), e4m3f((wk[q4] >> 8) & 0xff));
        __nv_bfloat162 k23 = __floats2bfloat162_rn(e4m3f((wk[q4] >> 16) & 0xff), e4m3f(wk[q4] >> 24));
        __nv_bfloat162 v01 = __floats2bfloat162_rn(e4m3f(wv[q4] & 0xff), e4m3f((wv[q4] >> 8) & 0xff));
        __nv_bfloat162 v23 = __floats2bfloat162_rn(e4m3f((wv[q4] >> 16) & 0xff), e4m3f(wv[q4] >> 24));
        *(__nv_bfloat162*)&ks[buf][kk][c + q4 * 4] = k01;
        *(__nv_bfloat162*)&ks[buf][kk][c + q4 * 4 + 2] = k23;
        *(__nv_bfloat162*)&vs[buf][kk][c + q4 * 4] = v01;
        *(__nv_bfloat162*)&vs[buf][kk][c + q4 * 4 + 2] = v23;
      }
    }
  };
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  uint32_t qf[8][4];
  if (t0 < t1) { fetch(t0); store(0); }
  if (t0 + 1 < t1) fetch(t0 + 1);
  __syncthreads();
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) ldsm_x4(qf[kk], &qsm[mt * 16 + (lane & 15)][kk * 16 + (lane >> 4) * 8]);
  const int g = lane >> 2, t = lane & 3;
  for (int tt = t0; tt < t1; ++tt) {
    const int buf = (tt - t0) & 1;
    // S = Q K^T for 32 keys (this warp's half): 4 n-tiles of 8 keys
    float s[4][4];
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      s[nn][0] = s[nn][1] = s[nn][2] = s[nn][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint32_t b0, b1;
        ldsm_x2(b0, b1, &ks[buf][kh * 32 + nn * 8 + (lane & 7)][kk * 16 + ((lane >> 3) & 1) * 8]);
        mma_bf16(s[nn], qf[kk], b0, b1);
      }
    }
    // mask + scale (log2 domain)
    const int kbase = kstart + tt * AT_TILE + kh * 32;
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nn = 0; nn < 4; ++nn)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int key = kbase + nn * 8 + 2 * t + (e & 1);
        float v = (key < seqlen && key >= kwin) ? s[nn][e] * sl2 : -INFINITY;
        s[nn][e] = v;
        mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
    float scale_old[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 1));
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 2));
      float mnew = fmaxf(m_run[hh], mx[hh]);
      scale_old[hh] = (m_run[hh] == -INFINITY) ? 0.f : exp2f(m_run[hh] - mnew);
      m_run[hh] = mnew;
      l_run[hh] *= scale_old[hh];
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      o[i][0] *= scale_old[0]; o[i][1] *= scale_old[0];
      o[i][2] *= scale_old[1]; o[i][3] *= scale_old[1];
    }
    uint32_t pa[2][4];  // P as A fragments: 2 k16 steps (keys 0-15, 16-31 of this half)
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      float p0 = m_run[0] == -INFINITY ? 0.f : exp2f(s[nn][0] - m_run[0]);
      float p1 = m_run[0] == -INFINITY ? 0.f : exp2f(s[nn][1] - m_run[0]);
      float p2 = m_run[1] == -INFINITY ? 0.f : exp2f(s[nn][2] - m_run[1]);
      float p3 = m_run[1] == -INFINITY ? 0.f : exp2f(s[nn][3] - m_run[1]);
      l_run[0] += p0 + p1;
      l_run[1] += p2 + p3;
      __nv_bfloat162 lo = __floats2bfloat162_rn(p0, p1), hi = __floats2bfloat162_rn(p2, p3);
      int ks16 = nn >> 1, half = nn & 1;
      pa[ks16][half * 2 + 0] = *(uint32_t*)&lo;
      pa[ks16][half * 2 + 1] = *(uint32_t*)&hi;
    }
    // O += P V : V tile rows = keys (k dim), cols = d (n dim) -> ldmatrix.trans
#pragma unroll
    for (int ks16 = 0; ks16 < 2; ++ks16) {
#pragma unroll
      for (int dn = 0; dn < 16; dn += 2) {
        uint32_t bv[4];
        ldsm_x4_t(bv, &vs[buf][kh * 32 + ks16 * 16 + (lane & 15)][dn * 8 + (lane >> 4) * 8]);
        mma_bf16(o[dn], pa[ks16], bv[0], bv[1]);
        mma_bf16(o[dn + 1], pa[ks16], bv[2], bv[3]);
      }
    }
    if (tt + 1 < t1) {
      store(buf ^ 1);
      if (tt + 2 < t1) fetch(tt + 2);
    }
    __syncthreads();
  }
  // row sums across the 4 lanes of a row
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    l_run[hh] += __shfl_xor_sync(0xffffffffu, l_run[hh], 1);
    l_run[hh] += __shfl_xor_sync(0xffffffffu, l_run[hh], 2);
  }
  // combine the two key halves (kh 0, 1) through shared memory: kh 1 writes m, l, o; kh 0 merges
  float* osm = (float*)&ks[0][0][0];  // reuse: [2 mt][16 rows][128] fp32 = 16 KB (ks is 2*64*136*2 = 34 KB)
  if (kh == 1) {
    if (t == 0) {
      comb[mt][g][0] = m_run[0]; comb[mt][g][1] = l_run[0];
      comb[mt][g + 8][0] = m_run[1]; comb[mt][g + 8][1] = l_run[1];
    }
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
      int c = dn * 8 + 2 * t;
      osm[(mt * 16 + g) * HD + c] = o[dn][0]; osm[(mt * 16 + g) * HD + c + 1] = o[dn][1];
      osm[(mt * 16 + g + 8) * HD + c] = o[dn][2]; osm[(mt * 16 + g + 8) * HD + c + 1] = o[dn][3];
    }
  }
  __syncthreads();
  if (kh == 0) {
    size_t pbase = (((size_t)r * KVH + h) * splits + sp) * 32;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      int row = g + hh * 8;
      float m2 = comb[mt][row][0], l2 = comb[mt][row][1];
      float mn = fmaxf(m_run[hh], m2);
      float a1 = m_run[hh] == -INFINITY ? 0.f : exp2f(m_run[hh] - mn);
      float a2 = m2 == -INFINITY ? 0.f : exp2f(m2 - mn);
      float l = l_run[hh] * a1 + l2 * a2;
      float il = l > 0.f ? 1.f / l : 0.f;
      int qrow = mt * 16 + row;
#pragma unroll
      for (int dn = 0; dn < 16; ++dn) {
        int c = dn * 8 + 2 * t;
        float v0 = (o[dn][hh * 2] * a1 + osm[qrow * HD + c] * a2) * il;
        float v1 = (o[dn][hh * 2 + 1] * a1 + osm[qrow * HD + c + 1] * a2) * il;
        *(float2*)&o_part[(pbase + qrow) * HD + c] = make_float2(v0, v1);
      }
      if (t == 0) { ml[(pbase + qrow) * 2] = mn; ml[(pbase + qrow) * 2 + 1] = l; }
    }
  }
}
// merge splits -> out [T][NH*HD] bf16. grid (n*KVH*24 query rows), block 128 (thread per d)
__global__ void attn_combine(const float* o_part, const float* ml, int splits, __nv_bfloat16* out) {
  const int qr = blockIdx.x % 24, rh = blockIdx.x / 24, r = rh / KVH, h = rh % KVH, d = threadIdx.x;
  const size_t base = ((size_t)r * KVH + h) * splits * 32;
  float mx = -INFINITY;
  for (int s = 0; s < splits; ++s) mx = fmaxf(mx, ml[(base + s * 32 + qr) * 2]);
  float num = 0.f, den = 0.f;
  for (int s = 0; s < splits; ++s) {
    float m = ml[(base + s * 32 + qr) * 2], l = ml[(base + s * 32 + qr) * 2 + 1];
    if (m == -INFINITY || l == 0.f) continue;
    float w = exp2f(m - mx) * l;
    num += o_part[(base + s * 32 + qr) * HD + d] * w;
    den += w;
  }
  const int j = qr / 3, g3 = qr % 3;
  out[((size_t)(r * BLK + j) * NH + h * 3 + g3) * HD + d] = __float2bfloat16_rn(den > 0.f ? num / den : 0.f);
}

// ------------------------------------------------------------------------------------------------ Markov head
struct Markov {
  int n;
  const int* anchor;                 // [n]
  const float* ctx;                  // [n*7][768] fp32 (row r*7+i)
  const __nv_bfloat16* w_state;      // [768][512] row-major (rnn_proj[:, :512])
  const __nv_bfloat16* w1;           // [VP][256]
  const __nv_bfloat16* fold_w;       // [VP][256]
  const __nv_bfloat16* fold_b;       // [VP]
  const float* rn;                   // [VP] ||fold_w_v||_2 rounded up
  const __nv_bfloat16* B;            // exact variant: base logits bf16 [n*7][VP]
  const float* Bq;                   // int8 variant: base logits from int8 head, fp32 [n*7][VP]
  const float* enorm;                // int8: ||head_v - s q_v||_2 (rounded up)
  const float* hrn;                  // int8: ||head_v||_2 (rounded up), for the fp32 summation slack
  const float* hnorm;                // int8: ||h_row||_2 (rounded up) [n*7]
  const float* hdn;                  // imma: ||h - s_h hq||_2 per head row (nullptr: f16 activations, no term)
  const float* qhn;                  // imma: ||q_v||_2 of the int8 head rows
  const float* hsc;                  // imma: s_w per head row (= head_scale)
  const float* hn_in;                // imma + outliers: ||h|| without the exactly-handled channels (else hnorm)
  const __nv_bfloat16* head;         // int8: exact bf16 head [VP][H] for candidates
  const __nv_bfloat16* h;            // int8: head input rows [n*7][H]
  __nv_bfloat16* state;              // [2][n][256]
  __nv_bfloat16* latent;             // [n][256]
  float* lat2;                       // [7][n] sum of latent^2 (atomic)
  unsigned long long* keys;          // [7][n] argmax keys
  unsigned long long* seedkeys;      // [n*7][MK_NCH] per-chunk best of (base + fold_b)
  const int8_t* fold_q;              // [VP][256] int8 rows of fold_w (row-major), scale fold_s, error norm efold
  const float* fold_s;
  const float* efold;
  const float* lrV;                  // [LRK][256] orthonormal rows (fp32)
  const __nv_bfloat16* lrU;          // [VP][LRK] bf16(fold_w V^T)
  const float* lrP;                  // [VP] ||P r_v|| (r = fold_w - U V), rounded up
  const float* lrQ;                  // [VP] ||(I - P) r_v||, rounded up
  int* drafts;                       // [n][7]
  int* nsurv;                        // [7][n] survivor counts (diagnostic)
  int* dbg;                          // optional [3]: rows passing stage 1, 1b, 2 (diagnostic)
  float* proj;                       // [n][LRK + 4]: p = V lat, ||p||, ||lat - V^T p||, sl, ||lat - sl lq|| (markov_zref)
  int8_t* latq;                      // [n][256] lq = rint(lat / sl), sl = max|lat| / 127 (markov_zref)
  const float* qnorm;                // [VP] ||q_v||_2 of the int8 fold rows (rounded up)
  int ksteps;                        // drafts to produce (1..7): the first k drafts equal a 7-step run's (exact)
  int vlim;                          // rows considered: VU (full vocabulary) or a shortlist prefix (first rows)
};

// exact config B base logit of row (hr, v) for the int8 variant: bf16(fp32 dot(h, head_v)) (warp-cooperative,
// 16 B per lane per load)
__device__ __forceinline__ float exact_base_warp(const Markov& mk, int hr, int v, int lane) {
  const uint4* hp = (const uint4*)(mk.h + (size_t)hr * H);
  const uint4* wp = (const uint4*)(mk.head + (size_t)v * H);
  float s = 0.f;
#pragma unroll
  for (int it = 0; it < H / 256; ++it) {
    uint4 a = hp[lane + it * 32], b = wp[lane + it * 32];
    const __nv_bfloat162* a2 = (const __nv_bfloat162*)&a;
    const __nv_bfloat162* b2 = (const __nv_bfloat162*)&b;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float2 x = __bfloat1622float2(a2[e]), y = __bfloat1622float2(b2[e]);
      s += x.x * y.x + x.y * y.y;
    }
  }
  return rb(warp_sum(s));
}
template <bool INT8>
__device__ __forceinline__ float base_val(const Markov& mk, int hr, int v, int lane) {
  if constexpr (INT8) return exact_base_warp(mk, hr, v, lane);
  else return bf2f(mk.B[(size_t)hr * VP + v]);
}
// exact z of (request r, step i, row v) given the latent in smem (warp-cooperative)
template <bool INT8>
__device__ __forceinline__ float exact_z_warp(const Markov& mk, int r, int i, int v, const float* lat, int lane) {
  const uint4 fv = ((const uint4*)(mk.fold_w + (size_t)v * RANK))[lane];   // 8 bf16 per lane
  const __nv_bfloat162* f2 = (const __nv_bfloat162*)&fv;
  float s = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    float2 b = __bfloat1622float2(f2[e]);
    s += lat[lane * 8 + 2 * e] * b.x + lat[lane * 8 + 2 * e + 1] * b.y;
  }
  s = warp_sum(s);
  float bias = rb(s + bf2f(mk.fold_b[v]));          // addmm(fold_b, latent, fold_w.T) -> bf16
  float B = base_val<INT8>(mk, r * 7 + i, v, lane);
  return softcap_z(rb(B + bias));                   // add_softcap
}

constexpr int LRK = 16;
// seeds: per head row and 2048-row chunk, argmax over v of (base + fold_b) (only used to get a good lower bound)
constexpr int MK_NCH = (VU + 2047) / 2048;  // 98
template <bool INT8>
__global__ void markov_seed(Markov mk) {
  int hr = blockIdx.y;
  int v0 = blockIdx.x * 2048;
  unsigned long long best = 0;
  for (int v = v0 + threadIdx.x; v < min(mk.vlim, v0 + 2048); v += blockDim.x) {
    float b = INT8 ? mk.Bq[(size_t)hr * VP + v] : bf2f(mk.B[(size_t)hr * VP + v]);
    float c = b + bf2f(mk.fold_b[v]);
    unsigned long long k = ((unsigned long long)__float_as_uint(c) ^ (c < 0 ? 0xFFFFFFFFull : 0x80000000ull)) << 32 |
                           (0xFFFFFFFFu - (uint32_t)v);
    best = k > best ? k : best;
  }
  for (int o = 16; o; o >>= 1) {
    unsigned long long x = __shfl_xor_sync(0xffffffffu, best, o);
    best = x > best ? x : best;
  }
  if ((threadIdx.x & 31) == 0) atomicMax(mk.seedkeys + (size_t)hr * MK_NCH + blockIdx.x, best);
}
__device__ __forceinline__ float key_z(unsigned long long k) {
  uint32_t o = (uint32_t)(k >> 32) & 0xFFFFu;
  uint32_t u = (o & 0x8000u) ? (o & 0x7FFFu) : (~o & 0xFFFFu);
  return __uint_as_float(u << 16);
}
// lower bound of step i: exact z of the 98 chunk seeds of each session; the best seeds the step's key.
// grid (ceil(98 / 8), n), 256 threads (warp per seed)
template <bool INT8>
__global__ void markov_zref(Markov mk, int i) {
  __shared__ float lat[RANK];
  const int r = blockIdx.y, hr = r * 7 + i, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int k = threadIdx.x; k < RANK; k += blockDim.x) lat[k] = bf2f(mk.latent[r * RANK + k]);
  __syncthreads();
  if (blockIdx.x == 0) {  // rank-16 projection of the latent, once per (session, step)
    __shared__ float pjs[LRK];
    for (int j = warp; j < LRK; j += 8) {
      float a = 0.f;
#pragma unroll
      for (int k = lane; k < RANK; k += 32) a += mk.lrV[j * RANK + k] * lat[k];
      a = warp_sum(a);
      if (lane == 0) pjs[j] = a;
    }
    __syncthreads();
    if (warp == 0) {
      float q2 = 0.f;
#pragma unroll
      for (int k = lane; k < RANK; k += 32) {
        float rec = 0.f;
#pragma unroll
        for (int j = 0; j < LRK; ++j) rec += pjs[j] * mk.lrV[j * RANK + k];
        float d = lat[k] - rec;
        q2 += d * d;
      }
      q2 = warp_sum(q2);
      float pn = 0.f;
#pragma unroll
      for (int j = 0; j < LRK; ++j) pn += pjs[j] * pjs[j];
      if (lane < LRK) mk.proj[r * (LRK + 4) + lane] = pjs[lane];
      if (lane == 0) {
        mk.proj[r * (LRK + 4) + LRK] = sqrtf(pn) * (1.0f + 1e-4f) + 1e-5f;
        mk.proj[r * (LRK + 4) + LRK + 1] = sqrtf(q2) * (1.0f + 1e-4f) + 1e-5f;
      }
    }
    if (warp == 1) {  // int8 latent for the dp4a stage: lq = rint(lat / sl), error norm ||lat - sl lq|| (fp32, slack)
      float mx = 0.f;
      for (int k = lane; k < RANK; k += 32) mx = fmaxf(mx, fabsf(lat[k]));
      mx = warp_max(mx);
      float sl = mx > 0.f ? mx / 127.f : 1.f;
      float e2 = 0.f;
      for (int k = lane; k < RANK; k += 32) {
        int qv = max(-127, min(127, __float2int_rn(lat[k] / sl)));
        mk.latq[r * RANK + k] = (int8_t)qv;
        float d = lat[k] - sl * (float)qv;
        e2 += d * d;
      }
      e2 = warp_sum(e2);
      if (lane == 0) {
        mk.proj[r * (LRK + 4) + LRK + 2] = sl;
        mk.proj[r * (LRK + 4) + LRK + 3] = sqrtf(e2) * (1.0f + 1e-4f) + 1e-6f;
      }
    }
  }
  int c = blockIdx.x * 8 + warp;
  if (c >= MK_NCH || c * 2048 >= mk.vlim) return;
  int v = key_index(mk.seedkeys[(size_t)hr * MK_NCH + c]);
  float z = exact_z_warp<INT8>(mk, r, i, v, lat, lane);
  if (lane == 0) atomicMax(mk.keys + i * mk.n + r, argkey(z, v));
}

// recurrence of step i: raw = ctx + [state, w1[prev]] . w_state^T (fp32), markov_gate -> state', latent, ||latent||^2
// grid 32 blocks x 256 threads: warp q handles rows (q, q+256, q+512) for all n sessions.
__global__ void markov_recur(Markov mk, int i) {
  __shared__ float inp[8][2 * RANK];
  const int n = mk.n;
  const __nv_bfloat16* st_in = mk.state + (size_t)(i & 1) * n * RANK;
  __nv_bfloat16* st_out = mk.state + (size_t)((i + 1) & 1) * n * RANK;
  for (int e = threadIdx.x; e < n * 2 * RANK; e += blockDim.x) {
    int r = e / (2 * RANK), k = e % (2 * RANK);
    float v;
    if (k < RANK) v = i == 0 ? 0.f : bf2f(st_in[r * RANK + k]);
    else {
      int prev = i == 0 ? mk.anchor[r] : key_index(mk.keys[(i - 1) * n + r]);
      v = bf2f(mk.w1[(size_t)prev * RANK + (k - RANK)]);
    }
    inp[r][k] = v;
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int q = blockIdx.x * 8 + warp;  // 0..255
  float raw[3][8];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const __nv_bfloat162* w = (const __nv_bfloat162*)(mk.w_state + (size_t)(q + c * RANK) * 2 * RANK);
    float wv[16];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      float2 f = __bfloat1622float2(w[lane + u * 32]);
      wv[2 * u] = f.x; wv[2 * u + 1] = f.y;
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (r < n) {
        float s = 0.f;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          int k = 2 * (lane + u * 32);
          s += wv[2 * u] * inp[r][k] + wv[2 * u + 1] * inp[r][k + 1];
        }
        raw[c][r] = warp_sum(s);
      }
    }
  }
  if (lane < n) {
    int r = lane;
    float rr[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      float v = raw[c][0];
#pragma unroll
      for (int rr2 = 1; rr2 < 8; ++rr2) if (rr2 == r) v = raw[c][rr2];
      rr[c] = v + mk.ctx[((size_t)r * 7 + i) * 768 + q + c * RANK];
    }
    float gate = rb(rr[0]), cand = rb(rr[1]), out = rb(rr[2]);
    float s = i == 0 ? 0.f : bf2f(st_in[r * RANK + q]);
    gate = rb(__fdiv_rn(1.0f, 1.0f + expf(-gate)));
    cand = rb(tanhf(cand));
    float keep = rb(gate * s);
    float upd = rb(rb(1.0f - gate) * cand);
    st_out[r * RANK + q] = __float2bfloat16_rn(keep + upd);
    float lat = bf2f(__float2bfloat16_rn(tanhf(out)));
    mk.latent[r * RANK + q] = __float2bfloat16_rn(lat);
    atomicAdd(mk.lat2 + i * n + r, lat * lat);
  }
}

// argmax of step i over the unpadded vocabulary with candidate pruning. grid (chunks of 2048, n), 256 threads.
// Lower bound zref = the step's key so far (exact z of the best chunk seed, markov_zref). Per row v:
//   stage 1: s_v <= B_up + fold_b + ||lat|| ||fold_w_v||                (Cauchy-Schwarz, no reads but norms)
//   stage 2: s_v <= B_up + fold_b + s8_v (lat . q_v) + ||lat|| e_v       (int8 fold row, 256 B, rigorous error)
//   rows whose bound z is not above zref - 2 ulp are skipped; the rest are evaluated exactly (exact_z_warp).
// B_up = B (exact variant) or Bq + ||h|| e_head_v + slack (int8 variant). Rounding slack covers config B's bf16
// rounding of bias and of base + bias (each <= 2^-8 relative) and fp32 summation order.
constexpr int MK_CHUNK = 2048;
template <bool INT8>
__global__ void __launch_bounds__(256) markov_argmax(Markov mk, int i) {
  __shared__ float lat[RANK];
  __shared__ float pj[LRK];
  __shared__ int surv[MK_CHUNK];
  __shared__ int c1v[MK_CHUNK];
  __shared__ float c1b[MK_CHUNK];
  __shared__ int c2v[MK_CHUNK];
  __shared__ float c2b[MK_CHUNK];
  __shared__ int nsv, n1, n2;
  __shared__ unsigned long long bestk;
  const int r = blockIdx.y, n = mk.n, hr = r * 7 + i;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int k = threadIdx.x; k < RANK; k += blockDim.x) lat[k] = bf2f(mk.latent[r * RANK + k]);
  if (threadIdx.x == 0) { nsv = 0; bestk = 0; n1 = 0; n2 = 0; }
  __syncthreads();
  __shared__ __align__(16) int8_t lqs[RANK];
  if (threadIdx.x < LRK) pj[threadIdx.x] = mk.proj[r * (LRK + 4) + threadIdx.x];
  if (threadIdx.x < RANK / 4) ((int*)lqs)[threadIdx.x] = ((const int*)(mk.latq + r * RANK))[threadIdx.x];
  __syncthreads();
  const float pn = mk.proj[r * (LRK + 4) + LRK];
  const float perp = mk.proj[r * (LRK + 4) + LRK + 1];
  const float sl = mk.proj[r * (LRK + 4) + LRK + 2];
  const float lerr = mk.proj[r * (LRK + 4) + LRK + 3];
  const float L = sqrtf(mk.lat2[i * n + r]) * (1.0f + 1e-5f) + 1e-6f;
  const float zref = key_z(*(volatile unsigned long long*)(mk.keys + i * n + r));
  // (two ulps: z is evaluated with the same rounding steps as config B; fp32 errors < 1 bf16 ulp)
  const float thr = zref - fmaxf(fabsf(zref) * 0.015625f, 1e-30f);
  const float hn = INT8 ? mk.hnorm[hr] : 0.f;
  // s-domain threshold: g(x) = rb(30 tanh(rb(x) / 30)) is monotone, so g(ub) <= thr whenever
  // ub < 30 atanh(thr / 30) - delta (delta covers the bf16 rounding of x, <= 2^-8 relative, and fp32 error);
  // thr itself sits two bf16 ulps under zref, so a skipped row has z < zref.
  float s_lim;
  {
    float tq = thr / SOFTCAP;
    if (tq >= 1.f) s_lim = INFINITY;
    else if (tq <= -1.f) s_lim = -INFINITY;
    else {
      float sl = SOFTCAP * atanhf(tq);
      s_lim = sl - fabsf(sl) * 0.0078125f - 1e-3f;
    }
  }
  // pass 1 (all rows of the chunk, 8 per thread): Cauchy-Schwarz bound; passing rows -> c1 list (shared)
  const int vbase = blockIdx.x * MK_CHUNK + threadIdx.x * 8;
  if (vbase < mk.vlim) {
    float Bu[8], Ba[8], fb[8], lr[8];
    {
      uint4 fbv = *(const uint4*)(mk.fold_b + vbase);
      const __nv_bfloat16* fbp = (const __nv_bfloat16*)&fbv;
      float4 r0 = *(const float4*)(mk.rn + vbase), r1 = *(const float4*)(mk.rn + vbase + 4);
      float rr[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) { fb[e] = bf2f(fbp[e]); lr[e] = L * rr[e]; }
    }
    if constexpr (INT8) {
      const float* bqp = mk.Bq + (size_t)hr * VP + vbase;
      float4 q0 = *(const float4*)bqp, q1 = *(const float4*)(bqp + 4);
      float4 e0 = *(const float4*)(mk.enorm + vbase), e1 = *(const float4*)(mk.enorm + vbase + 4);
      float4 h0 = *(const float4*)(mk.hrn + vbase), h1 = *(const float4*)(mk.hrn + vbase + 4);
      float bq[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
      float en[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
      float hr8[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        // int8 head error bound (Cauchy-Schwarz on the exact error row) + fp32 summation slack of both dots
        // (gamma_2304 < 1.4e-4 relative to sum |h_k w_k| <= ||h|| ||w||, doubled for truncating tensor-core adds; the measured v3 build used 3e-4) + f16 conversion of h (< 1e-5 here)
        float er = hn * (en[e] * (1.0f + 1e-3f) + 6e-4f * hr8[e]) + 1e-4f;
        Bu[e] = bq[e] + er;
        Ba[e] = fabsf(bq[e]) + er;
      }
    } else {
      uint4 bv = *(const uint4*)(mk.B + (size_t)hr * VP + vbase);
      const __nv_bfloat16* bp = (const __nv_bfloat16*)&bv;
#pragma unroll
      for (int e = 0; e < 8; ++e) { Bu[e] = bf2f(bp[e]); Ba[e] = fabsf(Bu[e]); }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float slack = (2.f * Ba[e] + 2.f * fabsf(fb[e]) + 2.f * lr[e]) * 0.0078125f + 1e-4f;
      if (vbase + e < mk.vlim && Bu[e] + fb[e] + lr[e] + slack >= s_lim) {
        int p = atomicAdd(&n1, 1);
        c1v[p] = vbase + e;
        c1b[p] = Bu[e] + fb[e] + slack + 1e-4f * lr[e];
      }
    }
  }
  __syncthreads();
  // pass 2: rank-16 part + residual bounds (|r.lat| <= ||Pr|| ||p|| + ||(I-P)r|| ||lat - V^T p||) -> c2 list
  for (int q = threadIdx.x; q < n1; q += blockDim.x) {
    const int v = c1v[q];
    const uint4* up = (const uint4*)(mk.lrU + (size_t)v * LRK);
    uint4 u0 = up[0], u1 = up[1];
    const __nv_bfloat162* ub = (const __nv_bfloat162*)&u0;
    const __nv_bfloat162* ub2 = (const __nv_bfloat162*)&u1;
    float dA = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 a2 = __bfloat1622float2(ub[j]), b2 = __bfloat1622float2(ub2[j]);
      dA += a2.x * pj[2 * j] + a2.y * pj[2 * j + 1] + b2.x * pj[8 + 2 * j] + b2.y * pj[8 + 2 * j + 1];
    }
    if (c1b[q] + dA + mk.lrP[v] * pn + mk.lrQ[v] * perp >= s_lim) {
      int p = atomicAdd(&n2, 1);
      c2v[p] = v; c2b[p] = c1b[q];
    }
  }
  __syncthreads();
  // pass 3 (thread per candidate): int8 fold row . int8 latent with dp4a (exact int32), bounded by
  //   |lat . f_v - sl s8 (lq . q_v)| <= ||lat|| e_v + s8 ||q_v|| ||lat - sl lq||   (Cauchy-Schwarz, both exact)
  for (int q = threadIdx.x; q < n2; q += blockDim.x) {
    const int v = c2v[q];
    const uint4* qp = (const uint4*)(mk.fold_q + (size_t)v * RANK);
    const int4* lp = (const int4*)lqs;
    int acc = 0;
#pragma unroll
    for (int cc = 0; cc < RANK / 16; ++cc) {
      uint4 w = qp[cc];
      int4 l = lp[cc];
      acc = __dp4a((int)w.x, l.x, acc);
      acc = __dp4a((int)w.y, l.y, acc);
      acc = __dp4a((int)w.z, l.z, acc);
      acc = __dp4a((int)w.w, l.w, acc);
    }
    const float s8 = mk.fold_s[v];
    const float est = (float)acc * s8 * sl;
    const float err = L * mk.efold[v] * (1.0f + 1e-3f) + s8 * mk.qnorm[v] * lerr * (1.0f + 1e-3f) + 1e-5f * fabsf(est);
    if (c2b[q] + est + err >= s_lim) {
      int p = atomicAdd(&nsv, 1);
      surv[p] = v;
    }
  }
  __syncthreads();
  const int ns = nsv;
  if (mk.dbg && threadIdx.x == 0) { atomicAdd(mk.dbg, n1); atomicAdd(mk.dbg + 1, n2); atomicAdd(mk.dbg + 2, ns); }
  unsigned long long best = 0;
  for (int s2 = warp; s2 < ns; s2 += 8) {
    int v = surv[s2];
    float z = exact_z_warp<INT8>(mk, r, i, v, lat, lane);
    unsigned long long k = argkey(z, v);
    best = k > best ? k : best;
  }
  if (lane == 0 && best) atomicMax(&bestk, best);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (bestk) atomicMax(mk.keys + i * n + r, bestk);
    if (mk.nsurv) atomicAdd(mk.nsurv + i * n + r, ns);
  }
}
// markov_argmax2: same bounds and result as markov_argmax, flatter: every row of the chunk gets the rank-16 bound
// directly (no Cauchy-Schwarz pre-filter, so no per-thread serial loop), then the dp4a int8 stage on the compacted
// candidates, then exact evaluation. 24 KB of shared memory -> one wave of CTAs.
template <bool INT8>
__global__ void __launch_bounds__(256) markov_argmax2(Markov mk, int i) {
  __shared__ float lat[RANK];
  __shared__ float pj[LRK];
  __shared__ __align__(16) int8_t lqs[RANK];
  __shared__ int c2v[MK_CHUNK];
  __shared__ float c2b[MK_CHUNK];
  __shared__ int surv[MK_CHUNK / 2];
  __shared__ int nsv, n2;
  __shared__ unsigned long long bestk;
  const int r = blockIdx.y, n = mk.n, hr = r * 7 + i;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  lat[threadIdx.x] = bf2f(mk.latent[r * RANK + threadIdx.x]);  // blockDim == RANK == 256
  if (threadIdx.x < LRK) pj[threadIdx.x] = mk.proj[r * (LRK + 4) + threadIdx.x];
  if (threadIdx.x < RANK / 4) ((int*)lqs)[threadIdx.x] = ((const int*)(mk.latq + r * RANK))[threadIdx.x];
  if (threadIdx.x == 0) { nsv = 0; n2 = 0; bestk = 0; }
  __syncthreads();
  const float pn = mk.proj[r * (LRK + 4) + LRK];
  const float perp = mk.proj[r * (LRK + 4) + LRK + 1];
  const float sl = mk.proj[r * (LRK + 4) + LRK + 2];
  const float lerr = mk.proj[r * (LRK + 4) + LRK + 3];
  const float L = sqrtf(mk.lat2[i * n + r]) * (1.0f + 1e-5f) + 1e-6f;
  const float zref = key_z(*(volatile unsigned long long*)(mk.keys + i * n + r));
  const float thr = zref - fmaxf(fabsf(zref) * 0.015625f, 1e-30f);
  float s_lim;
  {
    float tq = thr / SOFTCAP;
    if (tq >= 1.f) s_lim = INFINITY;
    else if (tq <= -1.f) s_lim = -INFINITY;
    else { float slv = SOFTCAP * atanhf(tq); s_lim = slv - fabsf(slv) * 0.0078125f - 1e-3f; }
  }
  const float hn = INT8 ? mk.hnorm[hr] : 0.f;
  const int vbase = blockIdx.x * MK_CHUNK + threadIdx.x * 8;
  if (vbase < mk.vlim) {
    float Bu[8], Ba[8], fb[8], rn8[8], P8[8], Q8[8];
    {
      uint4 fbv = *(const uint4*)(mk.fold_b + vbase);
      const __nv_bfloat16* fbp = (const __nv_bfloat16*)&fbv;
      float4 r0 = *(const float4*)(mk.rn + vbase), r1 = *(const float4*)(mk.rn + vbase + 4);
      float4 p0 = *(const float4*)(mk.lrP + vbase), p1 = *(const float4*)(mk.lrP + vbase + 4);
      float4 q0 = *(const float4*)(mk.lrQ + vbase), q1 = *(const float4*)(mk.lrQ + vbase + 4);
      float rr[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
      float pp[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
      float qq[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) { fb[e] = bf2f(fbp[e]); rn8[e] = rr[e]; P8[e] = pp[e]; Q8[e] = qq[e]; }
    }
    if constexpr (INT8) {
      const float* bqp = mk.Bq + (size_t)hr * VP + vbase;
      float4 q0 = *(const float4*)bqp, q1 = *(const float4*)(bqp + 4);
      float4 e0 = *(const float4*)(mk.enorm + vbase), e1 = *(const float4*)(mk.enorm + vbase + 4);
      float4 h0 = *(const float4*)(mk.hrn + vbase), h1 = *(const float4*)(mk.hrn + vbase + 4);
      float bq[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
      float en[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
      float hr8[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float er = hn * (en[e] * (1.0f + 1e-3f) + 6e-4f * hr8[e]) + 1e-4f;
        Bu[e] = bq[e] + er;
        Ba[e] = fabsf(bq[e]) + er;
      }
    } else {
      uint4 bv = *(const uint4*)(mk.B + (size_t)hr * VP + vbase);
      const __nv_bfloat16* bp = (const __nv_bfloat16*)&bv;
#pragma unroll
      for (int e = 0; e < 8; ++e) { Bu[e] = bf2f(bp[e]); Ba[e] = fabsf(Bu[e]); }
    }
    const uint4* up = (const uint4*)(mk.lrU + (size_t)vbase * LRK);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      uint4 u0 = up[2 * e], u1 = up[2 * e + 1];
      const __nv_bfloat162* ub = (const __nv_bfloat162*)&u0;
      const __nv_bfloat162* ub2 = (const __nv_bfloat162*)&u1;
      float dA = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 a2 = __bfloat1622float2(ub[j]), b2 = __bfloat1622float2(ub2[j]);
        dA += a2.x * pj[2 * j] + a2.y * pj[2 * j + 1] + b2.x * pj[8 + 2 * j] + b2.y * pj[8 + 2 * j + 1];
      }
      const float lr = L * rn8[e];
      const float slack = (2.f * Ba[e] + 2.f * fabsf(fb[e]) + 2.f * lr) * 0.0078125f + 1e-4f;
      const float base1 = Bu[e] + fb[e] + slack + 1e-4f * lr;
      if (vbase + e < mk.vlim && base1 + dA + P8[e] * pn + Q8[e] * perp >= s_lim) {
        int p = atomicAdd(&n2, 1);
        c2v[p] = vbase + e;
        c2b[p] = base1;
      }
    }
  }
  __syncthreads();
  for (int q = threadIdx.x; q < n2; q += blockDim.x) {
    const int v = c2v[q];
    const uint4* qp = (const uint4*)(mk.fold_q + (size_t)v * RANK);
    const int4* lp = (const int4*)lqs;
    int acc = 0;
#pragma unroll
    for (int cc = 0; cc < RANK / 16; ++cc) {
      uint4 w = qp[cc];
      int4 l = lp[cc];
      acc = __dp4a((int)w.x, l.x, acc);
      acc = __dp4a((int)w.y, l.y, acc);
      acc = __dp4a((int)w.z, l.z, acc);
      acc = __dp4a((int)w.w, l.w, acc);
    }
    const float s8 = mk.fold_s[v];
    const float est = (float)acc * s8 * sl;
    const float err = L * mk.efold[v] * (1.0f + 1e-3f) + s8 * mk.qnorm[v] * lerr * (1.0f + 1e-3f) + 1e-5f * fabsf(est);
    if (c2b[q] + est + err >= s_lim) {
      int p = atomicAdd(&nsv, 1);
      if (p < MK_CHUNK / 2) surv[p] = v;
    }
  }
  __syncthreads();
  const int ns = min(nsv, MK_CHUNK / 2);
  if (mk.dbg && threadIdx.x == 0) { atomicAdd(mk.dbg + 1, n2); atomicAdd(mk.dbg + 2, ns); if (nsv > MK_CHUNK / 2) atomicAdd(mk.dbg, 1); }
  unsigned long long best = 0;
  for (int s2 = warp; s2 < ns; s2 += 8) {
    int v = surv[s2];
    float z = exact_z_warp<INT8>(mk, r, i, v, lat, lane);
    unsigned long long k = argkey(z, v);
    best = k > best ? k : best;
  }
  if (lane == 0 && best) atomicMax(&bestk, best);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (bestk) atomicMax(mk.keys + i * n + r, bestk);
    if (mk.nsurv) atomicAdd(mk.nsurv + i * n + r, ns);
  }
}
struct ArgSmem {
  float lat[RANK];
  float pj[LRK];
  int8_t lqs[RANK];
  int c2v[MK_CHUNK];
  float c2b[MK_CHUNK];
  int surv[MK_CHUNK / 2];
  int nsv, n2;
  unsigned long long bestk;
};
template <bool INT8>
__device__ __forceinline__ void argmax_task(const Markov& mk, int i, int chunk, int r, ArgSmem& S) {
  float* lat = S.lat; float* pj = S.pj; int8_t* lqs = S.lqs; int* c2v = S.c2v; float* c2b = S.c2b; int* surv = S.surv;
  int& nsv = S.nsv; int& n2 = S.n2; unsigned long long& bestk = S.bestk;
  const int n = mk.n, hr = r * 7 + i;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  lat[threadIdx.x] = bf2f(mk.latent[r * RANK + threadIdx.x]);  // blockDim == RANK == 256
  if (threadIdx.x < LRK) pj[threadIdx.x] = mk.proj[r * (LRK + 4) + threadIdx.x];
  if (threadIdx.x < RANK / 4) ((int*)lqs)[threadIdx.x] = ((const int*)(mk.latq + r * RANK))[threadIdx.x];
  if (threadIdx.x == 0) { nsv = 0; n2 = 0; bestk = 0; }
  __syncthreads();
  const float pn = mk.proj[r * (LRK + 4) + LRK];
  const float perp = mk.proj[r * (LRK + 4) + LRK + 1];
  const float sl = mk.proj[r * (LRK + 4) + LRK + 2];
  const float lerr = mk.proj[r * (LRK + 4) + LRK + 3];
  const float L = sqrtf(mk.lat2[i * n + r]) * (1.0f + 1e-5f) + 1e-6f;
  const float zref = key_z(*(volatile unsigned long long*)(mk.keys + i * n + r));
  const float thr = zref - fmaxf(fabsf(zref) * 0.015625f, 1e-30f);
  float s_lim;
  {
    float tq = thr / SOFTCAP;
    if (tq >= 1.f) s_lim = INFINITY;
    else if (tq <= -1.f) s_lim = -INFINITY;
    else { float slv = SOFTCAP * atanhf(tq); s_lim = slv - fabsf(slv) * 0.0078125f - 1e-3f; }
  }
  const float hn = INT8 ? mk.hnorm[hr] : 0.f;
  const int vbase = chunk * MK_CHUNK + threadIdx.x * 8;
  if (vbase < mk.vlim) {
    float Bu[8], Ba[8], fb[8], rn8[8], P8[8], Q8[8];
    {
      uint4 fbv = *(const uint4*)(mk.fold_b + vbase);
      const __nv_bfloat16* fbp = (const __nv_bfloat16*)&fbv;
      float4 r0 = *(const float4*)(mk.rn + vbase), r1 = *(const float4*)(mk.rn + vbase + 4);
      float4 p0 = *(const float4*)(mk.lrP + vbase), p1 = *(const float4*)(mk.lrP + vbase + 4);
      float4 q0 = *(const float4*)(mk.lrQ + vbase), q1 = *(const float4*)(mk.lrQ + vbase + 4);
      float rr[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
      float pp[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
      float qq[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) { fb[e] = bf2f(fbp[e]); rn8[e] = rr[e]; P8[e] = pp[e]; Q8[e] = qq[e]; }
    }
    if constexpr (INT8) {
      const float* bqp = mk.Bq + (size_t)hr * VP + vbase;
      float4 q0 = *(const float4*)bqp, q1 = *(const float4*)(bqp + 4);
      float4 e0 = *(const float4*)(mk.enorm + vbase), e1 = *(const float4*)(mk.enorm + vbase + 4);
      float4 h0 = *(const float4*)(mk.hrn + vbase), h1 = *(const float4*)(mk.hrn + vbase + 4);
      float bq[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
      float en[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
      float hr8[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float er = hn * (en[e] * (1.0f + 1e-3f) + 6e-4f * hr8[e]) + 1e-4f;
        Bu[e] = bq[e] + er;
        Ba[e] = fabsf(bq[e]) + er;
      }
    } else {
      uint4 bv = *(const uint4*)(mk.B + (size_t)hr * VP + vbase);
      const __nv_bfloat16* bp = (const __nv_bfloat16*)&bv;
#pragma unroll
      for (int e = 0; e < 8; ++e) { Bu[e] = bf2f(bp[e]); Ba[e] = fabsf(Bu[e]); }
    }
    const uint4* up = (const uint4*)(mk.lrU + (size_t)vbase * LRK);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      uint4 u0 = up[2 * e], u1 = up[2 * e + 1];
      const __nv_bfloat162* ub = (const __nv_bfloat162*)&u0;
      const __nv_bfloat162* ub2 = (const __nv_bfloat162*)&u1;
      float dA = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 a2 = __bfloat1622float2(ub[j]), b2 = __bfloat1622float2(ub2[j]);
        dA += a2.x * pj[2 * j] + a2.y * pj[2 * j + 1] + b2.x * pj[8 + 2 * j] + b2.y * pj[8 + 2 * j + 1];
      }
      const float lr = L * rn8[e];
      const float slack = (2.f * Ba[e] + 2.f * fabsf(fb[e]) + 2.f * lr) * 0.0078125f + 1e-4f;
      const float base1 = Bu[e] + fb[e] + slack + 1e-4f * lr;
      if (vbase + e < mk.vlim && base1 + dA + P8[e] * pn + Q8[e] * perp >= s_lim) {
        int p = atomicAdd(&n2, 1);
        c2v[p] = vbase + e;
        c2b[p] = base1;
      }
    }
  }
  __syncthreads();
  for (int q = threadIdx.x; q < n2; q += blockDim.x) {
    const int v = c2v[q];
    const uint4* qp = (const uint4*)(mk.fold_q + (size_t)v * RANK);
    const int4* lp = (const int4*)lqs;
    int acc = 0;
#pragma unroll
    for (int cc = 0; cc < RANK / 16; ++cc) {
      uint4 w = qp[cc];
      int4 l = lp[cc];
      acc = __dp4a((int)w.x, l.x, acc);
      acc = __dp4a((int)w.y, l.y, acc);
      acc = __dp4a((int)w.z, l.z, acc);
      acc = __dp4a((int)w.w, l.w, acc);
    }
    const float s8 = mk.fold_s[v];
    const float est = (float)acc * s8 * sl;
    const float err = L * mk.efold[v] * (1.0f + 1e-3f) + s8 * mk.qnorm[v] * lerr * (1.0f + 1e-3f) + 1e-5f * fabsf(est);
    if (c2b[q] + est + err >= s_lim) {
      int p = atomicAdd(&nsv, 1);
      if (p < MK_CHUNK / 2) surv[p] = v;
    }
  }
  __syncthreads();
  const int ns = min(nsv, MK_CHUNK / 2);
  if (mk.dbg && threadIdx.x == 0) { atomicAdd(mk.dbg + 1, n2); atomicAdd(mk.dbg + 2, ns); if (nsv > MK_CHUNK / 2) atomicAdd(mk.dbg, 1); }
  unsigned long long best = 0;
  for (int s2 = warp; s2 < ns; s2 += 8) {
    int v = surv[s2];
    float z = exact_z_warp<INT8>(mk, r, i, v, lat, lane);
    unsigned long long k = argkey(z, v);
    best = k > best ? k : best;
  }
  if (lane == 0 && best) atomicMax(&bestk, best);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (bestk) atomicMax(mk.keys + i * n + r, bestk);
    if (mk.nsurv) atomicAdd(mk.nsurv + i * n + r, ns);
  }
  __syncthreads();  // smem reuse by the next task of this CTA
}

// ---- argmax v3: variable chunks (512 rows below id 16,384 where the frequent tokens and most candidates sit,
// 2,048 above), per row: CS bound (8-12 B read) -> compacted list -> rank-16 bound (thread per candidate) -> list
// -> dp4a int8 bound -> survivors -> exact. Same bounds as markov_argmax2, less shared/L2 traffic per row.
constexpr int MK3_HOT = 16384, MK3_HOTC = 512, MK3_NHOT = MK3_HOT / MK3_HOTC;
constexpr int MK3_NCH = MK3_NHOT + (VU - MK3_HOT + MK_CHUNK - 1) / MK_CHUNK;  // 32 + 90 = 122
__device__ __forceinline__ void chunk3(int c, int& v0, int& vlen) {
  if (c < MK3_NHOT) { v0 = c * MK3_HOTC; vlen = MK3_HOTC; }
  else { v0 = MK3_HOT + (c - MK3_NHOT) * MK_CHUNK; vlen = min(MK_CHUNK, VU - v0); }
}
struct ArgSmem3 {
  float lat[RANK];
  float pj[LRK];
  int8_t lqs[RANK];
  int c1v[MK_CHUNK];
  float c1b[MK_CHUNK];
  int c2v[MK_CHUNK];
  float c2b[MK_CHUNK];
  int surv[MK_CHUNK];
  int n1, n2, nsv;
  unsigned long long bestk;
};
template <bool INT8>
__device__ __forceinline__ void argmax_chunk(const Markov& mk, int i, int r, int c, ArgSmem3& S) {
  const int n = mk.n, hr = r * 7 + i;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int v0, vlen;
  chunk3(c, v0, vlen);
  S.lat[threadIdx.x] = bf2f(mk.latent[r * RANK + threadIdx.x]);
  if (threadIdx.x < LRK) S.pj[threadIdx.x] = mk.proj[r * (LRK + 4) + threadIdx.x];
  if (threadIdx.x < RANK / 4) ((int*)S.lqs)[threadIdx.x] = ((const int*)(mk.latq + r * RANK))[threadIdx.x];
  if (threadIdx.x == 0) { S.n1 = 0; S.n2 = 0; S.nsv = 0; S.bestk = 0; }
  __syncthreads();
  const float pn = mk.proj[r * (LRK + 4) + LRK];
  const float perp = mk.proj[r * (LRK + 4) + LRK + 1];
  const float sl = mk.proj[r * (LRK + 4) + LRK + 2];
  const float lerr = mk.proj[r * (LRK + 4) + LRK + 3];
  const float L = sqrtf(mk.lat2[i * n + r]) * (1.0f + 1e-5f) + 1e-6f;
  const float zref = key_z(*(volatile unsigned long long*)(mk.keys + i * n + r));
  const float thr = zref - fmaxf(fabsf(zref) * 0.015625f, 1e-30f);
  float s_lim;
  {
    float tq = thr / SOFTCAP;
    if (tq >= 1.f) s_lim = INFINITY;
    else if (tq <= -1.f) s_lim = -INFINITY;
    else { float slv = SOFTCAP * atanhf(tq); s_lim = slv - fabsf(slv) * 0.0078125f - 1e-3f; }
  }
  const float hn = INT8 ? mk.hnorm[hr] : 0.f;
  // pass 1: CS bound, coalesced (v = v0 + e * 256 + tid)
  for (int e = 0; e * 256 < vlen; ++e) {
    const int v = v0 + e * 256 + threadIdx.x;
    if (v >= v0 + vlen || v >= mk.vlim) continue;
    float fb = bf2f(mk.fold_b[v]);
    float lr = L * mk.rn[v];
    float Bu, Ba;
    if constexpr (INT8) {
      float bq = mk.Bq[(size_t)hr * VP + v];
      // |h.w - Bq| <= ||h|| ||w - s_w q|| + s_w ||q|| ||h - s_h hq|| (imma, exact int32 dot) or
      //             ||h|| ||w - s_w q|| + fp32 slack (f16-activation path)
      float er = mk.hdn ? (mk.hn_in ? mk.hn_in[hr] : hn) * mk.enorm[v] * (1.0f + 1e-3f) +
                              mk.hsc[v] * mk.qhn[v] * mk.hdn[hr] * (1.0f + 1e-3f) + 1e-6f * hn * mk.hrn[v] +
                              2e-6f * hn * mk.hrn[v] + 1e-4f
                        : hn * (mk.enorm[v] * (1.0f + 1e-3f) + 6e-4f * mk.hrn[v]) + 1e-4f;
      Bu = bq + er; Ba = fabsf(bq) + er;
    } else {
      Bu = bf2f(mk.B[(size_t)hr * VP + v]); Ba = fabsf(Bu);
    }
    const float slack = (2.f * Ba + 2.f * fabsf(fb) + 2.f * lr) * 0.0078125f + 1e-4f;
    if (Bu + fb + lr + slack >= s_lim) {
      int p = atomicAdd(&S.n1, 1);
      S.c1v[p] = v;
      S.c1b[p] = Bu + fb + slack + 1e-4f * lr;
    }
  }
  __syncthreads();
  // pass 2: rank-16 bound
  for (int q = threadIdx.x; q < S.n1; q += blockDim.x) {
    const int v = S.c1v[q];
    const uint4* up = (const uint4*)(mk.lrU + (size_t)v * LRK);
    uint4 u0 = up[0], u1 = up[1];
    const __nv_bfloat162* ub = (const __nv_bfloat162*)&u0;
    const __nv_bfloat162* ub2 = (const __nv_bfloat162*)&u1;
    float dA = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 a2 = __bfloat1622float2(ub[j]), b2 = __bfloat1622float2(ub2[j]);
      dA += a2.x * S.pj[2 * j] + a2.y * S.pj[2 * j + 1] + b2.x * S.pj[8 + 2 * j] + b2.y * S.pj[8 + 2 * j + 1];
    }
    if (S.c1b[q] + dA + mk.lrP[v] * pn + mk.lrQ[v] * perp >= s_lim) {
      int p = atomicAdd(&S.n2, 1);
      S.c2v[p] = v;
      S.c2b[p] = S.c1b[q];
    }
  }
  __syncthreads();
  // pass 3: dp4a int8 fold bound
  for (int q = threadIdx.x; q < S.n2; q += blockDim.x) {
    const int v = S.c2v[q];
    const uint4* qp = (const uint4*)(mk.fold_q + (size_t)v * RANK);
    const int4* lp = (const int4*)S.lqs;
    int acc = 0;
#pragma unroll
    for (int cc = 0; cc < RANK / 16; ++cc) {
      uint4 w = qp[cc];
      int4 l = lp[cc];
      acc = __dp4a((int)w.x, l.x, acc);
      acc = __dp4a((int)w.y, l.y, acc);
      acc = __dp4a((int)w.z, l.z, acc);
      acc = __dp4a((int)w.w, l.w, acc);
    }
    const float s8 = mk.fold_s[v];
    const float est = (float)acc * s8 * sl;
    const float err = L * mk.efold[v] * (1.0f + 1e-3f) + s8 * mk.qnorm[v] * lerr * (1.0f + 1e-3f) + 1e-5f * fabsf(est);
    if (S.c2b[q] + est + err >= s_lim) {
      int p = atomicAdd(&S.nsv, 1);
      S.surv[p] = v;
    }
  }
  __syncthreads();
  const int ns = S.nsv;
  if (mk.dbg && threadIdx.x == 0) {
    atomicAdd(mk.dbg, S.n1); atomicAdd(mk.dbg + 1, S.n2); atomicAdd(mk.dbg + 2, ns);

  }
  unsigned long long best = 0;
  for (int s2 = warp; s2 < ns; s2 += 8) {
    int v = S.surv[s2];
    float z = exact_z_warp<INT8>(mk, r, i, v, S.lat, lane);
    unsigned long long k = argkey(z, v);
    best = k > best ? k : best;
  }
  if (lane == 0 && best) atomicMax(&S.bestk, best);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (S.bestk) atomicMax(mk.keys + i * n + r, S.bestk);
    if (mk.nsurv) atomicAdd(mk.nsurv + i * n + r, ns);
  }
  __syncthreads();
}
template <bool INT8>
__global__ void __launch_bounds__(256) markov_argmax3(Markov mk, int i) {
  __shared__ __align__(16) ArgSmem3 S;
  argmax_chunk<INT8>(mk, i, blockIdx.y, blockIdx.x, S);
}

// markov_coop: the whole 7-step Markov loop in ONE cooperative persistent launch (grid = co-resident CTAs, >= 32).
// Per step: phase R (recurrence, CTAs 0..31, warp per output triple) | grid sync | phase Z (exact z of the 98 chunk
// seeds per session, warp per seed; the last n CTAs compute the rank-16 projection and the int8 latent) | grid sync |
// phase A (argmax tasks (chunk, session) round-robin over CTAs) | grid sync. Same arithmetic as the 3-kernel path.
template <bool INT8>
__global__ void __launch_bounds__(256) markov_coop(Markov mk) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  __shared__ __align__(16) unsigned char smraw[sizeof(ArgSmem3) > 8 * 2 * RANK * 4 ? sizeof(ArgSmem3) : 8 * 2 * RANK * 4];
  const int n = mk.n, G = gridDim.x, b = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int i = 0; i < mk.ksteps; ++i) {
    // ---------------- phase R
    if (b < 32) {
      float (*inp)[2 * RANK] = (float (*)[2 * RANK])smraw;
      const __nv_bfloat16* st_in = mk.state + (size_t)(i & 1) * n * RANK;
      __nv_bfloat16* st_out = mk.state + (size_t)((i + 1) & 1) * n * RANK;
      for (int e = threadIdx.x; e < n * 2 * RANK; e += blockDim.x) {
        int r = e / (2 * RANK), k = e % (2 * RANK);
        float v;
        if (k < RANK) v = i == 0 ? 0.f : bf2f(st_in[r * RANK + k]);
        else {
          int prev = i == 0 ? mk.anchor[r] : key_index(mk.keys[(i - 1) * n + r]);
          v = bf2f(mk.w1[(size_t)prev * RANK + (k - RANK)]);
        }
        inp[r][k] = v;
      }
      __syncthreads();
      const int q = b * 8 + warp;
      float raw[3][8];
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        const __nv_bfloat162* w = (const __nv_bfloat162*)(mk.w_state + (size_t)(q + c * RANK) * 2 * RANK);
        float wv[16];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          float2 f = __bfloat1622float2(w[lane + u * 32]);
          wv[2 * u] = f.x; wv[2 * u + 1] = f.y;
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          if (r < n) {
            float s = 0.f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              int k = 2 * (lane + u * 32);
              s += wv[2 * u] * inp[r][k] + wv[2 * u + 1] * inp[r][k + 1];
            }
            raw[c][r] = warp_sum(s);
          }
        }
      }
      if (lane < n) {
        int r = lane;
        float rr[3];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          float v = raw[c][0];
#pragma unroll
          for (int rr2 = 1; rr2 < 8; ++rr2) if (rr2 == r) v = raw[c][rr2];
          rr[c] = v + mk.ctx[((size_t)r * 7 + i) * 768 + q + c * RANK];
        }
        float gate = rb(rr[0]), cand = rb(rr[1]), out = rb(rr[2]);
        float s = i == 0 ? 0.f : bf2f(st_in[r * RANK + q]);
        gate = rb(__fdiv_rn(1.0f, 1.0f + expf(-gate)));
        cand = rb(tanhf(cand));
        float keep = rb(gate * s);
        float upd = rb(rb(1.0f - gate) * cand);
        st_out[r * RANK + q] = __float2bfloat16_rn(keep + upd);
        float lat = bf2f(__float2bfloat16_rn(tanhf(out)));
        mk.latent[r * RANK + q] = __float2bfloat16_rn(lat);
        atomicAdd(mk.lat2 + i * n + r, lat * lat);
      }
    }
    grid.sync();
    // ---------------- phase Z
    if (b >= G - n) {  // projection + int8 latent of session r
      const int r = G - 1 - b;
      float* lat = (float*)smraw;
      float* pjs = lat + RANK;
      lat[threadIdx.x] = bf2f(mk.latent[r * RANK + threadIdx.x]);
      __syncthreads();
      for (int j = warp; j < LRK; j += 8) {
        float a = 0.f;
#pragma unroll
        for (int k = lane; k < RANK; k += 32) a += mk.lrV[j * RANK + k] * lat[k];
        a = warp_sum(a);
        if (lane == 0) pjs[j] = a;
      }
      __syncthreads();
      if (warp == 0) {
        float q2 = 0.f;
#pragma unroll
        for (int k = lane; k < RANK; k += 32) {
          float rec = 0.f;
#pragma unroll
          for (int j = 0; j < LRK; ++j) rec += pjs[j] * mk.lrV[j * RANK + k];
          float d = lat[k] - rec;
          q2 += d * d;
        }
        q2 = warp_sum(q2);
        float pn = 0.f;
#pragma unroll
        for (int j = 0; j < LRK; ++j) pn += pjs[j] * pjs[j];
        if (lane < LRK) mk.proj[r * (LRK + 4) + lane] = pjs[lane];
        if (lane == 0) {
          mk.proj[r * (LRK + 4) + LRK] = sqrtf(pn) * (1.0f + 1e-4f) + 1e-5f;
          mk.proj[r * (LRK + 4) + LRK + 1] = sqrtf(q2) * (1.0f + 1e-4f) + 1e-5f;
        }
      }
      if (warp == 1) {
        float mx = 0.f;
        for (int k = lane; k < RANK; k += 32) mx = fmaxf(mx, fabsf(lat[k]));
        mx = warp_max(mx);
        float sl = mx > 0.f ? mx / 127.f : 1.f;
        float e2 = 0.f;
        for (int k = lane; k < RANK; k += 32) {
          int qv = max(-127, min(127, __float2int_rn(lat[k] / sl)));
          mk.latq[r * RANK + k] = (int8_t)qv;
          float d = lat[k] - sl * (float)qv;
          e2 += d * d;
        }
        e2 = warp_sum(e2);
        if (lane == 0) {
          mk.proj[r * (LRK + 4) + LRK + 2] = sl;
          mk.proj[r * (LRK + 4) + LRK + 3] = sqrtf(e2) * (1.0f + 1e-4f) + 1e-6f;
        }
      }
    } else {  // seed tasks: global warp gw -> (session, chunk)
      float* lat2r = (float*)smraw;  // two sessions' latents
      const int gw0 = b * 8, gw = gw0 + warp;
      const int ra = gw0 / MK_NCH, rbb = min(n - 1, (gw0 + 7) / MK_NCH);
      if (ra < n) {
        lat2r[threadIdx.x] = bf2f(mk.latent[ra * RANK + threadIdx.x]);
        lat2r[RANK + threadIdx.x] = bf2f(mk.latent[rbb * RANK + threadIdx.x]);
      }
      __syncthreads();
      if (gw < n * MK_NCH) {
        const int r = gw / MK_NCH, c = gw % MK_NCH;
        if (c * 2048 < mk.vlim) {
          const int hr = r * 7 + i;
          int v = key_index(mk.seedkeys[(size_t)hr * MK_NCH + c]);
          float z = exact_z_warp<INT8>(mk, r, i, v, r == ra ? lat2r : lat2r + RANK, lane);
          if (lane == 0) atomicMax(mk.keys + i * n + r, argkey(z, v));
        }
      }
    }
    grid.sync();
    // ---------------- phase A
    {
      ArgSmem3& S = *(ArgSmem3*)smraw;
      for (int t = b; t < MK3_NCH * n; t += G) argmax_chunk<INT8>(mk, i, t / MK3_NCH, t % MK3_NCH, S);
    }
    grid.sync();
  }
  if (b == 0 && threadIdx.x < n * 7) {
    int r = threadIdx.x / 7, ii = threadIdx.x % 7;
    mk.drafts[r * 7 + ii] = ii < mk.ksteps ? key_index(mk.keys[ii * n + r]) : -1;
  }
}
__global__ void markov_finish(Markov mk) {
  int e = threadIdx.x;
  if (e < mk.n * 7) {
    int r = e / 7, i = e % 7;
    mk.drafts[r * 7 + i] = i < mk.ksteps ? key_index(mk.keys[i * mk.n + r]) : -1;
  }
}

}  // namespace s2f
