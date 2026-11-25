// decode_dense candidate kernels (own code).
//
// sk2: the engine's skinny projection kernel (engine/native/src/proj_small.cuh, same Args / Job ABI) with the data
// movement pipelined and nothing else changed: every output element is still ONE warp's chain of
// mma.m16n8k16.row.col.f32.bf16.bf16.f32 over k in ascending 16-wide steps, with the weights as the A fragment
// (loaded from global with the same per-lane addresses) and the rows as the B fragment (same shared-memory layout),
// so its bits equal proj_small's (checked by dd_bench --mode bits). Changes: the weight fragments of chunk i + 1 are
// loaded into registers while chunk i is multiplied, and the row chunk i + 1 is copied to shared memory with cp.async
// into a second buffer (one __syncthreads per chunk pair instead of a load-sync-compute-sync round trip).
#pragma once
#include "../../native/src/proj_small.cuh"

namespace dd {

using projs::Args;
using projs::bf16;
using projs::Job;
using projs::KC;
using projs::WARPS;
using projs::XSTR;

__device__ __forceinline__ void cp16(void* s, const void* g) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int NT, int NW>
__global__ void __launch_bounds__(NW * 32) sk2_kernel(Args a) {
  int jb = 0;
  while (jb + 1 < a.njobs && (int)blockIdx.x >= a.j[jb + 1].cta0) ++jb;
  const Job J = a.j[jb];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int f0 = ((blockIdx.x - J.cta0) * NW + warp) * 16;
  const bool active = f0 < J.N;
  const bool r0ok = f0 + g < J.N, r1ok = f0 + g + 8 < J.N;
  __shared__ __align__(16) bf16 xs[2][NT * 8 * XSTR];
  const int M = a.M;
  // rows >= M stay zero in both buffers (never written by the copies)
  for (int i = threadIdx.x; i < NT * 8 * (KC / 8); i += NW * 32) {
    const int r = i / (KC / 8), c8 = i % (KC / 8);
    if (r >= M) {
      *reinterpret_cast<uint4*>(&xs[0][r * XSTR + c8 * 8]) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(&xs[1][r * XSTR + c8 * 8]) = make_uint4(0, 0, 0, 0);
    }
  }
  auto stage = [&](int buf, int k0) {
    for (int i = threadIdx.x; i < M * (KC / 8); i += NW * 32) {
      const int r = i / (KC / 8), c8 = i % (KC / 8);
      cp16(&xs[buf][r * XSTR + c8 * 8], J.x + (size_t)r * J.ldx + k0 + c8 * 8);
    }
    cp_commit();
  };
  const bf16* w0 = J.W + (size_t)(r0ok ? f0 + g : 0) * J.K + 2 * t;
  const bf16* w1 = J.W + (size_t)(r1ok ? f0 + g + 8 : 0) * J.K + 2 * t;
  auto loadA = [&](uint32_t (&A)[KC / 16][4], int k0) {
#pragma unroll
    for (int s = 0; s < KC / 16; ++s) {
      const int k = k0 + s * 16;
      A[s][0] = r0ok ? projs::ld32(w0 + k) : 0u;
      A[s][1] = r1ok ? projs::ld32(w1 + k) : 0u;
      A[s][2] = r0ok ? projs::ld32(w0 + k + 8) : 0u;
      A[s][3] = r1ok ? projs::ld32(w1 + k + 8) : 0u;
    }
  };
  float c[NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
  uint32_t A[2][KC / 16][4];
  const int nch = J.K / KC;
  stage(0, 0);
  if (active) loadA(A[0], 0);
#pragma unroll 1
  for (int ch = 0; ch < nch; ch += 2) {
    // even chunk ch in buffer 0 / A[0]; prefetch ch + 1 into buffer 1 / A[1]
    if (ch + 1 < nch) {
      stage(1, (ch + 1) * KC);
      if (active) loadA(A[1], (ch + 1) * KC);
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    if (active) {
#pragma unroll
      for (int s = 0; s < KC / 16; ++s)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const bf16* xr = &xs[0][(nt * 8 + g) * XSTR + s * 16 + 2 * t];
          projs::mma16816(c[nt], A[0][s][0], A[0][s][1], A[0][s][2], A[0][s][3], *reinterpret_cast<const uint32_t*>(xr),
                          *reinterpret_cast<const uint32_t*>(xr + 8));
        }
    }
    __syncthreads();   // buffer 0 free
    if (ch + 1 >= nch) break;
    // odd chunk ch + 1 in buffer 1 / A[1]; prefetch ch + 2 into buffer 0 / A[0]
    if (ch + 2 < nch) {
      stage(0, (ch + 2) * KC);
      if (active) loadA(A[0], (ch + 2) * KC);
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    if (active) {
#pragma unroll
      for (int s = 0; s < KC / 16; ++s)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const bf16* xr = &xs[1][(nt * 8 + g) * XSTR + s * 16 + 2 * t];
          projs::mma16816(c[nt], A[1][s][0], A[1][s][1], A[1][s][2], A[1][s][3], *reinterpret_cast<const uint32_t*>(xr),
                          *reinterpret_cast<const uint32_t*>(xr + 8));
        }
    }
    __syncthreads();   // buffer 1 free
  }
  if (!active) return;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int m0 = nt * 8 + 2 * t, m1 = m0 + 1;
    const int fa = f0 + g, fb = f0 + g + 8;
    if (m0 < M && r0ok) J.out[(size_t)m0 * J.ldo + fa] = __float2bfloat16_rn(c[nt][0]);
    if (m1 < M && r0ok) J.out[(size_t)m1 * J.ldo + fa] = __float2bfloat16_rn(c[nt][1]);
    if (m0 < M && r1ok) J.out[(size_t)m0 * J.ldo + fb] = __float2bfloat16_rn(c[nt][2]);
    if (m1 < M && r1ok) J.out[(size_t)m1 * J.ldo + fb] = __float2bfloat16_rn(c[nt][3]);
  }
}

// split copy of a fused projection row [q 3072 | gate 3072 | r 384 | sink 24 | k 1024 (| k_dyn 32) | v 1024 (| v_dyn 32)]
// (row stride ld) into the engine's separate buffers (q, gate, r, sink, k, kd, v, vd; each with row stride = its width)
struct ProjSplit {
  const bf16* f;
  int ld;
  bf16* out[8];
  int dyn;
};
__global__ void __launch_bounds__(256) proj_split_kernel(ProjSplit a) {
  const int t = blockIdx.x;
  const int w[8] = {3072, 3072, 384, 24, 1024, a.dyn ? 32 : 0, 1024, a.dyn ? 32 : 0};
  const uint4* src = reinterpret_cast<const uint4*>(a.f + (size_t)t * a.ld);
  int off = 0;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    uint4* dst = reinterpret_cast<uint4*>(a.out[s] + (size_t)t * w[s]);
    for (int i = threadIdx.x; i < w[s] / 8; i += blockDim.x) dst[i] = src[off / 8 + i];
    off += w[s];
  }
}
inline void launch_split(const ProjSplit& a, int M, cudaStream_t s) { proj_split_kernel<<<M, 256, 0, s>>>(a); }

template <int NW>
inline void launch_sk2(Args& a, cudaStream_t s) {
  int c = 0;
  for (int i = 0; i < a.njobs; ++i) { a.j[i].cta0 = c; c += (a.j[i].N + NW * 16 - 1) / (NW * 16); }
  switch ((a.M + 7) / 8) {
    case 1: sk2_kernel<1, NW><<<c, NW * 32, 0, s>>>(a); break;
    case 2: sk2_kernel<2, NW><<<c, NW * 32, 0, s>>>(a); break;
    case 3: sk2_kernel<3, NW><<<c, NW * 32, 0, s>>>(a); break;
    case 4: sk2_kernel<4, NW><<<c, NW * 32, 0, s>>>(a); break;
    case 5: sk2_kernel<5, NW><<<c, NW * 32, 0, s>>>(a); break;
    case 6: sk2_kernel<6, NW><<<c, NW * 32, 0, s>>>(a); break;
    case 7: sk2_kernel<7, NW><<<c, NW * 32, 0, s>>>(a); break;
    default: sk2_kernel<8, NW><<<c, NW * 32, 0, s>>>(a); break;
  }
}

}  // namespace dd
