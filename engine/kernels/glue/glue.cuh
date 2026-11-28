// inkling-cmp-engine native runtime: glue kernels, version 1 (own code; clarity over speed).
//
// Each kernel follows the S1 reference's op order (engine/ref/model.py, which transcribes config B's rounding points:
// docs/engine/configB-numerics.md). fp32 arithmetic whose order matters is written with __fmul_rn / __fadd_rn /
// __fdiv_rn / __fmaf_rn so nvcc cannot contract it; reductions (sum of squares, dots) use a fixed block tree (bits are
// a function of the row only: batch-invariant), which is not torch's order (the reference is not bit-reproducible
// there either; per-op tables give the difference).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace glue {

using bf16 = __nv_bfloat16;
constexpr int H = 2304, NH = 24, KVH = 8, HD = 128, DREL = 16, REL_EXT = 1024, NEXP = 128, TOPK = 8;

__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ bf16 r16(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rr16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// fixed-order block sum (blockDim.x multiple of 32, <= 1024); all threads get the result
// Row loops of the 256-thread row kernels (embed_norm, norm_row, attn_res, mg_in, attn_res_norm; every launch uses
// 256 threads): thread j visits i = j, j + 256, ... (9 steps over H = 2304), the order the reference-matched sums need.
// A compile-time stride lets the loop unroll, so a thread issues its 9 independent loads at once (latency-bound at the
// few rows of a decode call); arithmetic and order unchanged, bits unchanged.
constexpr int kRowThreads = 256;
static_assert(H % kRowThreads == 0, "row kernels: H multiple of 256");
#define GLUE_FOR_H(i) _Pragma("unroll") for (int i##_k = 0, i = threadIdx.x; i##_k < H / kRowThreads; ++i##_k, i += kRowThreads)

// Per value: the warp's xor tree, then the warps' partials added in warp order from 0.f. The loops over the NV values
// are innermost, so the NV independent chains interleave (the warp loop's trip count is only known at run time; with it
// innermost the NV chains of 8 dependent shared-memory adds ran one after another); each value's order is unchanged.
template <int NV>
__device__ __forceinline__ void block_sum(float (&v)[NV], float* sh /* >= 32 * NV */) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
#pragma unroll
    for (int i = 0; i < NV; ++i) v[i] = __fadd_rn(v[i], __shfl_xor_sync(0xffffffffu, v[i], o));
  const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
  if ((threadIdx.x & 31) == 0)
#pragma unroll
    for (int i = 0; i < NV; ++i) sh[i * 32 + w] = v[i];
  __syncthreads();
  float s[NV];
#pragma unroll
  for (int i = 0; i < NV; ++i) s[i] = 0.f;
  for (int j = 0; j < nw; ++j)
#pragma unroll
    for (int i = 0; i < NV; ++i) s[i] = __fadd_rn(s[i], sh[i * 32 + j]);
#pragma unroll
  for (int i = 0; i < NV; ++i) v[i] = s[i];
  __syncthreads();
}

// ---------------------------------------------------------------------------------------------- RMSNorm family
// y = x * rsqrt(sum(x^2) / N + eps) [* w], ref rms_norm (torch mean = sum / N; rsqrt ~ rsqrt.approx)
__device__ __forceinline__ float rstd_of(float ss, int n, float eps) {
  return rsqrtf(__fadd_rn(__fdiv_rn(ss, (float)n), eps));
}

// out bf16 [T, H] = bf16(rms_norm(embed[ids], w))
__global__ void embed_norm_kernel(const int* ids, const bf16* emb, const float* w, bf16* out, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x;
  const bf16* x = emb + (size_t)ids[t] * H;
  float v[1] = {0.f};
  GLUE_FOR_H(i) { float a = ld(x + i); v[0] = __fadd_rn(v[0], __fmul_rn(a, a)); }
  block_sum<1>(v, sh);
  const float r = rstd_of(v[0], H, eps);
  GLUE_FOR_H(i) out[(size_t)t * H + i] = r16(__fmul_rn(__fmul_rn(ld(x + i), r), w[i]));
}

// out bf16 [T, ldo] = bf16(rms_norm(in [T, ldi] (fp32 or bf16), w)) (w may be null); rmsnorm_g: grid (T, G), source y
struct NormG {
  const float* in[4];
  const float* w[4];
  bf16* out[4];
};
// Register-staged row kernels: thread j owns elements j + 256 k (k < KH); every load of the row is issued up front
// and the second pass reuses the registers (no reload, no shared-memory row). Same arithmetic in the same order as the
// GLUE_FOR_H loops: bits unchanged.
constexpr int KH = H / kRowThreads;   // 9
template <class TI>
__device__ __forceinline__ void norm_row(const TI* x, const float* w, bf16* o, float eps, float* sh) {
  float xv[KH], wv[KH];
#pragma unroll
  for (int k = 0; k < KH; ++k) {
    const int i = threadIdx.x + kRowThreads * k;
    xv[k] = ld(x + i);
    wv[k] = w ? w[i] : 1.f;
  }
  float v[1] = {0.f};
#pragma unroll
  for (int k = 0; k < KH; ++k) v[0] = __fadd_rn(v[0], __fmul_rn(xv[k], xv[k]));
  block_sum<1>(v, sh);
  const float r = rstd_of(v[0], H, eps);
#pragma unroll
  for (int k = 0; k < KH; ++k) {
    float y = __fmul_rn(xv[k], r);
    if (w) y = __fmul_rn(y, wv[k]);
    o[threadIdx.x + kRowThreads * k] = r16(y);
  }
}
template <class TI>
__global__ void __launch_bounds__(256) rmsnorm_kernel(const TI* in, int ldi, const float* w, bf16* out, int ldo, float eps) {
  __shared__ float sh[32];
  norm_row(in + (size_t)blockIdx.x * ldi, w, out + (size_t)blockIdx.x * ldo, eps, sh);
}
__global__ void __launch_bounds__(256) rmsnorm_g_kernel(NormG a, int ldi, int ldo, float eps) {
  __shared__ float sh[32];
  const int t = blockIdx.x, y = blockIdx.y;
  norm_row(a.in[y] + (size_t)t * ldi, a.w[y], a.out[y] + (size_t)t * ldo, eps, sh);
}

// ---------------------------------------------------------------------------------------------- AttnRes
// ref attn_res: sources x_n = blocks (bf16) then last (fp32 or bf16); qs = q * fp32(H^-0.5);
// rstd_n = rsqrt(fma(sum x_n^2, fp32(1/H), eps)); logit = (x_n . qs) * rstd_n; p = exp(l - max); inv = 1 / sum p;
// p *= rstd; acc = fma(p_n, x_n, acc) for n in order; out = acc * inv. out fp32 [T, Q, H].
struct ResSrc {
  const bf16* blk[4];
  int nb;
  const float* last32;   // one of last32 / last16
  const bf16* last16;
  int ld_last;           // row stride of the last source
};
// G > 0 (production path of loop-gate visits): the same block then also writes the G message-gate inputs of the row,
// mg_in_kernel's arithmetic: gate y's hidden = this kernel's output for query (gi.qsel ? y : 0), its message =
// gi.msg + y * H (row stride gi.ldm); g + y * gi.gs gets bf16(rms(hidden)) || bf16(rms(message)). The sums run over
// the same thread / i mapping and the same block_sum tree as mg_in_kernel: bit-identical to attn_res + mg_in.
struct GateIn {
  const float* msg;
  int ldm;
  bf16* g;
  long long gs;
  int qsel;
};
// The sources and queries of row t loaded up front (thread j: elements j + 256 k) and the phase-1 sums (sum x_n^2,
// x_n . qs) in the GLUE_FOR_H loop's order; shared by attn_res_kernel and attn_res_norm_kernel.
template <int Q>
__device__ __forceinline__ void attn_res_load(const ResSrc& s, const float* queries, int t, int N, float (&xs)[5][KH],
                                              float (&v)[5 * (Q + 1)], float* sh) {
  const float qsc = (float)(1.0 / 48.0);       // fp32(2304^-0.5) = 1/48 exactly
  float qs[Q][KH];
#pragma unroll
  for (int k = 0; k < KH; ++k) {
    const int i = threadIdx.x + kRowThreads * k;
#pragma unroll
    for (int q = 0; q < Q; ++q) qs[q][k] = __fmul_rn(queries[q * H + i], qsc);
#pragma unroll
    for (int n = 0; n < 5; ++n) {
      float x = 0.f;
      if (n < s.nb) x = ld(s.blk[n] + (size_t)t * H + i);
      else if (n == s.nb) x = s.last32 ? s.last32[(size_t)t * s.ld_last + i] : ld(s.last16 + (size_t)t * s.ld_last + i);
      xs[n][k] = x;
    }
  }
#pragma unroll
  for (int i = 0; i < 5 * (Q + 1); ++i) v[i] = 0.f;
#pragma unroll
  for (int k = 0; k < KH; ++k)
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) {
        const float x = xs[n][k];
        v[n * (Q + 1)] = __fadd_rn(v[n * (Q + 1)], __fmul_rn(x, x));
#pragma unroll
        for (int q = 0; q < Q; ++q) v[n * (Q + 1) + 1 + q] = __fadd_rn(v[n * (Q + 1) + 1 + q], __fmul_rn(x, qs[q][k]));
      }
  block_sum<5 * (Q + 1)>(v, sh);
}

// Query split (decode calls): grid (T, NQ / Q), block y computes queries y * Q .. y * Q + Q - 1 of NQ = gridDim.y * Q
// (out [T][NQ][H]); with qsel its gate inputs are those of its own queries' gates. A query's sums, softmax and output
// do not depend on the other queries of the block (each value has its own reduction tree): bits equal to one block per
// row with all NQ queries.
template <int Q, int G = 0>
__global__ void __launch_bounds__(256) attn_res_kernel(ResSrc s, const float* queries, float* out, float eps, GateIn gi = {}) {
  __shared__ float sh[32 * (5 * (Q + 1) > 2 * G ? 5 * (Q + 1) : 2 * G)];
  const int t = blockIdx.x, N = s.nb + 1, q0 = blockIdx.y * Q, NQ = gridDim.y * Q;
  const int goff = gi.qsel ? q0 : 0;           // first gate of this block (qsel: gate y = query y)
  const float invH = (float)(1.0 / 2304.0);    // fp32(1/H)
  float xs[5][KH], v[5 * (Q + 1)];
  attn_res_load<Q>(s, queries + (size_t)q0 * H, t, N, xs, v, sh);
  float rstd[5], p[Q][5], inv[Q];
#pragma unroll
  for (int n = 0; n < 5; ++n) rstd[n] = n < N ? rsqrtf(__fmaf_rn(v[n * (Q + 1)], invH, eps)) : 0.f;
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    float mx = -INFINITY, l[5];
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) { l[n] = __fmul_rn(v[n * (Q + 1) + 1 + q], rstd[n]); mx = fmaxf(mx, l[n]); }
    float sum = 0.f;
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) { p[q][n] = expf(__fadd_rn(l[n], -mx)); sum = __fadd_rn(sum, p[q][n]); }
    inv[q] = __fdiv_rn(1.0f, sum);
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) p[q][n] = __fmul_rn(p[q][n], rstd[n]);
  }
  float res[Q][KH];   // this thread's outputs (the values written to out), kept for the gate inputs
#pragma unroll
  for (int k = 0; k < KH; ++k) {
    const int i = threadIdx.x + kRowThreads * k;
    float acc[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) acc[q] = 0.f;
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) {
#pragma unroll
        for (int q = 0; q < Q; ++q) acc[q] = __fmaf_rn(p[q][n], xs[n][k], acc[q]);
      }
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      res[q][k] = __fmul_rn(acc[q], inv[q]);
      out[((size_t)t * NQ + q0 + q) * H + i] = res[q][k];
    }
  }
  if constexpr (G > 0) {
    float mv[G][KH];
#pragma unroll
    for (int k = 0; k < KH; ++k)
#pragma unroll
      for (int y = 0; y < G; ++y) mv[y][k] = gi.msg[(size_t)t * gi.ldm + (goff + y) * H + threadIdx.x + kRowThreads * k];
    float u[2 * G];
#pragma unroll
    for (int j = 0; j < 2 * G; ++j) u[j] = 0.f;
#pragma unroll
    for (int k = 0; k < KH; ++k) {
#pragma unroll
      for (int y = 0; y < G; ++y) {
        const float a = res[gi.qsel ? y : 0][k];
        const float b = mv[y][k];
        u[2 * y] = __fadd_rn(u[2 * y], __fmul_rn(a, a));
        u[2 * y + 1] = __fadd_rn(u[2 * y + 1], __fmul_rn(b, b));
      }
    }
    block_sum<2 * G>(u, sh);
    float r[2 * G];
#pragma unroll
    for (int j = 0; j < 2 * G; ++j) r[j] = rstd_of(u[j], H, eps);
#pragma unroll
    for (int k = 0; k < KH; ++k) {
      const int i = threadIdx.x + kRowThreads * k;
#pragma unroll
      for (int y = 0; y < G; ++y) {
        bf16* g = gi.g + (goff + y) * gi.gs + (size_t)t * 2 * H;
        g[i] = r16(__fmul_rn(res[gi.qsel ? y : 0][k], r[2 * y]));
        g[H + i] = r16(__fmul_rn(mv[y][k], r[2 * y + 1]));
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------- loop message gate
// g_in bf16 [T, 2H] = bf16(rms_norm(hidden)) || bf16(rms_norm(message))   (no weights)
// grid (T, G): gate y reads hid + y * hs, msg + y * ms and writes g + y * gs (strides in elements; 0 = shared)
__global__ void __launch_bounds__(256) mg_in_kernel(const float* hid, int ldh, const float* msg, int ldm, bf16* g, float eps, long long hs = 0,
                             long long ms = 0, long long gs = 0) {
  __shared__ float sh[64];
  const int t = blockIdx.x;
  hid += blockIdx.y * hs;
  msg += blockIdx.y * ms;
  g += blockIdx.y * gs;
  const float* a = hid + (size_t)t * ldh;
  const float* b = msg + (size_t)t * ldm;
  float av[KH], bv[KH];
#pragma unroll
  for (int k = 0; k < KH; ++k) { av[k] = a[threadIdx.x + kRowThreads * k]; bv[k] = b[threadIdx.x + kRowThreads * k]; }
  float v[2] = {0.f, 0.f};
#pragma unroll
  for (int k = 0; k < KH; ++k) {
    v[0] = __fadd_rn(v[0], __fmul_rn(av[k], av[k]));
    v[1] = __fadd_rn(v[1], __fmul_rn(bv[k], bv[k]));
  }
  block_sum<2>(v, sh);
  const float ra = rstd_of(v[0], H, eps), rb = rstd_of(v[1], H, eps);
#pragma unroll
  for (int k = 0; k < KH; ++k) {
    const int i = threadIdx.x + kRowThreads * k;
    g[(size_t)t * 2 * H + i] = r16(__fmul_rn(av[k], ra));
    g[(size_t)t * 2 * H + H + i] = r16(__fmul_rn(bv[k], rb));
  }
}
__global__ void bf16_to_f32_kernel(const bf16* a, float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = __bfloat162float(a[i]);
}
// out fp32 [T, ldo] = (1 - tanh(fp)) * hidden + tanh(gp) * message
__global__ void mg_out_kernel(const float* fp, const float* gp, const float* hid, int ldh, const float* msg, int ldm,
                              float* out, int ldo, int T) {
  const size_t n = (size_t)T * H;
  for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < n; k += (size_t)gridDim.x * blockDim.x) {
    const size_t t = k / H, i = k % H;
    const float f = tanhf(fp[k]), g = tanhf(gp[k]);
    out[t * ldo + i] = __fadd_rn(__fmul_rn(__fadd_rn(1.0f, -f), hid[t * ldh + i]), __fmul_rn(g, msg[t * ldm + i]));
  }
}

// ---------------------------------------------------------------------------------------------- q / rel post
// q = bf16(l2clip(qproj * q_scale, 22.63)); full: q = bf16(q * tau). rel likewise (16 dims, clip 8).
// l2clip: s = sum x^2; norm = s > 0 ? s * rsqrt(s) : 0; x * min(thr / (norm + 1e-6), 1).
__device__ __forceinline__ float tau_of(int pos) {
  float r = __fdiv_rn((float)(pos + 1), 8192.0f);
  r = fmaxf(r, 1.0f);
  return __fadd_rn(1.0f, __fmul_rn(0.1f, logf(r)));
}
// grid (T, 24), block 128: one head of q and (threads < 16) of rel
__global__ void qpost_kernel(const bf16* qproj, const float* q_scale, const bf16* rproj, const float* r_scale,
                             const int* row_pos, int full, bf16* q, bf16* rel) {
  __shared__ float sh[64];
  const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  float y = __fmul_rn(ld(qproj + ((size_t)t * NH + h) * HD + d), q_scale[h * HD + d]);
  float r = 0.f;
  if (d < DREL) r = __fmul_rn(ld(rproj + ((size_t)t * NH + h) * DREL + d), r_scale[h * DREL + d]);
  float v[2] = {__fmul_rn(y, y), __fmul_rn(r, r)};
  block_sum<2>(v, sh);
  auto clipf = [](float s, float thr) {
    const float norm = s > 0.f ? __fmul_rn(s, rsqrtf(s)) : 0.f;
    return fminf(__fdiv_rn(thr, __fadd_rn(norm, 1e-6f)), 1.0f);
  };
  float qo = rr16(__fmul_rn(y, clipf(v[0], 22.627416997969522f)));
  float ro = rr16(__fmul_rn(r, clipf(v[1], 8.0f)));
  if (full) {
    const float tau = tau_of(row_pos[t]);
    qo = rr16(__fmul_rn(qo, tau));
    ro = rr16(__fmul_rn(ro, tau));
  }
  q[((size_t)t * NH + h) * HD + d] = r16(qo);
  if (d < DREL) rel[((size_t)t * NH + h) * DREL + d] = r16(ro);
}
// rel_logits bf16 [T, 24, 1024] = bf16(rel [T, 24, 16] @ rel_proj [16, 1024]) (fp32 accumulation)
__global__ void rel_logits_kernel(const bf16* rel, const bf16* proj, bf16* out) {
  const int t = blockIdx.x, h = blockIdx.y;
  float r[DREL];
#pragma unroll
  for (int j = 0; j < DREL; ++j) r[j] = ld(rel + ((size_t)t * NH + h) * DREL + j);
  for (int d = threadIdx.x; d < REL_EXT; d += blockDim.x) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < DREL; ++j) acc = __fmaf_rn(r[j], ld(proj + j * REL_EXT + d), acc);
    out[((size_t)t * NH + h) * REL_EXT + d] = r16(acc);
  }
}
// ---------------------------------------------------------------------------------------------- short convs
// Row metadata of a forward call.
struct Rows {
  const int* pos;    // [T] absolute position
  const int* seg;    // [T] segment index
  const int* row0;   // [S] first row of the segment
  const int* slot;   // [S] session slot
};
// ext element j (0..3) for row t of channel c: (i + j < 3) ? state[i + j] : x[row0 + i + j - 3], i = t - row0
constexpr int SLD = 6656;   // conv state row stride (k 1024 | v 1024 | attn 2304 | mlp 2304)
__device__ __forceinline__ float ext_at(const bf16* x, int ldx, const bf16* st, int t, int i, int j, int c, int C) {
  const int e = i + j;
  (void)C;
  return e < 3 ? ld(st + (size_t)e * SLD + c) : ld(x + (size_t)(t - i + e - 3) * ldx + c);
}
// state of (slot, visit, kind): conv_state + ((slot * nvisits + visit) * 3) * 6656 + kind offset; C channels, rows 3
// static conv + residual: out = (w0 e0 + w1 e1 + w2 e2 + w3 e3 + x) * fp32(1/sqrt 5), unrounded (ref static_conv_ext)
__device__ __forceinline__ float static_conv(const float* w, const bf16* x, int ldx, const bf16* st, int t, int i, int c,
                                             int C) {
  float acc = __fmul_rn(w[c * 4 + 0], ext_at(x, ldx, st, t, i, 0, c, C));
#pragma unroll
  for (int j = 1; j < 4; ++j) acc = __fadd_rn(acc, __fmul_rn(w[c * 4 + j], ext_at(x, ldx, st, t, i, j, c, C)));
  return __fmul_rn(__fadd_rn(acc, ld(x + (size_t)t * ldx + c)), (float)(1.0 / 2.23606797749979));
}
// hidden conv (attn / mlp, C = H): y = bf16(conv). mode 0: out_bf = y. mode 1 (mlp): out_bf = bf16(partial32 + y).
// 2D grid (H / 256 = 9, T), block 256: zero 64-bit div, float4 weight load, bit-identical to original 1D grid-stride.
__global__ void __launch_bounds__(256) hidden_conv_kernel(const bf16* __restrict__ x, const float* __restrict__ w,
                                                          const bf16* __restrict__ state, long long st_slot_stride,
                                                          Rows R, int mode, const float* __restrict__ partial32,
                                                          bf16* __restrict__ out, int T) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  const int t = blockIdx.y;
  if (t >= T || c >= H) return;
  const int s = R.seg[t], i = t - R.row0[s];
  const bf16* st = state + (size_t)R.slot[s] * st_slot_stride;
  const float4 wv = *reinterpret_cast<const float4*>(w + (size_t)c * 4);
  const float x0 = ext_at(x, H, st, t, i, 0, c, H);
  const float x1 = ext_at(x, H, st, t, i, 1, c, H);
  const float x2 = ext_at(x, H, st, t, i, 2, c, H);
  const float x3 = ld(x + (size_t)t * H + c);
  float acc = __fmul_rn(wv.x, x0);
  acc = __fadd_rn(acc, __fmul_rn(wv.y, x1));
  acc = __fadd_rn(acc, __fmul_rn(wv.z, x2));
  acc = __fadd_rn(acc, __fmul_rn(wv.w, x3));
  const float y = rr16(__fmul_rn(__fadd_rn(acc, x3), (float)(1.0 / 2.23606797749979)));
  const size_t k = (size_t)t * H + c;
  out[k] = mode == 0 ? r16(y) : r16(__fadd_rn(partial32[k], y));
}
// key/value conv + post, grid (T, 8), block 128 (one kv head of k and v):
//   local: conv_k = bf16(static_conv(kproj)), dynamic (full): w_lag = static[lag] + dyn[t, head, lag], newest first,
//   lag 0 a product, lags 1..3 fma, (acc + x) * fp32(1/3); k = bf16(l2clip(conv_k, 22.63)); v = bf16(rms(conv_v)).
__global__ void kv_conv_kernel(const bf16* kx, const bf16* vx, const bf16* kst, const bf16* vst, long long st_slot_stride,
                               Rows R, int full, const float* kw, const float* vw, const bf16* kdyn, const bf16* vdyn,
                               bf16* kout, bf16* vout, float eps) {
  __shared__ float sh[64];
  const int t = blockIdx.x, g = blockIdx.y, d = threadIdx.x, c = g * HD + d;
  const int s = R.seg[t], r0 = R.row0[s], i = t - r0;
  const bf16* ks = kst + (size_t)R.slot[s] * st_slot_stride;
  const bf16* vs = vst + (size_t)R.slot[s] * st_slot_stride;
  float ck, cv;
  if (!full) {
    ck = static_conv(kw, kx, 1024, ks, t, i, c, 1024);
    cv = static_conv(vw, vx, 1024, vs, t, i, c, 1024);
  } else {
    auto dyn = [&](const float* sw, const bf16* x, const bf16* st, const bf16* dy) {
      float out = 0.f;
#pragma unroll
      for (int lag = 0; lag < 4; ++lag) {
        const float w = __fadd_rn(sw[lag * 1024 + c], ld(dy + (size_t)t * 32 + g * 4 + lag));
        const float xt = ext_at(x, 1024, st, t, i, 3 - lag, c, 1024);
        out = lag == 0 ? __fmul_rn(w, xt) : __fmaf_rn(w, xt, out);
      }
      return __fmul_rn(__fadd_rn(out, ld(x + (size_t)t * 1024 + c)), (float)(1.0 / 3.0));
    };
    ck = dyn(kw, kx, ks, kdyn);
    cv = dyn(vw, vx, vs, vdyn);
  }
  ck = rr16(ck);
  cv = rr16(cv);
  float v[2] = {__fmul_rn(ck, ck), __fmul_rn(cv, cv)};
  block_sum<2>(v, sh);
  const float norm = v[0] > 0.f ? __fmul_rn(v[0], rsqrtf(v[0])) : 0.f;
  const float f = fminf(__fdiv_rn(22.627416997969522f, __fadd_rn(norm, 1e-6f)), 1.0f);
  kout[(size_t)t * 1024 + c] = r16(__fmul_rn(ck, f));
  vout[(size_t)t * 1024 + c] = r16(__fmul_rn(cv, rstd_of(v[1], HD, eps)));
}
// conv state commit: new state = ext[c_s : c_s + 3] of the segment, for every (visit, kind) of the call.
// x of (visit, kind): convx + visit * convx_vstride + off * convx_kstride, ld C; kind offsets: 0, 1024, 2048, 4352.
struct CommitArgs {
  const bf16* convx;
  long long convx_kstride;   // rows per visit in convx (each kind stores convx_kstride * C elements)
  long long convx_vstride;   // elements between visits (convx_kstride * SLD)
  bf16* state;               // conv state base
  long long st_slot_stride, st_visit_stride;
  const int* row0;
  const int* slot;
  const int* cnt;            // [S] rows committed
  int nseg, nvisits;
};
__global__ void conv_commit_kernel(CommitArgs a) {
  const int v = blockIdx.y, s = blockIdx.z;
  const int kind = blockIdx.x / 9, part = blockIdx.x % 9;   // 9 x 256 >= 2304 channels
  const int C = kind < 2 ? 1024 : 2304;
  const int c = part * 256 + threadIdx.x;
  if (c >= C) return;
  const int off = kind == 0 ? 0 : kind == 1 ? 1024 : kind == 2 ? 2048 : 4352;
  bf16* st = a.state + (size_t)a.slot[s] * a.st_slot_stride + (size_t)v * a.st_visit_stride + off;
  const bf16* x = a.convx + (size_t)v * a.convx_vstride + (size_t)off * a.convx_kstride;
  const int cnt = a.cnt[s], r0 = a.row0[s];
  bf16 nv[3];
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    const int e = cnt + r;
    nv[r] = e < 3 ? st[(size_t)e * SLD + c] : x[(size_t)(r0 + e - 3) * C + c];
  }
#pragma unroll
  for (int r = 0; r < 3; ++r) st[(size_t)r * SLD + c] = nv[r];
}

// ---------------------------------------------------------------------------------------------- residual
// block start: blk = hidden (copy), partial32 = y; else partial32 = hidden + y (fp32, unrounded)
__global__ void partial_kernel(const bf16* hidden, const bf16* y, int block_start, bf16* blk, float* partial32, int T) {
  const size_t n = (size_t)T * H;
  for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < n; k += (size_t)gridDim.x * blockDim.x) {
    const float yy = ld(y + k);
    if (block_start) { blk[k] = hidden[k]; partial32[k] = yy; }
    else partial32[k] = __fadd_rn(ld(hidden + k), yy);
  }
}

// ---------------------------------------------------------------------------------------------- KV write
// k, v [T, 8, 128] -> head-major pages [page][8][1024][128] through a per-segment page table (pt row = segment)
__global__ void kv_write_kernel(const bf16* k, const bf16* v, bf16* kpool, bf16* vpool, const int* pt, int pt_stride,
                                Rows R) {
  const int t = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
  const int pos = R.pos[t], s = R.seg[t];
  const int page = pt[(size_t)s * pt_stride + (pos >> 10)];
  const size_t off = (((size_t)page * KVH + g) * 1024 + (pos & 1023)) * HD + d;
  kpool[off] = k[((size_t)t * KVH + g) * HD + d];
  vpool[off] = v[((size_t)t * KVH + g) * HD + d];
}
// E4M3 (FP8) KV write (global layers with --kv-fp8): inputs k, v are already bf16 -> exact IEEE E4M3FN RNE satfinite
__global__ void kv_write_fp8_kernel(const bf16* k, const bf16* v, uint8_t* kpool, uint8_t* vpool, const int* pt,
                                    int pt_stride, Rows R) {
  const int t = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
  const int pos = R.pos[t], s = R.seg[t];
  const int page = pt[(size_t)s * pt_stride + (pos >> 10)];
  const size_t off = (((size_t)page * KVH + g) * 1024 + (pos & 1023)) * HD + d;
  kpool[off] = f2e4m3(ld(k + ((size_t)t * KVH + g) * HD + d));
  vpool[off] = f2e4m3(ld(v + ((size_t)t * KVH + g) * HD + d));
}

// ---------------------------------------------------------------------------------------------- router
// logits[t, e] = (dot over k 0..1023) + (1024..2047) + (2048..2303) in fp32; keys = 1 / (1 + exp(-l)) + bias;
// top-8 by key (stable, ties to the lower expert); lp = min(l, 0) - log1p(exp(-|l|)); w = exp(lp - (max +
// log(sum8))) * 8 with sum8 = ((e0 + e2) + (e1 + e3)) + ((e4 + e6) + (e5 + e7)). Block = 128 threads (one expert).
__global__ void router_kernel(const bf16* x, const bf16* wr, const float* bias, int* idx, float* wts) {
  __shared__ float xs[H];
  __shared__ float lg[NEXP], key[NEXP];
  const int t = blockIdx.x, e = threadIdx.x;
  for (int i = threadIdx.x; i < H; i += blockDim.x) xs[i] = ld(x + (size_t)t * H + i);
  __syncthreads();
  const bf16* w = wr + (size_t)e * H;
  float tot = 0.f;
  for (int k0 = 0; k0 < H; k0 += 1024) {
    float part = 0.f;
    for (int k = k0; k < min(H, k0 + 1024); ++k) part = __fmaf_rn(xs[k], ld(w + k), part);
    tot = k0 == 0 ? part : __fadd_rn(tot, part);
  }
  lg[e] = tot;
  key[e] = __fadd_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-tot))), bias[e]);
  __syncthreads();
  if (e == 0) {
    int sel[TOPK];
    unsigned long long used0 = 0, used1 = 0;
    for (int j = 0; j < TOPK; ++j) {
      int best = -1;
      float bk = 0.f;
      for (int x2 = 0; x2 < NEXP; ++x2) {
        const bool u = x2 < 64 ? (used0 >> x2) & 1 : (used1 >> (x2 - 64)) & 1;
        if (u) continue;
        if (best < 0 || key[x2] > bk) { best = x2; bk = key[x2]; }
      }
      sel[j] = best;
      if (best < 64) used0 |= 1ull << best; else used1 |= 1ull << (best - 64);
    }
    float lp[TOPK], mx = -INFINITY;
    for (int j = 0; j < TOPK; ++j) {
      const float l = lg[sel[j]];
      lp[j] = __fadd_rn(fminf(l, 0.f), -log1pf(expf(-fabsf(l))));
      mx = fmaxf(mx, lp[j]);
    }
    float ex[TOPK];
    for (int j = 0; j < TOPK; ++j) ex[j] = expf(__fadd_rn(lp[j], -mx));
    const float s8 = __fadd_rn(__fadd_rn(__fadd_rn(ex[0], ex[2]), __fadd_rn(ex[1], ex[3])),
                               __fadd_rn(__fadd_rn(ex[4], ex[6]), __fadd_rn(ex[5], ex[7])));
    const float base = __fadd_rn(mx, logf(s8));
    for (int j = 0; j < TOPK; ++j) {
      idx[t * TOPK + j] = sel[j];
      wts[t * TOPK + j] = __fmul_rn(expf(__fadd_rn(lp[j], -base)), 8.0f);
    }
  }
}

// ---------------------------------------------------------------------------------------------- head
__global__ void gather_rows_bf16_kernel(const float* h, const int* rows, bf16* out) {
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < H; i += blockDim.x) out[(size_t)r * H + i] = r16(h[(size_t)rows[r] * H + i]);
}
// logits[r, :200058] = tanh(x / 30) * 30 in place (padded tail untouched); 2D grid (blocks, rows), float4 coalesced
__global__ void __launch_bounds__(256) softcap_kernel(float* lg, int rows, int ld, int vocab) {
  const int r = blockIdx.y;
  if (r >= rows) return;
  float* row = lg + (size_t)r * ld;
  const int c4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (c4 + 3 < vocab && (ld & 3) == 0) {
    float4 v = *reinterpret_cast<const float4*>(row + c4);
    v.x = __fmul_rn(tanhf(__fdiv_rn(v.x, 30.0f)), 30.0f);
    v.y = __fmul_rn(tanhf(__fdiv_rn(v.y, 30.0f)), 30.0f);
    v.z = __fmul_rn(tanhf(__fdiv_rn(v.z, 30.0f)), 30.0f);
    v.w = __fmul_rn(tanhf(__fdiv_rn(v.w, 30.0f)), 30.0f);
    *reinterpret_cast<float4*>(row + c4) = v;
  } else {
    for (int c = c4; c < vocab && c < c4 + 4; ++c) {
      row[c] = __fmul_rn(tanhf(__fdiv_rn(row[c], 30.0f)), 30.0f);
    }
  }
}


// ---------------------------------------------------------------------------------------------- fused-path helpers
// attention conv + residual: y = bf16(conv(wo)); block start: blk = hidden, partial = y; else partial = hidden + y
__global__ void attn_conv_partial_kernel(const bf16* x, const float* w, const bf16* state, long long st_slot_stride, Rows R,
                                         const bf16* hidden, int block_start, bf16* blk, bf16* y_out, float* partial32,
                                         int T) {
  const size_t n = (size_t)T * H;
  for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < n; k += (size_t)gridDim.x * blockDim.x) {
    const int t = (int)(k / H), c = (int)(k % H);
    const int s = R.seg[t], r0 = R.row0[s];
    const bf16* st = state + (size_t)R.slot[s] * st_slot_stride;
    const float yy = rr16(static_conv(w, x, H, st, t, t - r0, c, H));
    y_out[k] = r16(yy);
    if (block_start) { blk[k] = hidden[k]; partial32[k] = yy; }
    else partial32[k] = __fadd_rn(ld(hidden + k), yy);
  }
}


// router v3 selection: logits = (g0 + g1) + g2 of three fp32 GEMM outputs (K groups 0-1023, 1024-2047, 2048-2303,
// config B's group order); one warp per row: lane l holds experts l, l + 32, l + 64, l + 96; 8 rounds of a warp argmax
// on (key desc, expert asc) = the stable descending sort's first 8; weights as router_kernel.
__global__ void route_select_kernel(const float* g, int T, const float* bias, int* idx, float* wts) {
  const int t = blockIdx.x, lane = threadIdx.x;
  float lgv[4], key[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = lane + 32 * i;
    const float l = __fadd_rn(__fadd_rn(g[(size_t)t * NEXP + e], g[((size_t)T + t) * NEXP + e]),
                              g[((size_t)2 * T + t) * NEXP + e]);
    lgv[i] = l;
    key[i] = __fadd_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-l))), bias[e]);
  }
  unsigned used = 0;
  int sel[TOPK];
  float sl[TOPK];
  for (int j = 0; j < TOPK; ++j) {
    float bk = -INFINITY;
    int be = 1 << 30;
    float bl = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (!((used >> i) & 1) && (key[i] > bk || (key[i] == bk && lane + 32 * i < be))) { bk = key[i]; be = lane + 32 * i; bl = lgv[i]; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      const float ol = __shfl_xor_sync(0xffffffffu, bl, o);
      if (ok > bk || (ok == bk && oe < be)) { bk = ok; be = oe; bl = ol; }
    }
    sel[j] = be;
    sl[j] = bl;
    if ((be & 31) == lane) used |= 1u << (be >> 5);
  }
  if (lane == 0) {
    float lp[TOPK], mx = -INFINITY;
    for (int j = 0; j < TOPK; ++j) {
      lp[j] = __fadd_rn(fminf(sl[j], 0.f), -log1pf(expf(-fabsf(sl[j]))));
      mx = fmaxf(mx, lp[j]);
    }
    float ex[TOPK];
    for (int j = 0; j < TOPK; ++j) ex[j] = expf(__fadd_rn(lp[j], -mx));
    const float s8 = __fadd_rn(__fadd_rn(__fadd_rn(ex[0], ex[2]), __fadd_rn(ex[1], ex[3])),
                               __fadd_rn(__fadd_rn(ex[4], ex[6]), __fadd_rn(ex[5], ex[7])));
    const float base = __fadd_rn(mx, logf(s8));
    for (int j = 0; j < TOPK; ++j) {
      idx[t * TOPK + j] = sel[j];
      wts[t * TOPK + j] = __fmul_rn(expf(__fadd_rn(lp[j], -base)), 8.0f);
    }
  }
}

// message gate tail: f = gd . forget[h], g = gd . gain[h] (64-term fp32 fma, j ascending), out = (1 - tanh f) hid +
// tanh g msg. Replaces bf16->fp32 + 2 fp32 GEMMs + blend (fp32 order differs from cuBLAS; per row: batch-invariant).
// The kernel is pglue::mg_tail_rows_kernel (prefill_glue.cuh), for every row count (delta 2: mg_fused_kernel deleted).
struct MgTail {   // per gate (blockIdx.z): gd [T][64] bf16, transposed controller weights, blend inputs / output
  const bf16* gd[4];
  const float* FT[4];
  const float* GT[4];
  const float* hid[4];
  const float* msg[4];
  float* out[4];
};
// AttnRes + RMSNorm (production path): attn_res_kernel's arithmetic, output kept in registers, then for each query q
// rmsnorm_kernel's arithmetic with weight w[q] (same 256-thread loop and tree) -> bf16 out[q] (ld H). Bit-identical
// to attn_res_kernel + rmsnorm_kernel<float>. blockDim must be 256.
// Query split (decode calls) as attn_res_kernel: block y computes queries y * Q .. y * Q + Q - 1 (weights w, out o of
// those queries); bits unchanged.
template <int Q>
__global__ void __launch_bounds__(256) attn_res_norm_kernel(ResSrc s, const float* queries, const float* w0,
                                                            const float* w1, const float* w2, bf16* o0, bf16* o1,
                                                            bf16* o2, float eps) {
  __shared__ float sh[32 * 5 * (Q + 1)];
  const int t = blockIdx.x, N = s.nb + 1, q0 = blockIdx.y * Q;
  const float invH = (float)(1.0 / 2304.0);
  float xs[5][KH], v[5 * (Q + 1)];
  attn_res_load<Q>(s, queries + (size_t)q0 * H, t, N, xs, v, sh);
  float rstd[5], p[Q][5], inv[Q];
#pragma unroll
  for (int n = 0; n < 5; ++n) rstd[n] = n < N ? rsqrtf(__fmaf_rn(v[n * (Q + 1)], invH, eps)) : 0.f;
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    float mx = -INFINITY, l[5];
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) { l[n] = __fmul_rn(v[n * (Q + 1) + 1 + q], rstd[n]); mx = fmaxf(mx, l[n]); }
    float sum = 0.f;
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) { p[q][n] = expf(__fadd_rn(l[n], -mx)); sum = __fadd_rn(sum, p[q][n]); }
    inv[q] = __fdiv_rn(1.0f, sum);
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) p[q][n] = __fmul_rn(p[q][n], rstd[n]);
  }
  float res[Q][KH];   // the AttnRes row of this thread's elements (was a shared-memory row)
#pragma unroll
  for (int k = 0; k < KH; ++k) {
    float acc[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) acc[q] = 0.f;
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) {
#pragma unroll
        for (int q = 0; q < Q; ++q) acc[q] = __fmaf_rn(p[q][n], xs[n][k], acc[q]);
      }
#pragma unroll
    for (int q = 0; q < Q; ++q) res[q][k] = __fmul_rn(acc[q], inv[q]);
  }
  const float* wall[3] = {w0, w1, w2};
  bf16* oall[3] = {o0, o1, o2};
  const float* ws[Q];
  bf16* os[Q];
#pragma unroll
  for (int q = 0; q < Q; ++q) ws[q] = wall[q0 + q], os[q] = oall[q0 + q];
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    float wv[KH];
#pragma unroll
    for (int k = 0; k < KH; ++k) wv[k] = ws[q] ? ws[q][threadIdx.x + kRowThreads * k] : 1.f;
    float u[1] = {0.f};
#pragma unroll
    for (int k = 0; k < KH; ++k) u[0] = __fadd_rn(u[0], __fmul_rn(res[q][k], res[q][k]));
    block_sum<1>(u, sh);
    const float r = rstd_of(u[0], H, eps);
#pragma unroll
    for (int k = 0; k < KH; ++k) {
      float y = __fmul_rn(res[q][k], r);
      if (ws[q]) y = __fmul_rn(y, wv[k]);
      os[q][(size_t)t * H + threadIdx.x + kRowThreads * k] = r16(y);
    }
  }
}

// q / rel post + relative logits (blocks y < 24, qpost_kernel + rel_logits_kernel arithmetic) and key / value conv +
// KV write (y >= 24, kv_conv_kernel + kv_write_kernel arithmetic) in one launch; block 128 (production path)
__global__ void __launch_bounds__(128) qkv_post_kernel(const bf16* qproj, const float* q_scale, const bf16* rproj,
                                                       const float* r_scale, const int* row_pos, int full,
                                                       const bf16* proj, bf16* q, bf16* rel, bf16* rel_logits,
                                                       const bf16* kx, const bf16* vx, const bf16* kst, const bf16* vst,
                                                       long long st_slot_stride, Rows R, const float* kw, const float* vw,
                                                       const bf16* kdyn, const bf16* vdyn, bf16* kout, bf16* vout,
                                                       float eps, bf16* kpool, bf16* vpool, const int* pt, int pt_stride) {
  // the two device bodies are the originals' code (glue.cuh); dispatch on blockIdx.y
  if (blockIdx.y < NH) {
    __shared__ float sh[64];
    __shared__ float rs[DREL];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    float y = __fmul_rn(ld(qproj + ((size_t)t * NH + hh) * HD + d), q_scale[hh * HD + d]);
    float r = 0.f;
    if (d < DREL) r = __fmul_rn(ld(rproj + ((size_t)t * NH + hh) * DREL + d), r_scale[hh * DREL + d]);
    float v[2] = {__fmul_rn(y, y), __fmul_rn(r, r)};
    block_sum<2>(v, sh);
    auto clipf = [](float s2, float thr) {
      const float norm = s2 > 0.f ? __fmul_rn(s2, rsqrtf(s2)) : 0.f;
      return fminf(__fdiv_rn(thr, __fadd_rn(norm, 1e-6f)), 1.0f);
    };
    float qo = rr16(__fmul_rn(y, clipf(v[0], 22.627416997969522f)));
    float ro = rr16(__fmul_rn(r, clipf(v[1], 8.0f)));
    if (full) {
      const float tau = tau_of(row_pos[t]);
      qo = rr16(__fmul_rn(qo, tau));
      ro = rr16(__fmul_rn(ro, tau));
    }
    q[((size_t)t * NH + hh) * HD + d] = r16(qo);
    if (d < DREL) { rel[((size_t)t * NH + hh) * DREL + d] = r16(ro); rs[d] = ro; }
    __syncthreads();
    for (int x = d; x < REL_EXT; x += blockDim.x) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < DREL; ++j) acc = __fmaf_rn(rs[j], ld(proj + j * REL_EXT + x), acc);
      rel_logits[((size_t)t * NH + hh) * REL_EXT + x] = r16(acc);
    }
    return;
  }
  __shared__ float sh2[64];
  const int t = blockIdx.x, g = blockIdx.y - NH, d = threadIdx.x, c = g * HD + d;
  const int s = R.seg[t], r0 = R.row0[s], i = t - r0;
  const bf16* ks = kst + (size_t)R.slot[s] * st_slot_stride;
  const bf16* vs = vst + (size_t)R.slot[s] * st_slot_stride;
  float ck, cv;
  if (!full) {
    ck = static_conv(kw, kx, 1024, ks, t, i, c, 1024);
    cv = static_conv(vw, vx, 1024, vs, t, i, c, 1024);
  } else {
    auto dyn = [&](const float* sw, const bf16* x, const bf16* st, const bf16* dy) {
      float out = 0.f;
#pragma unroll
      for (int lag = 0; lag < 4; ++lag) {
        const float w = __fadd_rn(sw[lag * 1024 + c], ld(dy + (size_t)t * 32 + g * 4 + lag));
        const float xt = ext_at(x, 1024, st, t, i, 3 - lag, c, 1024);
        out = lag == 0 ? __fmul_rn(w, xt) : __fmaf_rn(w, xt, out);
      }
      return __fmul_rn(__fadd_rn(out, ld(x + (size_t)t * 1024 + c)), (float)(1.0 / 3.0));
    };
    ck = dyn(kw, kx, ks, kdyn);
    cv = dyn(vw, vx, vs, vdyn);
  }
  ck = rr16(ck);
  cv = rr16(cv);
  float v[2] = {__fmul_rn(ck, ck), __fmul_rn(cv, cv)};
  block_sum<2>(v, sh2);
  const float norm = v[0] > 0.f ? __fmul_rn(v[0], rsqrtf(v[0])) : 0.f;
  const float f = fminf(__fdiv_rn(22.627416997969522f, __fadd_rn(norm, 1e-6f)), 1.0f);
  const bf16 ko = r16(__fmul_rn(ck, f)), vo = r16(__fmul_rn(cv, rstd_of(v[1], HD, eps)));
  kout[(size_t)t * 1024 + c] = ko;
  vout[(size_t)t * 1024 + c] = vo;
  const int pos = R.pos[t];
  const int page = pt[(size_t)s * pt_stride + (pos >> 10)];
  const size_t off = (((size_t)page * KVH + g) * 1024 + (pos & 1023)) * HD + d;
  kpool[off] = ko;
  vpool[off] = vo;
}

// Same as qkv_post_kernel, writing global KV pages in FP8 E4M3 (unit scale; bf16-rounded ko/vo -> E4M3 so bits match
// pglue::qkv_post_rows_fp8_kernel and glue::kv_write_fp8_kernel).
__global__ void __launch_bounds__(128) qkv_post_fp8_kernel(
    const bf16* qproj, const float* q_scale, const bf16* rproj, const float* r_scale, const int* row_pos, int full,
    const bf16* proj, bf16* q, bf16* rel, bf16* rel_logits, const bf16* kx, const bf16* vx, const bf16* kst,
    const bf16* vst, long long st_slot_stride, Rows R, const float* kw, const float* vw, const bf16* kdyn,
    const bf16* vdyn, bf16* kout, bf16* vout, float eps, uint8_t* kpool, uint8_t* vpool, const int* pt, int pt_stride) {
  if (blockIdx.y < NH) {
    __shared__ float sh[64];
    __shared__ float rs[DREL];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    float y = __fmul_rn(ld(qproj + ((size_t)t * NH + hh) * HD + d), q_scale[hh * HD + d]);
    float r = 0.f;
    if (d < DREL) r = __fmul_rn(ld(rproj + ((size_t)t * NH + hh) * DREL + d), r_scale[hh * DREL + d]);
    float v[2] = {__fmul_rn(y, y), __fmul_rn(r, r)};
    block_sum<2>(v, sh);
    auto clipf = [](float s2, float thr) {
      const float norm = s2 > 0.f ? __fmul_rn(s2, rsqrtf(s2)) : 0.f;
      return fminf(__fdiv_rn(thr, __fadd_rn(norm, 1e-6f)), 1.0f);
    };
    float qo = rr16(__fmul_rn(y, clipf(v[0], 22.627416997969522f)));
    float ro = rr16(__fmul_rn(r, clipf(v[1], 8.0f)));
    if (full) {
      const float tau = tau_of(row_pos[t]);
      qo = rr16(__fmul_rn(qo, tau));
      ro = rr16(__fmul_rn(ro, tau));
    }
    q[((size_t)t * NH + hh) * HD + d] = r16(qo);
    if (d < DREL) { rel[((size_t)t * NH + hh) * DREL + d] = r16(ro); rs[d] = ro; }
    __syncthreads();
    for (int x = d; x < REL_EXT; x += blockDim.x) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < DREL; ++j) acc = __fmaf_rn(rs[j], ld(proj + j * REL_EXT + x), acc);
      rel_logits[((size_t)t * NH + hh) * REL_EXT + x] = r16(acc);
    }
    return;
  }
  __shared__ float sh2[64];
  const int t = blockIdx.x, g = blockIdx.y - NH, d = threadIdx.x, c = g * HD + d;
  const int s = R.seg[t], r0 = R.row0[s], i = t - r0;
  const bf16* ks = kst + (size_t)R.slot[s] * st_slot_stride;
  const bf16* vs = vst + (size_t)R.slot[s] * st_slot_stride;
  float ck, cv;
  if (!full) {
    ck = static_conv(kw, kx, 1024, ks, t, i, c, 1024);
    cv = static_conv(vw, vx, 1024, vs, t, i, c, 1024);
  } else {
    auto dyn = [&](const float* sw, const bf16* x, const bf16* st, const bf16* dy) {
      float out = 0.f;
#pragma unroll
      for (int lag = 0; lag < 4; ++lag) {
        const float w = __fadd_rn(sw[lag * 1024 + c], ld(dy + (size_t)t * 32 + g * 4 + lag));
        const float xt = ext_at(x, 1024, st, t, i, 3 - lag, c, 1024);
        out = lag == 0 ? __fmul_rn(w, xt) : __fmaf_rn(w, xt, out);
      }
      return __fmul_rn(__fadd_rn(out, ld(x + (size_t)t * 1024 + c)), (float)(1.0 / 3.0));
    };
    ck = dyn(kw, kx, ks, kdyn);
    cv = dyn(vw, vx, vs, vdyn);
  }
  ck = rr16(ck);
  cv = rr16(cv);
  float v[2] = {__fmul_rn(ck, ck), __fmul_rn(cv, cv)};
  block_sum<2>(v, sh2);
  const float norm = v[0] > 0.f ? __fmul_rn(v[0], rsqrtf(v[0])) : 0.f;
  const float f = fminf(__fdiv_rn(22.627416997969522f, __fadd_rn(norm, 1e-6f)), 1.0f);
  const bf16 ko = r16(__fmul_rn(ck, f)), vo = r16(__fmul_rn(cv, rstd_of(v[1], HD, eps)));
  kout[(size_t)t * 1024 + c] = ko;
  vout[(size_t)t * 1024 + c] = vo;
  const int pos = R.pos[t];
  const int page = pt[(size_t)s * pt_stride + (pos >> 10)];
  const size_t off = (((size_t)page * KVH + g) * 1024 + (pos & 1023)) * HD + d;
  kpool[off] = f2e4m3(__bfloat162float(ko));
  vpool[off] = f2e4m3(__bfloat162float(vo));
}

// DSpark features: feats[t][j][:] = src[t][:] (bf16) or bf16(src32) (j = 0..3: entering visits 5, 23, 41; final h)
__global__ void feat_copy_kernel(const bf16* src, const float* src32, bf16* feats, int j, int T) {
  const size_t n = (size_t)T * H;
  for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < n; k += (size_t)gridDim.x * blockDim.x) {
    const size_t t = k / H, c = k % H;
    feats[(t * 4 + j) * H + c] = src ? src[k] : r16(src32[k]);
  }
}

}  // namespace glue
