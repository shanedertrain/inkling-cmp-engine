// inkling-cmp-engine S2b: batched paged decode / speculative-verify attention for Inkling-2-Mini on sm_80 (CMP 170HX).
//
// Lineage: restructured from the P6 stand-in probes/p6-attn/verify_attn.cu, itself adapted from maya-inkling-mini
// (vendor/maya-inkling-mini @ e819852, src/ik2/ik2_kernels.cu: attn_block_mma_kernel 4438-4661 - (token, head) pairs as
// mma columns, K/V fragments loaded once per 16-key step and fed to every column tile, per-column masks and relative
// bias, online softmax in log2 units, P packed through warp shuffles; attn_block_combine_kernel 4662-4697: fixed-order
// merge; e4m3x4_h4 964-985). New here (2026-11-23): up to three n8 column tiles per warp with K as the A operand,
// K/V reused from the same registers by every tile, one instruction sequence per tile so a column's bits do not
// depend on its index (F1: verify rows == one-row decode rows); lazy rescale (per-column threshold, warp vote, reductions only in the rare slow path);
// 128-bit K loads through a dimension permutation; register-level K/V prefetch; absolute-anchored fixed key chunks
// (batch-invariant partition), warp-level chunk partials and a fixed-order combine; paged KV through a page table for
// both kinds (an SWA ring is a session's page-table row over a few recycled pages); config B's gate-then-sink
// epilogue with one bf16 rounding. Design B (SWA): shared cp.async ring, column-split warps.
//
// The maya-inkling-mini licence (MIT) applies to the adapted parts:
//   MIT License
//   Copyright (c) 2026 Peasant Smith and the Project Maya contributors
//   Copyright (c) 2026 Niko1221 and the Strata contributors
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
//   permit persons to whom the Software is furnished to do so, subject to the following conditions:
//   The above copyright notice and this permission notice shall be included in all copies or substantial portions of
//   the Software.
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//   THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.
//
// Semantics (config B, docs/engine/configB-numerics.md rows 41-44; engine/ref/model.py attend):
//   s = (q . k) * scale * log2e + rel[t, h, d] * log2e   for d = pos(t) - key < extent (else no bias)
//   visible: 0 <= d (causal), and d < window on sliding layers
//   p = 2^(s - m) (fp32), P.V with P rounded to bf16 (f16 for FP8 KV), row sum in fp32
//   o = O / L, lse = (M + log2 L) ln2, out = bf16( o * sigmoid(gate) * sigmoid(lse - sink) )   (one rounding)
// Batch invariance: a row's keys are cut into chunks of C keys anchored at absolute key positions (C fixed per kind
// and context bucket of the row's own session); each chunk is processed by one warp from a fresh state and written as
// one partial; the combine folds chunks in increasing order. Every column sits in an n8 tile run by the same
// instruction sequence, so nothing depends on the number of sessions, the grid, other rows' data, or the row's index
// inside its verify block (a verify row equals the one-row decode row at its position).
#pragma once
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace da {

constexpr int kD = 128, kHq = 24, kHkv = 8, kG = 3, kNW = 4;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr int kQsBytes = 3 * 8 * 32 * 8, kStageBytes = 24 * 128 * 4;  // Q fragments of 3 n8 tiles; partial stage
constexpr int kNP16 = 2;  // 2 warp-pairs per CTA in 16-row single-pass global kernel (4 warps/CTA)
constexpr int kQsBytes16 = 6 * 8 * 32 * 8;      // Q fragments of 6 n8 tiles (12,288 B)
constexpr int kKvPipeBytes16 = kNP16 * 2 * 2 * 8 * 32 * 16;  // 2 pairs x 2 stages x (K + V) x 4,096 B (32,768 B)
constexpr float kTau = 0.0f;  // lazy-rescale threshold (log2): 0 keeps p_max = 1 exactly in the step that sets the max
                              // (config B property; tau 8 cost +3.5 % real-KV error, LEDGER t13); reductions still
                              // only on steps with a new max
enum KvT : int { KV_BF16 = 0, KV_E4M3 = 1 };

struct Params {
  // ---- KV (bytes), paged for both kinds. Address of key k, head g of session s (s = index in this launch):
  //   kv + page_table[s * pt_stride + (k >> page_shift)] * page_stride + g * head_stride + (k & (2^page_shift - 1)) * row_stride
  // Design B and the recommended layout need head-major pages: row_stride = 128 * element bytes (a 16-key step is one
  // contiguous block). An SWA ring is the same thing: the session's page-table row maps logical page j to one of its
  // ring pages (e.g. slot * 3 + j % 3); the kernel never wraps by itself. See README "KV contract".
  const uint8_t* k;
  const uint8_t* v;
  const int* page_table;  // [n][pt_stride], row s = launch index s
  int pt_stride, page_shift;
  long long page_stride, head_stride;
  int row_stride;
  int window;  // keys visible: d < window (1 << 30 = global)
  // ---- rows: session s owns rows row0[s] .. row0[s] + R_s - 1 at positions ctx[s] .. ctx[s] + R_s - 1
  const int* ctx;
  const int* row0;
  const int* chunk_shift;  // [n] log2 chunk keys for this kind (fixed per kind and context bucket)
  int R;                 // rows per session (max when nrows is set); partial column stride = 3 R; combine grid 3 R
  const int* nrows;      // optional [n] per-session rows 1..R (ragged verify widths); nullptr = uniform R
  const uint16_t* q;     // [T][24][128] bf16
  const uint16_t* rel;   // [T][24][extent] bf16, by distance
  int extent;
  const uint16_t* gate;  // [T][24*128] bf16
  const uint16_t* sink;  // [T][24] bf16
  // ---- partials and output
  float* o_part;      // [n][8][max_chunks][3 R][128]
  float2* ml_part;    // [n][8][max_chunks][3 R] (max, sum), log2 units
  int max_chunks;     // >= the largest chunk count of any session in the call (may be larger: graph capture headroom)
  uint16_t* out;      // [T][24*128] bf16
  float* out32;       // optional [T][24*128] pre-epilogue o (oracle)
  float sc;           // scale * log2e * K dequant fold
  float vscale;       // V dequant fold
};

// ------------------------------------------------------------------------------------------- device helpers
template <bool BF>
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  if constexpr (BF)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <bool BF>
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  if constexpr (BF) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
  } else {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
  }
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// K/V loads (design A): streaming, no L1 allocation, 256-byte L2 prefetch (LEDGER t16/t18: -3 % vs plain ld.nc)
#define DA_LDG "ld.global.nc.L1::no_allocate.L2::256B.v4.u32"
__device__ __forceinline__ uint4 ldg16(const uint8_t* p) {
  uint4 r;
  asm volatile(DA_LDG " {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ uint16_t f2bf(float x) {
  __nv_bfloat16 h = __float2bfloat16_rn(x);
  return *reinterpret_cast<uint16_t*>(&h);
}
// bf16 pair -> f16 pair (q for the FP8 path; exact for |x| in f16 normal range)
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t w) {
  const float a = __uint_as_float(w << 16), b = __uint_as_float(w & 0xffff0000u);
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
// E4M3 placement decode (P6 fp8_decode, ours): byte b to the high byte of an f16 by one PRMT, magnitude >> 1, sign
// kept: f16 bits = s<<15 | (b & 0x7f) << 7 == E4M3 value * 2^-8, exact for every finite code incl. subnormals.
__device__ __forceinline__ uint32_t place2(uint32_t x, uint32_t y, uint32_t sel) {
  const uint32_t h = __byte_perm(x, y, sel);
  return ((h >> 1) & 0x3F803F80u) | (h & 0x80008000u);
}
__host__ __device__ constexpr uint32_t psel(int i, int j) {
  return (uint32_t)i | ((uint32_t)i << 4) | ((uint32_t)i << 8) | ((uint32_t)j << 12);
}
__device__ __forceinline__ float sigmoid_cfgb(float x) { return rcp(1.0f + ex2(-x * kLog2e)); }
__device__ __forceinline__ void cp16(void* sm, const void* gm) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(sm)), "l"(gm));
}
// 16-byte cp.async with zero fill: src_bytes 0 writes zeros (masked load, as config B's `other=0`)
__device__ __forceinline__ void cp16z(uint32_t sm, const void* gm, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sm), "l"(gm), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}

// ------------------------------------------------------------------------------------------- the kernel
// grid (chunk index from the session's first visible chunk, 8 KV heads, n sessions), 128 threads.
// Columns in n8 tiles (K as the A operand, Q as the B operand): tile u holds columns 8u..8u+7; NT = 1 tile (3R <= 8)
// or 3 tiles (WIDE, 3R > 8). Every tile runs the same instruction sequence, and inside a tile a column's arithmetic
// does not depend on its lane, so a column's bits do not depend on its index: verify row t of an R-row block equals a
// one-row step at pos0 + t bit for bit (F1, bench/raw/2026-11-23-f1-verify-rows/; the former m16 tile for columns
// 8-23 summed in another order).
// Register map per 16-key step (bf16):
//   K: rows gr, gr+8; 4 x 16 B each: dims j*32 + tq*8 + 0..7 -> mma k-block 2j (.x,.y) and 2j+1 (.z,.w)
//   V: rows 2tq, 2tq+1, 2tq+8, 2tq+9; 2 x 16 B each: dims gr*16 + 8i + 0..7
// FP8: K rows gr, gr+8, 2 x 16 B each: dims tq*32 + 0..31 -> k-block kk uses dims tq*32 + 4kk + 0..3;
//      V rows as bf16, 1 x 16 B each: dims gr*16 + 0..15.
// WIDE kernels stage the Q fragments in smem (frees registers: no spills, LEDGER A8 / review D5).
template <int KV, bool WIDE>
__global__ void __launch_bounds__(kNW * 32, 2) decode_attn_kernel(Params p) {
  constexpr bool QS = WIDE;
  constexpr int NT = WIDE ? 3 : 1;
  constexpr bool F8 = KV == KV_E4M3;
  constexpr bool BF = !F8;
  extern __shared__ __align__(16) uint8_t smem[];
  const int s = blockIdx.z, g = blockIdx.y;
  const int lane = threadIdx.x & 31, gr = lane >> 2, tq = lane & 3, w = threadIdx.x >> 5;
  const int R = p.nrows ? p.nrows[s] : p.R, NC = kG * R, NCs = kG * p.R;  // ragged rows; partial stride uses p.R = max
  bool dt[NT];  // tile u has columns (a ragged session with <= 8 columns runs tile 0 only)
#pragma unroll
  for (int u = 0; u < NT; ++u) dt[u] = 8 * u < NC;
  const int pos0 = p.ctx[s], qlast = pos0 + R - 1;
  const bool windowed = p.window < (1 << 30);
  const int k_lo = windowed ? max(0, pos0 - p.window + 1) : 0;
  const int cs = p.chunk_shift[s];
  // persistent worker: CTA blockIdx.x of gridDim.x takes chunks [i0, i1) of this session's visible chunks
  const int c_first = k_lo >> cs, nch = (qlast >> cs) - c_first + 1;
  const int i0 = (int)((long long)blockIdx.x * nch / gridDim.x), i1 = (int)((long long)(blockIdx.x + 1) * nch / gridDim.x);
  if (i0 >= i1) return;
  const int row0 = p.row0[s];
  const int rs = p.row_stride;
  const int pmask = (1 << p.page_shift) - 1;
  // warp w owns whole chunks i0 + w, i0 + w + 4, ... (< i1): its steps in chunk ci (first step key and count)
  auto wsteps = [&](int ci, int& st, int& n) {
    const int clo = (c_first + ci) << cs;
    const int a_lo = max(clo, k_lo), a_hi = min(clo + (1 << cs) - 1, qlast);
    st = a_lo & ~15;
    n = a_hi >= a_lo ? ((a_hi - st) >> 4) + 1 : 0;
  };
  // per-thread pointers to its K row gr and V row 2tq for the step starting at key st
  auto ptr_for = [&](int st, const uint8_t*& kq, const uint8_t*& vq) {
    const int page = p.page_table[(size_t)s * p.pt_stride + (st >> p.page_shift)];
    const long long off = (long long)page * p.page_stride + (long long)g * p.head_stride + (long long)(st & pmask) * rs;
    kq = p.k + off + gr * rs + (F8 ? tq * 32 : tq * 16);
    vq = p.v + off + 2 * tq * rs + gr * 16;  // bf16: dims i*64 + gr*8 + 0..7 (i = 0, 1); FP8: dims gr*16 + 0..15
  };
  const uint8_t* kp = nullptr;
  const uint8_t* vp = nullptr;
  // ---- Q fragments (B operand of tile u: column 8u + gr)
  const uint16_t* qbase = p.q + ((size_t)row0 * kHq + g * kG) * kD;  // + (t*24 + h)*128
  auto qrow = [&](int col) -> const uint16_t* {
    const int cc = col < NC ? col : 0;
    return qbase + ((size_t)(cc / kG) * kHq + cc % kG) * kD;
  };
  uint32_t qn[NT][8][2];
#pragma unroll
  for (int u = 0; u < NT; ++u) {
    const uint16_t* r = qrow(8 * u + gr);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if constexpr (!F8) {
        const uint4 v = *reinterpret_cast<const uint4*>(r + j * 32 + tq * 8);
        qn[u][2 * j][0] = v.x, qn[u][2 * j][1] = v.y, qn[u][2 * j + 1][0] = v.z, qn[u][2 * j + 1][1] = v.w;
      } else {
        // FP8: k-block kk uses dims tq*32 + 4kk + {0,1} (b0) and {2,3} (b1)
        const uint4 v = *reinterpret_cast<const uint4*>(r + tq * 32 + j * 8);
        qn[u][2 * j][0] = bf2_to_h2(v.x), qn[u][2 * j][1] = bf2_to_h2(v.y);
        qn[u][2 * j + 1][0] = bf2_to_h2(v.z), qn[u][2 * j + 1][1] = bf2_to_h2(v.w);
      }
    }
  }
  // QS: one per-lane fragment copy per CTA (every warp has the same (s, g) and lane mapping): [u][kk][lane] uint2;
  // warp w writes k-blocks w and w + 4.
  uint2* qs_n = reinterpret_cast<uint2*>(smem);
  if constexpr (QS) {
#pragma unroll
    for (int u = 0; u < NT; ++u)
#pragma unroll
      for (int kk = 0; kk < 8; ++kk)
        if ((kk & 3) == w) qs_n[(u * 8 + kk) * 32 + lane] = make_uint2(qn[u][kk][0], qn[u][kk][1]);
    __syncthreads();
  }
  const int ca = 2 * tq, cb = ca + 1;             // tile columns of this thread (+ 8u in tile u)
  const int extent = p.extent, window = p.window;
  const float sc = p.sc;

  float on[NT][8][4];   // accumulators: [tile][dim block mb][(dim gr*16+2mb | +1) x (col ca | cb)]
  float ma[NT], mb[NT], la[NT], lb[NT];  // running max (log2) and sum per column
#pragma unroll
  for (int u = 0; u < NT; ++u) {
#pragma unroll
    for (int i = 0; i < 8; ++i) on[u][i][0] = on[u][i][1] = on[u][i][2] = on[u][i][3] = 0.f;
    ma[u] = mb[u] = -INFINITY, la[u] = lb[u] = 0.f;
  }
  const int src = tq * 8 + (gr >> 1);
  const uint32_t psl = (gr & 1) ? 0x7632u : 0x5410u;

  // ---- register prefetch of the first step
  uint4 kr[2][F8 ? 2 : 4];
  uint4 vr[4][F8 ? 1 : 2];
  auto load_k = [&]() {
#pragma unroll
    for (int j = 0; j < (F8 ? 2 : 4); ++j) {
      kr[0][j] = ldg16(kp + j * (F8 ? 16 : 64));
      kr[1][j] = ldg16(kp + 8 * rs + j * (F8 ? 16 : 64));
    }
  };
  auto load_v = [&]() {
#pragma unroll
    for (int i = 0; i < (F8 ? 1 : 2); ++i) {
      vr[0][i] = ldg16(vp + i * 128);
      vr[1][i] = ldg16(vp + rs + i * 128);
      vr[2][i] = ldg16(vp + 8 * rs + i * 128);
      vr[3][i] = ldg16(vp + 9 * rs + i * 128);
    }
  };
  // load cursor (one step ahead of compute): chunk lci, step lj of lnst; kp/vp point at it
  int lci = i0 + w, lj = 0, lnst = 0, lst = 0;
  auto seek = [&](int ci) {  // first of this warp's chunks >= ci with steps
    for (lci = ci; lci < i1; lci += kNW) {
      wsteps(lci, lst, lnst);
      if (lnst > 0) break;
    }
    lj = 0;
    if (lci < i1) ptr_for(lst, kp, vp);
  };
  auto advance = [&]() {
    if (++lj < lnst) {
      const int key = lst + 16 * lj;
      if ((key & pmask) == 0) ptr_for(key, kp, vp);  // a page starts: look the next page up
      else kp += 16 * rs, vp += 16 * rs;
    } else {
      seek(lci + kNW);
    }
  };
  auto dimn = [&](int b, int rowg) { return F8 ? rowg * 16 + 2 * b : (b >> 2) * 64 + rowg * 8 + 2 * (b & 3); };
  float sn[NT][4];
  auto qk = [&]() {
    // ---- QK^T
#pragma unroll
    for (int u = 0; u < NT; ++u) sn[u][0] = sn[u][1] = sn[u][2] = sn[u][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      uint32_t a0, a1, a2, a3;
      if constexpr (!F8) {
        const uint4& x = kr[0][kk >> 1];
        const uint4& y = kr[1][kk >> 1];
        a0 = (kk & 1) ? x.z : x.x, a2 = (kk & 1) ? x.w : x.y;
        a1 = (kk & 1) ? y.z : y.x, a3 = (kk & 1) ? y.w : y.y;
      } else {
        // k-block kk: word kk of the 32 dims (bytes 4kk..4kk+3) -> f16x2 dims (0,1) and (2,3)
        const uint4& x = kr[0][kk >> 2];
        const uint4& y = kr[1][kk >> 2];
        const uint32_t wx = (kk & 3) == 0 ? x.x : (kk & 3) == 1 ? x.y : (kk & 3) == 2 ? x.z : x.w;
        const uint32_t wy = (kk & 3) == 0 ? y.x : (kk & 3) == 1 ? y.y : (kk & 3) == 2 ? y.z : y.w;
        a0 = place2(wx, wx, psel(0, 1)), a2 = place2(wx, wx, psel(2, 3));
        a1 = place2(wy, wy, psel(0, 1)), a3 = place2(wy, wy, psel(2, 3));
      }
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        if (!dt[u]) continue;
        uint32_t b0 = qn[u][kk][0], b1 = qn[u][kk][1];
        if constexpr (QS)
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1)
                       : "r"((unsigned)__cvta_generic_to_shared(qs_n + (u * 8 + kk) * 32 + lane)));
        mma<BF>(sn[u], a0, a1, a2, a3, b0, b1);
      }
    }
  };
  // EDGE: a step with keys outside [k_lo, qlast] (D3 masking). A separate instance keeps the zeroing out of the hot
  // loop (as one branch nvcc if-converts it to 32 SELs per step: +2 % n 4, +4.5 % n 1; LEDGER 12)
  auto rest = [&](int kt, auto edge_c) {
    constexpr bool EDGE = decltype(edge_c)::value;
    // ---- scores relative to the running offset u (log2 domain): t = acc*sc - u (+ bias), masked -> -inf
    // sn[u][0] (key kt+gr, col ca), [1] (kt+gr, cb), [2] (kt+gr+8, ca), [3] (kt+gr+8, cb); columns + 8u
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      if (!dt[u]) continue;
      const float ua = ma[u] == -INFINITY ? 0.f : ma[u], ub = mb[u] == -INFINITY ? 0.f : mb[u];
      sn[u][0] = fmaf(sn[u][0], sc, -ua), sn[u][1] = fmaf(sn[u][1], sc, -ub);
      sn[u][2] = fmaf(sn[u][2], sc, -ua), sn[u][3] = fmaf(sn[u][3], sc, -ub);
    }
    const bool near = pos0 - kt < extent + 15;
    const bool full = (kt + 15 <= pos0) && (!windowed || kt > qlast - window);
    if (near || !full) {
      auto cpos = [&](int col) { return col < NC ? pos0 + col / kG : -1; };
      auto reloff = [&](int col) -> int {
        const int cc = col < NC ? col : 0;
        return (int)((((size_t)row0 + cc / kG) * kHq + g * kG + cc % kG) * extent);
      };
      auto fix = [&](float& v, int pc, int key, int roff) {
        const int d = pc - key;
        if (pc >= 0 && d >= 0 && d < extent) v = fmaf(bf2f(__ldg(p.rel + roff + d)), kLog2e, v);
        const bool vis = pc >= 0 && d >= 0 && (!windowed || d < window);
        v = vis ? v : -INFINITY;
      };
      const int k0 = kt + gr, k1 = k0 + 8;
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        if (!dt[u]) continue;
        const int pa = cpos(8 * u + ca), pb = cpos(8 * u + cb), ra = reloff(8 * u + ca), rb = reloff(8 * u + cb);
        fix(sn[u][0], pa, k0, ra), fix(sn[u][1], pb, k0, rb), fix(sn[u][2], pa, k1, ra), fix(sn[u][3], pb, k1, rb);
      }
    }
    // ---- lazy rescale: column c rescales iff its step max exceeds m_c + tau (thr_c = tau, or -inf before the
    // first visible key); warp vote; reductions and rescale only in the slow path; per-column decisions only.
    {
      float ta[NT], tb[NT];
      bool trig = false;
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        ta[u] = ma[u] == -INFINITY ? -INFINITY : kTau, tb[u] = mb[u] == -INFINITY ? -INFINITY : kTau;
        trig = trig || (dt[u] && (sn[u][0] > ta[u] || sn[u][2] > ta[u] || sn[u][1] > tb[u] || sn[u][3] > tb[u]));
      }
      if (__any_sync(0xffffffffu, trig)) {
#pragma unroll
        for (int u = 0; u < NT; ++u) {
          if (!dt[u]) continue;
          // x = step max relative to the old offset; new offset = old + x
          float xa = fmaxf(sn[u][0], sn[u][2]), xb = fmaxf(sn[u][1], sn[u][3]);
#pragma unroll
          for (int sh = 4; sh < 32; sh <<= 1) {
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, sh));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, sh));
          }
          if (xa > ta[u]) {
            const float f = ma[u] == -INFINITY ? 0.f : ex2(-xa);
            ma[u] = (ma[u] == -INFINITY ? 0.f : ma[u]) + xa, la[u] *= f, sn[u][0] -= xa, sn[u][2] -= xa;
#pragma unroll
            for (int i = 0; i < 8; ++i) on[u][i][0] *= f, on[u][i][2] *= f;
          }
          if (xb > tb[u]) {
            const float f = mb[u] == -INFINITY ? 0.f : ex2(-xb);
            mb[u] = (mb[u] == -INFINITY ? 0.f : mb[u]) + xb, lb[u] *= f, sn[u][1] -= xb, sn[u][3] -= xb;
#pragma unroll
            for (int i = 0; i < 8; ++i) on[u][i][1] *= f, on[u][i][3] *= f;
          }
        }
      }
    }
    // ---- probabilities (P of tile u as its B operand: shuffled from the score layout)
    uint32_t pb0[NT], pb1[NT];
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      pb0[u] = pb1[u] = 0;
      if (!dt[u]) continue;
      const float pn0 = ex2(sn[u][0]), pn1 = ex2(sn[u][1]), pn2 = ex2(sn[u][2]), pn3 = ex2(sn[u][3]);
      la[u] += pn0 + pn2, lb[u] += pn1 + pn3;
      const uint32_t plo = pack2<BF>(pn0, pn1), phi = pack2<BF>(pn2, pn3);
      const uint32_t alo = __shfl_sync(0xffffffffu, plo, src), blo = __shfl_sync(0xffffffffu, plo, src + 4);
      const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src), bhi = __shfl_sync(0xffffffffu, phi, src + 4);
      pb0[u] = __byte_perm(alo, blo, psl), pb1[u] = __byte_perm(ahi, bhi, psl);
    }
    // ---- masked loads (review D3): keys of a boundary step outside [k_lo, qlast] are visible to no row; their V
    // rows may be unwritten memory, so they are zeroed (P is 0 there, so finite values gave the same bits)
    if constexpr (EDGE) {
      const int kv0 = kt + 2 * tq;
      const bool z0 = kv0 < k_lo || kv0 > qlast, z1 = kv0 + 1 < k_lo || kv0 + 1 > qlast;
      const bool z8 = kv0 + 8 < k_lo || kv0 + 8 > qlast, z9 = kv0 + 9 < k_lo || kv0 + 9 > qlast;
      const uint4 zz = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int i = 0; i < (F8 ? 1 : 2); ++i) {
        if (z0) vr[0][i] = zz;
        if (z1) vr[1][i] = zz;
        if (z8) vr[2][i] = zz;
        if (z9) vr[3][i] = zz;
      }
    }
    // ---- P.V: A = V^T (16 dims x 16 keys), shared by the tiles
    if constexpr (!F8) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint32_t x0[4] = {vr[0][i].x, vr[0][i].y, vr[0][i].z, vr[0][i].w};
        const uint32_t x1[4] = {vr[1][i].x, vr[1][i].y, vr[1][i].z, vr[1][i].w};
        const uint32_t x8[4] = {vr[2][i].x, vr[2][i].y, vr[2][i].z, vr[2][i].w};
        const uint32_t x9[4] = {vr[3][i].x, vr[3][i].y, vr[3][i].z, vr[3][i].w};
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int b = 4 * i + jj;
          const uint32_t lo01 = __byte_perm(x0[jj], x1[jj], 0x5410u), hi01 = __byte_perm(x0[jj], x1[jj], 0x7632u);
          const uint32_t lo89 = __byte_perm(x8[jj], x9[jj], 0x5410u), hi89 = __byte_perm(x8[jj], x9[jj], 0x7632u);
#pragma unroll
          for (int u = 0; u < NT; ++u)
            if (dt[u]) mma<BF>(on[u][b], lo01, hi01, lo89, hi89, pb0[u], pb1[u]);
        }
      }
    } else {
      const uint32_t x0[4] = {vr[0][0].x, vr[0][0].y, vr[0][0].z, vr[0][0].w};
      const uint32_t x1[4] = {vr[1][0].x, vr[1][0].y, vr[1][0].z, vr[1][0].w};
      const uint32_t x8[4] = {vr[2][0].x, vr[2][0].y, vr[2][0].z, vr[2][0].w};
      const uint32_t x9[4] = {vr[3][0].x, vr[3][0].y, vr[3][0].z, vr[3][0].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int b = 2 * i + hh;  // dims gr*16 + 4i + 2hh + {0,1}
          const uint32_t lo01 = place2(x0[i], x1[i], psel(2 * hh, 4 + 2 * hh));
          const uint32_t hi01 = place2(x0[i], x1[i], psel(2 * hh + 1, 5 + 2 * hh));
          const uint32_t lo89 = place2(x8[i], x9[i], psel(2 * hh, 4 + 2 * hh));
          const uint32_t hi89 = place2(x8[i], x9[i], psel(2 * hh + 1, 5 + 2 * hh));
#pragma unroll
          for (int u = 0; u < NT; ++u)
            if (dt[u]) mma<BF>(on[u][b], lo01, hi01, lo89, hi89, pb0[u], pb1[u]);
        }
      }
    }
  };
  // ---- chunk flush (this warp's columns only): row sums across lanes, partial (m, l, o) straight to global, reset
  auto flush = [&](int ci) {
    float sa[NT], sb[NT];
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      sa[u] = la[u], sb[u] = lb[u];
      if (!dt[u]) continue;
#pragma unroll
      for (int sh = 4; sh < 32; sh <<= 1) {
        sa[u] += __shfl_xor_sync(0xffffffffu, sa[u], sh);
        sb[u] += __shfl_xor_sync(0xffffffffu, sb[u], sh);
      }
    }
    const size_t pbase = ((size_t)s * kHkv + g) * p.max_chunks + ci;
    // coalesced partial store: this warp's columns go through its own smem stage [24][128] fp32, then float4 rows
    float* st = reinterpret_cast<float*>(smem + kQsBytes + w * kStageBytes);
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      if (!dt[u]) continue;
      const int cu = 8 * u + ca, cv = 8 * u + cb;
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        st[cu * kD + dimn(b, gr)] = on[u][b][0], st[cu * kD + dimn(b, gr) + 1] = on[u][b][2];
        st[cv * kD + dimn(b, gr)] = on[u][b][1], st[cv * kD + dimn(b, gr) + 1] = on[u][b][3];
      }
    }
    __syncwarp();
    float4* dst = reinterpret_cast<float4*>(p.o_part + pbase * NCs * kD);
    const float4* sv = reinterpret_cast<const float4*>(st);
    for (int i = lane; i < NC * (kD / 4); i += 32) dst[i] = sv[i];
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      if (gr == 0 && 8 * u + ca < NC) p.ml_part[pbase * NCs + 8 * u + ca] = make_float2(ma[u], sa[u]);
      if (gr == 0 && 8 * u + cb < NC) p.ml_part[pbase * NCs + 8 * u + cb] = make_float2(mb[u], sb[u]);
    }
    __syncwarp();
#pragma unroll
    for (int u = 0; u < NT; ++u) {
#pragma unroll
      for (int i = 0; i < 8; ++i) on[u][i][0] = on[u][i][1] = on[u][i][2] = on[u][i][3] = 0.f;
      ma[u] = mb[u] = -INFINITY, la[u] = lb[u] = 0.f;
    }
  };
  // ---- driver: warp w owns chunks i0 + w, i0 + w + 4, ...; one step of register prefetch across chunks
  seek(i0 + w);
  if (lci < i1) {
    load_k();
    load_v();
    advance();
  }
  for (int ci = i0 + w; ci < i1; ci += kNW) {
    int st_lo, nst;
    wsteps(ci, st_lo, nst);
    for (int it = 0; it < nst; ++it) {
      qk();
      if (lci < i1) load_k();  // K registers are free: prefetch the next step's K
      const int kt = st_lo + it * 16;
      if ((kt + 15 <= pos0) && (!windowed || kt > qlast - window)) rest(kt, std::false_type{});
      else rest(kt, std::true_type{});
      if (lci < i1) {
        load_v();
        advance();
      }
    }
    flush(ci);
  }
}

// 16-row single-pass global attention (9 <= R <= 16, up to 48 columns = 6 n8 tiles): 4 warps/CTA organized as 2
// independent warp-pairs (pair = w >> 1, half = w & 1). Each pair processes one chunk at a time (ci = i0 + pair + 2j):
// half=0 holds tiles 0..2 (cols 0..23, rows 0..7) and prefetches K from HBM; half=1 holds tiles 3..5 (cols 24..47,
// rows 8..15) and prefetches V from HBM. K and V are read from HBM ONCE per 16-key step and shared via a 2-stage
// conflict-free smem pipe synchronized with per-pair named barriers (bar.sync pair+1, 64). Every column runs the exact
// same Design A QK^T / softmax / P.V instruction sequence as decode_attn_kernel<KV, false>, guaranteeing F1 bit
// invariance with 1-row decode.
template <int KV>
__global__ void __launch_bounds__(kNW * 32, 2) decode_attn_kernel16(Params p) {
  constexpr int NT = 3;
  constexpr bool F8 = KV == KV_E4M3;
  constexpr bool BF = !F8;
  constexpr int kPrefN = F8 ? 4 : 8;
  extern __shared__ __align__(16) uint8_t smem[];
  const int s = blockIdx.z, g = blockIdx.y;
  const int lane = threadIdx.x & 31, gr = lane >> 2, tq = lane & 3, w = threadIdx.x >> 5;
  const int pair = w >> 1, half = w & 1;
  const int col_base = half * 24;
  const int R = p.nrows ? p.nrows[s] : p.R, NC = kG * R, NCs = kG * p.R;
  bool dt[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u) dt[u] = (col_base + 8 * u) < NC;
  const int pos0 = p.ctx[s], qlast = pos0 + R - 1;
  const bool windowed = p.window < (1 << 30);
  const int k_lo = windowed ? max(0, pos0 - p.window + 1) : 0;
  const int cs = p.chunk_shift[s];
  const int c_first = k_lo >> cs, nch = (qlast >> cs) - c_first + 1;
  const int i0 = (int)((long long)blockIdx.x * nch / gridDim.x), i1 = (int)((long long)(blockIdx.x + 1) * nch / gridDim.x);
  if (i0 >= i1) return;
  const int row0 = p.row0[s];
  const int rs = p.row_stride;
  const int pmask = (1 << p.page_shift) - 1;
  auto wsteps = [&](int ci, int& st, int& n) {
    const int clo = (c_first + ci) << cs;
    const int a_lo = max(clo, k_lo), a_hi = min(clo + (1 << cs) - 1, qlast);
    st = a_lo & ~15;
    n = a_hi >= a_lo ? ((a_hi - st) >> 4) + 1 : 0;
  };
  auto ptr_for = [&](int st, const uint8_t*& kq, const uint8_t*& vq) {
    const int page = p.page_table[(size_t)s * p.pt_stride + (st >> p.page_shift)];
    const long long off = (long long)page * p.page_stride + (long long)g * p.head_stride + (long long)(st & pmask) * rs;
    kq = p.k + off + gr * rs + (F8 ? tq * 32 : tq * 16);
    vq = p.v + off + 2 * tq * rs + gr * 16;
  };
  auto bar_pair = [&](int pr) {
    asm volatile("bar.sync %0, 64;" ::"r"(pr + 1) : "memory");
  };
  // ---- Q fragments of this half's 3 tiles -> smem qs_n[6][8][32]
  const uint16_t* qbase = p.q + ((size_t)row0 * kHq + g * kG) * kD;
  auto qrow = [&](int col) -> const uint16_t* {
    const int cc = col < NC ? col : 0;
    return qbase + ((size_t)(cc / kG) * kHq + cc % kG) * kD;
  };
  uint2* qs_n = reinterpret_cast<uint2*>(smem);
#pragma unroll
  for (int u = 0; u < NT; ++u) {
    const uint16_t* r = qrow(col_base + 8 * u + gr);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b00, b01, b10, b11;
      if constexpr (!F8) {
        const uint4 v = *reinterpret_cast<const uint4*>(r + j * 32 + tq * 8);
        b00 = v.x, b01 = v.y, b10 = v.z, b11 = v.w;
      } else {
        const uint4 v = *reinterpret_cast<const uint4*>(r + tq * 32 + j * 8);
        b00 = bf2_to_h2(v.x), b01 = bf2_to_h2(v.y);
        b10 = bf2_to_h2(v.z), b11 = bf2_to_h2(v.w);
      }
      const int tu = 3 * half + u;
      if (pair == 0) qs_n[(tu * 8 + 2 * j) * 32 + lane] = make_uint2(b00, b01);
      else qs_n[(tu * 8 + 2 * j + 1) * 32 + lane] = make_uint2(b10, b11);
    }
  }
  __syncthreads();

  uint4* kv_sm = reinterpret_cast<uint4*>(smem + kQsBytes16);
  const int ca = 2 * tq, cb = ca + 1;
  const int extent = p.extent, window = p.window;
  const float sc = p.sc;
  const int src = tq * 8 + (gr >> 1);
  const uint32_t psl = (gr & 1) ? 0x7632u : 0x5410u;

  float on[NT][8][4];
  float ma[NT], mb[NT], la[NT], lb[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u) {
#pragma unroll
    for (int i = 0; i < 8; ++i) on[u][i][0] = on[u][i][1] = on[u][i][2] = on[u][i][3] = 0.f;
    ma[u] = mb[u] = -INFINITY, la[u] = lb[u] = 0.f;
  }

  const uint8_t* kp = nullptr;
  const uint8_t* vp = nullptr;
  uint4 pref[kPrefN];
  auto load_half = [&]() {
    if (half == 0) {
#pragma unroll
      for (int j = 0; j < (F8 ? 2 : 4); ++j) {
        pref[j] = ldg16(kp + j * (F8 ? 16 : 64));
        pref[(F8 ? 2 : 4) + j] = ldg16(kp + 8 * rs + j * (F8 ? 16 : 64));
      }
    } else {
#pragma unroll
      for (int i = 0; i < (F8 ? 1 : 2); ++i) {
        pref[i] = ldg16(vp + i * 128);
        pref[(F8 ? 1 : 2) + i] = ldg16(vp + rs + i * 128);
        pref[(F8 ? 2 : 4) + i] = ldg16(vp + 8 * rs + i * 128);
        pref[(F8 ? 3 : 6) + i] = ldg16(vp + 9 * rs + i * 128);
      }
    }
  };
  auto store_half = [&](int stg) {
    uint4* dst = kv_sm + (((pair * 2 + stg) * 2 + half) * 8) * 32;
#pragma unroll
    for (int m = 0; m < kPrefN; ++m) dst[m * 32 + lane] = pref[m];
  };

  int lci = i0 + pair, lj = 0, lnst = 0, lst = 0;
  auto seek = [&](int ci) {
    for (lci = ci; lci < i1; lci += kNP16) {
      wsteps(lci, lst, lnst);
      if (lnst > 0) break;
    }
    lj = 0;
    if (lci < i1) ptr_for(lst, kp, vp);
  };
  auto advance = [&]() {
    if (++lj < lnst) {
      const int key = lst + 16 * lj;
      if ((key & pmask) == 0) ptr_for(key, kp, vp);
      else kp += 16 * rs, vp += 16 * rs;
    } else {
      seek(lci + kNP16);
    }
  };
  auto dimn = [&](int b, int rowg) { return F8 ? rowg * 16 + 2 * b : (b >> 2) * 64 + rowg * 8 + 2 * (b & 3); };

  float sn[NT][4];
  auto qk = [&](int cur) {
    const uint4* ks = kv_sm + (((pair * 2 + cur) * 2 + 0) * 8) * 32;
#pragma unroll
    for (int u = 0; u < NT; ++u) sn[u][0] = sn[u][1] = sn[u][2] = sn[u][3] = 0.f;
    if constexpr (!F8) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint4 x = ks[j * 32 + lane];
        const uint4 y = ks[(4 + j) * 32 + lane];
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int kk = 2 * j + sub;
          const uint32_t a0 = sub ? x.z : x.x, a2 = sub ? x.w : x.y;
          const uint32_t a1 = sub ? y.z : y.x, a3 = sub ? y.w : y.y;
#pragma unroll
          for (int u = 0; u < NT; ++u) {
            if (!dt[u]) continue;
            uint32_t b0, b1;
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1)
                         : "r"((unsigned)__cvta_generic_to_shared(qs_n + ((3 * half + u) * 8 + kk) * 32 + lane)));
            mma<BF>(sn[u], a0, a1, a2, a3, b0, b1);
          }
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint4 x = ks[j * 32 + lane];
        const uint4 y = ks[(2 + j) * 32 + lane];
        const uint32_t xs[4] = {x.x, x.y, x.z, x.w};
        const uint32_t ys[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
          const int kk = 4 * j + sub;
          const uint32_t a0 = place2(xs[sub], xs[sub], psel(0, 1)), a2 = place2(xs[sub], xs[sub], psel(2, 3));
          const uint32_t a1 = place2(ys[sub], ys[sub], psel(0, 1)), a3 = place2(ys[sub], ys[sub], psel(2, 3));
#pragma unroll
          for (int u = 0; u < NT; ++u) {
            if (!dt[u]) continue;
            uint32_t b0, b1;
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1)
                         : "r"((unsigned)__cvta_generic_to_shared(qs_n + ((3 * half + u) * 8 + kk) * 32 + lane)));
            mma<BF>(sn[u], a0, a1, a2, a3, b0, b1);
          }
        }
      }
    }
  };

  auto rest = [&](int kt, int cur, auto edge_c) {
    constexpr bool EDGE = decltype(edge_c)::value;
    if (!dt[0]) return;
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      if (!dt[u]) continue;
      const float ua = ma[u] == -INFINITY ? 0.f : ma[u], ub = mb[u] == -INFINITY ? 0.f : mb[u];
      sn[u][0] = fmaf(sn[u][0], sc, -ua), sn[u][1] = fmaf(sn[u][1], sc, -ub);
      sn[u][2] = fmaf(sn[u][2], sc, -ua), sn[u][3] = fmaf(sn[u][3], sc, -ub);
    }
    const bool near = pos0 - kt < extent + 15;
    const bool full = (kt + 15 <= pos0) && (!windowed || kt > qlast - window);
    if (near || !full) {
      auto cpos = [&](int col) { return col < NC ? pos0 + col / kG : -1; };
      auto reloff = [&](int col) -> int {
        const int cc = col < NC ? col : 0;
        return (int)((((size_t)row0 + cc / kG) * kHq + g * kG + cc % kG) * extent);
      };
      auto fix = [&](float& v, int pc, int key, int roff) {
        const int d = pc - key;
        if (pc >= 0 && d >= 0 && d < extent) v = fmaf(bf2f(__ldg(p.rel + roff + d)), kLog2e, v);
        const bool vis = pc >= 0 && d >= 0 && (!windowed || d < window);
        v = vis ? v : -INFINITY;
      };
      const int k0 = kt + gr, k1 = k0 + 8;
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        if (!dt[u]) continue;
        const int c0 = col_base + 8 * u;
        const int pa = cpos(c0 + ca), pb = cpos(c0 + cb), ra = reloff(c0 + ca), rb = reloff(c0 + cb);
        fix(sn[u][0], pa, k0, ra), fix(sn[u][1], pb, k0, rb), fix(sn[u][2], pa, k1, ra), fix(sn[u][3], pb, k1, rb);
      }
    }
    {
      float ta[NT], tb[NT];
      bool trig = false;
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        ta[u] = ma[u] == -INFINITY ? -INFINITY : kTau, tb[u] = mb[u] == -INFINITY ? -INFINITY : kTau;
        trig = trig || (dt[u] && (sn[u][0] > ta[u] || sn[u][2] > ta[u] || sn[u][1] > tb[u] || sn[u][3] > tb[u]));
      }
      if (__any_sync(0xffffffffu, trig)) {
#pragma unroll
        for (int u = 0; u < NT; ++u) {
          if (!dt[u]) continue;
          float xa = fmaxf(sn[u][0], sn[u][2]), xb = fmaxf(sn[u][1], sn[u][3]);
#pragma unroll
          for (int sh = 4; sh < 32; sh <<= 1) {
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, sh));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, sh));
          }
          if (xa > ta[u]) {
            const float f = ma[u] == -INFINITY ? 0.f : ex2(-xa);
            ma[u] = (ma[u] == -INFINITY ? 0.f : ma[u]) + xa, la[u] *= f, sn[u][0] -= xa, sn[u][2] -= xa;
#pragma unroll
            for (int i = 0; i < 8; ++i) on[u][i][0] *= f, on[u][i][2] *= f;
          }
          if (xb > tb[u]) {
            const float f = mb[u] == -INFINITY ? 0.f : ex2(-xb);
            mb[u] = (mb[u] == -INFINITY ? 0.f : mb[u]) + xb, lb[u] *= f, sn[u][1] -= xb, sn[u][3] -= xb;
#pragma unroll
            for (int i = 0; i < 8; ++i) on[u][i][1] *= f, on[u][i][3] *= f;
          }
        }
      }
    }
    uint32_t pb0[NT], pb1[NT];
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      pb0[u] = pb1[u] = 0;
      if (!dt[u]) continue;
      const float pn0 = ex2(sn[u][0]), pn1 = ex2(sn[u][1]), pn2 = ex2(sn[u][2]), pn3 = ex2(sn[u][3]);
      la[u] += pn0 + pn2, lb[u] += pn1 + pn3;
      const uint32_t plo = pack2<BF>(pn0, pn1), phi = pack2<BF>(pn2, pn3);
      const uint32_t alo = __shfl_sync(0xffffffffu, plo, src), blo = __shfl_sync(0xffffffffu, plo, src + 4);
      const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src), bhi = __shfl_sync(0xffffffffu, phi, src + 4);
      pb0[u] = __byte_perm(alo, blo, psl), pb1[u] = __byte_perm(ahi, bhi, psl);
    }
    const uint4* vs = kv_sm + (((pair * 2 + cur) * 2 + 1) * 8) * 32;
    const int kv0 = kt + 2 * tq;
    const bool z0 = EDGE && (kv0 < k_lo || kv0 > qlast), z1 = EDGE && (kv0 + 1 < k_lo || kv0 + 1 > qlast);
    const bool z8 = EDGE && (kv0 + 8 < k_lo || kv0 + 8 > qlast), z9 = EDGE && (kv0 + 9 < k_lo || kv0 + 9 > qlast);
    const uint4 zz = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (!F8) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 r0 = z0 ? zz : vs[(0 + i) * 32 + lane];
        const uint4 r1 = z1 ? zz : vs[(2 + i) * 32 + lane];
        const uint4 r8 = z8 ? zz : vs[(4 + i) * 32 + lane];
        const uint4 r9 = z9 ? zz : vs[(6 + i) * 32 + lane];
        const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w};
        const uint32_t x1[4] = {r1.x, r1.y, r1.z, r1.w};
        const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w};
        const uint32_t x9[4] = {r9.x, r9.y, r9.z, r9.w};
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int b = 4 * i + jj;
          const uint32_t lo01 = __byte_perm(x0[jj], x1[jj], 0x5410u), hi01 = __byte_perm(x0[jj], x1[jj], 0x7632u);
          const uint32_t lo89 = __byte_perm(x8[jj], x9[jj], 0x5410u), hi89 = __byte_perm(x8[jj], x9[jj], 0x7632u);
#pragma unroll
          for (int u = 0; u < NT; ++u)
            if (dt[u]) mma<BF>(on[u][b], lo01, hi01, lo89, hi89, pb0[u], pb1[u]);
        }
      }
    } else {
      const uint4 r0 = z0 ? zz : vs[0 * 32 + lane];
      const uint4 r1 = z1 ? zz : vs[1 * 32 + lane];
      const uint4 r8 = z8 ? zz : vs[2 * 32 + lane];
      const uint4 r9 = z9 ? zz : vs[3 * 32 + lane];
      const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w};
      const uint32_t x1[4] = {r1.x, r1.y, r1.z, r1.w};
      const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w};
      const uint32_t x9[4] = {r9.x, r9.y, r9.z, r9.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int b = 2 * i + hh;
          const uint32_t lo01 = place2(x0[i], x1[i], psel(2 * hh, 4 + 2 * hh));
          const uint32_t hi01 = place2(x0[i], x1[i], psel(2 * hh + 1, 5 + 2 * hh));
          const uint32_t lo89 = place2(x8[i], x9[i], psel(2 * hh, 4 + 2 * hh));
          const uint32_t hi89 = place2(x8[i], x9[i], psel(2 * hh + 1, 5 + 2 * hh));
#pragma unroll
          for (int u = 0; u < NT; ++u)
            if (dt[u]) mma<BF>(on[u][b], lo01, hi01, lo89, hi89, pb0[u], pb1[u]);
        }
      }
    }
  };

  auto flush = [&](int ci) {
    float sa[NT], sb[NT];
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      sa[u] = la[u], sb[u] = lb[u];
      if (!dt[u]) continue;
#pragma unroll
      for (int sh = 4; sh < 32; sh <<= 1) {
        sa[u] += __shfl_xor_sync(0xffffffffu, sa[u], sh);
        sb[u] += __shfl_xor_sync(0xffffffffu, sb[u], sh);
      }
    }
    const size_t pbase = ((size_t)s * kHkv + g) * p.max_chunks + ci;
    float* st = reinterpret_cast<float*>(smem + kQsBytes16 + kKvPipeBytes16 + pair * kStageBytes);
    const int nc_half = min(24, max(0, NC - col_base));
    auto write_half = [&]() {
      if (nc_half <= 0) return;
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        if (!dt[u]) continue;
        const int cu = 8 * u + ca, cv = 8 * u + cb;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
          st[cu * kD + dimn(b, gr)] = on[u][b][0], st[cu * kD + dimn(b, gr) + 1] = on[u][b][2];
          st[cv * kD + dimn(b, gr)] = on[u][b][1], st[cv * kD + dimn(b, gr) + 1] = on[u][b][3];
        }
      }
      __syncwarp();
      float4* dst = reinterpret_cast<float4*>(p.o_part + (pbase * NCs + col_base) * kD);
      const float4* sv = reinterpret_cast<const float4*>(st);
      for (int i = lane; i < nc_half * (kD / 4); i += 32) dst[i] = sv[i];
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        const int gca = col_base + 8 * u + ca, gcb = col_base + 8 * u + cb;
        if (gr == 0 && gca < NC) p.ml_part[pbase * NCs + gca] = make_float2(ma[u], sa[u]);
        if (gr == 0 && gcb < NC) p.ml_part[pbase * NCs + gcb] = make_float2(mb[u], sb[u]);
      }
      __syncwarp();
    };
    if (half == 0) write_half();
    bar_pair(pair);
    if (half == 1) write_half();
    bar_pair(pair);
#pragma unroll
    for (int u = 0; u < NT; ++u) {
#pragma unroll
      for (int i = 0; i < 8; ++i) on[u][i][0] = on[u][i][1] = on[u][i][2] = on[u][i][3] = 0.f;
      ma[u] = mb[u] = -INFINITY, la[u] = lb[u] = 0.f;
    }
  };

  int stage = 0;
  bool have_pref = false;
  seek(i0 + pair);
  if (lci < i1) {
    load_half();
    advance();
    store_half(stage);
    bar_pair(pair);
    if (lci < i1) {
      load_half();
      advance();
      have_pref = true;
    }
  }
  for (int ci = i0 + pair; ci < i1; ci += kNP16) {
    int st_lo, nst;
    wsteps(ci, st_lo, nst);
    for (int it = 0; it < nst; ++it) {
      const int cur = stage;
      const int nxt = stage ^ 1;
      qk(cur);
      const int kt = st_lo + it * 16;
      if ((kt + 15 <= pos0) && (!windowed || kt > qlast - window)) rest(kt, cur, std::false_type{});
      else rest(kt, cur, std::true_type{});
      if (have_pref) {
        store_half(nxt);
        bar_pair(pair);
        if (lci < i1) {
          load_half();
          advance();
        } else {
          have_pref = false;
        }
        stage = nxt;
      }
    }
    flush(ci);
  }
}

// Combine of one output element (session s, KV head g, column col, dim d): chunks folded in increasing order, then
// config B's gate-then-sink epilogue and one bf16 rounding.
__device__ __forceinline__ void combine_one(const Params& p, int s, int g, int col, int d) {
  const int R = p.nrows ? p.nrows[s] : p.R, NCs = kG * p.R;
  const int pos0 = p.ctx[s], qlast = pos0 + R - 1;
  const bool windowed = p.window < (1 << 30);
  const int k_lo = windowed ? max(0, pos0 - p.window + 1) : 0;
  const int cs = p.chunk_shift[s];
  const int nch = (qlast >> cs) - (k_lo >> cs) + 1;
  const size_t pbase = ((size_t)s * kHkv + g) * p.max_chunks;
  float M = -INFINITY;
  for (int i = 0; i < nch; ++i) M = fmaxf(M, __ldcg(&p.ml_part[(pbase + i) * NCs + col]).x);
  float L = 0.f, O = 0.f;
  for (int i = 0; i < nch; ++i) {
    const float2 ml = __ldcg(&p.ml_part[(pbase + i) * NCs + col]);
    const float f = ml.x == -INFINITY ? 0.f : ex2(ml.x - M);
    L = fmaf(ml.y, f, L);
    O = fmaf(__ldcg(&p.o_part[((pbase + i) * NCs + col) * kD + d]), f, O);
  }
  const int t = col / kG, h = g * kG + col % kG;
  const size_t row = (size_t)p.row0[s] + t;
  const float o = L > 0.f ? O / L * p.vscale : 0.f;
  const float lse = L > 0.f ? (M + __log2f(L)) * kLn2 : -INFINITY;
  const size_t oi = row * (kHq * kD) + (size_t)h * kD + d;
  float r = o * sigmoid_cfgb(bf2f(p.gate[oi]));
  r = r * sigmoid_cfgb(lse - bf2f(p.sink[row * kHq + h]));
  p.out[oi] = f2bf(r);
  if (p.out32) p.out32[oi] = o;
}
// Combine: one CTA per (column, KV head, session), 128 threads (dims).
__global__ void __launch_bounds__(128) combine_kernel(Params p) {
  const int col = blockIdx.x, g = blockIdx.y, s = blockIdx.z;
  const int R = p.nrows ? p.nrows[s] : p.R;
  if (col >= kG * R) return;
  combine_one(p, s, g, col, threadIdx.x);
}

// ------------------------------------------------------------------------------------------- design B (SWA)
// Persistent workers over fixed logical chunks. grid (W workers, 8 KV heads, n sessions); block NWC*32 threads,
// NWC = ceil(3R/8) warps, warp w owns columns 8w..8w+7 (one n8 tile, K as the A operand) for ALL keys of the worker.
// Worker j of session s streams chunks [j*nch/W, (j+1)*nch/W) as one continuous sequence of 16-key steps through a
// shared S-stage cp.async ring; at every chunk boundary each warp writes its columns' partial (m, l, o) for that
// chunk and resets. Partials (and so bits) depend only on the chunk, never on W, n or the other rows.
// bf16 stage: K and V 16 rows x 272 B (ldmatrix, conflict-free); FP8 stage: 16 rows x 144 B.
constexpr int kRS16 = 272, kRS8 = 144;
template <int KV>
constexpr int ring_stage_bytes() { return 16 * (KV == KV_BF16 ? kRS16 : kRS8) * 2; }

template <int KV, int NWC, int S>
__global__ void __launch_bounds__(NWC * 32) decode_attn_ring_kernel(Params p) {
  constexpr bool F8 = KV == KV_E4M3;
  constexpr bool BF = !F8;
  constexpr int RS = F8 ? kRS8 : kRS16;
  constexpr int STAGE = 16 * RS * 2;
  constexpr int ROWB = F8 ? 128 : 256;     // bytes of one key-head row
  constexpr int CPR = ROWB / 16;           // 16-byte chunks per row
  constexpr int NT = NWC * 32;
  extern __shared__ __align__(16) uint8_t smem[];
  const int s = blockIdx.z, g = blockIdx.y, wk = blockIdx.x, W = gridDim.x;
  const int lane = threadIdx.x & 31, gr = lane >> 2, tq = lane & 3, w = threadIdx.x >> 5;
  const int R = p.nrows ? p.nrows[s] : p.R, NC = kG * R, NCs = kG * p.R;  // ragged rows; partial stride uses p.R = max
  const int pos0 = p.ctx[s], qlast = pos0 + R - 1;
  const bool windowed = p.window < (1 << 30);
  const int k_lo = windowed ? max(0, pos0 - p.window + 1) : 0;
  const int cs = p.chunk_shift[s];
  const int c_first = k_lo >> cs, nch = (qlast >> cs) - c_first + 1;
  const int i0 = (int)((long long)wk * nch / W), i1 = (int)((long long)(wk + 1) * nch / W);
  if (i0 >= i1) return;
  const int key_lo = max((c_first + i0) << cs, k_lo), key_hi = min(((c_first + i1) << cs) - 1, qlast);
  const int st0 = key_lo & ~15;
  const int nst = ((key_hi - st0) >> 4) + 1;
  const int row0 = p.row0[s];
  const int pmask = (1 << p.page_shift) - 1;

  // ---- producer: step `it` -> smem stage it % S. Per-thread 16-byte chunk offsets precomputed once; the step base
  // advances by 16 rows, with a page-table lookup only when a page starts.
  constexpr int NCH = 16 * CPR;                 // chunks per K (or V) stage
  constexpr int PER = (NCH + NT - 1) / NT;      // chunks per thread
  // Head-major layout (row stride == ROWB, asserted on the host): the 16 rows of a step are one contiguous block, so
  // chunk c sits at byte 16c; thread t takes chunks t + i*NT (NT % CPR == 0): offsets are base + immediates.
  const int soff0 = (threadIdx.x / CPR) * RS + (threadIdx.x % CPR) * 16, goff0 = threadIdx.x * 16;
  const long long vdelta = p.v - p.k;
  int pk = st0;                                 // key of the next step to issue
  const uint8_t* gbase = nullptr;
  const int row_t = threadIdx.x / CPR;          // row of this thread's first chunk; + i * (NT / CPR) for chunk i
  auto issue = [&](int it) {
    if (it < nst) {
      if (gbase == nullptr || (pk & pmask) == 0) {
        const int page = __ldg(p.page_table + (size_t)s * p.pt_stride + (pk >> p.page_shift));
        gbase = p.k + (long long)page * p.page_stride + (long long)g * p.head_stride + (long long)(pk & pmask) * ROWB;
      }
      const uint32_t ks = (uint32_t)__cvta_generic_to_shared(smem + (it % S) * STAGE) + soff0;
      const uint8_t* gk = gbase + goff0;
      // masked loads (review D3): rows of keys outside [k_lo, qlast] are zero-filled, never read
      const bool edge = pk < k_lo || pk + 15 > qlast;
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        if (i * NT + (int)threadIdx.x < NCH) {
          const int key = pk + row_t + i * (NT / CPR);
          const int nb = (!edge || (key >= k_lo && key <= qlast)) ? 16 : 0;
          cp16z(ks + i * (NT / CPR) * RS, gk + i * NT * 16, nb);
          cp16z(ks + 16 * RS + i * (NT / CPR) * RS, gk + i * NT * 16 + vdelta, nb);
        }
      }
      pk += 16, gbase += 16 * ROWB;
    }
    cp_commit();
  };
#pragma unroll
  for (int i = 0; i < S - 1; ++i) issue(i);

  // ---- this warp's columns: n8 tile, columns 8w + {0..7}; thread column for Q = 8w + gr; for scores 8w + 2tq (+1)
  const int cq = 8 * w + gr;
  const uint16_t* qr = p.q + (((size_t)row0 + (cq < NC ? cq : 0) / kG) * kHq + g * kG + (cq < NC ? cq : 0) % kG) * kD;
  uint32_t qf[8][2];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    if constexpr (BF) {  // natural dims: b0 = dims kk*16 + 2tq,+1 ; b1 = +8
      qf[kk][0] = *reinterpret_cast<const uint32_t*>(qr + kk * 16 + 2 * tq);
      qf[kk][1] = *reinterpret_cast<const uint32_t*>(qr + kk * 16 + 8 + 2 * tq);
    } else {             // maya order: b0 = dims kk*16 + 4tq + {0,1}, b1 = + {2,3}
      const uint2 v = *reinterpret_cast<const uint2*>(qr + kk * 16 + 4 * tq);
      qf[kk][0] = bf2_to_h2(v.x), qf[kk][1] = bf2_to_h2(v.y);
    }
  }
  const int ca = 8 * w + 2 * tq, cb = ca + 1;
  const int pa = ca < NC ? pos0 + ca / kG : -1, pb = cb < NC ? pos0 + cb / kG : -1;
  const int ra_off = (int)((((size_t)row0 + (ca < NC ? ca : 0) / kG) * kHq + g * kG + (ca < NC ? ca : 0) % kG) * p.extent);
  const int rb_off = (int)((((size_t)row0 + (cb < NC ? cb : 0) / kG) * kHq + g * kG + (cb < NC ? cb : 0) % kG) * p.extent);
  const int extent = p.extent, window = p.window;
  const float sc = p.sc;
  const int src = tq * 8 + (gr >> 1);
  const uint32_t psl = (gr & 1) ? 0x7632u : 0x5410u;

  float o[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  int chunk = (st0 >> cs);  // chunk of the current step (steps never straddle chunks)

  auto flush = [&](int c) {
    float sa = la, sb = lb;
#pragma unroll
    for (int sh = 4; sh < 32; sh <<= 1) {
      sa += __shfl_xor_sync(0xffffffffu, sa, sh);
      sb += __shfl_xor_sync(0xffffffffu, sb, sh);
    }
    const size_t pbase = ((size_t)s * kHkv + g) * p.max_chunks + (c - c_first);
    {
      // coalesced: warp's 8 columns via its smem stage [8][128] (after the stages), then float4 rows
      float* st = reinterpret_cast<float*>(smem + S * STAGE + w * 8 * kD * 4);
      const int la0 = 2 * tq, lb0 = la0 + 1;
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const int d0 = BF ? b * 16 + gr : gr * 16 + 2 * b, d1 = BF ? d0 + 8 : d0 + 1;
        st[la0 * kD + d0] = o[b][0], st[la0 * kD + d1] = o[b][2];
        st[lb0 * kD + d0] = o[b][1], st[lb0 * kD + d1] = o[b][3];
      }
      __syncwarp();
      const int ncw = min(8, NC - 8 * w);
      float4* dst = reinterpret_cast<float4*>(p.o_part + (pbase * NCs + 8 * w) * kD);
      const float4* sv = reinterpret_cast<const float4*>(st);
      for (int i = lane; i < ncw * (kD / 4); i += 32) dst[i] = sv[i];
      if (gr == 0 && ca < NC) p.ml_part[pbase * NCs + ca] = make_float2(ma, sa);
      if (gr == 0 && cb < NC) p.ml_part[pbase * NCs + cb] = make_float2(mb, sb);
      __syncwarp();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    ma = mb = -INFINITY, la = lb = 0.f;
  };

  for (int it = 0; it < nst; ++it) {
    const int kt = st0 + it * 16;
    if ((kt >> cs) != chunk) { flush(chunk); chunk = kt >> cs; }
    cp_wait<S - 2>();
    __syncthreads();
    issue(it + S - 1);
    const uint8_t* ks = smem + (it % S) * STAGE;
    const uint8_t* vs = ks + 16 * RS;
    // ---- QK^T: A = K (16 keys x 16 dims), B = q (16 dims x 8 columns)
    float sa4[4] = {0.f, 0.f, 0.f, 0.f}, sb4[4] = {0.f, 0.f, 0.f, 0.f};
    if constexpr (BF) {
      const uint8_t* kp = ks + ((lane & 7) + ((lane >> 3) & 1) * 8) * RS + (lane >> 4) * 16;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint32_t a[4];
        ldsm4(a, kp + kk * 32);
        mma<BF>((kk & 1) ? sb4 : sa4, a[0], a[1], a[2], a[3], qf[kk][0], qf[kk][1]);
      }
    } else {
      const uint8_t* k0 = ks + gr * RS + tq * 4;
      const uint8_t* k1 = k0 + 8 * RS;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(k0 + kk * 16);
        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(k1 + kk * 16);
        mma<BF>((kk & 1) ? sb4 : sa4, place2(w0, w0, psel(0, 1)), place2(w1, w1, psel(0, 1)),
                place2(w0, w0, psel(2, 3)), place2(w1, w1, psel(2, 3)), qf[kk][0], qf[kk][1]);
      }
    }
    // sn[0] (key kt+gr, col ca), [1] (kt+gr, cb), [2] (kt+gr+8, ca), [3] (kt+gr+8, cb)
    float sn[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) sn[e] = (sa4[e] + sb4[e]) * sc;
    const bool near = pos0 - kt < extent + 15;
    const bool full = (kt + 15 <= pos0) && (!windowed || kt > qlast - window);
    if (near || !full) {
      auto fix = [&](float& v, int pc, int key, int roff) {
        const int d = pc - key;
        if (pc >= 0 && d >= 0 && d < extent) v = fmaf(bf2f(__ldg(p.rel + roff + d)), kLog2e, v);
        v = (pc >= 0 && d >= 0 && (!windowed || d < window)) ? v : -INFINITY;
      };
      const int k0 = kt + gr, k1 = k0 + 8;
      fix(sn[0], pa, k0, ra_off), fix(sn[1], pb, k0, rb_off), fix(sn[2], pa, k1, ra_off), fix(sn[3], pb, k1, rb_off);
    }
    {
      const float ta = ma + kTau, tb = mb + kTau;
      const bool trig = sn[0] > ta || sn[2] > ta || sn[1] > tb || sn[3] > tb;
      if (__any_sync(0xffffffffu, trig)) {
        float xa = fmaxf(sn[0], sn[2]), xb = fmaxf(sn[1], sn[3]);
#pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
          xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, sh));
          xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, sh));
        }
        if (xa > ta) {
          const float f = ma == -INFINITY ? 0.f : ex2(ma - xa);
          ma = xa, la *= f;
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i][0] *= f, o[i][2] *= f;
        }
        if (xb > tb) {
          const float f = mb == -INFINITY ? 0.f : ex2(mb - xb);
          mb = xb, lb *= f;
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i][1] *= f, o[i][3] *= f;
        }
      }
    }
    const float ua = ma == -INFINITY ? 0.f : ma, ub = mb == -INFINITY ? 0.f : mb;
    const float p0 = ex2(sn[0] - ua), p1 = ex2(sn[1] - ub), p2 = ex2(sn[2] - ua), p3 = ex2(sn[3] - ub);
    la += p0 + p2, lb += p1 + p3;
    const uint32_t plo = pack2<BF>(p0, p1), phi = pack2<BF>(p2, p3);
    const uint32_t alo = __shfl_sync(0xffffffffu, plo, src), blo = __shfl_sync(0xffffffffu, plo, src + 4);
    const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src), bhi = __shfl_sync(0xffffffffu, phi, src + 4);
    const uint32_t pb0 = __byte_perm(alo, blo, psl), pb1 = __byte_perm(ahi, bhi, psl);
    // ---- P.V: A = V^T (16 dims x 16 keys)
    if constexpr (BF) {
      const uint8_t* vp = vs + ((lane & 7) + ((lane >> 4) & 1) * 8) * RS + ((lane >> 3) & 1) * 16;
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        uint32_t a[4];
        ldsm4t(a, vp + b * 32);
        mma<BF>(o[b], a[0], a[1], a[2], a[3], pb0, pb1);
      }
    } else {
      const uint4 r0 = *reinterpret_cast<const uint4*>(vs + (2 * tq) * RS + gr * 16);
      const uint4 r1 = *reinterpret_cast<const uint4*>(vs + (2 * tq + 1) * RS + gr * 16);
      const uint4 r8 = *reinterpret_cast<const uint4*>(vs + (2 * tq + 8) * RS + gr * 16);
      const uint4 r9 = *reinterpret_cast<const uint4*>(vs + (2 * tq + 9) * RS + gr * 16);
      const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
      const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          mma<BF>(o[2 * i + hh], place2(x0[i], x1[i], psel(2 * hh, 4 + 2 * hh)), place2(x0[i], x1[i], psel(2 * hh + 1, 5 + 2 * hh)),
                  place2(x8[i], x9[i], psel(2 * hh, 4 + 2 * hh)), place2(x8[i], x9[i], psel(2 * hh + 1, 5 + 2 * hh)), pb0, pb1);
    }
  }
  cp_wait<0>();
  flush(chunk);
}

// ------------------------------------------------------------------------------------------- host side
// Every launch: grid x (workers per session and head) may exceed what a call needs (empty workers exit at once);
// Params::max_chunks may exceed the largest chunk count of the call; Params::R may exceed every session's nrows. So
// one captured graph can serve calls up to its captured bounds (re-capture only when a bound grows).
inline int smem_bytes() { return kQsBytes + kNW * kStageBytes; }
inline int smem_bytes16() { return kQsBytes16 + kKvPipeBytes16 + kNP16 * kStageBytes; }
template <int KV, int NWC, int S>
inline void* ring_ptr() { return (void*)decode_attn_ring_kernel<KV, NWC, S>; }
template <int KV, int S>
inline void* ring_pick_nw(int nw) {
  return nw == 1 ? ring_ptr<KV, 1, S>() : nw == 2 ? ring_ptr<KV, 2, S>() : nw == 3 ? ring_ptr<KV, 3, S>()
       : nw == 4 ? ring_ptr<KV, 4, S>() : nw == 5 ? ring_ptr<KV, 5, S>() : ring_ptr<KV, 6, S>();
}
template <int KV>
inline void* ring_pick_s(int nw, int S) {
  return S == 2 ? ring_pick_nw<KV, 2>(nw) : S == 3 ? ring_pick_nw<KV, 3>(nw) : S == 4 ? ring_pick_nw<KV, 4>(nw)
       : S == 6 ? ring_pick_nw<KV, 6>(nw) : ring_pick_nw<KV, 8>(nw);
}
inline int ring_nw(int R) { return (kG * R + 7) / 8; }
inline void* ring_pick(int kv, int R, int S) { return kv == KV_BF16 ? ring_pick_s<KV_BF16>(ring_nw(R), S) : ring_pick_s<KV_E4M3>(ring_nw(R), S); }
inline int ring_smem(int kv, int S, int nw = 3) {
  const int fw = nw > 3 ? nw : 3;
  return S * (kv == KV_BF16 ? ring_stage_bytes<KV_BF16>() : ring_stage_bytes<KV_E4M3>()) + fw * 8 * kD * 4;
}
inline cudaError_t ring_prepare(int kv, int R, int S) {
  return cudaFuncSetAttribute(ring_pick(kv, R, S), cudaFuncAttributeMaxDynamicSharedMemorySize, ring_smem(kv, S, ring_nw(R)));
}
inline int ring_ctas_per_sm(int kv, int R, int S) {
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ring_pick(kv, R, S), ring_nw(R) * 32, ring_smem(kv, S, ring_nw(R)));
  return occ;
}
// design B (SWA), then the combine. W workers per (session, head): physical choice only (bits do not depend on it).
// Needs head-major rows (row_stride = 128 * element bytes).
inline cudaError_t ring_launch(const Params& p, int kv, int n, int W, int S, cudaStream_t st) {
  if (p.row_stride != (kv == KV_BF16 ? 256 : 128)) return cudaErrorInvalidValue;
  void* args[] = {(void*)&p};
  const int nw = ring_nw(p.R);
  cudaError_t e = cudaLaunchKernel(ring_pick(kv, p.R, S), dim3(W, kHkv, n), dim3(nw * 32), args, ring_smem(kv, S, nw), st);
  if (e != cudaSuccess) return e;
  combine_kernel<<<dim3(kG * p.R, kHkv, n), kD, 0, st>>>(p);
  return cudaGetLastError();
}
inline void* pick(int kv, int R) {
  if (kG * R > 24) return kv == KV_BF16 ? (void*)decode_attn_kernel16<KV_BF16> : (void*)decode_attn_kernel16<KV_E4M3>;
  const bool wide = kG * R > 8;
  if (kv == KV_BF16) return wide ? (void*)decode_attn_kernel<KV_BF16, true> : (void*)decode_attn_kernel<KV_BF16, false>;
  return wide ? (void*)decode_attn_kernel<KV_E4M3, true> : (void*)decode_attn_kernel<KV_E4M3, false>;
}
inline cudaError_t prepare(int kv, int R) {
  return cudaFuncSetAttribute(pick(kv, R), cudaFuncAttributeMaxDynamicSharedMemorySize, kG * R > 24 ? smem_bytes16() : smem_bytes());
}
// design A (global; also usable for SWA), then the combine. grid_x = workers per (session, head) (physical choice).
inline cudaError_t launch(const Params& p, int kv, int n, int grid_x, cudaStream_t st) {
  void* args[] = {(void*)&p};
  const int sm = kG * p.R > 24 ? smem_bytes16() : smem_bytes();
  cudaError_t e = cudaLaunchKernel(pick(kv, p.R), dim3(grid_x, kHkv, n), dim3(kNW * 32), args, sm, st);
  if (e != cudaSuccess) return e;
  combine_kernel<<<dim3(kG * p.R, kHkv, n), kD, 0, st>>>(p);
  return cudaGetLastError();
}

}  // namespace da
