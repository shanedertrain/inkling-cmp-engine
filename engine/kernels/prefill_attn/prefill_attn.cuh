// Adapted from ingine native/src/kernels/prefill/attention.cu and native/src/kernels/prefill/common.cuh @ 306eaa8
// (Apache-2.0, Copyright 2026 satellitedown; see engine/NOTICE; common.cuh is itself adapted from fngine,
// Apache-2.0). Modified for inkling-cmp-engine:
//   - config B's numerics (vLLM models/inkling2/nvidia/ops/attention_sm90.py `_rel_attention_kernel` MODE_OUT_LSE
//     + ops/attention.py `_gate_sink_kernel` @ 6e9d1cce): log2-domain scores s = (q.k) * fp32(scale * log2e)
//     + bias * log2e, key blocks of 64 aligned to absolute key positions, ascending, one split; P rounded to BF16 for
//     P.V (BF16 mma, V stays BF16), row sum of the FP32 p; o = acc / l, lse = (m + log2 l) ln2;
//     out = bf16(o * sigmoid(gate) * sigmoid(lse - sink)), sigmoid = rcp.approx(1 + ex2.approx(-x log2e));
//   - paged KV pool (page table per segment, page size a power of two >= 64, strides as parameters: config B's
//     [page][slot][head][dim] or [page][head][slot][dim]); several segments (sessions) per launch through a tile
//     list; K/V staged with cp.async straight into swizzled shared memory (no register staging, no conversion);
//   - relative bias from a key-aligned bf16 table (column = key - ((pos & ~63) - 1024), zero outside distance
//     [0, 1024): same values as config B's sheared buffer, laid out per token so every 64-key block of a row is one
//     aligned 128-byte run), staged to shared memory with the K/V tiles of the blocks that need it;
//   - optional FP32 o / lse debug outputs.
//
// Invariance provided: every query row's output is a function of (its q, gate, sink, rel row, its position, the
// keys/values at positions it can see, the split rule) only. Each row visits the 64-key blocks that hold a key it can
// see, in ascending absolute order: blocks that are fully masked for a row are exact no-ops of the online softmax
// (alpha = 1, p = 0; or all zeros before the first visible key). Optional key splits (work items) cut a row tile's
// blocks at boundaries that are fixed absolute key positions (the caller's rule must depend on positions only, one rule
// per layer kind); each piece runs the same single-pass loop and the pieces are merged in piece order by whichever
// piece arrives last. So the output is bit-identical for any chunking of a prompt, any batch composition, any tile
// geometry / placement of the row, any arrival order and any page placement. Without splits the arithmetic is config
// B's single-split order (bit-equal to config B for ~99 % of outputs on synthetic data; the rest differ by fp32
// summation order inside a block); with splits it is a different, also fixed, fp32 order.
//
// Engine contract: rel table per token, kRelW = 1088 bf16 per (token, head): column c holds the bias for key
// ((pos & ~63) - 1024) + c, 0 where pos - key is outside [0, 1024). KV pool pages of >= 64 keys (power of two), any
// strides. Workspace (splits only): (sum over row tiles with > 1 piece of their pieces) x ws_floats_per_partial floats;
// counters: row tiles x 8 ints, zero before the first launch (self-resetting).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace pfa {

constexpr int kD = 128, kHeads = 24, kKvHeads = 8, kGqa = 3, kRelExt = 1024, kDrel = 16;
constexpr int kRelW = kRelExt + 64;
constexpr int kPbufRows = 112;   // covers max CTA token span (42) + BK (64) = 106 <= 112
constexpr int kBscCols = 74;     // covers max m16 token span (5) + BK (64) = 69 <= 74; 148 B stride avoids smem bank conflicts
#ifndef SKIP_RESCALE
#define SKIP_RESCALE 1
#endif
constexpr int DCH = kD / 8;  // 16-byte chunks per row
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

struct Seg {
  int q0;      // first token row of the segment in q / gate / sink / rel / out
  int len;     // new tokens (query rows) of the segment
  int past;    // cached keys before the segment (the segment's first token is at position `past`)
  int pt_off;  // offset of the segment's page table in `page_table`
};

struct Params {
  const __nv_bfloat16* q;     // [T][24][128] (clipped, tau-scaled, bf16)
  const __nv_bfloat16* k;     // pool
  const __nv_bfloat16* v;     // pool (same strides as k)
  long long s_page, s_slot, s_head;  // pool strides in elements
  int page_shift;             // log2(page size), >= 6
  const int* page_table;
  const __nv_bfloat16* rel;         // [T][24][16] relative query vectors (bf16, clipped & tau-scaled)
  const __nv_bfloat16* rel_proj_t;  // [1024][16] transposed relative projection (bf16, distance-major)
  const __nv_bfloat16* gate;        // [T][3072]
  const __nv_bfloat16* sink;  // [T][24]
  __nv_bfloat16* out;         // [T][3072]
  float* o32;                 // optional [T][24][128] fp32 o (pre-gate)
  float* lse;                 // optional [T][24] natural-log lse
  const Seg* segs;
  const int4* items;          // work items: (segment, first packed row, first key block, end key block)
  const int4* items2;         // (split index, splits of this row tile, first workspace partial, row-tile id)
  float* ws;                  // split partials: per (partial, KV head) NT x (64 MT o + 4 MT m/l) floats
  int* cnt;                   // per (row tile, KV head) arrival counters, zero between launches (self-reset)
  int window;                 // 0 = full (global), else sliding window (1024)
  float qk_scale;             // fp32(fp32(128^-0.5) * fp32(log2e)), as Triton forms it
};

__device__ __forceinline__ unsigned smem_addr(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t w) {
  const float a = __uint_as_float(w << 16), b = __uint_as_float(w & 0xffff0000u);
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
// E4M3 placement decode: byte -> f16 value * 2^-8 (exact for every finite code incl. subnormals; matches da::place2).
__device__ __forceinline__ uint32_t place2(uint32_t x, uint32_t y, uint32_t sel) {
  const uint32_t h = __byte_perm(x, y, sel);
  return ((h >> 1) & 0x3F803F80u) | (h & 0x80008000u);
}
__host__ __device__ constexpr uint32_t psel(int i, int j) {
  return (uint32_t)i | ((uint32_t)i << 4) | ((uint32_t)i << 8) | ((uint32_t)j << 12);
}
// [rows][8 chunks of 16 B], XOR swizzle on row & 7 for 128-byte E4M3 KV rows (byte offset)
__device__ __forceinline__ int swz8(int row, int chunk) { return row * kD + ((chunk ^ (row & 7)) << 4); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes = 16) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" : : "r"(smem_addr(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" : : "n"(N)); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rcp_approx(float x) { float y; asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float lg2(float x) { float y; asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float(uint32_t(u) << 16); }
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);  // cvt.rn.bf16x2.f32
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float sigmoid_cfgb(float x) { return rcp_approx(ex2(-x * kLog2e) + 1.0f); }
// [rows][16 chunks of 16 B], XOR swizzle on row & 7 (8 rows of an ldmatrix 8x8 hit distinct banks)
__device__ __forceinline__ int swz(int row, int chunk) { return row * (DCH * 8) + ((chunk ^ (row & 7)) << 3); }

// ------------------------------------------------------------------------------------------------ kernel
// One CTA = one KV head x BR packed query rows of one segment (row R of the segment = token R / 3, head 3 g + R % 3).
// NW warps, each MT m16 row tiles (BR = NW * 16 * MT). BK = 64 keys per key tile (config B's prefill BLOCK_N).
//   MT = 1: Q in registers (ingine). MT = 2: Q stays in shared memory, fragments re-read per k16 step (FA2): each
//           K / V fragment read from shared memory feeds twice the MMAs.
//   PIPE = 0: K and V tiles in a STAGES-deep cp.async ring, one barrier per key tile.
//   PIPE = 1: one K and one V buffer (FA2 order): V(i) loads during S(i) = Q K(i)^T and the softmax, K(i+1) during
//           O += P V(i); two barriers per key tile, half the shared memory (2 CTAs / SM at MT = 2).
template <int NW, int MT, int PIPE, int STAGES>
struct Cfg {
  static constexpr int NT = NW * 32, BR = NW * 16 * MT, BK = 64;
  static constexpr int TILE = BK * kD;                           // elements of one K or V tile
  static constexpr int Q_ELEMS = BR * kD;
  static constexpr int PT = kPbufRows * kDrel;                   // rel_proj_t tile elements (112 x 16 = 1792 bf16)
  static constexpr int BSCRATCH = NW * 16 * kBscCols;            // per-warp bias scratch (4 x 16 x 72 = 4608 bf16)
  static constexpr int STAGE = 2 * TILE + PT;                    // K, V, proj_t of one key tile
  static constexpr int KV_ELEMS = PIPE ? 2 * TILE + PT : STAGES * STAGE;
  static constexpr int Q_OFF = MT == 1 ? 0 : KV_ELEMS;           // MT = 1 stages Q over the ring once
  static constexpr int MAIN = MT == 1 ? (KV_ELEMS > Q_ELEMS ? KV_ELEMS : Q_ELEMS) : KV_ELEMS + Q_ELEMS;
  static constexpr int BSCRATCH_OFF = MAIN;
  static constexpr int SMEM = 2 * (BSCRATCH_OFF + BSCRATCH) + 16;
};

// CEIL = debug ceiling arm (wrong results): softmax, bias and masks skipped, raw scores packed as P; measures the
// MMA + data-movement rate of the structure at the power cap.
template <int NW, int MT, int PIPE, int STAGES, int MINB, bool CEIL = false>
__global__ void __launch_bounds__(NW * 32, MINB) prefill_attn_kernel(const Params p) {
  using C = Cfg<NW, MT, PIPE, STAGES>;
  constexpr int NT = C::NT, BR = C::BR, BK = C::BK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  uint16_t* sm = reinterpret_cast<uint16_t*>(smem_raw);
  auto kbuf = [&](int s) { return sm + (PIPE ? 0 : s * C::STAGE); };
  auto vbuf = [&](int s) { return sm + (PIPE ? 0 : s * C::STAGE) + C::TILE; };
  auto pbuf = [&](int s) { return sm + (PIPE ? 0 : s * C::STAGE) + 2 * C::TILE; };
  uint16_t* qs = sm + C::Q_OFF;
  uint16_t* b_warp = sm + C::BSCRATCH_OFF + (threadIdx.x >> 5) * (16 * kBscCols);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = blockIdx.y;
  const int4 it = p.items[blockIdx.x];
  const Seg sg = p.segs[it.x];
  const int rows = kGqa * sg.len;
  const int R0 = it.y;
  const int klen = sg.past + sg.len;
  const int* pt = p.page_table + sg.pt_off;

  // ---- Q tile -> shared
#pragma unroll
  for (int i = 0; i < (BR * DCH) / NT; ++i) {
    const int id = tid + i * NT, r = id >> 4, c = id & 15;
    const int R = min(R0 + r, rows - 1);
    const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
    cp_async16(qs + swz(r, c), p.q + (size_t(t) * kHeads + h) * kD + c * 8);
  }
  // ---- R tile [BR][16] -> kbuf(0) temporarily, then into warp registers rf[MT][4] (0 smem during key loop)
  const bool has_rel = p.rel != nullptr && p.rel_proj_t != nullptr;
  if (has_rel) {
    uint16_t* rs = kbuf(0);
#pragma unroll
    for (int i = 0; i < (BR * 2) / NT; ++i) {
      const int id = tid + i * NT, r = id >> 1, c = id & 1;
      const int R = min(R0 + r, rows - 1);
      const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
      cp_async16(rs + r * kDrel + ((c ^ (r & 1)) << 3), p.rel + (size_t(t) * kHeads + h) * kDrel + c * 8);
    }
  }
  cp_commit();
  // per-thread rows: (lane / 4) and (lane / 4 + 8) of each of the warp's MT 16-row tiles
  int qpos[MT][2], qmin_t[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    qmin_t[mt] = sg.past + min(R0 + (warp * MT + mt) * 16, rows - 1) / kGqa;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int R = min(R0 + (warp * MT + mt) * 16 + (lane >> 2) + r * 8, rows - 1);
      qpos[mt][r] = sg.past + R / kGqa;
    }
  }
  const int qmin = sg.past + R0 / kGqa;
  const int qmax = sg.past + min(R0 + BR - 1, rows - 1) / kGqa;
  const int lo = p.window ? max(0, qmin - p.window + 1) : 0;
  const int first = max(lo / BK, it.z);                 // this item's key blocks: [it.z, it.w) of the row tile's range
  const int ntiles = min(qmax / BK + 1, it.w) - first;

  unsigned rf[MT][4];
  if (has_rel) {
    cp_wait<0>();
    __syncthreads();
    const uint16_t* rs = kbuf(0);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = (warp * MT + mt) * 16 + (lane & 15);
      const int c = (lane >> 4) ^ (row & 1);
      ldmatrix_x4(rf[mt][0], rf[mt][1], rf[mt][2], rf[mt][3], smem_addr(rs + row * kDrel + (c << 3)));
    }
    __syncthreads();  // kbuf(0) is now free for K(0)
  }

  unsigned qf[MT == 1 ? DCH / 2 : 1][4];
  if constexpr (MT == 1) {
    if (!has_rel) {
      cp_wait<0>();
      __syncthreads();
    }
#pragma unroll
    for (int kk = 0; kk < DCH / 2; ++kk)
      ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], smem_addr(qs + swz(warp * 16 + (lane & 15), kk * 2 + (lane >> 4))));
    __syncthreads();  // the staging area is overwritten by the first key tiles
  }

  // ---- K / V tile loaders: 64 rows x 16 chunks per matrix; a 64-key tile lies in one page (page size >= 64)
  const long long kv_thread = (long long)(tid >> 4) * p.s_slot + (long long)g * p.s_head + (tid & 15) * 8;
  auto load_kv = [&](int i, uint16_t* dst, const __nv_bfloat16* src) {
    const int k0 = (first + i) * BK;
    const __nv_bfloat16* base = src + (long long)__ldg(pt + (k0 >> p.page_shift)) * p.s_page +
                                (long long)(k0 & ((1 << p.page_shift) - 1)) * p.s_slot + kv_thread;
#pragma unroll
    for (int j = 0; j < (BK * DCH) / NT; ++j) {
      const int row = (tid >> 4) + j * (NT / 16);
      const bool ok = k0 + row < klen;
      cp_async16(dst + swz(row, tid & 15), ok ? base + (long long)j * (NT / 16) * p.s_slot : src, ok ? 16 : 0);
    }
  };
  auto load_k = [&](int i, uint16_t* ks) { load_kv(i, ks, p.k); };
  auto load_v = [&](int i, uint16_t* vs) { load_kv(i, vs, p.v); };

  // ---- rel_proj_t tile loader: distances [d_lo, d_lo + 112), zero-padded outside [0, 1024), 3.5 KB from L2
  auto load_p = [&](int i, uint16_t* ps) {
    const int d_lo = qmin - ((first + i) * BK + BK - 1);
#pragma unroll
    for (int j = 0; j < (kPbufRows * 2 + NT - 1) / NT; ++j) {
      const int id = tid + j * NT;
      if (id < kPbufRows * 2) {
        const int r = id >> 1, c = id & 1;
        const int d = d_lo + r;
        const bool ok = d >= 0 && d < kRelExt;
        cp_async16(ps + r * kDrel + ((c ^ (r & 1)) << 3),
                   ok ? p.rel_proj_t + (size_t)d * kDrel + c * 8 : p.rel_proj_t,
                   ok ? 16 : 0);
      }
    }
  };
  auto tile_needs_bias = [&](int i) { return has_rel && qmin - ((first + i) * BK + BK - 1) < kRelExt; };

  float o[MT][kD / 8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) o[mt][i][0] = o[mt][i][1] = o[mt][i][2] = o[mt][i][3] = 0.f;
  float m[MT][2], l[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) m[mt][0] = m[mt][1] = -INFINITY, l[mt][0] = l[mt][1] = 0.f;

  if constexpr (PIPE) {
    if (ntiles > 0) { load_k(0, kbuf(0)); if (tile_needs_bias(0)) load_p(0, pbuf(0)); }
    cp_commit();
  } else {
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < ntiles) { load_k(s, kbuf(s)); load_v(s, vbuf(s)); if (tile_needs_bias(s)) load_p(s, pbuf(s)); }
      cp_commit();
    }
  }

#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    const int buf = PIPE ? 0 : i % STAGES;
    if constexpr (PIPE) {
      cp_wait<0>();     // K(i) (and Q on the first tile) landed
      __syncthreads();  // ... for every thread; every thread finished P V(i - 1): the V buffer is free
      load_v(i, vbuf(0));
      cp_commit();
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();  // tile i landed for every thread; every thread finished tile i - 1 (its buffer is free)
      const int nx = i + STAGES - 1, sx = nx % STAGES;
      if (nx < ntiles) { load_k(nx, kbuf(sx)); load_v(nx, vbuf(sx)); if (tile_needs_bias(nx)) load_p(nx, pbuf(sx)); }
      cp_commit();
    }

    const int kt = first + i;
    const int kmin = kt * BK, kmax = kt * BK + BK - 1;
    const int dmin = qmin - kmax, dmax = qmax - kmin;
    const bool need_mask = !(dmin >= 0 && (p.window == 0 || dmax < p.window));
    const bool need_bias = tile_needs_bias(i);
    const uint16_t* Ps = pbuf(buf);

    // ---- S = Q K^T
    float sc[MT][BK / 8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) sc[mt][nt][0] = sc[mt][nt][1] = sc[mt][nt][2] = sc[mt][nt][3] = 0.f;
    const uint16_t* Ks = kbuf(buf);
#pragma unroll
    for (int kk = 0; kk < DCH / 2; ++kk) {
      unsigned a[MT][4];
      if constexpr (MT == 1) {
#pragma unroll
        for (int x = 0; x < 4; ++x) a[0][x] = qf[kk][x];
      } else {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                      smem_addr(qs + swz((warp * MT + mt) * 16 + (lane & 15), kk * 2 + (lane >> 4))));
      }
      if constexpr (MT == 1) {  // all of the k16 group's key fragments first, then its MMAs
        unsigned b[BK / 16][4];
#pragma unroll
        for (int np = 0; np < BK / 16; ++np)
          ldmatrix_x4(b[np][0], b[np][1], b[np][2], b[np][3],
                      smem_addr(Ks + swz(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1))));
#pragma unroll
        for (int np = 0; np < BK / 16; ++np) {
          mma_bf16(sc[0][2 * np], a[0], b[np][0], b[np][1]);
          mma_bf16(sc[0][2 * np + 1], a[0], b[np][2], b[np][3]);
        }
      } else {  // one key fragment at a time (registers), each feeding 2 x MT MMAs
#pragma unroll
        for (int np = 0; np < BK / 16; ++np) {
          unsigned b[4];
          ldmatrix_x4(b[0], b[1], b[2], b[3],
                      smem_addr(Ks + swz(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1))));
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma_bf16(sc[mt][2 * np], a[mt], b[0], b[1]);
            mma_bf16(sc[mt][2 * np + 1], a[mt], b[2], b[3]);
          }
        }
      }
    }

    // ---- scores (log2 units), on-the-fly TC relative bias, mask, online softmax
#pragma unroll
    for (int mt = 0; mt < MT && !CEIL; ++mt) {
      float rmax[2] = {m[mt][0], m[mt][1]};
      if (!need_bias && !need_mask) {
#pragma unroll
        for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float x = sc[mt][nt][e] * p.qk_scale;
            sc[mt][nt][e] = x;
            rmax[e >> 1] = fmaxf(rmax[e >> 1], x);
          }
        }
      } else {
        if (need_bias) {
          const int off = qmin_t[mt] - qmin;
          const int r0_l = lane >> 2, r1_l = r0_l + 8, c_l = (lane & 3) * 2;
#pragma unroll
          for (int np = 0; np < 4; ++np) {
            const int pr = off + np * 16 + (lane & 7) + ((lane >> 4) << 3);
            const int pc = (lane >> 3) & 1;
            unsigned pb[4];
            ldmatrix_x4(pb[0], pb[1], pb[2], pb[3], smem_addr(Ps + pr * kDrel + ((pc ^ (pr & 1)) << 3)));
            float bc0[4] = {0.f, 0.f, 0.f, 0.f}, bc1[4] = {0.f, 0.f, 0.f, 0.f};
            mma_bf16(bc0, rf[mt], pb[0], pb[1]);
            mma_bf16(bc1, rf[mt], pb[2], pb[3]);
            *reinterpret_cast<uint32_t*>(b_warp + r0_l * kBscCols + (2 * np) * 8 + c_l) = pack_bf16x2(bc0[0], bc0[1]);
            *reinterpret_cast<uint32_t*>(b_warp + r1_l * kBscCols + (2 * np) * 8 + c_l) = pack_bf16x2(bc0[2], bc0[3]);
            *reinterpret_cast<uint32_t*>(b_warp + r0_l * kBscCols + (2 * np + 1) * 8 + c_l) = pack_bf16x2(bc1[0], bc1[1]);
            *reinterpret_cast<uint32_t*>(b_warp + r1_l * kBscCols + (2 * np + 1) * 8 + c_l) = pack_bf16x2(bc1[2], bc1[3]);
          }
          {
            const int pr = min(off + 64 + (lane & 7) + ((lane >> 4) << 3), kPbufRows - 1);
            const int pc = (lane >> 3) & 1;
            unsigned pb[4];
            ldmatrix_x4(pb[0], pb[1], pb[2], pb[3], smem_addr(Ps + pr * kDrel + ((pc ^ (pr & 1)) << 3)));
            float bc0[4] = {0.f, 0.f, 0.f, 0.f};
            mma_bf16(bc0, rf[mt], pb[0], pb[1]);
            *reinterpret_cast<uint32_t*>(b_warp + r0_l * kBscCols + 64 + c_l) = pack_bf16x2(bc0[0], bc0[1]);
            *reinterpret_cast<uint32_t*>(b_warp + r1_l * kBscCols + 64 + c_l) = pack_bf16x2(bc0[2], bc0[3]);
          }
          __syncwarp();
        }
        const int sh0 = qpos[mt][0] - qmin_t[mt];
        const int sh1 = qpos[mt][1] - qmin_t[mt];
        const int bo0 = (lane >> 2) * kBscCols + sh0 + 63;
        const int bo1 = ((lane >> 2) + 8) * kBscCols + sh1 + 63;
        if (!need_mask) {
#pragma unroll
          for (int nt = 0; nt < BK / 8; ++nt) {
            const int jl0 = nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int r = e >> 1, jl = jl0 + (e & 1);
              const uint16_t bv = b_warp[(r ? bo1 : bo0) - jl];
              const float x = fmaf(bf2f(bv), kLog2e, sc[mt][nt][e] * p.qk_scale);
              sc[mt][nt][e] = x;
              rmax[r] = fmaxf(rmax[r], x);
            }
          }
        } else {
#pragma unroll
          for (int nt = 0; nt < BK / 8; ++nt) {
            const int jl0 = nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int r = e >> 1, jl = jl0 + (e & 1);
              float x = sc[mt][nt][e] * p.qk_scale;
              if (need_bias) {
                const uint16_t bv = b_warp[(r ? bo1 : bo0) - jl];
                x = fmaf(bf2f(bv), kLog2e, x);
              }
              const int d = qpos[mt][r] - (kmin + jl);
              x = (d >= 0 && (p.window == 0 || d < p.window)) ? x : -INFINITY;
              sc[mt][nt][e] = x;
              rmax[r] = fmaxf(rmax[r], x);
            }
          }
        }
      }
      float mref[2], alpha[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1));
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2));
        mref[r] = rmax[r] == -INFINITY ? 0.f : rmax[r];
        alpha[r] = ex2(m[mt][r] - mref[r]);
        m[mt][r] = rmax[r];
        l[mt][r] *= alpha[r];
      }
      // o *= alpha only if some row's max moved: alpha == 1 exactly is an identity, so skipping it changes no bit
      if (!SKIP_RESCALE || __any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt) {
          o[mt][dt][0] *= alpha[0];
          o[mt][dt][1] *= alpha[0];
          o[mt][dt][2] *= alpha[1];
          o[mt][dt][3] *= alpha[1];
        }
      }
      float ts[2] = {0.f, 0.f};  // this key tile's row sums first, then l = l * alpha + sum (config B's order)
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float pv = ex2(sc[mt][nt][e] - mref[e >> 1]);
          ts[e >> 1] += pv;
          sc[mt][nt][e] = pv;
        }
      }
      l[mt][0] += ts[0];
      l[mt][1] += ts[1];
    }

    if constexpr (PIPE) {
      cp_wait<0>();     // V(i) landed
      __syncthreads();  // ... for every thread; every thread finished S(i): the K buffer is free
      if (i + 1 < ntiles) { load_k(i + 1, kbuf(0)); if (tile_needs_bias(i + 1)) load_p(i + 1, pbuf(0)); }
      cp_commit();
    }

    // ---- O += P V (P bf16, V bf16)
    const uint16_t* Vs = vbuf(buf);
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      unsigned a[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        a[mt][0] = pack_bf16x2(sc[mt][2 * kk][0], sc[mt][2 * kk][1]);
        a[mt][1] = pack_bf16x2(sc[mt][2 * kk][2], sc[mt][2 * kk][3]);
        a[mt][2] = pack_bf16x2(sc[mt][2 * kk + 1][0], sc[mt][2 * kk + 1][1]);
        a[mt][3] = pack_bf16x2(sc[mt][2 * kk + 1][2], sc[mt][2 * kk + 1][3]);
      }
#pragma unroll
      for (int dq = 0; dq < kD / 32; ++dq) {
        constexpr int HB = MT == 1 ? 2 : 1;  // value fragments loaded together
#pragma unroll
        for (int h0 = 0; h0 < 2; h0 += HB) {
          unsigned b[HB][4];
#pragma unroll
          for (int h = 0; h < HB; ++h)
            ldmatrix_x4_t(b[h][0], b[h][1], b[h][2], b[h][3],
                          smem_addr(Vs + swz(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, (2 * dq + h0 + h) * 2 + (lane >> 4))));
#pragma unroll
          for (int h = 0; h < HB; ++h)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              mma_bf16(o[mt][2 * (2 * dq + h0 + h)], a[mt], b[h][0], b[h][1]);
              mma_bf16(o[mt][2 * (2 * dq + h0 + h) + 1], a[mt], b[h][2], b[h][3]);
            }
        }
      }
    }
  }
  cp_wait<0>();

  // ---- split items: store the partial (unnormalised o, running max, per-thread row sum); the last of the row
  // tile's items to arrive merges all partials in split order (fixed: deterministic, independent of arrival order)
  const int4 it2 = p.items2[blockIdx.x];
  if (it2.y > 1) {
    constexpr int NO = 16 * MT;  // float4 of o per thread
    auto wsp = [&](int part) { return p.ws + (size_t)(part * kKvHeads + g) * NT * (NO * 4 + 4 * MT); };
    {
      float4* w4 = reinterpret_cast<float4*>(wsp(it2.z + it2.x));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt) w4[(mt * 16 + dt) * NT + tid] = make_float4(o[mt][dt][0], o[mt][dt][1], o[mt][dt][2], o[mt][dt][3]);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) w4[(NO + mt) * NT + tid] = make_float4(m[mt][0], m[mt][1], l[mt][0], l[mt][1]);
    }
    __threadfence();
    __syncthreads();  // every thread left the key loop: b_scratch is free (no static smem: 2 CTAs / SM)
    volatile int* s_last = reinterpret_cast<volatile int*>(sm + C::BSCRATCH_OFF);
    if (tid == 0) {
      const int old = atomicAdd(p.cnt + it2.w * kKvHeads + g, 1);
      s_last[0] = old == it2.y - 1;
      if (old == it2.y - 1) p.cnt[it2.w * kKvHeads + g] = 0;  // every item of this (tile, head) has arrived: reset
    }
    __syncthreads();
    if (!s_last[0]) return;
    __threadfence();
    // merge: M = max_s m_s, w_s = 2^(m_s - M), l = sum_s w_s l_s, o = sum_s w_s o_s, in split order
    float M[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) M[mt][0] = M[mt][1] = -INFINITY;
    for (int sp = 0; sp < it2.y; ++sp) {
      const float4* w4 = reinterpret_cast<const float4*>(wsp(it2.z + sp));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float4 ml = __ldcg(w4 + (NO + mt) * NT + tid);
        M[mt][0] = fmaxf(M[mt][0], ml.x);
        M[mt][1] = fmaxf(M[mt][1], ml.y);
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      l[mt][0] = l[mt][1] = 0.f;
      m[mt][0] = M[mt][0];
      m[mt][1] = M[mt][1];
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) o[mt][dt][0] = o[mt][dt][1] = o[mt][dt][2] = o[mt][dt][3] = 0.f;
    }
    for (int sp = 0; sp < it2.y; ++sp) {
      const float4* w4 = reinterpret_cast<const float4*>(wsp(it2.z + sp));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float4 ml = __ldcg(w4 + (NO + mt) * NT + tid);
        const float w0 = ml.x == -INFINITY ? 0.f : ex2(ml.x - M[mt][0]);
        const float w1 = ml.y == -INFINITY ? 0.f : ex2(ml.y - M[mt][1]);
        l[mt][0] = fmaf(ml.z, w0, l[mt][0]);
        l[mt][1] = fmaf(ml.w, w1, l[mt][1]);
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt) {
          const float4 v = __ldcg(w4 + (mt * 16 + dt) * NT + tid);
          o[mt][dt][0] = fmaf(v.x, w0, o[mt][dt][0]);
          o[mt][dt][1] = fmaf(v.y, w0, o[mt][dt][1]);
          o[mt][dt][2] = fmaf(v.z, w1, o[mt][dt][2]);
          o[mt][dt][3] = fmaf(v.w, w1, o[mt][dt][3]);
        }
      }
    }
  }

  // ---- epilogue: o = acc / l, lse, gate then sink, one bf16 rounding
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float lr = l[mt][r];
      lr += __shfl_xor_sync(0xffffffffu, lr, 1);
      lr += __shfl_xor_sync(0xffffffffu, lr, 2);
      const int R = R0 + (warp * MT + mt) * 16 + (lane >> 2) + r * 8;
      if (R >= rows) continue;
      const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
      const bool empty = lr == 0.f;
      const float den = empty ? 1.f : lr;
      const float lse = (empty ? -INFINITY : m[mt][r] + lg2(lr)) * kLn2;
      const float fs = sigmoid_cfgb(lse - bf2f(reinterpret_cast<const uint16_t*>(p.sink)[size_t(t) * kHeads + h]));
      const uint16_t* grow = reinterpret_cast<const uint16_t*>(p.gate) + size_t(t) * (kHeads * kD) + h * kD;
      uint16_t* orow = reinterpret_cast<uint16_t*>(p.out) + size_t(t) * (kHeads * kD) + h * kD;
      if (p.lse && (lane & 3) == 0) p.lse[size_t(t) * kHeads + h] = lse;
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) {
        const int col = dt * 8 + (lane & 3) * 2;
        const float o0 = o[mt][dt][2 * r] / den, o1 = o[mt][dt][2 * r + 1] / den;
        if (p.o32) *reinterpret_cast<float2*>(p.o32 + (size_t(t) * kHeads + h) * kD + col) = make_float2(o0, o1);
        const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + col);
        const float y0 = (o0 * sigmoid_cfgb(bf_lo(gw))) * fs;
        const float y1 = (o1 * sigmoid_cfgb(bf_hi(gw))) * fs;
        *reinterpret_cast<uint32_t*>(orow + col) = pack_bf16x2(y0, y1);
      }
    }
}

// ------------------------------------------------------------------------------------------------ host side
// workspace floats per split partial (all KV heads) of a variant: 8 x NT x (64 MT + 4 MT)
template <int NW, int MT>
constexpr size_t ws_floats_per_partial() { return (size_t)kKvHeads * NW * 32 * (64 * MT + 4 * MT); }

template <int NW, int MT, int PIPE, int STAGES, int MINB, bool CEIL = false>
inline cudaError_t launch(const Params& p, int nitems, cudaStream_t s) {
  using C = Cfg<NW, MT, PIPE, STAGES>;
  static cudaError_t attr = cudaFuncSetAttribute(prefill_attn_kernel<NW, MT, PIPE, STAGES, MINB, CEIL>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  if (attr != cudaSuccess) return attr;
  if (nitems <= 0) return cudaSuccess;
  prefill_attn_kernel<NW, MT, PIPE, STAGES, MINB, CEIL><<<dim3(nitems, kKvHeads), C::NT, C::SMEM, s>>>(p);
  return cudaGetLastError();
}

// ------------------------------------------------------------------------------------------------ FAR-only BF16 kernel
// Global-layer unmasked, bias-free prefill attention for work items whose key blocks all lie at distance >= kRelExt
// (1024) from every query in the tile (qmin >= it.w * 64 + 1023, window == 0). Dropping the relative-bias table,
// bias scratch, rf/qpos/qmin_t state and mask branches eliminates all register spills (local = 0 vs 40 B), cuts
// dynamic smem to 65,552 B (2 CTAs/SM), and matches prefill_attn_kernel bit-for-bit on every FAR item.
template <int NW, int MT, int PIPE, int STAGES>
struct CfgFar {
  static constexpr int NT = NW * 32, BR = NW * 16 * MT, BK = 64;
  static constexpr int TILE = BK * kD;
  static constexpr int Q_ELEMS = BR * kD;
  static constexpr int STAGE = 2 * TILE;
  static constexpr int KV_ELEMS = PIPE ? 2 * TILE : STAGES * STAGE;
  static constexpr int Q_OFF = MT == 1 ? 0 : KV_ELEMS;
  static constexpr int SMEM = 2 * (MT == 1 ? (KV_ELEMS > Q_ELEMS ? KV_ELEMS : Q_ELEMS) : KV_ELEMS + Q_ELEMS) + 16;
};

template <int NW, int MT, int PIPE, int STAGES, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB) prefill_attn_far_kernel(const Params p) {
  static_assert(MT == 2, "prefill_attn_far_kernel expects MT=2");
  using C = CfgFar<NW, MT, PIPE, STAGES>;
  constexpr int NT = C::NT, BR = C::BR, BK = C::BK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  uint16_t* sm = reinterpret_cast<uint16_t*>(smem_raw);
  auto kbuf = [&](int s) { return sm + (PIPE ? 0 : s * C::STAGE); };
  auto vbuf = [&](int s) { return sm + (PIPE ? 0 : s * C::STAGE) + C::TILE; };
  uint16_t* qs = sm + C::Q_OFF;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = blockIdx.y;
  const int4 it = p.items[blockIdx.x];
  const Seg sg = p.segs[it.x];
  const int rows = kGqa * sg.len;
  const int R0 = it.y;
  const int* pt = p.page_table + sg.pt_off;

#pragma unroll
  for (int i = 0; i < (BR * DCH) / NT; ++i) {
    const int id = tid + i * NT, r = id >> 4, c = id & 15;
    const int R = min(R0 + r, rows - 1);
    const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
    cp_async16(qs + swz(r, c), p.q + (size_t(t) * kHeads + h) * kD + c * 8);
  }
  cp_commit();

  const int first = it.z;
  const int ntiles = it.w - first;
  const long long kv_thread = (long long)(tid >> 4) * p.s_slot + (long long)g * p.s_head + (tid & 15) * 8;
  auto load_kv = [&](int i, uint16_t* dst, const __nv_bfloat16* src) {
    const int k0 = (first + i) * BK;
    const __nv_bfloat16* base = src + (long long)__ldg(pt + (k0 >> p.page_shift)) * p.s_page +
                                (long long)(k0 & ((1 << p.page_shift) - 1)) * p.s_slot + kv_thread;
#pragma unroll
    for (int j = 0; j < (BK * DCH) / NT; ++j) {
      const int row = (tid >> 4) + j * (NT / 16);
      cp_async16(dst + swz(row, tid & 15), base + (long long)j * (NT / 16) * p.s_slot, 16);
    }
  };

  float o[MT][kD / 8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) o[mt][i][0] = o[mt][i][1] = o[mt][i][2] = o[mt][i][3] = 0.f;
  float m[MT][2], l[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) m[mt][0] = m[mt][1] = -INFINITY, l[mt][0] = l[mt][1] = 0.f;

  if constexpr (PIPE) {
    if (ntiles > 0) load_kv(0, kbuf(0), p.k);
    cp_commit();
  } else {
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < ntiles) { load_kv(s, kbuf(s), p.k); load_kv(s, vbuf(s), p.v); }
      cp_commit();
    }
  }

#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    const int buf = PIPE ? 0 : i % STAGES;
    if constexpr (PIPE) {
      cp_wait<0>();
      __syncthreads();
      load_kv(i, vbuf(0), p.v);
      cp_commit();
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();
      const int nx = i + STAGES - 1, sx = nx % STAGES;
      if (nx < ntiles) { load_kv(nx, kbuf(sx), p.k); load_kv(nx, vbuf(sx), p.v); }
      cp_commit();
    }

    float sc[MT][BK / 8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) sc[mt][nt][0] = sc[mt][nt][1] = sc[mt][nt][2] = sc[mt][nt][3] = 0.f;
    const uint16_t* Ks = kbuf(buf);
#pragma unroll
    for (int kk = 0; kk < DCH / 2; ++kk) {
      unsigned a[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                    smem_addr(qs + swz((warp * MT + mt) * 16 + (lane & 15), kk * 2 + (lane >> 4))));
#pragma unroll
      for (int np = 0; np < BK / 16; ++np) {
        unsigned b[4];
        ldmatrix_x4(b[0], b[1], b[2], b[3],
                    smem_addr(Ks + swz(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1))));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(sc[mt][2 * np], a[mt], b[0], b[1]);
          mma_bf16(sc[mt][2 * np + 1], a[mt], b[2], b[3]);
        }
      }
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      float rmax[2] = {m[mt][0], m[mt][1]};
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float x = sc[mt][nt][e] * p.qk_scale;
          sc[mt][nt][e] = x;
          rmax[e >> 1] = fmaxf(rmax[e >> 1], x);
        }
      }
      float mref[2], alpha[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1));
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2));
        mref[r] = rmax[r];
        alpha[r] = ex2(m[mt][r] - mref[r]);
        m[mt][r] = rmax[r];
        l[mt][r] *= alpha[r];
      }
      if (!SKIP_RESCALE || __any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt) {
          o[mt][dt][0] *= alpha[0];
          o[mt][dt][1] *= alpha[0];
          o[mt][dt][2] *= alpha[1];
          o[mt][dt][3] *= alpha[1];
        }
      }
      float ts[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float pv = ex2(sc[mt][nt][e] - mref[e >> 1]);
          ts[e >> 1] += pv;
          sc[mt][nt][e] = pv;
        }
      }
      l[mt][0] += ts[0];
      l[mt][1] += ts[1];
    }

    if constexpr (PIPE) {
      cp_wait<0>();
      __syncthreads();
      if (i + 1 < ntiles) load_kv(i + 1, kbuf(0), p.k);
      cp_commit();
    }

    const uint16_t* Vs = vbuf(buf);
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      unsigned a[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        a[mt][0] = pack_bf16x2(sc[mt][2 * kk][0], sc[mt][2 * kk][1]);
        a[mt][1] = pack_bf16x2(sc[mt][2 * kk][2], sc[mt][2 * kk][3]);
        a[mt][2] = pack_bf16x2(sc[mt][2 * kk + 1][0], sc[mt][2 * kk + 1][1]);
        a[mt][3] = pack_bf16x2(sc[mt][2 * kk + 1][2], sc[mt][2 * kk + 1][3]);
      }
#pragma unroll
      for (int dq = 0; dq < kD / 32; ++dq) {
#pragma unroll
        for (int h0 = 0; h0 < 2; ++h0) {
          unsigned b[4];
          ldmatrix_x4_t(b[0], b[1], b[2], b[3],
                        smem_addr(Vs + swz(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, (2 * dq + h0) * 2 + (lane >> 4))));
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma_bf16(o[mt][2 * (2 * dq + h0)], a[mt], b[0], b[1]);
            mma_bf16(o[mt][2 * (2 * dq + h0) + 1], a[mt], b[2], b[3]);
          }
        }
      }
    }
  }
  cp_wait<0>();

  const int4 it2 = p.items2[blockIdx.x];
  if (it2.y > 1) {
    constexpr int NO = 16 * MT;
    auto wsp = [&](int part) { return p.ws + (size_t)(part * kKvHeads + g) * NT * (NO * 4 + 4 * MT); };
    {
      float4* w4 = reinterpret_cast<float4*>(wsp(it2.z + it2.x));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt)
          w4[(mt * 16 + dt) * NT + tid] = make_float4(o[mt][dt][0], o[mt][dt][1], o[mt][dt][2], o[mt][dt][3]);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) w4[(NO + mt) * NT + tid] = make_float4(m[mt][0], m[mt][1], l[mt][0], l[mt][1]);
    }
    __threadfence();
    __syncthreads();
    volatile int* s_last = reinterpret_cast<volatile int*>(sm);
    if (tid == 0) {
      const int old = atomicAdd(p.cnt + it2.w * kKvHeads + g, 1);
      s_last[0] = old == it2.y - 1;
      if (old == it2.y - 1) p.cnt[it2.w * kKvHeads + g] = 0;
    }
    __syncthreads();
    if (!s_last[0]) return;
    __threadfence();
    float M[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) M[mt][0] = M[mt][1] = -INFINITY;
    for (int sp = 0; sp < it2.y; ++sp) {
      const float4* w4 = reinterpret_cast<const float4*>(wsp(it2.z + sp));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float4 ml = __ldcg(w4 + (NO + mt) * NT + tid);
        M[mt][0] = fmaxf(M[mt][0], ml.x);
        M[mt][1] = fmaxf(M[mt][1], ml.y);
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      l[mt][0] = l[mt][1] = 0.f;
      m[mt][0] = M[mt][0];
      m[mt][1] = M[mt][1];
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) o[mt][dt][0] = o[mt][dt][1] = o[mt][dt][2] = o[mt][dt][3] = 0.f;
    }
    for (int sp = 0; sp < it2.y; ++sp) {
      const float4* w4 = reinterpret_cast<const float4*>(wsp(it2.z + sp));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float4 ml = __ldcg(w4 + (NO + mt) * NT + tid);
        const float w0 = ml.x == -INFINITY ? 0.f : ex2(ml.x - M[mt][0]);
        const float w1 = ml.y == -INFINITY ? 0.f : ex2(ml.y - M[mt][1]);
        l[mt][0] = fmaf(ml.z, w0, l[mt][0]);
        l[mt][1] = fmaf(ml.w, w1, l[mt][1]);
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt) {
          const float4 v = __ldcg(w4 + (mt * 16 + dt) * NT + tid);
          o[mt][dt][0] = fmaf(v.x, w0, o[mt][dt][0]);
          o[mt][dt][1] = fmaf(v.y, w0, o[mt][dt][1]);
          o[mt][dt][2] = fmaf(v.z, w1, o[mt][dt][2]);
          o[mt][dt][3] = fmaf(v.w, w1, o[mt][dt][3]);
        }
      }
    }
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float lr = l[mt][r];
      lr += __shfl_xor_sync(0xffffffffu, lr, 1);
      lr += __shfl_xor_sync(0xffffffffu, lr, 2);
      const int R = R0 + (warp * MT + mt) * 16 + (lane >> 2) + r * 8;
      if (R >= rows) continue;
      const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
      const bool empty = lr == 0.f;
      const float den = empty ? 1.f : lr;
      const float lse = (empty ? -INFINITY : m[mt][r] + lg2(lr)) * kLn2;
      const float fs = sigmoid_cfgb(lse - bf2f(reinterpret_cast<const uint16_t*>(p.sink)[size_t(t) * kHeads + h]));
      const uint16_t* grow = reinterpret_cast<const uint16_t*>(p.gate) + size_t(t) * (kHeads * kD) + h * kD;
      uint16_t* orow = reinterpret_cast<uint16_t*>(p.out) + size_t(t) * (kHeads * kD) + h * kD;
      if (p.lse && (lane & 3) == 0) p.lse[size_t(t) * kHeads + h] = lse;
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) {
        const int col = dt * 8 + (lane & 3) * 2;
        const float o0 = o[mt][dt][2 * r] / den, o1 = o[mt][dt][2 * r + 1] / den;
        if (p.o32) *reinterpret_cast<float2*>(p.o32 + (size_t(t) * kHeads + h) * kD + col) = make_float2(o0, o1);
        const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + col);
        const float y0 = (o0 * sigmoid_cfgb(bf_lo(gw))) * fs;
        const float y1 = (o1 * sigmoid_cfgb(bf_hi(gw))) * fs;
        *reinterpret_cast<uint32_t*>(orow + col) = pack_bf16x2(y0, y1);
      }
    }
}

// ------------------------------------------------------------------------------------------------ FP8 (E4M3) KV kernel
// Global-layer E4M3 KV prefill attention (--kv-fp8): K/V tiles staged as 1-byte E4M3 (half the global bytes and smem)
// with cp.async, decoded in registers by place2 (exact E4M3 * 2^-8 -> f16), Q converted once to Maya-ordered f16 in
// smem, P packed to f16, f16 tensor-core MMAs; p.qk_scale carries the 256x K dequant fold, epilogue applies 256x V
// dequant fold. Same fixed absolute key-block and split-K merge order as prefill_attn_kernel (chunk/batch invariant).
template <int NW, int MT, int PIPE, int STAGES>
struct CfgFp8 {
  static constexpr int NT = NW * 32, BR = NW * 16 * MT, BK = 64;
  static constexpr int TILE_B = BK * kD;                                 // 8,192 B per E4M3 K or V tile
  static constexpr int PT_B = kPbufRows * kDrel * 2;                     // 3,584 B per bf16 rel_proj_t tile
  static constexpr int STAGE_B = 2 * TILE_B + PT_B;                      // 19,968 B per stage (128-B aligned)
  static constexpr int KV_B = PIPE ? STAGE_B : STAGES * STAGE_B;
  static constexpr int Q_B = BR * kD * 2;                                // 32,768 B per f16 Q tile
  static constexpr int Q_OFF_B = MT == 1 ? 0 : KV_B;
  static constexpr int BSCRATCH_OFF_B = (MT == 1 ? (KV_B > Q_B ? KV_B : Q_B) : KV_B + Q_B);
  static constexpr int BSCRATCH_B = NW * 16 * kBscCols * 2;              // 9,472 B per-warp TC bias scratch
  static constexpr int SMEM = BSCRATCH_OFF_B + BSCRATCH_B + 16;          // 62,224 B
};

template <int NW, int MT, int PIPE, int STAGES, int MINB, bool CEIL = false>
__global__ void __launch_bounds__(NW * 32, MINB) prefill_attn_fp8_kernel(const Params p) {
  static_assert(PIPE == 1 && MT == 2, "prefill_attn_fp8_kernel expects PIPE=1, MT=2");
  using C = CfgFp8<NW, MT, PIPE, STAGES>;
  constexpr int NT = C::NT, BR = C::BR, BK = C::BK, DCH8 = kD / 16;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  uint8_t* kbuf0 = smem_raw;
  uint8_t* vbuf0 = smem_raw + C::TILE_B;
  uint16_t* pbuf0 = reinterpret_cast<uint16_t*>(smem_raw + 2 * C::TILE_B);
  uint16_t* qs = reinterpret_cast<uint16_t*>(smem_raw + C::Q_OFF_B);
  uint16_t* b_warp = reinterpret_cast<uint16_t*>(smem_raw + C::BSCRATCH_OFF_B) + (threadIdx.x >> 5) * (16 * kBscCols);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g_lane = lane >> 2, t4 = lane & 3;
  const int g = blockIdx.y;
  const int4 it = p.items[blockIdx.x];
  const Seg sg = p.segs[it.x];
  const int rows = kGqa * sg.len;
  const int R0 = it.y;
  const int klen = sg.past + sg.len;
  const int* pt = p.page_table + sg.pt_off;

  // ---- Q tile -> shared (bf16)
#pragma unroll
  for (int i = 0; i < (BR * DCH) / NT; ++i) {
    const int id = tid + i * NT, r = id >> 4, c = id & 15;
    const int R = min(R0 + r, rows - 1);
    const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
    cp_async16(qs + swz(r, c), p.q + (size_t(t) * kHeads + h) * kD + c * 8);
  }
  // ---- R tile [BR][16] -> kbuf0 temporarily, then into warp registers rf[MT][4]
  const bool has_rel = p.rel != nullptr && p.rel_proj_t != nullptr;
  if (has_rel) {
    uint16_t* rs = reinterpret_cast<uint16_t*>(kbuf0);
#pragma unroll
    for (int i = 0; i < (BR * 2) / NT; ++i) {
      const int id = tid + i * NT, r = id >> 1, c = id & 1;
      const int R = min(R0 + r, rows - 1);
      const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
      cp_async16(rs + r * kDrel + ((c ^ (r & 1)) << 3), p.rel + (size_t(t) * kHeads + h) * kDrel + c * 8);
    }
  }
  cp_commit();

  int qpos[MT][2], qmin_t[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    qmin_t[mt] = sg.past + min(R0 + (warp * MT + mt) * 16, rows - 1) / kGqa;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int R = min(R0 + (warp * MT + mt) * 16 + g_lane + r * 8, rows - 1);
      qpos[mt][r] = sg.past + R / kGqa;
    }
  }
  const int qmin = sg.past + R0 / kGqa;
  const int qmax = sg.past + min(R0 + BR - 1, rows - 1) / kGqa;
  const int lo = p.window ? max(0, qmin - p.window + 1) : 0;
  const int first = max(lo / BK, it.z);
  const int ntiles = min(qmax / BK + 1, it.w) - first;

  // Wait for Q (and R) to land; load rf[MT][4] and convert qs in-place from natural bf16 to Maya-ordered f16
  cp_wait<0>();
  __syncthreads();
  unsigned rf[MT][4];
  if (has_rel) {
    const uint16_t* rs = reinterpret_cast<const uint16_t*>(kbuf0);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = (warp * MT + mt) * 16 + (lane & 15);
      const int c = (lane >> 4) ^ (row & 1);
      ldmatrix_x4(rf[mt][0], rf[mt][1], rf[mt][2], rf[mt][3], smem_addr(rs + row * kDrel + (c << 3)));
    }
  }
#pragma unroll 1
  for (int j = 0; j < (BR * (DCH / 2)) / NT; ++j) {
    const int idx = tid + j * NT;
    const int r = idx >> 3, kk = idx & 7;
    uint4* p0 = reinterpret_cast<uint4*>(qs + swz(r, 2 * kk));
    uint4* p1 = reinterpret_cast<uint4*>(qs + swz(r, 2 * kk + 1));
    const uint4 u0 = *p0, u1 = *p1;
    const uint32_t h0 = bf2_to_h2(u0.x), h1 = bf2_to_h2(u0.y), h2 = bf2_to_h2(u0.z), h3 = bf2_to_h2(u0.w);
    const uint32_t h4 = bf2_to_h2(u1.x), h5 = bf2_to_h2(u1.y), h6 = bf2_to_h2(u1.z), h7 = bf2_to_h2(u1.w);
    *p0 = make_uint4(h0, h2, h4, h6);
    *p1 = make_uint4(h1, h3, h5, h7);
  }
  __syncthreads();

  // ---- E4M3 K / V tile loaders: 64 rows x 8 chunks of 16 bytes (half the cp.async count of bf16)
  const uint8_t* kbase = reinterpret_cast<const uint8_t*>(p.k);
  const uint8_t* vbase = reinterpret_cast<const uint8_t*>(p.v);
  const long long kv_thread = (long long)(tid >> 3) * p.s_slot + (long long)g * p.s_head + (tid & 7) * 16;
  auto load_kv8 = [&](int i, uint8_t* dst, const uint8_t* src) {
    const int k0 = (first + i) * BK;
    const uint8_t* base = src + (long long)__ldg(pt + (k0 >> p.page_shift)) * p.s_page +
                          (long long)(k0 & ((1 << p.page_shift) - 1)) * p.s_slot + kv_thread;
#pragma unroll
    for (int j = 0; j < (BK * DCH8) / NT; ++j) {
      const int row = (tid >> 3) + j * (NT / DCH8);
      const bool ok = k0 + row < klen;
      cp_async16(dst + swz8(row, tid & 7), ok ? base + (long long)j * (NT / DCH8) * p.s_slot : src, ok ? 16 : 0);
    }
  };
  auto load_p = [&](int i, uint16_t* ps) {
    const int d_lo = qmin - ((first + i) * BK + BK - 1);
#pragma unroll
    for (int j = 0; j < (kPbufRows * 2 + NT - 1) / NT; ++j) {
      const int id = tid + j * NT;
      if (id < kPbufRows * 2) {
        const int r = id >> 1, c = id & 1;
        const int d = d_lo + r;
        const bool ok = d >= 0 && d < kRelExt;
        cp_async16(ps + r * kDrel + ((c ^ (r & 1)) << 3),
                   ok ? p.rel_proj_t + (size_t)d * kDrel + c * 8 : p.rel_proj_t,
                   ok ? 16 : 0);
      }
    }
  };
  auto tile_needs_bias = [&](int i) { return has_rel && qmin - ((first + i) * BK + BK - 1) < kRelExt; };

  float o[MT][kD / 8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) o[mt][i][0] = o[mt][i][1] = o[mt][i][2] = o[mt][i][3] = 0.f;
  float m[MT][2], l[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) m[mt][0] = m[mt][1] = -INFINITY, l[mt][0] = l[mt][1] = 0.f;

  const uint32_t vsel = psel(g_lane & 3, 4 + (g_lane & 3));

  if (ntiles > 0) {
    load_kv8(0, kbuf0, kbase);
    if (tile_needs_bias(0)) load_p(0, pbuf0);
  }
  cp_commit();

#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    cp_wait<0>();
    __syncthreads();
    load_kv8(i, vbuf0, vbase);
    cp_commit();

    const int kt = first + i;
    const int kmin = kt * BK, kmax = kt * BK + BK - 1;
    const int dmin = qmin - kmax, dmax = qmax - kmin;
    const bool need_mask = !(dmin >= 0 && (p.window == 0 || dmax < p.window));
    const bool need_bias = tile_needs_bias(i);

    // ---- S = Q K^T (f16 mma, K decoded from E4M3 via place2)
    float sc[MT][BK / 8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) sc[mt][nt][0] = sc[mt][nt][1] = sc[mt][nt][2] = sc[mt][nt][3] = 0.f;
#pragma unroll 1
    for (int kk = 0; kk < DCH / 2; ++kk) {
      unsigned a[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                    smem_addr(qs + swz((warp * MT + mt) * 16 + (lane & 15), kk * 2 + (lane >> 4))));
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(kbuf0 + swz8(nt * 8 + g_lane, kk) + t4 * 4);
        const unsigned b0 = place2(w, w, psel(0, 1));
        const unsigned b1 = place2(w, w, psel(2, 3));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma_f16(sc[mt][nt], a[mt], b0, b1);
      }
    }

    // ---- scores (log2 units, p.qk_scale includes 256x K dequant fold), in-kernel TC bias, mask, online softmax
#pragma unroll
    for (int mt = 0; mt < MT && !CEIL; ++mt) {
      float rmax[2] = {m[mt][0], m[mt][1]};
      if (!need_bias && !need_mask) {
#pragma unroll
        for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float x = sc[mt][nt][e] * p.qk_scale;
            sc[mt][nt][e] = x;
            rmax[e >> 1] = fmaxf(rmax[e >> 1], x);
          }
        }
      } else {
        if (need_bias) {
          const int off = qmin_t[mt] - qmin;
          const int r0_l = g_lane, r1_l = r0_l + 8, c_l = t4 * 2;
#pragma unroll
          for (int np = 0; np < 4; ++np) {
            const int pr = off + np * 16 + (lane & 7) + ((lane >> 4) << 3);
            const int pc = (lane >> 3) & 1;
            unsigned pb[4];
            ldmatrix_x4(pb[0], pb[1], pb[2], pb[3], smem_addr(pbuf0 + pr * kDrel + ((pc ^ (pr & 1)) << 3)));
            float bc0[4] = {0.f, 0.f, 0.f, 0.f}, bc1[4] = {0.f, 0.f, 0.f, 0.f};
            mma_bf16(bc0, rf[mt], pb[0], pb[1]);
            mma_bf16(bc1, rf[mt], pb[2], pb[3]);
            *reinterpret_cast<uint32_t*>(b_warp + r0_l * kBscCols + (2 * np) * 8 + c_l) = pack_bf16x2(bc0[0], bc0[1]);
            *reinterpret_cast<uint32_t*>(b_warp + r1_l * kBscCols + (2 * np) * 8 + c_l) = pack_bf16x2(bc0[2], bc0[3]);
            *reinterpret_cast<uint32_t*>(b_warp + r0_l * kBscCols + (2 * np + 1) * 8 + c_l) = pack_bf16x2(bc1[0], bc1[1]);
            *reinterpret_cast<uint32_t*>(b_warp + r1_l * kBscCols + (2 * np + 1) * 8 + c_l) = pack_bf16x2(bc1[2], bc1[3]);
          }
          {
            const int pr = min(off + 64 + (lane & 7) + ((lane >> 4) << 3), kPbufRows - 1);
            const int pc = (lane >> 3) & 1;
            unsigned pb[4];
            ldmatrix_x4(pb[0], pb[1], pb[2], pb[3], smem_addr(pbuf0 + pr * kDrel + ((pc ^ (pr & 1)) << 3)));
            float bc0[4] = {0.f, 0.f, 0.f, 0.f};
            mma_bf16(bc0, rf[mt], pb[0], pb[1]);
            *reinterpret_cast<uint32_t*>(b_warp + r0_l * kBscCols + 64 + c_l) = pack_bf16x2(bc0[0], bc0[1]);
            *reinterpret_cast<uint32_t*>(b_warp + r1_l * kBscCols + 64 + c_l) = pack_bf16x2(bc0[2], bc0[3]);
          }
          __syncwarp();
        }
        const int sh0 = qpos[mt][0] - qmin_t[mt];
        const int sh1 = qpos[mt][1] - qmin_t[mt];
        const int bo0 = g_lane * kBscCols + sh0 + 63;
        const int bo1 = (g_lane + 8) * kBscCols + sh1 + 63;
#pragma unroll
        for (int nt = 0; nt < BK / 8; ++nt) {
          const int jl0 = nt * 8 + t4 * 2;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int r = e >> 1, jl = jl0 + (e & 1);
            float x = sc[mt][nt][e] * p.qk_scale;
            if (need_bias) {
              const uint16_t bv = b_warp[(r ? bo1 : bo0) - jl];
              x = fmaf(bf2f(bv), kLog2e, x);
            }
            if (need_mask) {
              const int d = qpos[mt][r] - (kmin + jl);
              x = (d >= 0 && (p.window == 0 || d < p.window)) ? x : -INFINITY;
            }
            sc[mt][nt][e] = x;
            rmax[r] = fmaxf(rmax[r], x);
          }
        }
      }
      float mref[2], alpha[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1));
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2));
        mref[r] = rmax[r] == -INFINITY ? 0.f : rmax[r];
        alpha[r] = ex2(m[mt][r] - mref[r]);
        m[mt][r] = rmax[r];
        l[mt][r] *= alpha[r];
      }
      if (!SKIP_RESCALE || __any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt) {
          o[mt][dt][0] *= alpha[0];
          o[mt][dt][1] *= alpha[0];
          o[mt][dt][2] *= alpha[1];
          o[mt][dt][3] *= alpha[1];
        }
      }
      float ts[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float pv = ex2(sc[mt][nt][e] - mref[e >> 1]);
          ts[e >> 1] += pv;
          sc[mt][nt][e] = pv;
        }
      }
      l[mt][0] += ts[0];
      l[mt][1] += ts[1];
    }

    cp_wait<0>();
    __syncthreads();
    if (i + 1 < ntiles) {
      load_kv8(i + 1, kbuf0, kbase);
      if (tile_needs_bias(i + 1)) load_p(i + 1, pbuf0);
    }
    cp_commit();

    // ---- O += P V (P f16, V decoded from E4M3 via place2)
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      unsigned a[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        a[mt][0] = pack_f16x2(sc[mt][2 * kk][0], sc[mt][2 * kk][1]);
        a[mt][1] = pack_f16x2(sc[mt][2 * kk][2], sc[mt][2 * kk][3]);
        a[mt][2] = pack_f16x2(sc[mt][2 * kk + 1][0], sc[mt][2 * kk + 1][1]);
        a[mt][3] = pack_f16x2(sc[mt][2 * kk + 1][2], sc[mt][2 * kk + 1][3]);
      }
      const int r0 = kk * 16 + 2 * t4;
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) {
        const int c16 = dt >> 1;
        const int woff = ((dt & 1) * 2 + (g_lane >> 2)) * 4;
        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(vbuf0 + swz8(r0, c16) + woff);
        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(vbuf0 + swz8(r0 + 1, c16) + woff);
        const uint32_t w8 = *reinterpret_cast<const uint32_t*>(vbuf0 + swz8(r0 + 8, c16) + woff);
        const uint32_t w9 = *reinterpret_cast<const uint32_t*>(vbuf0 + swz8(r0 + 9, c16) + woff);
        const unsigned b0 = place2(w0, w1, vsel);
        const unsigned b1 = place2(w8, w9, vsel);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma_f16(o[mt][dt], a[mt], b0, b1);
      }
    }
  }
  cp_wait<0>();

  // ---- split items merge (identical to prefill_attn_kernel; o stays in 2^-8 scale until epilogue)
  const int4 it2 = p.items2[blockIdx.x];
  if (it2.y > 1) {
    constexpr int NO = 16 * MT;
    auto wsp = [&](int part) { return p.ws + (size_t)(part * kKvHeads + g) * NT * (NO * 4 + 4 * MT); };
    {
      float4* w4 = reinterpret_cast<float4*>(wsp(it2.z + it2.x));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt)
          w4[(mt * 16 + dt) * NT + tid] = make_float4(o[mt][dt][0], o[mt][dt][1], o[mt][dt][2], o[mt][dt][3]);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) w4[(NO + mt) * NT + tid] = make_float4(m[mt][0], m[mt][1], l[mt][0], l[mt][1]);
    }
    __threadfence();
    __syncthreads();
    volatile int* s_last = reinterpret_cast<volatile int*>(smem_raw + C::BSCRATCH_OFF_B);
    if (tid == 0) {
      const int old = atomicAdd(p.cnt + it2.w * kKvHeads + g, 1);
      s_last[0] = old == it2.y - 1;
      if (old == it2.y - 1) p.cnt[it2.w * kKvHeads + g] = 0;
    }
    __syncthreads();
    if (!s_last[0]) return;
    __threadfence();
    float M[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) M[mt][0] = M[mt][1] = -INFINITY;
    for (int sp = 0; sp < it2.y; ++sp) {
      const float4* w4 = reinterpret_cast<const float4*>(wsp(it2.z + sp));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float4 ml = __ldcg(w4 + (NO + mt) * NT + tid);
        M[mt][0] = fmaxf(M[mt][0], ml.x);
        M[mt][1] = fmaxf(M[mt][1], ml.y);
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      l[mt][0] = l[mt][1] = 0.f;
      m[mt][0] = M[mt][0];
      m[mt][1] = M[mt][1];
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) o[mt][dt][0] = o[mt][dt][1] = o[mt][dt][2] = o[mt][dt][3] = 0.f;
    }
    for (int sp = 0; sp < it2.y; ++sp) {
      const float4* w4 = reinterpret_cast<const float4*>(wsp(it2.z + sp));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float4 ml = __ldcg(w4 + (NO + mt) * NT + tid);
        const float w0 = ml.x == -INFINITY ? 0.f : ex2(ml.x - M[mt][0]);
        const float w1 = ml.y == -INFINITY ? 0.f : ex2(ml.y - M[mt][1]);
        l[mt][0] = fmaf(ml.z, w0, l[mt][0]);
        l[mt][1] = fmaf(ml.w, w1, l[mt][1]);
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt) {
          const float4 v = __ldcg(w4 + (mt * 16 + dt) * NT + tid);
          o[mt][dt][0] = fmaf(v.x, w0, o[mt][dt][0]);
          o[mt][dt][1] = fmaf(v.y, w0, o[mt][dt][1]);
          o[mt][dt][2] = fmaf(v.z, w1, o[mt][dt][2]);
          o[mt][dt][3] = fmaf(v.w, w1, o[mt][dt][3]);
        }
      }
    }
  }

  // ---- epilogue: o = (acc / l) * 256 (V dequant fold), lse, gate then sink, one bf16 rounding
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float lr = l[mt][r];
      lr += __shfl_xor_sync(0xffffffffu, lr, 1);
      lr += __shfl_xor_sync(0xffffffffu, lr, 2);
      const int R = R0 + (warp * MT + mt) * 16 + g_lane + r * 8;
      if (R >= rows) continue;
      const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
      const bool empty = lr == 0.f;
      const float den = empty ? 1.f : lr;
      const float lse = (empty ? -INFINITY : m[mt][r] + lg2(lr)) * kLn2;
      const float fs = sigmoid_cfgb(lse - bf2f(reinterpret_cast<const uint16_t*>(p.sink)[size_t(t) * kHeads + h]));
      const uint16_t* grow = reinterpret_cast<const uint16_t*>(p.gate) + size_t(t) * (kHeads * kD) + h * kD;
      uint16_t* orow = reinterpret_cast<uint16_t*>(p.out) + size_t(t) * (kHeads * kD) + h * kD;
      if (p.lse && t4 == 0) p.lse[size_t(t) * kHeads + h] = lse;
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) {
        const int col = dt * 8 + t4 * 2;
        const float o0 = (o[mt][dt][2 * r] / den) * 256.0f, o1 = (o[mt][dt][2 * r + 1] / den) * 256.0f;
        if (p.o32) *reinterpret_cast<float2*>(p.o32 + (size_t(t) * kHeads + h) * kD + col) = make_float2(o0, o1);
        const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + col);
        const float y0 = (o0 * sigmoid_cfgb(bf_lo(gw))) * fs;
        const float y1 = (o1 * sigmoid_cfgb(bf_hi(gw))) * fs;
        *reinterpret_cast<uint32_t*>(orow + col) = pack_bf16x2(y0, y1);
      }
    }
}

// ------------------------------------------------------------------------------------------------ FAR-only FP8 kernel
// Global-layer unmasked, bias-free E4M3 KV prefill attention for FAR items (qmin >= it.w * 64 + 1023, window == 0).
// Eliminates all register spills (local = 0 vs 8 B), cuts smem to 49,168 B (2 CTAs/SM), bit-exact vs prefill_attn_fp8_kernel.
template <int NW, int MT>
struct CfgFp8Far {
  static constexpr int NT = NW * 32, BR = NW * 16 * MT, BK = 64;
  static constexpr int TILE_B = BK * kD;        // 8,192 B per E4M3 K or V tile
  static constexpr int KV_B = 2 * TILE_B;       // 16,384 B
  static constexpr int Q_B = BR * kD * 2;       // 32,768 B per f16 Q tile
  static constexpr int Q_OFF_B = KV_B;
  static constexpr int SMEM = KV_B + Q_B + 16;  // 49,168 B
};

template <int NW, int MT, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB) prefill_attn_fp8_far_kernel(const Params p) {
  static_assert(MT == 2, "prefill_attn_fp8_far_kernel expects MT=2");
  using C = CfgFp8Far<NW, MT>;
  constexpr int NT = C::NT, BR = C::BR, BK = C::BK, DCH8 = kD / 16;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  uint8_t* kbuf0 = smem_raw;
  uint8_t* vbuf0 = smem_raw + C::TILE_B;
  uint16_t* qs = reinterpret_cast<uint16_t*>(smem_raw + C::Q_OFF_B);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g_lane = lane >> 2, t4 = lane & 3;
  const int g = blockIdx.y;
  const int4 it = p.items[blockIdx.x];
  const Seg sg = p.segs[it.x];
  const int rows = kGqa * sg.len;
  const int R0 = it.y;
  const int* pt = p.page_table + sg.pt_off;

#pragma unroll
  for (int i = 0; i < (BR * DCH) / NT; ++i) {
    const int id = tid + i * NT, r = id >> 4, c = id & 15;
    const int R = min(R0 + r, rows - 1);
    const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
    cp_async16(qs + swz(r, c), p.q + (size_t(t) * kHeads + h) * kD + c * 8);
  }
  cp_commit();
  const int first = it.z;
  const int ntiles = it.w - first;

  cp_wait<0>();
  __syncthreads();
#pragma unroll 1
  for (int j = 0; j < (BR * (DCH / 2)) / NT; ++j) {
    const int idx = tid + j * NT;
    const int r = idx >> 3, kk = idx & 7;
    uint4* p0 = reinterpret_cast<uint4*>(qs + swz(r, 2 * kk));
    uint4* p1 = reinterpret_cast<uint4*>(qs + swz(r, 2 * kk + 1));
    const uint4 u0 = *p0, u1 = *p1;
    const uint32_t h0 = bf2_to_h2(u0.x), h1 = bf2_to_h2(u0.y), h2 = bf2_to_h2(u0.z), h3 = bf2_to_h2(u0.w);
    const uint32_t h4 = bf2_to_h2(u1.x), h5 = bf2_to_h2(u1.y), h6 = bf2_to_h2(u1.z), h7 = bf2_to_h2(u1.w);
    *p0 = make_uint4(h0, h2, h4, h6);
    *p1 = make_uint4(h1, h3, h5, h7);
  }
  __syncthreads();

  const uint8_t* kbase = reinterpret_cast<const uint8_t*>(p.k);
  const uint8_t* vbase = reinterpret_cast<const uint8_t*>(p.v);
  const long long kv_thread = (long long)(tid >> 3) * p.s_slot + (long long)g * p.s_head + (tid & 7) * 16;
  auto load_kv8 = [&](int i, uint8_t* dst, const uint8_t* src) {
    const int k0 = (first + i) * BK;
    const uint8_t* base = src + (long long)__ldg(pt + (k0 >> p.page_shift)) * p.s_page +
                          (long long)(k0 & ((1 << p.page_shift) - 1)) * p.s_slot + kv_thread;
#pragma unroll
    for (int j = 0; j < (BK * DCH8) / NT; ++j) {
      const int row = (tid >> 3) + j * (NT / DCH8);
      cp_async16(dst + swz8(row, tid & 7), base + (long long)j * (NT / DCH8) * p.s_slot, 16);
    }
  };

  float o[MT][kD / 8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) o[mt][i][0] = o[mt][i][1] = o[mt][i][2] = o[mt][i][3] = 0.f;
  float m[MT][2], l[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) m[mt][0] = m[mt][1] = -INFINITY, l[mt][0] = l[mt][1] = 0.f;

  const uint32_t vsel = psel(g_lane & 3, 4 + (g_lane & 3));
  if (ntiles > 0) load_kv8(0, kbuf0, kbase);
  cp_commit();

#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    cp_wait<0>();
    __syncthreads();
    load_kv8(i, vbuf0, vbase);
    cp_commit();

    float sc[MT][BK / 8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) sc[mt][nt][0] = sc[mt][nt][1] = sc[mt][nt][2] = sc[mt][nt][3] = 0.f;
#pragma unroll 1
    for (int kk = 0; kk < DCH / 2; ++kk) {
      unsigned a[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                    smem_addr(qs + swz((warp * MT + mt) * 16 + (lane & 15), kk * 2 + (lane >> 4))));
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(kbuf0 + swz8(nt * 8 + g_lane, kk) + t4 * 4);
        const unsigned b0 = place2(w, w, psel(0, 1));
        const unsigned b1 = place2(w, w, psel(2, 3));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma_f16(sc[mt][nt], a[mt], b0, b1);
      }
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      float rmax[2] = {m[mt][0], m[mt][1]};
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float x = sc[mt][nt][e] * p.qk_scale;
          sc[mt][nt][e] = x;
          rmax[e >> 1] = fmaxf(rmax[e >> 1], x);
        }
      }
      float mref[2], alpha[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1));
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2));
        mref[r] = rmax[r];
        alpha[r] = ex2(m[mt][r] - mref[r]);
        m[mt][r] = rmax[r];
        l[mt][r] *= alpha[r];
      }
      if (!SKIP_RESCALE || __any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt) {
          o[mt][dt][0] *= alpha[0];
          o[mt][dt][1] *= alpha[0];
          o[mt][dt][2] *= alpha[1];
          o[mt][dt][3] *= alpha[1];
        }
      }
      float ts[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float pv = ex2(sc[mt][nt][e] - mref[e >> 1]);
          ts[e >> 1] += pv;
          sc[mt][nt][e] = pv;
        }
      }
      l[mt][0] += ts[0];
      l[mt][1] += ts[1];
    }

    cp_wait<0>();
    __syncthreads();
    if (i + 1 < ntiles) load_kv8(i + 1, kbuf0, kbase);
    cp_commit();

#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      unsigned a[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        a[mt][0] = pack_f16x2(sc[mt][2 * kk][0], sc[mt][2 * kk][1]);
        a[mt][1] = pack_f16x2(sc[mt][2 * kk][2], sc[mt][2 * kk][3]);
        a[mt][2] = pack_f16x2(sc[mt][2 * kk + 1][0], sc[mt][2 * kk + 1][1]);
        a[mt][3] = pack_f16x2(sc[mt][2 * kk + 1][2], sc[mt][2 * kk + 1][3]);
      }
      const int r0 = kk * 16 + 2 * t4;
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) {
        const int c16 = dt >> 1;
        const int woff = ((dt & 1) * 2 + (g_lane >> 2)) * 4;
        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(vbuf0 + swz8(r0, c16) + woff);
        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(vbuf0 + swz8(r0 + 1, c16) + woff);
        const uint32_t w8 = *reinterpret_cast<const uint32_t*>(vbuf0 + swz8(r0 + 8, c16) + woff);
        const uint32_t w9 = *reinterpret_cast<const uint32_t*>(vbuf0 + swz8(r0 + 9, c16) + woff);
        const unsigned b0 = place2(w0, w1, vsel);
        const unsigned b1 = place2(w8, w9, vsel);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma_f16(o[mt][dt], a[mt], b0, b1);
      }
    }
  }
  cp_wait<0>();

  const int4 it2 = p.items2[blockIdx.x];
  if (it2.y > 1) {
    constexpr int NO = 16 * MT;
    auto wsp = [&](int part) { return p.ws + (size_t)(part * kKvHeads + g) * NT * (NO * 4 + 4 * MT); };
    {
      float4* w4 = reinterpret_cast<float4*>(wsp(it2.z + it2.x));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt)
          w4[(mt * 16 + dt) * NT + tid] = make_float4(o[mt][dt][0], o[mt][dt][1], o[mt][dt][2], o[mt][dt][3]);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) w4[(NO + mt) * NT + tid] = make_float4(m[mt][0], m[mt][1], l[mt][0], l[mt][1]);
    }
    __threadfence();
    __syncthreads();
    volatile int* s_last = reinterpret_cast<volatile int*>(smem_raw);
    if (tid == 0) {
      const int old = atomicAdd(p.cnt + it2.w * kKvHeads + g, 1);
      s_last[0] = old == it2.y - 1;
      if (old == it2.y - 1) p.cnt[it2.w * kKvHeads + g] = 0;
    }
    __syncthreads();
    if (!s_last[0]) return;
    __threadfence();
    float M[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) M[mt][0] = M[mt][1] = -INFINITY;
    for (int sp = 0; sp < it2.y; ++sp) {
      const float4* w4 = reinterpret_cast<const float4*>(wsp(it2.z + sp));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float4 ml = __ldcg(w4 + (NO + mt) * NT + tid);
        M[mt][0] = fmaxf(M[mt][0], ml.x);
        M[mt][1] = fmaxf(M[mt][1], ml.y);
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      l[mt][0] = l[mt][1] = 0.f;
      m[mt][0] = M[mt][0];
      m[mt][1] = M[mt][1];
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) o[mt][dt][0] = o[mt][dt][1] = o[mt][dt][2] = o[mt][dt][3] = 0.f;
    }
    for (int sp = 0; sp < it2.y; ++sp) {
      const float4* w4 = reinterpret_cast<const float4*>(wsp(it2.z + sp));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float4 ml = __ldcg(w4 + (NO + mt) * NT + tid);
        const float w0 = ml.x == -INFINITY ? 0.f : ex2(ml.x - M[mt][0]);
        const float w1 = ml.y == -INFINITY ? 0.f : ex2(ml.y - M[mt][1]);
        l[mt][0] = fmaf(ml.z, w0, l[mt][0]);
        l[mt][1] = fmaf(ml.w, w1, l[mt][1]);
#pragma unroll
        for (int dt = 0; dt < kD / 8; ++dt) {
          const float4 v = __ldcg(w4 + (mt * 16 + dt) * NT + tid);
          o[mt][dt][0] = fmaf(v.x, w0, o[mt][dt][0]);
          o[mt][dt][1] = fmaf(v.y, w0, o[mt][dt][1]);
          o[mt][dt][2] = fmaf(v.z, w1, o[mt][dt][2]);
          o[mt][dt][3] = fmaf(v.w, w1, o[mt][dt][3]);
        }
      }
    }
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float lr = l[mt][r];
      lr += __shfl_xor_sync(0xffffffffu, lr, 1);
      lr += __shfl_xor_sync(0xffffffffu, lr, 2);
      const int R = R0 + (warp * MT + mt) * 16 + g_lane + r * 8;
      if (R >= rows) continue;
      const int t = sg.q0 + R / kGqa, h = g * kGqa + R % kGqa;
      const bool empty = lr == 0.f;
      const float den = empty ? 1.f : lr;
      const float lse = (empty ? -INFINITY : m[mt][r] + lg2(lr)) * kLn2;
      const float fs = sigmoid_cfgb(lse - bf2f(reinterpret_cast<const uint16_t*>(p.sink)[size_t(t) * kHeads + h]));
      const uint16_t* grow = reinterpret_cast<const uint16_t*>(p.gate) + size_t(t) * (kHeads * kD) + h * kD;
      uint16_t* orow = reinterpret_cast<uint16_t*>(p.out) + size_t(t) * (kHeads * kD) + h * kD;
      if (p.lse && t4 == 0) p.lse[size_t(t) * kHeads + h] = lse;
#pragma unroll
      for (int dt = 0; dt < kD / 8; ++dt) {
        const int col = dt * 8 + t4 * 2;
        const float o0 = (o[mt][dt][2 * r] / den) * 256.0f, o1 = (o[mt][dt][2 * r + 1] / den) * 256.0f;
        if (p.o32) *reinterpret_cast<float2*>(p.o32 + (size_t(t) * kHeads + h) * kD + col) = make_float2(o0, o1);
        const uint32_t gw = *reinterpret_cast<const uint32_t*>(grow + col);
        const float y0 = (o0 * sigmoid_cfgb(bf_lo(gw))) * fs;
        const float y1 = (o1 * sigmoid_cfgb(bf_hi(gw))) * fs;
        *reinterpret_cast<uint32_t*>(orow + col) = pack_bf16x2(y0, y1);
      }
    }
}

}  // namespace pfa
