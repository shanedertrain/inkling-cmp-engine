// inkling-cmp-engine S4: prefill expert (MoE) grouped GEMM for sm_80 (CMP 170HX), own code.
//
// Math (config B, docs/engine/configB-numerics.md "Expert gate/up, down", "SiTU", "Combine"):
//   w13:  gu[r, n] = bf16( (sum_k x[t(r), k] * Wb[e(r), n, k]) * factor[e] )      fp32 accumulation, Wb exact in bf16
//         h[r, i]  = bf16( ((softcap4(g) * sigmoid(g)) * softcap25(u)) * gamma[r] ), g = gu[r, 2i], u = gu[r, 2i+1]
//   w2:   y[r, n]  = bf16( (sum_k h[r, k] * Wb2[e(r), n, k]) * factor2[e] )
//   out[t, n] = bf16( 0 + y[8t + 0, n] + y[8t + 1, n] + ... + y[8t + 7, n] )   (fp32, slot order)
//   NVFP4: Wb = e2m1 * e4m3 block scale (per 16 k), factor = amax / denominator; MXFP8: Wb = e4m3 * 2^(e8m0 - 127)
//   (per 32 k), factor = 1. r = flat row 8 t + slot.
//
// Structure ("swap AB"): the weights are the A operand of mma.m16n8k16 (16 output features x 16 k), the token rows
// the B operand (8 rows per n-tile), so an expert's rows are covered in 8-row steps (no 64/128-row padding) and each
// decoded weight fragment feeds 8 MMAs. Weights are pre-permuted at load (repack_* in bench.cu) into the exact
// per-lane fragment order: one 32-bit word = a lane's 8 NVFP4 weights of one m16 x k16 fragment (MXFP8: 2 words), so
// the shared -> register path is one LDS.128 per m16 tile per 64-k stage and the decode is shifts / masks + one bf16
// multiply by the block scale.
//
// Exactness of the decode: the e2m1 code is placed in a bf16 as sign | exponent bits | mantissa bit (value = v *
// 2^-126, bf16 subnormal for v = 0.5) and multiplied (bf16 fma, subnormals supported on sm_80) by the block scale
// pre-biased by 2^118: the product v * s * 2^-8 has <= 6 significant bits and is a normal bf16, so it is exact.
// MXFP8: e4m3 bits -> bf16 (value w * 2^-120) times 2^(e8m0 - 15) = w * 2^(e8m0 - 127) * 2^-8, exact. The fp32
// accumulator is thus 2^-8 x config B's; the epilogue multiplies by factor * 256 (exactly config B's product
// acc * factor, one rounding) then rounds to bf16 once.
//
// Invariance: every output element is one warp's mma chain over k in fixed order (k-steps of 64, then 4 k16 in
// order); no split-K, no atomics, no dependence on the row's position in its tile, on the tile size or on other
// rows. So a row's output depends only on its own input row, its expert and the layer weights (bit-for-bit
// independent of chunking and batch composition). Tested in bench.cu (--mode inv).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace pmoe {

constexpr int H = 2304;        // hidden (w13 K, w2 N)
constexpr int I = 1152;        // intermediate (w13 N = 2 I, w2 K)
constexpr int E = 128;
constexpr int TOPK = 8;
constexpr int BK = 64;         // k per pipeline stage
constexpr int WT = 64;         // weight rows per warp tile (4 m16 tiles)
constexpr int BN = 128;        // max token rows per CTA tile (2 row-warps x 8 n8 tiles, interleaved)

enum Fmt { NVFP4 = 0, MXFP8 = 1, NVFP4B = 2, NVFP4_N1S1 = 3 };   // NVFP4B: n1 + bf16 scales; NVFP4_N1S1: n1 + 1B scales
enum Mode { W13_SITU = 0, W13_PLAIN = 1, W2_PLAIN = 2 };

// bytes per (expert, warp tile, k-stage) block
template <int FMT> struct FmtInfo;
template <> struct FmtInfo<NVFP4> { static constexpr int WBYTES = WT * BK / 2; static constexpr int SBYTES = WT * BK / 16; };
template <> struct FmtInfo<MXFP8> { static constexpr int WBYTES = WT * BK; static constexpr int SBYTES = WT * BK / 32; };
template <> struct FmtInfo<NVFP4B> { static constexpr int WBYTES = WT * BK / 2; static constexpr int SBYTES = WT * BK / 16 * 2; };
template <> struct FmtInfo<NVFP4_N1S1> { static constexpr int WBYTES = WT * BK / 2; static constexpr int SBYTES = WT * BK / 16; };

struct Item {             // one CTA tile: expert e, weight tile ft (WF warp tiles), sorted rows [p0, p0 + cnt)
  int e, ft, p0, cnt;
};

struct Params {
  const __nv_bfloat16* A;     // activations: x [Q, H] (w13) or h [R, I] sorted (w2)
  const int* a_row;           // per sorted position p: row of A (a_row * K / 8 < 2^31: the kernels address A in 16-byte chunks)
  const uint8_t* W;           // repacked weights [E][N / 64][K / 64] blocks
  const uint8_t* S;           // repacked scales, same block order
  const float* F;             // per expert: factor * 256 (NVFP4) or 256 (MXFP8)
  const Item* items;
  int n_items;                // item count (host-built lists; the launch grid)
  const int* n_items_dev;     // optional: item count in device memory (device-built lists, grid = an upper bound):
                              // CTAs at or past it exit before reading an item
  int K;                      // reduction length
  int NWT;                    // weight tiles (of 64 rows) per expert = N / 64
  // outputs
  __nv_bfloat16* out;         // W13_SITU: h [R, I] sorted; *_PLAIN: [R, N] at row out_row[p]
  const int* out_row;         // per sorted position p (PLAIN modes)
  const float* gamma;         // per sorted position p (W13_SITU)
};

// ------------------------------------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {   // bf16x2 a * b (fma.rn.bf16x2 with -0)
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// NVFP4 fragment word -> 4 bf16x2 (values v * 2^-126). reg j: low half from nibble bits [12-4j, 15-4j], high half
// from [28-4j, 31-4j]; bf16 bits = sign << 15 | (code & 7) << 6.
__device__ __forceinline__ void dec_nvfp4(uint32_t q, uint32_t* r) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t v = q << (4 * j);
    r[j] = (v & 0x80008000u) | ((v >> 6) & 0x01C001C0u);
  }
}
// NVFP4B ("n1" layout): reg 0 = q & M (code bits 0-2 at 6-8, sign at 15; +16 for the high half); reg 1 the same at
// -3 (bits 3-5, 12), reg 2 at -6 (bits 0-2, 9), reg 3 from the leftover bits (code bits 0, 1 at 10, 11; bit 2 at 13;
// sign at 14). 11 integer ops for 4 bf16x2 instead of ~14.
__device__ __forceinline__ void dec_nvfp4b(uint32_t q, uint32_t* r) {
  r[0] = q & 0x81C081C0u;
  r[1] = (q << 3) & 0x81C081C0u;
  r[2] = (q << 6) & 0x81C081C0u;
  r[3] = ((q >> 4) & 0x00C000C0u) | ((q >> 5) & 0x01000100u) | ((q << 1) & 0x80008000u);
}
// MXFP8: q0 = bytes (g,2t) (g,2t+1) (g+8,2t) (g+8,2t+1) as [b1 b3] -> reg0 (row g), [b0 b2] -> reg1 (row g+8);
// e4m3 bits -> bf16 bits sign << 15 | (code & 0x7f) << 4 (value w * 2^-120).
__device__ __forceinline__ void dec_mxfp8(uint32_t q, uint32_t& hi, uint32_t& lo) {
  hi = (q & 0x80008000u) | ((q >> 4) & 0x07F007F0u);
  lo = ((q << 8) & 0x80008000u) | ((q << 4) & 0x07F007F0u);
}

// ------------------------------------------------------------------------------------------------ SiTU (config B)
// activation.py _sigmoid_approx / _div_rn / _softcap / _situ, op for op (mul.rn, ex2.approx.ftz, add.rn,
// rcp.approx.ftz, tanh.approx; no contraction).
__device__ __forceinline__ float ex2_ftz(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float rcp_ftz(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float tanh_ap(float x) { float r; asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float situ(float g, float u, float gamma) {
  float t = __fmul_rn(g, __uint_as_float(0xBFB8AA3Bu));
  t = ex2_ftz(t);
  t = __fadd_rn(t, 1.0f);
  float sig = rcp_ftz(t);
  float gc = __fmul_rn(4.0f, tanh_ap(__fmul_rn(g, 0.25f)));
  float q = __fmul_rn(u, 1.0f / 25.0f);
  float res = fmaxf(__fmaf_rn(q, 25.0f, -u), -3.4028234663852886e38f);
  q = __fmaf_rn(-res, 1.0f / 25.0f, q);
  float uc = __fmul_rn(25.0f, tanh_ap(q));
  return __fmul_rn(__fmul_rn(__fmul_rn(gc, sig), uc), gamma);
}

// ------------------------------------------------------------------------------------------------ the GEMM
// WF warp tiles of 64 weight rows per CTA (features), 2 row-warps per weight tile; CTA = 2 WF warps.
// Row-warp wr takes n8 tiles wr, wr + 2, ..., wr + 14 of the CTA's (up to) 128 rows.
template <int FMT, int MODE, int WF, int STAGES>
struct Cfg {
  static constexpr int WARPS = 2 * WF;
  static constexpr int THREADS = 32 * WARPS;
  static constexpr int XBYTES = BN * BK * 2;                       // 16 KB
  static constexpr int WBYTES = WF * FmtInfo<FMT>::WBYTES;
  static constexpr int SBYTES = WF * FmtInfo<FMT>::SBYTES;
  static constexpr int STAGE = XBYTES + WBYTES + SBYTES;
  static constexpr int EPI = MODE == W13_SITU ? BN * (WF * 32) * 2 : BN * (WF * 64) * 2;   // output tile bf16
  static constexpr int SMEM = (STAGE * STAGES > EPI ? STAGE * STAGES : EPI) + BN * 4 * 2;  // + row tables
};

template <int FMT, int MODE, int WF, int STAGES>
__global__ void __launch_bounds__(64 * WF, WF == 2 ? 2 : 1) moe_gemm(Params p) {
  using C = Cfg<FMT, MODE, WF, STAGES>;
  extern __shared__ __align__(128) uint8_t smem[];
  int* s_arow = reinterpret_cast<int*>(smem + C::SMEM - BN * 8);   // A row offsets (elements / 8 = 16-B chunks)
  int* s_orow = s_arow + BN;

  if ((int)blockIdx.x >= (p.n_items_dev ? *p.n_items_dev : p.n_items)) return;
  const Item it = p.items[blockIdx.x];
  if (it.cnt > BN) __trap();   // item contract: at most BN rows (the shared row tables and epilogue tile)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wf = warp >> 1, wr = warp & 1;
  const int K = p.K, KS = K / BK;

  // row tables
  for (int i = tid; i < BN; i += C::THREADS) {
    int pp = it.p0 + (i < it.cnt ? i : 0);
    s_arow[i] = p.a_row[pp] * (p.K / 8);   // 16-byte chunk offset of the A row (< 2^31: rows < 2^34 / K)
    if (MODE != W13_SITU) s_orow[i] = p.out_row[pp];
  }
  __syncthreads();

  const uint8_t* wbase = p.W + (size_t)(it.e * p.NWT + it.ft * WF) * KS * FmtInfo<FMT>::WBYTES;
  const uint8_t* sbase = p.S + (size_t)(it.e * p.NWT + it.ft * WF) * KS * FmtInfo<FMT>::SBYTES;
  const size_t wt_stride_w = (size_t)KS * FmtInfo<FMT>::WBYTES;   // next warp tile, same k-stage
  const size_t wt_stride_s = (size_t)KS * FmtInfo<FMT>::SBYTES;

  auto load_stage = [&](int ks, int slot) {
    uint8_t* st = smem + slot * C::STAGE;
    const uint32_t xs = smem_u32(st);
    // X: 128 rows x 8 chunks, swizzled chunk' = chunk ^ (row & 7)
#pragma unroll
    for (int j = 0; j < BN * 8 / C::THREADS; ++j) {
      int idx = tid + j * C::THREADS;
      int row = idx >> 3, ch = idx & 7;
      const uint4* src = reinterpret_cast<const uint4*>(p.A) + (s_arow[row] + ks * (BK / 8) + ch);
      cp_async16(xs + row * 128 + ((ch ^ (row & 7)) << 4), src, row < it.cnt ? 16 : 0);
    }
    const uint32_t ws = xs + C::XBYTES;
    constexpr int WCH = FmtInfo<FMT>::WBYTES / 16;
#pragma unroll
    for (int j = 0; j < (WF * WCH + C::THREADS - 1) / C::THREADS; ++j) {
      int idx = tid + j * C::THREADS;
      if (idx < WF * WCH) {
        int w = idx / WCH, c = idx % WCH;
        cp_async16(ws + idx * 16, wbase + w * wt_stride_w + (size_t)ks * FmtInfo<FMT>::WBYTES + c * 16, 16);
      }
    }
    const uint32_t ss = ws + C::WBYTES;
    constexpr int SCH = FmtInfo<FMT>::SBYTES / 16;
    if (tid < WF * SCH) {
      int w = tid / SCH, c = tid % SCH;
      cp_async16(ss + tid * 16, sbase + w * wt_stride_s + (size_t)ks * FmtInfo<FMT>::SBYTES + c * 16, 16);
    }
  };

  const int n_tiles = (it.cnt + 7) >> 3;                 // n8 tiles in this CTA tile
  const int my_nt = n_tiles > wr ? (n_tiles - wr + 1) >> 1 : 0;   // tiles wr, wr+2, ... below n_tiles

  float acc[4][8][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KS) load_stage(s, s);
    cp_async_commit();
  }

  // ldmatrix lane address parts: matrix m = lane >> 3 (tile = m >> 1, k-half = m & 1), row-in-matrix = lane & 7
  const int lm = lane >> 3, lr = lane & 7;
  const int g = lane >> 2;

  for (int ks = 0; ks < KS; ++ks) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      int nk = ks + STAGES - 1;
      if (nk < KS) load_stage(nk, nk % STAGES);
      cp_async_commit();
    }
    const uint8_t* st = smem + (ks % STAGES) * C::STAGE;
    const uint32_t xs = smem_u32(st);
    const uint8_t* ws = st + C::XBYTES + wf * FmtInfo<FMT>::WBYTES;
    const uint8_t* ss = st + C::XBYTES + C::WBYTES + wf * FmtInfo<FMT>::SBYTES;

    // weight words for this lane: [mt][kk]
    uint32_t wq[4][FMT == NVFP4 ? 4 : 8];
    uint32_t sw[4][FMT == NVFP4 ? 2 : 1];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      if (FMT == NVFP4) {
        uint4 v = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 16);
        wq[mt][0] = v.x; wq[mt][1] = v.y; wq[mt][2] = v.z; wq[mt][3] = v.w;
        uint2 sv = *reinterpret_cast<const uint2*>(ss + (mt * 8 + g) * 8);
        sw[mt][0] = sv.x; sw[mt][1] = sv.y;
      } else {
        uint4 v0 = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 32);
        uint4 v1 = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 32 + 16);
        wq[mt][0] = v0.x; wq[mt][1] = v0.y; wq[mt][2] = v0.z; wq[mt][3] = v0.w;
        wq[mt][4] = v1.x; wq[mt][5] = v1.y; wq[mt][6] = v1.z; wq[mt][7] = v1.w;
        sw[mt][0] = *reinterpret_cast<const uint32_t*>(ss + (mt * 8 + g) * 4);
      }
    }

#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      // B fragments for my n8 tiles (pairs of tiles per ldmatrix.x4)
      uint32_t b[8][2];
#pragma unroll
      for (int pr = 0; pr < 4; ++pr) {
        int t0 = 2 * pr;                     // my tile indices t0, t0 + 1 -> CTA tiles wr + 2 t
        if (t0 < my_nt) {
          int ct = wr + 2 * (t0 + (lm >> 1));
          int row = ct * 8 + lr;
          int ch = 2 * kk + (lm & 1);
          ldmatrix_x4(xs + row * 128 + ((ch ^ (row & 7)) << 4), b[t0][0], b[t0][1], b[t0 + 1][0], b[t0 + 1][1]);
        }
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        uint32_t a[4];
        if (FMT == NVFP4) {
          dec_nvfp4(wq[mt][kk], a);
          // scales: byte (kk, row g) and (kk, row g + 8) at bytes 2kk, 2kk + 1 of sw[mt][kk >> 1]
          uint32_t sv = sw[mt][kk >> 1];
          uint32_t sel_lo = (kk & 1) ? 0x4242u : 0x4040u;     // [b_g, 0, b_g, 0] (byte 4 of {sv, 0} = 0)
          uint32_t sel_hi = (kk & 1) ? 0x4343u : 0x4141u;
          uint32_t s_g = prmt(sv, 0u, sel_lo) * 16u + 0x76007600u;
          uint32_t s_h = prmt(sv, 0u, sel_hi) * 16u + 0x76007600u;
          a[0] = bmul2(a[0], s_g);
          a[1] = bmul2(a[1], s_h);
          a[2] = bmul2(a[2], s_g);
          a[3] = bmul2(a[3], s_h);
        } else {
          dec_mxfp8(wq[mt][2 * kk], a[0], a[1]);
          dec_mxfp8(wq[mt][2 * kk + 1], a[2], a[3]);
          uint32_t sv = sw[mt][0];                   // bytes: [grp0 g, grp0 g+8, grp1 g, grp1 g+8]
          uint32_t sel_lo = (kk >> 1) ? 0x4242u : 0x4040u;
          uint32_t sel_hi = (kk >> 1) ? 0x4343u : 0x4141u;
          // bf16 bits (c + 112) << 7 in both halves: prmt gives [c, 0, c, 0]
          uint32_t s_g = prmt(sv, 0u, sel_lo) * 128u + 0x38003800u;
          uint32_t s_h = prmt(sv, 0u, sel_hi) * 128u + 0x38003800u;
          a[0] = bmul2(a[0], s_g);
          a[1] = bmul2(a[1], s_h);
          a[2] = bmul2(a[2], s_g);
          a[3] = bmul2(a[3], s_h);
        }
#pragma unroll
        for (int t = 0; t < 8; ++t)
          if (t < my_nt) mma_bf16(acc[mt][t], a, b[t][0], b[t][1]);
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // ---------------------------------------------------------------- epilogue
  const float fe = p.F[it.e];
  const int t4 = lane & 3;
  __nv_bfloat16* so = reinterpret_cast<__nv_bfloat16*>(smem);
  if (MODE == W13_SITU) {
    // tile [BN rows][WF * 32 features]; feature (local) = wf * 32 + mt * 8 + g; rows (g) = gate, (g + 8) = up
    constexpr int FW = WF * 32;
    float gm[8][2];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      int r0 = (wr + 2 * t) * 8 + 2 * t4;
      gm[t][0] = r0 < it.cnt ? p.gamma[it.p0 + r0] : 0.f;
      gm[t][1] = r0 + 1 < it.cnt ? p.gamma[it.p0 + r0 + 1] : 0.f;
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      int f = wf * 32 + mt * 8 + g;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        if (t < my_nt) {
          int r0 = (wr + 2 * t) * 8 + 2 * t4;
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            float gv = __bfloat162float(__float2bfloat16_rn(acc[mt][t][c] * fe));
            float uv = __bfloat162float(__float2bfloat16_rn(acc[mt][t][c + 2] * fe));
            so[(r0 + c) * FW + (f ^ ((((r0 + c) >> 1) & 3) << 3))] = __float2bfloat16_rn(situ(gv, uv, gm[t][c]));
          }
        }
      }
    }
    __syncthreads();
    // rows of FW features (FW * 2 bytes) to h[p0 + r, ft * FW ...]
    constexpr int CPR = FW * 2 / 16;     // 16-byte chunks per row
    for (int idx = tid; idx < it.cnt * CPR; idx += C::THREADS) {
      int r = idx / CPR, c = idx % CPR;
      // un-swizzle: chunk c holds features 8c..8c+7, stored at f ^ (sw << 3) -> chunk c ^ sw
      int swz = (r >> 1) & 3;
      uint4 v = *reinterpret_cast<const uint4*>(so + r * FW + ((c ^ swz) << 3));
      *reinterpret_cast<uint4*>(p.out + (size_t)(it.p0 + r) * I + it.ft * FW + c * 8) = v;
    }
  } else {
    // tile [BN rows][WF * 64 out features]
    constexpr int FW = WF * 64;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        int lf = mt * 16 + g + 8 * hh;          // row within the warp tile
        int f;
        if (MODE == W13_PLAIN) {                // undo the gate/up permutation: local (g) = gate i, (g+8) = up i
          int i = wf * 32 + mt * 8 + g;
          f = 2 * i + hh;
        } else {
          f = wf * 64 + lf;
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          if (t < my_nt) {
            int r0 = (wr + 2 * t) * 8 + 2 * t4;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
              int r = r0 + c;
              so[r * FW + (f ^ (((r >> 1) & 7) << 3))] = __float2bfloat16_rn(acc[mt][t][2 * hh + c] * fe);
            }
          }
        }
      }
    }
    __syncthreads();
    constexpr int CPR = FW * 2 / 16;
    const int N = p.NWT * 64;
    for (int idx = tid; idx < it.cnt * CPR; idx += C::THREADS) {
      int r = idx / CPR, c = idx % CPR;
      int swz = (r >> 1) & 7;
      uint4 v = *reinterpret_cast<const uint4*>(so + r * FW + ((c ^ swz) << 3));
      *reinterpret_cast<uint4*>(p.out + (size_t)s_orow[r] * N + it.ft * FW + c * 8) = v;
    }
  }
}

// ------------------------------------------------------------------------------------------------ v2 GEMM
// CTA = 4 warps, 128 weight rows (2 warp tiles of 64) x up to 128 token rows. Warp w owns weight rows 32w..32w+31
// (warp tile w >> 1, m16 tiles 2 (w & 1) + {0, 1}) for ALL the CTA's token rows, so each weight fragment is decoded
// once per CTA and feeds 2 x NT MMAs; NT (even number of n8 tiles, 2..16) is a template parameter, so no MMA is
// predicated (v1's predicated MMAs cost a WARPSYNC each) and X rows beyond 8 NT are not even loaded.
// WARPS = 4 (128 weight rows, 2 CTAs / SM) or 8 (256 weight rows, 1 CTA / SM: half the X traffic per FLOP).
// DBG: 0 = real kernel; 1 = ceiling arm without the weight decode (raw words used as bf16, WRONG results);
// 2 = ceiling arm without the X ldmatrix (one B fragment per stage reused, WRONG results).
template <int FMT, int MODE, int STAGES, int WARPS = 4>
struct Cfg2 {
  static constexpr int THREADS = 32 * WARPS;
  static constexpr int WTS = WARPS / 2;                 // warp tiles (64 weight rows) per CTA
  static constexpr int XBYTES = BN * BK * 2;
  static constexpr int WBYTES = WTS * FmtInfo<FMT>::WBYTES;
  static constexpr int SBYTES = WTS * FmtInfo<FMT>::SBYTES;
  static constexpr int STAGE = XBYTES + WBYTES + SBYTES;
  static constexpr int FWS = WTS * 32, FWP = WTS * 64;   // output features per CTA tile (SiTU / plain)
  static constexpr int EPI = MODE == W13_SITU ? BN * FWS * 2 : BN * FWP * 2;
  static constexpr int SMEM = (STAGE * STAGES > EPI ? STAGE * STAGES : EPI) + BN * 4 * 2;
};

template <int FMT, int MODE, int STAGES, int WARPS, int DBG, int NT>
__device__ __forceinline__ void body2(const Params& p, const Item& it, uint8_t* smem, const int* s_arow,
                                      const int* s_orow) {
  using C = Cfg2<FMT, MODE, STAGES, WARPS>;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int K = p.K, KS = K / BK;
  constexpr int XROWS_PER_PASS = C::THREADS / 8;
  constexpr int XJ = (NT * 8 + XROWS_PER_PASS - 1) / XROWS_PER_PASS;   // X chunks per thread per stage
  const int xch = tid & 7, xr0 = tid >> 3;
  const uint8_t* wbase = p.W + (size_t)(it.e * p.NWT + it.ft * C::WTS) * KS * FmtInfo<FMT>::WBYTES;
  const uint8_t* sbase = p.S + (size_t)(it.e * p.NWT + it.ft * C::WTS) * KS * FmtInfo<FMT>::SBYTES;
  const size_t wts_ = (size_t)KS * FmtInfo<FMT>::WBYTES, sts_ = (size_t)KS * FmtInfo<FMT>::SBYTES;

  auto load_stage = [&](int ks, int slot) {
    const uint32_t xs = smem_u32(smem + slot * C::STAGE);
#pragma unroll
    for (int j = 0; j < XJ; ++j) {
      int row = xr0 + XROWS_PER_PASS * j;
      if (NT * 8 % XROWS_PER_PASS != 0 && row >= NT * 8) break;
      cp_async16(xs + row * 128 + ((xch ^ (row & 7)) << 4), reinterpret_cast<const uint4*>(p.A) + (s_arow[row] + ks * (BK / 8) + xch),
                 row < it.cnt ? 16 : 0);
    }
    const uint32_t ws = xs + C::XBYTES;
    constexpr int WCH = FmtInfo<FMT>::WBYTES / 16;          // 128 (NVFP4) / 256 (MXFP8) chunks per warp tile
#pragma unroll
    for (int j = 0; j < C::WTS * WCH / C::THREADS; ++j) {
      int idx = tid + j * C::THREADS, wt = idx / WCH, c = idx % WCH;
      cp_async16(ws + idx * 16, wbase + wt * wts_ + (size_t)ks * FmtInfo<FMT>::WBYTES + c * 16, 16);
    }
    constexpr int SCH = FmtInfo<FMT>::SBYTES / 16;          // 16 / 8
    if (tid < C::WTS * SCH) {
      int wt = tid / SCH, c = tid % SCH;
      cp_async16(ws + C::WBYTES + tid * 16, sbase + wt * sts_ + (size_t)ks * FmtInfo<FMT>::SBYTES + c * 16, 16);
    }
  };

  float acc[2][NT][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][t][c] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KS) load_stage(s, s);
    cp_async_commit();
  }
  const int lm = lane >> 3, lr = lane & 7, g = lane >> 2;
  uint32_t ao[4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk)
    ao[kk] = ((lm >> 1) * 8 + lr) * 128 + ((((kk ^ (lr >> 1)) << 1) | ((lm & 1) ^ (lr & 1))) << 4);
  const int wofs = (w >> 1) * FmtInfo<FMT>::WBYTES, sofs = (w >> 1) * FmtInfo<FMT>::SBYTES, mt0 = 2 * (w & 1);

  for (int ks = 0; ks < KS; ++ks) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      int nk = ks + STAGES - 1;
      if (nk < KS) load_stage(nk, nk % STAGES);
      cp_async_commit();
    }
    const uint8_t* st = smem + (ks % STAGES) * C::STAGE;
    const uint32_t xs = smem_u32(st);
    const uint8_t* ws = st + C::XBYTES + wofs;
    const uint8_t* ss = st + C::XBYTES + C::WBYTES + sofs;
    uint32_t wq[2][FMT == MXFP8 ? 8 : 4], sw[2][FMT == NVFP4B ? 4 : FMT == MXFP8 ? 1 : 2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int mt = mt0 + i;
      if (FMT == NVFP4 || FMT == NVFP4_N1S1) {
        uint4 v = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 16);
        wq[i][0] = v.x; wq[i][1] = v.y; wq[i][2] = v.z; wq[i][3] = v.w;
        uint2 sv = *reinterpret_cast<const uint2*>(ss + (mt * 8 + g) * 8);
        sw[i][0] = sv.x; sw[i][1] = sv.y;
      } else if (FMT == NVFP4B) {
        uint4 v = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 16);
        wq[i][0] = v.x; wq[i][1] = v.y; wq[i][2] = v.z; wq[i][3] = v.w;
        uint4 sv = *reinterpret_cast<const uint4*>(ss + (mt * 8 + g) * 16);
        sw[i][0] = sv.x; sw[i][1] = sv.y; sw[i][2] = sv.z; sw[i][3] = sv.w;
      } else {
        uint4 v0 = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 32);
        uint4 v1 = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 32 + 16);
        wq[i][0] = v0.x; wq[i][1] = v0.y; wq[i][2] = v0.z; wq[i][3] = v0.w;
        wq[i][4] = v1.x; wq[i][5] = v1.y; wq[i][6] = v1.z; wq[i][7] = v1.w;
        sw[i][0] = *reinterpret_cast<const uint32_t*>(ss + (mt * 8 + g) * 4);
      }
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t a[2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint32_t s2;
        if (DBG == 1) {
          a[i][0] = wq[i][kk]; a[i][1] = wq[i][kk] ^ 0x1000u; a[i][2] = wq[i][kk] ^ 0x2000u; a[i][3] = wq[i][kk] ^ 0x3000u;
          continue;
        }
        if (FMT == NVFP4) {
          dec_nvfp4(wq[i][kk], a[i]);
          uint32_t s_g = prmt(sw[i][kk >> 1], 0u, (kk & 1) ? 0x4242u : 0x4040u) * 16u + 0x76007600u;
          uint32_t s_h = prmt(sw[i][kk >> 1], 0u, (kk & 1) ? 0x4343u : 0x4141u) * 16u + 0x76007600u;
          a[i][0] = bmul2(a[i][0], s_g); a[i][1] = bmul2(a[i][1], s_h);
          a[i][2] = bmul2(a[i][2], s_g); a[i][3] = bmul2(a[i][3], s_h);
          continue;
        } else if (FMT == NVFP4_N1S1) {
          dec_nvfp4b(wq[i][kk], a[i]);
          uint32_t s_g = prmt(sw[i][kk >> 1], 0u, (kk & 1) ? 0x4242u : 0x4040u) * 16u + 0x76007600u;
          uint32_t s_h = prmt(sw[i][kk >> 1], 0u, (kk & 1) ? 0x4343u : 0x4141u) * 16u + 0x76007600u;
          a[i][0] = bmul2(a[i][0], s_g); a[i][1] = bmul2(a[i][1], s_h);
          a[i][2] = bmul2(a[i][2], s_g); a[i][3] = bmul2(a[i][3], s_h);
          continue;
        } else if (FMT == NVFP4B) {
          dec_nvfp4b(wq[i][kk], a[i]);
          s2 = sw[i][kk];                            // (S_g, S_h) stored as bf16 bits
        } else {
          dec_mxfp8(wq[i][2 * kk], a[i][0], a[i][1]);
          dec_mxfp8(wq[i][2 * kk + 1], a[i][2], a[i][3]);
          s2 = prmt(sw[i][0], 0u, (kk >> 1) ? 0x4342u : 0x4140u) * 128u + 0x38003800u;
        }
        uint32_t s_g = prmt(s2, s2, 0x1010u), s_h = prmt(s2, s2, 0x3232u);
        a[i][0] = bmul2(a[i][0], s_g);
        a[i][1] = bmul2(a[i][1], s_h);
        a[i][2] = bmul2(a[i][2], s_g);
        a[i][3] = bmul2(a[i][3], s_h);
      }
#pragma unroll
      for (int pr = 0; pr < NT / 2; ++pr) {
        uint32_t b0, b1, b2, b3;
        if (DBG == 2) {
          if (pr == 0) ldmatrix_x4(xs + ao[kk], b0, b1, b2, b3);
          b0 += pr; b1 += pr; b2 += pr; b3 += pr;
        } else {
          ldmatrix_x4(xs + ao[kk] + pr * 2048, b0, b1, b2, b3);
        }
        mma_bf16(acc[0][2 * pr], a[0], b0, b1);
        mma_bf16(acc[1][2 * pr], a[1], b0, b1);
        mma_bf16(acc[0][2 * pr + 1], a[0], b2, b3);
        mma_bf16(acc[1][2 * pr + 1], a[1], b2, b3);
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // ---------------------------------------------------------------- epilogue (same tiles / swizzles as v1, WF = 2)
  const float fe = p.F[it.e];
  const int t4 = lane & 3;
  __nv_bfloat16* so = reinterpret_cast<__nv_bfloat16*>(smem);
  if (MODE == W13_SITU) {
    constexpr int FW = C::FWS;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int r = t * 8 + 2 * t4 + c;
        const float gm = r < it.cnt ? p.gamma[it.p0 + r] : 0.f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int f = 16 * w + 8 * i + g;
          float gv = __bfloat162float(__float2bfloat16_rn(acc[i][t][c] * fe));
          float uv = __bfloat162float(__float2bfloat16_rn(acc[i][t][c + 2] * fe));
          so[r * FW + (f ^ (((r >> 1) & 3) << 3))] = __float2bfloat16_rn(situ(gv, uv, gm));
        }
      }
    }
    __syncthreads();
    constexpr int CPR = FW * 2 / 16;
    for (int idx = tid; idx < it.cnt * CPR; idx += C::THREADS) {
      int r = idx / CPR, c = idx % CPR, swz = (r >> 1) & 3;
      uint4 v = *reinterpret_cast<const uint4*>(so + r * FW + ((c ^ swz) << 3));
      *reinterpret_cast<uint4*>(p.out + (size_t)(it.p0 + r) * I + it.ft * FW + c * 8) = v;
    }
  } else {
    constexpr int FW = C::FWP;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int f = MODE == W13_PLAIN ? 2 * (16 * w + 8 * i + g) + hh : 32 * w + 16 * i + g + 8 * hh;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const int r = t * 8 + 2 * t4 + c;
            so[r * FW + (f ^ (((r >> 1) & 7) << 3))] = __float2bfloat16_rn(acc[i][t][2 * hh + c] * fe);
          }
        }
      }
    }
    __syncthreads();
    constexpr int CPR = FW * 2 / 16;
    const int N = p.NWT * 64;
    for (int idx = tid; idx < it.cnt * CPR; idx += C::THREADS) {
      int r = idx / CPR, c = idx % CPR, swz = (r >> 1) & 7;
      uint4 v = *reinterpret_cast<const uint4*>(so + r * FW + ((c ^ swz) << 3));
      *reinterpret_cast<uint4*>(p.out + (size_t)s_orow[r] * N + it.ft * FW + c * 8) = v;
    }
  }
}

template <int FMT, int MODE, int STAGES, int WARPS = 4, int DBG = 0>
__global__ void __launch_bounds__(32 * WARPS, WARPS == 4 ? 2 : 1) moe_gemm2(Params p) {
  using C = Cfg2<FMT, MODE, STAGES, WARPS>;
  extern __shared__ __align__(128) uint8_t smem[];
  int* s_arow = reinterpret_cast<int*>(smem + C::SMEM - BN * 8);
  int* s_orow = s_arow + BN;
  if ((int)blockIdx.x >= (p.n_items_dev ? *p.n_items_dev : p.n_items)) return;
  const Item it = p.items[blockIdx.x];
  if (it.cnt > BN) __trap();   // item contract: at most BN rows (the shared row tables and epilogue tile)
  for (int i = threadIdx.x; i < BN; i += C::THREADS) {
    int pp = it.p0 + (i < it.cnt ? i : 0);
    s_arow[i] = p.a_row[pp] * (p.K / 8);   // 16-byte chunk offset of the A row (< 2^31: rows < 2^34 / K)
    if (MODE != W13_SITU) s_orow[i] = p.out_row[pp];
  }
  __syncthreads();
  const int nt = (((it.cnt + 7) >> 3) + 1) & ~1;
  switch (nt) {
    case 2: body2<FMT, MODE, STAGES, WARPS, DBG, 2>(p, it, smem, s_arow, s_orow); break;
    case 4: body2<FMT, MODE, STAGES, WARPS, DBG, 4>(p, it, smem, s_arow, s_orow); break;
    case 6: body2<FMT, MODE, STAGES, WARPS, DBG, 6>(p, it, smem, s_arow, s_orow); break;
    case 8: body2<FMT, MODE, STAGES, WARPS, DBG, 8>(p, it, smem, s_arow, s_orow); break;
    case 10: body2<FMT, MODE, STAGES, WARPS, DBG, 10>(p, it, smem, s_arow, s_orow); break;
    case 12: body2<FMT, MODE, STAGES, WARPS, DBG, 12>(p, it, smem, s_arow, s_orow); break;
    case 14: body2<FMT, MODE, STAGES, WARPS, DBG, 14>(p, it, smem, s_arow, s_orow); break;
    default: body2<FMT, MODE, STAGES, WARPS, DBG, 16>(p, it, smem, s_arow, s_orow); break;
  }
}

// ------------------------------------------------------------------------------------------------ decode GEMM
// Single-copy decode MoE kernel (BN_DEC = 16, WARPS = 4 -> 128 weight rows/CTA, FT_DEC = 18 CTAs/expert, 0 spills).
// 100% bit-identical to moe_gemm2 (same BK=64 k-order, decode + bmul2 + mma_bf16, epilogue / SiTU).
// Specialised on NT in {1, 2}: NT=1 (1..8 rows, >85% of decode experts) uses ldmatrix.x2 and halves MMAs, registers,
// X loads and epilogue work; 5-stage cp.async pipeline (~32.6 KB SMEM for NVFP4_N1S1 -> 4-5 CTAs/SM, ~52 KB MXFP8 -> 3).
constexpr int BN_DEC = 16;

template <int FMT, int MODE, int STAGES, int WARPS = 4>
struct CfgDec {
  static constexpr int THREADS = 32 * WARPS;
  static constexpr int WTS = WARPS / 2;
  static constexpr int XBYTES = BN_DEC * BK * 2;
  static constexpr int WBYTES = WTS * FmtInfo<FMT>::WBYTES;
  static constexpr int SBYTES = WTS * FmtInfo<FMT>::SBYTES;
  static constexpr int STAGE = XBYTES + WBYTES + SBYTES;
  static constexpr int FWS = WTS * 32, FWP = WTS * 64;
  static constexpr int EPI = MODE == W13_SITU ? BN_DEC * FWS * 2 : BN_DEC * FWP * 2;
  static constexpr int SMEM = (STAGE * STAGES > EPI ? STAGE * STAGES : EPI) + BN_DEC * 4 * 2;
};

template <int MODE, int WARPS, int NT>
__device__ __forceinline__ void epi_dec(const Params& p, const Item& it, uint8_t* smem, const int* s_orow,
                                        const float acc[2][NT][4]) {
  constexpr int THREADS = 32 * WARPS;
  constexpr int WTS = WARPS / 2;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const float fe = p.F[it.e];
  __nv_bfloat16* so = reinterpret_cast<__nv_bfloat16*>(smem);
  if (MODE == W13_SITU) {
    constexpr int FW = WTS * 32;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int r = t * 8 + 2 * t4 + c;
        if (r < it.cnt) {
          const float gm = p.gamma[it.p0 + r];
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int f = 16 * w + 8 * i + g;
            float gv = __bfloat162float(__float2bfloat16_rn(acc[i][t][c] * fe));
            float uv = __bfloat162float(__float2bfloat16_rn(acc[i][t][c + 2] * fe));
            so[r * FW + (f ^ (((r >> 1) & 3) << 3))] = __float2bfloat16_rn(situ(gv, uv, gm));
          }
        }
      }
    }
    __syncthreads();
    constexpr int CPR = FW * 2 / 16;
    for (int idx = tid; idx < it.cnt * CPR; idx += THREADS) {
      int r = idx / CPR, c = idx % CPR, swz = (r >> 1) & 3;
      uint4 v = *reinterpret_cast<const uint4*>(so + r * FW + ((c ^ swz) << 3));
      *reinterpret_cast<uint4*>(p.out + (size_t)(it.p0 + r) * I + it.ft * FW + c * 8) = v;
    }
  } else {
    constexpr int FW = WTS * 64;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int f = MODE == W13_PLAIN ? 2 * (16 * w + 8 * i + g) + hh : 32 * w + 16 * i + g + 8 * hh;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const int r = t * 8 + 2 * t4 + c;
            if (r < it.cnt) {
              so[r * FW + (f ^ (((r >> 1) & 7) << 3))] = __float2bfloat16_rn(acc[i][t][2 * hh + c] * fe);
            }
          }
        }
      }
    }
    __syncthreads();
    constexpr int CPR = FW * 2 / 16;
    const int N = p.NWT * 64;
    for (int idx = tid; idx < it.cnt * CPR; idx += THREADS) {
      int r = idx / CPR, c = idx % CPR, swz = (r >> 1) & 7;
      uint4 v = *reinterpret_cast<const uint4*>(so + r * FW + ((c ^ swz) << 3));
      *reinterpret_cast<uint4*>(p.out + (size_t)s_orow[r] * N + it.ft * FW + c * 8) = v;
    }
  }
}

template <int FMT, int MODE, int STAGES, int WARPS, int NT>
__device__ __forceinline__ void body_dec(const Params& p, const Item& it, uint8_t* smem, const int* s_arow,
                                         const int* s_orow) {
  using C = CfgDec<FMT, MODE, STAGES, WARPS>;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int K = p.K, KS = K / BK;
  const int xch = tid & 7, xr0 = tid >> 3;
  const uint8_t* wbase = p.W + (size_t)(it.e * p.NWT + it.ft * C::WTS) * KS * FmtInfo<FMT>::WBYTES;
  const uint8_t* sbase = p.S + (size_t)(it.e * p.NWT + it.ft * C::WTS) * KS * FmtInfo<FMT>::SBYTES;
  const size_t wts_ = (size_t)KS * FmtInfo<FMT>::WBYTES, sts_ = (size_t)KS * FmtInfo<FMT>::SBYTES;

  auto load_stage = [&](int ks, int slot) {
    const uint32_t xs = smem_u32(smem + slot * C::STAGE);
    if (tid < NT * 64) {
      int row = xr0;
      cp_async16(xs + row * 128 + ((xch ^ (row & 7)) << 4),
                 reinterpret_cast<const uint4*>(p.A) + (s_arow[row] + ks * (BK / 8) + xch),
                 row < it.cnt ? 16 : 0);
    }
    const uint32_t ws = xs + C::XBYTES;
    constexpr int WCH = FmtInfo<FMT>::WBYTES / 16;
#pragma unroll
    for (int j = 0; j < C::WTS * WCH / C::THREADS; ++j) {
      int idx = tid + j * C::THREADS, wt = idx / WCH, c = idx % WCH;
      cp_async16(ws + idx * 16, wbase + wt * wts_ + (size_t)ks * FmtInfo<FMT>::WBYTES + c * 16, 16);
    }
    constexpr int SCH = FmtInfo<FMT>::SBYTES / 16;
    if (tid < C::WTS * SCH) {
      int wt = tid / SCH, c = tid % SCH;
      cp_async16(ws + C::WBYTES + tid * 16, sbase + wt * sts_ + (size_t)ks * FmtInfo<FMT>::SBYTES + c * 16, 16);
    }
  };

  float acc[2][NT][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][t][c] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KS) load_stage(s, s);
    cp_async_commit();
  }
  const int lm = lane >> 3, lr = lane & 7, g = lane >> 2;
  uint32_t ao[4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    const int r_ld = (NT == 1) ? lr : ((lm >> 1) * 8 + lr);
    ao[kk] = r_ld * 128 + ((((kk ^ (lr >> 1)) << 1) | ((lm & 1) ^ (lr & 1))) << 4);
  }
  const int wofs = (w >> 1) * FmtInfo<FMT>::WBYTES, sofs = (w >> 1) * FmtInfo<FMT>::SBYTES, mt0 = 2 * (w & 1);

  for (int ks = 0; ks < KS; ++ks) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      int nk = ks + STAGES - 1;
      if (nk < KS) load_stage(nk, nk % STAGES);
      cp_async_commit();
    }
    const uint8_t* st = smem + (ks % STAGES) * C::STAGE;
    const uint32_t xs = smem_u32(st);
    const uint8_t* ws = st + C::XBYTES + wofs;
    const uint8_t* ss = st + C::XBYTES + C::WBYTES + sofs;
    uint32_t wq[2][FMT == MXFP8 ? 8 : 4], sw[2][FMT == NVFP4B ? 4 : FMT == MXFP8 ? 1 : 2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int mt = mt0 + i;
      if constexpr (FMT == NVFP4 || FMT == NVFP4_N1S1) {
        uint4 v = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 16);
        wq[i][0] = v.x; wq[i][1] = v.y; wq[i][2] = v.z; wq[i][3] = v.w;
        uint2 sv = *reinterpret_cast<const uint2*>(ss + (mt * 8 + g) * 8);
        sw[i][0] = sv.x; sw[i][1] = sv.y;
      } else if constexpr (FMT == NVFP4B) {
        uint4 v = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 16);
        wq[i][0] = v.x; wq[i][1] = v.y; wq[i][2] = v.z; wq[i][3] = v.w;
        uint4 sv = *reinterpret_cast<const uint4*>(ss + (mt * 8 + g) * 16);
        sw[i][0] = sv.x; sw[i][1] = sv.y; sw[i][2] = sv.z; sw[i][3] = sv.w;
      } else {
        uint4 v0 = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 32);
        uint4 v1 = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 32 + 16);
        wq[i][0] = v0.x; wq[i][1] = v0.y; wq[i][2] = v0.z; wq[i][3] = v0.w;
        wq[i][4] = v1.x; wq[i][5] = v1.y; wq[i][6] = v1.z; wq[i][7] = v1.w;
        sw[i][0] = *reinterpret_cast<const uint32_t*>(ss + (mt * 8 + g) * 4);
      }
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t b0, b1, b2 = 0, b3 = 0;
      if constexpr (NT == 1) ldmatrix_x2(xs + ao[kk], b0, b1);
      else ldmatrix_x4(xs + ao[kk], b0, b1, b2, b3);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint32_t a[4], s_g, s_h;
        if constexpr (FMT == NVFP4) {
          dec_nvfp4(wq[i][kk], a);
          s_g = prmt(sw[i][kk >> 1], 0u, (kk & 1) ? 0x4242u : 0x4040u) * 16u + 0x76007600u;
          s_h = prmt(sw[i][kk >> 1], 0u, (kk & 1) ? 0x4343u : 0x4141u) * 16u + 0x76007600u;
        } else if constexpr (FMT == NVFP4_N1S1) {
          dec_nvfp4b(wq[i][kk], a);
          s_g = prmt(sw[i][kk >> 1], 0u, (kk & 1) ? 0x4242u : 0x4040u) * 16u + 0x76007600u;
          s_h = prmt(sw[i][kk >> 1], 0u, (kk & 1) ? 0x4343u : 0x4141u) * 16u + 0x76007600u;
        } else if constexpr (FMT == NVFP4B) {
          dec_nvfp4b(wq[i][kk], a);
          uint32_t s2 = sw[i][kk];
          s_g = prmt(s2, s2, 0x1010u);
          s_h = prmt(s2, s2, 0x3232u);
        } else {
          dec_mxfp8(wq[i][2 * kk], a[0], a[1]);
          dec_mxfp8(wq[i][2 * kk + 1], a[2], a[3]);
          s_g = prmt(sw[i][0], 0u, (kk >> 1) ? 0x4242u : 0x4040u) * 128u + 0x38003800u;
          s_h = prmt(sw[i][0], 0u, (kk >> 1) ? 0x4343u : 0x4141u) * 128u + 0x38003800u;
        }
        a[0] = bmul2(a[0], s_g); a[1] = bmul2(a[1], s_h);
        a[2] = bmul2(a[2], s_g); a[3] = bmul2(a[3], s_h);
        mma_bf16(acc[i][0], a, b0, b1);
        if constexpr (NT == 2) mma_bf16(acc[i][1], a, b2, b3);
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();
  epi_dec<MODE, WARPS, NT>(p, it, smem, s_orow, acc);
}

template <int FMT, int MODE, int STAGES, int WARPS = 4>
__global__ void __launch_bounds__(32 * WARPS, FMT == MXFP8 ? 3 : 4) moe_gemm_dec(Params p) {
  using C = CfgDec<FMT, MODE, STAGES, WARPS>;
  extern __shared__ __align__(128) uint8_t smem[];
  int* s_arow = reinterpret_cast<int*>(smem + C::SMEM - BN_DEC * 8);
  int* s_orow = s_arow + BN_DEC;
  if ((int)blockIdx.x >= (p.n_items_dev ? *p.n_items_dev : p.n_items)) return;
  const Item it = p.items[blockIdx.x];
  if (it.cnt > BN_DEC) __trap();
  if (threadIdx.x < BN_DEC) {
    int pp = it.p0 + (threadIdx.x < it.cnt ? threadIdx.x : 0);
    s_arow[threadIdx.x] = p.a_row[pp] * (p.K / 8);
    if (MODE != W13_SITU) s_orow[threadIdx.x] = p.out_row[pp];
  }
  __syncthreads();
  if (it.cnt <= 8)
    body_dec<FMT, MODE, STAGES, WARPS, 1>(p, it, smem, s_arow, s_orow);
  else
    body_dec<FMT, MODE, STAGES, WARPS, 2>(p, it, smem, s_arow, s_orow);
}

// ------------------------------------------------------------------------------------------------ v3 GEMM
// CTA = 8 warps, 128 weight rows x up to 256 token rows (BN3). Warp w owns ONE m16 tile (warp tile w >> 2, m16 tile
// w & 3) for all the CTA's token rows: a decoded fragment feeds NT (up to 32) MMAs, twice v2's for experts with
// >= 256 rows. NT in {2, 4, ..., 16, 32} (template); items are 256-row tiles, then one 128-row tile, then the rest.
constexpr int BN3 = 256;
template <int FMT, int MODE, int STAGES>
struct Cfg3 {
  static constexpr int THREADS = 256;
  static constexpr int XBYTES = BN3 * BK * 2;            // 32 KB
  static constexpr int WBYTES = 2 * FmtInfo<FMT>::WBYTES;
  static constexpr int SBYTES = 2 * FmtInfo<FMT>::SBYTES;
  static constexpr int STAGE = XBYTES + WBYTES + SBYTES;
  static constexpr int EPI = MODE == W13_SITU ? BN3 * 64 * 2 : BN3 * 128 * 2;
  static constexpr int SMEM = (STAGE * STAGES > EPI ? STAGE * STAGES : EPI) + BN3 * 4 * 2;
};

template <int FMT, int MODE, int STAGES, int NT>
__device__ __forceinline__ void body3(const Params& p, const Item& it, uint8_t* smem, const int* s_arow,
                                      const int* s_orow) {
  using C = Cfg3<FMT, MODE, STAGES>;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int K = p.K, KS = K / BK;
  constexpr int XJ = (NT * 8 + 31) / 32;                 // 32 rows per pass (256 threads x 8 chunks)
  const int xch = tid & 7, xr0 = tid >> 3;
  const uint8_t* wbase = p.W + (size_t)(it.e * p.NWT + it.ft * 2) * KS * FmtInfo<FMT>::WBYTES;
  const uint8_t* sbase = p.S + (size_t)(it.e * p.NWT + it.ft * 2) * KS * FmtInfo<FMT>::SBYTES;
  const size_t wts_ = (size_t)KS * FmtInfo<FMT>::WBYTES, sts_ = (size_t)KS * FmtInfo<FMT>::SBYTES;

  auto load_stage = [&](int ks, int slot) {
    const uint32_t xs = smem_u32(smem + slot * C::STAGE);
#pragma unroll
    for (int j = 0; j < XJ; ++j) {
      int row = xr0 + 32 * j;
      if ((NT * 8) % 32 != 0 && row >= NT * 8) break;
      cp_async16(xs + row * 128 + ((xch ^ (row & 7)) << 4), reinterpret_cast<const uint4*>(p.A) + (s_arow[row] + ks * (BK / 8) + xch),
                 row < it.cnt ? 16 : 0);
    }
    const uint32_t ws = xs + C::XBYTES;
    constexpr int WCH = FmtInfo<FMT>::WBYTES / 16;
#pragma unroll
    for (int j = 0; j < (2 * WCH + 255) / 256; ++j) {
      int idx = tid + j * 256;
      if (idx < 2 * WCH) {
        int wt = idx / WCH, c = idx % WCH;
        cp_async16(ws + idx * 16, wbase + wt * wts_ + (size_t)ks * FmtInfo<FMT>::WBYTES + c * 16, 16);
      }
    }
    constexpr int SCH = FmtInfo<FMT>::SBYTES / 16;
    if (tid < 2 * SCH) {
      int wt = tid / SCH, c = tid % SCH;
      cp_async16(ws + C::WBYTES + tid * 16, sbase + wt * sts_ + (size_t)ks * FmtInfo<FMT>::SBYTES + c * 16, 16);
    }
  };

  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[t][c] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KS) load_stage(s, s);
    cp_async_commit();
  }
  const int lm = lane >> 3, lr = lane & 7, g = lane >> 2;
  uint32_t ao[4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk)
    ao[kk] = ((lm >> 1) * 8 + lr) * 128 + ((((kk ^ (lr >> 1)) << 1) | ((lm & 1) ^ (lr & 1))) << 4);
  const int wofs = (w >> 2) * FmtInfo<FMT>::WBYTES, sofs = (w >> 2) * FmtInfo<FMT>::SBYTES, mt = w & 3;

  for (int ks = 0; ks < KS; ++ks) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      int nk = ks + STAGES - 1;
      if (nk < KS) load_stage(nk, nk % STAGES);
      cp_async_commit();
    }
    const uint8_t* st = smem + (ks % STAGES) * C::STAGE;
    const uint32_t xs = smem_u32(st);
    const uint8_t* ws = st + C::XBYTES + wofs;
    const uint8_t* ss = st + C::XBYTES + C::WBYTES + sofs;
    uint32_t wq[FMT == NVFP4 ? 4 : 8], sw[FMT == NVFP4 ? 2 : 1];
    if (FMT == NVFP4) {
      uint4 v = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 16);
      wq[0] = v.x; wq[1] = v.y; wq[2] = v.z; wq[3] = v.w;
      uint2 sv = *reinterpret_cast<const uint2*>(ss + (mt * 8 + g) * 8);
      sw[0] = sv.x; sw[1] = sv.y;
    } else {
      uint4 v0 = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 32);
      uint4 v1 = *reinterpret_cast<const uint4*>(ws + (mt * 32 + lane) * 32 + 16);
      wq[0] = v0.x; wq[1] = v0.y; wq[2] = v0.z; wq[3] = v0.w;
      wq[4] = v1.x; wq[5] = v1.y; wq[6] = v1.z; wq[7] = v1.w;
      sw[0] = *reinterpret_cast<const uint32_t*>(ss + (mt * 8 + g) * 4);
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t a[4], s2;
      if (FMT == NVFP4) {
        dec_nvfp4(wq[kk], a);
        s2 = prmt(sw[kk >> 1], 0u, (kk & 1) ? 0x4342u : 0x4140u) * 16u + 0x76007600u;
      } else {
        dec_mxfp8(wq[2 * kk], a[0], a[1]);
        dec_mxfp8(wq[2 * kk + 1], a[2], a[3]);
        s2 = prmt(sw[0], 0u, (kk >> 1) ? 0x4342u : 0x4140u) * 128u + 0x38003800u;
      }
      uint32_t s_g = prmt(s2, s2, 0x1010u), s_h = prmt(s2, s2, 0x3232u);
      a[0] = bmul2(a[0], s_g);
      a[1] = bmul2(a[1], s_h);
      a[2] = bmul2(a[2], s_g);
      a[3] = bmul2(a[3], s_h);
#pragma unroll
      for (int pr = 0; pr < NT / 2; ++pr) {
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4(xs + ao[kk] + pr * 2048, b0, b1, b2, b3);
        mma_bf16(acc[2 * pr], a, b0, b1);
        mma_bf16(acc[2 * pr + 1], a, b2, b3);
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  const float fe = p.F[it.e];
  const int t4 = lane & 3;
  __nv_bfloat16* so = reinterpret_cast<__nv_bfloat16*>(smem);
  if (MODE == W13_SITU) {
    constexpr int FW = 64;
    const int f = 8 * w + g;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int r = t * 8 + 2 * t4 + c;
        const float gm = r < it.cnt ? p.gamma[it.p0 + r] : 0.f;
        float gv = __bfloat162float(__float2bfloat16_rn(acc[t][c] * fe));
        float uv = __bfloat162float(__float2bfloat16_rn(acc[t][c + 2] * fe));
        so[r * FW + (f ^ (((r >> 1) & 3) << 3))] = __float2bfloat16_rn(situ(gv, uv, gm));
      }
    }
    __syncthreads();
    constexpr int CPR = FW * 2 / 16;
    for (int idx = tid; idx < it.cnt * CPR; idx += 256) {
      int r = idx / CPR, c = idx % CPR, swz = (r >> 1) & 3;
      uint4 v = *reinterpret_cast<const uint4*>(so + r * FW + ((c ^ swz) << 3));
      *reinterpret_cast<uint4*>(p.out + (size_t)(it.p0 + r) * I + it.ft * FW + c * 8) = v;
    }
  } else {
    constexpr int FW = 128;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int f = MODE == W13_PLAIN ? 2 * (8 * w + g) + hh : 16 * w + g + 8 * hh;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int r = t * 8 + 2 * t4 + c;
          so[r * FW + (f ^ (((r >> 1) & 7) << 3))] = __float2bfloat16_rn(acc[t][2 * hh + c] * fe);
        }
      }
    }
    __syncthreads();
    constexpr int CPR = FW * 2 / 16;
    const int N = p.NWT * 64;
    for (int idx = tid; idx < it.cnt * CPR; idx += 256) {
      int r = idx / CPR, c = idx % CPR, swz = (r >> 1) & 7;
      uint4 v = *reinterpret_cast<const uint4*>(so + r * FW + ((c ^ swz) << 3));
      *reinterpret_cast<uint4*>(p.out + (size_t)s_orow[r] * N + it.ft * FW + c * 8) = v;
    }
  }
}

template <int FMT, int MODE, int STAGES>
__global__ void __launch_bounds__(256, 1) moe_gemm3(Params p) {
  using C = Cfg3<FMT, MODE, STAGES>;
  extern __shared__ __align__(128) uint8_t smem[];
  int* s_arow = reinterpret_cast<int*>(smem + C::SMEM - BN3 * 8);
  int* s_orow = s_arow + BN3;
  if ((int)blockIdx.x >= (p.n_items_dev ? *p.n_items_dev : p.n_items)) return;
  const Item it = p.items[blockIdx.x];
  if (it.cnt > BN3) __trap();   // item contract: at most BN3 rows (the shared row tables and epilogue tile)
  for (int i = threadIdx.x; i < BN3; i += 256) {
    int pp = it.p0 + (i < it.cnt ? i : 0);
    s_arow[i] = p.a_row[pp] * (p.K / 8);   // 16-byte chunk offset of the A row (< 2^31: rows < 2^34 / K)
    if (MODE != W13_SITU) s_orow[i] = p.out_row[pp];
  }
  __syncthreads();
  const int nt = (((it.cnt + 7) >> 3) + 1) & ~1;
  switch (nt) {
    case 2: body3<FMT, MODE, STAGES, 2>(p, it, smem, s_arow, s_orow); break;
    case 4: body3<FMT, MODE, STAGES, 4>(p, it, smem, s_arow, s_orow); break;
    case 6: body3<FMT, MODE, STAGES, 6>(p, it, smem, s_arow, s_orow); break;
    case 8: body3<FMT, MODE, STAGES, 8>(p, it, smem, s_arow, s_orow); break;
    case 10: body3<FMT, MODE, STAGES, 10>(p, it, smem, s_arow, s_orow); break;
    case 12: body3<FMT, MODE, STAGES, 12>(p, it, smem, s_arow, s_orow); break;
    case 14: body3<FMT, MODE, STAGES, 14>(p, it, smem, s_arow, s_orow); break;
    case 16: body3<FMT, MODE, STAGES, 16>(p, it, smem, s_arow, s_orow); break;
    default: body3<FMT, MODE, STAGES, 32>(p, it, smem, s_arow, s_orow); break;
  }
}

// ------------------------------------------------------------------------------------------------ combine
// out[t, :] = bf16(0 + y[8t] + ... + y[8t + 7]) in slot order (config B _combine_kernel).
__global__ void moe_combine(const __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ out, int tokens) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;       // one thread per 8 columns of one token
  int per = H / 8;
  if (idx >= tokens * per) return;
  int t = idx / per, c = idx % per;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
#pragma unroll
  for (int s = 0; s < TOPK; ++s) {
    uint4 v = *reinterpret_cast<const uint4*>(y + ((size_t)t * TOPK + s) * H + c * 8);
    const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = __fadd_rn(acc[j], __bfloat162float(hv[j]));
  }
  uint4 o;
  __nv_bfloat16* ov = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
  for (int j = 0; j < 8; ++j) ov[j] = __float2bfloat16_rn(acc[j]);
  *reinterpret_cast<uint4*>(out + (size_t)t * H + c * 8) = o;
}

// SiTU of interleaved gate/up rows (flat order) times the routing weight: the Humming arm's swiglu (config B
// _swiglu_kernel numerics).
__global__ void moe_swiglu(const __nv_bfloat16* __restrict__ gu, const float* __restrict__ wts,
                           __nv_bfloat16* __restrict__ h, int rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;      // one thread per 8 features of one row
  int per = I / 8;
  if (idx >= rows * per) return;
  int r = idx / per, c = idx % per;
  const uint4* src = reinterpret_cast<const uint4*>(gu + (size_t)r * 2 * I + c * 16);
  uint4 v0 = src[0], v1 = src[1];
  const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(&v0);
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v1);
  float gamma = wts[r];
  uint4 o;
  __nv_bfloat16* ov = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
  for (int j = 0; j < 4; ++j) ov[j] = __float2bfloat16_rn(situ(__bfloat162float(a[2 * j]), __bfloat162float(a[2 * j + 1]), gamma));
#pragma unroll
  for (int j = 0; j < 4; ++j) ov[4 + j] = __float2bfloat16_rn(situ(__bfloat162float(b[2 * j]), __bfloat162float(b[2 * j + 1]), gamma));
  *reinterpret_cast<uint4*>(h + (size_t)r * I + c * 8) = o;
}

}  // namespace pmoe
