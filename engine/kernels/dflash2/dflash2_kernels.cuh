// Native sm_80 CUDA kernels for Inkling-2-Mini-DFlash2 (block-diffusion speculative drafter).
// Own code of this project; zero dependencies on engine/native/.
//
// Model constants (incoai/Inkling-2-Mini-DFlash2):
//   H = 2304, NH = 32, KVH = 8, HD = 128, INTER = 6912, NLAYERS = 5,
//   BLK = 8 (1 anchor + 7 masks), DRAFTS = 7, SWA = 2048, RING = 4096 (mask 4095),
//   CONV_GROUPS = 144, CONV_GS = 16, CONV_TAPS = 2,
//   VALID_VOCAB = 200058, TOP_K = 16, SEL_RANK = 256, SOFTCAP = 30.0f, EPS = 1e-6f.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>

namespace df2 {

constexpr int H = 2304;
constexpr int NH = 32;
constexpr int KVH = 8;
constexpr int HD = 128;
constexpr int GQA = NH / KVH;  // 4
constexpr int INTER = 6912;
constexpr int BLK = 8;
constexpr int DRAFTS = 7;
constexpr int SWA = 2048;
constexpr int RING = 4096;
constexpr int RING_MASK = RING - 1;
constexpr int CONV_GROUPS = 144;
constexpr int CONV_GS = 16;
constexpr int VALID_VOCAB = 200058;
constexpr int TOP_K = 16;
constexpr int SEL_RANK = 256;
constexpr int TOPK_CHUNKS = 64;
constexpr int TOPK_CHUNK_LEN = 3136;  // multiple of 16; 64 * 3136 = 200704 >= VALID_VOCAB
constexpr int AT_TILE = 64;
constexpr int AT_KS = HD + 8;  // 136 bf16 -> conflict-free ldmatrix
constexpr int GLOBAL_ATTN_SMEM = 0;  // Design A uses 8 KB static smem (zero dynamic smem)
constexpr float RMS_EPS = 1e-6f;
constexpr float SOFTCAP = 30.0f;
constexpr float ATTN_SCALE = 0.08838834764831845f;           // 1 / sqrt(128)
constexpr float ATTN_SCALE_LOG2 = 0.12751743515238057f;      // (1 / sqrt(128)) * log2(e)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

__device__ __forceinline__ float block_sum_256(float v, float* smem) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  float s = (threadIdx.x < 8) ? smem[threadIdx.x] : 0.0f;
  if (wid == 0) {
    s = warp_sum(s);
    if (lane == 0) smem[0] = s;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float block_max_256(float v, float* smem) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  v = warp_max(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  float m = (threadIdx.x < 8) ? smem[threadIdx.x] : -1e38f;
  if (wid == 0) {
    m = warp_max(m);
    if (lane == 0) smem[0] = m;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ void block_argmax_256(float val, int idx, float* sv, int* si, float& out_v, int& out_i) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, val, off);
    int oi = __shfl_xor_sync(0xffffffffu, idx, off);
    if (ov > val || (ov == val && oi < idx)) {
      val = ov;
      idx = oi;
    }
  }
  if (lane == 0) {
    sv[wid] = val;
    si[wid] = idx;
  }
  __syncthreads();
  if (wid == 0) {
    float bv = (lane < 8) ? sv[lane] : -1e38f;
    int bi = (lane < 8) ? si[lane] : 0x7fffffff;
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, bv, off);
      int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      sv[0] = bv;
      si[0] = bi;
    }
  }
  __syncthreads();
  out_v = sv[0];
  out_i = si[0];
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16_r(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg16_stream(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t pack2_f16(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t w) {
  const float a = __uint_as_float(w << 16);
  const float b = __uint_as_float(w & 0xffff0000u);
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// E4M3 -> f16x2 register decode (exact for all 254 finite E4M3 codes incl. subnormals): value = e4m3(b) * 2^-8.
__device__ __forceinline__ uint32_t place2(uint32_t x, uint32_t y, uint32_t sel) {
  const uint32_t h = __byte_perm(x, y, sel);
  return ((h >> 1) & 0x3F803F80u) | (h & 0x80008000u);
}

__host__ __device__ constexpr uint32_t psel(int i, int j) {
  return (uint32_t)i | ((uint32_t)i << 4) | ((uint32_t)i << 8) | ((uint32_t)j << 12);
}

__device__ __forceinline__ float ex2_fast(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a));
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void cp_async16(void* s, const void* g, bool pred) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
  int n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(g), "r"(n));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// 1. Precompute NeoX RoPE cos/sin table [max_pos, 128] bf16 (cos[0..64], sin[64..128]).
// Grid: ((max_pos * 64 + 255) / 256), Block: 256.
__global__ void rope_table_kernel(__nv_bfloat16* __restrict__ cos_sin, int max_pos, float theta) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = max_pos * 64;
  if (tid >= total) return;
  int pos = tid >> 6;
  int i = tid & 63;
  double inv_freq = 1.0 / pow((double)theta, (double)(2 * i) / (double)HD);
  double ang = (double)pos * inv_freq;
  cos_sin[(size_t)pos * HD + i] = __float2bfloat16((float)cos(ang));
  cos_sin[(size_t)pos * HD + 64 + i] = __float2bfloat16((float)sin(ang));
}

// 2. Build draft block embeddings [k, 8, H]: row j==0 is embed_norm(embed[anchor[sel[bi]]]), rows j>0 are mask_emb.
// Grid: (k * 8), Block: 256.
__global__ void embed_block_kernel(const __nv_bfloat16* __restrict__ embed,
                                   const __nv_bfloat16* __restrict__ embed_norm,
                                   const __nv_bfloat16* __restrict__ mask_emb,
                                   const int* __restrict__ anchor,
                                   const int* __restrict__ sel,
                                   __nv_bfloat16* __restrict__ out,
                                   int k) {
  int row = blockIdx.x;
  if (row >= k * BLK) return;
  int bi = row >> 3;
  int j = row & 7;
  __nv_bfloat16* dst = out + (size_t)row * H;
  if (j > 0) {
    for (int c = threadIdx.x; c < H; c += 256) dst[c] = mask_emb[c];
    return;
  }
  int tok = anchor[sel[bi]];
  const __nv_bfloat16* src = embed + (size_t)tok * H;
  __shared__ float smem[8];
  float ss = 0.0f;
  for (int c = threadIdx.x; c < H; c += 256) {
    float v = __bfloat162float(src[c]);
    ss += v * v;
  }
  float inv = rsqrtf(block_sum_256(ss, smem) / (float)H + RMS_EPS);
  for (int c = threadIdx.x; c < H; c += 256) {
    float v = __bfloat162float(src[c]);
    float nv = __bfloat162float(__float2bfloat16(v * inv));
    dst[c] = __float2bfloat16(nv * __bfloat162float(embed_norm[c]));
  }
}

// 3. RMSNorm [rows, H] -> [rows, H]. When skip_anchor != 0, reads row (bi*8 + slot + 1) for output row (bi*7 + slot).
// Grid: (rows), Block: 256.
__global__ void rms_norm_kernel(const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ w,
                                __nv_bfloat16* __restrict__ out,
                                int rows,
                                int skip_anchor) {
  int r = blockIdx.x;
  if (r >= rows) return;
  int src_r = skip_anchor ? ((r / DRAFTS) * BLK + (r % DRAFTS) + 1) : r;
  const __nv_bfloat16* src = x + (size_t)src_r * H;
  __nv_bfloat16* dst = out + (size_t)r * H;
  __shared__ float smem[8];
  float ss = 0.0f;
  for (int c = threadIdx.x; c < H; c += 256) {
    float v = __bfloat162float(src[c]);
    ss += v * v;
  }
  float inv = rsqrtf(block_sum_256(ss, smem) / (float)H + RMS_EPS);
  for (int c = threadIdx.x; c < H; c += 256) {
    float v = __bfloat162float(src[c]);
    float nv = __bfloat162float(__float2bfloat16(v * inv));
    dst[c] = __float2bfloat16(nv * __bfloat162float(w[c]));
  }
}

// 4. 2-tap grouped dynamic causal conv (input side = 0) across each 8-token draft block.
// dyn: [k*8, 576] = [k*8, 2(side), 2(tap), 144(group)]. base: [2, 2, H].
// Grid: (k * 8), Block: 256.
__global__ void grouped_conv_in_kernel(const __nv_bfloat16* __restrict__ h,
                                       const __nv_bfloat16* __restrict__ dyn,
                                       const __nv_bfloat16* __restrict__ base,
                                       __nv_bfloat16* __restrict__ out,
                                       int k) {
  int row = blockIdx.x;
  if (row >= k * BLK) return;
  int j = row & 7;
  const __nv_bfloat16* h0 = h + (size_t)row * H;
  const __nv_bfloat16* h1 = (j > 0) ? (h + (size_t)(row - 1) * H) : nullptr;
  const __nv_bfloat16* d0 = dyn + (size_t)row * 576;
  const __nv_bfloat16* d1 = d0 + CONV_GROUPS;
  const __nv_bfloat16* b0 = base;
  const __nv_bfloat16* b1 = base + H;
  __nv_bfloat16* dst = out + (size_t)row * H;
  for (int c = threadIdx.x; c < H; c += 256) {
    int g = c >> 4;
    float c0 = __bfloat162float(__float2bfloat16(__bfloat162float(b0[c]) + __bfloat162float(d0[g])));
    float acc = __bfloat162float(__float2bfloat16(c0 * __bfloat162float(h0[c])));
    if (j > 0) {
      float c1 = __bfloat162float(__float2bfloat16(__bfloat162float(b1[c]) + __bfloat162float(d1[g])));
      acc += __bfloat162float(__float2bfloat16(c1 * __bfloat162float(h1[c])));
    }
    dst[c] = __float2bfloat16(acc);
  }
}

// 5. Fused 2-tap grouped dynamic causal conv (output side = 1) + residual add (in-place on residual) + next RMSNorm.
// When skip_anchor_norm != 0, only rows j > 0 write to norm_out[bi*7 + (j-1)] (and optional f16_out for W8A16 head).
// Grid: (k * 8), Block: 256.
__global__ void grouped_conv_out_res_norm_kernel(const __nv_bfloat16* __restrict__ sub,
                                                 const __nv_bfloat16* __restrict__ dyn,
                                                 const __nv_bfloat16* __restrict__ base,
                                                 __nv_bfloat16* __restrict__ residual,
                                                 const __nv_bfloat16* __restrict__ norm_w,
                                                 __nv_bfloat16* __restrict__ norm_out,
                                                 __half* __restrict__ f16_out,
                                                 int k,
                                                 int skip_anchor_norm) {
  int row = blockIdx.x;
  if (row >= k * BLK) return;
  int bi = row >> 3;
  int j = row & 7;
  const __nv_bfloat16* s0 = sub + (size_t)row * H;
  const __nv_bfloat16* s1 = (j > 0) ? (sub + (size_t)(row - 1) * H) : nullptr;
  const __nv_bfloat16* d0 = dyn + (size_t)row * 576 + 2 * CONV_GROUPS;
  const __nv_bfloat16* d1 = d0 + CONV_GROUPS;
  const __nv_bfloat16* b0 = base + 2 * H;
  const __nv_bfloat16* b1 = base + 3 * H;
  __nv_bfloat16* res = residual + (size_t)row * H;

  __shared__ float smem[8];
  float vals[9];  // H / 256 == 9 exact
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    int c = threadIdx.x + i * 256;
    int g = c >> 4;
    float c0 = __bfloat162float(__float2bfloat16(__bfloat162float(b0[c]) + __bfloat162float(d0[g])));
    float acc = __bfloat162float(__float2bfloat16(c0 * __bfloat162float(s0[c])));
    if (j > 0) {
      float c1 = __bfloat162float(__float2bfloat16(__bfloat162float(b1[c]) + __bfloat162float(d1[g])));
      acc += __bfloat162float(__float2bfloat16(c1 * __bfloat162float(s1[c])));
    }
    __nv_bfloat16 conv_bf = __float2bfloat16(acc);
    __nv_bfloat16 xnew = __float2bfloat16(__bfloat162float(res[c]) + __bfloat162float(conv_bf));
    res[c] = xnew;
    float xf = __bfloat162float(xnew);
    vals[i] = xf;
    ss += xf * xf;
  }
  if (!norm_w || (skip_anchor_norm && j == 0)) return;
  float inv = rsqrtf(block_sum_256(ss, smem) / (float)H + RMS_EPS);
  int out_row = skip_anchor_norm ? (bi * DRAFTS + (j - 1)) : row;
  __nv_bfloat16* dst = norm_out + (size_t)out_row * H;
  __half* hdst = f16_out ? (f16_out + (size_t)out_row * H) : nullptr;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    int c = threadIdx.x + i * 256;
    float nv = __bfloat162float(__float2bfloat16(vals[i] * inv));
    float outv = nv * __bfloat162float(norm_w[c]);
    dst[c] = __float2bfloat16(outv);
    if (hdst) hdst[c] = __float2half_rn(outv);
  }
}

// 6. Commit context K/V: k_norm + NeoX RoPE on K, store K & V into head-major [sessions, KVH=8, RING=4096, HD=128].
// kv: [n, 2048] (k: [0..1024), v: [1024..2048)).
// Grid: (n, KVH=8), Block: 128.
__global__ void kv_norm_rope_store_kernel(const __nv_bfloat16* __restrict__ kv,
                                          const __nv_bfloat16* __restrict__ k_norm,
                                          const __nv_bfloat16* __restrict__ cos_sin,
                                          const int* __restrict__ pos,
                                          const int* __restrict__ sess,
                                          __nv_bfloat16* __restrict__ k_ring,
                                          __nv_bfloat16* __restrict__ v_ring,
                                          int n) {
  int r = blockIdx.x;
  int kvh = blockIdx.y;
  int d = threadIdx.x;
  if (r >= n) return;
  int p = pos[r];
  int s = sess[r];

  const __nv_bfloat16* ksrc = kv + (size_t)r * 2048 + kvh * HD;
  const __nv_bfloat16* vsrc = kv + (size_t)r * 2048 + 1024 + kvh * HD;

  __shared__ float sk[HD];
  __shared__ float sw[4];
  float kf = __bfloat162float(ksrc[d]);
  float ws = warp_sum(kf * kf);
  if ((d & 31) == 0) sw[d >> 5] = ws;
  __syncthreads();
  float tot = sw[0] + sw[1] + sw[2] + sw[3];
  float inv = rsqrtf(tot / (float)HD + RMS_EPS);
  float kn = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(kf * inv)) * __bfloat162float(k_norm[d])));
  sk[d] = kn;
  __syncthreads();

  int dh = d & 63;
  float cv = __bfloat162float(cos_sin[(size_t)p * HD + dh]);
  float sv = __bfloat162float(cos_sin[(size_t)p * HD + 64 + dh]);
  float rot = (d < 64) ? (sk[d] * cv - sk[d + 64] * sv) : (sk[d] * cv + sk[d - 64] * sv);

  size_t dst_off = (((size_t)s * KVH + (size_t)kvh) * RING + (size_t)(p & RING_MASK)) * HD + (size_t)d;
  k_ring[dst_off] = __float2bfloat16(rot);
  v_ring[dst_off] = vsrc[d];
}

// 6b. Commit context K/V for global ("full_attention") layers:
// k_norm + NeoX RoPE (cos_sin_global) on K, per-head-token max-abs E4M3 quantization on K & V,
// stored linearly in head-major [sessions, KVH=8, kv_stride, HD=128] uint8_t with fp32 scales
// [sessions, KVH=8, kv_stride].
// Grid: (n, KVH=8), Block: 128.
__global__ void kv_norm_rope_store_global_e4m3_kernel(const __nv_bfloat16* __restrict__ kv,
                                                      const __nv_bfloat16* __restrict__ k_norm,
                                                      const __nv_bfloat16* __restrict__ cos_sin,
                                                      const int* __restrict__ pos,
                                                      const int* __restrict__ sess,
                                                      uint8_t* __restrict__ k_glob,
                                                      uint8_t* __restrict__ v_glob,
                                                      float* __restrict__ ks_glob,
                                                      float* __restrict__ vs_glob,
                                                      int n,
                                                      int max_ctx,
                                                      int kv_stride) {
  int r = blockIdx.x;
  int kvh = blockIdx.y;
  int d = threadIdx.x;
  if (r >= n) return;
  int p = pos[r];
  if (p < 0 || p >= max_ctx) return;
  int s = sess[r];

  const __nv_bfloat16* ksrc = kv + (size_t)r * 2048 + kvh * HD;
  const __nv_bfloat16* vsrc = kv + (size_t)r * 2048 + 1024 + kvh * HD;

  __shared__ float sk[HD];
  __shared__ float sw[4];
  __shared__ float swk[4];
  __shared__ float swv[4];
  float kf = __bfloat162float(ksrc[d]);
  float ws = warp_sum(kf * kf);
  if ((d & 31) == 0) sw[d >> 5] = ws;
  __syncthreads();
  float tot = sw[0] + sw[1] + sw[2] + sw[3];
  float inv = rsqrtf(tot / (float)HD + RMS_EPS);
  float kn = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(kf * inv)) * __bfloat162float(k_norm[d])));
  sk[d] = kn;
  __syncthreads();

  int dh = d & 63;
  float cv = __bfloat162float(cos_sin[(size_t)p * HD + dh]);
  float sv = __bfloat162float(cos_sin[(size_t)p * HD + 64 + dh]);
  float rot = (d < 64) ? (sk[d] * cv - sk[d + 64] * sv) : (sk[d] * cv + sk[d - 64] * sv);
  float rot_bf = __bfloat162float(__float2bfloat16(rot));
  float vf = __bfloat162float(vsrc[d]);

  float wk = warp_max(fabsf(rot_bf));
  float wv = warp_max(fabsf(vf));
  if ((d & 31) == 0) {
    swk[d >> 5] = wk;
    swv[d >> 5] = wv;
  }
  __syncthreads();
  float amax_k = fmaxf(fmaxf(swk[0], swk[1]), fmaxf(swk[2], swk[3]));
  float amax_v = fmaxf(fmaxf(swv[0], swv[1]), fmaxf(swv[2], swv[3]));
  float scale_k = (amax_k > 0.0f) ? (amax_k * (1.0f / 448.0f)) : 1.0f;
  float inv_sk = (amax_k > 0.0f) ? (448.0f / amax_k) : 0.0f;
  float scale_v = (amax_v > 0.0f) ? (amax_v * (1.0f / 448.0f)) : 1.0f;
  float inv_sv = (amax_v > 0.0f) ? (448.0f / amax_v) : 0.0f;

  size_t tok_off = ((size_t)s * KVH + (size_t)kvh) * (size_t)kv_stride + (size_t)p;
  size_t dst_off = tok_off * HD + (size_t)d;
  k_glob[dst_off] = f2e4m3(rot_bf * inv_sk);
  v_glob[dst_off] = f2e4m3(vf * inv_sv);
  if (d == 0) {
    ks_glob[tok_off] = scale_k;
    vs_glob[tok_off] = scale_v;
  }
}

// 7. Draft block QKV post: q_norm + NeoX RoPE on Q (32 heads), k_norm + NeoX RoPE on K (8 heads), copy V (8 heads).
// qkv: [k*8, 6144] (Q: 0..4096, K: 4096..5120, V: 5120..6144).
// Grid: (k * 8, 40), Block: 128.
__global__ void qkv_norm_rope_blk_kernel(const __nv_bfloat16* __restrict__ qkv,
                                         const __nv_bfloat16* __restrict__ q_norm,
                                         const __nv_bfloat16* __restrict__ k_norm,
                                         const __nv_bfloat16* __restrict__ cos_sin,
                                         const int* __restrict__ pos,
                                         const int* __restrict__ sel,
                                         __nv_bfloat16* __restrict__ q_blk,
                                         __nv_bfloat16* __restrict__ k_blk,
                                         __nv_bfloat16* __restrict__ v_blk,
                                         int k) {
  int row = blockIdx.x;
  int hidx = blockIdx.y;
  int d = threadIdx.x;
  if (row >= k * BLK) return;
  int bi = row >> 3;
  int j = row & 7;
  int p = pos[sel[bi]] + j;

  bool is_q = (hidx < NH);
  const __nv_bfloat16* src = is_q ? (qkv + (size_t)row * 6144 + hidx * HD)
                                  : (qkv + (size_t)row * 6144 + 4096 + (hidx - NH) * HD);
  const __nv_bfloat16* nw = is_q ? q_norm : k_norm;

  __shared__ float sx[HD];
  __shared__ float sw[4];
  float xf = __bfloat162float(src[d]);
  float ws = warp_sum(xf * xf);
  if ((d & 31) == 0) sw[d >> 5] = ws;
  __syncthreads();
  float inv = rsqrtf((sw[0] + sw[1] + sw[2] + sw[3]) / (float)HD + RMS_EPS);
  float xn = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(xf * inv)) * __bfloat162float(nw[d])));
  sx[d] = xn;
  __syncthreads();

  int dh = d & 63;
  float cv = __bfloat162float(cos_sin[(size_t)p * HD + dh]);
  float sv = __bfloat162float(cos_sin[(size_t)p * HD + 64 + dh]);
  float rot = (d < 64) ? (sx[d] * cv - sx[d + 64] * sv) : (sx[d] * cv + sx[d - 64] * sv);

  if (is_q) {
    q_blk[((size_t)row * NH + (size_t)hidx) * HD + d] = __float2bfloat16(rot);
  } else {
    int kvh = hidx - NH;
    size_t off = ((size_t)row * KVH + (size_t)kvh) * HD + d;
    k_blk[off] = __float2bfloat16(rot);
    v_blk[off] = qkv[(size_t)row * 6144 + 5120 + kvh * HD + d];
  }
}

// 8a. Tensor Core MMA split-KV SWA-2048 + bidirectional 8-token block attention.
// All 32 query rows per KV head (8 block queries x 4 GQA heads) are active with zero wasted MMA rows.
// Dynamic shared memory: (4 * AT_TILE + 32) * AT_KS * 2 = 78,336 bytes.
// Grid: (splits, KVH=8, k), Block: 128 (4 warps: mt = warp&1 selects 16 Q rows, kh = warp>>1 selects 32 keys).
// Outputs: part_ml [k, KVH, splits, 32, 2] f32 (m_log2, l), part_o [k, KVH, splits, 32, HD] f32.
__global__ void __launch_bounds__(128) swa_attn_split_kernel(const __nv_bfloat16* __restrict__ q_blk,
                                                             const __nv_bfloat16* __restrict__ k_blk,
                                                             const __nv_bfloat16* __restrict__ v_blk,
                                                             const __nv_bfloat16* __restrict__ k_ring,
                                                             const __nv_bfloat16* __restrict__ v_ring,
                                                             const int* __restrict__ pos,
                                                             const int* __restrict__ sel,
                                                             float* __restrict__ part_ml,
                                                             float* __restrict__ part_o,
                                                             int k,
                                                             int splits) {
  extern __shared__ __align__(16) unsigned char at_smem[];
  auto ks = (__nv_bfloat16(*)[AT_TILE][AT_KS])at_smem;
  auto vs = (__nv_bfloat16(*)[AT_TILE][AT_KS])(at_smem + 2 * AT_TILE * AT_KS * 2);
  auto qsm = (__nv_bfloat16(*)[AT_KS])(at_smem + 4 * AT_TILE * AT_KS * 2);
  __shared__ float comb[2][32][2];

  const int sp = blockIdx.x;
  const int kvh = blockIdx.y;
  const int bi = blockIdx.z;
  if (bi >= k) return;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int mt = warp & 1;
  const int kh = warp >> 1;
  const int s = sel[bi];
  const int p0 = pos[s];

  const int ctx_start = (p0 > SWA - 1) ? (p0 - (SWA - 1)) : 0;
  const int ctx_len = p0 - ctx_start;
  const int total_keys = ctx_len + BLK;
  const int ntile_total = (total_keys + AT_TILE - 1) / AT_TILE;
  const int per = (ntile_total + splits - 1) / splits;
  const int t0 = sp * per;
  const int t1 = min(ntile_total, t0 + per);

  // Load 32 Q rows: row = j * 4 + g4 (j in 0..7, g4 in 0..3).
  for (int i = threadIdx.x; i < 32 * (HD / 8); i += 128) {
    int row = i >> 4;
    int c = (i & 15) * 8;
    int j = row >> 2;
    int g4 = row & 3;
    *(uint4*)&qsm[row][c] = *(const uint4*)(q_blk + ((size_t)(bi * BLK + j) * NH + kvh * GQA + g4) * HD + c);
  }
  __syncthreads();

  const size_t ring_base = ((size_t)s * KVH + (size_t)kvh) * RING;
  auto load_tile = [&](int tt, int buf) {
    const int kb = tt * AT_TILE;
#pragma unroll
    for (int it = 0; it < (AT_TILE * HD / 8) / 128; ++it) {
      int ii = threadIdx.x + it * 128;
      int kk = ii >> 4;
      int c = (ii & 15) * 8;
      int ki = kb + kk;
      if (ki < ctx_len) {
        size_t off = (ring_base + (size_t)((ctx_start + ki) & RING_MASK)) * HD + c;
        cp_async16(&ks[buf][kk][c], k_ring + off, true);
        cp_async16(&vs[buf][kk][c], v_ring + off, true);
      } else if (ki < total_keys) {
        int bj = ki - ctx_len;
        size_t off = ((size_t)(bi * BLK + bj) * KVH + (size_t)kvh) * HD + c;
        cp_async16(&ks[buf][kk][c], k_blk + off, true);
        cp_async16(&vs[buf][kk][c], v_blk + off, true);
      } else {
        cp_async16(&ks[buf][kk][c], k_blk, false);
        cp_async16(&vs[buf][kk][c], v_blk, false);
      }
    }
    cp_commit();
  };

  float m_run[2] = {-INFINITY, -INFINITY};
  float l_run[2] = {0.f, 0.f};
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;

  uint32_t qf[8][4];
  if (t0 < t1) load_tile(t0, 0);
  __syncthreads();
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) ldsm_x4(qf[kk], &qsm[mt * 16 + (lane & 15)][kk * 16 + (lane >> 4) * 8]);

  const int g = lane >> 2;
  const int t = lane & 3;
  const int q_pos0 = p0 + ((mt * 16 + g) >> 2);
  const int q_pos1 = p0 + ((mt * 16 + g + 8) >> 2);

  for (int tt = t0; tt < t1; ++tt) {
    const int buf = (tt - t0) & 1;
    if (tt + 1 < t1) {
      load_tile(tt + 1, buf ^ 1);
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();

    float sc[4][4];
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      sc[nn][0] = sc[nn][1] = sc[nn][2] = sc[nn][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint32_t b0, b1;
        ldsm_x2(b0, b1, &ks[buf][kh * 32 + nn * 8 + (lane & 7)][kk * 16 + ((lane >> 3) & 1) * 8]);
        mma_bf16(sc[nn], qf[kk], b0, b1);
      }
    }

    const int kbase = tt * AT_TILE + kh * 32;
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int ki = kbase + nn * 8 + 2 * t + (e & 1);
        int qp = (e < 2) ? q_pos0 : q_pos1;
        int kp = ctx_start + ki;
        float v = (ki < total_keys && (qp - kp < SWA)) ? (sc[nn][e] * ATTN_SCALE_LOG2) : -INFINITY;
        sc[nn][e] = v;
        mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
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
      o[i][0] *= scale_old[0];
      o[i][1] *= scale_old[0];
      o[i][2] *= scale_old[1];
      o[i][3] *= scale_old[1];
    }

    uint32_t pa[2][4];
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      float p0v = (m_run[0] == -INFINITY) ? 0.f : exp2f(sc[nn][0] - m_run[0]);
      float p1v = (m_run[0] == -INFINITY) ? 0.f : exp2f(sc[nn][1] - m_run[0]);
      float p2v = (m_run[1] == -INFINITY) ? 0.f : exp2f(sc[nn][2] - m_run[1]);
      float p3v = (m_run[1] == -INFINITY) ? 0.f : exp2f(sc[nn][3] - m_run[1]);
      l_run[0] += p0v + p1v;
      l_run[1] += p2v + p3v;
      __nv_bfloat162 lo = __floats2bfloat162_rn(p0v, p1v);
      __nv_bfloat162 hi = __floats2bfloat162_rn(p2v, p3v);
      int ks16 = nn >> 1;
      int half_idx = nn & 1;
      pa[ks16][half_idx * 2 + 0] = *(uint32_t*)&lo;
      pa[ks16][half_idx * 2 + 1] = *(uint32_t*)&hi;
    }

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

#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    l_run[hh] += __shfl_xor_sync(0xffffffffu, l_run[hh], 1);
    l_run[hh] += __shfl_xor_sync(0xffffffffu, l_run[hh], 2);
  }

  float* osm = (float*)&ks[0][0][0];  // [32][128] fp32 = 16 KB (fits inside ks = 34 KB)
  if (kh == 1) {
    if (t == 0) {
      comb[mt][g][0] = m_run[0];
      comb[mt][g][1] = l_run[0];
      comb[mt][g + 8][0] = m_run[1];
      comb[mt][g + 8][1] = l_run[1];
    }
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
      int c = dn * 8 + 2 * t;
      osm[(mt * 16 + g) * HD + c] = o[dn][0];
      osm[(mt * 16 + g) * HD + c + 1] = o[dn][1];
      osm[(mt * 16 + g + 8) * HD + c] = o[dn][2];
      osm[(mt * 16 + g + 8) * HD + c + 1] = o[dn][3];
    }
  }
  __syncthreads();
  if (kh == 0) {
    size_t pbase = (((size_t)bi * KVH + (size_t)kvh) * (size_t)splits + (size_t)sp) * 32;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      int row = g + hh * 8;
      float m2 = comb[mt][row][0];
      float l2 = comb[mt][row][1];
      float mn = fmaxf(m_run[hh], m2);
      float a1 = (m_run[hh] == -INFINITY) ? 0.f : exp2f(m_run[hh] - mn);
      float a2 = (m2 == -INFINITY) ? 0.f : exp2f(m2 - mn);
      float l = l_run[hh] * a1 + l2 * a2;
      float il = (l > 0.f) ? (1.f / l) : 0.f;
      int qrow = mt * 16 + row;
#pragma unroll
      for (int dn = 0; dn < 16; ++dn) {
        int c = dn * 8 + 2 * t;
        float v0 = (o[dn][hh * 2] * a1 + osm[qrow * HD + c] * a2) * il;
        float v1 = (o[dn][hh * 2 + 1] * a1 + osm[qrow * HD + c + 1] * a2) * il;
        *(float2*)&part_o[(pbase + qrow) * HD + c] = make_float2(v0, v1);
      }
      if (t == 0) {
        part_ml[(pbase + qrow) * 2] = mn;
        part_ml[(pbase + qrow) * 2 + 1] = l;
      }
    }
  }
}

// 8c. Design-A warp-independent Tensor Core f16 MMA split-KV Global E4M3 attention over linear context [0..pos) +
// exact BF16->F16 8-token block keys [pos..pos+7]:
// - Zero smem K/V staging, zero __syncthreads() in the hot loop, direct streaming ldg16_stream into registers.
// - K as A operand (16 keys x 16 dims), Q in 8 KB static smem as B operand (4 x n8 tiles = all 32 Q rows):
//   every `place2` E4M3 decode is reused across all 4 tiles (4x decode reuse).
// - 64 threads (2 warps) per CTA, each warp is an independent split `sp = blockIdx.x * 2 + warp` (splits = gridDim.x * 2).
// - Grid: (splits / 2, KVH=8, k), Block: 64, Dynamic smem: 0. Outputs feed `swa_attn_reduce_kernel` directly.
__global__ void __launch_bounds__(64, 4) global_attn_e4m3_split_kernel(const __nv_bfloat16* __restrict__ q_blk,
                                                                       const __nv_bfloat16* __restrict__ k_blk,
                                                                       const __nv_bfloat16* __restrict__ v_blk,
                                                                       const uint8_t* __restrict__ k_glob,
                                                                       const uint8_t* __restrict__ v_glob,
                                                                       const float* __restrict__ ks_glob,
                                                                       const float* __restrict__ vs_glob,
                                                                       const int* __restrict__ pos,
                                                                       const int* __restrict__ sel,
                                                                       float* __restrict__ part_ml,
                                                                       float* __restrict__ part_o,
                                                                       int k,
                                                                       int max_ctx,
                                                                       int kv_stride,
                                                                       int splits) {
  // 8,192 B static smem: holds qs_n[4][8][32] uint2 during compute, reused as [2 warps][8][128] fp32 at flush.
  __shared__ __align__(16) uint2 qs_n[4 * 8 * 32];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int gr = lane >> 2;  // 0..7
  const int tq = lane & 3;   // 0..3
  const int sp = blockIdx.x * 2 + warp;
  const int kvh = blockIdx.y;
  const int bi = blockIdx.z;
  if (bi >= k) return;

  // Stage all 32 Q rows (4 n8 tiles) into qs_n[u][kk][lane] in f16 maya order (each warp writes half of kk).
  {
    const __nv_bfloat16* qbase = q_blk + ((size_t)(bi * BLK) * NH + (size_t)(kvh * GQA)) * HD;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      int col = 8 * u + gr;
      int j = col >> 2;
      int g4 = col & 3;
      const uint16_t* qr = reinterpret_cast<const uint16_t*>(qbase + ((size_t)j * NH + (size_t)g4) * HD);
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        if ((jj & 1) == warp) {
          const uint4 v = *reinterpret_cast<const uint4*>(qr + tq * 32 + jj * 8);
          qs_n[(u * 8 + 2 * jj) * 32 + lane] = make_uint2(bf2_to_h2(v.x), bf2_to_h2(v.y));
          qs_n[(u * 8 + 2 * jj + 1) * 32 + lane] = make_uint2(bf2_to_h2(v.z), bf2_to_h2(v.w));
        }
      }
    }
  }
  __syncthreads();

  const int s = sel[bi];
  const int p0 = pos[s];
  const int ctx_len = (p0 > 0) ? min(p0, max_ctx) : 0;
  const int nsteps_total = (ctx_len + 15) >> 4;
  const int s0 = (int)((long long)sp * nsteps_total / splits);
  const int s1 = (int)((long long)(sp + 1) * nsteps_total / splits);

  const size_t kv_base = ((size_t)s * KVH + (size_t)kvh) * (size_t)kv_stride;
  const uint8_t* k_head = k_glob + kv_base * HD;
  const uint8_t* v_head = v_glob + kv_base * HD;
  const float* ks_head = ks_glob + kv_base;
  const float* vs_head = vs_glob + kv_base;

  float on[4][8][4];
  float ma[4], mb[4], la[4], lb[4];
#pragma unroll
  for (int u = 0; u < 4; ++u) {
#pragma unroll
    for (int i = 0; i < 8; ++i) on[u][i][0] = on[u][i][1] = on[u][i][2] = on[u][i][3] = 0.f;
    ma[u] = mb[u] = -INFINITY;
    la[u] = lb[u] = 0.f;
  }

  const int src = tq * 8 + (gr >> 1);
  const uint32_t psl = (gr & 1) ? 0x7632u : 0x5410u;
  constexpr float K_SCALE_LOG2 = 256.0f * ATTN_SCALE_LOG2;

  uint4 kr[2][2];
  uint4 vr[4];
  float ks0_r = 0.f, ks1_r = 0.f, vs0_r = 0.f, vs1_r = 0.f;

  auto load_k = [&](int st, auto edge_c) {
    constexpr bool EDGE = decltype(edge_c)::value;
    const int k0 = st + gr;
    const int k1 = k0 + 8;
    if constexpr (!EDGE) {
      const uint8_t* p0k = k_head + (size_t)k0 * HD + tq * 32;
      const uint8_t* p1k = p0k + 8 * HD;
      kr[0][0] = ldg16_stream(p0k);
      kr[0][1] = ldg16_stream(p0k + 16);
      kr[1][0] = ldg16_stream(p1k);
      kr[1][1] = ldg16_stream(p1k + 16);
      ks0_r = __ldg(ks_head + k0) * K_SCALE_LOG2;
      ks1_r = __ldg(ks_head + k1) * K_SCALE_LOG2;
    } else {
      const uint4 zz = make_uint4(0u, 0u, 0u, 0u);
      if (k0 < ctx_len) {
        const uint8_t* p0k = k_head + (size_t)k0 * HD + tq * 32;
        kr[0][0] = ldg16_stream(p0k);
        kr[0][1] = ldg16_stream(p0k + 16);
        ks0_r = __ldg(ks_head + k0) * K_SCALE_LOG2;
      } else {
        kr[0][0] = kr[0][1] = zz;
        ks0_r = 0.f;
      }
      if (k1 < ctx_len) {
        const uint8_t* p1k = k_head + (size_t)k1 * HD + tq * 32;
        kr[1][0] = ldg16_stream(p1k);
        kr[1][1] = ldg16_stream(p1k + 16);
        ks1_r = __ldg(ks_head + k1) * K_SCALE_LOG2;
      } else {
        kr[1][0] = kr[1][1] = zz;
        ks1_r = 0.f;
      }
    }
  };

  auto load_v = [&](int st, auto edge_c) {
    constexpr bool EDGE = decltype(edge_c)::value;
    const int kv0 = st + 2 * tq;
    const int k0 = st + gr;
    const int k1 = k0 + 8;
    if constexpr (!EDGE) {
      const uint8_t* pv = v_head + (size_t)kv0 * HD + gr * 16;
      vr[0] = ldg16_stream(pv);
      vr[1] = ldg16_stream(pv + HD);
      vr[2] = ldg16_stream(pv + 8 * HD);
      vr[3] = ldg16_stream(pv + 9 * HD);
      vs0_r = __ldg(vs_head + k0) * 256.0f;
      vs1_r = __ldg(vs_head + k1) * 256.0f;
    } else {
      const uint4 zz = make_uint4(0u, 0u, 0u, 0u);
      vr[0] = (kv0 < ctx_len) ? ldg16_stream(v_head + (size_t)kv0 * HD + gr * 16) : zz;
      vr[1] = (kv0 + 1 < ctx_len) ? ldg16_stream(v_head + (size_t)(kv0 + 1) * HD + gr * 16) : zz;
      vr[2] = (kv0 + 8 < ctx_len) ? ldg16_stream(v_head + (size_t)(kv0 + 8) * HD + gr * 16) : zz;
      vr[3] = (kv0 + 9 < ctx_len) ? ldg16_stream(v_head + (size_t)(kv0 + 9) * HD + gr * 16) : zz;
      vs0_r = (k0 < ctx_len) ? (__ldg(vs_head + k0) * 256.0f) : 0.f;
      vs1_r = (k1 < ctx_len) ? (__ldg(vs_head + k1) * 256.0f) : 0.f;
    }
  };

  float sn[4][4];
  auto run_qk = [&]() {
#pragma unroll
    for (int u = 0; u < 4; ++u) sn[u][0] = sn[u][1] = sn[u][2] = sn[u][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const uint4& x = kr[0][kk >> 2];
      const uint4& y = kr[1][kk >> 2];
      const uint32_t wx = (kk & 3) == 0 ? x.x : (kk & 3) == 1 ? x.y : (kk & 3) == 2 ? x.z : x.w;
      const uint32_t wy = (kk & 3) == 0 ? y.x : (kk & 3) == 1 ? y.y : (kk & 3) == 2 ? y.z : y.w;
      const uint32_t a0 = place2(wx, wx, psel(0, 1));
      const uint32_t a2 = place2(wx, wx, psel(2, 3));
      const uint32_t a1 = place2(wy, wy, psel(0, 1));
      const uint32_t a3 = place2(wy, wy, psel(2, 3));
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint32_t b0, b1;
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                     : "=r"(b0), "=r"(b1)
                     : "r"((unsigned)__cvta_generic_to_shared(qs_n + (u * 8 + kk) * 32 + lane)));
        mma_f16_r(sn[u], a0, a1, a2, a3, b0, b1);
      }
    }
  };

  auto run_softmax_pv = [&](int kt, float ks0, float ks1, float vs0, float vs1, auto edge_c) {
    constexpr bool EDGE = decltype(edge_c)::value;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float ua = (ma[u] == -INFINITY) ? 0.f : ma[u];
      const float ub = (mb[u] == -INFINITY) ? 0.f : mb[u];
      sn[u][0] = fmaf(sn[u][0], ks0, -ua);
      sn[u][1] = fmaf(sn[u][1], ks0, -ub);
      sn[u][2] = fmaf(sn[u][2], ks1, -ua);
      sn[u][3] = fmaf(sn[u][3], ks1, -ub);
      if constexpr (EDGE) {
        if (kt + gr >= ctx_len) sn[u][0] = sn[u][1] = -INFINITY;
        if (kt + gr + 8 >= ctx_len) sn[u][2] = sn[u][3] = -INFINITY;
      }
    }
    bool trig = false;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float ta = (ma[u] == -INFINITY) ? -INFINITY : 0.f;
      const float tb = (mb[u] == -INFINITY) ? -INFINITY : 0.f;
      trig = trig || (sn[u][0] > ta || sn[u][2] > ta || sn[u][1] > tb || sn[u][3] > tb);
    }
    if (__any_sync(0xffffffffu, trig)) {
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const float ta = (ma[u] == -INFINITY) ? -INFINITY : 0.f;
        const float tb = (mb[u] == -INFINITY) ? -INFINITY : 0.f;
        float xa = fmaxf(sn[u][0], sn[u][2]);
        float xb = fmaxf(sn[u][1], sn[u][3]);
#pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
          xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, sh));
          xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, sh));
        }
        if (xa > ta) {
          const float f = (ma[u] == -INFINITY) ? 0.f : ex2_fast(-xa);
          ma[u] = ((ma[u] == -INFINITY) ? 0.f : ma[u]) + xa;
          la[u] *= f;
          sn[u][0] -= xa;
          sn[u][2] -= xa;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            on[u][i][0] *= f;
            on[u][i][2] *= f;
          }
        }
        if (xb > tb) {
          const float f = (mb[u] == -INFINITY) ? 0.f : ex2_fast(-xb);
          mb[u] = ((mb[u] == -INFINITY) ? 0.f : mb[u]) + xb;
          lb[u] *= f;
          sn[u][1] -= xb;
          sn[u][3] -= xb;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            on[u][i][1] *= f;
            on[u][i][3] *= f;
          }
        }
      }
    }
    uint32_t pb0[4], pb1[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float pn0 = ex2_fast(sn[u][0]);
      const float pn1 = ex2_fast(sn[u][1]);
      const float pn2 = ex2_fast(sn[u][2]);
      const float pn3 = ex2_fast(sn[u][3]);
      la[u] += pn0 + pn2;
      lb[u] += pn1 + pn3;
      const uint32_t plo = pack2_f16(pn0 * vs0, pn1 * vs0);
      const uint32_t phi = pack2_f16(pn2 * vs1, pn3 * vs1);
      const uint32_t alo = __shfl_sync(0xffffffffu, plo, src);
      const uint32_t blo = __shfl_sync(0xffffffffu, plo, src + 4);
      const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src);
      const uint32_t bhi = __shfl_sync(0xffffffffu, phi, src + 4);
      pb0[u] = __byte_perm(alo, blo, psl);
      pb1[u] = __byte_perm(ahi, bhi, psl);
    }
    const uint32_t x0[4] = {vr[0].x, vr[0].y, vr[0].z, vr[0].w};
    const uint32_t x1[4] = {vr[1].x, vr[1].y, vr[1].z, vr[1].w};
    const uint32_t x8[4] = {vr[2].x, vr[2].y, vr[2].z, vr[2].w};
    const uint32_t x9[4] = {vr[3].x, vr[3].y, vr[3].z, vr[3].w};
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
        for (int u = 0; u < 4; ++u) mma_f16_r(on[u][b], lo01, hi01, lo89, hi89, pb0[u], pb1[u]);
      }
    }
  };

  // Register-prefetched streaming loop over [s0, s1): branchless interior loop + at most 1 edge step.
  const int nsteps_full = ctx_len >> 4;
  const int s1_full = min(s1, nsteps_full);
  if (s0 < s1_full) {
    load_k(s0 << 4, std::false_type{});
    load_v(s0 << 4, std::false_type{});
    for (int st = s0; st < s1_full; ++st) {
      const float ks0 = ks0_r, ks1 = ks1_r, vs0 = vs0_r, vs1 = vs1_r;
      run_qk();
      if (st + 1 < s1_full) load_k((st + 1) << 4, std::false_type{});
      run_softmax_pv(st << 4, ks0, ks1, vs0, vs1, std::false_type{});
      if (st + 1 < s1_full) load_v((st + 1) << 4, std::false_type{});
    }
  }
  if (s1 > s1_full && s0 <= s1_full) {
    const int kt = s1_full << 4;
    load_k(kt, std::true_type{});
    load_v(kt, std::true_type{});
    run_qk();
    run_softmax_pv(kt, ks0_r, ks1_r, vs0_r, vs1_r, std::true_type{});
  }

  // Exact BF16 (exact in F16 normal range) 8-token bidirectional block keys [pos..pos+7], once on sp == splits - 1.
  if (sp == splits - 1) {
#pragma unroll
    for (int u = 0; u < 4; ++u) sn[u][0] = sn[u][1] = sn[u][2] = sn[u][3] = 0.f;
    const uint16_t* krow =
        reinterpret_cast<const uint16_t*>(k_blk + ((size_t)(bi * BLK + gr) * KVH + (size_t)kvh) * HD + tq * 32);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const uint4 v = *reinterpret_cast<const uint4*>(krow + jj * 8);
      const uint32_t a0_0 = bf2_to_h2(v.x), a2_0 = bf2_to_h2(v.y);
      const uint32_t a0_1 = bf2_to_h2(v.z), a2_1 = bf2_to_h2(v.w);
      const int kk0 = 2 * jj;
      const int kk1 = 2 * jj + 1;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint32_t b0, b1;
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                     : "=r"(b0), "=r"(b1)
                     : "r"((unsigned)__cvta_generic_to_shared(qs_n + (u * 8 + kk0) * 32 + lane)));
        mma_f16_r(sn[u], a0_0, 0u, a2_0, 0u, b0, b1);
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                     : "=r"(b0), "=r"(b1)
                     : "r"((unsigned)__cvta_generic_to_shared(qs_n + (u * 8 + kk1) * 32 + lane)));
        mma_f16_r(sn[u], a0_1, 0u, a2_1, 0u, b0, b1);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float ua = (ma[u] == -INFINITY) ? 0.f : ma[u];
      const float ub = (mb[u] == -INFINITY) ? 0.f : mb[u];
      sn[u][0] = fmaf(sn[u][0], ATTN_SCALE_LOG2, -ua);
      sn[u][1] = fmaf(sn[u][1], ATTN_SCALE_LOG2, -ub);
      sn[u][2] = -INFINITY;
      sn[u][3] = -INFINITY;
    }
    bool trig = false;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float ta = (ma[u] == -INFINITY) ? -INFINITY : 0.f;
      const float tb = (mb[u] == -INFINITY) ? -INFINITY : 0.f;
      trig = trig || (sn[u][0] > ta || sn[u][1] > tb);
    }
    if (__any_sync(0xffffffffu, trig)) {
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const float ta = (ma[u] == -INFINITY) ? -INFINITY : 0.f;
        const float tb = (mb[u] == -INFINITY) ? -INFINITY : 0.f;
        float xa = sn[u][0];
        float xb = sn[u][1];
#pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
          xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, sh));
          xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, sh));
        }
        if (xa > ta) {
          const float f = (ma[u] == -INFINITY) ? 0.f : ex2_fast(-xa);
          ma[u] = ((ma[u] == -INFINITY) ? 0.f : ma[u]) + xa;
          la[u] *= f;
          sn[u][0] -= xa;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            on[u][i][0] *= f;
            on[u][i][2] *= f;
          }
        }
        if (xb > tb) {
          const float f = (mb[u] == -INFINITY) ? 0.f : ex2_fast(-xb);
          mb[u] = ((mb[u] == -INFINITY) ? 0.f : mb[u]) + xb;
          lb[u] *= f;
          sn[u][1] -= xb;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            on[u][i][1] *= f;
            on[u][i][3] *= f;
          }
        }
      }
    }
    uint32_t pb0[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float pn0 = ex2_fast(sn[u][0]);
      const float pn1 = ex2_fast(sn[u][1]);
      la[u] += pn0;
      lb[u] += pn1;
      const uint32_t plo = pack2_f16(pn0, pn1);
      const uint32_t alo = __shfl_sync(0xffffffffu, plo, src);
      const uint32_t blo = __shfl_sync(0xffffffffu, plo, src + 4);
      pb0[u] = __byte_perm(alo, blo, psl);
    }
    const __nv_bfloat16* v0p = v_blk + ((size_t)(bi * BLK + 2 * tq) * KVH + (size_t)kvh) * HD + gr * 16;
    const __nv_bfloat16* v1p = v_blk + ((size_t)(bi * BLK + 2 * tq + 1) * KVH + (size_t)kvh) * HD + gr * 16;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const float a00 = __bfloat162float(v0p[2 * b]);
      const float a01 = __bfloat162float(v1p[2 * b]);
      const float a10 = __bfloat162float(v0p[2 * b + 1]);
      const float a11 = __bfloat162float(v1p[2 * b + 1]);
      const uint32_t lo01 = pack2_f16(a00, a01);
      const uint32_t hi01 = pack2_f16(a10, a11);
#pragma unroll
      for (int u = 0; u < 4; ++u) mma_f16_r(on[u][b], lo01, hi01, 0u, 0u, pb0[u], 0u);
    }
  }

  // Wait for both warps to finish reading qs_n, then reuse qs_n (8 KB) as [2 warps][8][128] fp32 coalesced flush stage.
  __syncthreads();
  float* st_buf = reinterpret_cast<float*>(qs_n) + warp * 8 * HD;
  const size_t pbase = (((size_t)bi * KVH + (size_t)kvh) * (size_t)splits + (size_t)sp) * 32;
  const int ca = 2 * tq;
  const int cb = ca + 1;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    float sa = la[u];
    float sb = lb[u];
#pragma unroll
    for (int sh = 4; sh < 32; sh <<= 1) {
      sa += __shfl_xor_sync(0xffffffffu, sa, sh);
      sb += __shfl_xor_sync(0xffffffffu, sb, sh);
    }
    const float ia = (sa > 0.f) ? (1.f / sa) : 0.f;
    const float ib = (sb > 0.f) ? (1.f / sb) : 0.f;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int d = gr * 16 + 2 * b;
      st_buf[ca * HD + d] = on[u][b][0] * ia;
      st_buf[ca * HD + d + 1] = on[u][b][2] * ia;
      st_buf[cb * HD + d] = on[u][b][1] * ib;
      st_buf[cb * HD + d + 1] = on[u][b][3] * ib;
    }
    __syncwarp();
    float4* dst = reinterpret_cast<float4*>(part_o + (pbase + (size_t)(8 * u)) * HD);
    const float4* sv = reinterpret_cast<const float4*>(st_buf);
    for (int i = lane; i < 8 * (HD / 4); i += 32) dst[i] = sv[i];
    if (gr == 0) {
      part_ml[(pbase + (size_t)(8 * u + ca)) * 2] = ma[u];
      part_ml[(pbase + (size_t)(8 * u + ca)) * 2 + 1] = sa;
      part_ml[(pbase + (size_t)(8 * u + cb)) * 2] = mb[u];
      part_ml[(pbase + (size_t)(8 * u + cb)) * 2 + 1] = sb;
    }
    __syncwarp();
  }
}

// 8b. Reduce split-KV partial accumulators -> attn_out [k*8, 4096] bf16.
// Grid: (k * KVH * 32), Block: 128.
__global__ void swa_attn_reduce_kernel(const float* __restrict__ part_ml,
                                       const float* __restrict__ part_o,
                                       __nv_bfloat16* __restrict__ out,
                                       int k,
                                       int splits) {
  int idx = blockIdx.x;
  if (idx >= k * KVH * 32) return;
  int qr = idx & 31;
  int rh = idx >> 5;
  int bi = rh / KVH;
  int kvh = rh % KVH;
  int d = threadIdx.x;

  size_t base = ((size_t)bi * KVH + (size_t)kvh) * (size_t)splits * 32;
  float mx = -INFINITY;
  for (int sp = 0; sp < splits; ++sp) {
    mx = fmaxf(mx, part_ml[(base + (size_t)sp * 32 + qr) * 2]);
  }
  float num = 0.f;
  float den = 0.f;
  for (int sp = 0; sp < splits; ++sp) {
    float m = part_ml[(base + (size_t)sp * 32 + qr) * 2];
    float l = part_ml[(base + (size_t)sp * 32 + qr) * 2 + 1];
    if (m == -INFINITY || l == 0.f) continue;
    float w = exp2f(m - mx) * l;
    num += part_o[(base + (size_t)sp * 32 + qr) * HD + d] * w;
    den += w;
  }
  int j = qr >> 2;
  int g4 = qr & 3;
  out[((size_t)(bi * BLK + j) * NH + (size_t)(kvh * GQA + g4)) * HD + d] =
      __float2bfloat16_rn(den > 0.f ? (num / den) : 0.f);
}

// 9. Fused SiLU(gate) * up: gate_up [rows, 2*INTER] -> out [rows, INTER] bf16.
// Grid: (rows, 27), Block: 256 (27 * 256 == 6912).
__global__ void silu_mul_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                __nv_bfloat16* __restrict__ out,
                                int rows) {
  int r = blockIdx.x;
  int c = blockIdx.y * 256 + threadIdx.x;
  if (r >= rows || c >= INTER) return;
  const __nv_bfloat16* src = gate_up + (size_t)r * (2 * INTER);
  float g = __bfloat162float(src[c]);
  float u = __bfloat162float(src[INTER + c]);
  float sg = g / (1.0f + expf(-g));
  out[(size_t)r * INTER + c] = __float2bfloat16(sg * u);
}

// 10a. Exact 2-pass threshold top-16, pass 1: float4 vectorized chunk maxima [rows, 64] + full-vocab softcapped
// exp-sum chunk_z [rows, 64] + zero cand_count[rows].
// Grid: (rows, 64), Block: 256.
__global__ void topk16_chunk_max_kernel(const float* __restrict__ logits,
                                        float* __restrict__ chunk_max,
                                        float* __restrict__ chunk_z,
                                        int* __restrict__ cand_count,
                                        const float* __restrict__ samp_temp,
                                        const int* __restrict__ sel,
                                        int rows,
                                        int logit_stride) {
  int r = blockIdx.x;
  int ch = blockIdx.y;
  if (r >= rows) return;
  if (ch == 0 && threadIdx.x == 0) cand_count[r] = 0;

  int bi = r / DRAFTS;
  float t = (samp_temp && sel) ? samp_temp[sel[bi]] : 0.0f;
  float inv_t = (t >= 1e-5f) ? (1.0f / t) : 1.0f;

  const float* rowp = logits + (size_t)r * (size_t)logit_stride;
  int beg = ch * TOPK_CHUNK_LEN;
  int end = min(beg + TOPK_CHUNK_LEN, VALID_VOCAB);
  float mx = -1e38f;
  float zsum = 0.0f;
  constexpr float INV_SC = 1.0f / SOFTCAP;
  // TOPK_CHUNK_LEN = 3136 -> 784 float4s; logit_stride = 200064 is multiple of 4.
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    int p = threadIdx.x + u * 256;
    int idx4 = beg + p * 4;
    if (p < (TOPK_CHUNK_LEN / 4) && idx4 < logit_stride) {
      float4 v = *reinterpret_cast<const float4*>(rowp + idx4);
      if (idx4 + 0 < end) {
        mx = fmaxf(mx, v.x);
        zsum += __expf((SOFTCAP * tanhf(v.x * INV_SC) - SOFTCAP) * inv_t);
      }
      if (idx4 + 1 < end) {
        mx = fmaxf(mx, v.y);
        zsum += __expf((SOFTCAP * tanhf(v.y * INV_SC) - SOFTCAP) * inv_t);
      }
      if (idx4 + 2 < end) {
        mx = fmaxf(mx, v.z);
        zsum += __expf((SOFTCAP * tanhf(v.z * INV_SC) - SOFTCAP) * inv_t);
      }
      if (idx4 + 3 < end) {
        mx = fmaxf(mx, v.w);
        zsum += __expf((SOFTCAP * tanhf(v.w * INV_SC) - SOFTCAP) * inv_t);
      }
    }
  }
  __shared__ float smem[8];
  float bm = block_max_256(mx, smem);
  if (threadIdx.x == 0) chunk_max[(size_t)r * TOPK_CHUNKS + ch] = bm;
  __syncthreads();
  zsum = warp_sum(zsum);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = zsum;
  __syncthreads();
  if (threadIdx.x == 0 && chunk_z) {
    float bz = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) bz += smem[w];
    chunk_z[(size_t)r * TOPK_CHUNKS + ch] = bz;
  }
}

// 10b. Exact 2-pass threshold top-16, pass 2: compute 16th-largest chunk_max as guaranteed lower bound `thresh`,
// skip all chunks with chunk_max < thresh (~75% of HBM traffic skipped), emit exact <= 16 winners per surviving chunk.
// Grid: (rows, 64), Block: 256.
__global__ void topk16_filter_kernel(const float* __restrict__ logits,
                                     const float* __restrict__ chunk_max,
                                     float* __restrict__ s1_vals,
                                     int* __restrict__ s1_ids,
                                     int* __restrict__ cand_count,
                                     int rows,
                                     int logit_stride) {
  int r = blockIdx.x;
  int ch = blockIdx.y;
  if (r >= rows) return;

  __shared__ float s_thresh;
  if (threadIdx.x < 32) {
    int lane = threadIdx.x;
    float a = chunk_max[(size_t)r * TOPK_CHUNKS + lane];
    float b = chunk_max[(size_t)r * TOPK_CHUNKS + 32 + lane];
    float kth = -1e38f;
    for (int t = 0; t < TOP_K; ++t) {
      float mv = (a >= b) ? a : b;
      int my_id = (a >= b) ? lane : (32 + lane);
      float bv = mv;
      int bi = my_id;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (bi == lane) a = -1e38f;
      if (bi == 32 + lane) b = -1e38f;
      if (t == TOP_K - 1) kth = bv;
    }
    if (lane == 0) s_thresh = kth;
  }
  __syncthreads();

  float thresh = s_thresh;
  if (chunk_max[(size_t)r * TOPK_CHUNKS + ch] < thresh) return;

  const float* rowp = logits + (size_t)r * (size_t)logit_stride;
  int beg = ch * TOPK_CHUNK_LEN;
  int end = min(beg + TOPK_CHUNK_LEN, VALID_VOCAB);

  __shared__ float sh_v[256];
  __shared__ int sh_i[256];
  __shared__ int sh_n;
  __shared__ int s_base;
  __shared__ float sv[8];
  __shared__ int si[8];
  if (threadIdx.x == 0) sh_n = 0;
  __syncthreads();

  // Scan chunk with float4 and collect all elements >= thresh.
  float rv[16];
  int ri[16];
  int rn = 0;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    int p = threadIdx.x + u * 256;
    int idx4 = beg + p * 4;
    if (p < (TOPK_CHUNK_LEN / 4) && idx4 < logit_stride) {
      float4 v = *reinterpret_cast<const float4*>(rowp + idx4);
      float arr[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        int id = idx4 + q;
        float val = (id < end) ? arr[q] : -1e38f;
        rv[rn] = val;
        ri[rn] = (id < end) ? id : 0x7fffffff;
        ++rn;
        if (id < end && val >= thresh) {
          int pos = atomicAdd(&sh_n, 1);
          if (pos < 256) {
            sh_v[pos] = val;
            sh_i[pos] = id;
          }
        }
      }
    }
  }
  __syncthreads();

  int total_found = sh_n;
  if (total_found <= 0) return;

  if (total_found <= 256) {
    int take = min(total_found, TOP_K);
    if (threadIdx.x == 0) s_base = atomicAdd(&cand_count[r], take);
    __syncthreads();
    int base = s_base;
    float lv = (threadIdx.x < total_found) ? sh_v[threadIdx.x] : -1e38f;
    int li = (threadIdx.x < total_found) ? sh_i[threadIdx.x] : 0x7fffffff;
    for (int t = 0; t < take; ++t) {
      float gv;
      int gi;
      block_argmax_256(lv, li, sv, si, gv, gi);
      if (li == gi) {
        lv = -1e38f;
        li = 0x7fffffff;
      }
      if (threadIdx.x == 0) {
        s1_vals[(size_t)r * (TOPK_CHUNKS * TOP_K) + base + t] = gv;
        s1_ids[(size_t)r * (TOPK_CHUNKS * TOP_K) + base + t] = gi;
      }
    }
  } else {
    // Degenerate fallback (>256 exact ties >= thresh inside one chunk): exact 16-pass over this chunk.
    if (threadIdx.x == 0) s_base = atomicAdd(&cand_count[r], TOP_K);
    __syncthreads();
    int base = s_base;
    for (int t = 0; t < TOP_K; ++t) {
      float best_v = -1e38f;
      int best_i = 0x7fffffff;
      int best_u = -1;
      for (int u = 0; u < rn; ++u) {
        if (rv[u] > best_v || (rv[u] == best_v && ri[u] < best_i)) {
          best_v = rv[u];
          best_i = ri[u];
          best_u = u;
        }
      }
      float gv;
      int gi;
      block_argmax_256(best_v, best_i, sv, si, gv, gi);
      if (best_u >= 0 && best_i == gi) {
        rv[best_u] = -1e38f;
        ri[best_u] = 0x7fffffff;
      }
      if (threadIdx.x == 0) {
        s1_vals[(size_t)r * (TOPK_CHUNKS * TOP_K) + base + t] = gv;
        s1_ids[(size_t)r * (TOPK_CHUNKS * TOP_K) + base + t] = gi;
      }
    }
  }
}

// 10c. Exact top-16 stage 2: merge cand_count[r] <= 1024 survivors per row -> sorted top-16 + full-vocab row_z[r].
// Grid: (rows), Block: 256.
__global__ void topk16_stage2_kernel(const float* __restrict__ s1_vals,
                                     const int* __restrict__ s1_ids,
                                     const int* __restrict__ cand_count,
                                     const float* __restrict__ chunk_z,
                                     float* __restrict__ unary_out,
                                     int* __restrict__ cand_ids_out,
                                     float* __restrict__ row_z_out,
                                     int rows) {
  int r = blockIdx.x;
  if (r >= rows) return;
  __shared__ float sv[8];
  __shared__ int si[8];
  if (chunk_z && row_z_out) {
    float zl = (threadIdx.x < TOPK_CHUNKS) ? chunk_z[(size_t)r * TOPK_CHUNKS + threadIdx.x] : 0.0f;
    zl = warp_sum(zl);
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = zl;
    __syncthreads();
    if (threadIdx.x == 0) row_z_out[r] = sv[0] + sv[1];
    __syncthreads();
  }
  int cnt = cand_count[r];
  if (cnt > TOPK_CHUNKS * TOP_K) cnt = TOPK_CHUNKS * TOP_K;
  const float* iv = s1_vals + (size_t)r * (TOPK_CHUNKS * TOP_K);
  const int* ii = s1_ids + (size_t)r * (TOPK_CHUNKS * TOP_K);
  float lv[4];
  int li[4];
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    int p = threadIdx.x + u * 256;
    if (p < cnt) {
      lv[u] = iv[p];
      li[u] = ii[p];
    } else {
      lv[u] = -1e38f;
      li[u] = 0x7fffffff;
    }
  }
  float* ov = unary_out + (size_t)r * TOP_K;
  int* oi = cand_ids_out + (size_t)r * TOP_K;

  for (int t = 0; t < TOP_K; ++t) {
    float best_v = -1e38f;
    int best_i = 0x7fffffff;
    int best_u = -1;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (lv[u] > best_v || (lv[u] == best_v && li[u] < best_i)) {
        best_v = lv[u];
        best_i = li[u];
        best_u = u;
      }
    }
    float gv;
    int gi;
    block_argmax_256(best_v, best_i, sv, si, gv, gi);
    if (best_u >= 0 && best_i == gi) {
      lv[best_u] = -1e38f;
      li[best_u] = 0x7fffffff;
    }
    if (threadIdx.x == 0) {
      ov[t] = gv;
      oi[t] = gi;
    }
  }
}

// 11. 16-warp candidate-parallel path selector walk (7 slots, top_k=16, rank=256, softcap=30.0).
// Greedy when samp_temp[s] < 1e-5f; otherwise samples c* ~ softmax(sc / temp) over the 16 candidates quantized to
// exact integer weights summing to PROP_QTOT = 2^22, writing (prop_ids, prop_w) for exact Leviathan rejection sampling.
// Grid: (k), Block: 512 (16 warps: warp c scores candidate c in parallel across 32 lanes x 8 rank dims).
constexpr unsigned int PROP_QTOT = 1u << 22;
constexpr unsigned int DF2_DRFT_PURPOSE = 0x44524654u;  // "DRFT"

__device__ __forceinline__ uint4 df2_philox(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const unsigned int lo0 = 0xD2511F53u * c.x, hi0 = __umulhi(0xD2511F53u, c.x);
    const unsigned int lo1 = 0xCD9E8D57u * c.z, hi1 = __umulhi(0xCD9E8D57u, c.z);
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

__global__ void selector_walk_kernel(const int* __restrict__ cand_ids,
                                     const float* __restrict__ unary,
                                     const float* __restrict__ row_z,
                                     const float* __restrict__ hp,
                                     const __nv_bfloat16* __restrict__ pred_cb,
                                     const __nv_bfloat16* __restrict__ succ_cb,
                                     const int* __restrict__ anchor,
                                     const int* __restrict__ pos,
                                     const int* __restrict__ sel,
                                     const float* __restrict__ samp_temp,
                                     const float* __restrict__ samp_top_p,
                                     const unsigned long long* __restrict__ samp_seed,
                                     int* __restrict__ drafts,
                                     int* __restrict__ prop_ids,
                                     unsigned int* __restrict__ prop_w,
                                     float* __restrict__ conf,
                                     int conf_mode,
                                     int k) {
  int bi = blockIdx.x;
  if (bi >= k) return;
  int s = sel[bi];
  int wid = threadIdx.x >> 5;   // 0..15 (candidate index)
  int lane = threadIdx.x & 31;  // 0..31

  __shared__ float sh_ph[SEL_RANK];
  __shared__ float sh_sc[TOP_K];
  __shared__ float sh_u[TOP_K];
  __shared__ float sh_ex[TOP_K];
  __shared__ unsigned int sh_w[TOP_K];
  __shared__ int sh_id[TOP_K];
  __shared__ int s_pred;
  if (threadIdx.x == 0) s_pred = anchor[s];
  __syncthreads();

  float temp = samp_temp ? samp_temp[s] : 0.0f;
  float top_p = samp_top_p ? samp_top_p[s] : 1.0f;
  unsigned long long seed = samp_seed ? samp_seed[s] : 0ull;
  int p0 = pos ? pos[s] : 0;
  float cum_conf = 0.0f;

  for (int slot = 0; slot < DRAFTS; ++slot) {
    int pred = s_pred;
    int row = bi * DRAFTS + slot;
    if (threadIdx.x < SEL_RANK) {
      sh_ph[threadIdx.x] =
          __bfloat162float(pred_cb[(size_t)pred * SEL_RANK + threadIdx.x]) * hp[(size_t)row * SEL_RANK + threadIdx.x];
    }
    __syncthreads();

    int cid = cand_ids[(size_t)row * TOP_K + wid];
    const __nv_bfloat16* scb = succ_cb + (size_t)cid * SEL_RANK;
    float dot = 0.0f;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      int d = lane + u * 32;
      dot += sh_ph[d] * __bfloat162float(scb[d]);
    }
    dot = warp_sum(dot);
    if (lane == 0) {
      float uv = unary[(size_t)row * TOP_K + wid];
      float sc = uv + dot;
      sh_u[wid] = uv;
      sh_sc[wid] = SOFTCAP * tanhf(sc / SOFTCAP);
      sh_id[wid] = cid;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      // Exact corrected full-vocab partition function Z_corr at inv_t:
      // replace top-16 unary contributions with bilinear selector scores while retaining exact tail mass.
      const float inv_t = (temp >= 1e-5f) ? (1.0f / temp) : 1.0f;
      float z_u16 = 0.0f;
      float z_s16 = 0.0f;
      constexpr float INV_SC = 1.0f / SOFTCAP;
      for (int c = 0; c < TOP_K; ++c) {
        z_u16 += expf((SOFTCAP * tanhf(sh_u[c] * INV_SC) - SOFTCAP) * inv_t);
        z_s16 += expf((sh_sc[c] - SOFTCAP) * inv_t);
      }
      float z_tail = row_z ? fmaxf(0.0f, row_z[row] - z_u16) : 0.0f;

      // Sort sh_sc/sh_id in-place descending (zero register spill).
      for (int c = 1; c < TOP_K; ++c) {
        float sv = sh_sc[c];
        int iv = sh_id[c];
        int j = c;
        while (j > 0 && (sv > sh_sc[j - 1] || (sv == sh_sc[j - 1] && iv < sh_id[j - 1]))) {
          sh_sc[j] = sh_sc[j - 1];
          sh_id[j] = sh_id[j - 1];
          --j;
        }
        sh_sc[j] = sv;
        sh_id[j] = iv;
      }
      float best_sc = sh_sc[0];
      float z_corr = fmaxf(z_s16 + z_tail, expf((best_sc - SOFTCAP) * inv_t));

      if (!(temp >= 1e-5f) || !prop_ids || !prop_w) {
        cum_conf += (best_sc - SOFTCAP) * inv_t - logf(z_corr);
        if (conf) conf[(size_t)s * DRAFTS + slot] = cum_conf;
        s_pred = sh_id[0];
        drafts[(size_t)s * DRAFTS + slot] = sh_id[0];
      } else {
        float sum = 0.0f;
        for (int c = 0; c < TOP_K; ++c) {
          float e = expf((sh_sc[c] - best_sc) * inv_t);
          sh_ex[c] = (e == e && e > 0.0f) ? e : 0.0f;
          sum += sh_ex[c];
        }
        int keep = TOP_K;
        if (top_p > 0.0f && top_p < 1.0f && sum > 0.0f) {
          float denom = sum;
          if (row_z) {
            float z_full = z_corr * expf((SOFTCAP - best_sc) * inv_t);
            if (isfinite(z_full) && z_full > denom) denom = z_full;
          }
          float thresh = top_p * denom;
          float cum = 0.0f;
          for (int c = 0; c < TOP_K; ++c) {
            cum += sh_ex[c];
            if (cum >= thresh) {
              keep = c + 1;
              break;
            }
          }
        }
        float keep_sum = 0.0f;
        for (int c = 0; c < keep; ++c) keep_sum += sh_ex[c];
        if (keep == 1 || !(keep_sum > 0.0f)) {
          sh_w[0] = PROP_QTOT;
          for (int c = 1; c < TOP_K; ++c) sh_w[c] = 0u;
        } else {
          float inv_s = (float)PROP_QTOT / keep_sum;
          unsigned int wsum = 0u;
          for (int c = 0; c < keep; ++c) {
            float vf = floorf(sh_ex[c] * inv_s);
            unsigned int wi = (vf > 0.0f) ? (vf >= (float)PROP_QTOT ? PROP_QTOT : (unsigned int)vf) : 0u;
            if (wi > PROP_QTOT - wsum) wi = PROP_QTOT - wsum;
            sh_w[c] = wi;
            wsum += wi;
          }
          for (int c = keep; c < TOP_K; ++c) sh_w[c] = 0u;
          sh_w[0] += (PROP_QTOT - wsum);
        }
        uint2 key = make_uint2((unsigned int)seed, (unsigned int)(seed >> 32));
        uint4 rnd = df2_philox(make_uint4((unsigned int)(p0 + slot + 1), DF2_DRFT_PURPOSE, 0u, 0u), key);
        unsigned int R = rnd.x & (PROP_QTOT - 1u);
        unsigned int acc = 0u;
        int pick = 0;
        for (int c = 0; c < keep; ++c) {
          acc += sh_w[c];
          if (R < acc) {
            pick = c;
            break;
          }
        }
        if (conf_mode == 1) {
          cum_conf += (sh_sc[pick] - SOFTCAP) * inv_t - logf(z_corr);
        } else if (conf_mode == 2) {
          cum_conf += logf(fmaxf((float)sh_w[pick] / (float)PROP_QTOT, 1e-9f));
        } else {
          cum_conf += (best_sc - SOFTCAP) * inv_t - logf(z_corr);
        }
        if (conf) conf[(size_t)s * DRAFTS + slot] = cum_conf;
        s_pred = sh_id[pick];
        drafts[(size_t)s * DRAFTS + slot] = sh_id[pick];
        size_t pbase = ((size_t)s * DRAFTS + (size_t)slot) * TOP_K;
        for (int c = 0; c < TOP_K; ++c) {
          prop_ids[pbase + c] = sh_id[c];
          prop_w[pbase + c] = sh_w[c];
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace df2
