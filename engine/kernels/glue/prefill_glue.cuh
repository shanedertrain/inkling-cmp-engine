// inkling-cmp-engine native runtime: glue kernels at prefill row counts (own code).
//
// The decode-shaped glue (glue.cuh) gives every row its own block and re-reads small per-layer weights from L2 once per
// row (the message-gate controllers: 2 x 590 KB per row; the relative projection: 32 KB per row and head). At 2,048
// rows that L2 traffic dominates the prefill glue. The kernels here keep one weight slice per block (registers or shared
// memory) and loop over many rows. Every output element is the same arithmetic in the same order as the kernel it
// replaces (explicit __fmaf_rn / __fmul_rn / __fadd_rn chains), so results are bit-identical to glue.cuh whatever the
// row count: chunk and batch invariance are unchanged.
#pragma once
#include "glue.cuh"
#include "moe_dev.cuh"

namespace pglue {

using glue::bf16;
using glue::H;
using glue::NH;
using glue::DREL;
using glue::REL_EXT;

// Message-gate tail of every row count (bit-identical to the deleted per-row glue::mg_fused_kernel): f = gd . forget[h],
// g = gd . gain[h] (64-term fp32 fma, j ascending), out = (1 - tanh f) hid + tanh g msg. Thread = one column h with its 2 x 64 controller weights in
// registers; block = 128 columns x kRows rows. Grid (H / 128, ceil(T / kRows), G), block 128.
constexpr int kMgRows = 32;
__global__ void __launch_bounds__(128) mg_tail_rows_kernel(glue::MgTail m, int T, int ldh, int ldm, int ldo) {
  const int z = blockIdx.z;
  const bf16* gd = m.gd[z];
  const float *FT = m.FT[z], *GT = m.GT[z], *hid = m.hid[z], *msg = m.msg[z];
  float* out = m.out[z];
  __shared__ __align__(16) float g[kMgRows][64];
  const int t0 = blockIdx.y * kMgRows, nr = min(kMgRows, T - t0);
  for (int i = threadIdx.x; i < nr * 64; i += blockDim.x) g[i / 64][i % 64] = __bfloat162float(gd[(size_t)t0 * 64 + i]);
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  float fw[64], gw[64];
#pragma unroll
  for (int j = 0; j < 64; ++j) {
    fw[j] = FT[(size_t)j * H + h];
    gw[j] = GT[(size_t)j * H + h];
  }
  __syncthreads();
  // two rows per step (four independent fma chains), the gate inputs read as float4 (one shared load per 8 fma for a
  // row instead of one per 2): issue slots go to the fma pipe; each chain is still j ascending from 0
  auto blend = [&](int t, float f, float a) {
    f = tanhf(f);
    a = tanhf(a);
    out[(size_t)t * ldo + h] = __fadd_rn(__fmul_rn(__fadd_rn(1.0f, -f), hid[(size_t)t * ldh + h]),
                                         __fmul_rn(a, msg[(size_t)t * ldm + h]));
  };
  int r = 0;
  for (; r + 1 < nr; r += 2) {
    float f0 = 0.f, a0 = 0.f, f1 = 0.f, a1 = 0.f;
#pragma unroll
    for (int q = 0; q < 16; ++q) {
      const float4 u = *reinterpret_cast<const float4*>(&g[r][4 * q]);
      const float4 w = *reinterpret_cast<const float4*>(&g[r + 1][4 * q]);
      const float ux[4] = {u.x, u.y, u.z, u.w}, wx[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int j = 4 * q + k;
        f0 = __fmaf_rn(ux[k], fw[j], f0);
        a0 = __fmaf_rn(ux[k], gw[j], a0);
        f1 = __fmaf_rn(wx[k], fw[j], f1);
        a1 = __fmaf_rn(wx[k], gw[j], a1);
      }
    }
    blend(t0 + r, f0, a0);
    blend(t0 + r + 1, f1, a1);
  }
  if (r < nr) {
    float f = 0.f, a = 0.f;
#pragma unroll
    for (int q = 0; q < 16; ++q) {
      const float4 u = *reinterpret_cast<const float4*>(&g[r][4 * q]);
      const float ux[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        f = __fmaf_rn(ux[k], fw[4 * q + k], f);
        a = __fmaf_rn(ux[k], gw[4 * q + k], a);
      }
    }
    blend(t0 + r, f, a);
  }
}

// Key-aligned bias table for prefill_attn straight from rel (bf16 [T][24][16]) and the layer's rel_proj (bf16
// [16][1024]): relk[t][h][col] = bf16(sum_j rel[t][h][j] * proj[j][d]) for d = pos - ((pos & ~63) - 1024 + col) in
// [0, 1024), else 0. The dot is glue::qkv_post_kernel's rel-logit arithmetic (fma chain from 0, j ascending, operands
// = the bf16 rel the q-post wrote, which equals its fp32 rs), so the table equals build_relk(rel_logits) bit for bit.
// With rel_logits != nullptr the per-distance logits [t][h][1024] are written too (decode segments of a mixed call).
// Thread = 4 distances d with their 4 x 16 proj values in registers, looping over kRelRows rows of one head.
// Grid (ceil(T / kRelRows), 24), block 256.
constexpr int kRelRows = 32;
__global__ void __launch_bounds__(256) relk_direct_kernel(const bf16* rel, const bf16* proj, const int* row_pos, int T,
                                                          bf16* relk, bf16* rel_logits) {
  __shared__ float rs[kRelRows][DREL];
  const int h = blockIdx.y, t0 = blockIdx.x * kRelRows, nr = min(kRelRows, T - t0);
  for (int i = threadIdx.x; i < nr * DREL; i += blockDim.x)
    rs[i / DREL][i % DREL] = __bfloat162float(rel[((size_t)(t0 + i / DREL) * NH + h) * DREL + i % DREL]);
  float pw[4][DREL];
#pragma unroll
  for (int k = 0; k < 4; ++k)
#pragma unroll
    for (int j = 0; j < DREL; ++j) pw[k][j] = __bfloat162float(proj[j * REL_EXT + threadIdx.x + k * 256]);
  __syncthreads();
  const bf16 zero = __float2bfloat16_rn(0.f);
  for (int r = 0; r < nr; ++r) {
    const int t = t0 + r, pos = row_pos[t], off = (pos & 63) + REL_EXT;   // col = off - d
    bf16* dst = relk + ((size_t)t * NH + h) * (REL_EXT + 64);
    bf16* dl = rel_logits ? rel_logits + ((size_t)t * NH + h) * REL_EXT : nullptr;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int d = threadIdx.x + k * 256;
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < DREL; ++j) acc = __fmaf_rn(rs[r][j], pw[k][j], acc);
      const bf16 o = __float2bfloat16_rn(acc);
      dst[off - d] = o;
      if (dl) dl[d] = o;
    }
    if (threadIdx.x < 64) {   // the 64 columns outside distance [0, 1024): col <= pos & 63 or col > off
      const int col = threadIdx.x <= (pos & 63) ? threadIdx.x : threadIdx.x + REL_EXT;
      dst[col] = zero;
    }
  }
}

// Routing layout for prefill_moe at prefill row counts, the same outputs as moedev::moe_build_kernel (perm with rows
// ascending per expert, a_row, out_row, gamma, item list with experts by count descending, ties to the lower expert)
// from three launches:
//   count:   grid nb = ceil(R / kBuildRows), per block the expert counts of its flat rows -> cntb [nb][128]
//   plan:    one CTA of 128: warp-parallel scan128 for expert offsets and item offsets, per-block bases, rank, items
//   scatter: grid nb, thread e walks its block's rows in ascending order and writes expert e's entries from its base
constexpr int kBuildRows = 128;
__global__ void __launch_bounds__(128) moe_count_kernel(const int* idx, int R, int* cntb) {
  __shared__ int cnt[pmoe::E];
  const int tid = threadIdx.x;
  cnt[tid] = 0;
  __syncthreads();
  const int r = blockIdx.x * kBuildRows + tid;
  if (r < R) atomicAdd(&cnt[idx[r]], 1);
  __syncthreads();
  cntb[blockIdx.x * pmoe::E + tid] = cnt[tid];
}
__global__ void __launch_bounds__(pmoe::E) moe_plan_kernel(const int* cntb, int nb, int* baseb, pmoe::Item* items,
                                                          int* n_items) {
  __shared__ int cnt[pmoe::E], eoff[pmoe::E + 1], order[pmoe::E], tiles[pmoe::E], ioff[pmoe::E + 1];
  const int e = threadIdx.x;
  int c = 0;
  for (int b = 0; b < nb; ++b) c += cntb[b * pmoe::E + e];
  cnt[e] = c;
  __syncthreads();
  moedev::scan128(cnt, eoff);
  int rk = 0;   // stable descending rank by count
  for (int f = 0; f < pmoe::E; ++f) rk += cnt[f] > c || (cnt[f] == c && f < e);
  order[rk] = e;
  __syncthreads();
  int base = eoff[e];
  for (int b = 0; b < nb; ++b) {
    baseb[b * pmoe::E + e] = base;
    base += cntb[b * pmoe::E + e];
  }
  const int x = order[e], cx = cnt[x], nt = (cx + pmoe::BN - 1) / pmoe::BN;
  tiles[e] = moedev::FT * nt;
  __syncthreads();
  moedev::scan128(tiles, ioff);
  __syncthreads();
  if (e == 0) *n_items = ioff[pmoe::E];
  int o = ioff[e];
  for (int ft = 0; ft < moedev::FT; ++ft)
    for (int t = 0; t < nt; ++t) items[o++] = {x, ft, eoff[x] + t * pmoe::BN, min(pmoe::BN, cx - t * pmoe::BN)};
}
__global__ void __launch_bounds__(pmoe::E) moe_scatter_kernel(const int* idx, const float* wts, int R, const int* baseb,
                                                             int* a13, int* a2, int* orow, float* gamma) {
  __shared__ int si[kBuildRows];
  __shared__ float sw[kBuildRows];
  const int r0 = blockIdx.x * kBuildRows, n = min(R - r0, kBuildRows), e = threadIdx.x;
  if (e < n) {
    si[e] = idx[r0 + e];
    sw[e] = wts[r0 + e];
  }
  __syncthreads();
  int p = baseb[blockIdx.x * pmoe::E + e];
  for (int i = 0; i < n; ++i)
    if (si[i] == e) {
      const int r = r0 + i;
      a13[p] = r / pmoe::TOPK;
      a2[p] = p;
      orow[p] = r;
      gamma[p] = sw[i];
      ++p;
    }
}

// q / rel post and key / value conv + KV write of prefill calls (glue::qkv_post_kernel without the rel logits, which
// relk_direct_kernel produces): one warp per (row, head) instead of one 128-thread block. Lane l holds the 4
// consecutive dims 4 l .. 4 l + 3 (8-byte loads and stores), i.e. virtual threads d = 4 l + e of the original 128-thread
// block; bsum4c reproduces its block_sum exactly: each virtual warp's xor tree (offsets 16, 8, 4 = lanes l ^ 4, l ^ 2,
// l ^ 1; offsets 2, 1 = e ^ 2, e ^ 1, in that order), then the 4 virtual-warp results added from 0 in warp order
// (virtual warp w = lanes 8 w .. 8 w + 7). Per element the arithmetic is the original's: bit-identical.
// Grid (T, 4), block 256: warp w of block y takes head y * 8 + w (heads 0-23 q, 24-31 kv).
template <int NV>
__device__ __forceinline__ void bsum4c(float (&v)[4][NV]) {   // v[e][i] of virtual thread 4 lane + e
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    float a[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) a[e] = v[e][i];
#pragma unroll
    for (int o = 4; o > 0; o >>= 1)
#pragma unroll
      for (int e = 0; e < 4; ++e) a[e] = __fadd_rn(a[e], __shfl_xor_sync(0xffffffffu, a[e], o));
    float b[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) b[e] = __fadd_rn(a[e], a[e ^ 2]);
    const float w = __fadd_rn(b[0], b[1]);   // virtual lane 0 of virtual warp lane / 8
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) s = __fadd_rn(s, __shfl_sync(0xffffffffu, w, 8 * k));
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e][i] = s;
  }
}
__device__ __forceinline__ void ld4(const bf16* p, float (&v)[4]) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
  const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
  v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}
__device__ __forceinline__ void ld4(const float* p, float (&v)[4]) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
}
__device__ __forceinline__ void st4(bf16* p, const float (&v)[4]) {
  uint2 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
  h[0] = __floats2bfloat162_rn(v[0], v[1]);
  h[1] = __floats2bfloat162_rn(v[2], v[3]);
  *reinterpret_cast<uint2*>(p) = u;
}
__global__ void __launch_bounds__(256) qkv_post_rows_kernel(const bf16* qproj, const float* q_scale, const bf16* rproj,
                                                            const float* r_scale, const int* row_pos, int full, bf16* q,
                                                            bf16* rel, const bf16* kx, const bf16* vx, const bf16* kst,
                                                            const bf16* vst, long long st_slot_stride, glue::Rows R,
                                                            const float* kw, const float* vw, const bf16* kdyn,
                                                            const bf16* vdyn, float eps, bf16* kpool, bf16* vpool,
                                                            const int* pt, int pt_stride) {
  using glue::ld;
  using glue::rr16;
  const int t = blockIdx.x, hh = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  constexpr int HD = glue::HD;
  const int d0 = 4 * lane;
  if (hh < NH) {
    float y[4], r[4] = {0.f, 0.f, 0.f, 0.f}, v[4][2], a[4], b[4];
    ld4(qproj + ((size_t)t * NH + hh) * HD + d0, a);
    ld4(q_scale + hh * HD + d0, b);
#pragma unroll
    for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(a[e], b[e]);
    if (d0 < DREL) {
      ld4(rproj + ((size_t)t * NH + hh) * DREL + d0, a);
      ld4(r_scale + hh * DREL + d0, b);
#pragma unroll
      for (int e = 0; e < 4; ++e) r[e] = __fmul_rn(a[e], b[e]);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[e][0] = __fmul_rn(y[e], y[e]);
      v[e][1] = __fmul_rn(r[e], r[e]);
    }
    bsum4c<2>(v);
    auto clipf = [](float s2, float thr) {
      const float norm = s2 > 0.f ? __fmul_rn(s2, rsqrtf(s2)) : 0.f;
      return fminf(__fdiv_rn(thr, __fadd_rn(norm, 1e-6f)), 1.0f);
    };
    const float cq = clipf(v[0][0], 22.627416997969522f), cr = clipf(v[0][1], 8.0f);
    const float tau = full ? glue::tau_of(row_pos[t]) : 1.f;
    float qo[4], ro[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      qo[e] = rr16(__fmul_rn(y[e], cq));
      ro[e] = rr16(__fmul_rn(r[e], cr));
      if (full) {
        qo[e] = rr16(__fmul_rn(qo[e], tau));
        ro[e] = rr16(__fmul_rn(ro[e], tau));
      }
    }
    st4(q + ((size_t)t * NH + hh) * HD + d0, qo);
    if (d0 < DREL) st4(rel + ((size_t)t * NH + hh) * DREL + d0, ro);
    return;
  }
  const int g = hh - NH;
  const int s = R.seg[t], r0 = R.row0[s], i = t - r0;
  const bf16* ks = kst + (size_t)R.slot[s] * st_slot_stride;
  const bf16* vs = vst + (size_t)R.slot[s] * st_slot_stride;
  const int c0 = g * HD + d0;
  auto ext = [&](const bf16* x, const bf16* st, int j, float (&o)[4]) {
    const int e = i + j;
    if (e < 3) ld4(st + (size_t)e * glue::SLD + c0, o);
    else ld4(x + (size_t)(t + j - 3) * 1024 + c0, o);
  };
  auto conv = [&](const float* w, const bf16* x, const bf16* st, const bf16* dy, float (&out)[4]) {
    float xw[4][4], xt[4];
    ld4(x + (size_t)t * 1024 + c0, xt);
    if (!full) {
      float wv[4][4];
#pragma unroll
      for (int e = 0; e < 4; ++e) ld4(w + (size_t)(c0 + e) * 4, wv[e]);
#pragma unroll
      for (int j = 0; j < 3; ++j) ext(x, st, j, xw[j]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float acc = __fmul_rn(wv[e][0], xw[0][e]);
        acc = __fadd_rn(acc, __fmul_rn(wv[e][1], xw[1][e]));
        acc = __fadd_rn(acc, __fmul_rn(wv[e][2], xw[2][e]));
        acc = __fadd_rn(acc, __fmul_rn(wv[e][3], xt[e]));
        out[e] = __fmul_rn(__fadd_rn(acc, xt[e]), (float)(1.0 / 2.23606797749979));
      }
    } else {
#pragma unroll
      for (int j = 0; j < 3; ++j) ext(x, st, j, xw[j]);
#pragma unroll
      for (int e = 0; e < 4; ++e) xw[3][e] = xt[e];
      float acc[4];
#pragma unroll
      for (int lag = 0; lag < 4; ++lag) {
        float sw[4];
        ld4(w + lag * 1024 + c0, sw);
        const float dv = ld(dy + (size_t)t * 32 + g * 4 + lag);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float wl = __fadd_rn(sw[e], dv);
          acc[e] = lag == 0 ? __fmul_rn(wl, xw[3 - lag][e]) : __fmaf_rn(wl, xw[3 - lag][e], acc[e]);
        }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) out[e] = __fmul_rn(__fadd_rn(acc[e], xt[e]), (float)(1.0 / 3.0));
    }
  };
  float ck[4], cv[4], v[4][2];
  conv(kw, kx, ks, kdyn, ck);
  conv(vw, vx, vs, vdyn, cv);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    ck[e] = rr16(ck[e]);
    cv[e] = rr16(cv[e]);
    v[e][0] = __fmul_rn(ck[e], ck[e]);
    v[e][1] = __fmul_rn(cv[e], cv[e]);
  }
  bsum4c<2>(v);
  const float norm = v[0][0] > 0.f ? __fmul_rn(v[0][0], rsqrtf(v[0][0])) : 0.f;
  const float f = fminf(__fdiv_rn(22.627416997969522f, __fadd_rn(norm, 1e-6f)), 1.0f);
  const float rv = glue::rstd_of(v[0][1], HD, eps);
  const int pos = R.pos[t];
  const int page = pt[(size_t)s * pt_stride + (pos >> 10)];
  float ko[4], vo[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    ko[e] = __fmul_rn(ck[e], f);
    vo[e] = __fmul_rn(cv[e], rv);
  }
  const size_t off = (((size_t)page * glue::KVH + g) * 1024 + (pos & 1023)) * HD + d0;
  st4(kpool + off, ko);
  st4(vpool + off, vo);
}

__device__ __forceinline__ void st4_e4m3(uint8_t* p, const float (&v)[4]) {
  const uint32_t b0 = glue::f2e4m3(glue::rr16(v[0]));
  const uint32_t b1 = glue::f2e4m3(glue::rr16(v[1]));
  const uint32_t b2 = glue::f2e4m3(glue::rr16(v[2]));
  const uint32_t b3 = glue::f2e4m3(glue::rr16(v[3]));
  *reinterpret_cast<uint32_t*>(p) = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}
// Same as qkv_post_rows_kernel, writing global KV pages in FP8 E4M3 (unit scale; bf16-rounded before E4M3 so bits
// match glue::qkv_post_fp8_kernel and glue::kv_write_fp8_kernel).
__global__ void __launch_bounds__(256) qkv_post_rows_fp8_kernel(
    const bf16* qproj, const float* q_scale, const bf16* rproj, const float* r_scale, const int* row_pos, int full,
    bf16* q, bf16* rel, const bf16* kx, const bf16* vx, const bf16* kst, const bf16* vst, long long st_slot_stride,
    glue::Rows R, const float* kw, const float* vw, const bf16* kdyn, const bf16* vdyn, float eps, uint8_t* kpool,
    uint8_t* vpool, const int* pt, int pt_stride) {
  using glue::ld;
  using glue::rr16;
  const int t = blockIdx.x, hh = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  constexpr int HD = glue::HD;
  const int d0 = 4 * lane;
  if (hh < NH) {
    float y[4], r[4] = {0.f, 0.f, 0.f, 0.f}, v[4][2], a[4], b[4];
    ld4(qproj + ((size_t)t * NH + hh) * HD + d0, a);
    ld4(q_scale + hh * HD + d0, b);
#pragma unroll
    for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(a[e], b[e]);
    if (d0 < DREL) {
      ld4(rproj + ((size_t)t * NH + hh) * DREL + d0, a);
      ld4(r_scale + hh * DREL + d0, b);
#pragma unroll
      for (int e = 0; e < 4; ++e) r[e] = __fmul_rn(a[e], b[e]);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[e][0] = __fmul_rn(y[e], y[e]);
      v[e][1] = __fmul_rn(r[e], r[e]);
    }
    bsum4c<2>(v);
    auto clipf = [](float s2, float thr) {
      const float norm = s2 > 0.f ? __fmul_rn(s2, rsqrtf(s2)) : 0.f;
      return fminf(__fdiv_rn(thr, __fadd_rn(norm, 1e-6f)), 1.0f);
    };
    const float cq = clipf(v[0][0], 22.627416997969522f), cr = clipf(v[0][1], 8.0f);
    const float tau = full ? glue::tau_of(row_pos[t]) : 1.f;
    float qo[4], ro[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      qo[e] = rr16(__fmul_rn(y[e], cq));
      ro[e] = rr16(__fmul_rn(r[e], cr));
      if (full) {
        qo[e] = rr16(__fmul_rn(qo[e], tau));
        ro[e] = rr16(__fmul_rn(ro[e], tau));
      }
    }
    st4(q + ((size_t)t * NH + hh) * HD + d0, qo);
    if (d0 < DREL) st4(rel + ((size_t)t * NH + hh) * DREL + d0, ro);
    return;
  }
  const int g = hh - NH;
  const int s = R.seg[t], r0 = R.row0[s], i = t - r0;
  const bf16* ks = kst + (size_t)R.slot[s] * st_slot_stride;
  const bf16* vs = vst + (size_t)R.slot[s] * st_slot_stride;
  const int c0 = g * HD + d0;
  auto ext = [&](const bf16* x, const bf16* st, int j, float (&o)[4]) {
    const int e = i + j;
    if (e < 3) ld4(st + (size_t)e * glue::SLD + c0, o);
    else ld4(x + (size_t)(t + j - 3) * 1024 + c0, o);
  };
  auto conv = [&](const float* w, const bf16* x, const bf16* st, const bf16* dy, float (&out)[4]) {
    float xw[4][4], xt[4];
    ld4(x + (size_t)t * 1024 + c0, xt);
    if (!full) {
      float wv[4][4];
#pragma unroll
      for (int e = 0; e < 4; ++e) ld4(w + (size_t)(c0 + e) * 4, wv[e]);
#pragma unroll
      for (int j = 0; j < 3; ++j) ext(x, st, j, xw[j]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float acc = __fmul_rn(wv[e][0], xw[0][e]);
        acc = __fadd_rn(acc, __fmul_rn(wv[e][1], xw[1][e]));
        acc = __fadd_rn(acc, __fmul_rn(wv[e][2], xw[2][e]));
        acc = __fadd_rn(acc, __fmul_rn(wv[e][3], xt[e]));
        out[e] = __fmul_rn(__fadd_rn(acc, xt[e]), (float)(1.0 / 2.23606797749979));
      }
    } else {
#pragma unroll
      for (int j = 0; j < 3; ++j) ext(x, st, j, xw[j]);
#pragma unroll
      for (int e = 0; e < 4; ++e) xw[3][e] = xt[e];
      float acc[4];
#pragma unroll
      for (int lag = 0; lag < 4; ++lag) {
        float sw[4];
        ld4(w + lag * 1024 + c0, sw);
        const float dv = ld(dy + (size_t)t * 32 + g * 4 + lag);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float wl = __fadd_rn(sw[e], dv);
          acc[e] = lag == 0 ? __fmul_rn(wl, xw[3 - lag][e]) : __fmaf_rn(wl, xw[3 - lag][e], acc[e]);
        }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) out[e] = __fmul_rn(__fadd_rn(acc[e], xt[e]), (float)(1.0 / 3.0));
    }
  };
  float ck[4], cv[4], v[4][2];
  conv(kw, kx, ks, kdyn, ck);
  conv(vw, vx, vs, vdyn, cv);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    ck[e] = rr16(ck[e]);
    cv[e] = rr16(cv[e]);
    v[e][0] = __fmul_rn(ck[e], ck[e]);
    v[e][1] = __fmul_rn(cv[e], cv[e]);
  }
  bsum4c<2>(v);
  const float norm = v[0][0] > 0.f ? __fmul_rn(v[0][0], rsqrtf(v[0][0])) : 0.f;
  const float f = fminf(__fdiv_rn(22.627416997969522f, __fadd_rn(norm, 1e-6f)), 1.0f);
  const float rv = glue::rstd_of(v[0][1], HD, eps);
  const int pos = R.pos[t];
  const int page = pt[(size_t)s * pt_stride + (pos >> 10)];
  float ko[4], vo[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    ko[e] = __fmul_rn(ck[e], f);
    vo[e] = __fmul_rn(cv[e], rv);
  }
  const size_t off = (((size_t)page * glue::KVH + g) * 1024 + (pos & 1023)) * HD + d0;
  st4_e4m3(kpool + off, ko);
  st4_e4m3(vpool + off, vo);
}

// Hidden short convs of prefill calls (glue::attn_conv_partial_kernel without its unused y output, and
// glue::hidden_conv_kernel mode 1): thread = 8 consecutive channels (16-byte loads and stores) x kConvRows consecutive
// rows, the conv window's three earlier x rows kept in registers, so every x row is read once instead of four times and
// no 64-bit index division runs per element. Per element the arithmetic is glue::static_conv's (product, three
// mul + add, + x, x fp32(5^-0.5)) and the residual update of the kernel it replaces: bit-identical.
//   MODE 0 (attention conv): yy = bf16r(conv(x)); block start: blk = hin, partial = yy; else partial = hin + yy
//   MODE 1 (MLP conv):       out = bf16(partial + bf16r(conv(x)))
// Grid (H / 8 / 96, ceil(T / kConvRows)), block 96.
constexpr int kConvRows = 8, kConvThreads = 96;
static_assert(H / 8 == 3 * kConvThreads, "conv rows: 288 channel groups");
__device__ __forceinline__ void ld8(const bf16* p, float (&v)[8]) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f = __bfloat1622float2(h[k]);
    v[2 * k] = f.x;
    v[2 * k + 1] = f.y;
  }
}
__device__ __forceinline__ void st8(bf16* p, const float (&v)[8]) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int k = 0; k < 4; ++k) h[k] = __floats2bfloat162_rn(v[2 * k], v[2 * k + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}
template <int MODE>
__global__ void __launch_bounds__(kConvThreads) conv_rows_kernel(const bf16* x, const float* w, const bf16* state,
                                                                 long long st_slot_stride, glue::Rows R, int T,
                                                                 const bf16* hin, int block_start, bf16* blk,
                                                                 float* partial32, bf16* out) {
  const int c0 = 8 * (blockIdx.x * kConvThreads + threadIdx.x);
  const int t0 = blockIdx.y * kConvRows, t1 = min(T, t0 + kConvRows);
  float wv[8][4];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float4 q = *reinterpret_cast<const float4*>(w + (size_t)(c0 + e) * 4);
    wv[e][0] = q.x; wv[e][1] = q.y; wv[e][2] = q.z; wv[e][3] = q.w;
  }
  float prev[3][8];   // x rows t - 3, t - 2, t - 1 (rows before 0 are never selected)
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int r = t0 - 3 + j;
    if (r >= 0) ld8(x + (size_t)r * H + c0, prev[j]);
    else {
#pragma unroll
      for (int e = 0; e < 8; ++e) prev[j][e] = 0.f;
    }
  }
  constexpr float kInv5 = (float)(1.0 / 2.23606797749979);
  for (int t = t0; t < t1; ++t) {
    float cur[8];
    ld8(x + (size_t)t * H + c0, cur);
    const int s = R.seg[t], i = t - R.row0[s];
    float ex[3][8];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      if (i + j < 3) ld8(state + (size_t)R.slot[s] * st_slot_stride + (size_t)(i + j) * glue::SLD + c0, ex[j]);
      else {
#pragma unroll
        for (int e = 0; e < 8; ++e) ex[j][e] = prev[j][e];
      }
    }
    float y[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float acc = __fmul_rn(wv[e][0], ex[0][e]);
      acc = __fadd_rn(acc, __fmul_rn(wv[e][1], ex[1][e]));
      acc = __fadd_rn(acc, __fmul_rn(wv[e][2], ex[2][e]));
      acc = __fadd_rn(acc, __fmul_rn(wv[e][3], cur[e]));
      y[e] = glue::rr16(__fmul_rn(__fadd_rn(acc, cur[e]), kInv5));
    }
    const size_t o = (size_t)t * H + c0;
    if constexpr (MODE == 0) {
      float p[8];
      if (block_start) {
        *reinterpret_cast<uint4*>(blk + o) = *reinterpret_cast<const uint4*>(hin + o);
#pragma unroll
        for (int e = 0; e < 8; ++e) p[e] = y[e];
      } else {
        float h[8];
        ld8(hin + o, h);
#pragma unroll
        for (int e = 0; e < 8; ++e) p[e] = __fadd_rn(h[e], y[e]);
      }
      *reinterpret_cast<float4*>(partial32 + o) = make_float4(p[0], p[1], p[2], p[3]);
      *reinterpret_cast<float4*>(partial32 + o + 4) = make_float4(p[4], p[5], p[6], p[7]);
    } else {
      const float4 a = *reinterpret_cast<const float4*>(partial32 + o);
      const float4 b = *reinterpret_cast<const float4*>(partial32 + o + 4);
      const float pa[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      float r[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) r[e] = __fadd_rn(pa[e], y[e]);
      st8(out + o, r);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      prev[0][e] = prev[1][e];
      prev[1][e] = prev[2][e];
      prev[2][e] = cur[e];
    }
  }
}

// ---------------------------------------------------------------------------------------------- AttnRes and norms, warp per row
// The 256-thread row kernels of glue.cuh (attn_res_kernel, attn_res_norm_kernel, rmsnorm_g_kernel, rmsnorm_kernel)
// run here with one warp per row and 16-byte loads: lane l holds the 8 consecutive elements 256 k + 8 l .. + 7 of
// slice k = 0..8, i.e. virtual threads j = 8 l + e (e = 0..7) of the 256-thread decomposition (virtual thread j owns
// elements j + 256 k). Every per-thread sum runs over k ascending as in the original; vblock_sum then reproduces
// glue::block_sum exactly: the xor tree of each virtual warp (offsets 16 and 8 = lanes l ^ 2 and l ^ 1, offsets 4, 2, 1
// = e ^ 4, e ^ 2, e ^ 1, in that order) and the sum of the 8 virtual-warp results from 0 in warp order (virtual warp w
// = lanes 4 w .. 4 w + 3). Products, fma and every rounding are the originals' (bit-identical outputs). No shared memory
// and no barrier; four rows per 128-thread block.
constexpr int kWarpRows = 4;
__device__ __forceinline__ float vblock_sum(float (&a)[8]) {
#pragma unroll
  for (int e = 0; e < 8; ++e) a[e] = __fadd_rn(a[e], __shfl_xor_sync(0xffffffffu, a[e], 2));
#pragma unroll
  for (int e = 0; e < 8; ++e) a[e] = __fadd_rn(a[e], __shfl_xor_sync(0xffffffffu, a[e], 1));
  float b[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) b[e] = __fadd_rn(a[e], a[e ^ 4]);
#pragma unroll
  for (int e = 0; e < 8; ++e) a[e] = __fadd_rn(b[e], b[e ^ 2]);
  const float w = __fadd_rn(a[0], a[1]);   // virtual lane 0 of virtual warp lane / 4
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) s = __fadd_rn(s, __shfl_sync(0xffffffffu, w, 4 * k));
  return s;
}
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) {
  const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void st8(float* p, const float (&v)[8]) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}
// AttnRes of one row, glue::attn_res_kernel's arithmetic (sources s.blk[0 .. nb) bf16 then the last source fp32 or
// bf16): zero shared memory (full occupancy) for Q = 1 and Q = 3, with k-outer Pass 2 for Q = 4 (msgs) so the 2 bf16
// sources are read ONCE from HBM (staged in 36 KB smem = 4 CTAs/SM = 16 warps/SM) instead of 5x, and Q = 3 Pass 2
// writes gin hidden-part directly from registers (eliminating 2x HBM reloads of out). Bit-identical.
__device__ __forceinline__ void ld8_bf16_smem(const uint4* base, int k, int lane, float (&x)[8]) {
  const uint4 u = base[k * 32 + lane];
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(h[j]);
    x[2 * j] = f.x;
    x[2 * j + 1] = f.y;
  }
}
template <int Q, class F>
__device__ __forceinline__ void ar_row(const glue::ResSrc& s, const float* queries, int t, int lane, float eps, F&& f) {
  const int N = s.nb + 1;
  const float qsc = (float)(1.0 / 48.0);
  const float invH = (float)(1.0 / 2304.0);
  auto x_at = [&](int n, int i, float (&x)[8]) {
    if (n < s.nb) ld8(s.blk[n] + (size_t)t * H + i, x);
    else if (s.last32) ld8(s.last32 + (size_t)t * s.ld_last + i, x);
    else ld8(s.last16 + (size_t)t * s.ld_last + i, x);
  };
  float rstd[5], lg[Q][5];
#pragma unroll
  for (int n = 0; n < 5; ++n) {
    if (n < N) {
      float ss[8], dq[Q][8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        ss[e] = 0.f;
#pragma unroll
        for (int q = 0; q < Q; ++q) dq[q][e] = 0.f;
      }
#pragma unroll
      for (int k = 0; k < 9; ++k) {
        const int i = 256 * k + 8 * lane;
        float x[8];
        x_at(n, i, x);
#pragma unroll
        for (int q = 0; q < Q; ++q) {
          float qv[8];
          ld8(queries + q * H + i, qv);
#pragma unroll
          for (int e = 0; e < 8; ++e) dq[q][e] = __fadd_rn(dq[q][e], __fmul_rn(x[e], __fmul_rn(qv[e], qsc)));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) ss[e] = __fadd_rn(ss[e], __fmul_rn(x[e], x[e]));
      }
      rstd[n] = rsqrtf(__fmaf_rn(vblock_sum(ss), invH, eps));
#pragma unroll
      for (int q = 0; q < Q; ++q) lg[q][n] = __fmul_rn(vblock_sum(dq[q]), rstd[n]);
    } else {
      rstd[n] = 0.f;
    }
  }
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    float mx = -INFINITY, p[5];
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) mx = fmaxf(mx, lg[q][n]);
    float sum = 0.f;
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) { p[n] = expf(__fadd_rn(lg[q][n], -mx)); sum = __fadd_rn(sum, p[n]); }
    const float inv = __fdiv_rn(1.0f, sum);
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) p[n] = __fmul_rn(p[n], rstd[n]);
    float res[9][8];
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const int i = 256 * k + 8 * lane;
      float acc[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
      for (int n = 0; n < 5; ++n)
        if (n < N) {
          float x[8];
          x_at(n, i, x);
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[e] = __fmaf_rn(p[n], x[e], acc[e]);
        }
#pragma unroll
      for (int e = 0; e < 8; ++e) res[k][e] = __fmul_rn(acc[e], inv);
    }
    f(q, res);
  }
}
// RMSNorm of a row held as res[9][8] (rmsnorm_kernel's arithmetic, weight w or none) -> bf16 o (ld H)
__device__ __forceinline__ void norm_out(const float (&res)[9][8], const float* w, bf16* o, int lane, float eps) {
  float u[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) u[e] = 0.f;
#pragma unroll
  for (int k = 0; k < 9; ++k)
#pragma unroll
    for (int e = 0; e < 8; ++e) u[e] = __fadd_rn(u[e], __fmul_rn(res[k][e], res[k][e]));
  const float r = glue::rstd_of(vblock_sum(u), H, eps);
#pragma unroll
  for (int k = 0; k < 9; ++k) {
    const int i = 256 * k + 8 * lane;
    float y[8], wv[8];
    if (w) ld8(w + i, wv);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      y[e] = __fmul_rn(res[k][e], r);
      if (w) y[e] = __fmul_rn(y[e], wv[e]);
    }
    st8(o + i, y);
  }
}
// = glue::attn_res_norm_kernel<Q> (bf16 outputs o[q] with weights w[q]); grid ceil(T / 4), block 128
template <int Q>
__global__ void __launch_bounds__(32 * kWarpRows) ar_norm_rows_kernel(glue::ResSrc s, const float* queries, const float* w0,
                                                                      const float* w1, const float* w2, bf16* o0,
                                                                      bf16* o1, bf16* o2, float eps, int T) {
  const int t = blockIdx.x * kWarpRows + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= T) return;
  const float* ws[3] = {w0, w1, w2};
  bf16* os[3] = {o0, o1, o2};
  ar_row<Q>(s, queries, t, lane, eps, [&](int q, const float (&res)[9][8]) {
    norm_out(res, ws[q], os[q] + (size_t)t * H, lane, eps);
  });
}
// = glue::attn_res_kernel<Q, G> (fp32 out [T][Q][H]; with G the loop-gate inputs of mg_in_kernel's arithmetic)
template <int Q, int G>
__global__ void __launch_bounds__(32 * kWarpRows) ar_rows_kernel(glue::ResSrc s, const float* queries, float* out,
                                                                 float eps, glue::GateIn gi, int T) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t = blockIdx.x * kWarpRows + warp;
  if (t >= T) return;
  if constexpr (Q == 4 && G == 0) {
    // msgs fast path: N = 2 bf16 sources staged once in 36 KB static smem (4 CTAs/SM = 16 warps/SM) -> 1x HBM read
    // instead of 5x, k-outer Pass 2 writes all 4 outputs directly with zero register pressure.
    __shared__ __align__(16) uint4 msmem[kWarpRows][2][9 * 32];
    const float qsc = (float)(1.0 / 48.0);
    const float invH = (float)(1.0 / 2304.0);
    float rstd[2], lg[4][2];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      float ss[8], dq[4][8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        ss[e] = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) dq[q][e] = 0.f;
      }
      const bf16* src = n == 0 ? s.blk[0] + (size_t)t * H : s.last16 + (size_t)t * s.ld_last;
#pragma unroll 1
      for (int k = 0; k < 9; ++k) {
        const int i = 256 * k + 8 * lane;
        const uint4 u = *reinterpret_cast<const uint4*>(src + i);
        msmem[warp][n][k * 32 + lane] = u;
        float x[8];
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 fv = __bfloat1622float2(h[j]);
          x[2 * j] = fv.x; x[2 * j + 1] = fv.y;
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float qv[8];
          ld8(queries + q * H + i, qv);
#pragma unroll
          for (int e = 0; e < 8; ++e) dq[q][e] = __fadd_rn(dq[q][e], __fmul_rn(x[e], __fmul_rn(qv[e], qsc)));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) ss[e] = __fadd_rn(ss[e], __fmul_rn(x[e], x[e]));
      }
      rstd[n] = rsqrtf(__fmaf_rn(vblock_sum(ss), invH, eps));
#pragma unroll
      for (int q = 0; q < 4; ++q) lg[q][n] = __fmul_rn(vblock_sum(dq[q]), rstd[n]);
    }
    float p[4][2], inv[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float mx = fmaxf(lg[q][0], lg[q][1]);
      const float p0 = expf(__fadd_rn(lg[q][0], -mx));
      const float p1 = expf(__fadd_rn(lg[q][1], -mx));
      inv[q] = __fdiv_rn(1.0f, __fadd_rn(p0, p1));
      p[q][0] = __fmul_rn(p0, rstd[0]);
      p[q][1] = __fmul_rn(p1, rstd[1]);
    }
#pragma unroll 1
    for (int k = 0; k < 9; ++k) {
      const int i = 256 * k + 8 * lane;
      float x0[8], x1[8];
      ld8_bf16_smem(msmem[warp][0], k, lane, x0);
      ld8_bf16_smem(msmem[warp][1], k, lane, x1);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float y[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float acc = __fmaf_rn(p[q][1], x1[e], __fmaf_rn(p[q][0], x0[e], 0.f));
          y[e] = __fmul_rn(acc, inv[q]);
        }
        st8(out + ((size_t)t * 4 + q) * H + i, y);
      }
    }
    return;
  }
  uint4 h_bf16[Q == 1 && G > 0 ? 9 : 1];
  ar_row<Q>(s, queries, t, lane, eps, [&](int q, const float (&res)[9][8]) {
#pragma unroll
    for (int k = 0; k < 9; ++k) st8(out + ((size_t)t * Q + q) * H + 256 * k + 8 * lane, res[k]);
    if constexpr (G > 0) {
      if constexpr (Q == 1) {
        float ua[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) ua[e] = 0.f;
#pragma unroll
        for (int k = 0; k < 9; ++k)
#pragma unroll
          for (int e = 0; e < 8; ++e) ua[e] = __fadd_rn(ua[e], __fmul_rn(res[k][e], res[k][e]));
        const float ra1 = glue::rstd_of(vblock_sum(ua), H, eps);
#pragma unroll
        for (int k = 0; k < 9; ++k) {
          float ya[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) ya[e] = __fmul_rn(res[k][e], ra1);
          uint4 u;
          __nv_bfloat162* hv = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
          for (int j = 0; j < 4; ++j) hv[j] = __floats2bfloat162_rn(ya[2 * j], ya[2 * j + 1]);
          h_bf16[k] = u;
        }
      } else {
        if (gi.qsel && q < G) {
          float ua[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) ua[e] = 0.f;
#pragma unroll
          for (int k = 0; k < 9; ++k)
#pragma unroll
            for (int e = 0; e < 8; ++e) ua[e] = __fadd_rn(ua[e], __fmul_rn(res[k][e], res[k][e]));
          const float ra = glue::rstd_of(vblock_sum(ua), H, eps);
          bf16* g = gi.g + q * gi.gs + (size_t)t * 2 * H;
#pragma unroll
          for (int k = 0; k < 9; ++k) {
            float ya[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) ya[e] = __fmul_rn(res[k][e], ra);
            st8(g + 256 * k + 8 * lane, ya);
          }
        }
      }
    }
  });
  if constexpr (G > 0) {
#pragma unroll
    for (int y = 0; y < G; ++y) {
      const float* b = gi.msg + (size_t)t * gi.ldm + y * H;
      float ub[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) ub[e] = 0.f;
#pragma unroll
      for (int k = 0; k < 9; ++k) {
        float bv[8];
        ld8(b + 256 * k + 8 * lane, bv);
#pragma unroll
        for (int e = 0; e < 8; ++e) ub[e] = __fadd_rn(ub[e], __fmul_rn(bv[e], bv[e]));
      }
      const float rb = glue::rstd_of(vblock_sum(ub), H, eps);
      bf16* g = gi.g + y * gi.gs + (size_t)t * 2 * H;
#pragma unroll
      for (int k = 0; k < 9; ++k) {
        const int i = 256 * k + 8 * lane;
        if constexpr (Q == 1) {
          *reinterpret_cast<uint4*>(g + i) = h_bf16[k];
        }
        float bv[8], yb[8];
        ld8(b + i, bv);
#pragma unroll
        for (int e = 0; e < 8; ++e) yb[e] = __fmul_rn(bv[e], rb);
        st8(g + H + i, yb);
      }
    }
  }
}
// = glue::rmsnorm_g_kernel (G fp32 sources, row stride ldi, -> bf16 ld ldo) and rmsnorm_kernel<float> (G = 1)
__global__ void __launch_bounds__(32 * kWarpRows) norm_rows_kernel(glue::NormG a, int ldi, int ldo, float eps, int T) {
  const int t = blockIdx.x * kWarpRows + (threadIdx.x >> 5), lane = threadIdx.x & 31, y = blockIdx.y;
  if (t >= T) return;
  float res[9][8];
#pragma unroll
  for (int k = 0; k < 9; ++k) ld8(a.in[y] + (size_t)t * ldi + 256 * k + 8 * lane, res[k]);
  norm_out(res, a.w[y], a.out[y] + (size_t)t * ldo, lane, eps);
}

// Message-gate tail on tensor cores (BIT-CHANGING candidate): f = gd . forget[h], g = gd . gain[h] by bf16 mma.sync
// m16n8k16 with fp32 accumulation (the controller weights are bf16 in the checkpoint, so the products are exact; only
// the summation order differs from the fp32 fma chain), blend in the epilogue: out = (1 - tanh f) hid + tanh g msg.
// W: bf16 [2][H][64] per gate (forget rows, then gain rows; k contiguous). Block 128 = 4 warps over 128 columns
// (32 each), 64 rows; grid (H / 128, ceil(T / 64), G).
struct MgTc {
  const bf16* gd[4];
  const bf16* W[4];
  const float* hid[4];
  const float* msg[4];
  float* out[4];
};
constexpr int kTcRows = 32, kTcLd = 72;   // 32 rows x 128 cols: halves accum regs (64 floats -> 3-4 CTAs/SM), coalesced float4 out
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__global__ void __launch_bounds__(128) mg_tail_tc_kernel(MgTc m, int T, int ldh, int ldm, int ldo) {
  const int z = blockIdx.z, n0 = blockIdx.x * 128, t0 = blockIdx.y * kTcRows;
  __shared__ __align__(16) bf16 sa[kTcRows * kTcLd];
  __shared__ __align__(16) bf16 sb[2][128 * kTcLd];
  const int tid = threadIdx.x;
  for (int i = tid; i < kTcRows * 8; i += 128) {   // 16-byte chunks: row i / 8, chunk i % 8
    const int r = i / 8, c = i % 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (t0 + r < T) v = *reinterpret_cast<const uint4*>(m.gd[z] + (size_t)(t0 + r) * 64 + 8 * c);
    *reinterpret_cast<uint4*>(sa + r * kTcLd + 8 * c) = v;
  }
  for (int i = tid; i < 2 * 128 * 8; i += 128) {
    const int w = i / 1024, r = (i / 8) % 128, c = i % 8;
    *reinterpret_cast<uint4*>(sb[w] + r * kTcLd + 8 * c) =
        *reinterpret_cast<const uint4*>(m.W[z] + ((size_t)w * H + n0 + r) * 64 + 8 * c);
  }
  __syncthreads();
  const int warp = tid >> 5, lane = tid & 31, gq = lane >> 2, t4 = lane & 3;
  float acc[2][2][4][4];   // [f / g][m tile: 2][n tile: 4][frag: 4] = 64 floats
#pragma unroll
  for (int w = 0; w < 2; ++w)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[w][i][j][e] = 0.f;
  auto u32 = [](const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); };
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const int k0 = 16 * ks + 2 * t4;
    uint32_t a[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const bf16* r0 = sa + (16 * i + gq) * kTcLd, *r1 = r0 + 8 * kTcLd;
      a[i][0] = u32(r0 + k0); a[i][1] = u32(r1 + k0); a[i][2] = u32(r0 + k0 + 8); a[i][3] = u32(r1 + k0 + 8);
    }
#pragma unroll
    for (int w = 0; w < 2; ++w)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const bf16* bn = sb[w] + (32 * warp + 8 * j + gq) * kTcLd;
        const uint32_t b0 = u32(bn + k0), b1 = u32(bn + k0 + 8);
#pragma unroll
        for (int i = 0; i < 2; ++i) mma16816(acc[w][i][j], a[i], b0, b1);
      }
  }
  __syncthreads();
  // Reuse sb (36,864 B >= 32 * 128 * 4 = 16,384 B) for coalesced float4 hid/msg/out transfers
  float* stmp = reinterpret_cast<float*>(sb);
  const float *hid = m.hid[z], *msg = m.msg[z];
  float* out = m.out[z];
  const int nr = min(kTcRows, T - t0);
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int r = 16 * i + gq + 8 * hf;
      if (r >= nr) continue;
      const int t = t0 + r;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int lc = 32 * warp + 8 * j + 2 * t4;
        const int h = n0 + lc;
        const float2 hv = *reinterpret_cast<const float2*>(hid + (size_t)t * ldh + h);
        const float2 mv = *reinterpret_cast<const float2*>(msg + (size_t)t * ldm + h);
        float2 o;
        {
          const float f = tanhf(acc[0][i][j][2 * hf]), a = tanhf(acc[1][i][j][2 * hf]);
          o.x = __fadd_rn(__fmul_rn(__fadd_rn(1.0f, -f), hv.x), __fmul_rn(a, mv.x));
        }
        {
          const float f = tanhf(acc[0][i][j][2 * hf + 1]), a = tanhf(acc[1][i][j][2 * hf + 1]);
          o.y = __fadd_rn(__fmul_rn(__fadd_rn(1.0f, -f), hv.y), __fmul_rn(a, mv.y));
        }
        *reinterpret_cast<float2*>(stmp + r * 128 + lc) = o;
      }
    }
  __syncthreads();
  for (int idx = tid; idx < nr * 32; idx += 128) {
    const int r = idx >> 5, c = idx & 31;
    *reinterpret_cast<float4*>(out + (size_t)(t0 + r) * ldo + n0 + c * 4) =
        *reinterpret_cast<const float4*>(stmp + r * 128 + c * 4);
  }
}
// W of one gate from the fp32 transposed controller weights FT / GT [64][H] (exact bf16 values): grid 2 H, block 64
__global__ void mg_tc_weights_kernel(const float* FT, const float* GT, bf16* W) {
  const int n = blockIdx.x, k = threadIdx.x;
  const float w = n < H ? FT[(size_t)k * H + n] : GT[(size_t)k * H + n - H];
  W[(size_t)n * 64 + k] = __float2bfloat16_rn(w);
}
// Transposed relative projection [1024][16] bf16 from rel_proj [16][1024] bf16 (for in-kernel TC bias in prefill_attn):
// grid 1024, block 16.
__global__ void rel_proj_t_kernel(const bf16* rel_proj, bf16* rel_proj_t) {
  const int d = blockIdx.x, j = threadIdx.x;
  rel_proj_t[(size_t)d * DREL + j] = rel_proj[(size_t)j * REL_EXT + d];
}

// ---------------------------------------------------------------------------------------------- decode-shaped fast glue & fused TC kernels

// Decode relative logits: out [T*24, 1024] = rel [T*24, 16] @ proj [16, 1024], bit-identical to glue::qkv_post_kernel.
// Weight-stationary over 16 rows per CTA (4 warps x 4 rows), 128 cols per CTA (lane l owns 4 consecutive cols -> 8-byte ld4/st4).
// Grid (1024 / 128 = 8, ceil(T * 24 / 16)), block 128.
constexpr int kRelDecRows = 16;
__global__ void __launch_bounds__(128) rel_logits_decode_kernel(const bf16* __restrict__ rel,
                                                                const bf16* __restrict__ proj, int R,
                                                                bf16* __restrict__ out) {
  __shared__ float rs[kRelDecRows][DREL];
  const int d0 = blockIdx.x * 128 + (threadIdx.x & 31) * 4;
  const int r0 = blockIdx.y * kRelDecRows;
  const int nr = min(kRelDecRows, R - r0);
  const int tid = threadIdx.x, warp = tid >> 5;
  for (int i = tid; i < nr * DREL; i += 128)
    rs[i >> 4][i & 15] = __bfloat162float(rel[(size_t)(r0 + (i >> 4)) * DREL + (i & 15)]);
  float pw[DREL][4];
#pragma unroll
  for (int j = 0; j < DREL; ++j) ld4(proj + j * REL_EXT + d0, pw[j]);
  __syncthreads();
  const int rw0 = warp * 4;
  const int rw1 = min(nr, rw0 + 4);
  for (int r = rw0; r < rw1; ++r) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < DREL; ++j) {
      const float rv = rs[r][j];
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(rv, pw[j][e], acc[e]);
    }
    st4(out + (size_t)(r0 + r) * REL_EXT + d0, acc);
  }
}

// Single-launch fused decode Router GEMM + Top-8 selection (MT = 1, 2, 4 for M <= 16, 32, 64).
// Coalesced 16-byte uint4 smem tiling (128 K-cols/stage x 2 stages = 256 K-cols/CTA), exact 3-K-group reduction
// ([0..1024), [1024..2048), [2048..2304)) and row-count invariance.
// Grid (8, 9, 1) = 72 CTAs: bx = 16 experts, by = 9 K-slices of 256 cols (0..3 -> kg0, 4..7 -> kg1, 8 -> kg2).
// Block 256 = 8 warps (2 N-warps x 4 K-warps). Last CTA (atomicAdd(cnt, 1) == 71) runs 8-warp route_select with __ldcg.
constexpr int kRtLd = 136;   // 128 bf16 + 8 bf16 pad (272 B) for conflict-free 32-bit fragment loads
template <int MT>
__global__ void __launch_bounds__(256) router_decode_kernel(const bf16* __restrict__ x, const bf16* __restrict__ wr,
                                                            const float* __restrict__ bias, float* __restrict__ rlog,
                                                            int* __restrict__ cnt, int M, int* __restrict__ idx,
                                                            float* __restrict__ wts) {
  union {
    struct {
      __align__(16) bf16 sx[MT * 16 * kRtLd];
      __align__(16) bf16 sw[16 * kRtLd];
    } tile;
    float sp[4][MT * 16][16];
  } __shared__ smem;
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nw_n = warp >> 2, nw_k = warp & 3;
  const int gq = lane >> 2, t4 = lane & 3;
  const int e0 = bx * 16;
  const int k_cta = by * 256;
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
    acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
  auto u32 = [](const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); };
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const int k_st = k_cta + st * 128;
    // 16 x 16 uint4 = 256 uint4s -> 1 coalesced 16-byte load/thread for W
    {
      const int r = tid >> 4, c = tid & 15;
      *reinterpret_cast<uint4*>(smem.tile.sw + r * kRtLd + c * 8) =
          *reinterpret_cast<const uint4*>(wr + (size_t)(e0 + r) * H + k_st + c * 8);
    }
    // MT*16 x 16 uint4 -> 1..4 coalesced 16-byte loads/thread for X
    for (int i = tid; i < MT * 16 * 16; i += 256) {
      const int r = i >> 4, c = i & 15;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < M) v = *reinterpret_cast<const uint4*>(x + (size_t)r * H + k_st + c * 8);
      *reinterpret_cast<uint4*>(smem.tile.sx + r * kRtLd + c * 8) = v;
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int kl = nw_k * 32 + ks * 16 + 2 * t4;
      const bf16* bn = smem.tile.sw + (nw_n * 8 + gq) * kRtLd + kl;
      const uint32_t b0 = u32(bn), b1 = u32(bn + 8);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const bf16* r0 = smem.tile.sx + (16 * mt + gq) * kRtLd + kl;
        const bf16* r1 = r0 + 8 * kRtLd;
        uint32_t a[4] = {u32(r0), u32(r1), u32(r0 + 8), u32(r1 + 8)};
        mma16816(acc[mt], a, b0, b1);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    smem.sp[nw_k][16 * mt + gq][nw_n * 8 + 2 * t4] = acc[mt][0];
    smem.sp[nw_k][16 * mt + gq][nw_n * 8 + 2 * t4 + 1] = acc[mt][1];
    smem.sp[nw_k][16 * mt + gq + 8][nw_n * 8 + 2 * t4] = acc[mt][2];
    smem.sp[nw_k][16 * mt + gq + 8][nw_n * 8 + 2 * t4 + 1] = acc[mt][3];
  }
  __syncthreads();
  for (int i = tid; i < M * 16; i += 256) {
    const int m = i >> 4, e = i & 15;
    const float v = __fadd_rn(__fadd_rn(smem.sp[0][m][e], smem.sp[1][m][e]),
                              __fadd_rn(smem.sp[2][m][e], smem.sp[3][m][e]));
    rlog[((size_t)by * M + m) * glue::NEXP + e0 + e] = v;
  }
  __threadfence();
  __syncthreads();
  __shared__ int is_last;
  if (tid == 0) {
    is_last = 0;
    if (atomicAdd(cnt, 1) == 71) {
      atomicExch(cnt, 0);
      is_last = 1;
    }
  }
  __syncthreads();
  if (!is_last) return;
  // Last CTA runs route_select for all M rows across 8 warps using L1-bypassing __ldcg.
  float bv[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) bv[i] = bias[lane + 32 * i];
  const size_t rstride = (size_t)M * glue::NEXP;
  for (int m = warp; m < M; m += 8) {
    float lgv[4], key[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int e = lane + 32 * i;
      const size_t col = (size_t)m * glue::NEXP + e;
      const float p0 = __ldcg(&rlog[0 * rstride + col]);
      const float p1 = __ldcg(&rlog[1 * rstride + col]);
      const float p2 = __ldcg(&rlog[2 * rstride + col]);
      const float p3 = __ldcg(&rlog[3 * rstride + col]);
      const float p4 = __ldcg(&rlog[4 * rstride + col]);
      const float p5 = __ldcg(&rlog[5 * rstride + col]);
      const float p6 = __ldcg(&rlog[6 * rstride + col]);
      const float p7 = __ldcg(&rlog[7 * rstride + col]);
      const float p8 = __ldcg(&rlog[8 * rstride + col]);
      const float g0 = __fadd_rn(__fadd_rn(p0, p1), __fadd_rn(p2, p3));
      const float g1 = __fadd_rn(__fadd_rn(p4, p5), __fadd_rn(p6, p7));
      const float l = __fadd_rn(__fadd_rn(g0, g1), p8);
      lgv[i] = l;
      key[i] = __fadd_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-l))), bv[i]);
    }
    unsigned used = 0;
    int sel[glue::TOPK];
    float sl[glue::TOPK];
#pragma unroll
    for (int j = 0; j < glue::TOPK; ++j) {
      float bk = -INFINITY;
      int be = 1 << 30;
      float bl = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (!((used >> i) & 1) && (key[i] > bk || (key[i] == bk && lane + 32 * i < be))) {
          bk = key[i]; be = lane + 32 * i; bl = lgv[i];
        }
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
    if (lane < 8) {
      const float my_l = sl[lane];
      const float my_lp = __fadd_rn(fminf(my_l, 0.f), -log1pf(expf(-fabsf(my_l))));
      float mx = my_lp;
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, o));
      const float my_ex = expf(__fadd_rn(my_lp, -mx));
      const float e_pair = __fadd_rn(my_ex, __shfl_xor_sync(0xffu, my_ex, 2));
      const float e_quad = __fadd_rn(e_pair, __shfl_xor_sync(0xffu, e_pair, 1));
      const float s8 = __fadd_rn(e_quad, __shfl_xor_sync(0xffu, e_quad, 4));
      const float base = __fadd_rn(mx, logf(s8));
      idx[m * glue::TOPK + lane] = sel[lane];
      wts[m * glue::TOPK + lane] = __fmul_rn(expf(__fadd_rn(my_lp, -base)), 8.0f);
    }
  }
}

// Split decode Router GEMM (no atomic sync; M = 9..64) + multi-CTA route_select9_kernel<<<M, 32>>>:
// Bit-identical to router_decode_kernel<1> (exact same 9-slice MMA + ((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7)) + p8 tree).
template <int MT>
__global__ void __launch_bounds__(256) router_gemm_dec_kernel(const bf16* __restrict__ x, const bf16* __restrict__ wr,
                                                              float* __restrict__ rlog, int M) {
  union {
    struct {
      __align__(16) bf16 sx[MT * 16 * kRtLd];
      __align__(16) bf16 sw[16 * kRtLd];
    } tile;
    float sp[4][MT * 16][16];
  } __shared__ smem;
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nw_n = warp >> 2, nw_k = warp & 3;
  const int gq = lane >> 2, t4 = lane & 3;
  const int e0 = bx * 16;
  const int k_cta = by * 256;
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
    acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
  auto u32 = [](const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); };
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const int k_st = k_cta + st * 128;
    {
      const int r = tid >> 4, c = tid & 15;
      *reinterpret_cast<uint4*>(smem.tile.sw + r * kRtLd + c * 8) =
          *reinterpret_cast<const uint4*>(wr + (size_t)(e0 + r) * H + k_st + c * 8);
    }
    for (int i = tid; i < MT * 16 * 16; i += 256) {
      const int r = i >> 4, c = i & 15;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < M) v = *reinterpret_cast<const uint4*>(x + (size_t)r * H + k_st + c * 8);
      *reinterpret_cast<uint4*>(smem.tile.sx + r * kRtLd + c * 8) = v;
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int kl = nw_k * 32 + ks * 16 + 2 * t4;
      const bf16* bn = smem.tile.sw + (nw_n * 8 + gq) * kRtLd + kl;
      const uint32_t b0 = u32(bn), b1 = u32(bn + 8);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const bf16* r0 = smem.tile.sx + (16 * mt + gq) * kRtLd + kl;
        const bf16* r1 = r0 + 8 * kRtLd;
        uint32_t a[4] = {u32(r0), u32(r1), u32(r0 + 8), u32(r1 + 8)};
        mma16816(acc[mt], a, b0, b1);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    smem.sp[nw_k][16 * mt + gq][nw_n * 8 + 2 * t4] = acc[mt][0];
    smem.sp[nw_k][16 * mt + gq][nw_n * 8 + 2 * t4 + 1] = acc[mt][1];
    smem.sp[nw_k][16 * mt + gq + 8][nw_n * 8 + 2 * t4] = acc[mt][2];
    smem.sp[nw_k][16 * mt + gq + 8][nw_n * 8 + 2 * t4 + 1] = acc[mt][3];
  }
  __syncthreads();
  for (int i = tid; i < M * 16; i += 256) {
    const int m = i >> 4, e = i & 15;
    const float v = __fadd_rn(__fadd_rn(smem.sp[0][m][e], smem.sp[1][m][e]),
                              __fadd_rn(smem.sp[2][m][e], smem.sp[3][m][e]));
    rlog[((size_t)by * M + m) * glue::NEXP + e0 + e] = v;
  }
}

__global__ void __launch_bounds__(32) route_select9_kernel(const float* __restrict__ rlog, int M,
                                                           const float* __restrict__ bias, int* __restrict__ idx,
                                                           float* __restrict__ wts) {
  const int m = blockIdx.x, lane = threadIdx.x;
  const size_t rstride = (size_t)M * glue::NEXP;
  float lgv[4], key[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = lane + 32 * i;
    const size_t col = (size_t)m * glue::NEXP + e;
    const float p0 = rlog[0 * rstride + col];
    const float p1 = rlog[1 * rstride + col];
    const float p2 = rlog[2 * rstride + col];
    const float p3 = rlog[3 * rstride + col];
    const float p4 = rlog[4 * rstride + col];
    const float p5 = rlog[5 * rstride + col];
    const float p6 = rlog[6 * rstride + col];
    const float p7 = rlog[7 * rstride + col];
    const float p8 = rlog[8 * rstride + col];
    const float g0 = __fadd_rn(__fadd_rn(p0, p1), __fadd_rn(p2, p3));
    const float g1 = __fadd_rn(__fadd_rn(p4, p5), __fadd_rn(p6, p7));
    const float l = __fadd_rn(__fadd_rn(g0, g1), p8);
    lgv[i] = l;
    key[i] = __fadd_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-l))), bias[e]);
  }
  unsigned used = 0;
  int sel[glue::TOPK];
  float sl[glue::TOPK];
#pragma unroll
  for (int j = 0; j < glue::TOPK; ++j) {
    float bk = -INFINITY;
    int be = 1 << 30;
    float bl = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (!((used >> i) & 1) && (key[i] > bk || (key[i] == bk && lane + 32 * i < be))) {
        bk = key[i]; be = lane + 32 * i; bl = lgv[i];
      }
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
  if (lane < 8) {
    const float my_l = sl[lane];
    const float my_lp = __fadd_rn(fminf(my_l, 0.f), -log1pf(expf(-fabsf(my_l))));
    float mx = my_lp;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, o));
    const float my_ex = expf(__fadd_rn(my_lp, -mx));
    const float e_pair = __fadd_rn(my_ex, __shfl_xor_sync(0xffu, my_ex, 2));
    const float e_quad = __fadd_rn(e_pair, __shfl_xor_sync(0xffu, e_pair, 1));
    const float s8 = __fadd_rn(e_quad, __shfl_xor_sync(0xffu, e_quad, 4));
    const float base = __fadd_rn(mx, logf(s8));
    idx[m * glue::TOPK + lane] = sel[lane];
    wts[m * glue::TOPK + lane] = __fmul_rn(expf(__fadd_rn(my_lp, -base)), 8.0f);
  }
}

// Single-launch fused decode message-gate down projection for all G gates (MT = 1, 2, 4 for M <= 16, 32, 64).
// Coalesced 16-byte uint4 smem tiling (128 K-cols/stage x 2 stages = 256 K-cols/CTA).
// X[z]: bf16 [M, 4608], W[z]: bf16 [64, 4608], out[z]: bf16 [M, 64].
// Grid (4, 18, G): bx = 16 output cols, by = 18 K-slices of 256 cols (4608 / 18 = 256), bz = gate. Block 256 = 8 warps.
struct MgDown {
  const bf16* x[4];
  const bf16* W[4];
  bf16* out[4];
  float* ws;
  int* cnt;
};
template <int MT>
__global__ void __launch_bounds__(256) mg_down_dec_kernel(MgDown m, int M) {
  union {
    struct {
      __align__(16) bf16 sx[MT * 16 * kRtLd];
      __align__(16) bf16 sw[16 * kRtLd];
    } tile;
    float sp[4][MT * 16][16];
  } __shared__ smem;
  const int bx = blockIdx.x, ks = blockIdx.y, z = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nw_n = warp >> 2, nw_k = warp & 3;
  const int gq = lane >> 2, t4 = lane & 3;
  const int n0 = bx * 16;
  const int k_cta = ks * 256;
  const bf16* xp = m.x[z];
  const bf16* wp = m.W[z];
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
    acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
  auto u32 = [](const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); };
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const int k_st = k_cta + st * 128;
    {
      const int r = tid >> 4, c = tid & 15;
      *reinterpret_cast<uint4*>(smem.tile.sw + r * kRtLd + c * 8) =
          *reinterpret_cast<const uint4*>(wp + (size_t)(n0 + r) * (2 * H) + k_st + c * 8);
    }
    for (int i = tid; i < MT * 16 * 16; i += 256) {
      const int r = i >> 4, c = i & 15;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < M) v = *reinterpret_cast<const uint4*>(xp + (size_t)r * (2 * H) + k_st + c * 8);
      *reinterpret_cast<uint4*>(smem.tile.sx + r * kRtLd + c * 8) = v;
    }
    __syncthreads();
#pragma unroll
    for (int ksub = 0; ksub < 2; ++ksub) {
      const int kl = nw_k * 32 + ksub * 16 + 2 * t4;
      const bf16* bn = smem.tile.sw + (nw_n * 8 + gq) * kRtLd + kl;
      const uint32_t b0 = u32(bn), b1 = u32(bn + 8);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const bf16* r0 = smem.tile.sx + (16 * mt + gq) * kRtLd + kl;
        const bf16* r1 = r0 + 8 * kRtLd;
        uint32_t a[4] = {u32(r0), u32(r1), u32(r0 + 8), u32(r1 + 8)};
        mma16816(acc[mt], a, b0, b1);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    smem.sp[nw_k][16 * mt + gq][nw_n * 8 + 2 * t4] = acc[mt][0];
    smem.sp[nw_k][16 * mt + gq][nw_n * 8 + 2 * t4 + 1] = acc[mt][1];
    smem.sp[nw_k][16 * mt + gq + 8][nw_n * 8 + 2 * t4] = acc[mt][2];
    smem.sp[nw_k][16 * mt + gq + 8][nw_n * 8 + 2 * t4 + 1] = acc[mt][3];
  }
  __syncthreads();
  const size_t base_z = (size_t)z * 18 * M * 64;
  for (int i = tid; i < M * 16; i += 256) {
    const int row = i >> 4, e = i & 15;
    const float v = __fadd_rn(__fadd_rn(smem.sp[0][row][e], smem.sp[1][row][e]),
                              __fadd_rn(smem.sp[2][row][e], smem.sp[3][row][e]));
    m.ws[base_z + ((size_t)ks * M + row) * 64 + n0 + e] = v;
  }
  __threadfence();
  __syncthreads();
  __shared__ int is_last;
  if (tid == 0) {
    is_last = 0;
    if (atomicAdd(m.cnt + z * 4 + bx, 1) == 17) {
      atomicExch(m.cnt + z * 4 + bx, 0);
      is_last = 1;
    }
  }
  __syncthreads();
  if (!is_last) return;
  bf16* out = m.out[z];
  const size_t stride = (size_t)M * 64;
  for (int i = tid; i < M * 16; i += 256) {
    const int row = i >> 4, e = i & 15;
    const size_t col_off = (size_t)row * 64 + n0 + e;
    float p[18];
#pragma unroll
    for (int s = 0; s < 18; ++s) p[s] = __ldcg(&m.ws[base_z + (size_t)s * stride + col_off]);
    const float a0 = __fadd_rn(__fadd_rn(__fadd_rn(p[0], p[1]), __fadd_rn(p[2], p[3])),
                               __fadd_rn(__fadd_rn(p[4], p[5]), __fadd_rn(p[6], p[7])));
    const float a1 = __fadd_rn(__fadd_rn(__fadd_rn(p[8], p[9]), __fadd_rn(p[10], p[11])),
                               __fadd_rn(__fadd_rn(p[12], p[13]), __fadd_rn(p[14], p[15])));
    const float sum = __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(p[16], p[17]));
    out[col_off] = __float2bfloat16_rn(sum);
  }
}

// Decode BF16 tensor-core message-gate tail (MT = 1, 2, 4 for T <= 16, 32, 64), bit-identical to mg_tail_tc_kernel.
// Tile N = 64 cols/CTA (4 warps x 16 cols = 2 N-tiles of 8): grid (H / 64 = 36, 1, G), block 128.
template <int MT>
__global__ void __launch_bounds__(128) mg_tail_tc_dec_kernel(MgTc m, int T, int ldh, int ldm, int ldo) {
  const int z = blockIdx.z, n0 = blockIdx.x * 64;
  __shared__ __align__(16) bf16 sa[MT * 16 * kTcLd];
  __shared__ __align__(16) bf16 sb[2][64 * kTcLd];
  const int tid = threadIdx.x;
  for (int i = tid; i < MT * 16 * 8; i += 128) {
    const int r = i >> 3, c = i & 7;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < T) v = *reinterpret_cast<const uint4*>(m.gd[z] + (size_t)r * 64 + 8 * c);
    *reinterpret_cast<uint4*>(sa + r * kTcLd + 8 * c) = v;
  }
  for (int i = tid; i < 2 * 64 * 8; i += 128) {
    const int w = i >> 9, r = (i >> 3) & 63, c = i & 7;
    *reinterpret_cast<uint4*>(sb[w] + r * kTcLd + 8 * c) =
        *reinterpret_cast<const uint4*>(m.W[z] + ((size_t)w * H + n0 + r) * 64 + 8 * c);
  }
  __syncthreads();
  const int warp = tid >> 5, lane = tid & 31, gq = lane >> 2, t4 = lane & 3;
  float acc[2][MT][2][4];
#pragma unroll
  for (int w = 0; w < 2; ++w)
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        acc[w][i][j][0] = acc[w][i][j][1] = acc[w][i][j][2] = acc[w][i][j][3] = 0.f;
  auto u32 = [](const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); };
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const int k0 = 16 * ks + 2 * t4;
    uint32_t a[MT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      const bf16* r0 = sa + (16 * i + gq) * kTcLd, *r1 = r0 + 8 * kTcLd;
      a[i][0] = u32(r0 + k0); a[i][1] = u32(r1 + k0); a[i][2] = u32(r0 + k0 + 8); a[i][3] = u32(r1 + k0 + 8);
    }
#pragma unroll
    for (int w = 0; w < 2; ++w)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const bf16* bn = sb[w] + (16 * warp + 8 * j + gq) * kTcLd;
        const uint32_t b0 = u32(bn + k0), b1 = u32(bn + k0 + 8);
#pragma unroll
        for (int i = 0; i < MT; ++i) mma16816(acc[w][i][j], a[i], b0, b1);
      }
  }
  const float *hid = m.hid[z], *msg = m.msg[z];
  float* out = m.out[z];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int t = 16 * i + gq + 8 * hf;
      if (t >= T) continue;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int h = n0 + 16 * warp + 8 * j + 2 * t4;
        const float2 hv = *reinterpret_cast<const float2*>(hid + (size_t)t * ldh + h);
        const float2 mv = *reinterpret_cast<const float2*>(msg + (size_t)t * ldm + h);
        float2 o;
        {
          const float f = tanhf(acc[0][i][j][2 * hf]), a = tanhf(acc[1][i][j][2 * hf]);
          o.x = __fadd_rn(__fmul_rn(__fadd_rn(1.0f, -f), hv.x), __fmul_rn(a, mv.x));
        }
        {
          const float f = tanhf(acc[0][i][j][2 * hf + 1]), a = tanhf(acc[1][i][j][2 * hf + 1]);
          o.y = __fadd_rn(__fmul_rn(__fadd_rn(1.0f, -f), hv.y), __fmul_rn(a, mv.y));
        }
        *reinterpret_cast<float2*>(out + (size_t)t * ldo + h) = o;
      }
    }
}

// ---------------------------------------------------------------------------------------------- 288-thread vectorized AttnRes / Norm / Conv+AttnRes
// 288 threads = 9 warps x 8 consecutive elements = 2304 (16-byte uint4 / 2x float4 loads and stores, single-sync block sum).
// Used for ALL row counts on the production path so decode, prefill, and mixed rows share identical reduction order.
template <int NV>
__device__ __forceinline__ void block_sum288(float (&v)[NV], float* sh /* >= 9 * NV */) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
#pragma unroll
    for (int i = 0; i < NV; ++i) v[i] = __fadd_rn(v[i], __shfl_xor_sync(0xffffffffu, v[i], o));
  const int w = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
#pragma unroll
    for (int i = 0; i < NV; ++i) sh[i * 9 + w] = v[i];
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 9; ++k) s = __fadd_rn(s, sh[i * 9 + k]);
    v[i] = s;
  }
}

__device__ __forceinline__ void ar288_finish(const float (&xs)[5][8], int N, const float* __restrict__ qptr, int c0,
                                             float eps, float* sh, float (&res)[8]) {
  const float qsc = (float)(1.0 / 48.0);
  const float invH = (float)(1.0 / 2304.0);
  float qv[8];
  ld8(qptr + c0, qv);
#pragma unroll
  for (int e = 0; e < 8; ++e) qv[e] = __fmul_rn(qv[e], qsc);
  float v[10];
#pragma unroll
  for (int i = 0; i < 10; ++i) v[i] = 0.f;
#pragma unroll
  for (int n = 0; n < 5; ++n) {
    if (n < N) {
      float ss = 0.f, dq = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float x = xs[n][e];
        ss = __fadd_rn(ss, __fmul_rn(x, x));
        dq = __fadd_rn(dq, __fmul_rn(x, qv[e]));
      }
      v[2 * n] = ss;
      v[2 * n + 1] = dq;
    }
  }
  block_sum288<10>(v, sh);
  float rstd[5], l[5], p[5];
  float mx = -INFINITY;
#pragma unroll
  for (int n = 0; n < 5; ++n) {
    if (n < N) {
      rstd[n] = rsqrtf(__fmaf_rn(v[2 * n], invH, eps));
      l[n] = __fmul_rn(v[2 * n + 1], rstd[n]);
      mx = fmaxf(mx, l[n]);
    } else {
      rstd[n] = 0.f;
      l[n] = 0.f;
    }
  }
  float sum = 0.f;
#pragma unroll
  for (int n = 0; n < 5; ++n) {
    if (n < N) {
      p[n] = expf(__fadd_rn(l[n], -mx));
      sum = __fadd_rn(sum, p[n]);
    } else {
      p[n] = 0.f;
    }
  }
  const float inv = __fdiv_rn(1.0f, sum);
#pragma unroll
  for (int n = 0; n < 5; ++n)
    if (n < N) p[n] = __fmul_rn(p[n], rstd[n]);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float acc = 0.f;
#pragma unroll
    for (int n = 0; n < 5; ++n)
      if (n < N) acc = __fmaf_rn(p[n], xs[n][e], acc);
    res[e] = __fmul_rn(acc, inv);
  }
}

__device__ __forceinline__ void norm288_out(const float (&res)[8], const float* __restrict__ w, bf16* __restrict__ out,
                                            int c0, float eps, float* sh) {
  float wv[8];
  if (w) ld8(w + c0, wv);
  float u[1] = {0.f};
#pragma unroll
  for (int e = 0; e < 8; ++e) u[0] = __fadd_rn(u[0], __fmul_rn(res[e], res[e]));
  block_sum288<1>(u, sh);
  const float r = glue::rstd_of(u[0], H, eps);
  float y[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float v = __fmul_rn(res[e], r);
    if (w) v = __fmul_rn(v, wv[e]);
    y[e] = v;
  }
  st8(out + c0, y);
}

// AttnRes (G == 0: grid (T, Q); G == 1: grid (T, NG) with gate- & query-split), block 288.
template <int G>
__global__ void __launch_bounds__(288) ar288_kernel(glue::ResSrc s, const float* __restrict__ queries,
                                                    float* __restrict__ out, float eps, glue::GateIn gi) {
  __shared__ float sh1[9 * 10];
  const int t = blockIdx.x, c0 = threadIdx.x * 8;
  const int q_idx = (G == 0 || gi.qsel) ? (int)blockIdx.y : 0;
  const int NQ = (G == 0 || gi.qsel) ? (int)gridDim.y : 1;
  const int g_idx = (int)blockIdx.y;
  const int N = s.nb + 1;
  float xs[5][8];
#pragma unroll
  for (int n = 0; n < 5; ++n) {
    if (n < s.nb) ld8(s.blk[n] + (size_t)t * H + c0, xs[n]);
    else if (n == s.nb) {
      if (s.last32) ld8(s.last32 + (size_t)t * s.ld_last + c0, xs[n]);
      else ld8(s.last16 + (size_t)t * s.ld_last + c0, xs[n]);
    }
  }
  float res[8];
  ar288_finish(xs, N, queries + (size_t)q_idx * H, c0, eps, sh1, res);
  if constexpr (G == 0) {
    st8(out + ((size_t)t * NQ + q_idx) * H + c0, res);
  } else {
    if (gi.qsel || g_idx == 0) st8(out + ((size_t)t * NQ + q_idx) * H + c0, res);
    __shared__ float sh2[9 * 2];
    float mv[8];
    ld8(gi.msg + (size_t)t * gi.ldm + (size_t)g_idx * H + c0, mv);
    float u[2] = {0.f, 0.f};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      u[0] = __fadd_rn(u[0], __fmul_rn(res[e], res[e]));
      u[1] = __fadd_rn(u[1], __fmul_rn(mv[e], mv[e]));
    }
    block_sum288<2>(u, sh2);
    const float ra = glue::rstd_of(u[0], H, eps);
    const float rb = glue::rstd_of(u[1], H, eps);
    float ya[8], yb[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      ya[e] = __fmul_rn(res[e], ra);
      yb[e] = __fmul_rn(mv[e], rb);
    }
    bf16* gdst = gi.g + (size_t)g_idx * gi.gs + (size_t)t * 2 * H;
    st8(gdst + c0, ya);
    st8(gdst + H + c0, yb);
  }
}

// AttnRes + RMSNorm: grid (T, Q), block 288.
__global__ void __launch_bounds__(288) ar_norm288_kernel(glue::ResSrc s, const float* __restrict__ queries,
                                                         const float* w0, const float* w1, const float* w2,
                                                         bf16* o0, bf16* o1, bf16* o2, float eps) {
  __shared__ float sh1[9 * 10];
  __shared__ float sh2[9];
  const int t = blockIdx.x, q0 = blockIdx.y, c0 = threadIdx.x * 8;
  const int N = s.nb + 1;
  float xs[5][8];
#pragma unroll
  for (int n = 0; n < 5; ++n) {
    if (n < s.nb) ld8(s.blk[n] + (size_t)t * H + c0, xs[n]);
    else if (n == s.nb) {
      if (s.last32) ld8(s.last32 + (size_t)t * s.ld_last + c0, xs[n]);
      else ld8(s.last16 + (size_t)t * s.ld_last + c0, xs[n]);
    }
  }
  float res[8];
  ar288_finish(xs, N, queries + (size_t)q0 * H, c0, eps, sh1, res);
  const float* ws[3] = {w0, w1, w2};
  bf16* os[3] = {o0, o1, o2};
  norm288_out(res, ws[q0], os[q0] + (size_t)t * H, c0, eps, sh2);
}

// Helper for fused attention conv + partial32 + blk update inside 288-thread MLP-input kernels.
__device__ __forceinline__ void conv_partial288(const bf16* __restrict__ ax, const float* __restrict__ cw,
                                                const bf16* __restrict__ state, long long st_slot_stride,
                                                glue::Rows R, const bf16* __restrict__ hin, int block_start,
                                                bf16* __restrict__ blk_out, float* __restrict__ partial32,
                                                int t, int c0, float (&h_out)[8], float (&p_out)[8]) {
  float y[8];
  {
    float wv[8][4];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float4 q = *reinterpret_cast<const float4*>(cw + (size_t)(c0 + e) * 4);
      wv[e][0] = q.x; wv[e][1] = q.y; wv[e][2] = q.z; wv[e][3] = q.w;
    }
    const int s = R.seg[t], i = t - R.row0[s];
    const bf16* st = state + (size_t)R.slot[s] * st_slot_stride;
    float cur[8], ex[3][8];
    ld8(ax + (size_t)t * H + c0, cur);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      if (i + j < 3) ld8(st + (size_t)(i + j) * glue::SLD + c0, ex[j]);
      else ld8(ax + (size_t)(t + j - 3) * H + c0, ex[j]);
    }
    constexpr float kInv5 = (float)(1.0 / 2.23606797749979);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float acc = __fmul_rn(wv[e][0], ex[0][e]);
      acc = __fadd_rn(acc, __fmul_rn(wv[e][1], ex[1][e]));
      acc = __fadd_rn(acc, __fmul_rn(wv[e][2], ex[2][e]));
      acc = __fadd_rn(acc, __fmul_rn(wv[e][3], cur[e]));
      y[e] = glue::rr16(__fmul_rn(__fadd_rn(acc, cur[e]), kInv5));
    }
  }
  const size_t o = (size_t)t * H + c0;
  const uint4 h_raw = *reinterpret_cast<const uint4*>(hin + o);
  const __nv_bfloat162* hh = reinterpret_cast<const __nv_bfloat162*>(&h_raw);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 a = __bfloat1622float2(hh[k]);
    h_out[2 * k] = a.x;
    h_out[2 * k + 1] = a.y;
  }
  if (block_start) {
    *reinterpret_cast<uint4*>(blk_out + o) = h_raw;
#pragma unroll
    for (int e = 0; e < 8; ++e) p_out[e] = y[e];
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) p_out[e] = __fadd_rn(h_out[e], y[e]);
  }
  st8(partial32 + o, p_out);
}

// Fused conv_partial + ar_mlp + mlp_norm (plain visits, !has_msgs): grid (T, 1), block 288.
// s.nb = nb BEFORE bstart increment; s.blk[0..nb-1] are completed blocks before this layer.
__global__ void __launch_bounds__(288) conv_ar_norm288_kernel(const bf16* __restrict__ ax, const float* __restrict__ cw,
                                                              const bf16* __restrict__ state, long long st_slot_stride,
                                                              glue::Rows R, const bf16* __restrict__ hin,
                                                              int block_start, bf16* __restrict__ blk_out,
                                                              float* __restrict__ partial32, glue::ResSrc s,
                                                              const float* __restrict__ query,
                                                              const float* __restrict__ norm_w,
                                                              bf16* __restrict__ xm_out, float eps) {
  __shared__ float sh1[9 * 10];
  __shared__ float sh2[9];
  const int t = blockIdx.x, c0 = threadIdx.x * 8;
  float h_v[8], p_v[8];
  conv_partial288(ax, cw, state, st_slot_stride, R, hin, block_start, blk_out, partial32, t, c0, h_v, p_v);
  const int nb_prev = s.nb;
  const int N = nb_prev + (block_start ? 2 : 1);
  const size_t o = (size_t)t * H + c0;
  float xs[5][8];
#pragma unroll
  for (int n = 0; n < 5; ++n) {
    if (n < nb_prev) {
      ld8(s.blk[n] + o, xs[n]);
    } else if (block_start && n == nb_prev) {
#pragma unroll
      for (int e = 0; e < 8; ++e) xs[n][e] = h_v[e];
    } else if (n == N - 1) {
#pragma unroll
      for (int e = 0; e < 8; ++e) xs[n][e] = p_v[e];
    }
  }
  float res[8];
  ar288_finish(xs, N, query, c0, eps, sh1, res);
  norm288_out(res, norm_w, xm_out + (size_t)t * H, c0, eps, sh2);
}

// Fused conv_partial + ar_mlp_m + mg_in (loop-message visits, has_msgs, Q=1, G=1): grid (T, 1), block 288.
__global__ void __launch_bounds__(288) conv_ar_gate288_kernel(const bf16* __restrict__ ax, const float* __restrict__ cw,
                                                              const bf16* __restrict__ state, long long st_slot_stride,
                                                              glue::Rows R, const bf16* __restrict__ hin,
                                                              int block_start, bf16* __restrict__ blk_out,
                                                              float* __restrict__ partial32, glue::ResSrc s,
                                                              const float* __restrict__ query,
                                                              float* __restrict__ ar_out, float eps, glue::GateIn gi) {
  __shared__ float sh1[9 * 10];
  __shared__ float sh2[9 * 2];
  const int t = blockIdx.x, c0 = threadIdx.x * 8;
  float h_v[8], p_v[8];
  conv_partial288(ax, cw, state, st_slot_stride, R, hin, block_start, blk_out, partial32, t, c0, h_v, p_v);
  const int nb_prev = s.nb;
  const int N = nb_prev + (block_start ? 2 : 1);
  const size_t o = (size_t)t * H + c0;
  float xs[5][8];
#pragma unroll
  for (int n = 0; n < 5; ++n) {
    if (n < nb_prev) {
      ld8(s.blk[n] + o, xs[n]);
    } else if (block_start && n == nb_prev) {
#pragma unroll
      for (int e = 0; e < 8; ++e) xs[n][e] = h_v[e];
    } else if (n == N - 1) {
#pragma unroll
      for (int e = 0; e < 8; ++e) xs[n][e] = p_v[e];
    }
  }
  float res[8];
  ar288_finish(xs, N, query, c0, eps, sh1, res);
  st8(ar_out + o, res);
  float mv[8];
  ld8(gi.msg + (size_t)t * gi.ldm + c0, mv);
  float u[2] = {0.f, 0.f};
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    u[0] = __fadd_rn(u[0], __fmul_rn(res[e], res[e]));
    u[1] = __fadd_rn(u[1], __fmul_rn(mv[e], mv[e]));
  }
  block_sum288<2>(u, sh2);
  const float ra = glue::rstd_of(u[0], H, eps);
  const float rb = glue::rstd_of(u[1], H, eps);
  float ya[8], yb[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    ya[e] = __fmul_rn(res[e], ra);
    yb[e] = __fmul_rn(mv[e], rb);
  }
  bf16* gdst = gi.g + (size_t)t * 2 * H;
  st8(gdst + c0, ya);
  st8(gdst + H + c0, yb);
}

}  // namespace pglue
