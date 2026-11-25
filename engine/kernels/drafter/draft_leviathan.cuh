// inkling-cmp-engine: DSpark exact Leviathan top-16 proposal & rejection sampling (own code).
//
// When any session in a draft batch has temperature >= 1e-5, the sampled head path runs:
//   memset_zero(mk.proj)
//   for i in 0..7:
//     markov_recur_proj(mk, i, lb)         // recurrence + fused rank-16 projection p = lrV @ lat + step init
//     markov_lev_scan_fused<N>(mk, i, lb)  // 1x lrU/fold_b read across all N sessions, 3-sync parallel-warp top-16 + Z
//     markov_lev_finish(mk, i, lb)         // 16-warp v16_lo global top-16 + parallel full-rank fold_w logits ->
//                                          // exact 2^22 weights -> Philox4x32-10 draw -> keys, drafts, prop_ids/w, conf_c
//
// Target verification reuses ink::sample_leviathan_kernel (exact 128-bit arithmetic on W * 2^22), preserving the
// exact target filtered distribution p(x) while keeping greedy (temp < 1e-5) bit-identical via the existing path.
#pragma once

namespace s2f {

constexpr int LEV_K = 16;
constexpr unsigned LEV_QTOT = 1u << 22;
constexpr int LEV_ROWS = 512;
constexpr int LEV_NCH = (VU + LEV_ROWS - 1) / LEV_ROWS;  // 391
constexpr int LEV_NCAND = LEV_NCH * LEV_K;               // 6,256

struct LevBuf {
  float* chunk_val;                    // [n][LEV_NCAND] compacted candidate scores
  int* chunk_id;                       // [n][LEV_NCAND] compacted candidate ids
  float* chunk_z;                      // [LEV_NCH][n]
  float* chunk_cmax;                   // [LEV_NCH][n]
  int* cand_cnt;                       // [n]
  unsigned int* g_lb;                  // [n] ordered f32 lower bound on global 16th-largest
  const float* samp_temp;              // [n_total]
  const float* samp_top_p;             // [n_total]
  const unsigned long long* samp_seed; // [n_total]
  const int* pos_c;                    // [k] (P_c)
  const int* sel;                      // [k] (compact r -> slot)
  int* prop_ids;                       // [n_total][7][16]
  unsigned int* prop_w;                // [n_total][7][16]
  float* conf_c;                       // [k][7]
  int full_verify;                     // 1: conf_c=0 for sampled sessions (w8); 0: use conf_mode
  int conf_mode;                       // 0: log p_max (default), 1: log p(c*), 2: log q(c*)
};

// Philox4x32-10 returning the first 32-bit word (matches df2::philox4x32_10 & ink::samp_philox).
__device__ __forceinline__ unsigned int lev_philox4x32_10(unsigned long long seed, int pos, int slot,
                                                          unsigned int domain) {
  unsigned int c0 = (unsigned int)pos;
  unsigned int c1 = (unsigned int)slot;
  unsigned int c2 = domain;
  unsigned int c3 = 0u;
  unsigned int k0 = (unsigned int)(seed & 0xffffffffull);
  unsigned int k1 = (unsigned int)(seed >> 32);
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    unsigned int hi0 = __umulhi(0xD2511F53u, c0);
    unsigned int lo0 = 0xD2511F53u * c0;
    unsigned int hi1 = __umulhi(0xCD9E8D57u, c2);
    unsigned int lo1 = 0xCD9E8D57u * c2;
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c0;
}

__device__ __forceinline__ bool lev_better(float va, int ia, float vb, int ib) {
  return (va > vb) || (va == vb && ia < ib);
}

__device__ __forceinline__ unsigned int lev_f32_to_ord(float f) {
  unsigned int u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float lev_ord_to_f32(unsigned int o) {
  unsigned int u = (o & 0x80000000u) ? (o & 0x7fffffffu) : ~o;
  return __uint_as_float(u);
}

// grid 32, block 256: fused markov_recur + rank-16 projection p = lrV @ lat + per-step lb init.
__global__ void __launch_bounds__(256) markov_recur_proj(Markov mk, int i, LevBuf lb) {
  __shared__ float inp[8][2 * RANK];
  __shared__ float sm_lat[8][8];
  __shared__ float sm_v[LRK][8];
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
  if (blockIdx.x == 0 && threadIdx.x < n) {
    lb.cand_cnt[threadIdx.x] = 0;
    lb.g_lb[threadIdx.x] = 0x007FFFFFu;  // lev_f32_to_ord(-INFINITY)
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int q0 = blockIdx.x * 8;
  const int q = q0 + warp;
  float raw[3][8];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const __nv_bfloat162* w = (const __nv_bfloat162*)(mk.w_state + (size_t)(q + c * RANK) * 2 * RANK);
    float wv[16];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      float2 f = __bfloat1622float2(w[lane + u * 32]);
      wv[2 * u] = f.x;
      wv[2 * u + 1] = f.y;
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
      for (int rr2 = 1; rr2 < 8; ++rr2)
        if (rr2 == r) v = raw[c][rr2];
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
    sm_lat[r][warp] = lat;
  }
  if (threadIdx.x < LRK * 8) {
    const int j = threadIdx.x >> 3;
    const int o = threadIdx.x & 7;
    sm_v[j][o] = mk.lrV[j * RANK + q0 + o];
  }
  __syncthreads();
  if (threadIdx.x < n * LRK) {
    const int r = threadIdx.x >> 4;
    const int j = threadIdx.x & 15;
    float acc = 0.f;
#pragma unroll
    for (int o = 0; o < 8; ++o) acc += sm_v[j][o] * sm_lat[r][o];
    atomicAdd(&mk.proj[r * (LRK + 4) + j], acc);
  }
}

// grid (LEV_NCH=391, 1), block 256: reads lrU & fold_b ONCE for all N sessions;
// only 3 __syncthreads() across all N sessions combined, and warps 0..N-1 extract top-16 in parallel!
template <int N>
__global__ void __launch_bounds__(256) markov_lev_scan_fused(Markov mk, int i, LevBuf lb) {
  constexpr int CAP = 64;
  __shared__ float pr[N][LRK];
  __shared__ float w1s[N][8];
  __shared__ float w2s[N][8];
  __shared__ float zs[N][8];
  __shared__ float cmax_s[N];
  __shared__ float thresh_s[N];
  __shared__ float invt_s[N];
  __shared__ int skip_s[N];
  __shared__ int cnt_s[N];
  __shared__ float cand_v[N][CAP];
  __shared__ int cand_i[N][CAP];
  __shared__ int any_overflow;

  const int chunk = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int e = tid; e < N * LRK; e += 256) {
    pr[e >> 4][e & 15] = mk.proj[(e >> 4) * (LRK + 4) + (e & 15)];
  }
  if (tid < N) {
    const float t = lb.samp_temp[lb.sel[tid]];
    invt_s[tid] = (t >= 1e-5f) ? (1.0f / t) : 1.0f;
  }
  if (tid == 0) any_overflow = 0;
  __syncthreads();

  const int v0 = chunk * LEV_ROWS + tid;
  const int v1 = v0 + 256;
  float sc[N][2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int v = j == 0 ? v0 : v1;
    if (v < VU) {
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
          for (int r = 0; r < N; ++r) {
            dot[r] += pr[r][c * 8 + 2 * e] * w.x + pr[r][c * 8 + 2 * e + 1] * w.y;
          }
        }
      }
      const float fb = bf2f(mk.fold_b[v]);
#pragma unroll
      for (int r = 0; r < N; ++r) {
        const float s = rb(mk.Bq[(size_t)(r * 7 + i) * VP + v] + rb(fb + dot[r]));
        sc[r][j] = SOFTCAP * tanh_fast(s * (1.0f / SOFTCAP));
      }
    } else {
#pragma unroll
      for (int r = 0; r < N; ++r) sc[r][j] = -INFINITY;
    }
  }

  // Phase 1: warp top-2 for ALL N sessions before sync #1
#pragma unroll
  for (int r = 0; r < N; ++r) {
    const float s0 = sc[r][0], s1 = sc[r][1];
    const float hi = s0 >= s1 ? s0 : s1;
    const float lo = s0 >= s1 ? s1 : s0;
    float w1 = hi;
    int wl = lane;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, w1, off);
      int ol = __shfl_xor_sync(0xffffffffu, wl, off);
      if (ov > w1 || (ov == w1 && ol < wl)) { w1 = ov; wl = ol; }
    }
    const float w2 = warp_max(lane == wl ? lo : hi);
    if (lane == 0) {
      w1s[r][warp] = w1;
      w2s[r][warp] = w2;
    }
  }
  __syncthreads();  // Sync #1

  // Threads 0..N-1 compute cmax, vlo, atomicMax(g_lb), thresh, skip in parallel!
  if (tid < N) {
    float cm = w1s[tid][0], vlo = w2s[tid][0];
#pragma unroll
    for (int w = 1; w < 8; ++w) {
      cm = fmaxf(cm, w1s[tid][w]);
      vlo = fminf(vlo, w2s[tid][w]);
    }
    cmax_s[tid] = cm;
    const unsigned int vlo_ord = vlo > -INFINITY ? lev_f32_to_ord(vlo) : 0x007FFFFFu;
    const unsigned int old_ord = vlo > -INFINITY ? atomicMax(&lb.g_lb[tid], vlo_ord) : atomicOr(&lb.g_lb[tid], 0u);
    const unsigned int cur_ord = old_ord > vlo_ord ? old_ord : vlo_ord;
    const float cur_lb = lev_ord_to_f32(cur_ord);
    thresh_s[tid] = fmaxf(vlo, cur_lb);
    skip_s[tid] = (cm < cur_lb) ? 1 : 0;
    cnt_s[tid] = 0;
  }
  __syncthreads();  // Sync #2

  // Phase 2: compute Z and compact >= thresh for ALL N sessions before sync #3
#pragma unroll
  for (int r = 0; r < N; ++r) {
    const float s0 = sc[r][0], s1 = sc[r][1];
    const float cmax = cmax_s[r];
    const float it = invt_s[r];
    float zsum = 0.f;
    if (v0 < VU) zsum += __expf((s0 - cmax) * it);
    if (v1 < VU) zsum += __expf((s1 - cmax) * it);
    zsum = warp_sum(zsum);
    if (lane == 0) zs[r][warp] = zsum;

    if (!skip_s[r]) {
      const float thr = thresh_s[r];
      if (v0 < VU && s0 >= thr) {
        int pos = atomicAdd(&cnt_s[r], 1);
        if (pos < CAP) { cand_v[r][pos] = s0; cand_i[r][pos] = v0; }
      }
      if (v1 < VU && s1 >= thr) {
        int pos = atomicAdd(&cnt_s[r], 1);
        if (pos < CAP) { cand_v[r][pos] = s1; cand_i[r][pos] = v1; }
      }
    }
  }
  __syncthreads();  // Sync #3

  // Phase 3: Warps 0..N-1 run IN PARALLEL — each warp writes chunk_z/cmax and extracts top-16 for session r = warp!
  if (warp < N) {
    const int r = warp;
    if (lane == 0) {
      float zt = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) zt += zs[r][w];
      lb.chunk_z[chunk * N + r] = zt;
      lb.chunk_cmax[chunk * N + r] = cmax_s[r];
    }
    if (!skip_s[r]) {
      const int cnt = cnt_s[r];
      if (cnt <= CAP) {
        const int take = cnt < LEV_K ? cnt : LEV_K;
        if (take > 0) {
          int base = 0;
          if (lane == 0) base = atomicAdd(&lb.cand_cnt[r], take);
          base = __shfl_sync(0xffffffffu, base, 0);
          float lv0 = lane < cnt ? cand_v[r][lane] : -INFINITY;
          int li0 = lane < cnt ? cand_i[r][lane] : VU;
          float lv1 = (lane + 32) < cnt ? cand_v[r][lane + 32] : -INFINITY;
          int li1 = (lane + 32) < cnt ? cand_i[r][lane + 32] : VU;
          float last_v = -INFINITY;
          for (int p = 0; p < take; ++p) {
            const bool pick1 = lev_better(lv1, li1, lv0, li0);
            float wv = pick1 ? lv1 : lv0;
            int wi = pick1 ? li1 : li0;
            const float myv = wv;
            const int myi = wi;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
              float ov = __shfl_xor_sync(0xffffffffu, wv, off);
              int oi = __shfl_xor_sync(0xffffffffu, wi, off);
              if (lev_better(ov, oi, wv, wi)) { wv = ov; wi = oi; }
            }
            if (myi == wi && myv == wv) {
              if (pick1) { lv1 = -INFINITY; li1 = VU; }
              else { lv0 = -INFINITY; li0 = VU; }
            }
            if (lane == 0 && base + p < LEV_NCAND) {
              lb.chunk_val[(size_t)r * LEV_NCAND + base + p] = wv;
              lb.chunk_id[(size_t)r * LEV_NCAND + base + p] = wi < VU ? wi : -1;
              last_v = wv;
            }
          }
          if (lane == 0 && take == LEV_K && last_v > -INFINITY) {
            atomicMax(&lb.g_lb[r], lev_f32_to_ord(last_v));
          }
        }
      } else if (lane == 0) {
        any_overflow = 1;
      }
    }
  }
  __syncthreads();  // Sync #4 (only needed for rare overflow check; zero cost if no overflow!)
  if (!any_overflow) return;

  // Rare degenerate tie fallback (cnt_s[r] > 64): exact 16-pass over (sc[r][0], sc[r][1])
  __shared__ int fb_id[8];
  __shared__ int fb_tid[8];
  __shared__ int fb_win;
  __shared__ int fb_base;
#pragma unroll
  for (int r = 0; r < N; ++r) {
    if (!skip_s[r] && cnt_s[r] > CAP) {
      if (tid == 0) fb_base = atomicAdd(&lb.cand_cnt[r], LEV_K);
      __syncthreads();
      const int base = fb_base;
      float r0 = sc[r][0], r1 = sc[r][1];
      int i0 = v0 < VU ? v0 : VU, i1 = v1 < VU ? v1 : VU;
      for (int p = 0; p < LEV_K; ++p) {
        const bool p1 = lev_better(r1, i1, r0, i0);
        float bv = p1 ? r1 : r0;
        int bi = p1 ? i1 : i0;
        int bt = tid;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          float ov = __shfl_xor_sync(0xffffffffu, bv, off);
          int oi = __shfl_xor_sync(0xffffffffu, bi, off);
          int ot = __shfl_xor_sync(0xffffffffu, bt, off);
          if (lev_better(ov, oi, bv, bi)) { bv = ov; bi = oi; bt = ot; }
        }
        if (lane == 0) { w1s[0][warp] = bv; fb_id[warp] = bi; fb_tid[warp] = bt; }
        __syncthreads();
        if (warp == 0) {
          float wv = lane < 8 ? w1s[0][lane] : -INFINITY;
          int wi = lane < 8 ? fb_id[lane] : VU;
          int wt = lane < 8 ? fb_tid[lane] : -1;
#pragma unroll
          for (int off = 4; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, wv, off);
            int oi = __shfl_xor_sync(0xffffffffu, wi, off);
            int ot = __shfl_xor_sync(0xffffffffu, wt, off);
            if (lev_better(ov, oi, wv, wi)) { wv = ov; wi = oi; wt = ot; }
          }
          if (lane == 0) {
            if (base + p < LEV_NCAND) {
              lb.chunk_val[(size_t)r * LEV_NCAND + base + p] = wv;
              lb.chunk_id[(size_t)r * LEV_NCAND + base + p] = wi < VU ? wi : -1;
            }
            fb_win = wt;
          }
        }
        __syncthreads();
        if (tid == fb_win) {
          if (lev_better(r1, i1, r0, i0)) { r1 = -INFINITY; i1 = VU; }
          else { r0 = -INFINITY; i0 = VU; }
        }
        __syncthreads();
      }
    }
  }
}

// grid k, block 512 (16 warps): 2-sync v16_lo global top-16 -> 16-warp parallel exact_fold_z_warp ->
// exact 2^22 weights -> Philox4x32-10 draw -> mk.keys, mk.drafts, prop_ids, prop_w, conf_c, reset mk.proj.
__global__ void __launch_bounds__(512) markov_lev_finish(Markov mk, int i, LevBuf lb) {
  constexpr int FIN_CAP = 128;
  __shared__ float lat[RANK];
  __shared__ int top_id[LEV_K];
  __shared__ float exact_z[LEV_K];
  __shared__ float sv[LEV_K];
  __shared__ int si[LEV_K];
  __shared__ float sm_v[16];
  __shared__ int sm_id[16];
  __shared__ int sm_idx[16];
  __shared__ float fin_v[FIN_CAP];
  __shared__ int fin_i[FIN_CAP];
  __shared__ int fin_cnt;
  __shared__ float fin_thr;
  __shared__ int win_idx;
  __shared__ float approx_max;
  __shared__ float z_tot;

  const int r = blockIdx.x, k = mk.n;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int slot = lb.sel[r];
  const float temp = lb.samp_temp[slot];
  const float inv_t = (temp >= 1e-5f) ? (1.0f / temp) : 1.0f;

  if (tid < RANK) lat[tid] = bf2f(mk.latent[r * RANK + tid]);
  if (tid < LRK) mk.proj[r * (LRK + 4) + tid] = 0.f;  // ready for step i + 1

  const float my_cz = tid < LEV_NCH ? lb.chunk_z[tid * k + r] : 0.f;
  const float my_cm = tid < LEV_NCH ? lb.chunk_cmax[tid * k + r] : -INFINITY;
  float cm_max = warp_max(my_cm);
  if (lane == 0) sm_v[warp] = cm_max;
  __syncthreads();
  if (warp == 0) {
    float am = lane < 16 ? sm_v[lane] : -INFINITY;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
    if (lane == 0) approx_max = am;
  }
  __syncthreads();
  float zloc = tid < LEV_NCH ? my_cz * expf((my_cm - approx_max) * inv_t) : 0.f;
  zloc = warp_sum(zloc);
  if (lane == 0) sm_v[warp] = zloc;
  __syncthreads();
  if (warp == 0) {
    float zt = lane < 16 ? sm_v[lane] : 0.f;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) zt += __shfl_xor_sync(0xffffffffu, zt, off);
    if (lane == 0) {
      z_tot = zt;
      fin_cnt = 0;
    }
  }
  __syncthreads();  // wait for warp 0 to finish reading sm_v[0..15] and writing z_tot/fin_cnt

  int cnt = lb.cand_cnt[r];
  if (cnt > LEV_NCAND) cnt = LEV_NCAND;
  float* g_val = lb.chunk_val + (size_t)r * LEV_NCAND;
  int* g_id = lb.chunk_id + (size_t)r * LEV_NCAND;

  if (cnt <= 512) {
    // Fast register-resident exact global top-16 (zero global re-reads)
    float cv = tid < cnt ? g_val[tid] : -INFINITY;
    int ci = tid < cnt ? g_id[tid] : VU;
    if (ci < 0) { cv = -INFINITY; ci = VU; }
    for (int p = 0; p < LEV_K; ++p) {
      float bv = cv;
      int bi = ci;
      int bt = tid;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        int ot = __shfl_xor_sync(0xffffffffu, bt, off);
        if (lev_better(ov, oi, bv, bi)) { bv = ov; bi = oi; bt = ot; }
      }
      if (lane == 0) { sm_v[warp] = bv; sm_id[warp] = bi; sm_idx[warp] = bt; }
      __syncthreads();
      if (warp == 0) {
        float wv = lane < 16 ? sm_v[lane] : -INFINITY;
        int wi = lane < 16 ? sm_id[lane] : VU;
        int wt = lane < 16 ? sm_idx[lane] : -1;
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
          float ov = __shfl_xor_sync(0xffffffffu, wv, off);
          int oi = __shfl_xor_sync(0xffffffffu, wi, off);
          int ot = __shfl_xor_sync(0xffffffffu, wt, off);
          if (lev_better(ov, oi, wv, wi)) { wv = ov; wi = oi; wt = ot; }
        }
        if (lane == 0) {
          top_id[p] = wi < VU ? wi : 0;
          win_idx = wt;
        }
      }
      __syncthreads();
      if (tid == win_idx) { cv = -INFINITY; ci = VU; }
    }
  } else {
    // cnt > 512: all 16 warps are non-empty => 16 distinct warp maxima => min <= global 16th-largest!
    float wmax = -INFINITY;
    for (int e = tid; e < cnt; e += 512) {
      float v = g_val[e];
      int id = g_id[e];
      if (id >= 0 && v > wmax) wmax = v;
    }
    wmax = warp_max(wmax);
    if (lane == 0) sm_v[warp] = wmax;
    __syncthreads();
    if (warp == 0) {
      float mn = lane < 16 ? sm_v[lane] : INFINITY;
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
      if (lane == 0) fin_thr = (mn < INFINITY) ? mn : -INFINITY;
    }
    __syncthreads();

    const float thr = fin_thr;
    for (int e = tid; e < cnt; e += 512) {
      float v = g_val[e];
      int id = g_id[e];
      if (id >= 0 && v >= thr) {
        int pos = atomicAdd(&fin_cnt, 1);
        if (pos < FIN_CAP) {
          fin_v[pos] = v;
          fin_i[pos] = id;
        }
      }
    }
    __syncthreads();

    if (fin_cnt <= FIN_CAP && fin_cnt >= LEV_K) {
      if (warp == 0) {
        const int fc = fin_cnt;
        float lv[4];
        int li[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          int idx = lane + u * 32;
          if (idx < fc) { lv[u] = fin_v[idx]; li[u] = fin_i[idx]; }
          else { lv[u] = -INFINITY; li[u] = VU; }
        }
        for (int p = 0; p < LEV_K; ++p) {
          float bv = lv[0]; int bi = li[0]; int bu = 0;
          if (lev_better(lv[1], li[1], bv, bi)) { bv = lv[1]; bi = li[1]; bu = 1; }
          if (lev_better(lv[2], li[2], bv, bi)) { bv = lv[2]; bi = li[2]; bu = 2; }
          if (lev_better(lv[3], li[3], bv, bi)) { bv = lv[3]; bi = li[3]; bu = 3; }
          float wv = bv; int wi = bi;
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, wv, off);
            int oi = __shfl_xor_sync(0xffffffffu, wi, off);
            if (lev_better(ov, oi, wv, wi)) { wv = ov; wi = oi; }
          }
          if (bi == wi && bv == wv) {
            if (bu == 0) { lv[0] = -INFINITY; li[0] = VU; }
            else if (bu == 1) { lv[1] = -INFINITY; li[1] = VU; }
            else if (bu == 2) { lv[2] = -INFINITY; li[2] = VU; }
            else { lv[3] = -INFINITY; li[3] = VU; }
          }
          if (lane == 0) top_id[p] = wi < VU ? wi : 0;
        }
      }
      __syncthreads();
    } else {
      for (int p = 0; p < LEV_K; ++p) {
        float bv = -INFINITY;
        int bi = VU;
        int be = -1;
        for (int e = tid; e < cnt; e += 512) {
          float v = g_val[e];
          int id = g_id[e];
          if (id < 0) { v = -INFINITY; id = VU; }
          if (lev_better(v, id, bv, bi)) { bv = v; bi = id; be = e; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          float ov = __shfl_xor_sync(0xffffffffu, bv, off);
          int oi = __shfl_xor_sync(0xffffffffu, bi, off);
          int oe = __shfl_xor_sync(0xffffffffu, be, off);
          if (lev_better(ov, oi, bv, bi)) { bv = ov; bi = oi; be = oe; }
        }
        if (lane == 0) { sm_v[warp] = bv; sm_id[warp] = bi; sm_idx[warp] = be; }
        __syncthreads();
        if (warp == 0) {
          float wv = lane < 16 ? sm_v[lane] : -INFINITY;
          int wi = lane < 16 ? sm_id[lane] : VU;
          int we = lane < 16 ? sm_idx[lane] : -1;
#pragma unroll
          for (int off = 8; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, wv, off);
            int oi = __shfl_xor_sync(0xffffffffu, wi, off);
            int oe = __shfl_xor_sync(0xffffffffu, we, off);
            if (lev_better(ov, oi, wv, wi)) { wv = ov; wi = oi; we = oe; }
          }
          if (lane == 0) {
            top_id[p] = wi < VU ? wi : 0;
            win_idx = we;
          }
        }
        __syncthreads();
        if (tid == 0 && win_idx >= 0) {
          g_val[win_idx] = -INFINITY;
          g_id[win_idx] = VU;
        }
        __syncthreads();
      }
    }
  }

  // Exact full-rank BF16 Markov logits for ALL 16 candidates in parallel (16 warps x 1 candidate!)
  {
    const int vid = top_id[warp];
    const float ez = (vid >= 0 && vid < VU) ? exact_z_warp<true>(mk, r, i, vid, lat, lane) : -INFINITY;
    if (lane == 0) exact_z[warp] = ez;
  }
  __syncthreads();

  if (tid == 0) {
    const float top_p = lb.samp_top_p ? lb.samp_top_p[slot] : 1.0f;
    const int dst = (slot * 7 + i) * LEV_K;

    // Sort the 16 exact candidates in-place in shared memory descending by (exact_z, -top_id).
    for (int c = 1; c < LEV_K; ++c) {
      float zv = exact_z[c];
      int iv = top_id[c];
      int j = c;
      while (j > 0 && lev_better(zv, iv, exact_z[j - 1], top_id[j - 1])) {
        exact_z[j] = exact_z[j - 1];
        top_id[j] = top_id[j - 1];
        --j;
      }
      exact_z[j] = zv;
      top_id[j] = iv;
    }

    const float z_max = exact_z[0];
    int d = top_id[0];
    float z_d = z_max;
    float log_p = 0.f;
    float z_full = z_tot * expf((approx_max - z_max) * inv_t);
    if (!isfinite(z_full) || z_full < 1.0f) z_full = 1.0f;

    if (temp < 1e-5f) {
      lb.prop_ids[dst + 0] = d;
      lb.prop_w[dst + 0] = LEV_QTOT;
      for (int c = 1; c < LEV_K; ++c) {
        lb.prop_ids[dst + c] = 0;
        lb.prop_w[dst + c] = 0u;
      }
      log_p = -logf(z_full);
    } else {
      float sum_p = 0.f;
      for (int c = 0; c < LEV_K; ++c) {
        float e = expf((exact_z[c] - z_max) * inv_t);
        sv[c] = (e == e && e > 0.f) ? e : 0.f;
        sum_p += sv[c];
      }
      const float denom = fmaxf(sum_p, z_full);
      int keep = LEV_K;
      if (top_p > 0.f && top_p < 1.f) {
        const float thresh = top_p * denom;
        float cum = 0.f;
        for (int c = 0; c < LEV_K; ++c) {
          cum += sv[c];
          if (cum >= thresh) {
            keep = c + 1;
            break;
          }
        }
      }
      float keep_sum = 0.f;
      for (int c = 0; c < keep; ++c) keep_sum += sv[c];

      unsigned int w_sum = 0u;
      if (keep == 1 || !(keep_sum > 0.f)) {
        si[0] = (int)LEV_QTOT;
        for (int c = 1; c < LEV_K; ++c) si[c] = 0;
      } else {
        const float inv_s = (float)LEV_QTOT / keep_sum;
        for (int c = 0; c < keep; ++c) {
          long long qv = llroundf(sv[c] * inv_s);
          if (qv < 0) qv = 0;
          if (qv > (long long)LEV_QTOT) qv = (long long)LEV_QTOT;
          si[c] = (int)qv;
          w_sum += (unsigned int)qv;
        }
        for (int c = keep; c < LEV_K; ++c) si[c] = 0;
        if (w_sum < LEV_QTOT) {
          si[0] += (int)(LEV_QTOT - w_sum);
        } else if (w_sum > LEV_QTOT) {
          unsigned int excess = w_sum - LEV_QTOT;
          if ((unsigned int)si[0] >= excess) {
            si[0] -= (int)excess;
          } else {
            for (int c = 0; c < keep && excess > 0u; ++c) {
              unsigned int wi = (unsigned int)si[c];
              unsigned int take = wi < excess ? wi : excess;
              si[c] = (int)(wi - take);
              excess -= take;
            }
          }
        }
      }
      const unsigned long long seed = lb.samp_seed[slot];
      const int pos = lb.pos_c[r] + 1 + i;
      const unsigned int u = lev_philox4x32_10(seed, pos, slot, 0x44524654u) & (LEV_QTOT - 1u);
      unsigned int acc = 0u;
      int chosen = 0;
      for (int c = 0; c < keep; ++c) {
        acc += (unsigned int)si[c];
        if (u < acc) {
          chosen = c;
          break;
        }
      }
      d = top_id[chosen];
      z_d = exact_z[chosen];
      for (int c = 0; c < LEV_K; ++c) {
        lb.prop_ids[dst + c] = top_id[c];
        lb.prop_w[dst + c] = (unsigned int)si[c];
      }
      if (lb.full_verify) {
        log_p = 0.f;
      } else if (lb.conf_mode == 1) {
        log_p = (z_d - z_max) * inv_t - logf(z_full);
      } else if (lb.conf_mode == 2) {
        log_p = logf(fmaxf((float)si[chosen] / (float)LEV_QTOT, 1e-9f));
      } else {
        log_p = -logf(z_full);
      }
    }

    mk.keys[i * k + r] = argkey(z_d, d);
    mk.drafts[r * 7 + i] = d;
    const float prev_conf = i == 0 ? 0.f : lb.conf_c[r * 7 + i - 1];
    lb.conf_c[r * 7 + i] = prev_conf + log_p;
  }
}

}  // namespace s2f
