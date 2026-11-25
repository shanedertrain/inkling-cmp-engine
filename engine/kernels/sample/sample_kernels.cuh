// inkling-cmp-engine native runtime: device sampling of logit rows (own code). One block of 1,024 threads per row.
//
// Distribution (the target's filtered distribution; README of bench/raw/2026-11-23-spec-sampling/ has the proof of
// the speculative rule built on it):
//   max M = the row's largest logit (first index on ties, NaN never wins: argmax_rows_kernel's rule);
//   e_i = __expf((x_i - M) * (1 / T)) in fp32 (__frcp_rn, round-to-nearest ops), key_i = the float bits of e_i;
//   weight w_i = floor(e_i * 2^40) (an integer <= 2^40; the largest logit has w = 2^40);
//   candidates: w_i >= wmin (min-p; wmin >= 1, so tokens below 2^-40 of the mode are not candidates);
//   top-k: keep keys >= the k-th largest candidate key (ties kept); top-p: keep the smallest set {key >= t} whose weight
//   reaches ceil(W * p24 / 2^24) of the weight W left by top-k (ties kept, at least the mode);
//   the token is drawn with probability w_i / W_kept, exactly: R = uniform integer in [0, W_kept) (Philox4x32-10 keyed
//   by the seed, counter = (position, "SAMP", draw, 0); Lemire's unbiased multiply-and-reject), and the token is the
//   kept entry whose cumulative weight interval holds R, entries in the fixed order (i mod 1024, i / 1024).
// Every sum is an integer sum, so the result does not depend on thread order, batch or launch: a row's token is a
// function of (logits, parameters, seed, position) only. Greedy rows (temperature < 1e-5) take the argmax.
// Thresholds by radix descent on the key (exponent field, then mantissa bits 22-15, 14-7, 6-0). One full pass counts and
// weighs the candidates and collects those with e >= 2^-14 into a shared list (warp-aggregated append); when the list
// holds the threshold (the usual case: its weight reaches the target) the descent reads only the list. Otherwise the
// exponent is found by binary search with full passes (block sums, no atomics) and that exponent's candidates are
// collected. Full passes: argmax, candidates, draw (+ about 8 when the threshold lies below 2^-14).
#pragma once
#include <cstdint>

#include "sample.hpp"

namespace ink {
namespace {

constexpr int kSampT = 1024;
constexpr unsigned kSampPurpose = 0x53414d50u;         // "SAMP"
constexpr int kSampListCap = 8192;

struct SampShared {
  unsigned long long hist[256];
  unsigned long long red[32];
  unsigned list[kSampListCap];
  float sv[32];
  int si[32];
  int nlist, sel, tok;
  unsigned long long above, last;
};

// Philox4x32-10 (Salmon et al., SC 2011; Random123 constants and round order)
__device__ __forceinline__ uint4 samp_philox(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const unsigned lo0 = 0xD2511F53u * c.x, hi0 = __umulhi(0xD2511F53u, c.x);
    const unsigned lo1 = 0xCD9E8D57u * c.z, hi1 = __umulhi(0xCD9E8D57u, c.z);
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

// uniform integer in [0, W), W >= 1: Lemire's multiply with rejection over the 64-bit halves of successive draws
__device__ __forceinline__ unsigned long long samp_uniform(unsigned long long W, unsigned long long seed, unsigned pos) {
  const uint2 key = make_uint2((unsigned)seed, (unsigned)(seed >> 32));
  const unsigned long long t = (0ull - W) % W;
  for (unsigned it = 0;; ++it) {
    const uint4 r = samp_philox(make_uint4(pos, kSampPurpose, it, 0u), key);
    const unsigned long long x[2] = {r.x | ((unsigned long long)r.y << 32), r.z | ((unsigned long long)r.w << 32)};
#pragma unroll
    for (int h = 0; h < 2; ++h)
      if (x[h] * W >= t) return __umul64hi(x[h], W);
  }
}

__device__ __forceinline__ unsigned long long samp_w(unsigned key) {
  return (unsigned long long)__fmul_rn(__uint_as_float(key), 1099511627776.0f);   // exact: e * 2^40, truncated
}

// candidate key of logit v (0: not a candidate)
// iT = 1 / temperature (__frcp_rn; exactly 1 at temperature 1); e = __expf (ex2.approx: deterministic on sm_80,
// relative error ~1e-6 against exp at the arguments that matter, |z| <= 27.7)
__device__ __forceinline__ unsigned samp_key(float v, float M, float iT, unsigned long long wmin) {
  if (!(v == v)) return 0u;
  const unsigned k = __float_as_uint(__expf(__fmul_rn(__fsub_rn(v, M), iT)));
  return samp_w(k) >= wmin ? k : 0u;
}

__device__ __forceinline__ unsigned long long samp_mix(unsigned long long z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

__device__ unsigned long long samp_sum(unsigned long long v, SampShared& s) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) s.red[threadIdx.x >> 5] = v;
  __syncthreads();
  unsigned long long t = 0;
  for (int w = 0; w < kSampT / 32; ++w) t += s.red[w];
  return t;
}

// exclusive prefix sum over the block's threads; *total = the sum
__device__ unsigned long long samp_scan(unsigned long long v, SampShared& s, unsigned long long* total) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  unsigned long long inc = v;
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long n = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += n;
  }
  __syncthreads();
  if (lane == 31) s.red[w] = inc;
  __syncthreads();
  if (w == 0) {
    unsigned long long x = s.red[lane];
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long n = __shfl_up_sync(0xffffffffu, x, o);
      if (lane >= o) x += n;
    }
    s.red[lane] = x;
  }
  __syncthreads();
  *total = s.red[31];
  return (w ? s.red[w - 1] : 0ull) + inc - v;
}

struct SampRow {
  const float* x;
  int V;
  float M, T;   // T: 1 / temperature
  unsigned long long wmin;
};

// the shared list holds every candidate with lo <= key < hi (unless full: more than the capacity)
struct SampList {
  unsigned lo, hi;
  int n;
  bool full;
  __device__ bool covers(unsigned x, unsigned y) const { return !full && lo <= x && y <= hi; }
};

constexpr unsigned kSampKeyEnd = 0xffffffffu;   // above every key (keys are positive floats <= 1)
constexpr unsigned kSampListKey = 113u << 23;    // the first list: e >= 2^-14

// f(key) for every candidate with x <= key < y: from the list when it holds them, else from the row
template <class F>
__device__ __forceinline__ void samp_range_each(const SampRow& q, const SampList& L, const SampShared& s, unsigned x,
                                                unsigned y, F f) {
  if (L.covers(x, y)) {
    for (int i = threadIdx.x; i < L.n; i += kSampT) {
      const unsigned k = s.list[i];
      if (k >= x && k < y) f(k);
    }
  } else {
    for (int c = threadIdx.x; c < q.V; c += kSampT) {
      const unsigned k = samp_key(q.x[c], q.M, q.T, q.wmin);
      if (k && k >= x && k < y) f(k);
    }
  }
}

// measure (count, or weight when by_mass) of the candidates with key >= x, on every thread
__device__ unsigned long long samp_measure(SampShared& s, const SampRow& q, const SampList& L, bool by_mass, unsigned x) {
  unsigned long long v = 0;
  samp_range_each(q, L, s, x, kSampKeyEnd, [&](unsigned k) { v += by_mass ? samp_w(k) : 1ull; });
  return samp_sum(v, s);
}

// one full pass: the list := the candidates with lo <= key < hi (warp-aggregated append)
__device__ void samp_collect(SampShared& s, const SampRow& q, unsigned lo, unsigned hi, SampList* L, unsigned* flags) {
  __syncthreads();
  if (threadIdx.x == 0) s.nlist = 0;
  __syncthreads();
  for (int b0 = 0; b0 < q.V; b0 += kSampT) {
    const int c = b0 + threadIdx.x;
    const unsigned k = c < q.V ? samp_key(q.x[c], q.M, q.T, q.wmin) : 0u;
    const bool in = k && k >= lo && k < hi;
    const unsigned bal = __ballot_sync(0xffffffffu, in);
    if (bal) {
      const int lane = threadIdx.x & 31, lead = __ffs(bal) - 1;
      int base = 0;
      if (lane == lead) base = atomicAdd(&s.nlist, __popc(bal));
      base = __shfl_sync(0xffffffffu, base, lead);
      const int i = base + __popc(bal & ((1u << lane) - 1));
      if (in && i < kSampListCap) s.list[i] = k;
    }
  }
  __syncthreads();
  L->lo = lo;
  L->hi = hi;
  L->n = min(s.nlist, kSampListCap);
  L->full = s.nlist > kSampListCap;
  if (L->full) *flags |= kSampFlagOverflow;
  __syncthreads();
}

// thread 0: the highest digit d of hist[0..n) with above + sum(hist[d..n)) >= target; s.sel, s.above (= above +
// sum(hist(d..n))), s.last = hist[d]; s.sel = -1 when none (s.above then holds the whole sum)
__device__ void samp_pick(SampShared& s, int lo, int n, unsigned long long above, unsigned long long target) {
  if (threadIdx.x == 0) {
    unsigned long long acc = above;
    int sel = -1;
    for (int d = n - 1; d >= lo; --d) {
      if (acc + s.hist[d] >= target) { sel = d; break; }
      acc += s.hist[d];
    }
    s.sel = sel;
    s.above = acc;
    s.last = sel >= 0 ? s.hist[sel] : 0ull;
  }
  __syncthreads();
}

// The largest key t with measure{candidates: key >= max(t, floor)} >= target (count, or weight when by_mass): radix
// descent over the key. Exponent field: a histogram of the list when the list (every candidate above its lo) already
// holds the target, else a binary search over the exponent with one full pass (block sum, no atomics) per step.
// Mantissa (bits 22-15, 14-7, 6-0): histograms of the chosen exponent's candidates (collected into the list when it does
// not hold them). *kept = measure{key >= t}. Needs measure{key >= floor} >= target >= 1.
__device__ unsigned samp_select(SampShared& s, const SampRow& q, bool by_mass, unsigned long long target, unsigned floor,
                                SampList* L, unsigned long long* kept, unsigned* flags) {
  auto meas = [&](unsigned k) { return by_mass ? samp_w(k) : 1ull; };
  int bin;
  unsigned long long above;
  const unsigned x0 = max(floor, L->lo);
  if (L->covers(x0, kSampKeyEnd) && samp_measure(s, q, *L, by_mass, x0) >= target) {
    for (int i = threadIdx.x; i < 256; i += kSampT) s.hist[i] = 0;
    __syncthreads();
    samp_range_each(q, *L, s, x0, kSampKeyEnd, [&](unsigned k) { atomicAdd(&s.hist[k >> 23], meas(k)); });
    __syncthreads();
    samp_pick(s, 0, 256, 0ull, target);
    bin = max(0, s.sel);
    above = s.above;
  } else {
    *flags |= kSampFlagLump;
    int lo = (int)(floor >> 23), hi = 128;   // measure at lo's exponent >= target, at hi's < target
    unsigned long long g_hi = 0;
    while (hi - lo > 1) {
      const int m = (lo + hi) / 2;
      const unsigned long long g = samp_measure(s, q, *L, by_mass, max(floor, (unsigned)m << 23));
      if (g >= target) lo = m;
      else { hi = m; g_hi = g; }
    }
    bin = lo;
    above = g_hi;   // measure{key >= (bin + 1) << 23}
  }
  const unsigned blo = max(floor, (unsigned)bin << 23), bhi = (unsigned)(bin + 1) << 23;
  if (!L->covers(blo, bhi)) samp_collect(s, q, (unsigned)bin << 23, bhi, L, flags);
  unsigned prefix = (unsigned)bin;
  const int shifts[3] = {15, 7, 0}, widths[3] = {8, 8, 7};
#pragma unroll
  for (int ps = 0; ps < 3; ++ps) {
    const int sh = shifts[ps], wd = widths[ps];
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += kSampT) s.hist[i] = 0;
    __syncthreads();
    samp_range_each(q, *L, s, blo, bhi, [&](unsigned k) {
      if ((k >> (sh + wd)) == prefix) atomicAdd(&s.hist[(k >> sh) & ((1u << wd) - 1)], meas(k));
    });
    __syncthreads();
    samp_pick(s, 0, 1 << wd, above, target);
    prefix = (prefix << wd) | (unsigned)max(0, s.sel);
    above = s.above;
  }
  *kept = above + s.last;
  return prefix;
}

__global__ void __launch_bounds__(kSampT) sample_rows_kernel(SampleArgs a) {
  __shared__ SampShared s;
  const int r = blockIdx.x, t = threadIdx.x;
  int lr, pi, pos;
  if (a.lrow) {
    const int c = a.lrow[r];
    lr = r;
    pi = a.call_seg[c];
    pos = a.call_pos[c] + 1;
  } else {
    lr = a.t_lrow[r];
    pi = a.t_par[r];
    pos = a.t_pos[r];
  }
  const SampleParam p = a.par[pi];
  const float* x = a.lg + (size_t)lr * a.ld;
  const int V = a.vocab;
  // the argmax, as argmax_rows_kernel
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int c = t; c < V; c += kSampT) {
    const float v = x[c];
    if (v > bv || (v == bv && c < bi)) { bv = v; bi = c; }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if ((t & 31) == 0) { s.sv[t >> 5] = bv; s.si[t >> 5] = bi; }
  __syncthreads();
  if (t == 0) {
    for (int w = 1; w < kSampT / 32; ++w)
      if (s.sv[w] > bv || (s.sv[w] == bv && s.si[w] < bi)) { bv = s.sv[w]; bi = s.si[w]; }
    s.sv[0] = bv;
    s.si[0] = bi;
    s.nlist = 0;
  }
  __syncthreads();
  bv = s.sv[0];
  bi = s.si[0];
  const int amax = bi == 0x7fffffff ? 0 : bi;
  unsigned flags = 0;
  if (p.greedy || bi == 0x7fffffff) {
    if (t == 0) {
      a.out[r] = amax;
      if (a.stats) {
        unsigned long long* st = a.stats + (size_t)r * 8;
        st[0] = __float_as_uint(bv) | ((unsigned long long)amax << 32);
        for (int i = 1; i < 8; ++i) st[i] = 0;
        st[6] = amax;
        st[7] = kSampFlagGreedy;
      }
    }
    return;
  }
  const SampRow q{x, V, bv, p.temperature == 1.0f ? 1.0f : __frcp_rn(p.temperature), p.wmin};
  // candidates: count and weight; the list of those with e >= 2^-14
  unsigned long long ac = 0, am = 0;
  if (t == 0) s.nlist = 0;
  __syncthreads();
  for (int b0 = 0; b0 < V; b0 += kSampT) {
    const int c = b0 + t;
    const unsigned k = c < V ? samp_key(x[c], q.M, q.T, q.wmin) : 0u;
    if (k) { ++ac; am += samp_w(k); }
    const bool in = k >= kSampListKey;
    const unsigned bal = __ballot_sync(0xffffffffu, in);
    if (bal) {
      const int lane = t & 31, lead = __ffs(bal) - 1;
      int base = 0;
      if (lane == lead) base = atomicAdd(&s.nlist, __popc(bal));
      base = __shfl_sync(0xffffffffu, base, lead);
      const int i = base + __popc(bal & ((1u << lane) - 1));
      if (in && i < kSampListCap) s.list[i] = k;
    }
  }
  const unsigned long long cnt_all = samp_sum(ac, s), mass_all = samp_sum(am, s);
  SampList L{kSampListKey, kSampKeyEnd, min(s.nlist, kSampListCap), s.nlist > kSampListCap};
  if (L.full) flags |= kSampFlagOverflow;
  unsigned tau = 1;
  unsigned long long W = mass_all, kept = 0;
  if (p.top_k > 0 && cnt_all > (unsigned long long)p.top_k) {
    tau = samp_select(s, q, false, (unsigned long long)p.top_k, 1u, &L, &kept, &flags);
    W = samp_measure(s, q, L, true, tau);
  }
  if (p.p24 < (1u << 24)) {
    const unsigned long long hi = __umul64hi(W, (unsigned long long)p.p24), lo = W * (unsigned long long)p.p24;
    unsigned long long target = (hi << 40) | (lo >> 24);
    target += (lo & 0xffffffull) != 0;
    target = max(target, 1ull);
    tau = samp_select(s, q, true, target, tau, &L, &kept, &flags);
    W = kept;
  }
  const unsigned long long R = samp_uniform(W, p.seed, (unsigned)pos);
  // the draw: kept weight per thread (entries c = t + 1024 j), exclusive scan, the owner of R walks its entries
  unsigned long long mine = 0, nk = 0, hs = 0;
  for (int c = t; c < V; c += kSampT) {
    const unsigned k = samp_key(x[c], q.M, q.T, q.wmin);
    if (a.keys) a.keys[(size_t)r * V + c] = k;
    if (k >= tau && k) {
      const unsigned long long w = samp_w(k);
      mine += w;
      ++nk;
      hs += samp_mix(((unsigned long long)c << 41) | w);
    }
  }
  unsigned long long total = 0;
  const unsigned long long ex = samp_scan(mine, s, &total);
  if (t == 0) s.tok = -1;
  __syncthreads();
  if (total == W && mine && R >= ex && R - ex < mine) {
    unsigned long long acc = ex;
    for (int c = t; c < V; c += kSampT) {
      const unsigned k = samp_key(x[c], q.M, q.T, q.wmin);
      if (k >= tau && k) {
        const unsigned long long w = samp_w(k);
        if (R - acc < w) { s.tok = c; break; }
        acc += w;
      }
    }
  }
  __syncthreads();
  const int tok = s.tok >= 0 ? s.tok : amax;
  if (total != W || s.tok < 0) flags |= kSampFlagMismatch;
  const unsigned long long nkept = samp_sum(nk, s), hash = samp_sum(hs, s);
  if (t == 0) {
    a.out[r] = tok;
    if (a.stats) {
      unsigned long long* st = a.stats + (size_t)r * 8;
      st[0] = __float_as_uint(bv) | ((unsigned long long)amax << 32);
      st[1] = tau;
      st[2] = W;
      st[3] = nkept;
      st[4] = hash;
      st[5] = R;
      st[6] = (unsigned)tok;
      st[7] = flags;
    }
  }
}

// Exact 128-bit unsigned arithmetic for lossless Leviathan rejection + residual sampling on W * 2^22 scale.
constexpr unsigned long long kLevQtot = 1ull << 22;
constexpr unsigned kLevPurposeRej = 0x52454a54u;   // "REJT"
constexpr unsigned kLevPurposeRes = 0x52455344u;   // "RESD"
constexpr int kLevTopK = 16;

struct SampU128 {
  unsigned long long lo, hi;
};

__device__ __forceinline__ SampU128 u128_add(SampU128 a, SampU128 b) {
  unsigned long long lo = a.lo + b.lo;
  return SampU128{lo, a.hi + b.hi + (lo < a.lo ? 1ull : 0ull)};
}

__device__ __forceinline__ SampU128 u128_sub(SampU128 a, SampU128 b) {
  unsigned long long lo = a.lo - b.lo;
  return SampU128{lo, a.hi - b.hi - (a.lo < b.lo ? 1ull : 0ull)};
}

__device__ __forceinline__ bool u128_lt(SampU128 a, SampU128 b) {
  return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

__device__ __forceinline__ SampU128 u128_mul_u64_u32(unsigned long long a, unsigned int b) {
  return SampU128{a * (unsigned long long)b, __umul64hi(a, (unsigned long long)b)};
}

// Unbiased uniform 128-bit integer in [0, M), M >= 1, via bit-width mask + rejection over Philox4x32-10.
__device__ __forceinline__ SampU128 samp_uniform_u128(SampU128 M, unsigned long long seed, unsigned pos,
                                                      unsigned purpose) {
  if (M.hi == 0ull && M.lo <= 1ull) return SampU128{0ull, 0ull};
  const SampU128 m1 = u128_sub(M, SampU128{1ull, 0ull});
  const int nb = m1.hi ? (64 + (64 - __clzll((long long)m1.hi))) : (64 - __clzll((long long)m1.lo));
  const uint2 key = make_uint2((unsigned)seed, (unsigned)(seed >> 32));
  for (unsigned it = 0;; ++it) {
    const uint4 r = samp_philox(make_uint4(pos, purpose, it, 0u), key);
    unsigned long long xlo = r.x | ((unsigned long long)r.y << 32);
    unsigned long long xhi = r.z | ((unsigned long long)r.w << 32);
    if (nb <= 64) {
      xhi = 0ull;
      if (nb < 64) xlo &= (1ull << nb) - 1ull;
    } else {
      const int hb = nb - 64;
      if (hb < 64) xhi &= (1ull << hb) - 1ull;
    }
    const SampU128 X{xlo, xhi};
    if (u128_lt(X, M)) return X;
  }
}

// Exclusive prefix sum over the block's threads in 128-bit unsigned arithmetic (reuses s.hist[0..63]).
__device__ SampU128 samp_scan_u128(SampU128 v, SampShared& s, SampU128* total) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  SampU128 inc = v;
  for (int o = 1; o < 32; o <<= 1) {
    const SampU128 n{__shfl_up_sync(0xffffffffu, inc.lo, o), __shfl_up_sync(0xffffffffu, inc.hi, o)};
    if (lane >= o) inc = u128_add(inc, n);
  }
  __syncthreads();
  if (lane == 31) {
    s.hist[w] = inc.lo;
    s.hist[32 + w] = inc.hi;
  }
  __syncthreads();
  if (w == 0) {
    SampU128 x{s.hist[lane], s.hist[32 + lane]};
    for (int o = 1; o < 32; o <<= 1) {
      const SampU128 n{__shfl_up_sync(0xffffffffu, x.lo, o), __shfl_up_sync(0xffffffffu, x.hi, o)};
      if (lane >= o) x = u128_add(x, n);
    }
    s.hist[lane] = x.lo;
    s.hist[32 + lane] = x.hi;
  }
  __syncthreads();
  *total = SampU128{s.hist[31], s.hist[63]};
  const SampU128 base = w ? SampU128{s.hist[w - 1], s.hist[32 + w - 1]} : SampU128{0ull, 0ull};
  return u128_sub(u128_add(base, inc), v);
}

// Exact Leviathan rejection + residual sampling over a sparse 16-candidate proposal distribution (sum q = 2^22).
// If accepted, writes out[r] = draft d; if rejected, draws from max(0, p - q) which has weight 0 at d (so out[r] != d).
// Hence accept_kernel's `out[r0 + j] == drafts[slot * 7 + j]` check is exact with zero modification.
__global__ void __launch_bounds__(kSampT) sample_leviathan_kernel(LeviathanArgs a) {
  __shared__ SampShared s;
  __shared__ int s_pid[kLevTopK];
  __shared__ unsigned int s_pq[kLevTopK];
  __shared__ unsigned long long s_qlo[kLevTopK];
  __shared__ unsigned long long s_qhi[kLevTopK];
  __shared__ unsigned long long s_rlo, s_rhi;
  __shared__ int s_accept;

  const int r = blockIdx.x, t = threadIdx.x;
  const int c_row = a.lrow[r];
  const int pi = a.call_seg[c_row];
  const int pos = a.call_pos[c_row] + 1;
  const SampleParam p = a.par[pi];
  const float* x = a.lg + (size_t)r * a.ld;
  const int V = a.vocab;

  // 1. Argmax (identical to sample_rows_kernel).
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int c = t; c < V; c += kSampT) {
    const float v = x[c];
    if (v > bv || (v == bv && c < bi)) { bv = v; bi = c; }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if ((t & 31) == 0) { s.sv[t >> 5] = bv; s.si[t >> 5] = bi; }
  __syncthreads();
  if (t == 0) {
    for (int w = 1; w < kSampT / 32; ++w)
      if (s.sv[w] > bv || (s.sv[w] == bv && s.si[w] < bi)) { bv = s.sv[w]; bi = s.si[w]; }
    s.sv[0] = bv;
    s.si[0] = bi;
    s.nlist = 0;
  }
  __syncthreads();
  bv = s.sv[0];
  bi = s.si[0];
  const int amax = bi == 0x7fffffff ? 0 : bi;
  unsigned flags = 0;
  if (p.greedy || bi == 0x7fffffff) {
    if (t == 0) a.out[r] = amax;
    return;
  }

  // 2. Exact target filtered distribution (tau, W), identical to sample_rows_kernel.
  const SampRow q{x, V, bv, p.temperature == 1.0f ? 1.0f : __frcp_rn(p.temperature), p.wmin};
  unsigned long long ac = 0, am = 0;
  if (t == 0) s.nlist = 0;
  __syncthreads();
  for (int b0 = 0; b0 < V; b0 += kSampT) {
    const int c = b0 + t;
    const unsigned k = c < V ? samp_key(x[c], q.M, q.T, q.wmin) : 0u;
    if (k) { ++ac; am += samp_w(k); }
    const bool in = k >= kSampListKey;
    const unsigned bal = __ballot_sync(0xffffffffu, in);
    if (bal) {
      const int lane = t & 31, lead = __ffs(bal) - 1;
      int base = 0;
      if (lane == lead) base = atomicAdd(&s.nlist, __popc(bal));
      base = __shfl_sync(0xffffffffu, base, lead);
      const int i = base + __popc(bal & ((1u << lane) - 1));
      if (in && i < kSampListCap) s.list[i] = k;
    }
  }
  const unsigned long long cnt_all = samp_sum(ac, s), mass_all = samp_sum(am, s);
  SampList L{kSampListKey, kSampKeyEnd, min(s.nlist, kSampListCap), s.nlist > kSampListCap};
  unsigned tau = 1;
  unsigned long long W = mass_all, kept = 0;
  if (p.top_k > 0 && cnt_all > (unsigned long long)p.top_k) {
    tau = samp_select(s, q, false, (unsigned long long)p.top_k, 1u, &L, &kept, &flags);
    W = samp_measure(s, q, L, true, tau);
  }
  if (p.p24 < (1u << 24)) {
    const unsigned long long hi = __umul64hi(W, (unsigned long long)p.p24), lo = W * (unsigned long long)p.p24;
    unsigned long long target = (hi << 40) | (lo >> 24);
    target += (lo & 0xffffffull) != 0;
    target = max(target, 1ull);
    tau = samp_select(s, q, true, target, tau, &L, &kept, &flags);
    W = kept;
  }

  // 3. Determine if this row verifies a draft token (j < k) or is the bonus row (j == k).
  const int r0 = a.call_row0[pi];
  const int slot = a.call_slot[pi];
  const int k_draft = (pi + 1 < a.nseg ? a.call_row0[pi + 1] : a.T) - r0 - 1;
  const int j = c_row - r0;
  const int d = (j >= 0 && j < k_draft && j < 7) ? a.drafts[slot * 7 + j] : -1;

  if (d < 0 || d >= V) {
    // Bonus row: plain exact draw from p.
    const unsigned long long R = samp_uniform(W, p.seed, (unsigned)pos);
    unsigned long long mine = 0;
    for (int c = t; c < V; c += kSampT) {
      const unsigned k = samp_key(x[c], q.M, q.T, q.wmin);
      if (k >= tau && k) mine += samp_w(k);
    }
    unsigned long long total = 0;
    const unsigned long long ex = samp_scan(mine, s, &total);
    if (t == 0) s.tok = -1;
    __syncthreads();
    if (total == W && mine && R >= ex && R - ex < mine) {
      unsigned long long acc = ex;
      for (int c = t; c < V; c += kSampT) {
        const unsigned k = samp_key(x[c], q.M, q.T, q.wmin);
        if (k >= tau && k) {
          const unsigned long long w = samp_w(k);
          if (R - acc < w) { s.tok = c; break; }
          acc += w;
        }
      }
    }
    __syncthreads();
    if (t == 0) a.out[r] = s.tok >= 0 ? s.tok : amax;
    return;
  }

  // 4. Load the 16 sparse proposal pairs (pid, pq) and test exact acceptance u < p(d) / q(d).
  if (t < kLevTopK) {
    const size_t pbase = ((size_t)slot * 7 + (size_t)j) * kLevTopK + (size_t)t;
    s_pid[t] = a.prop_ids[pbase];
    const unsigned int qw = a.prop_w[pbase];
    s_pq[t] = qw;
    const SampU128 qs = u128_mul_u64_u32(W, qw);
    s_qlo[t] = qs.lo;
    s_qhi[t] = qs.hi;
  }
  __syncthreads();

  if (t == 0) {
    unsigned int q_d = 0u;
    for (int m = 0; m < kLevTopK; ++m) {
      if (s_pid[m] == d) { q_d = s_pq[m]; break; }
    }
    if (q_d == 0u) {
      s_accept = -1;  // malformed proposal fallback
    } else {
      const unsigned kd = samp_key(x[d], q.M, q.T, q.wmin);
      const unsigned long long wd = (kd && kd >= tau) ? samp_w(kd) : 0ull;
      const SampU128 p_d{wd * kLevQtot, 0ull};
      const SampU128 q_ds = u128_mul_u64_u32(W, q_d);
      if (!u128_lt(p_d, q_ds)) {
        s_accept = 1;  // p(d) >= q(d): accept with prob 1
      } else if (wd == 0ull) {
        s_accept = 0;  // p(d) == 0: reject with prob 1
      } else {
        const SampU128 U = samp_uniform_u128(q_ds, p.seed, (unsigned)pos, kLevPurposeRej);
        s_accept = u128_lt(U, p_d) ? 1 : 0;
      }
    }
  }
  __syncthreads();

  if (s_accept == 1) {
    if (t == 0) a.out[r] = d;
    return;
  }

  // 5. Rejected (p(d) < q(d)): draw exact correction token from residual max(0, p - q) on W * 2^22 scale.
  auto res_weight = [&](int c) -> unsigned long long {
    const unsigned k = samp_key(x[c], q.M, q.T, q.wmin);
    if (!(k && k >= tau)) return 0ull;
    const unsigned long long pc = samp_w(k) * kLevQtot;
    if (s_accept == -1) return pc;
#pragma unroll
    for (int m = 0; m < kLevTopK; ++m) {
      if (c == s_pid[m]) {
        return (s_qhi[m] == 0ull && pc > s_qlo[m]) ? (pc - s_qlo[m]) : 0ull;
      }
    }
    return pc;
  };

  SampU128 mine{0ull, 0ull};
  for (int c = t; c < V; c += kSampT) {
    const unsigned long long rw = res_weight(c);
    if (rw) mine = u128_add(mine, SampU128{rw, 0ull});
  }
  SampU128 total{0ull, 0ull};
  const SampU128 ex = samp_scan_u128(mine, s, &total);
  if (t == 0) {
    s.tok = -1;
    if (total.hi != 0ull || total.lo != 0ull) {
      const SampU128 R = samp_uniform_u128(total, p.seed, (unsigned)pos, kLevPurposeRes);
      s_rlo = R.lo;
      s_rhi = R.hi;
    } else {
      s_rlo = 0ull;
      s_rhi = 0ull;
    }
  }
  __syncthreads();
  const SampU128 R{s_rlo, s_rhi};
  const SampU128 ex_end = u128_add(ex, mine);
  if ((mine.hi != 0ull || mine.lo != 0ull) && !u128_lt(R, ex) && u128_lt(R, ex_end)) {
    SampU128 acc = ex;
    for (int c = t; c < V; c += kSampT) {
      const unsigned long long rw = res_weight(c);
      if (rw) {
        const SampU128 next = u128_add(acc, SampU128{rw, 0ull});
        if (u128_lt(R, next)) { s.tok = c; break; }
        acc = next;
      }
    }
  }
  __syncthreads();
  if (t == 0) a.out[r] = s.tok >= 0 ? s.tok : amax;
}

}  // namespace
}  // namespace ink
