// S4 prefill attention: standalone C++/CUDA bench and correctness harness (our code). No Python engine dependency.
//
// Modes (JSON lines on stdout, first line = meta):
//   info   registers / smem / occupancy per kernel variant
//   gen    write the synthetic data set to --dir (read by cfgb_prefill.py, so config B runs on the same values)
//   check  FP64 oracle on sampled rows at real shapes, for our kernel and (if --cfgb-dir has its outputs) config B's
//   inv    invariance: chunking, batch composition, page placement / layout, repeat determinism (bit compare)
//   time   sustained timing (>= 20 s per arm, first 2 s dropped, 10 Hz NVML, energy counter), variants x cells
//
// Data (synthetic, realistic statistics; real KV was not captured, see README): keys/values by absolute position
// (K = norm-clipped at 22.627 with per-channel scales and per-head means, V = RMS-normed; bf16), query-side rows by
// position % QMAX (q norm U(15, 22.6) global / U(8, 16.6) SWA, real-KV medians 20.0 / 12.3 in P6 F4; rel logits
// N(0, 0.8); gate N(0, 1.5); sink N(1, 2)), all bf16.
#include "harness.hpp"
#include "prefill_attn.cuh"

#include <cuda_bf16.h>

#include <map>
#include <numeric>
#include <random>

using p6::Line;
static const char* PROBE = "s4_prefill_attn";
#define CK(x) P6_CK(PROBE, x)

constexpr int QMAX = 8192;
constexpr int KVROW = pfa::kKvHeads * pfa::kD;  // 1024 elements per position per matrix

// ------------------------------------------------------------------------------------------------ generators
__host__ __device__ inline float gauss(uint64_t h) {
  const float u1 = p6::u01(h), u2 = p6::u01(p6::splitmix64(h ^ 0x5bd1e995ull));
  return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
__device__ inline uint16_t f2bf(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }

// one warp per (position, KV head): same statistics as probes/p6-attn/kvgen.cuh
__global__ void gen_kv(uint16_t* K, uint16_t* V, long long npos, uint64_t seed) {
  const long long wid = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (wid >= npos * pfa::kKvHeads) return;
  const int head = (int)(wid % pfa::kKvHeads);
  const uint64_t hk = p6::splitmix64(seed ^ (uint64_t)wid * 0x9E3779B97F4A7C15ull);
  float k[4], v[4], nk = 0.f, nv = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = lane * 4 + i;
    const uint64_t ch = p6::splitmix64(0xC0FFEEull ^ ((uint64_t)head << 8) ^ (uint64_t)c);
    const float sig = expf(0.5f * gauss(ch)) * (p6::u01(p6::splitmix64(ch)) < 0.03f ? 4.0f : 1.0f);
    const float mu = 0.3f * sig * gauss(p6::splitmix64(ch ^ 1));
    k[i] = mu + sig * gauss(p6::splitmix64(hk ^ (uint64_t)(2 * c)));
    v[i] = sig * gauss(p6::splitmix64(hk ^ (uint64_t)(2 * c + 1)));
    nk += k[i] * k[i];
    nv += v[i] * v[i];
  }
#pragma unroll
  for (int sh = 16; sh; sh >>= 1) { nk += __shfl_xor_sync(0xffffffffu, nk, sh); nv += __shfl_xor_sync(0xffffffffu, nv, sh); }
  const float target = 22.627f * (0.6f + 0.6f * p6::u01(p6::splitmix64(hk ^ 0xABCDull)));
  const float ks = (target / (sqrtf(nk) + 1e-6f)) * fminf(1.0f, 22.627f / (target + 1e-6f));
  const float vs = rsqrtf(nv / 128.0f + 1e-6f);
  const size_t base = (size_t)wid * pfa::kD + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) { K[base + i] = f2bf(k[i] * ks); V[base + i] = f2bf(v[i] * vs); }
}
// one warp per (row, head): random direction with the key channel scales, norm U(nlo, nhi)
__global__ void gen_q(uint16_t* Q, int rows, float nlo, float nhi, uint64_t seed) {
  const long long wid = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (wid >= (long long)rows * pfa::kHeads) return;
  const int head = (int)(wid % pfa::kHeads), kvh = head / pfa::kGqa;
  const uint64_t hq = p6::splitmix64(seed ^ (uint64_t)wid * 0xD1B54A32D192ED03ull);
  float x[4], n = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = lane * 4 + i;
    const uint64_t ch = p6::splitmix64(0xC0FFEEull ^ ((uint64_t)kvh << 8) ^ (uint64_t)c);
    const float sig = expf(0.5f * gauss(ch)) * (p6::u01(p6::splitmix64(ch)) < 0.03f ? 4.0f : 1.0f);
    const float mu = 0.3f * sig * gauss(p6::splitmix64(ch ^ 1));
    x[i] = 0.5f * mu + sig * gauss(p6::splitmix64(hq ^ (uint64_t)c));
    n += x[i] * x[i];
  }
#pragma unroll
  for (int sh = 16; sh; sh >>= 1) n += __shfl_xor_sync(0xffffffffu, n, sh);
  const float target = nlo + (nhi - nlo) * p6::u01(p6::splitmix64(hq ^ 0x77ull));
  const float s = target / (sqrtf(n) + 1e-6f);
  const size_t base = (size_t)wid * pfa::kD + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) Q[base + i] = f2bf(x[i] * s);
}
__global__ void gen_gauss(uint16_t* x, long long n, float mu, float sigma, uint64_t seed) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    x[i] = f2bf(mu + sigma * gauss(p6::splitmix64(seed ^ (uint64_t)i * 0x9E3779B97F4A7C15ull)));
}
// logical [pos][8][128] -> pool through the page table; one thread per 16-byte chunk
__global__ void scatter_pool(const uint4* Kl, const uint4* Vl, uint16_t* Kp, uint16_t* Vp, const int* pt, int shift,
                             long long sp, long long ss, long long sh, long long npos) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < npos * 128; i += (long long)gridDim.x * blockDim.x) {
    const long long pos = i >> 7;
    const int h = (int)((i >> 4) & 7), c = (int)(i & 15);
    const long long off = (long long)pt[pos >> shift] * sp + (pos & ((1 << shift) - 1)) * ss + h * sh + c * 8;
    *reinterpret_cast<uint4*>(Kp + off) = Kl[i];
    *reinterpret_cast<uint4*>(Vp + off) = Vl[i];
  }
}
// rows (pos % QMAX) of a [QMAX][w] table -> [rows][w] (w in 16-bit elements, multiple of 8)
__global__ void gather_rows(const uint16_t* src, uint16_t* dst, int w, int pos0, int rows) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const uint4* s = reinterpret_cast<const uint4*>(src + (size_t)((pos0 + r) % QMAX) * w);
  uint4* d = reinterpret_cast<uint4*>(dst + (size_t)r * w);
  for (int i = threadIdx.x; i < w / 8; i += blockDim.x) d[i] = s[i];
}

// key-aligned bias rows of a segment: out[r][h][col] = REL[(pos % QMAX)][h][pos - key], key = (pos & ~63) - 1024 + col,
// 0 where the distance is outside [0, 1024) (the layout the kernel reads; same values as the by-distance table)
__global__ void build_relk(const uint16_t* REL, uint16_t* out, int pos0, int rows) {
  const int r = blockIdx.x, h = blockIdx.y;
  if (r >= rows) return;
  const int pos = pos0 + r, base = (pos & ~63) - pfa::kRelExt;
  const uint16_t* src = REL + ((size_t)(pos % QMAX) * pfa::kHeads + h) * pfa::kRelExt;
  uint16_t* dst = out + ((size_t)r * pfa::kHeads + h) * pfa::kRelW;
  for (int col = threadIdx.x; col < pfa::kRelW; col += blockDim.x) {
    const int d = pos - (base + col);
    dst[col] = (d >= 0 && d < pfa::kRelExt) ? src[d] : 0;
  }
}

// ------------------------------------------------------------------------------------------------ data set
struct Data {
  long long npos = 0;
  uint16_t *Kl = nullptr, *Vl = nullptr;                      // logical [npos][8][128]
  uint16_t *Qg = nullptr, *Ql = nullptr, *REL = nullptr, *GATE = nullptr, *SINK = nullptr;  // [QMAX][...]
  std::vector<uint16_t> hK, hV, hQg, hQl, hREL, hGATE, hSINK;  // host copies (oracle)
};
static void make_data(Data& d, long long npos, uint64_t seed, bool host) {
  d.npos = npos;
  CK(cudaMalloc(&d.Kl, npos * KVROW * 2));
  CK(cudaMalloc(&d.Vl, npos * KVROW * 2));
  const size_t qn = (size_t)QMAX * pfa::kHeads * pfa::kD, rn = (size_t)QMAX * pfa::kHeads * pfa::kRelExt;
  CK(cudaMalloc(&d.Qg, qn * 2));
  CK(cudaMalloc(&d.Ql, qn * 2));
  CK(cudaMalloc(&d.REL, rn * 2));
  CK(cudaMalloc(&d.GATE, qn * 2));
  CK(cudaMalloc(&d.SINK, (size_t)QMAX * pfa::kHeads * 2));
  gen_kv<<<(unsigned)((npos * 8 * 32 + 255) / 256), 256>>>(d.Kl, d.Vl, npos, seed);
  gen_q<<<(unsigned)((QMAX * 24 * 32 + 255) / 256), 256>>>(d.Qg, QMAX, 15.0f, 22.6f, seed ^ 0x1111);
  gen_q<<<(unsigned)((QMAX * 24 * 32 + 255) / 256), 256>>>(d.Ql, QMAX, 8.0f, 16.6f, seed ^ 0x2222);
  gen_gauss<<<4096, 256>>>(d.REL, rn, 0.0f, 0.8f, seed ^ 0x3333);
  gen_gauss<<<4096, 256>>>(d.GATE, qn, 0.0f, 1.5f, seed ^ 0x4444);
  gen_gauss<<<64, 256>>>(d.SINK, (long long)QMAX * 24, 1.0f, 2.0f, seed ^ 0x5555);
  CK(cudaDeviceSynchronize());
  if (host) {
    auto cp = [](std::vector<uint16_t>& h, const uint16_t* dp, size_t n) {
      h.resize(n);
      CK(cudaMemcpy(h.data(), dp, n * 2, cudaMemcpyDeviceToHost));
    };
    cp(d.hK, d.Kl, npos * KVROW);
    cp(d.hV, d.Vl, npos * KVROW);
    cp(d.hQg, d.Qg, qn);
    cp(d.hQl, d.Ql, qn);
    cp(d.hREL, d.REL, rn);
    cp(d.hGATE, d.GATE, qn);
    cp(d.hSINK, d.SINK, (size_t)QMAX * 24);
  }
}

// Paged pool holding the logical sequence. layout 0 = config B [page][slot][head][dim], 1 = [page][head][slot][dim].
struct Pool {
  int page = 64, shift = 6, layout = 0;
  long long sp, ss, sh, npages;
  uint16_t *K = nullptr, *V = nullptr;
  int* pt = nullptr;  // [npages], logical page -> physical page
  std::vector<int> hpt;
};
static void make_pool(Pool& P, const Data& d, int page, int layout, uint64_t seed) {
  P.page = page;
  P.shift = 0;
  while ((1 << P.shift) < page) ++P.shift;
  P.layout = layout;
  P.npages = (d.npos + page - 1) / page;
  if (layout == 0) { P.ss = KVROW; P.sh = pfa::kD; P.sp = (long long)page * KVROW; }
  else { P.sh = (long long)page * pfa::kD; P.ss = pfa::kD; P.sp = (long long)page * KVROW; }
  CK(cudaMalloc(&P.K, P.npages * P.sp * 2));
  CK(cudaMalloc(&P.V, P.npages * P.sp * 2));
  P.hpt.resize(P.npages);
  std::iota(P.hpt.begin(), P.hpt.end(), 0);
  std::mt19937_64 rng(seed);
  std::shuffle(P.hpt.begin(), P.hpt.end(), rng);
  CK(cudaMalloc(&P.pt, P.npages * sizeof(int)));
  CK(cudaMemcpy(P.pt, P.hpt.data(), P.npages * sizeof(int), cudaMemcpyHostToDevice));
  scatter_pool<<<8192, 256>>>(reinterpret_cast<const uint4*>(d.Kl), reinterpret_cast<const uint4*>(d.Vl), P.K, P.V, P.pt,
                              P.shift, P.sp, P.ss, P.sh, d.npos);
  CK(cudaDeviceSynchronize());
}
static void free_pool(Pool& P) { cudaFree(P.K); cudaFree(P.V); cudaFree(P.pt); P.K = P.V = nullptr; P.pt = nullptr; }

// ------------------------------------------------------------------------------------------------ one launch problem
struct SegSpec { int past, len; };
struct Problem {
  bool swa = false;
  std::vector<SegSpec> segs;
  int T = 0;
  uint16_t *q = nullptr, *rel = nullptr, *gate = nullptr, *sink = nullptr, *out = nullptr;
  float *o32 = nullptr, *lse = nullptr;
  pfa::Seg* dsegs = nullptr;
  int4 *dtiles = nullptr, *ditems2 = nullptr;
  float* ws = nullptr;
  int* cnt = nullptr;
  int ntiles = 0, br = 0, nrowtiles = 0, nparts = 0;
  size_t ws_bytes = 0;
  double flops = 0;  // P9 model: 4 * 24 * 128 * sum of visible keys per query row
};
static void build_problem(Problem& pr, const Data& d, const Pool& P, bool swa, const std::vector<SegSpec>& segs, bool dbg) {
  pr.swa = swa;
  pr.segs = segs;
  pr.T = 0;
  for (auto& s : segs) pr.T += s.len;
  const int T = pr.T;
  CK(cudaMalloc(&pr.q, (size_t)T * 3072 * 2));
  CK(cudaMalloc(&pr.rel, (size_t)T * 24 * pfa::kRelW * 2));
  CK(cudaMalloc(&pr.gate, (size_t)T * 3072 * 2));
  CK(cudaMalloc(&pr.sink, (size_t)T * 24 * 2 + 16));
  CK(cudaMalloc(&pr.out, (size_t)T * 3072 * 2));
  if (dbg) {
    CK(cudaMalloc(&pr.o32, (size_t)T * 3072 * 4));
    CK(cudaMalloc(&pr.lse, (size_t)T * 24 * 4));
  }
  std::vector<pfa::Seg> hs;
  int q0 = 0;
  pr.flops = 0;
  for (auto& s : segs) {
    gather_rows<<<s.len, 128>>>(swa ? d.Ql : d.Qg, pr.q + (size_t)q0 * 3072, 3072, s.past, s.len);
    build_relk<<<dim3(s.len, 24), 256>>>(d.REL, pr.rel + (size_t)q0 * 24 * pfa::kRelW, s.past, s.len);
    gather_rows<<<s.len, 128>>>(d.GATE, pr.gate + (size_t)q0 * 3072, 3072, s.past, s.len);
    for (int i = 0; i < s.len; ++i)  // sink rows are 24 wide (not a multiple of 8): copy row by row
      CK(cudaMemcpy(pr.sink + (size_t)(q0 + i) * 24, d.SINK + (size_t)((s.past + i) % QMAX) * 24, 48, cudaMemcpyDeviceToDevice));
    hs.push_back({q0, s.len, s.past, 0});
    for (int i = 0; i < s.len; ++i) {
      const long long pos = s.past + i;
      pr.flops += 4.0 * 24 * 128 * (double)(swa ? std::min<long long>(pos + 1, 1024) : pos + 1);
    }
    q0 += s.len;
  }
  CK(cudaMalloc(&pr.dsegs, hs.size() * sizeof(pfa::Seg)));
  CK(cudaMemcpy(pr.dsegs, hs.data(), hs.size() * sizeof(pfa::Seg), cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());
}
// work items for a row-tile size BR: each row tile's key blocks cut at the absolute split boundaries (every
// g_split keys, 0 = no split; a position-only rule, so a row's arithmetic does not depend on the launch), heaviest
// first (stable). Workspace partials are allocated for row tiles with more than one item.
static int g_split = 0;
static int g_order = 0;
static int g_geo = 0;   // geometric split boundaries  // 0 = key-range-major with splits, 1 = heaviest first always (ledger A/B)
static void build_tiles(Problem& pr, int br, size_t ws_per_part) {
  if (pr.br == br && pr.dtiles) return;
  if (pr.dtiles) { cudaFree(pr.dtiles); cudaFree(pr.ditems2); cudaFree(pr.ws); cudaFree(pr.cnt); }
  struct It { int4 a, b; long long w; };
  std::vector<It> v;
  int tile = 0, parts = 0;
  for (size_t si = 0; si < pr.segs.size(); ++si) {
    const auto& s = pr.segs[si];
    for (int r0 = 0; r0 < 3 * s.len; r0 += br, ++tile) {
      const int qmax = s.past + std::min(r0 + br - 1, 3 * s.len - 1) / 3, qmin = s.past + r0 / 3;
      const int b0 = (pr.swa ? std::max(0, qmin - 1023) : 0) / 64, b1 = qmax / 64 + 1;
      std::vector<int> cuts = {b0};
      if (g_split > 0) {
        // boundaries (in 64-key blocks): every sb; with g_geo, every sb below 8 sb, 2 sb below 16 sb, 4 sb beyond
        const int sb = g_split / 64;
        auto step = [&](int c) { return !g_geo ? sb : c < 8 * sb ? sb : c < 16 * sb ? 2 * sb : 4 * sb; };
        for (int c = 0; c < b1; c += step(c))
          if (c > b0) cuts.push_back(c);
      }
      cuts.push_back(b1);
      const int n = (int)cuts.size() - 1;
      for (int k = 0; k < n; ++k)
        v.push_back({make_int4((int)si, r0, cuts[k], cuts[k + 1]), make_int4(k, n, parts, tile), (long long)(cuts[k + 1] - cuts[k])});
      if (n > 1) parts += n;
    }
  }
  // split: key-range-major (items running together read the same keys: L2 reuse), heaviest first within a range;
  // no split: heaviest first
  if (g_split > 0 && g_order == 0)
    std::stable_sort(v.begin(), v.end(), [](const It& a, const It& b) {
      if (g_geo) return a.a.z != b.a.z ? a.a.z > b.a.z : a.w > b.w;  // geometric: biggest (latest) ranges first
      return a.a.z != b.a.z ? a.a.z < b.a.z : a.w > b.w;
    });
  else
    std::stable_sort(v.begin(), v.end(), [](const It& a, const It& b) { return a.w > b.w; });
  std::vector<int4> h1, h2;
  for (auto& t : v) { h1.push_back(t.a); h2.push_back(t.b); }
  pr.ntiles = (int)h1.size();
  pr.nrowtiles = tile;
  pr.nparts = parts;
  pr.br = br;
  CK(cudaMalloc(&pr.dtiles, h1.size() * sizeof(int4)));
  CK(cudaMalloc(&pr.ditems2, h2.size() * sizeof(int4)));
  CK(cudaMemcpy(pr.dtiles, h1.data(), h1.size() * sizeof(int4), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(pr.ditems2, h2.data(), h2.size() * sizeof(int4), cudaMemcpyHostToDevice));
  pr.ws_bytes = (size_t)std::max(parts, 1) * ws_per_part * 4;
  CK(cudaMalloc(&pr.ws, pr.ws_bytes));
  CK(cudaMalloc(&pr.cnt, (size_t)tile * 8 * sizeof(int)));
  CK(cudaMemset(pr.cnt, 0, (size_t)tile * 8 * sizeof(int)));
}
static void free_problem(Problem& pr) {
  for (void* x : {(void*)pr.q, (void*)pr.rel, (void*)pr.gate, (void*)pr.sink, (void*)pr.out, (void*)pr.o32, (void*)pr.lse,
                  (void*)pr.dsegs, (void*)pr.dtiles, (void*)pr.ditems2, (void*)pr.ws, (void*)pr.cnt})
    if (x) cudaFree(x);
  pr = Problem{};
}

// ------------------------------------------------------------------------------------------------ variants
struct Variant {
  const char* name;
  int br;
  size_t ws_per_part;
  cudaError_t (*fn)(const pfa::Params&, int, cudaStream_t);
  const void* kern;
};
// name: warps x m-tiles per warp, p = K/V pipeline (0 ring, 1 FA2 interleave), s = ring stages, m = min CTAs / SM
#define VAR(NW, MT, PI, ST, MB)                                                                  \
  {#NW "x" #MT "p" #PI "s" #ST "m" #MB, NW * 16 * MT, pfa::ws_floats_per_partial<NW, MT>(), &pfa::launch<NW, MT, PI, ST, MB>,         \
   (const void*)&pfa::prefill_attn_kernel<NW, MT, PI, ST, MB>}
static const Variant VARIANTS[] = {VAR(8, 1, 0, 2, 1), VAR(7, 1, 0, 2, 1), VAR(6, 1, 0, 2, 1), VAR(5, 1, 0, 2, 1),
                                   VAR(8, 1, 0, 3, 1), VAR(4, 1, 0, 2, 2), VAR(4, 2, 1, 1, 2), VAR(4, 2, 0, 2, 1),
                                   VAR(8, 2, 1, 1, 1), VAR(8, 1, 1, 1, 1),
                                   {"4x2p1s1m2-ceil", 128, pfa::ws_floats_per_partial<4, 2>(), &pfa::launch<4, 2, 1, 1, 2, true>,
                                    (const void*)&pfa::prefill_attn_kernel<4, 2, 1, 1, 2, true>}};
// "auto": pick the row-tile size per launch among the 1-CTA/SM ring variants (5..8 warps) by an LPT makespan
// estimate over 74 SMs, CTA cost = rows x key tiles (rows results do not depend on the choice: inv test 'variant')
static const char* AUTO_SET[] = {"8x1p0s2m1", "7x1p0s2m1", "6x1p0s2m1", "5x1p0s2m1"};
static const Variant* find_var(const std::string& n);
static const Variant* choose_auto(const std::vector<SegSpec>& segs, bool swa, int sms, double* eff_out);
static int nthreads(const Variant& v) { return std::atoi(v.name) * 32; }
static const Variant* find_var(const std::string& n) {
  if (n == "auto") return nullptr;
  for (auto& v : VARIANTS)
    if (n == v.name) return &v;
  return nullptr;
}

static const Variant* choose_auto(const std::vector<SegSpec>& segs, bool swa, int sms, double* eff_out) {
  const Variant* best = nullptr;
  double best_t = 1e300, best_eff = 0;
  for (const char* n : AUTO_SET) {
    const Variant* v = find_var(n);
    std::vector<double> w;
    double tot = 0;
    for (auto& s : segs)
      for (int r0 = 0; r0 < 3 * s.len; r0 += v->br) {
        const long long qmax = s.past + std::min(r0 + v->br - 1, 3 * s.len - 1) / 3, qmin = s.past + r0 / 3;
        const long long lo = swa ? std::max(0LL, qmin - 1023) : 0;
        const double c = (double)v->br * (qmax / 64 - lo / 64 + 1);
        for (int g = 0; g < 8; ++g) w.push_back(c);
        tot += 8 * c;
      }
    std::sort(w.begin(), w.end(), std::greater<double>());
    std::vector<double> sm(sms, 0.0);
    for (double c : w) *std::min_element(sm.begin(), sm.end()) += c;
    const double t = *std::max_element(sm.begin(), sm.end());
    if (t < best_t * 0.995) { best_t = t; best = v; best_eff = tot / (t * sms); }
  }
  if (eff_out) *eff_out = best_eff;
  return best;
}
static bool g_nobias = false;  // debug timing arm: skip the relative bias
static pfa::Params params(const Problem& pr, const Pool& P) {
  pfa::Params p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(pr.q);
  p.k = reinterpret_cast<const __nv_bfloat16*>(P.K);
  p.v = reinterpret_cast<const __nv_bfloat16*>(P.V);
  p.s_page = P.sp; p.s_slot = P.ss; p.s_head = P.sh;
  p.page_shift = P.shift;
  p.page_table = P.pt;
  p.rel = g_nobias ? nullptr : reinterpret_cast<const __nv_bfloat16*>(pr.rel);
  p.gate = reinterpret_cast<const __nv_bfloat16*>(pr.gate);
  p.sink = reinterpret_cast<const __nv_bfloat16*>(pr.sink);
  p.out = reinterpret_cast<__nv_bfloat16*>(pr.out);
  p.o32 = pr.o32;
  p.lse = pr.lse;
  p.segs = pr.dsegs;
  p.items = pr.dtiles;
  p.items2 = pr.ditems2;
  p.ws = pr.ws;
  p.cnt = pr.cnt;
  p.window = pr.swa ? 1024 : 0;
  p.qk_scale = (float)(0.08838834764831845f) * 1.4426950408889634f;  // fp32 * fp32, rounded
  return p;
}
static cudaError_t run(Problem& pr, const Pool& P, const Variant& v, cudaStream_t s) {
  build_tiles(pr, v.br, v.ws_per_part);
  return v.fn(params(pr, P), pr.ntiles, s);
}

// ------------------------------------------------------------------------------------------------ FP64 oracle
struct OracleRow { std::vector<double> out, o; std::vector<double> lse; };
static void oracle_row(const Data& d, bool swa, long long pos, OracleRow& r) {
  const int j = (int)(pos % QMAX);
  const uint16_t* q = (swa ? d.hQl.data() : d.hQg.data()) + (size_t)j * 3072;
  r.out.assign(3072, 0); r.o.assign(3072, 0); r.lse.assign(24, 0);
  const long long lo = swa ? std::max(0LL, pos - 1023) : 0;
  const long long n = pos - lo + 1;
  std::vector<double> s(n);
  const double scale = 1.0 / std::sqrt(128.0);
  for (int h = 0; h < 24; ++h) {
    const int g = h / 3;
    double qd[128];
    for (int c = 0; c < 128; ++c) qd[c] = p6::bf16_to_f32(q[h * 128 + c]);
    double mx = -INFINITY;
    for (long long kp = lo; kp <= pos; ++kp) {
      const uint16_t* k = d.hK.data() + (size_t)kp * KVROW + g * 128;
      double acc = 0;
      for (int c = 0; c < 128; ++c) acc += qd[c] * (double)p6::bf16_to_f32(k[c]);
      double x = acc * scale;
      const long long dist = pos - kp;
      if (dist < 1024) x += p6::bf16_to_f32(d.hREL[((size_t)j * 24 + h) * 1024 + dist]);
      s[kp - lo] = x;
      mx = std::max(mx, x);
    }
    double den = 0, o[128] = {0};
    for (long long kp = lo; kp <= pos; ++kp) {
      const double e = std::exp(s[kp - lo] - mx);
      den += e;
      const uint16_t* v = d.hV.data() + (size_t)kp * KVROW + g * 128;
      for (int c = 0; c < 128; ++c) o[c] += e * (double)p6::bf16_to_f32(v[c]);
    }
    const double lse = mx + std::log(den);
    const double sk = p6::bf16_to_f32(d.hSINK[(size_t)j * 24 + h]);
    const double fs = 1.0 / (1.0 + std::exp(-(lse - sk)));
    r.lse[h] = lse;
    for (int c = 0; c < 128; ++c) {
      const double oc = o[c] / den;
      const double gt = p6::bf16_to_f32(d.hGATE[(size_t)j * 3072 + h * 128 + c]);
      r.o[h * 128 + c] = oc;
      r.out[h * 128 + c] = oc / (1.0 + std::exp(-gt)) * fs;
    }
  }
}

struct Err { double se = 0, sr = 0, maxabs = 0; long long n = 0;
  void add(double x, double ref) { const double e = x - ref; se += e * e; sr += ref * ref; maxabs = std::max(maxabs, std::fabs(e)); ++n; }
  double rel() const { return std::sqrt(se / std::max(sr, 1e-300)); } };

static bool read_file(const std::string& path, void* dst, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  const size_t got = std::fread(dst, 1, bytes, f);
  std::fclose(f);
  return got == bytes;
}
static void write_file(const std::string& path, const void* src, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) p6::harness_fail(PROBE, "fopen", path.c_str());
  std::fwrite(src, 1, bytes, f);
  std::fclose(f);
}

// ------------------------------------------------------------------------------------------------ main
int main(int argc, char** argv) {
  p6::Ctx c;
  c.probe = PROBE;
  c.a = {argc, argv};
  const std::string mode = c.a.get("--mode", "info");
  p6::open_device(c);
  const uint64_t seed = (uint64_t)c.a.num("--seed", 20261123);

  if (mode == "info") {
    for (auto& v : VARIANTS) {
      cudaFuncAttributes fa{};
      cudaFuncGetAttributes(&fa, v.kern);
      pfa::Params dummy{};
      v.fn(dummy, 0, c.s);  // sets the smem attribute only
      int nb = 0;
      cudaFuncGetAttributes(&fa, v.kern);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, v.kern, nthreads(v), (size_t)fa.maxDynamicSharedSizeBytes);
      Line(PROBE, "info").kv("var", v.name).kv("regs", fa.numRegs).kv("local_bytes", (long long)fa.localSizeBytes)
          .kv("smem_dyn", fa.maxDynamicSharedSizeBytes).kv("threads", nthreads(v)).kv("ctas_per_sm", nb).emit();
    }
    return 0;
  }

  const long long nmax = (long long)c.a.num("--nmax", 245760 + 8192);
  Data d;
  const bool need_host = mode == "check" || mode == "gen";
  make_data(d, nmax, seed, need_host);
  Line(PROBE, "data").kv("npos", nmax).kv("qmax", QMAX).kv("seed", (unsigned long long)seed).kv("host", need_host).emit();

  if (mode == "gen") {
    const std::string dir = c.a.get("--dir", "/out/data");
    write_file(dir + "/K.bin", d.hK.data(), d.hK.size() * 2);
    write_file(dir + "/V.bin", d.hV.data(), d.hV.size() * 2);
    write_file(dir + "/qg.bin", d.hQg.data(), d.hQg.size() * 2);
    write_file(dir + "/ql.bin", d.hQl.data(), d.hQl.size() * 2);
    write_file(dir + "/rel.bin", d.hREL.data(), d.hREL.size() * 2);
    write_file(dir + "/gate.bin", d.hGATE.data(), d.hGATE.size() * 2);
    write_file(dir + "/sink.bin", d.hSINK.data(), d.hSINK.size() * 2);
    Line(PROBE, "gen").kv("ok", true).kv("dir", dir).kv("npos", nmax).emit();
    return 0;
  }

  const int page = (int)c.a.num("--page", 64);
  const int layout = (int)c.a.num("--layout", 0);
  Pool P;
  make_pool(P, d, page, layout, seed ^ 0xBEEF);
  const auto kinds = c.a.slist("--kind", "global,swa");
  const auto Cs = c.a.list("--C", "0,8192,32768,65536,122880,245760");
  const auto Qs = c.a.list("--Q", "2048");
  const auto vars = c.a.slist("--var", "8x1p0s2m1");
  g_nobias = c.a.has("--nobias");
  g_split = (int)c.a.num("--split", 0);
  g_order = (int)c.a.num("--order", 0);
  g_geo = (int)c.a.num("--geo", 0);
  if (g_split % 64) p6::harness_fail(PROBE, "split", "must be a multiple of 64 keys");

  if (mode == "check") {
    const std::string cdir = c.a.get("--cfgb-dir", "");
    const int nsamp = (int)c.a.num("--samples", 32);
    const Variant* v = find_var(vars[0]);
    if (!v) p6::harness_fail(PROBE, "variant", vars[0].c_str());
    for (auto& kind : kinds)
      for (long long C : Cs)
        for (long long Q : Qs) {
          const bool swa = kind == "swa";
          if (C + Q > nmax) continue;
          Problem pr;
          build_problem(pr, d, P, swa, {{(int)C, (int)Q}}, true);
          CK(run(pr, P, *v, c.s));
          CK(cudaStreamSynchronize(c.s));
          const size_t T = Q;
          std::vector<uint16_t> out(T * 3072), out2(T * 3072);
          std::vector<float> o32(T * 3072), lse(T * 24);
          CK(cudaMemcpy(out.data(), pr.out, out.size() * 2, cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(o32.data(), pr.o32, o32.size() * 4, cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(lse.data(), pr.lse, lse.size() * 4, cudaMemcpyDeviceToHost));
          bool det = true;  // 2 more runs, bit compare
          for (int rep = 0; rep < 2; ++rep) {
            CK(cudaMemsetAsync(pr.out, 0xff, out2.size() * 2, c.s));  // poison, so a stale buffer cannot pass
            CK(run(pr, P, *v, c.s));
            CK(cudaStreamSynchronize(c.s));
            CK(cudaMemcpy(out2.data(), pr.out, out2.size() * 2, cudaMemcpyDeviceToHost));
            det = det && out2 == out;
          }
          // config B outputs (cfgb_prefill.py --mode check), same cell
          char nm[128];
          std::snprintf(nm, sizeof nm, "%s_C%lld_Q%lld", kind.c_str(), C, Q);
          std::vector<uint16_t> cb_out(T * 3072);
          std::vector<float> cb_o(T * 3072), cb_lse(24 * T);
          bool have_cb = !cdir.empty() && read_file(cdir + "/cfgb_out_" + nm + ".bin", cb_out.data(), cb_out.size() * 2) &&
                         read_file(cdir + "/cfgb_o32_" + nm + ".bin", cb_o.data(), cb_o.size() * 4) &&
                         read_file(cdir + "/cfgb_lse_" + nm + ".bin", cb_lse.data(), cb_lse.size() * 4);
          // sampled rows: edges + spread + random
          std::vector<long long> rows = {0, 1, 2, 41, 42, 63, 64, Q / 2, Q - 2, Q - 1};
          std::mt19937_64 rng(seed ^ (uint64_t)(C * 7 + Q));
          while ((int)rows.size() < nsamp) rows.push_back((long long)(rng() % Q));
          std::vector<OracleRow> orc(rows.size());
          {
            std::vector<std::thread> th;
            std::atomic<size_t> next{0};
            const int nth = (int)c.a.num("--threads", 4);
            for (int t = 0; t < nth; ++t)
              th.emplace_back([&] { for (size_t i; (i = next++) < rows.size();) oracle_row(d, swa, C + rows[i], orc[i]); });
            for (auto& t : th) t.join();
          }
          Err e_out, e_o, e_lse, c_out, c_o, c_lse, e_out_bfref;
          for (size_t i = 0; i < rows.size(); ++i) {
            const size_t t = rows[i];
            for (int x = 0; x < 3072; ++x) {
              e_out.add(p6::bf16_to_f32(out[t * 3072 + x]), orc[i].out[x]);
              e_o.add(o32[t * 3072 + x], orc[i].o[x]);
              // floor: the oracle itself rounded to bf16 (the output format's own error)
              e_out_bfref.add(p6::bf16_to_f32(p6::f32_to_bf16_rne((float)orc[i].out[x])), orc[i].out[x]);
              if (have_cb) { c_out.add(p6::bf16_to_f32(cb_out[t * 3072 + x]), orc[i].out[x]); c_o.add(cb_o[t * 3072 + x], orc[i].o[x]); }
            }
            for (int h = 0; h < 24; ++h) {
              e_lse.add(lse[t * 24 + h], orc[i].lse[h]);
              if (have_cb) c_lse.add(cb_lse[(size_t)h * T + t], orc[i].lse[h]);
            }
          }
          Line l(PROBE, "oracle");
          l.kv("kind", kind).kv("C", C).kv("Q", Q).kv("var", v->name).kv("split", g_split).kv("page", page).kv("layout", layout)
              .kv("rows", (long long)rows.size()).kv("deterministic_3runs", det)
              .kv("ours_out_rel_rms", e_out.rel()).kv("ours_out_max_abs", e_out.maxabs)
              .kv("ours_o32_rel_rms", e_o.rel()).kv("ours_o32_max_abs", e_o.maxabs).kv("ours_lse_max_abs", e_lse.maxabs)
              .kv("bf16_floor_out_rel_rms", e_out_bfref.rel()).kv("cfgb_present", have_cb);
          if (have_cb) {
            long long same = 0;
            double maxd = 0;
            for (size_t x = 0; x < out.size(); ++x) {
              same += out[x] == cb_out[x];
              maxd = std::max(maxd, (double)std::fabs(p6::bf16_to_f32(out[x]) - p6::bf16_to_f32(cb_out[x])));
            }
            double od = 0, on = 0;
            for (size_t x = 0; x < o32.size(); ++x) { od += (o32[x] - cb_o[x]) * (double)(o32[x] - cb_o[x]); on += (double)cb_o[x] * cb_o[x]; }
            l.kv("cfgb_out_rel_rms", c_out.rel()).kv("cfgb_out_max_abs", c_out.maxabs).kv("cfgb_o32_rel_rms", c_o.rel())
                .kv("cfgb_o32_max_abs", c_o.maxabs).kv("cfgb_lse_max_abs", c_lse.maxabs)
                .kv("ours_vs_cfgb_out_bit_equal_frac", (double)same / out.size()).kv("ours_vs_cfgb_out_max_abs", maxd)
                .kv("ours_vs_cfgb_o32_rel_rms_full", std::sqrt(od / std::max(on, 1e-300)))
                .kv("ratio_ours_over_cfgb_o32_err", e_o.rel() / std::max(c_o.rel(), 1e-300));
          }
          l.kv("pass", det && e_out.rel() <= 1.25 * std::max(have_cb ? c_out.rel() : e_out_bfref.rel() * 1.5, e_out_bfref.rel()));
          l.emit();
          free_problem(pr);
        }
    return 0;
  }

  auto diffstat = [](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    long long n = 0, first = -1;
    double mx = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
      if (a[i] != b[i]) {
        if (first < 0) first = (long long)i;
        ++n;
        mx = std::max(mx, (double)std::fabs(p6::bf16_to_f32(a[i]) - p6::bf16_to_f32(b[i])));
      }
    char s[160];
    std::snprintf(s, sizeof s, "{\"n\":%lld,\"first_row\":%lld,\"first_col\":%lld,\"max_abs\":%.3g}", n,
                  first < 0 ? -1 : first / 3072, first < 0 ? -1 : first % 3072, mx);
    return std::string(s);
  };
  if (mode == "inv") {
    const Variant* v = find_var(vars[0]);
    if (!v) p6::harness_fail(PROBE, "variant", vars[0].c_str());
    auto fetch = [&](Problem& pr) {
      CK(cudaStreamSynchronize(c.s));  // kernels run on a non-blocking stream: cudaMemcpy alone does not wait for them
      std::vector<uint16_t> h((size_t)pr.T * 3072);
      CK(cudaMemcpy(h.data(), pr.out, h.size() * 2, cudaMemcpyDeviceToHost));
      return h;
    };
    for (auto& kind : kinds) {
      const bool swa = kind == "swa";
      for (long long C : Cs) {
        const int N = (int)c.a.num("--invN", 8192);
        if (C + N > nmax) continue;
        // reference: one chunk of N rows at past C
        Problem ref;
        build_problem(ref, d, P, swa, {{(int)C, N}}, false);
        CK(run(ref, P, *v, c.s));
        auto href = fetch(ref);
        // (a) chunking: chunks of 512 / 2048 / 3000 (odd) rows, run one after another
        for (int ch : {512, 2048, 3000}) {
          std::vector<uint16_t> acc;
          for (int s0 = 0; s0 < N; s0 += ch) {
            Problem pr;
            build_problem(pr, d, P, swa, {{(int)C + s0, std::min(ch, N - s0)}}, false);
            CK(run(pr, P, *v, c.s));
            auto h = fetch(pr);
            acc.insert(acc.end(), h.begin(), h.end());
            free_problem(pr);
          }
          Line(PROBE, "inv").kv("kind", kind).kv("C", C).kv("N", N).kv("check", "chunking").kv("chunk", ch)
              .kv("bit_identical", acc == href).raw("diff", diffstat(acc, href)).emit();
        }
        // (b) batch composition: the same segment between two other segments (other lengths / contexts)
        {
          Problem pr;
          build_problem(pr, d, P, swa, {{777, 1500}, {(int)C, N}, {12345, 333}}, false);
          CK(run(pr, P, *v, c.s));
          auto h = fetch(pr);
          std::vector<uint16_t> mid(h.begin() + (size_t)1500 * 3072, h.begin() + (size_t)(1500 + N) * 3072);
          Line(PROBE, "inv").kv("kind", kind).kv("C", C).kv("N", N).kv("check", "batch").kv("bit_identical", mid == href).emit();
          free_problem(pr);
        }
        // (c) other row-tile size (other variant geometry), if available
        for (auto& vn : vars) {
          const Variant* v2 = find_var(vn);
          if (!v2 || v2 == v) continue;
          CK(run(ref, P, *v2, c.s));
          auto h = fetch(ref);
          Line(PROBE, "inv").kv("kind", kind).kv("C", C).kv("N", N).kv("check", "variant").kv("var2", v2->name)
              .kv("bit_identical", h == href).emit();
        }
        // (d) other page placement and layout
        {
          Pool P2;
          make_pool(P2, d, page == 64 ? 256 : 64, 1 - layout, seed ^ 0xF00D);
          CK(run(ref, P2, *v, c.s));
          auto h = fetch(ref);
          Line(PROBE, "inv").kv("kind", kind).kv("C", C).kv("N", N).kv("check", "pages").kv("page2", P2.page)
              .kv("layout2", P2.layout).kv("bit_identical", h == href).emit();
          free_pool(P2);
        }
        free_problem(ref);
      }
    }
    return 0;
  }

  if (mode == "time") {
    // per cell: variants in sandwich order (first variant = baseline), --rounds rounds
    for (auto& kind : kinds)
      for (long long C : Cs)
        for (long long Q : Qs) {
          const bool swa = kind == "swa";
          if (C + Q > nmax) continue;
          Problem pr;
          build_problem(pr, d, P, swa, {{(int)C, (int)Q}}, false);
          std::vector<const Variant*> vs;
          double auto_eff = 0;
          for (auto& vn : vars) {
            const Variant* v = vn == "auto" ? choose_auto(pr.segs, swa, c.prop.multiProcessorCount, &auto_eff) : find_var(vn);
            if (!v) p6::harness_fail(PROBE, "variant", vn.c_str());
            vs.push_back(v);
          }
          // calls per chunk ~ 40 ms
          std::vector<int> cpc(vs.size());
          for (size_t i = 0; i < vs.size(); ++i) {
            CK(run(pr, P, *vs[i], c.s));
            cudaEvent_t a, b;
            cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a, c.s);
            for (int k = 0; k < 3; ++k) CK(run(pr, P, *vs[i], c.s));
            cudaEventRecord(b, c.s);
            CK(cudaEventSynchronize(b));
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            cpc[i] = std::max(1, (int)(40.0 / (ms / 3)));
          }
          for (auto& sl : p6::sandwich((int)vs.size(), c.rounds)) {
            const Variant* v = vs[sl.arm];
            Line l(PROBE, "time");
            l.kv("kind", kind).kv("C", C).kv("Q", Q).kv("var", vars[sl.arm]).kv("impl", v->name).kv("page", page).kv("layout", layout)
                .kv("nobias", g_nobias).kv("auto_lpt_eff", vars[sl.arm] == "auto" ? auto_eff : 0.0).kv("split", g_split).kv("order", g_order).kv("geo", g_geo)
                .kv("items", pr.ntiles).kv("ws_mb", pr.ws_bytes / 1e6).kv("round", sl.round).kv("pos", sl.pos).kv("sandwich_tail", sl.sandwich_tail).kv("flops", pr.flops);
            auto r = p6::sustained(c, l, [&](cudaStream_t s) { run(pr, P, *v, s); }, pr.flops / 1e12, cpc[sl.arm], "TFLOP");
            if (std::isfinite(r.ms_per_call_median)) {
              l.kv("ms_per_chunk", r.ms_per_call_median).kv("tflops", pr.flops / (r.ms_per_call_median * 1e-3) / 1e12);
              if (std::isfinite(r.j_per_call)) l.kv("j_per_new_token", r.j_per_call / Q);
            }
            l.emit();
          }
          free_problem(pr);
        }
    return 0;
  }
  p6::harness_fail(PROBE, "mode", mode.c_str());
}
