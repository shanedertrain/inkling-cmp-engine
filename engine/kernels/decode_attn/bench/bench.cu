// S2b decode attention bench (our code): FP64 oracle on synthetic and real KV, batch-invariance test, sustained
// exec / step-equivalent timing in the P6 discipline (common.hpp: >= 20 s cells, 2 s dropped, 10 Hz NVML, energy,
// sandwich, rounds). Modes: info, check, inv, exec, step, realkv.
#include "common.hpp"
#include "../decode_attn.cuh"

#include <map>
#include <thread>

static const char* P = "s2b_decode_attn";
using namespace da;

// ------------------------------------------------------------------------------------------------ data
__host__ __device__ inline float gauss(uint64_t h) {
  const float u1 = p6::u01(h), u2 = p6::u01(p6::splitmix64(h ^ 0x5bd1e995ull));
  return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
__host__ __device__ inline uint8_t f32_to_e4m3(float x) {  // RNE, saturate to 448 (kvgen.cuh)
  const uint8_t sign = x < 0 ? 0x80 : 0;
  float a = fabsf(x);
  if (!(a < 448.0f)) return sign | 0x7e;
  if (a < 0.015625f) return sign | (uint8_t)rintf(a * 512.0f);
  int e;
  const float f = frexpf(a, &e);
  int E = e - 1, m = (int)rintf((2.0f * f - 1.0f) * 8.0f);
  if (m == 8) { m = 0; ++E; }
  int code = ((E + 7) << 3) | m;
  if (code > 0x7e) code = 0x7e;
  return sign | (uint8_t)code;
}

// KV layout (one execution = one layer visit), paged for both kinds (pages of 1024 keys). ring = SWA: each session
// owns kRingPages physical pages and its page-table row maps logical page j to ring page j % kRingPages (the native
// runtime's per-slot rings); global: one physical page per logical page.
constexpr int kRingPages = 3;
struct Layout {
  bool ring = false, headmajor = false;
  int eb = 2, page_shift = 10;
  long long page_stride = 0, head_stride = 0;
  int row_stride = 0;
  void init(bool ring_, bool hm, int eb_) {
    ring = ring_, headmajor = hm, eb = eb_;
    const long long rowb = (long long)kD * eb;
    row_stride = (int)(hm ? rowb : kHkv * rowb);
    head_stride = hm ? (1LL << page_shift) * rowb : rowb;
    page_stride = (1LL << page_shift) * kHkv * rowb;
  }
  __host__ __device__ long long addr(const int* pt, int pt_stride, int s, long long key, int g) const {
    const int page = pt[(size_t)s * pt_stride + (key >> page_shift)];
    return (long long)page * page_stride + g * head_stride + (key & ((1 << page_shift) - 1)) * row_stride;
  }
};
// page table: global -> n * pt_stride shuffled pages; ring -> session s's row maps j to its pages perm[s * 3 + j % 3]
static std::vector<int> make_pt(bool ring, int n, int pt_stride, uint64_t seed, int& npages) {
  npages = ring ? n * kRingPages : n * pt_stride;
  std::vector<int> perm(npages);
  for (int i = 0; i < npages; ++i) perm[i] = i;
  for (int i = npages - 1; i > 0; --i) {  // deterministic shuffle
    const int j = (int)(p6::splitmix64(seed ^ (uint64_t)i) % (uint64_t)(i + 1));
    std::swap(perm[i], perm[j]);
  }
  std::vector<int> pt((size_t)n * pt_stride);
  for (int s = 0; s < n; ++s)
    for (int j = 0; j < pt_stride; ++j) pt[(size_t)s * pt_stride + j] = ring ? perm[s * kRingPages + j % kRingPages] : perm[s * pt_stride + j];
  return pt;
}

// Synthetic KV value (kvgen.cuh statistics) of session s, key position kpos, head g, in a per-execution stream.
// One warp per (key, head). Writes bf16 or E4M3(value * scale) into the layout.
__global__ void gen_kv(uint8_t* K, uint8_t* V, Layout lay, const int* pt, int pt_stride, int s, int sid, long long k0,
                       long long nkeys, int kv, float kscale, float vscale, uint64_t seed) {
  const long long wid = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (wid >= nkeys * kHkv) return;
  const long long key = k0 + wid / kHkv;
  const int head = (int)(wid % kHkv);
  const uint64_t hk = p6::splitmix64(seed ^ ((uint64_t)sid << 52) ^ ((uint64_t)key * 8 + head) * 0x9E3779B97F4A7C15ull);
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
  const long long a = lay.addr(pt, pt_stride, s, key, head);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float kb = __bfloat162float(__float2bfloat16_rn(k[i] * ks));
    const float vb = __bfloat162float(__float2bfloat16_rn(v[i] * vs));
    const int d = lane * 4 + i;
    if (kv == KV_BF16) {
      reinterpret_cast<uint16_t*>(K + a)[d] = f2bf(kb);
      reinterpret_cast<uint16_t*>(V + a)[d] = f2bf(vb);
    } else {
      K[a + d] = f32_to_e4m3(kb * kscale);
      V[a + d] = f32_to_e4m3(vb * vscale);
    }
  }
}
// q [T][24][128] bf16 (clip 22.627, tau on global), rel [T][24][E], gate [T][3072], sink [T][24]: all seeded by
// (session id, row t, head) only, so a session's data do not depend on n or on its position in the batch.
__global__ void gen_rows(uint16_t* q, uint16_t* rel, uint16_t* gate, uint16_t* sink, const int* sid, const int* ctx,
                         int n, int R, int E, int global, uint64_t seed) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= n * R * kHq) return;
  const int si = wid / (R * kHq), t = (wid / kHq) % R, h = wid % kHq;
  const uint64_t hq = p6::splitmix64(seed ^ ((uint64_t)sid[si] << 40) ^ ((uint64_t)t << 20) ^ (uint64_t)h);
  float x[4], nn = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = lane * 4 + i;
    const uint64_t ch = p6::splitmix64(0xFACEull ^ ((uint64_t)h << 8) ^ (uint64_t)c);
    x[i] = expf(0.5f * gauss(ch)) * gauss(p6::splitmix64(hq ^ (uint64_t)c));
    nn += x[i] * x[i];
  }
#pragma unroll
  for (int sh = 16; sh; sh >>= 1) nn += __shfl_xor_sync(0xffffffffu, nn, sh);
  const float target = 22.627f * (0.6f + 0.6f * p6::u01(p6::splitmix64(hq ^ 7)));
  const float sc = (target / (sqrtf(nn) + 1e-6f)) * fminf(1.0f, 22.627f / (target + 1e-6f));
  const float pos = (float)(ctx[si] + t);
  const float tau = 1.0f + 0.1f * logf(fmaxf((pos + 1.0f) / 8192.0f, 1.0f));
  const size_t row = (size_t)si * R + t;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float v = __bfloat162float(__float2bfloat16_rn(x[i] * sc));
    if (global) v = __bfloat162float(__float2bfloat16_rn(v * tau));
    q[(row * kHq + h) * kD + lane * 4 + i] = f2bf(v);
    gate[(row * kHq + h) * kD + lane * 4 + i] = f2bf(1.5f * gauss(p6::splitmix64(hq ^ 0x33ull ^ (uint64_t)(lane * 4 + i))));
  }
  for (int d = lane; d < E; d += 32)
    rel[(row * kHq + h) * E + d] = f2bf(1.5f * gauss(p6::splitmix64(hq ^ 0x11ull ^ (uint64_t)d * 0x9E37ull)) * expf(-d / 256.0f));
  if (lane == 0) sink[row * kHq + h] = f2bf(2.0f * gauss(p6::splitmix64(hq ^ 0x22ull)));
}

static size_t g_alloc = 0;
static double g_cap_gb = 56.0;
static void* dalloc(size_t b) {
  if ((g_alloc + b) / 1e9 > g_cap_gb) p6::harness_fail(P, "memory cap", "raise --mem-cap-gb");
  void* p = nullptr;
  P6_CK(P, cudaMalloc(&p, b));
  g_alloc += b;
  return p;
}
template <class T>
static std::vector<T> d2h(const T* p, size_t n) {
  std::vector<T> v(n);
  cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}

// One execution's KV for n sessions (global: paged pool with shuffled pages; SWA: rings)
struct KvSet {
  Layout lay;
  int kv = 0, n = 0, pt_stride = 0;
  uint8_t* K = nullptr;
  uint8_t* V = nullptr;
  int* pt = nullptr;  // device
  std::vector<int> hpt;
  size_t bytes = 0;
};
static float g_kscale = 1.0f, g_vscale = 1.0f;  // E4M3 storage scales (value * scale stored)
// ---- real-KV timing data: K/V (and q) of captured config B executions, tiled to any length (key k <- dump key k % N)
static std::string g_kvdump;
struct DumpExec { int nkeys = 0; uint16_t* k = nullptr; uint16_t* v = nullptr; std::vector<uint16_t> q; };
static std::map<int, DumpExec>* g_dumpc = nullptr;
static std::vector<uint8_t> readf(const std::string& f);
static DumpExec& dump_exec(int e) {
  auto it = g_dumpc->find(e);
  if (it != g_dumpc->end()) return it->second;
  DumpExec d;
  char pre[600];
  std::snprintf(pre, sizeof pre, "%s/e%02d.", g_kvdump.c_str(), e);
  auto K = readf(std::string(pre) + "k.bin"), V = readf(std::string(pre) + "v.bin"), Q = readf(std::string(pre) + "q.bin");
  d.nkeys = (int)(K.size() / (kHkv * kD * 2));
  P6_CK(P, cudaMalloc(&d.k, K.size()));
  P6_CK(P, cudaMalloc(&d.v, V.size()));
  P6_CK(P, cudaMemcpy(d.k, K.data(), K.size(), cudaMemcpyHostToDevice));
  P6_CK(P, cudaMemcpy(d.v, V.data(), V.size(), cudaMemcpyHostToDevice));
  d.q.resize(Q.size() / 2);
  std::memcpy(d.q.data(), Q.data(), Q.size());
  return (*g_dumpc)[e] = d;
}
// global executions of the dump (visits of layers 5, 11, 17, 23) and the 40 SWA ones, in visit order
static const int kDumpGlobal[8] = {5, 11, 17, 23, 29, 35, 41, 47};
static int dump_swa(int i) { int c = 0; for (int e = 0; e < 48; ++e) { bool gl = false; for (int x : kDumpGlobal) gl |= x == e; if (!gl && c++ == i) return e; } return 0; }
__global__ void scatter_kv(uint8_t* K, uint8_t* V, Layout lay, const int* pt, int pt_stride, int s, long long k0, long long nkeys,
                           const uint16_t* dk, const uint16_t* dv, int dn, int koff, int kv, float kscale, float vscale) {
  const long long wid = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (wid >= nkeys * kHkv) return;
  const long long key = k0 + wid / kHkv;
  const int head = (int)(wid % kHkv);
  const long long src = (((key + koff) % dn) * kHkv + head) * kD;
  const long long a = lay.addr(pt, pt_stride, s, key, head);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 4 + i;
    const uint16_t kb = dk[src + d], vb = dv[src + d];
    if (kv == KV_BF16) {
      reinterpret_cast<uint16_t*>(K + a)[d] = kb;
      reinterpret_cast<uint16_t*>(V + a)[d] = vb;
    } else {
      K[a + d] = f32_to_e4m3(__uint_as_float((uint32_t)kb << 16) * kscale);
      V[a + d] = f32_to_e4m3(__uint_as_float((uint32_t)vb << 16) * vscale);
    }
  }
}

// kend: optional per-session key end (ring: the session's own last 2,048 keys are written; default maxkeys)
static KvSet make_kv(int kv, bool ring, bool hm, int n, const std::vector<int>& sids, long long maxkeys, uint64_t seed,
                     cudaStream_t st, int dump_e = -1, const std::vector<long long>* kend = nullptr) {
  KvSet k;
  k.kv = kv, k.n = n;
  k.lay.init(ring, hm, kv == KV_BF16 ? 2 : 1);
  {
    k.pt_stride = (int)((maxkeys + 1023) >> 10) + 1;
    int npages = 0;
    k.hpt = make_pt(ring, n, k.pt_stride, seed, npages);
    k.bytes = (size_t)npages * k.lay.page_stride;
    k.pt = (int*)dalloc(k.hpt.size() * 4);
    P6_CK(P, cudaMemcpy(k.pt, k.hpt.data(), k.hpt.size() * 4, cudaMemcpyHostToDevice));
  }
  k.K = (uint8_t*)dalloc(k.bytes);
  k.V = (uint8_t*)dalloc(k.bytes);
  P6_CK(P, cudaMemsetAsync(k.K, 0, k.bytes, st));
  P6_CK(P, cudaMemsetAsync(k.V, 0, k.bytes, st));
  for (int s = 0; s < n; ++s) {
    // ring: only the last 2,048 keys are written (they map to distinct slots of the 3 ring pages and cover every window)
    const long long ke = kend ? (*kend)[s] : maxkeys;
    const long long k0 = ring ? std::max(0LL, ke - 2048) : 0, nk = ke - k0;
    if (dump_e >= 0) {
      DumpExec& d = dump_exec(dump_e);
      scatter_kv<<<(unsigned)((nk * kHkv * 32 + 255) / 256), 256, 0, st>>>(k.K, k.V, k.lay, k.pt, k.pt_stride, s, k0, nk, d.k, d.v, d.nkeys,
                                                                       (int)((uint64_t)sids[s] * 7919 % d.nkeys), kv, g_kscale, g_vscale);
      continue;
    }
    gen_kv<<<(unsigned)((nk * kHkv * 32 + 255) / 256), 256, 0, st>>>(k.K, k.V, k.lay, k.pt, k.pt_stride, s, sids[s], k0, nk, kv,
                                                                   g_kscale, g_vscale, seed);
  }
  P6_CK(P, cudaStreamSynchronize(st));
  return k;
}

struct Rows {
  int n = 0, R = 0, E = 1024, max_chunks = 0;
  uint16_t *q = nullptr, *rel = nullptr, *gate = nullptr, *sink = nullptr, *out = nullptr;
  float *o_part = nullptr, *out32 = nullptr;
  float2* ml = nullptr;
  int *ctx = nullptr, *row0 = nullptr, *cshift = nullptr, *sid = nullptr, *nrows = nullptr;
  std::vector<int> widths;  // ragged per-session widths (empty = uniform R)
  std::vector<int> hctx, hcs;
  int grid_x = 0;
};
static int g_cs_swa = 0;  // override of the SWA rule (only to compare against runs made with another rule)
static int chunk_shift_for(bool global, long long ctx, int fixed) {
  if (fixed > 0) return fixed;
  if (!global) return g_cs_swa > 0 ? g_cs_swa : 7;    // SWA: 128-key chunks (LEDGER B3/B4: fastest for designs A and B)
  return ctx < 98304 ? 10 : 12;   // global: 1024 keys below 96K context, 4096 above (LEDGER C1, t17)
}
static std::vector<int> g_widths;  // set around make_rows calls for ragged batches
static Rows make_rows(int n, int R, const std::vector<int>& ctx, const std::vector<int>& sids, bool global, int fixed_cs,
                      uint64_t seed, cudaStream_t st) {
  Rows m;
  m.n = n, m.R = R, m.hctx = ctx;
  const int T = n * R, NC = kG * R;
  m.hcs.resize(n);
  int gx = 0;
  for (int s = 0; s < n; ++s) {
    m.hcs[s] = chunk_shift_for(global, ctx[s], fixed_cs);
    const int Rs = g_widths.empty() ? R : g_widths[s % g_widths.size()];
    const long long qlast = ctx[s] + Rs - 1, klo = global ? 0 : std::max(0LL, (long long)ctx[s] - 1023);
    gx = std::max(gx, (int)((qlast >> m.hcs[s]) - (klo >> m.hcs[s]) + 1));
  }
  m.grid_x = gx, m.max_chunks = gx;
  m.q = (uint16_t*)dalloc((size_t)T * kHq * kD * 2);
  m.rel = (uint16_t*)dalloc((size_t)T * kHq * m.E * 2);
  m.gate = (uint16_t*)dalloc((size_t)T * kHq * kD * 2);
  m.sink = (uint16_t*)dalloc((size_t)T * kHq * 2);
  m.out = (uint16_t*)dalloc((size_t)T * kHq * kD * 2);
  m.out32 = (float*)dalloc((size_t)T * kHq * kD * 4);
  m.o_part = (float*)dalloc((size_t)n * kHkv * gx * NC * kD * 4);
  m.ml = (float2*)dalloc((size_t)n * kHkv * gx * NC * 8);
  m.ctx = (int*)dalloc(n * 4), m.row0 = (int*)dalloc(n * 4), m.cshift = (int*)dalloc(n * 4), m.sid = (int*)dalloc(n * 4);
  std::vector<int> r0(n);
  for (int s = 0; s < n; ++s) r0[s] = s * R;
  if (!g_widths.empty()) {
    for (int s = 0; s < n; ++s) m.widths.push_back(g_widths[s % g_widths.size()]);
    m.nrows = (int*)dalloc(n * 4);
    cudaMemcpy(m.nrows, m.widths.data(), n * 4, cudaMemcpyHostToDevice);
  }
  cudaMemcpy(m.ctx, ctx.data(), n * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(m.row0, r0.data(), n * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(m.cshift, m.hcs.data(), n * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(m.sid, sids.data(), n * 4, cudaMemcpyHostToDevice);
  gen_rows<<<(T * kHq * 32 + 255) / 256, 256, 0, st>>>(m.q, m.rel, m.gate, m.sink, m.sid, m.ctx, n, R, m.E, global, seed);
  P6_CK(P, cudaStreamSynchronize(st));
  return m;
}
static Params make_params(const KvSet& k, const Rows& m, bool global) {
  Params p{};
  p.k = k.K, p.v = k.V, p.page_table = k.pt, p.pt_stride = k.pt_stride, p.page_shift = k.lay.page_shift;
  p.page_stride = k.lay.page_stride, p.head_stride = k.lay.head_stride;
  p.row_stride = k.lay.row_stride;
  p.window = global ? (1 << 30) : 1024;
  p.ctx = m.ctx, p.row0 = m.row0, p.chunk_shift = m.cshift, p.R = m.R, p.nrows = m.nrows;
  p.q = m.q, p.rel = m.rel, p.extent = m.E, p.gate = m.gate, p.sink = m.sink;
  p.o_part = m.o_part, p.ml_part = m.ml, p.max_chunks = m.max_chunks, p.out = m.out, p.out32 = m.out32;
  const float l2s = kLog2e / sqrtf((float)kD);
  if (k.kv == KV_BF16) p.sc = l2s, p.vscale = 1.0f;
  else p.sc = l2s * 256.0f / g_kscale, p.vscale = 256.0f / g_vscale;  // placement decode = code value * 2^-8
  return p;
}

// ------------------------------------------------------------------------------------------------ FP64 oracle
struct Err {
  double se = 0, sr = 0, mx = 0, se32 = 0, sr32 = 0;
  size_t n = 0;
  void add(double got, double ref) { se += (got - ref) * (got - ref); sr += ref * ref; mx = std::max(mx, std::fabs(got - ref)); ++n; }
  double rel() const { return std::sqrt(se / std::max(sr, 1e-300)); }
};
// out[h][d] (gated, fp64) and o[h][d] (pre-gate) of one row from decoded K/V [nk][8][128] (key i at position lo + i)
static void oracle_row(const double* Kd, const double* Vd, long long lo, long long nk, long long pos, bool global,
                       const float* q /*24x128*/, const float* rel /*24xE*/, int E, const float* gate, const float* sink,
                       double* out, double* o32) {
  const double sc = 1.0 / std::sqrt(128.0);
  std::vector<double> s(nk);
  for (int h = 0; h < kHq; ++h) {
    const int g = h / kG;
    double M = -INFINITY;
    for (long long i = 0; i < nk; ++i) {
      const long long key = lo + i, d = pos - key;
      if (d < 0 || (!global && d >= 1024)) { s[i] = -INFINITY; continue; }
      const double* kr = Kd + (i * kHkv + g) * kD;
      double a = 0;
      for (int x = 0; x < kD; ++x) a += (double)q[h * kD + x] * kr[x];
      a *= sc;
      if (d < E) a += rel[h * E + d];
      s[i] = a, M = std::max(M, a);
    }
    double L = 0, o[kD] = {0};
    for (long long i = 0; i < nk; ++i) {
      if (s[i] == -INFINITY) continue;
      const double e = std::exp(s[i] - M);
      L += e;
      const double* vr = Vd + (i * kHkv + g) * kD;
      for (int x = 0; x < kD; ++x) o[x] += e * vr[x];
    }
    const double lse = M + std::log(L);
    const double ss = 1.0 / (1.0 + std::exp(-(lse - sink[h])));
    for (int x = 0; x < kD; ++x) {
      const double ov = o[x] / L;
      o32[h * kD + x] = ov;
      out[h * kD + x] = ov * (1.0 / (1.0 + std::exp(-(double)gate[h * kD + x]))) * ss;
    }
  }
}
static double dec(const uint8_t* p, int kv, double scale) {
  return kv == KV_BF16 ? p6::bf16_to_f32(*(const uint16_t*)p) : p6::e4m3_to_f64(*p) / scale;
}
// gather session s's keys [lo, lo+nk) of a KvSet into decoded host arrays
static void gather(const KvSet& k, int s, long long lo, long long nk, std::vector<double>& Kd, std::vector<double>& Vd) {
  auto hK = d2h(k.K, k.bytes);
  auto hV = d2h(k.V, k.bytes);
  Kd.assign(nk * kHkv * kD, 0), Vd.assign(nk * kHkv * kD, 0);
  for (long long i = 0; i < nk; ++i)
    for (int g = 0; g < kHkv; ++g) {
      const long long a = k.lay.addr(k.hpt.data(), k.pt_stride, s, lo + i, g);
      for (int x = 0; x < kD; ++x) {
        Kd[(i * kHkv + g) * kD + x] = dec(&hK[a + x * k.lay.eb], k.kv, g_kscale);
        Vd[(i * kHkv + g) * kD + x] = dec(&hV[a + x * k.lay.eb], k.kv, g_vscale);
      }
    }
}
static std::vector<float> tof(const std::vector<uint16_t>& v) {
  std::vector<float> f(v.size());
  for (size_t i = 0; i < v.size(); ++i) f[i] = p6::bf16_to_f32(v[i]);
  return f;
}

// variant: 0 = design A, S > 0 = design B with an S-stage cp.async ring
static int g_var = 0;
static double g_wmul = 1.0;
// Physical worker count per (session, head): minimise ceil(waves) x ceil(chunks per worker) (bits do not depend on it)
static int pick_workers(int n, int nch, int slots, double wmul) {
  if (wmul <= 0) return nch;  // one CTA per chunk
  int best = 1;
  double bt = 1e30;
  for (int W = 1; W <= nch; ++W) {
    const double t = std::ceil((double)n * kHkv * W / slots) * std::ceil((double)nch / W);
    if (t < bt - 1e-9) bt = t, best = W;
  }
  return best;
}
// design A: each CTA's chunks are spread over its 4 warps (warp-level partials)
static int pick_workers_warp(int n, int nch, int slots, double wmul) {
  if (wmul < 0) return std::max(1, std::min(nch, (int)(-wmul)));  // explicit W
  if (wmul == 0) return (nch + kNW - 1) / kNW;
  int best = 1;
  double bt = 1e30;
  for (int W = 1; W <= nch; ++W) {
    const double t = std::ceil((double)n * kHkv * W / slots) * std::ceil(std::ceil((double)nch / W) / kNW);
    if (t < bt - 1e-9) bt = t, best = W;
  }
  return best;
}
static int ring_workers(int kv, int R, int S, int n, int maxchunks, double wmul) {
  return pick_workers(n, maxchunks, 74 * std::max(1, ring_ctas_per_sm(kv, R, S)), wmul);
}
static void run_var(int kv, const Params& p, int n, int gx, int var, double wmul, cudaStream_t st) {
  cudaError_t e = var == 0 ? da::launch(p, kv, n, pick_workers_warp(n, gx, 148, wmul), st)
                           : da::ring_launch(p, kv, n, ring_workers(kv, p.R, var, n, gx, wmul), var, st);
  if (e != cudaSuccess) p6::harness_fail(P, "launch", cudaGetErrorName(e));
}
static void run_once(int kv, const Params& p, int n, int gx, cudaStream_t st) { run_var(kv, p, n, gx, g_var, g_wmul, st); }
static void prep_all(int kv, int R) {
  prepare(kv, R);
  for (int S : {2, 3, 4, 6, 8}) ring_prepare(kv, R, S);
}

// ------------------------------------------------------------------------------------------------ modes
static void mode_info(p6::Ctx& cx) {
  for (int kv : {0, 1})
    for (int R : {1, 8}) {
      cudaFuncAttributes fa{};
      void* f = pick(kv, R);
      prep_all(kv, R);
      cudaFuncGetAttributes(&fa, f);
      int occ = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, kNW * 32, smem_bytes());
      p6::Line(P, "kernel_info").kv("kv", kv ? "e4m3" : "bf16").kv("R", R).kv("m16", kG * R > 8).kv("regs", fa.numRegs)
          .kv("local_bytes", (long long)fa.localSizeBytes).kv("smem_dyn", smem_bytes()).kv("ctas_per_sm", occ).emit();
      for (int S : {2, 3, 4, 6, 8}) {
        cudaFuncAttributes fb{};
        cudaFuncGetAttributes(&fb, ring_pick(kv, R, S));
        p6::Line(P, "kernel_info_ring").kv("kv", kv ? "e4m3" : "bf16").kv("R", R).kv("S", S).kv("warps", ring_nw(R))
            .kv("regs", fb.numRegs).kv("local_bytes", (long long)fb.localSizeBytes).kv("smem_dyn", ring_smem(kv, S))
            .kv("ctas_per_sm", ring_ctas_per_sm(kv, R, S)).emit();
      }
    }
  (void)cx;
}

// check: oracle on synthetic KV
static void mode_check(p6::Ctx& cx) {
  const auto Ls = cx.a.list("--L", "2000,44729");
  const auto ns = cx.a.list("--n", "1,4");
  const auto Rs = cx.a.list("--R", "1,3,8");
  const auto kvs = cx.a.slist("--kv", "bf16,e4m3");
  const auto kinds = cx.a.slist("--kind", "global,swa");
  const auto lays = cx.a.slist("--layout", "head,row");
  const int fcs = (int)cx.a.num("--cs", 0);
  for (auto& kind : kinds)
    for (auto& kvn : kvs)
      for (auto& ly : lays)
        for (auto L : Ls)
          for (auto n : ns)
            for (auto R : Rs) {
              const bool global = kind == "global";
              const int kv = kvn == "bf16" ? KV_BF16 : KV_E4M3;
              std::vector<int> ctx(n), sids(n);
              for (int s = 0; s < n; ++s) ctx[s] = (int)(L - 37 * s), sids[s] = s;
              prep_all(kv, (int)R);
              const size_t a0 = g_alloc;
              KvSet k = make_kv(kv, !global, ly == "head", (int)n, sids, L + 16, 0x6100, cx.s);
              KvSet kb;
              if (kv != KV_BF16) kb = make_kv(KV_BF16, !global, ly == "head", (int)n, sids, L + 16, 0x6100, cx.s);
              Rows m = make_rows((int)n, (int)R, ctx, sids, global, fcs, 0x9000, cx.s);
              Params p = make_params(k, m, global);
              run_once(kv, p, (int)n, m.grid_x, cx.s);
              cudaError_t e = cudaStreamSynchronize(cx.s);
              auto out1 = d2h(m.out, (size_t)n * R * kHq * kD);
              run_once(kv, p, (int)n, m.grid_x, cx.s);
              run_once(kv, p, (int)n, m.grid_x, cx.s);
              cudaError_t e2 = cudaStreamSynchronize(cx.s);
              auto out3 = d2h(m.out, out1.size());
              const bool det = std::memcmp(out1.data(), out3.data(), out1.size() * 2) == 0;
              auto o32 = d2h(m.out32, out1.size());
              auto hq = tof(d2h(m.q, out1.size())), hrel = tof(d2h(m.rel, (size_t)n * R * kHq * m.E));
              auto hg = tof(d2h(m.gate, out1.size())), hs = tof(d2h(m.sink, (size_t)n * R * kHq));
              Err eo, e32, eq;
              std::vector<int> ss = {0};
              if (n > 1) ss.push_back((int)n - 1);
              for (int s : ss) {
                const long long qlast = ctx[s] + R - 1, lo = global ? 0 : std::max(0LL, (long long)ctx[s] - 1023);
                std::vector<double> Kd, Vd, Kb, Vb;
                gather(k, s, lo, qlast - lo + 1, Kd, Vd);
                if (kv != KV_BF16) gather(kb, s, lo, qlast - lo + 1, Kb, Vb);
                std::vector<int> ts = {0};
                if (R > 1) ts.push_back((int)R - 1);
                for (int t : ts) {
                  const size_t row = (size_t)s * R + t;
                  std::vector<double> ref(kHq * kD), r32(kHq * kD), refb(kHq * kD), rb32(kHq * kD);
                  oracle_row(Kd.data(), Vd.data(), lo, qlast - lo + 1, ctx[s] + t, global, &hq[row * kHq * kD],
                             &hrel[row * kHq * m.E], m.E, &hg[row * kHq * kD], &hs[row * kHq], ref.data(), r32.data());
                  if (kv != KV_BF16)
                    oracle_row(Kb.data(), Vb.data(), lo, qlast - lo + 1, ctx[s] + t, global, &hq[row * kHq * kD],
                               &hrel[row * kHq * m.E], m.E, &hg[row * kHq * kD], &hs[row * kHq], refb.data(), rb32.data());
                  for (int i = 0; i < kHq * kD; ++i) {
                    eo.add(p6::bf16_to_f32(out1[row * kHq * kD + i]), ref[i]);
                    e32.add(o32[row * kHq * kD + i], r32[i]);
                    if (kv != KV_BF16) eq.add(p6::bf16_to_f32(out1[row * kHq * kD + i]), refb[i]);
                  }
                }
              }
              const double tol = kv == KV_BF16 ? 6e-3 : 1e-2;
              p6::Line l(P, "oracle");
              l.rt(e != cudaSuccess ? e : e2).kv("kind", kind).kv("kv", kvn).kv("layout", ly).kv("L", L).kv("n", n).kv("R", R)
                  .kv("grid_x", m.grid_x).kv("chunk_shift0", m.hcs[0]).kv("checked", (unsigned long long)eo.n)
                  .kv("rel_rms_err", eo.rel()).kv("max_abs_err", eo.mx).kv("pregate_rel_rms_err", e32.rel())
                  .kv("deterministic_3_runs", det).kv("tol", tol).kv("pass", e == cudaSuccess && det && eo.rel() <= tol);
              if (kv != KV_BF16) l.kv("vs_bf16_truth_rel_rms", eq.rel()).kv("vs_bf16_truth_max_abs", eq.mx);
              l.emit();
              (void)a0;  // buffers are leaked on purpose (check mode is short; no frees of GB ranges at run time)
            }
}

// inv: session 0's rows bit-identical alone (n=1), in n=4 and n=8 batches (other sessions at other contexts), and
// row 0 at R = 1 vs R = 8. Also a permuted batch where session 0's data sit in slot 3.
static void mode_inv(p6::Ctx& cx) {
  const auto Ls = cx.a.list("--L", "2000,44729,131072");
  const auto kvs = cx.a.slist("--kv", "bf16,e4m3");
  const auto kinds = cx.a.slist("--kind", "global,swa");
  const int R8 = (int)cx.a.num("--R", 8);
  for (auto& kind : kinds)
    for (auto& kvn : kvs)
      for (auto L : Ls) {
        const bool global = kind == "global";
        const int kv = kvn == "bf16" ? KV_BF16 : KV_E4M3;
        struct Case { int n, R, slot; bool ragged; };
        std::vector<Case> cases = {{1, R8, 0, false}, {4, R8, 0, false}, {8, R8, 0, false}, {4, R8, 3, false}, {1, 1, 0, false},
                                   {8, 1, 5, false}, {4, R8, 1, true}, {8, 1, 2, true}};
        std::vector<uint16_t> ref8, ref1;
        bool all = true, ok = true;
        std::string detail;
        for (auto& cs : cases) {
          std::vector<int> ctx(cs.n), sids(cs.n);
          for (int s = 0; s < cs.n; ++s) {
            const int id = s == cs.slot ? 0 : 100 + s;
            sids[s] = id;
            ctx[s] = id == 0 ? (int)L : (int)std::max(64LL, L / (1 + s) + 811 * s);
          }
          // ragged: rows allocated at width R8 per session, session under test has width cs.R, others 8/3/1/5
          const int Ralloc = cs.ragged ? R8 : cs.R;
          if (cs.ragged) {
            const int pat[4] = {8, 3, 1, 5};
            g_widths.assign(cs.n, 0);
            for (int s = 0; s < cs.n; ++s) g_widths[s] = s == cs.slot ? cs.R : std::min(R8, pat[s % 4]);
          }
          long long mk = 0;
          for (int c : ctx) mk = std::max(mk, (long long)c + Ralloc + 16);
          prep_all(kv, Ralloc);
          std::vector<long long> kend(cs.n);
          for (int s = 0; s < cs.n; ++s) kend[s] = (long long)ctx[s] + Ralloc + 16;
          KvSet k = make_kv(kv, !global, true, cs.n, sids, mk, 0x6100, cx.s, -1, &kend);
          Rows m = make_rows(cs.n, Ralloc, ctx, sids, global, 0, 0x9000, cx.s);
          g_widths.clear();
          Params p = make_params(k, m, global);
          const int vsmall = (int)cx.a.num("--stages-n1", g_var);
          run_var(kv, p, cs.n, m.grid_x, cs.n == 1 ? vsmall : g_var, g_wmul, cx.s);
          ok &= cudaStreamSynchronize(cx.s) == cudaSuccess;
          auto out = d2h(m.out, (size_t)cs.n * Ralloc * kHq * kD);
          std::vector<uint16_t> mine(out.begin() + (size_t)cs.slot * Ralloc * kHq * kD,
                                     out.begin() + ((size_t)cs.slot * Ralloc + cs.R) * kHq * kD);
          bool same = true;
          if (cs.R == R8) {
            if (ref8.empty()) ref8 = mine;
            else same = std::memcmp(ref8.data(), mine.data(), mine.size() * 2) == 0;
          } else {
            if (ref1.empty()) ref1 = mine;
            else same = std::memcmp(ref1.data(), mine.data(), mine.size() * 2) == 0;
            // row 0 at R=1 vs row 0 at R=8
            same = same && std::memcmp(ref8.data(), mine.data(), (size_t)kHq * kD * 2) == 0;
          }
          all &= same;
          detail += "n" + std::to_string(cs.n) + "R" + std::to_string(cs.R) + "slot" + std::to_string(cs.slot) + (cs.ragged ? "ragged" : "") + (same ? ":same " : ":DIFF ");
          cudaFree(k.K), cudaFree(k.V);
          g_alloc -= 2 * k.bytes;
        }
        p6::Line(P, "invariance").kv("ok", ok).kv("kind", kind).kv("kv", kvn).kv("L", L).kv("bit_identical", all)
            .kv("detail", detail).kv("compare", "session 0 rows vs n=1 R=8 alone; R=1 row 0 vs R=8 row 0").emit();
      }
}

// ---------------------------------------------------------------------------------------------- timing
static std::vector<int> visit_kinds() {
  std::vector<int> layers, k;
  for (int l = 0; l <= 5; ++l) layers.push_back(l);
  for (int r = 0; r < 3; ++r)
    for (int l = 6; l <= 17; ++l) layers.push_back(l);
  for (int l = 18; l <= 23; ++l) layers.push_back(l);
  for (int l : layers) k.push_back(l == 5 || l == 11 || l == 17 || l == 23);
  return k;
}
struct ArmCfg {
  int kvg = KV_BF16, kvw = KV_BF16, R = 8, csg = 0, csw = 0, varg = 0, varw = 0;
  std::vector<int> widths;
  double wmul = 1.0;
  bool hm = true;
  std::string name;
};
static ArmCfg parse_arm(const std::string& s) {  // kvg[/kvw]:layout:R[:csg:csw]
  ArmCfg a;
  a.name = s;
  std::vector<std::string> f;
  size_t p0 = 0;
  while (p0 <= s.size()) {
    size_t q = s.find(':', p0);
    if (q == std::string::npos) q = s.size();
    f.push_back(s.substr(p0, q - p0));
    p0 = q + 1;
  }
  auto kvof = [](const std::string& x) { return x == "e4m3" ? KV_E4M3 : KV_BF16; };
  const size_t sl = f[0].find('/');
  a.kvg = kvof(sl == std::string::npos ? f[0] : f[0].substr(0, sl));
  a.kvw = kvof(sl == std::string::npos ? f[0] : f[0].substr(sl + 1));
  if (f.size() > 1) a.hm = f[1] != "row";
  if (f.size() > 2) {
    a.R = 0;
    size_t q0 = 0;
    while (q0 < f[2].size()) {
      size_t q1 = f[2].find('x', q0);
      if (q1 == std::string::npos) q1 = f[2].size();
      const int w = std::atoi(f[2].substr(q0, q1 - q0).c_str());
      a.widths.push_back(w), a.R = std::max(a.R, w);
      q0 = q1 + 1;
    }
    if (a.widths.size() == 1) a.widths.clear();
  }
  if (f.size() > 3) a.csg = std::atoi(f[3].c_str());
  if (f.size() > 4) a.csw = std::atoi(f[4].c_str());
  // field 5: variant for global[/SWA]: A or B<S> (e.g. B4, B4/A); field 6: worker multiplier
  auto vof = [](const std::string& x) { return x.empty() || x[0] == 'A' ? 0 : std::atoi(x.c_str() + 1); };  // A | B<S>
  if (f.size() > 5) {
    const size_t q = f[5].find('/');
    a.varg = vof(q == std::string::npos ? f[5] : f[5].substr(0, q));
    a.varw = vof(q == std::string::npos ? f[5] : f[5].substr(q + 1));
  }
  if (f.size() > 6) a.wmul = std::atof(f[6].c_str());
  return a;
}
struct Built {
  cudaGraphExec_t exec = nullptr;
  double bytes = 0;
  int dg = 0, gxg = 0, gxw = 0;
};
static std::map<std::string, KvSet>* g_kv = nullptr;
static KvSet& kv_for(int kv, bool ring, bool hm, int n, const std::vector<int>& sids, long long mk, int idx, cudaStream_t st) {
  char key[96];
  std::snprintf(key, sizeof key, "%d-%d-%d-%d-%lld-%d", kv, (int)ring, (int)hm, n, mk, idx);
  auto it = g_kv->find(key);
  if (it != g_kv->end()) return it->second;
  const int de = g_kvdump.empty() ? -1 : ring ? dump_swa(idx % 40) : kDumpGlobal[idx % 8];
  return (*g_kv)[key] = make_kv(kv, ring, hm, n, sids, mk, (ring ? 0x5300 : 0x6100) + idx * 0x1000ull, st, de);
}
static double kv_bytes_read(bool global, int kv, const std::vector<int>& ctx, int R) {
  double b = 0;
  for (int c : ctx) {
    const long long qlast = c + R - 1, lo = global ? 0 : std::max(0, c - 1023);
    b += (double)((qlast - (lo & ~15)) / 16 + 1) * 16 * 2 * kHkv * kD * (kv == KV_BF16 ? 2 : 1);
  }
  return b;
}
// kinds: 0 = global only (8 execs), 1 = SWA only (40), 2 = step (8 + 40 in visit order)
static Built build(p6::Ctx& cx, const ArmCfg& a, int n, long long L, int what, double budget_gb, std::vector<Rows>& keep) {
  Built b;
  std::vector<int> ctx(n), sids(n);
  for (int s = 0; s < n; ++s) ctx[s] = (int)L, sids[s] = s;
  const long long mk = L + 8 + 16;  // independent of R: R = 1 and R = 8 arms share the KV sets
  const double per = 2.0 * n * (double)mk * kHkv * kD * (a.kvg == KV_BF16 ? 2 : 1);
  b.dg = std::max(1, std::min(8, (int)std::floor(budget_gb * 1e9 / per)));
  g_widths = a.widths;
  Rows mg = make_rows(n, a.R, ctx, sids, true, a.csg, 0x9000, cx.s);
  Rows mw = make_rows(n, a.R, ctx, sids, false, a.csw, 0x9100, cx.s);
  g_widths.clear();
  if (!g_kvdump.empty()) {
    auto setq = [&](Rows& m, int e) {
      DumpExec& d = dump_exec(e);
      std::vector<uint16_t> hq((size_t)n * a.R * kHq * kD);
      for (int r = 0; r < n * a.R; ++r)
        std::memcpy(&hq[(size_t)r * kHq * kD], &d.q[(size_t)(r % a.R) * kHq * kD], kHq * kD * 2);
      cudaMemcpy(m.q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice);
    };
    setq(mg, kDumpGlobal[0]);
    setq(mw, dump_swa(0));
  }
  keep.push_back(mg), keep.push_back(mw);
  b.gxg = mg.grid_x, b.gxw = mw.grid_x;
  prep_all(a.kvg, a.R), prep_all(a.kvw, a.R);
  auto kinds = visit_kinds();
  int gi = 0, wi = 0;
  std::vector<std::pair<Params, int>> calls;  // params, kv
  std::vector<int> vars;
  std::vector<int> isg;
  for (int v = 0; v < 48; ++v) {
    const bool g = kinds[v];
    if (g && what == 1) continue;
    if (!g && what == 0) continue;
    if (g) {
      KvSet& k = kv_for(a.kvg, false, a.hm, n, sids, mk, gi % b.dg, cx.s);
      calls.push_back({make_params(k, mg, true), a.kvg});
      vars.push_back(a.varg);
      b.bytes += kv_bytes_read(true, a.kvg, ctx, a.R);
      ++gi;
    } else {
      KvSet& k = kv_for(a.kvw, true, a.hm, n, sids, mk, wi, cx.s);
      calls.push_back({make_params(k, mw, false), a.kvw});
      vars.push_back(a.varw);
      b.bytes += kv_bytes_read(false, a.kvw, ctx, a.R);
      ++wi;
    }
    isg.push_back(g);
  }
  // warm-up outside capture
  for (size_t i = 0; i < calls.size(); ++i) run_var(calls[i].second, calls[i].first, n, isg[i] ? b.gxg : b.gxw, vars[i], a.wmul, cx.s);
  P6_CK(P, cudaStreamSynchronize(cx.s));
  cudaGraph_t gr;
  P6_CK(P, cudaStreamBeginCapture(cx.s, cudaStreamCaptureModeThreadLocal));
  for (size_t i = 0; i < calls.size(); ++i) run_var(calls[i].second, calls[i].first, n, isg[i] ? b.gxg : b.gxw, vars[i], a.wmul, cx.s);
  P6_CK(P, cudaStreamEndCapture(cx.s, &gr));
  P6_CK(P, cudaGraphInstantiate(&b.exec, gr, 0));
  cudaGraphDestroy(gr);
  return b;
}
static void mode_timed(p6::Ctx& cx, int what) {
  const auto arms_s = cx.a.slist("--arms", "bf16:head:8");
  const int n = (int)cx.a.num("--n", 4);
  const long long L = (long long)cx.a.num("--L", 44729);
  const double budget = cx.a.num("--kv-budget-gb", 40.0);
  std::vector<Built> built;
  std::vector<std::vector<Rows>> keeps(arms_s.size());
  std::vector<ArmCfg> arms;
  for (size_t i = 0; i < arms_s.size(); ++i) {
    arms.push_back(parse_arm(arms_s[i]));
    built.push_back(build(cx, arms.back(), n, L, what, budget / arms_s.size(), keeps[i]));
    p6::Line(P, "arm").kv("ok", true).kv("arm", arms_s[i]).kv("what", what == 0 ? "global8" : what == 1 ? "swa40" : "step")
        .kv("n", n).kv("L", L).kv("distinct_global_sets", built.back().dg).kv("kv_bytes", built.back().bytes)
        .kv("grid_x_global", built.back().gxg).kv("grid_x_swa", built.back().gxw).emit();
  }
  const bool sandwich = cx.a.num("--sandwich", 1) > 0 && arms.size() > 1;
  std::vector<p6::Slot> order;
  if (sandwich) order = p6::sandwich((int)arms.size(), cx.rounds, 0);
  else
    for (int r = 0; r < cx.rounds; ++r)
      for (int i = 0; i < (int)arms.size(); ++i) order.push_back({r, i, i, false});
  for (auto& sl : order) {
    Built& b = built[sl.arm];
    p6::Line l(P, what == 2 ? "step" : "exec");
    l.kv("arm", arms[sl.arm].name).kv("what", what == 0 ? "global8" : what == 1 ? "swa40" : "step").kv("round", sl.round)
        .kv("pos", sl.pos).kv("sandwich_tail", sl.sandwich_tail).kv("n", n).kv("L", L).kv("kv_bytes", b.bytes);
    auto r = p6::sustained(cx, l, [&](cudaStream_t st) { cudaGraphLaunch(b.exec, st); }, b.bytes / 1e9, 2, "GB");
    l.kv("ms_per_graph", r.ms_per_call_median);
    l.emit();
  }
}

// ---------------------------------------------------------------------------------------------- real KV
static std::vector<uint8_t> readf(const std::string& f) {
  FILE* fp = std::fopen(f.c_str(), "rb");
  if (!fp) p6::harness_fail(P, "open", f.c_str());
  std::fseek(fp, 0, SEEK_END);
  const long sz = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::vector<uint8_t> v(sz);
  if (std::fread(v.data(), 1, sz, fp) != (size_t)sz) p6::harness_fail(P, "read", f.c_str());
  std::fclose(fp);
  return v;
}
static long long jnum(const std::string& line, const char* key) {
  const std::string k = std::string("\"") + key + "\": ";
  const size_t p = line.find(k);
  return p == std::string::npos ? -1 : std::atoll(line.c_str() + p + k.size());
}
static void mode_realkv(p6::Ctx& cx) {
  const std::string dir = cx.a.get("--dump", "/dump");
  const int maxe = (int)cx.a.num("--max-exec", 48);
  const auto kvs = cx.a.slist("--kv", "bf16,e4m3");
  FILE* fp = std::fopen((dir + "/meta.jsonl").c_str(), "r");
  if (!fp) p6::harness_fail(P, "meta", dir.c_str());
  char buf[4096];
  int ex = 0;
  Err tot_ours[2], tot_cfgb, tot_vs_cfgb;
  while (std::fgets(buf, sizeof buf, fp) && ex < maxe) {
    const std::string line(buf);
    const int e = (int)jnum(line, "exec");
    const bool local = line.find("\"local\": true") != std::string::npos;
    const long long seq = jnum(line, "seq_len"), lo = jnum(line, "lo"), nk = jnum(line, "keys");
    const int T = (int)jnum(line, "T");
    char pre[512];
    std::snprintf(pre, sizeof pre, "%s/e%02d.", dir.c_str(), e);
    auto Kb = readf(std::string(pre) + "k.bin"), Vb = readf(std::string(pre) + "v.bin");
    auto qb = readf(std::string(pre) + "q.bin"), rb = readf(std::string(pre) + "rel.bin");
    auto gb = readf(std::string(pre) + "gate.bin"), sb = readf(std::string(pre) + "sink.bin"), ob = readf(std::string(pre) + "out.bin");
    const uint16_t* K16 = (const uint16_t*)Kb.data();
    const uint16_t* V16 = (const uint16_t*)Vb.data();
    // decoded bf16 truth
    std::vector<double> Kd(nk * kHkv * kD), Vd(nk * kHkv * kD);
    for (size_t i = 0; i < Kd.size(); ++i) Kd[i] = p6::bf16_to_f32(K16[i]), Vd[i] = p6::bf16_to_f32(V16[i]);
    auto vec16 = [](const std::vector<uint8_t>& b) { std::vector<uint16_t> v(b.size() / 2); std::memcpy(v.data(), b.data(), b.size()); return v; };
    auto hq = tof(vec16(qb)), hrel = tof(vec16(rb)), hg = tof(vec16(gb)), hs = tof(vec16(sb));
    auto hout_cfgb = vec16(ob);
    const int R = T;
    const long long ctx0 = seq - T;
    // oracle (all rows, all heads), threads over rows
    std::vector<double> ref((size_t)T * kHq * kD), r32((size_t)T * kHq * kD);
    {
      std::vector<std::thread> th;
      for (int t = 0; t < T; ++t)
        th.emplace_back([&, t] {
          oracle_row(Kd.data(), Vd.data(), lo, nk, ctx0 + t, !local, &hq[(size_t)t * kHq * kD], &hrel[(size_t)t * kHq * 1024],
                     1024, &hg[(size_t)t * kHq * kD], &hs[(size_t)t * kHq], &ref[(size_t)t * kHq * kD], &r32[(size_t)t * kHq * kD]);
        });
      for (auto& x : th) x.join();
    }
    Err ecfgb;
    for (size_t i = 0; i < ref.size(); ++i) ecfgb.add(p6::bf16_to_f32(hout_cfgb[i]), ref[i]), tot_cfgb.add(p6::bf16_to_f32(hout_cfgb[i]), ref[i]);
    p6::Line l(P, "realkv");
    l.kv("ok", true).kv("exec", e).kv("local", local).kv("seq_len", seq).kv("keys", nk).kv("cfgb_rel_rms_vs_oracle", ecfgb.rel())
        .kv("cfgb_max_abs_vs_oracle", ecfgb.mx);
    for (auto& kvn : kvs) {
      const int kv = kvn == "bf16" ? KV_BF16 : KV_E4M3;
      // one session; KV placed through the engine layout (head-major, shuffled pages / ring)
      KvSet k;
      k.kv = kv, k.n = 1;
      k.lay.init(local, true, kv == KV_BF16 ? 2 : 1);
      {
        k.pt_stride = (int)((seq + 1023) >> 10) + 1;
        int npages = 0;
        k.hpt = make_pt(local, 1, k.pt_stride, 0x77ull + e, npages);  // SWA: 3 ring pages through the page table
        k.bytes = (size_t)npages * k.lay.page_stride;
        k.pt = (int*)dalloc(k.hpt.size() * 4);
        cudaMemcpy(k.pt, k.hpt.data(), k.hpt.size() * 4, cudaMemcpyHostToDevice);
      }
      std::vector<uint8_t> hK(k.bytes, 0), hV(k.bytes, 0);
      for (long long i = 0; i < nk; ++i)
        for (int g = 0; g < kHkv; ++g) {
          const long long a = k.lay.addr(k.hpt.data(), k.pt_stride, 0, lo + i, g);
          for (int x = 0; x < kD; ++x) {
            const uint16_t kv16 = K16[(i * kHkv + g) * kD + x], vv16 = V16[(i * kHkv + g) * kD + x];
            if (kv == KV_BF16) {
              std::memcpy(&hK[a + 2 * x], &kv16, 2), std::memcpy(&hV[a + 2 * x], &vv16, 2);
            } else {
              hK[a + x] = f32_to_e4m3(p6::bf16_to_f32(kv16) * g_kscale);
              hV[a + x] = f32_to_e4m3(p6::bf16_to_f32(vv16) * g_vscale);
            }
          }
        }
      k.K = (uint8_t*)dalloc(k.bytes), k.V = (uint8_t*)dalloc(k.bytes);
      cudaMemcpy(k.K, hK.data(), k.bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(k.V, hV.data(), k.bytes, cudaMemcpyHostToDevice);
      std::vector<int> ctx = {(int)ctx0}, sids = {0};
      Rows m = make_rows(1, R, ctx, sids, !local, 0, 1, cx.s);
      cudaMemcpy(m.q, qb.data(), qb.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(m.rel, rb.data(), rb.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(m.gate, gb.data(), gb.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(m.sink, sb.data(), sb.size(), cudaMemcpyHostToDevice);
      prep_all(kv, R);
      Params p = make_params(k, m, !local);
      run_once(kv, p, 1, m.grid_x, cx.s);
      cudaError_t ce = cudaStreamSynchronize(cx.s);
      auto out = d2h(m.out, (size_t)T * kHq * kD);
      auto o32 = d2h(m.out32, (size_t)T * kHq * kD);
      Err eo, ec, e32;
      for (size_t i = 0; i < r32.size(); ++i) e32.add(o32[i], r32[i]);
      for (size_t i = 0; i < ref.size(); ++i) {
        eo.add(p6::bf16_to_f32(out[i]), ref[i]);
        tot_ours[kv].add(p6::bf16_to_f32(out[i]), ref[i]);
        if (kv == KV_BF16) ec.add(p6::bf16_to_f32(out[i]), p6::bf16_to_f32(hout_cfgb[i])), tot_vs_cfgb.add(p6::bf16_to_f32(out[i]), p6::bf16_to_f32(hout_cfgb[i]));
      }
      // bit fingerprint of this execution's outputs (bf16 out and fp32 pre-epilogue o): FNV-1a 64
      uint64_t hb = 1469598103934665603ull, h32 = 1469598103934665603ull;
      for (uint16_t x : out) hb = (hb ^ x) * 1099511628211ull;
      for (float x : o32) { uint32_t u; std::memcpy(&u, &x, 4); h32 = (h32 ^ u) * 1099511628211ull; }
      char hx[40];
      std::snprintf(hx, sizeof hx, "%016llx/%016llx", (unsigned long long)hb, (unsigned long long)h32);
      const std::string pf = kvn + "_";
      l.kv((pf + "out_hash").c_str(), hx);
      l.kv((pf + "cuda_ok").c_str(), ce == cudaSuccess).kv((pf + "rel_rms_vs_oracle").c_str(), eo.rel())
          .kv((pf + "max_abs_vs_oracle").c_str(), eo.mx).kv((pf + "pregate_rel_rms_vs_oracle").c_str(), e32.rel());
      if (kv == KV_BF16) l.kv("bf16_rel_rms_vs_cfgb", ec.rel()).kv("bf16_max_abs_vs_cfgb", ec.mx);
      cudaFree(k.K), cudaFree(k.V);
      if (k.pt) cudaFree(k.pt);
      g_alloc -= 2 * k.bytes;
    }
    l.emit();
    ++ex;
  }
  std::fclose(fp);
  p6::Line(P, "realkv_total").kv("ok", true).kv("execs", ex).kv("cfgb_rel_rms_vs_oracle", tot_cfgb.rel())
      .kv("bf16_rel_rms_vs_oracle", tot_ours[0].rel()).kv("e4m3_rel_rms_vs_oracle", tot_ours[1].rel())
      .kv("bf16_rel_rms_vs_cfgb", tot_vs_cfgb.rel()).kv("cfgb_max_abs", tot_cfgb.mx).kv("bf16_max_abs", tot_ours[0].mx)
      .kv("e4m3_max_abs", tot_ours[1].mx).kv("e4m3_scales", std::to_string(g_kscale) + "/" + std::to_string(g_vscale)).emit();
}

int main(int argc, char** argv) {
  p6::Ctx cx;
  cx.probe = P;
  cx.a = {argc, argv};
  g_cap_gb = cx.a.num("--mem-cap-gb", 56.0);
  g_kscale = (float)cx.a.num("--kscale", 1.0);
  g_vscale = (float)cx.a.num("--vscale", 1.0);
  g_var = (int)cx.a.num("--stages", 0);
  g_cs_swa = (int)cx.a.num("--cs-swa", 0);
  if (const char* v = std::getenv("DA_VERSION")) p6::Line(P, "version").kv("component", "decode_attn").kv("version", v).emit();
  g_kvdump = cx.a.get("--kvdump", "");
  std::map<int, DumpExec> dumpc;
  g_dumpc = &dumpc;
  g_wmul = cx.a.num("--wmul", 1.0);
  std::map<std::string, KvSet> cache;
  g_kv = &cache;
  p6::open_device(cx);
  const std::string mode = cx.a.get("--mode", "info");
  if (mode == "info") mode_info(cx);
  else if (mode == "check") mode_check(cx);
  else if (mode == "inv") mode_inv(cx);
  else if (mode == "exec_g") mode_timed(cx, 0);
  else if (mode == "exec_w") mode_timed(cx, 1);
  else if (mode == "step") mode_timed(cx, 2);
  else if (mode == "realkv") mode_realkv(cx);
  else p6::harness_fail(P, "mode", mode.c_str());
  p6::Line(P, "done").kv("ok", true).kv("device_bytes_allocated", (unsigned long long)g_alloc).emit();
  cx.tele->stop();
  return 0;
}
