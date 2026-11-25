// S4 prefill expert (MoE) kernel: standalone C++/CUDA bench and correctness harness (our code), with config B's
// Humming cubins as the baseline arm (humming_launch.hpp, Apache-2.0 attribution there).
//
// Modes (JSON lines on stdout, first line = meta):
//   info   registers / smem / occupancy of our kernel variants and the Humming cubins used
//   stats  rows per expert of the real routing for the chunk cells
//   check  FP64 oracle (exact decoded checkpoint weights, real x and routing) on sampled tokens, ours and Humming
//   inv    invariance: chunking and batch composition (bit compare of every token's output rows)
//   time   sustained timing (>= 20 s per arm, first 2 s dropped, 10 Hz NVML, energy counter), Humming sandwiched
//
// Data: layer weights straight from the checkpoint (ours: repacked here; Humming: its own repacked tensors from the
// H1 dump of the same checkpoint), MoE inputs x and routing (ids, weights) captured by capture_routing.py with the
// S1 reference on real text (job Q's pool). One call = one chunk-layer: w13 (+SiTU), w2, combine of Q tokens.
// FLOP convention: 2 x 8 Q x (2304 x 2304 + 2304 x 1152) per call (valid rows only; no padding counted).
#include "harness.hpp"
#include "humming_launch.hpp"
#include "prefill_moe.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <numeric>
#include <random>
#include <thread>

using p6::Line;
using hum::SafeTensors;
static const char* PROBE = "s4_prefill_moe";
#define CK(x) P6_CK(PROBE, x)
using namespace pmoe;

static double flop_per_token() { return 2.0 * TOPK * ((double)H * 2 * I + (double)H * I); }

// ------------------------------------------------------------------------------------------------ device buffers
struct Buf {
  void* p = nullptr;
  size_t n = 0;
  Buf() = default;
  explicit Buf(size_t bytes) { alloc(bytes); }
  void alloc(size_t bytes) { free(); n = bytes; if (bytes) CK(cudaMalloc(&p, bytes)); }
  void free() { if (p) cudaFree(p); p = nullptr; n = 0; }
  ~Buf() { free(); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  template <class T> T* as() const { return reinterpret_cast<T*>(p); }
  void up(const void* h, size_t bytes) { if (n < bytes) alloc(bytes); CK(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice)); }
  template <class T> void up(const std::vector<T>& v) { up(v.data(), v.size() * sizeof(T)); }
  template <class T> std::vector<T> down(cudaStream_t s, size_t count) const {
    std::vector<T> v(count);
    CK(cudaStreamSynchronize(s));            // S4 attention lesson: sync the non-blocking stream before the copy
    CK(cudaMemcpy(v.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};

static inline float bf(uint16_t h) { return p6::bf16_to_f32(h); }

// ------------------------------------------------------------------------------------------------ formats (host)
static const double E2M1V[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};

// e4m3 scale code (>= 0) -> our byte: ((E + 9) << 3) | mantissa3 of the normalised value 2^E (1 + m / 8)
static uint8_t enc_scale(uint8_t c) {
  int e = (c >> 3) & 15, m = c & 7;
  int E, mm;
  if (e == 0) {
    REQUIRE(m != 0, "zero scale must be handled by the caller");
    if (m == 1) { E = -9; mm = 0; }
    else if (m < 4) { E = -8; mm = (m - 2) * 4; }
    else { E = -7; mm = (m - 4) * 2; }
  } else { E = e - 7; mm = m; }
  REQUIRE(E >= -9 && E <= 8, "scale exponent range");
  return (uint8_t)(((E + 9) << 3) | mm);
}

struct LayerW {
  int layer = 0, fmt = NVFP4;
  // checkpoint (host, for the oracle and the repack)
  std::vector<uint8_t> w[2], s[2];       // [proj]
  std::vector<float> fac[2];             // NVFP4 per-expert factor (fp32 amax / denominator)
  // ours (device); Wn / Sn = NVFP4 in the n1 layout
  Buf W[2], S[2], F[2], Wn[2], Sn[2];
  // Humming (device)
  Buf hW[2], hS[2], hS2[2];
  bool has_hs2 = false;
  int K(int pj) const { return pj == 0 ? H : I; }
  int N(int pj) const { return pj == 0 ? 2 * I : H; }
  // exact value of weight (e, n, k) of projection pj (fp64)
  double wv(int pj, int e, int n, int k) const {
    const int K_ = K(pj), N_ = N(pj);
    if (fmt == NVFP4) {
      uint8_t byte = w[pj][((size_t)e * N_ + n) * (K_ / 2) + k / 2];
      uint8_t code = (k & 1) ? byte >> 4 : byte & 15;
      double sc = p6::e4m3_to_f64(s[pj][((size_t)e * N_ + n) * (K_ / 16) + k / 16]);
      return E2M1V[code] * sc * (double)fac[pj][e];
    } else {
      double v = p6::e4m3_to_f64(w[pj][((size_t)e * N_ + n) * K_ + k]);
      int c = s[pj][((size_t)e * N_ + n) * (K_ / 32) + k / 32];
      return v * std::ldexp(1.0, c - 127);
    }
  }
};

// original row n of local row (mt, g, hh) in weight tile wt
static inline int orig_row(int pj, int wt, int mt, int g, int hh) {
  if (pj == 0) return 2 * (wt * 32 + mt * 8 + g) + hh;   // w13: (g) = gate i, (g + 8) = up i
  return wt * 64 + mt * 16 + g + 8 * hh;
}

static void repack(LayerW& L, int pj) {
  const int K_ = L.K(pj), N_ = L.N(pj), KS = K_ / BK, NWT = N_ / WT;
  if (L.fmt == NVFP4) {
    const size_t WB = FmtInfo<NVFP4>::WBYTES, SB = FmtInfo<NVFP4>::SBYTES;
    std::vector<uint8_t> W((size_t)E * NWT * KS * WB), S((size_t)E * NWT * KS * SB);
    size_t neg = 0, zero = 0, sub = 0;
    auto work = [&](int e0, int e1) {
      for (int e = e0; e < e1; ++e)
        for (int wt = 0; wt < NWT; ++wt)
          for (int ks = 0; ks < KS; ++ks) {
            uint8_t* wb = W.data() + (((size_t)e * NWT + wt) * KS + ks) * WB;
            uint8_t* sb = S.data() + (((size_t)e * NWT + wt) * KS + ks) * SB;
            for (int mt = 0; mt < 4; ++mt) {
              for (int g = 0; g < 8; ++g)
                for (int kk = 0; kk < 4; ++kk)
                  for (int hh = 0; hh < 2; ++hh) {
                    int n = orig_row(pj, wt, mt, g, hh);
                    uint8_t c = L.s[pj][((size_t)e * N_ + n) * (K_ / 16) + ks * 4 + kk] & 0x7f;
                    sb[(mt * 8 + g) * 8 + kk * 2 + hh] = c ? enc_scale(c) : 0;
                  }
              for (int lane = 0; lane < 32; ++lane) {
                int g = lane >> 2, t = lane & 3;
                for (int kk = 0; kk < 4; ++kk) {
                  uint32_t q = 0;
                  for (int j = 0; j < 4; ++j)
                    for (int h = 0; h < 2; ++h) {
                      int n = orig_row(pj, wt, mt, g, j & 1);
                      int k = ks * 64 + kk * 16 + 2 * t + h + 8 * (j >> 1);
                      uint8_t byte = L.w[pj][((size_t)e * N_ + n) * (K_ / 2) + k / 2];
                      uint32_t code = (k & 1) ? byte >> 4 : byte & 15;
                      uint8_t sc = L.s[pj][((size_t)e * N_ + n) * (K_ / 16) + k / 16];
                      if (sc & 0x80) code ^= 8;           // negative scale: flip the weight's sign (exact)
                      if ((sc & 0x7f) == 0) code = 0;     // zero scale: weight 0
                      q |= code << (h == 0 ? 12 - 4 * j : 28 - 4 * j);
                    }
                  std::memcpy(wb + ((mt * 32 + lane) * 4 + kk) * 4, &q, 4);
                }
              }
            }
          }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 8; ++i) th.emplace_back(work, i * 16, i * 16 + 16);
    for (auto& t : th) t.join();
    for (uint8_t c : L.s[pj]) { neg += (c & 0x80) != 0; zero += (c & 0x7f) == 0; sub += (c & 0x78) == 0 && (c & 7); }
    Line("repack", "nvfp4").kv("proj", pj).kv("neg_scales", (unsigned long long)neg)
        .kv("zero_scales", (unsigned long long)zero).kv("subnormal_scales", (unsigned long long)sub).emit();
    L.W[pj].up(W);
    L.S[pj].up(S);
    std::vector<float> F(E);
    for (int e = 0; e < E; ++e) F[e] = L.fac[pj][e] * 256.0f;
    L.F[pj].up(F);
    // n1 layout: same blocks; new nibble positions; scales as bf16 bits (S_g | S_h << 16) per (mt, g, kk)
    {
      const size_t SBn = FmtInfo<NVFP4B>::SBYTES;
      std::vector<uint8_t> Wn(W.size()), Sn((size_t)E * NWT * KS * SBn);
      auto work2 = [&](int e0, int e1) {
        static const int pos[4][4] = {{6, 7, 8, 15}, {3, 4, 5, 12}, {0, 1, 2, 9}, {10, 11, 13, 14}};
        for (int e = e0; e < e1; ++e)
          for (int wt = 0; wt < NWT; ++wt)
            for (int ks = 0; ks < KS; ++ks) {
              uint8_t* wb = Wn.data() + (((size_t)e * NWT + wt) * KS + ks) * WB;
              uint8_t* sb = Sn.data() + (((size_t)e * NWT + wt) * KS + ks) * SBn;
              for (int mt = 0; mt < 4; ++mt) {
                for (int g = 0; g < 8; ++g)
                  for (int kk = 0; kk < 4; ++kk) {
                    uint32_t wd = 0;
                    for (int hh = 0; hh < 2; ++hh) {
                      int n = orig_row(pj, wt, mt, g, hh);
                      uint8_t c = L.s[pj][((size_t)e * N_ + n) * (K_ / 16) + ks * 4 + kk] & 0x7f;
                      uint32_t bits = ((uint32_t)(c ? enc_scale(c) : 0) << 4) + 0x7600u;
                      wd |= bits << (16 * hh);
                    }
                    std::memcpy(sb + ((mt * 8 + g) * 4 + kk) * 4, &wd, 4);
                  }
                for (int lane = 0; lane < 32; ++lane) {
                  int g = lane >> 2, t = lane & 3;
                  for (int kk = 0; kk < 4; ++kk) {
                    uint32_t q = 0;
                    for (int j = 0; j < 4; ++j)
                      for (int h = 0; h < 2; ++h) {
                        int n = orig_row(pj, wt, mt, g, j & 1);
                        int k = ks * 64 + kk * 16 + 2 * t + h + 8 * (j >> 1);
                        uint8_t byte = L.w[pj][((size_t)e * N_ + n) * (K_ / 2) + k / 2];
                        uint32_t code = (k & 1) ? byte >> 4 : byte & 15;
                        uint8_t sc = L.s[pj][((size_t)e * N_ + n) * (K_ / 16) + k / 16];
                        if (sc & 0x80) code ^= 8;
                        if ((sc & 0x7f) == 0) code = 0;
                        for (int bt = 0; bt < 4; ++bt)
                          if (code >> bt & 1) q |= 1u << (pos[j][bt] + 16 * h);
                      }
                    std::memcpy(wb + ((mt * 32 + lane) * 4 + kk) * 4, &q, 4);
                  }
                }
              }
            }
      };
      std::vector<std::thread> th2;
      for (int i = 0; i < 8; ++i) th2.emplace_back(work2, i * 16, i * 16 + 16);
      for (auto& t : th2) t.join();
      L.Wn[pj].up(Wn);
      L.Sn[pj].up(Sn);
    }
  } else {
    const size_t WB = FmtInfo<MXFP8>::WBYTES, SB = FmtInfo<MXFP8>::SBYTES;
    std::vector<uint8_t> W((size_t)E * NWT * KS * WB), S((size_t)E * NWT * KS * SB);
    int cmin = 255, cmax = 0;
    for (uint8_t c : L.s[pj]) { cmin = std::min(cmin, (int)c); cmax = std::max(cmax, (int)c); }
    REQUIRE(cmin >= 40 && cmax <= 142, "e8m0 scale outside the exact range of the decode");
    auto work = [&](int e0, int e1) {
      for (int e = e0; e < e1; ++e)
        for (int wt = 0; wt < NWT; ++wt)
          for (int ks = 0; ks < KS; ++ks) {
            uint8_t* wb = W.data() + (((size_t)e * NWT + wt) * KS + ks) * WB;
            uint8_t* sb = S.data() + (((size_t)e * NWT + wt) * KS + ks) * SB;
            for (int mt = 0; mt < 4; ++mt) {
              for (int g = 0; g < 8; ++g)
                for (int grp = 0; grp < 2; ++grp)
                  for (int hh = 0; hh < 2; ++hh) {
                    int n = orig_row(pj, wt, mt, g, hh);
                    sb[(mt * 8 + g) * 4 + grp * 2 + hh] = L.s[pj][((size_t)e * N_ + n) * (K_ / 32) + ks * 2 + grp];
                  }
              for (int lane = 0; lane < 32; ++lane) {
                int g = lane >> 2, t = lane & 3;
                int ng = orig_row(pj, wt, mt, g, 0), nh = orig_row(pj, wt, mt, g, 1);
                for (int kk = 0; kk < 4; ++kk)
                  for (int half = 0; half < 2; ++half) {
                    int k = ks * 64 + kk * 16 + 2 * t + 8 * half;
                    auto at = [&](int n, int kx) { return L.w[pj][((size_t)e * N_ + n) * K_ + kx]; };
                    uint8_t* d = wb + ((mt * 32 + lane) * 4 + kk) * 8 + half * 4;
                    d[0] = at(nh, k); d[1] = at(ng, k); d[2] = at(nh, k + 1); d[3] = at(ng, k + 1);
                  }
              }
            }
          }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 8; ++i) th.emplace_back(work, i * 16, i * 16 + 16);
    for (auto& t : th) t.join();
    Line("repack", "mxfp8").kv("proj", pj).kv("e8m0_min", cmin).kv("e8m0_max", cmax).emit();
    L.W[pj].up(W);
    L.S[pj].up(S);
    std::vector<float> F(E, 256.0f);
    L.F[pj].up(F);
  }
}

static void load_layer(LayerW& L, int layer, const std::string& model, const std::string& h1) {
  L.layer = layer;
  L.fmt = layer < 20 ? NVFP4 : MXFP8;
  SafeTensors ck(model);
  const std::string p = "model.llm.layers." + std::to_string(layer) + ".mlp.experts.";
  const char* proj[2] = {"w13", "w2"};
  for (int pj = 0; pj < 2; ++pj) {
    L.w[pj] = ck.read(p + proj[pj] + "_weight");
    L.s[pj] = ck.read(p + proj[pj] + "_weight_scale");
    if (L.fmt == NVFP4) {
      auto am = ck.read(p + proj[pj] + "_weight_amax");
      auto de = ck.read(p + proj[pj] + "_weight_outer_scale_denominator");
      REQUIRE(am.size() == E * 4 && de.size() == 4, "amax / denominator shape");
      float d;
      std::memcpy(&d, de.data(), 4);
      L.fac[pj].resize(E);
      for (int e = 0; e < E; ++e) {
        float a;
        std::memcpy(&a, am.data() + 4 * e, 4);
        volatile float q = a / d;                     // fp32 division (config B / formats.nvfp4_factor)
        L.fac[pj][e] = q;
      }
    }
    size_t expect_w = (size_t)E * L.N(pj) * L.K(pj) / (L.fmt == NVFP4 ? 2 : 1);
    REQUIRE(L.w[pj].size() == expect_w, "checkpoint weight size");
    repack(L, pj);
    // Humming's own tensors (H1 dump of the same checkpoint)
    char fn[64];
    std::snprintf(fn, sizeof fn, "/weights/L%02d.safetensors", layer);
    SafeTensors hs(h1 + fn);
    L.hW[pj].up(hs.read(std::string(proj[pj]) + ".weight"));
    L.hS[pj].up(hs.read(std::string(proj[pj]) + ".weight_scale"));
    if (hs.has(std::string(proj[pj]) + ".weight_scale_2")) {
      auto s2 = hs.read(std::string(proj[pj]) + ".weight_scale_2");
      // cross-check: Humming's tensor scale == our factor, bit for bit
      REQUIRE(L.fmt == NVFP4 && std::memcmp(s2.data(), L.fac[pj].data(), E * 4) == 0,
              "Humming weight_scale_2 != amax / denominator");
      L.hS2[pj].up(s2);
      L.has_hs2 = true;
    }
  }
}

// ------------------------------------------------------------------------------------------------ chunks
struct Routing {             // one captured prompt, one visit
  int n = 0;
  std::vector<uint16_t> x;   // [n, H] bf16
  std::vector<int32_t> idx;  // [n, 8]
  std::vector<float> wts;    // [n, 8]
};
static Routing load_routing(const std::string& path, int visit) {
  SafeTensors st(path);
  char nm[32];
  Routing r;
  std::snprintf(nm, sizeof nm, "v%02d.idx", visit);
  auto b = st.read(nm);
  r.n = (int)(b.size() / 32);
  r.idx.resize((size_t)r.n * 8);
  std::memcpy(r.idx.data(), b.data(), b.size());
  std::snprintf(nm, sizeof nm, "v%02d.wts", visit);
  b = st.read(nm);
  r.wts.resize((size_t)r.n * 8);
  std::memcpy(r.wts.data(), b.data(), b.size());
  std::snprintf(nm, sizeof nm, "v%02d.x", visit);
  b = st.read(nm);
  REQUIRE(b.size() == (size_t)r.n * H * 2, "x size");
  r.x.resize((size_t)r.n * H);
  std::memcpy(r.x.data(), b.data(), b.size());
  return r;
}

// Humming's interval table (job E tunings_full, identical for L0 / L20 at these sizes): valid rows -> block_m
static int humming_block_m(int rows) {
  struct Iv { int lo, hi, bm; };
  static const Iv t[] = {{5504, 8192, 32},   {8192, 11136, 96},  {11136, 14848, 128}, {14848, 22144, 64},
                         {22144, 29568, 128}, {29568, 33280, 96}, {33280, 36864, 64},  {36864, 44288, 128},
                         {44288, 51712, 64},  {51712, 55296, 96}, {55296, 59008, 128}, {59008, 1 << 30, 64}};
  for (auto& iv : t)
    if (rows > iv.lo && rows <= iv.hi) return iv.bm;
  return 0;   // no prefill cubin mapped (the Humming arm cannot run this chunk)
}

struct Chunk {
  int Q = 0, R = 0;
  std::vector<uint16_t> x;            // [Q, H]
  std::vector<int32_t> idx;           // [Q, 8]
  std::vector<float> wts;             // [Q, 8]
  std::vector<int> perm, cnt, eoff;   // sorted position -> flat row; rows per expert; expert offsets
  // device
  Buf dx, dwts_flat, d_arow13, d_arow2, d_orow, d_gamma, d_items13, d_items2;
  int n_items13 = 0, n_items2 = 0;
  Buf h, y, out, gu;                  // ours: h sorted [R, I], y flat [R, H], out [Q, H], gu flat [R, 2I]
  Buf hc1, hact, hy, hout;            // Humming
  Buf h_sorted, h_eids, h_ntp;
  int hbm = 0, h_ntp_v = 0;
};

static std::vector<Item> make_items(const Chunk& c, int WF, int NWT, bool full_first, bool v3 = false) {
  std::vector<int> order(E);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return c.cnt[a] > c.cnt[b]; });
  std::vector<Item> it;
  const int FT = NWT / WF;
  for (int e : order) {
    int n = c.cnt[e];
    if (!n) continue;
    std::vector<std::pair<int, int>> tiles;     // (start, count)
    if (v3) {                                    // v3: 256-row tiles, then one 128-row tile, then the rest
      int p0 = 0;
      while (n - p0 >= 256) { tiles.push_back({p0, 256}); p0 += 256; }
      if (n - p0 > 128) { tiles.push_back({p0, 128}); p0 += 128; }
      if (n - p0 > 0) tiles.push_back({p0, n - p0});
    } else {
      int nt = (n + BN - 1) / BN;
      int size = full_first ? BN : ((n + nt - 1) / nt + 7) / 8 * 8;   // v2: full 128-row tiles, then the remainder
      for (int p0 = 0; p0 < n; p0 += size) tiles.push_back({p0, std::min(size, n - p0)});
    }
    for (int ft = 0; ft < FT; ++ft)
      for (auto& tl : tiles) it.push_back({e, ft, c.eoff[e] + tl.first, tl.second});
  }
  return it;
}

static void build_chunk(Chunk& c, const Routing& r, const std::vector<int>& tokens, int WF) {
  const bool full_first = WF < 0;   // WF < 0: v2 kernel (-2: 128-row CTA tiles, -4: 256-row), full tiles first
  const bool v3 = WF == -8;         // v3 kernel: 128-row CTA tiles, token tiles 256 / 128 / rest
  if (WF < 0) WF = v3 ? 2 : -WF;
  c.Q = (int)tokens.size();
  c.R = c.Q * TOPK;
  c.x.resize((size_t)c.Q * H);
  c.idx.resize((size_t)c.R);
  c.wts.resize((size_t)c.R);
  for (int i = 0; i < c.Q; ++i) {
    int t = tokens[i];
    std::memcpy(&c.x[(size_t)i * H], &r.x[(size_t)t * H], H * 2);
    std::memcpy(&c.idx[(size_t)i * 8], &r.idx[(size_t)t * 8], 32);
    std::memcpy(&c.wts[(size_t)i * 8], &r.wts[(size_t)t * 8], 32);
  }
  c.cnt.assign(E, 0);
  for (int v : c.idx) { REQUIRE(v >= 0 && v < E, "expert id"); c.cnt[v]++; }
  c.eoff.assign(E + 1, 0);
  for (int e = 0; e < E; ++e) c.eoff[e + 1] = c.eoff[e] + c.cnt[e];
  c.perm.assign(c.R, 0);
  std::vector<int> fill(c.eoff.begin(), c.eoff.end() - 1);
  for (int rr = 0; rr < c.R; ++rr) c.perm[fill[c.idx[rr]]++] = rr;   // ascending flat row within an expert
  std::vector<int> a13(c.R), a2(c.R), orow(c.R);
  std::vector<float> gam(c.R);
  for (int p = 0; p < c.R; ++p) { a13[p] = c.perm[p] / TOPK; a2[p] = p; orow[p] = c.perm[p]; gam[p] = c.wts[c.perm[p]]; }
  c.dx.up(c.x);
  c.dwts_flat.up(c.wts);
  c.d_arow13.up(a13);
  c.d_arow2.up(a2);
  c.d_orow.up(orow);
  c.d_gamma.up(gam);
  auto i13 = make_items(c, WF, (2 * I) / WT, full_first, v3), i2 = make_items(c, WF, H / WT, full_first, v3);
  c.n_items13 = (int)i13.size();
  c.n_items2 = (int)i2.size();
  c.d_items13.up(i13);
  c.d_items2.up(i2);
  c.h.alloc((size_t)c.R * I * 2);
  c.y.alloc((size_t)c.R * H * 2);
  c.out.alloc((size_t)c.Q * H * 2);
  c.gu.alloc((size_t)c.R * 2 * I * 2);
  c.hc1.alloc((size_t)c.R * 2 * I * 2);
  c.hact.alloc((size_t)c.R * I * 2);
  c.hy.alloc((size_t)c.R * H * 2);
  c.hout.alloc((size_t)c.Q * H * 2);
  // Humming alignment (vLLM moe_align_block_size layout, as the H1 dumps show it): per expert ascending flat rows,
  // padded to block_m with the sentinel R; expert id per block, -1 after the last; num_tokens_post_pad.
  c.hbm = humming_block_m(c.R);
  if (!c.hbm) return;
  const int bm = c.hbm;
  size_t maxp = (size_t)c.R + (size_t)E * (bm - 1);
  std::vector<int32_t> sid(maxp, c.R), eid(maxp / bm, -1);
  size_t pos = 0;
  for (int e = 0; e < E; ++e) {
    if (!c.cnt[e]) continue;
    for (int i = 0; i < c.cnt[e]; ++i) sid[pos + i] = c.perm[c.eoff[e] + i];
    int blocks = (c.cnt[e] + bm - 1) / bm;
    for (int b = 0; b < blocks; ++b) eid[pos / bm + b] = e;
    pos += (size_t)blocks * bm;
  }
  c.h_ntp_v = (int)pos;
  std::vector<int32_t> ntp = {c.h_ntp_v};
  c.h_sorted.up(sid);
  c.h_eids.up(eid);
  c.h_ntp.up(ntp);
}

// ------------------------------------------------------------------------------------------------ kernels / arms
// v1: "2x3" (WF x STAGES). v2: "v2s3" (4 warps, 3 stages), "v2w8s4" (8 warps, 4 stages), "...d1" / "...d2" ceiling arms
struct Variant { int WF, STAGES, WARPS = 4, DBG = 0, NIB = 0; };   // WF < 0: v2 kernel (moe_gemm2); NIB 1: n1 layout
static Variant parse_variant(const std::string& v) {
  if (v.rfind("v3", 0) == 0) return {-8, std::atoi(v.c_str() + v.find('s') + 1), 8, 0};
  if (v[0] == 'v') {
    Variant r{-2, 3, 4, 0};
    size_t pw = v.find('w'), ps = v.find('s'), pd = v.find('d');
    if (pw != std::string::npos) r.WARPS = std::atoi(v.c_str() + pw + 1);
    r.STAGES = std::atoi(v.c_str() + ps + 1);
    if (pd != std::string::npos) r.DBG = std::atoi(v.c_str() + pd + 1);
    if (v.find('n') != std::string::npos) r.NIB = 1;
    if (r.WARPS == 8) r.WF = -4;
    return r;
  }
  return {std::atoi(v.substr(0, v.find('x')).c_str()), std::atoi(v.substr(v.find('x') + 1).c_str())};
}
template <int FMT, int MODE, int ST, int WA, int DB>
static void launch_t2(const Params& p, cudaStream_t s) {
  moe_gemm2<FMT, MODE, ST, WA, DB><<<p.n_items, 32 * WA, Cfg2<FMT, MODE, ST, WA>::SMEM, s>>>(p);
}
template <int FMT, int MODE, int ST, int WA, int DB>
static void set_attr2() {
  CK(cudaFuncSetAttribute(moe_gemm2<FMT, MODE, ST, WA, DB>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg2<FMT, MODE, ST, WA>::SMEM));
}

template <int FMT, int MODE, int WF, int ST>
static void set_attr() {
  using Cf = Cfg<FMT, MODE, WF, ST>;
  CK(cudaFuncSetAttribute(moe_gemm<FMT, MODE, WF, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cf::SMEM));
}
template <int FMT, int MODE, int WF, int ST>
static void launch_t(const Params& p, cudaStream_t s) {
  using Cf = Cfg<FMT, MODE, WF, ST>;
  moe_gemm<FMT, MODE, WF, ST><<<p.n_items, Cf::THREADS, Cf::SMEM, s>>>(p);
}
template <int MODE, int WF, int ST>
static void launch_m(int fmt, const Params& p, cudaStream_t s) {
  if (fmt == NVFP4) launch_t<NVFP4, MODE, WF, ST>(p, s); else launch_t<MXFP8, MODE, WF, ST>(p, s);
}
template <int FMT, int MODE, int ST>
static void launch_t3(const Params& p, cudaStream_t s) {
  moe_gemm3<FMT, MODE, ST><<<p.n_items, 256, Cfg3<FMT, MODE, ST>::SMEM, s>>>(p);
}
template <int FMT, int MODE, int ST>
static void set_attr3() {
  CK(cudaFuncSetAttribute(moe_gemm3<FMT, MODE, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg3<FMT, MODE, ST>::SMEM));
}
static void launch_gemm(int fmt, int mode, Variant v, const Params& p, cudaStream_t s) {
  if (v.WF == -8) {
#define V3(ST_)                                                                                        \
    if (v.STAGES == ST_) {                                                                             \
      if (fmt == NVFP4) {                                                                              \
        if (mode == W13_SITU) launch_t3<NVFP4, W13_SITU, ST_>(p, s);                                   \
        else if (mode == W13_PLAIN) launch_t3<NVFP4, W13_PLAIN, ST_>(p, s);                            \
        else launch_t3<NVFP4, W2_PLAIN, ST_>(p, s);                                                    \
      } else {                                                                                         \
        if (mode == W13_SITU) launch_t3<MXFP8, W13_SITU, ST_>(p, s);                                   \
        else if (mode == W13_PLAIN) launch_t3<MXFP8, W13_PLAIN, ST_>(p, s);                            \
        else launch_t3<MXFP8, W2_PLAIN, ST_>(p, s);                                                    \
      }                                                                                                \
      return;                                                                                          \
    }
    V3(3) V3(4)
#undef V3
    hum::die("unknown v3 stages");
  }
  if (v.WF < 0) {
#define V2(ST_, WA_, DB_)                                                                              \
    if (v.STAGES == ST_ && v.WARPS == WA_ && v.DBG == DB_) {                                           \
      if (fmt == NVFP4 && v.NIB) {                                                                     \
        if (mode == W13_SITU) launch_t2<NVFP4B, W13_SITU, ST_, WA_, DB_>(p, s);                        \
        else if (mode == W13_PLAIN) launch_t2<NVFP4B, W13_PLAIN, ST_, WA_, DB_>(p, s);                 \
        else launch_t2<NVFP4B, W2_PLAIN, ST_, WA_, DB_>(p, s);                                         \
      } else if (fmt == NVFP4) {                                                                       \
        if (mode == W13_SITU) launch_t2<NVFP4, W13_SITU, ST_, WA_, DB_>(p, s);                         \
        else if (mode == W13_PLAIN) launch_t2<NVFP4, W13_PLAIN, ST_, WA_, DB_>(p, s);                  \
        else launch_t2<NVFP4, W2_PLAIN, ST_, WA_, DB_>(p, s);                                          \
      } else {                                                                                         \
        if (mode == W13_SITU) launch_t2<MXFP8, W13_SITU, ST_, WA_, DB_>(p, s);                         \
        else if (mode == W13_PLAIN) launch_t2<MXFP8, W13_PLAIN, ST_, WA_, DB_>(p, s);                  \
        else launch_t2<MXFP8, W2_PLAIN, ST_, WA_, DB_>(p, s);                                          \
      }                                                                                                \
      return;                                                                                          \
    }
    V2(3, 4, 0) V2(3, 4, 1) V2(3, 4, 2) V2(3, 8, 0) V2(4, 8, 0) V2(4, 8, 1)
#undef V2
    hum::die("unknown v2 stages");
  }
#define VAR(WF_, ST_)                                                         \
  if (v.WF == WF_ && v.STAGES == ST_) {                                       \
    if (mode == W13_SITU) launch_m<W13_SITU, WF_, ST_>(fmt, p, s);            \
    else if (mode == W13_PLAIN) launch_m<W13_PLAIN, WF_, ST_>(fmt, p, s);     \
    else launch_m<W2_PLAIN, WF_, ST_>(fmt, p, s);                             \
    return;                                                                   \
  }
  VAR(2, 3) VAR(2, 4) VAR(4, 3) VAR(4, 4)
#undef VAR
  hum::die("unknown variant");
}
template <int WF, int ST>
static void attrs_all() {
  set_attr<NVFP4, W13_SITU, WF, ST>(); set_attr<NVFP4, W13_PLAIN, WF, ST>(); set_attr<NVFP4, W2_PLAIN, WF, ST>();
  set_attr<MXFP8, W13_SITU, WF, ST>(); set_attr<MXFP8, W13_PLAIN, WF, ST>(); set_attr<MXFP8, W2_PLAIN, WF, ST>();
}

struct Ctx2 {
  p6::Ctx* c;
  LayerW* L;
  std::map<int, std::unique_ptr<hum::Kernel>> hk[2];   // block_m -> kernel, per proj
  std::string hcache;
  Variant var{2, 3};
};

static const char* hum_md5_prefix(int fmt, int pj, int bm) {
  // config B's cache entries with the prefill tiles (probes/humming/h0_cubin_meta.json)
  struct E_ { int fmt, pj, bm; const char* p; };
  static const E_ t[] = {{0, 0, 32, "2a9c3e4f"}, {0, 0, 64, "1d1660a1"}, {0, 0, 96, "d3625ff5"}, {0, 0, 128, "a97d2636"},
                         {0, 1, 32, "2ed61365"}, {0, 1, 64, "a639b797"}, {0, 1, 96, "a21b37f5"}, {0, 1, 128, "3c65923e"},
                         {1, 0, 32, "49ea2fe5"}, {1, 0, 64, "827c4403"}, {1, 0, 96, "94e7165f"}, {1, 0, 128, "05646568"},
                         {1, 1, 32, "c0b87984"}, {1, 1, 64, "047f157e"}, {1, 1, 96, "ae9fc565"}, {1, 1, 128, "4deb7df7"}};
  for (auto& x : t)
    if (x.fmt == fmt && x.pj == pj && x.bm == bm) return x.p;
  return nullptr;
}
static hum::Kernel& hum_kernel(Ctx2& X, int pj, int bm) {
  auto& m = X.hk[pj];
  auto it = m.find(bm);
  if (it != m.end()) return *it->second;
  const char* pre = hum_md5_prefix(X.L->fmt, pj, bm);
  REQUIRE(pre, "no Humming cubin for block_m " + std::to_string(bm));
  std::string md5;
  for (auto& d : std::filesystem::directory_iterator(X.hcache)) {
    std::string n = d.path().filename().string();
    if (n.rfind(pre, 0) == 0) md5 = n;
  }
  REQUIRE(!md5.empty(), std::string("cubin dir not found: ") + pre);
  auto k = hum::load_kernel(md5, X.hcache + "/" + md5 + "/kernel.cubin");
  REQUIRE((int)k->m.block_shape_m == bm, "block_m of cubin");
  REQUIRE((int)k->m.problem_shape_k == X.L->K(pj) && (int)k->m.problem_shape_n == X.L->N(pj), "cubin shape");
  REQUIRE(k->m.num_experts == E, "cubin experts");
  Line("info", "humming_cubin").kv("proj", pj).kv("block_m", bm).kv("md5", md5).kv("regs", k->regs)
      .kv("local_bytes", k->local_bytes).kv("smem", (int)k->m.smem_size).kv("ctas_per_sm", (int)k->m.num_ctas_per_sm)
      .kv("block_n", (int)k->m.block_shape_n).kv("block_k", (int)k->m.block_shape_k).kv("stages", (int)k->m.num_stages).emit();
  auto& ref = *k;
  m[bm] = std::move(k);
  return ref;
}

static Buf g_locks;
static void hum_gemm(Ctx2& X, Chunk& c, int pj, const void* A, void* C, cudaStream_t s) {
  hum::Kernel& k = hum_kernel(X, pj, c.hbm);
  hum::Launch L;
  L.k = &k;
  L.grid_x = k.m.num_ctas_per_sm * 74;
  L.A = (CUdeviceptr)A;
  L.B = (CUdeviceptr)X.L->hW[pj].p;
  L.C = (CUdeviceptr)C;
  L.BS = (CUdeviceptr)X.L->hS[pj].p;
  L.BS2 = X.L->has_hs2 ? (CUdeviceptr)X.L->hS2[pj].p : 0;
  if (k.m.is_tensor_weight_scale_2) REQUIRE(L.BS2 != 0, "cubin needs BS2");
  L.sorted_ids = (CUdeviceptr)c.h_sorted.p;
  L.expert_ids = (CUdeviceptr)c.h_eids.p;
  L.num_tokens_padded = (CUdeviceptr)c.h_ntp.p;
  L.locks = (CUdeviceptr)g_locks.p;
  L.shape_m = pj == 0 ? c.Q : c.R;
  L.top_k = pj == 0 ? TOPK : 1;
  hum::launch(L, (CUstream)s);
}

static Params params(Ctx2& X, Chunk& c, int pj, int mode) {
  Params p{};
  p.K = X.L->K(pj);
  p.NWT = X.L->N(pj) / WT;
  const bool n1 = X.var.NIB && X.L->fmt == NVFP4 && X.var.WF < 0 && X.var.WF != -8;
  p.W = (n1 ? X.L->Wn[pj] : X.L->W[pj]).as<uint8_t>();
  p.S = (n1 ? X.L->Sn[pj] : X.L->S[pj]).as<uint8_t>();
  p.F = X.L->F[pj].as<float>();
  if (pj == 0) {
    p.A = c.dx.as<__nv_bfloat16>();
    p.a_row = c.d_arow13.as<int>();
    p.items = c.d_items13.as<Item>();
    p.n_items = c.n_items13;
  } else {
    p.A = c.h.as<__nv_bfloat16>();
    p.a_row = c.d_arow2.as<int>();
    p.items = c.d_items2.as<Item>();
    p.n_items = c.n_items2;
  }
  p.out_row = c.d_orow.as<int>();
  p.gamma = c.d_gamma.as<float>();
  if (mode == W13_SITU) p.out = c.h.as<__nv_bfloat16>();
  else if (mode == W13_PLAIN) p.out = c.gu.as<__nv_bfloat16>();
  else p.out = c.y.as<__nv_bfloat16>();
  return p;
}

// arms: one call = one chunk-layer
enum Arm { OURS = 0, HUM = 1, OURS_G = 2, HUM_G = 3 };
static const char* arm_name(int a) { return a == OURS ? "ours" : a == HUM ? "humming" : a == OURS_G ? "ours_gemm" : "humming_gemm"; }
static void run_arm(Ctx2& X, Chunk& c, int arm, cudaStream_t s) {
  const int fmt = X.L->fmt;
  const int cb = (c.Q * (H / 8) + 255) / 256;
  if (arm == OURS) {
    launch_gemm(fmt, W13_SITU, X.var, params(X, c, 0, W13_SITU), s);
    launch_gemm(fmt, W2_PLAIN, X.var, params(X, c, 1, W2_PLAIN), s);
    moe_combine<<<cb, 256, 0, s>>>(c.y.as<__nv_bfloat16>(), c.out.as<__nv_bfloat16>(), c.Q);
  } else if (arm == HUM) {
    hum_gemm(X, c, 0, c.dx.p, c.hc1.p, s);
    moe_swiglu<<<(c.R * (I / 8) + 255) / 256, 256, 0, s>>>(c.hc1.as<__nv_bfloat16>(), c.dwts_flat.as<float>(),
                                                          c.hact.as<__nv_bfloat16>(), c.R);
    hum_gemm(X, c, 1, c.hact.p, c.hy.p, s);
    moe_combine<<<cb, 256, 0, s>>>(c.hy.as<__nv_bfloat16>(), c.hout.as<__nv_bfloat16>(), c.Q);
  } else if (arm == OURS_G) {
    launch_gemm(fmt, W13_PLAIN, X.var, params(X, c, 0, W13_PLAIN), s);
    launch_gemm(fmt, W2_PLAIN, X.var, params(X, c, 1, W2_PLAIN), s);
  } else {
    hum_gemm(X, c, 0, c.dx.p, c.hc1.p, s);
    hum_gemm(X, c, 1, c.hact.p, c.hy.p, s);
  }
}

// ------------------------------------------------------------------------------------------------ oracle
static double situ_exact(double g, double u, double gamma) {
  return 4.0 * std::tanh(g / 4.0) * (1.0 / (1.0 + std::exp(-g))) * 25.0 * std::tanh(u / 25.0) * gamma;
}
struct Err {
  double se = 0, sr = 0, maxrel = 0;
  size_t n = 0, eq = 0;
  void add(double got, double ref) {
    se += (got - ref) * (got - ref);
    sr += ref * ref;
    ++n;
  }
  double rel() const { return std::sqrt(se / std::max(sr, 1e-300)); }
};

static void mode_check(Ctx2& X, const Routing& r, int Q, int off, int nsample, const std::string& cell) {
  p6::Ctx& c0 = *X.c;
  Chunk c;
  std::vector<int> tok(Q);
  std::iota(tok.begin(), tok.end(), off);
  build_chunk(c, r, tok, X.var.WF);
  cudaStream_t s = c0.s;
  CK(cudaMemsetAsync(c.h.p, 0xff, c.h.n, s));
  CK(cudaMemsetAsync(c.y.p, 0xff, c.y.n, s));
  CK(cudaMemsetAsync(c.gu.p, 0xff, c.gu.n, s));
  CK(cudaMemsetAsync(c.out.p, 0xff, c.out.n, s));
  run_arm(X, c, OURS, s);
  run_arm(X, c, HUM, s);
  run_arm(X, c, OURS_G, s);     // gu (flat, unfused w13) + y again from our h
  CK(cudaStreamSynchronize(s));
  CK(cudaGetLastError());
  auto gu = c.gu.down<uint16_t>(s, (size_t)c.R * 2 * I);
  auto hc1 = c.hc1.down<uint16_t>(s, (size_t)c.R * 2 * I);
  auto h = c.h.down<uint16_t>(s, (size_t)c.R * I);
  auto hact = c.hact.down<uint16_t>(s, (size_t)c.R * I);
  auto y = c.y.down<uint16_t>(s, (size_t)c.R * H);
  auto hy = c.hy.down<uint16_t>(s, (size_t)c.R * H);
  run_arm(X, c, OURS, s);      // out again (OURS_G overwrote y with the same values; out unchanged)
  auto out = c.out.down<uint16_t>(s, (size_t)c.Q * H);
  auto hout = c.hout.down<uint16_t>(s, (size_t)c.Q * H);
  std::vector<int> pos_of(c.R);
  for (int p = 0; p < c.R; ++p) pos_of[c.perm[p]] = p;
  // whole-chunk agreement ours vs Humming (bit-equal fractions, not a gate)
  size_t eq_gu = 0, eq_out = 0, nan_ours = 0;
  for (size_t i = 0; i < gu.size(); ++i) { eq_gu += gu[i] == hc1[i]; nan_ours += (gu[i] & 0x7f80) == 0x7f80; }
  for (size_t i = 0; i < out.size(); ++i) { eq_out += out[i] == hout[i]; nan_ours += (out[i] & 0x7f80) == 0x7f80; }
  // sampled tokens: FP64 chain
  const LayerW& L = *X.L;
  std::vector<int> st;
  for (int i = 0; i < nsample; ++i) st.push_back((int)((long long)i * Q / nsample + (i * 7919) % std::max(1, Q / nsample)) % Q);
  Err e_gu[2], e_h[2], e_y[2], e_out[2], e_y_own[2];
  std::mutex mu;
  auto work = [&](int i0, int i1) {
    Err l_gu[2], l_h[2], l_y[2], l_out[2], l_yo[2];
    std::vector<double> gref(2 * I), href(I), yref(H), oref(H), yown(H);
    for (int si = i0; si < i1; ++si) {
      int t = st[si];
      std::fill(oref.begin(), oref.end(), 0.0);
      const uint16_t* xr = &c.x[(size_t)t * H];
      for (int sl = 0; sl < TOPK; ++sl) {
        int rr = t * TOPK + sl, e = c.idx[rr], p = pos_of[rr];
        double gam = c.wts[rr];
        for (int n = 0; n < 2 * I; ++n) {
          double a = 0;
          for (int k = 0; k < H; ++k) a += (double)bf(xr[k]) * L.wv(0, e, n, k);
          gref[n] = a;
          l_gu[0].add(bf(gu[(size_t)rr * 2 * I + n]), a);
          l_gu[1].add(bf(hc1[(size_t)rr * 2 * I + n]), a);
        }
        for (int i = 0; i < I; ++i) {
          href[i] = situ_exact(gref[2 * i], gref[2 * i + 1], gam);
          l_h[0].add(bf(h[(size_t)p * I + i]), href[i]);
          l_h[1].add(bf(hact[(size_t)rr * I + i]), href[i]);
        }
        for (int n = 0; n < H; ++n) {
          double a = 0, ao = 0, ah = 0;
          for (int k = 0; k < I; ++k) {
            double w = L.wv(1, e, n, k);
            a += href[k] * w;
            ao += (double)bf(h[(size_t)p * I + k]) * w;       // GEMM-only error: same bf16 input as the kernel saw
            ah += (double)bf(hact[(size_t)rr * I + k]) * w;
          }
          yref[n] = a;
          oref[n] += a;
          l_y[0].add(bf(y[(size_t)rr * H + n]), a);
          l_y[1].add(bf(hy[(size_t)rr * H + n]), a);
          l_yo[0].add(bf(y[(size_t)rr * H + n]), ao);
          l_yo[1].add(bf(hy[(size_t)rr * H + n]), ah);
        }
      }
      for (int n = 0; n < H; ++n) {
        l_out[0].add(bf(out[(size_t)t * H + n]), oref[n]);
        l_out[1].add(bf(hout[(size_t)t * H + n]), oref[n]);
      }
    }
    std::lock_guard<std::mutex> g(mu);
    for (int a = 0; a < 2; ++a) {
      for (auto pr : {std::make_pair(&e_gu[a], &l_gu[a]), std::make_pair(&e_h[a], &l_h[a]), std::make_pair(&e_y[a], &l_y[a]),
                      std::make_pair(&e_out[a], &l_out[a]), std::make_pair(&e_y_own[a], &l_yo[a])}) {
        pr.first->se += pr.second->se; pr.first->sr += pr.second->sr; pr.first->n += pr.second->n;
      }
    }
  };
  int nth = std::max(1, std::min(8, nsample));
  std::vector<std::thread> th;
  for (int i = 0; i < nth; ++i) th.emplace_back(work, i * nsample / nth, (i + 1) * nsample / nth);
  for (auto& t : th) t.join();
  for (int a = 0; a < 2; ++a) {
    Line l("check", a == 0 ? "ours" : "humming");
    l.kv("cell", cell).kv("layer", L.layer).kv("fmt", L.fmt == NVFP4 ? "nvfp4" : "mxfp8").kv("Q", Q).kv("offset", off)
        .kv("tokens_sampled", nsample).kv("variant", X.var.WF == -8 ? "v3s" + std::to_string(X.var.STAGES) : X.var.WF < 0 ? "v2w" + std::to_string(X.var.WARPS) + "s" + std::to_string(X.var.STAGES) + "d" + std::to_string(X.var.DBG) + (X.var.NIB ? "n1" : "") : std::to_string(X.var.WF) + "x" + std::to_string(X.var.STAGES))
        .kv("rel_gu_w13", e_gu[a].rel()).kv("rel_h_situ", e_h[a].rel()).kv("rel_y_w2_chain", e_y[a].rel())
        .kv("rel_y_w2_own_input", e_y_own[a].rel()).kv("rel_out_moe", e_out[a].rel()).kv("n_gu", (unsigned long long)e_gu[a].n);
    if (a == 0) l.kv("bit_equal_gu_vs_humming", (double)eq_gu / gu.size()).kv("bit_equal_out_vs_humming", (double)eq_out / out.size())
        .kv("nonfinite_ours", (unsigned long long)nan_ours).kv("items13", c.n_items13).kv("items2", c.n_items2);
    else l.kv("block_m", c.hbm).kv("padded_rows", c.h_ntp_v).kv("pad_frac", 1.0 - (double)c.R / c.h_ntp_v);
    l.kv("ok", nan_ours == 0).emit();
  }
}

// ------------------------------------------------------------------------------------------------ invariance
// Per token t (absolute index in the prompt): out row, y rows (8 slots), h rows (8 slots) must be bit-identical
// whatever chunk the token is processed in and whichever other tokens share the chunk.
struct TokOut { std::vector<uint16_t> out, y, h; };
static std::map<int, TokOut> run_tokens(Ctx2& X, const Routing& r, const std::vector<int>& tok, bool humming) {
  Chunk c;
  build_chunk(c, r, tok, X.var.WF);
  cudaStream_t s = X.c->s;
  for (Buf* b : {&c.h, &c.y, &c.out, &c.hact, &c.hy, &c.hout}) CK(cudaMemsetAsync(b->p, 0xff, b->n, s));
  run_arm(X, c, humming ? HUM : OURS, s);
  auto out = (humming ? c.hout : c.out).down<uint16_t>(s, (size_t)c.Q * H);
  auto y = (humming ? c.hy : c.y).down<uint16_t>(s, (size_t)c.R * H);
  auto h = (humming ? c.hact : c.h).down<uint16_t>(s, (size_t)c.R * I);
  std::vector<int> pos_of(c.R);
  for (int p = 0; p < c.R; ++p) pos_of[c.perm[p]] = p;
  std::map<int, TokOut> m;
  for (int i = 0; i < c.Q; ++i) {
    TokOut o;
    o.out.assign(out.begin() + (size_t)i * H, out.begin() + (size_t)(i + 1) * H);
    for (int sl = 0; sl < TOPK; ++sl) {
      int rr = i * TOPK + sl;
      o.y.insert(o.y.end(), y.begin() + (size_t)rr * H, y.begin() + (size_t)(rr + 1) * H);
      size_t hp = humming ? (size_t)rr : (size_t)pos_of[rr];
      o.h.insert(o.h.end(), h.begin() + hp * I, h.begin() + (hp + 1) * I);
    }
    m[tok[i]] = std::move(o);
  }
  return m;
}

static void mode_inv(Ctx2& X, const Routing& r, const Routing& r2, int Q, int off, const std::string& cell) {
  for (int humming = 0; humming < 2; ++humming) {
    std::vector<int> base(Q);
    std::iota(base.begin(), base.end(), off);
    auto ref = run_tokens(X, r, base, humming);
    struct Case { std::string name; std::vector<std::vector<int>> chunks; bool mix; };
    std::vector<Case> cases;
    cases.push_back({"repeat", {base}, false});
    if (!humming) {   // uneven chunking (Humming has no cubins for chunks below 689 tokens: not run there)
      std::vector<std::vector<int>> ch;
      int sizes[] = {777, 1, 1300, 64};
      int p = 0, k = 0;
      while (p < Q) { int n = std::min(sizes[k++ % 4], Q - p); ch.emplace_back(base.begin() + p, base.begin() + p + n); p += n; }
      cases.push_back({"chunks_777_1_1300_64", ch, false});
    }
    {   // two halves
      std::vector<std::vector<int>> ch;
      ch.emplace_back(base.begin(), base.begin() + Q / 2);
      ch.emplace_back(base.begin() + Q / 2, base.end());
      cases.push_back({"halves", ch, false});
    }
    {   // reversed token order in one chunk
      std::vector<int> rv(base.rbegin(), base.rend());
      cases.push_back({"reversed", {rv}, false});
    }
    cases.push_back({"mixed_with_other_prompt", {base}, true});
    for (auto& cs : cases) {
      size_t diff_out = 0, diff_y = 0, diff_h = 0, tokens = 0;
      for (auto& ch : cs.chunks) {
        std::map<int, TokOut> got;
        if (!cs.mix) {
          got = run_tokens(X, r, ch, humming);
        } else {
          // tokens of this prompt interleaved with as many tokens of another prompt (other routing, other x):
          // build a merged Routing on the host
          Routing m;
          int n2 = std::min(r2.n, (int)ch.size());
          m.n = (int)ch.size() + n2;
          m.x.resize((size_t)m.n * H);
          m.idx.resize((size_t)m.n * 8);
          m.wts.resize((size_t)m.n * 8);
          std::vector<int> back(m.n, -1);
          // simple deterministic interleave: even slots ours, odd slots other prompt
          int a = 0, b = 0;
          std::vector<int> mtok;
          for (int i = 0; i < m.n; ++i) {
            bool mine = (b >= n2) || (a < (int)ch.size() && (i & 1) == 0);
            if (mine) {
              int t = ch[a++];
              std::memcpy(&m.x[(size_t)i * H], &r.x[(size_t)t * H], H * 2);
              std::memcpy(&m.idx[(size_t)i * 8], &r.idx[(size_t)t * 8], 32);
              std::memcpy(&m.wts[(size_t)i * 8], &r.wts[(size_t)t * 8], 32);
              back[i] = t;
            } else {
              int t = b++;
              std::memcpy(&m.x[(size_t)i * H], &r2.x[(size_t)t * H], H * 2);
              std::memcpy(&m.idx[(size_t)i * 8], &r2.idx[(size_t)t * 8], 32);
              std::memcpy(&m.wts[(size_t)i * 8], &r2.wts[(size_t)t * 8], 32);
            }
            mtok.push_back(i);
          }
          auto gm = run_tokens(X, m, mtok, humming);
          for (auto& kv : gm)
            if (back[kv.first] >= 0) got[back[kv.first]] = std::move(kv.second);
        }
        for (auto& kv : got) {
          const TokOut& a = ref.at(kv.first);
          diff_out += std::memcmp(a.out.data(), kv.second.out.data(), a.out.size() * 2) != 0;
          diff_y += std::memcmp(a.y.data(), kv.second.y.data(), a.y.size() * 2) != 0;
          diff_h += std::memcmp(a.h.data(), kv.second.h.data(), a.h.size() * 2) != 0;
          ++tokens;
        }
      }
      Line("inv", humming ? "humming" : "ours").kv("cell", cell).kv("layer", X.L->layer).kv("Q", Q).kv("case", cs.name)
          .kv("tokens", (unsigned long long)tokens).kv("tokens_out_differ", (unsigned long long)diff_out)
          .kv("tokens_y_differ", (unsigned long long)diff_y).kv("tokens_h_differ", (unsigned long long)diff_h)
          .kv("ok", tokens == (size_t)Q && diff_out == 0 && diff_y == 0 && diff_h == 0).emit();
    }
  }
}

// ------------------------------------------------------------------------------------------------ contract tests
// Review defects 2 and 4 (S4-prefill-moe-review): (a) device item count + upper-bound grid: CTAs past n_items_dev exit
// before reading an item (the padded item buffer holds hostile items that would overwrite rows); (b) A row index x K
// beyond 2^31 elements (w13 rows moved past 932K rows of a 4.3 GB buffer). Both outputs must be bit-identical to the
// plain launch. Defect 3 (cnt > BN traps) runs as --mode contract_trap in its own process (a trap ends the context).
static void mode_contract(Ctx2& X, const Routing& r, int Q, int off, const std::string& cell) {
  cudaStream_t s = X.c->s;
  Chunk c;
  std::vector<int> tok(Q);
  std::iota(tok.begin(), tok.end(), off);
  build_chunk(c, r, tok, X.var.WF);
  const int fmt = X.L->fmt;
  run_arm(X, c, OURS, s);
  auto h_ref = c.h.down<uint16_t>(s, (size_t)c.R * I);
  auto y_ref = c.y.down<uint16_t>(s, (size_t)c.R * H);
  // (a) w2 with grid n_items + 512 over a buffer whose tail items point at real rows of other experts
  {
    std::vector<Item> items(c.n_items2);
    CK(cudaMemcpy(items.data(), c.d_items2.p, items.size() * sizeof(Item), cudaMemcpyDeviceToHost));
    std::vector<Item> pad(items);
    for (int k = 0; k < 512; ++k) { Item b = items[k % items.size()]; b.e = (b.e + 1 + k) % E; pad.push_back(b); }
    Buf dpad, dn;
    dpad.up(pad);
    std::vector<int> n = {c.n_items2};
    dn.up(n);
    Params p = params(X, c, 1, W2_PLAIN);
    p.items = dpad.as<Item>();
    p.n_items = (int)pad.size();   // the launch grid
    p.n_items_dev = dn.as<int>();
    CK(cudaMemsetAsync(c.y.p, 0xff, c.y.n, s));
    launch_gemm(fmt, W2_PLAIN, X.var, p, s);
    auto y = c.y.down<uint16_t>(s, (size_t)c.R * H);
    CK(cudaGetLastError());
    const bool ok = std::memcmp(y.data(), y_ref.data(), y.size() * 2) == 0;
    Line("contract", "upper_bound_grid").kv("cell", cell).kv("items", c.n_items2).kv("grid", (int)pad.size())
        .kv("bit_identical", ok).emit();
  }
  // (b) w13 reading A rows at index >= 2^31 / K
  {
    const long long shift = (1LL << 31) / H + 1024;   // first A row index whose element offset exceeds 2^31
    Buf big((size_t)(shift + Q) * H * 2);
    CK(cudaMemcpyAsync(big.as<uint16_t>() + (size_t)shift * H, c.dx.p, (size_t)Q * H * 2, cudaMemcpyDeviceToDevice, s));
    std::vector<int> arow(c.R);
    CK(cudaMemcpy(arow.data(), c.d_arow13.p, arow.size() * 4, cudaMemcpyDeviceToHost));
    for (auto& a : arow) a += (int)shift;
    Buf darow;
    darow.up(arow);
    Params p = params(X, c, 0, W13_SITU);
    p.A = big.as<__nv_bfloat16>();
    p.a_row = darow.as<int>();
    CK(cudaMemsetAsync(c.h.p, 0xff, c.h.n, s));
    launch_gemm(fmt, W13_SITU, X.var, p, s);
    auto h = c.h.down<uint16_t>(s, (size_t)c.R * I);
    CK(cudaGetLastError());
    const bool ok = std::memcmp(h.data(), h_ref.data(), h.size() * 2) == 0;
    Line("contract", "row_offset_beyond_2e31").kv("cell", cell).kv("first_row", shift).kv("bit_identical", ok).emit();
  }
}
// one item of 129 rows: the kernel must trap (launch error), not read past its 128-row tables
static void mode_contract_trap(Ctx2& X, const Routing& r, int Q, int off) {
  cudaStream_t s = X.c->s;
  Chunk c;
  std::vector<int> tok(Q);
  std::iota(tok.begin(), tok.end(), off);
  build_chunk(c, r, tok, X.var.WF);
  std::vector<Item> one = {{0, 0, 0, BN + 1}};
  Buf d;
  d.up(one);
  Params p = params(X, c, 1, W2_PLAIN);
  p.items = d.as<Item>();
  p.n_items = 1;
  launch_gemm(X.L->fmt, W2_PLAIN, X.var, p, s);
  const cudaError_t e = cudaStreamSynchronize(s);
  Line("contract", "cnt_over_128_traps").kv("error", cudaGetErrorName(e)).kv("trapped", e != cudaSuccess).emit();
  std::exit(e != cudaSuccess ? 0 : 1);
}

// ------------------------------------------------------------------------------------------------ main
int main(int argc, char** argv) {
  p6::Ctx c;
  c.probe = PROBE;
  c.a = {argc, argv};
  p6::open_device(c);
  Ctx2 X;
  X.c = &c;
  X.hcache = c.a.get("--hcache", "/hcache");
  std::string mode = c.a.get("--mode", "check");
  std::string var = c.a.get("--variant", "2x3");
  X.var = parse_variant(var);
  attrs_all<2, 3>(); attrs_all<2, 4>(); attrs_all<4, 3>(); attrs_all<4, 4>();
#define A2(ST_, WA_, DB_) set_attr2<NVFP4, W13_SITU, ST_, WA_, DB_>(); set_attr2<NVFP4, W13_PLAIN, ST_, WA_, DB_>(); \
  set_attr2<NVFP4, W2_PLAIN, ST_, WA_, DB_>(); set_attr2<MXFP8, W13_SITU, ST_, WA_, DB_>(); \
  set_attr2<MXFP8, W13_PLAIN, ST_, WA_, DB_>(); set_attr2<MXFP8, W2_PLAIN, ST_, WA_, DB_>();
  A2(3, 4, 0) A2(3, 4, 1) A2(3, 4, 2) A2(3, 8, 0) A2(4, 8, 0) A2(4, 8, 1)
#undef A2
  set_attr2<NVFP4B, W13_SITU, 3, 8, 0>(); set_attr2<NVFP4B, W13_PLAIN, 3, 8, 0>(); set_attr2<NVFP4B, W2_PLAIN, 3, 8, 0>();
  set_attr2<NVFP4B, W13_SITU, 3, 4, 0>(); set_attr2<NVFP4B, W13_PLAIN, 3, 4, 0>(); set_attr2<NVFP4B, W2_PLAIN, 3, 4, 0>();
#define A3(ST_) set_attr3<NVFP4, W13_SITU, ST_>(); set_attr3<NVFP4, W13_PLAIN, ST_>(); set_attr3<NVFP4, W2_PLAIN, ST_>(); \
  set_attr3<MXFP8, W13_SITU, ST_>(); set_attr3<MXFP8, W13_PLAIN, ST_>(); set_attr3<MXFP8, W2_PLAIN, ST_>();
  A3(3) A3(4)
#undef A3
  g_locks.alloc(2048 * 4);
  CK(cudaMemset(g_locks.p, 0, g_locks.n));

  if (mode == "info") {
    auto show = [&](const char* name, const void* fn, int threads, int smem) {
      cudaFuncAttributes fa{};
      CK(cudaFuncGetAttributes(&fa, fn));
      int occ = 0;
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, smem));
      Line("info", name).kv("regs", fa.numRegs).kv("local_bytes", (unsigned long long)fa.localSizeBytes)
          .kv("smem_dyn", smem).kv("threads", threads).kv("ctas_per_sm", occ).emit();
    };
#define SHOW2(F, M, S) show("v2_" #F "_" #M "_s" #S, (const void*)moe_gemm2<F, M, S, 4, 0>, 128, Cfg2<F, M, S, 4>::SMEM); \
  show("v2w8_" #F "_" #M "_s" #S, (const void*)moe_gemm2<F, M, S, 8, 0>, 256, Cfg2<F, M, S, 8>::SMEM)
#define SHOW(F, M, W, S) show(#F "_" #M "_" #W "x" #S, (const void*)moe_gemm<F, M, W, S>, Cfg<F, M, W, S>::THREADS, Cfg<F, M, W, S>::SMEM)
    SHOW(NVFP4, W13_SITU, 2, 3); SHOW(NVFP4, W2_PLAIN, 2, 3); SHOW(NVFP4, W13_PLAIN, 2, 3);
    SHOW(MXFP8, W13_SITU, 2, 3); SHOW(MXFP8, W2_PLAIN, 2, 3);
    SHOW(NVFP4, W13_SITU, 2, 4); SHOW(NVFP4, W13_SITU, 4, 3); SHOW(NVFP4, W13_SITU, 4, 4); SHOW(NVFP4, W2_PLAIN, 4, 4);
    SHOW(MXFP8, W13_SITU, 4, 4); SHOW(MXFP8, W2_PLAIN, 4, 4); SHOW(MXFP8, W13_SITU, 4, 3);
    SHOW2(NVFP4, W13_SITU, 3); SHOW2(NVFP4, W2_PLAIN, 3); SHOW2(MXFP8, W13_SITU, 3); SHOW2(MXFP8, W2_PLAIN, 3);
    SHOW2(NVFP4, W13_SITU, 4); SHOW2(NVFP4, W2_PLAIN, 4); SHOW2(MXFP8, W13_SITU, 4);
#undef SHOW
#undef SHOW2
    return 0;
  }

  const std::string rdir = c.a.get("--routing", "/routing");
  if (mode == "stats") {
    for (int pr = 0; pr < 4; ++pr) {
      for (int visit : {0, 7, 20, 44, 47}) {
        auto r = load_routing(rdir + "/p" + std::to_string(pr) + ".safetensors", visit);
        for (long long Q : c.a.list("--Q", "1600,2048,6000")) {
          if (Q > r.n) continue;
          std::vector<int> cnt(E, 0);
          for (long long i = 0; i < Q * 8; ++i) cnt[r.idx[i]]++;
          std::vector<int> sc(cnt);
          std::sort(sc.rbegin(), sc.rend());
          int touched = (int)std::count_if(cnt.begin(), cnt.end(), [](int v) { return v > 0; });
          long long pad64 = 0, pad128 = 0, pad8 = 0, tiles128 = 0;
          for (int v : cnt) { pad64 += (v + 63) / 64 * 64; pad128 += (v + 127) / 128 * 128; pad8 += (v + 7) / 8 * 8; tiles128 += (v + 127) / 128; }
          Line("stats", "rows_per_expert").kv("prompt", pr).kv("visit", visit).kv("Q", (long long)Q).kv("touched", touched)
              .kv("max", sc[0]).kv("p90", sc[12]).kv("median", sc[64]).kv("p10", sc[115]).kv("min", sc[127])
              .kv("pad8_frac", 1.0 - (double)Q * 8 / pad8).kv("pad64_frac", 1.0 - (double)Q * 8 / pad64)
              .kv("pad128_frac", 1.0 - (double)Q * 8 / pad128).kv("row_tiles128", tiles128).emit();
        }
      }
    }
    return 0;
  }

  LayerW L;
  int layer = (int)c.a.num("--layer", 0);
  int visit = layer == 0 ? 0 : layer == 7 ? 7 : layer == 20 ? 44 : layer == 23 ? 47 : -1;
  REQUIRE(visit >= 0, "layer must be 0, 7, 20 or 23 (captured x)");
  load_layer(L, layer, c.a.get("--model", "/model/model-00001-of-00001.safetensors"), c.a.get("--h1", "/h1"));
  X.L = &L;
  int prompt = (int)c.a.num("--prompt", 0);
  Routing r = load_routing(rdir + "/p" + std::to_string(prompt) + ".safetensors", visit);
  int off = (int)c.a.num("--offset", 0);
  auto Qs = c.a.list("--Q", "1600,2048,6000");

  if (mode == "check") {
    for (long long Q : Qs) {
      REQUIRE(off + Q <= r.n, "chunk beyond the captured prompt");
      mode_check(X, r, (int)Q, off, (int)c.a.num("--samples", 48), "L" + std::to_string(layer) + "/Q" + std::to_string(Q));
    }
  } else if (mode == "contract") {
    for (long long Q : Qs) mode_contract(X, r, (int)Q, off, "L" + std::to_string(layer) + "/Q" + std::to_string(Q));
  } else if (mode == "contract_trap") {
    mode_contract_trap(X, r, (int)Qs[0], off);
  } else if (mode == "inv") {
    Routing r2 = load_routing(rdir + "/p" + std::to_string((prompt + 1) % 4) + ".safetensors", visit);
    for (long long Q : Qs) mode_inv(X, r, r2, (int)Q, off, "L" + std::to_string(layer) + "/Q" + std::to_string(Q));
  } else if (mode == "time") {
    // arms: comma list of ours,humming,ours_gemm,humming_gemm; sandwich baseline = first arm of each group
    auto groups = c.a.slist("--groups", "humming+ours,humming_gemm+ours_gemm");
    auto variants = c.a.slist("--variants", var.c_str());
    for (long long Q : Qs) {
      REQUIRE(off + Q <= r.n, "chunk beyond the captured prompt");
      Chunk ch;
      std::vector<int> tok(Q);
      std::iota(tok.begin(), tok.end(), off);
      build_chunk(ch, r, tok, X.var.WF);
      // per variant: own items (WF changes the tile list)
      std::map<std::string, std::unique_ptr<Chunk>> vch;
      for (auto& v : variants) {
        auto cp = std::make_unique<Chunk>();
        build_chunk(*cp, r, tok, parse_variant(v).WF);
        vch[v] = std::move(cp);
      }
      for (auto& gname : groups) {
        // "base+armA+armB..." ; ours arms expand over variants
        std::vector<std::string> names;
        size_t p0 = 0;
        while (p0 <= gname.size()) { size_t q = gname.find('+', p0); if (q == std::string::npos) q = gname.size(); names.push_back(gname.substr(p0, q - p0)); p0 = q + 1; }
        struct A { int arm; std::string var; };
        std::vector<A> arms;
        for (auto& n : names) {
          int a = n == "ours" ? OURS : n == "humming" ? HUM : n == "ours_gemm" ? OURS_G : HUM_G;
          if (a == OURS || a == OURS_G) for (auto& v : variants) arms.push_back({a, v});
          else arms.push_back({a, ""});
        }
        double units = flop_per_token() * Q;
        double est_ms = units / 70e12 * 1e3;
        int cpc = std::max(1, (int)std::ceil(25.0 / est_ms));
        for (auto& sl : p6::sandwich((int)arms.size(), c.rounds, 0)) {
          A& a = arms[sl.arm];
          Chunk& cc = a.var.empty() ? ch : *vch[a.var];
          if (!a.var.empty()) X.var = parse_variant(a.var);
          run_arm(X, cc, a.arm, c.s);         // warm (and load cubins)
          CK(cudaStreamSynchronize(c.s));
          Line l("time", arm_name(a.arm));
          l.kv("cell", "L" + std::to_string(layer) + "/Q" + std::to_string(Q)).kv("layer", layer)
              .kv("fmt", L.fmt == NVFP4 ? "nvfp4" : "mxfp8").kv("Q", (long long)Q).kv("prompt", prompt).kv("offset", off)
              .kv("variant", a.var).kv("round", sl.round).kv("pos", sl.pos).kv("sandwich_tail", sl.sandwich_tail)
              .kv("humming_block_m", cc.hbm).kv("items13", cc.n_items13).kv("items2", cc.n_items2);
          auto res = p6::sustained(c, l, [&](cudaStream_t s) { run_arm(X, cc, a.arm, s); }, units, cpc, "FLOP");
          if (std::isfinite(res.rate_span)) {
            l.kv("tflops", res.rate_span / 1e12).kv("ms_per_call", res.ms_per_call_mean)
                .kv("j_per_token", std::isfinite(res.j_per_call) ? res.j_per_call / Q : NAN);
          }
          l.emit();
        }
      }
    }
  }
  c.tele->stop();
  return 0;
}
