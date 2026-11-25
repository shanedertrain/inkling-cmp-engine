// gd_bench (own code): the loop message-gate down projections of one decode step through the engine's LtGemm
// (engine/native/src/gemm_lt.hpp), standalone: timing / energy, bits against the engine's current path, row (batch)
// invariance and run-to-run determinism. Lever G1 (bench/raw/2026-11-23-g1-gate-proj/).
//
// One "call" = one step's gate down projections: 24 loop visits x (a 3-gate stage + a 1-gate stage), out[T, 64] =
// gin[T, 4608] @ W[64, 4608]^T per gate, bf16 in / out, fp32 accumulation. Weights: 24 x 4 distinct random matrices
// (56.6 MB, more than L2) laid out like the engine's arena (a layer's 4 gate matrices at a constant stride), inputs
// RMS-normalised-scale random rows. Variants:
//   seq           engine before G1: one run() per gate (split-K 1, the pinned decode algorithm)
//   b             the G gates of a stage as one strided-batch runb() on the same algorithm (bit-identical by design)
//   b_sk<S>r<R>   b on the same algorithm with SPLITK_NUM S and reduction scheme R (2 COMPUTE_TYPE, 4 OUTPUT_TYPE)
// The pinned decode algorithm is read from --pin "ID,TILE,STAGES,SWIZZLE" (pins file entry n 64 k 4608 decode).
//
//   gd_bench --rows 1,8,32,64 --splits 4,8,16,32,64 --red 2,4 --secs 4 --rounds 2 [--pin 31,15,17,1]
#include <cublasLt.h>

#include <random>

#include "../decode_attn/bench/common.hpp"   // p6::Ctx, Tele, Line, sustained, sandwich
#include "../common/common.hpp"
#include "../../native/src/gemm_lt.hpp"
#include "../../native/src/gemm_pins.hpp"

using ink::bf16;
static const char* kProbe = "gate_down";
constexpr int H = 2304, K = 2 * H, GR = 64, NL = 24, TMAX = 64;
constexpr long long WSTRIDE = (long long)GR * K + 4096;   // elements between a layer's gate matrices (padded, like the arena)

struct Variant {
  std::string name;
  int splits = 1;
  uint32_t red = 0;
  bool batched = false;
  ink::LtGemm* lt = nullptr;
};

int main(int argc, char** argv) {
  p6::Ctx c;
  c.probe = kProbe;
  c.a = {argc, argv};
  p6::open_device(c);
  const auto rows = c.a.list("--rows", "1,8,32,64");
  const auto splits = c.a.list("--splits", "4,8,16,32,64");
  const auto reds = c.a.list("--red", "2,4");
  const auto pin = c.a.list("--pin", "31,15,17,1");
  INK_REQUIRE(pin.size() == 4, "--pin ID,TILE,STAGES,SWIZZLE");

  // weights [NL][4] at WSTRIDE, inputs gin [3][TMAX][K] (gate stride TMAX * K), outputs [3][TMAX][GR]
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> hw((size_t)NL * 4 * WSTRIDE, __float2bfloat16(0.f));
  for (int l = 0; l < NL; ++l)
    for (int g = 0; g < 4; ++g)
      for (long long i = 0; i < (long long)GR * K; ++i) hw[((size_t)l * 4 + g) * WSTRIDE + i] = __float2bfloat16(0.02f * nd(rng));
  std::vector<bf16> hx((size_t)3 * TMAX * K);
  for (auto& v : hx) v = __float2bfloat16(nd(rng));
  bf16 *W, *X, *Y, *Yref;
  CK(cudaMalloc(&W, hw.size() * 2));
  CK(cudaMalloc(&X, hx.size() * 2));
  CK(cudaMalloc(&Y, (size_t)NL * 4 * TMAX * GR * 2));
  CK(cudaMalloc(&Yref, (size_t)NL * 4 * TMAX * GR * 2));
  CK(cudaMemcpy(W, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(X, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  const size_t wsb = 16 << 20;
  void* ws;
  CK(cudaMalloc(&ws, wsb));
  const long long xs = (long long)TMAX * K, os = (long long)TMAX * GR;

  std::vector<Variant> vs;
  vs.push_back({"seq", 1, 0, false});
  vs.push_back({"b", 1, 0, true});
  for (long long s : splits)
    for (long long r : reds) vs.push_back({"b_sk" + std::to_string(s) + "r" + std::to_string(r), (int)s, (uint32_t)r, true});
  ink::gemm_pins::AlgoPin ap;
  ap.id = (int)pin[0], ap.tile = (uint32_t)pin[1], ap.stages = (uint32_t)pin[2], ap.swizzle = (uint32_t)pin[3];
  ink::gemm_pins::Shape shp;
  shp.n = GR, shp.k = K, shp.ldw = K;
  for (auto& v : vs) {
    v.lt = new ink::LtGemm();
    v.lt->init(c.s);
    v.lt->set_workspace(ws, wsb);
    v.lt->run(X, K, W, GR, K, Y, GR, 1, false);   // plans the shape (class 0)
    v.lt->for_each_plan([&](const std::tuple<int, int, int>&, cublasLtMatmulAlgo_t& a) {
      a = ink::gemm_pins::build_algo(v.lt->handle(), shp, ap);
    });
    v.lt->recheck();
    if (v.splits > 1) v.lt->split_k(GR, K, 0, v.splits, v.red);
  }
  auto step = [&](Variant& v, bf16* out, int M, cudaStream_t) {
    for (int l = 0; l < NL; ++l) {
      const bf16* w = W + (size_t)l * 4 * WSTRIDE;
      bf16* o = out + (size_t)l * 4 * os;
      if (v.batched) {
        v.lt->runb(X, K, xs, w, WSTRIDE, GR, K, o, GR, os, M, 3);
        v.lt->runb(X, K, xs, w + 3 * WSTRIDE, WSTRIDE, GR, K, o + 3 * os, GR, os, M, 1);
      } else {
        for (int g = 0; g < 4; ++g) v.lt->run(X + (g < 3 ? g : 0) * xs, K, w + g * WSTRIDE, GR, K, o + g * os, GR, M, false);
      }
    }
  };
  auto fetch = [&](const bf16* d) {
    std::vector<uint16_t> h((size_t)NL * 4 * TMAX * GR);
    CK(cudaMemcpy(h.data(), d, h.size() * 2, cudaMemcpyDeviceToHost));
    return h;
  };
  // bits: every variant at every M vs seq at the same M (rows < M); invariance: rows < M at M vs the same rows at
  // M = TMAX; determinism: a second run at TMAX
  for (auto& v : vs) {
    CK(cudaMemset(Y, 0, (size_t)NL * 4 * TMAX * GR * 2));
    step(v, Y, TMAX, c.s);
    CK(cudaStreamSynchronize(c.s));
    auto full = fetch(Y);
    step(v, Y, TMAX, c.s);
    CK(cudaStreamSynchronize(c.s));
    auto again = fetch(Y);
    size_t nondet = 0;
    for (size_t i = 0; i < full.size(); ++i) nondet += full[i] != again[i];
    for (long long M : rows) {
      CK(cudaMemset(Y, 0, (size_t)NL * 4 * TMAX * GR * 2));
      CK(cudaMemset(Yref, 0, (size_t)NL * 4 * TMAX * GR * 2));
      step(v, Y, (int)M, c.s);
      step(vs[0], Yref, (int)M, c.s);
      CK(cudaStreamSynchronize(c.s));
      auto a = fetch(Y), r = fetch(Yref);
      size_t n = 0, diff_seq = 0, diff_inv = 0;
      for (int lg = 0; lg < NL * 4; ++lg)
        for (long long t = 0; t < M; ++t)
          for (int j = 0; j < GR; ++j) {
            const size_t i = ((size_t)lg * TMAX + t) * GR + j;
            ++n;
            diff_seq += a[i] != r[i];
            diff_inv += a[i] != full[i];
          }
      p6::Line(kProbe, "bits").kv("variant", v.name).kv("rows", (int)M).kv("elements", (unsigned long long)n)
          .kv("differ_vs_seq", (unsigned long long)diff_seq).kv("differ_vs_rows64", (unsigned long long)diff_inv)
          .kv("nondeterministic_at_64", (unsigned long long)nondet).kv("fallbacks", v.lt->fallbacks).emit();
    }
  }
  // timing: one graph per variant and M, sandwich rounds with seq as the baseline
  for (long long M : rows) {
    std::vector<cudaGraphExec_t> gs;
    for (auto& v : vs) {
      step(v, Y, (int)M, c.s);   // AlgoCheck outside the capture
      CK(cudaStreamSynchronize(c.s));
      cudaGraph_t g;
      CK(cudaStreamBeginCapture(c.s, cudaStreamCaptureModeThreadLocal));
      step(v, Y, (int)M, c.s);
      CK(cudaStreamEndCapture(c.s, &g));
      cudaGraphExec_t ge;
      CK(cudaGraphInstantiate(&ge, g, 0));
      gs.push_back(ge);
    }
    for (auto& sl : p6::sandwich((int)vs.size(), c.rounds)) {
      p6::Line l(kProbe, "sweep");
      l.kv("variant", vs[sl.arm].name).kv("rows", (int)M).kv("round", sl.round).kv("pos", sl.pos);
      auto ge = gs[sl.arm];
      p6::sustained(c, l, [&](cudaStream_t) { cudaGraphLaunch(ge, c.s); }, (double)NL * 4 * GR * K * 2 / 1e9, 4);
      l.emit();
    }
    for (auto g : gs) cudaGraphExecDestroy(g);
  }
  p6::Line(kProbe, "end").emit();
  c.tele->stop();
  return 0;
}
