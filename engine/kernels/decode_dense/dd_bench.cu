// decode_dense: standalone timing / energy / bit harness for the dense GEMMs of one decode step (own code).
//
// Real Inkling-2-Mini weights (config B checkpoint, mmap), the engine's visit plan (48 visits), M rows per call.
// Every "call" of an op = that op over all 48 visits (distinct weights per layer, so nothing stays in L2 across
// visits, as in the step), captured as one CUDA graph and replayed back to back (p6::sustained: chunks of graph
// launches between events, NVML energy counter at 10 Hz, J per call). Ops and variants:
//
//   proj  : the visit's input projections q, gate, k, v, r, sink (+ k_dyn, v_dyn on full layers)
//     skinny     engine: proj_small.cuh, one launch per visit (decode-only calls, M <= 64)       [engine today]
//     lt_sep     engine: one cublasLt call per matrix, fixed algo tuned at M = 32 (gemm_lt.hpp)  [engine, prefill class]
//     lt_fused   cublasLt over the concatenated weight [q|gate|r|sink|k|v(|dyn)]: one call per visit with one input,
//                three (q side, k side, v side) when q / k / v inputs differ (full layers, message-gate visits)
//     ex_fused   cublasGemmEx(CUBLAS_GEMM_DEFAULT_TENSOR_OP) on the same fused layout: config B's structure
//                (MergedColumnParallelLinear + F.linear), our cuBLAS
//     lt_pin     the D1 patch: lt_fused with ONE pinned algorithm (heuristic #1 at N = 8600) for every group, then a
//                split copy into the engine's separate buffers (bits = skinny's, checked)
//     sk2, sk2w8 candidate skinny kernel (dd_kernels.cuh; 4 / 8 warps per CTA), bit-identical to skinny by
//                construction (checked)
//   wo    : o projection [2304 x 3072]: lt (engine), ex
//   gdown : loop message-gate down projections [64 x 4608] (4 per message visit): lt (engine)
//   router: three K-group fp32 GEMMs [128 x 1024|1024|256]: lt (engine)
//   head  : unembedding [201024 x 2304] fp32 out over M rows: lt (engine), ex
//
// Modes: --mode sweep (every op x variant x M, sandwich rounds), --mode algos (every cublasLt heuristic algorithm of
// every shape at every M, plus gemmEx default and the skinny kernel), --mode bits (differing bf16 elements of each
// variant vs the engine's path, and row invariance across M).
//
//   dd_bench --model /model --mode sweep --rows 8,32,64 --ops proj,wo --secs 6 --rounds 3
#include <cublas_v2.h>
#include <cublasLt.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <map>
#include <random>

#include "../decode_attn/bench/common.hpp"   // p6::Ctx, Tele, Line, sustained, sandwich
#include "../common/common.hpp"
#include "../../native/src/gemm_lt.hpp"
#include "../../native/src/proj_small.cuh"
#include "../prefill_moe/humming_launch.hpp"   // hum::JsonParser
#include "dd_kernels.cuh"

using ink::bf16;
static const char* kProbe = "decode_dense";
constexpr int H = 2304, QD = 3072, KD = 1024, NRS = 384, NSINK = 24, NDYN = 32, NEXP = 128, GR = 64;
constexpr int VOCAB_PAD = 201024, NLAYERS = 24, NV = 48;
static bool is_full(int l) { return l == 5 || l == 11 || l == 17 || l == 23; }

#define CBS(x)                                                                         \
  do {                                                                                 \
    cublasStatus_t s_ = (x);                                                           \
    if (s_ != CUBLAS_STATUS_SUCCESS) ink::fatal(std::string(#x) + " " + std::to_string((int)s_)); \
  } while (0)

// ------------------------------------------------------------------------------------------------ checkpoint
struct Mapped {
  struct T { std::string dtype; std::vector<int64_t> shape; const uint8_t* p; size_t bytes; };
  std::map<std::string, T> m;
  explicit Mapped(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    INK_REQUIRE(fd >= 0, "open " + path);
    struct stat st{};
    ::fstat(fd, &st);
    auto* base = (const uint8_t*)::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    INK_REQUIRE(base != MAP_FAILED, "mmap");
    uint64_t hl = 0;
    std::memcpy(&hl, base, 8);
    hum::Json j = hum::JsonParser(std::string((const char*)base + 8, hl)).parse();
    for (auto& kv : j.o) {
      if (kv.first == "__metadata__") continue;
      T t;
      t.dtype = kv.second["dtype"].str();
      for (auto& d : kv.second["shape"].a) t.shape.push_back(d.i());
      t.p = base + 8 + hl + kv.second["data_offsets"][0].u();
      t.bytes = kv.second["data_offsets"][1].u() - kv.second["data_offsets"][0].u();
      m[kv.first] = t;
    }
  }
  const T& at(const std::string& n) const {
    auto it = m.find(n);
    INK_REQUIRE(it != m.end(), "missing " + n);
    return it->second;
  }
};

struct Layer {
  bf16 *wq, *wg, *wk, *wv, *wr, *ws, *wkd = nullptr, *wvd = nullptr, *wo, *router;
  bf16* cdown[4] = {};
  bf16* fused = nullptr;   // [q | gate | r | sink | k (| kdyn) | v (| vdyn)] x H
  int n_qside = QD + QD + NRS + NSINK, n_kside = KD, n_vside = KD;
};

static bf16* dev_copy(const Mapped::T& t) {
  INK_REQUIRE(t.dtype == "BF16", "not bf16");
  bf16* d;
  CK(cudaMalloc(&d, t.bytes));
  CK(cudaMemcpy(d, t.p, t.bytes, cudaMemcpyHostToDevice));
  return d;
}

struct Model {
  std::vector<Layer> L;
  bf16* unembed = nullptr;
  int vlayer[NV];
  bool vmsgs[NV];
  size_t proj_bytes_step = 0, wo_bytes_step = 0, gdown_bytes_step = 0, router_bytes_step = 0;
  void load(const std::string& dir) {
    Mapped m(dir + "/model-00001-of-00001.safetensors");
    L.resize(NLAYERS);
    for (int l = 0; l < NLAYERS; ++l) {
      Layer& y = L[l];
      const std::string p = "model.llm.layers." + std::to_string(l) + ".";
      y.wq = dev_copy(m.at(p + "attn.wq_du.weight"));
      y.wk = dev_copy(m.at(p + "attn.wk_dv.weight"));
      y.wv = dev_copy(m.at(p + "attn.wv_dv.weight"));
      y.wr = dev_copy(m.at(p + "attn.wr_du.weight"));
      y.wo = dev_copy(m.at(p + "attn.wo_ud.weight"));
      y.wg = dev_copy(m.at(p + "attn.gate_attn.weight"));
      y.ws = dev_copy(m.at(p + "attn.attn_sink_proj.weight"));
      y.router = dev_copy(m.at(p + "mlp.gate.weight"));
      if (is_full(l)) {
        y.wkd = dev_copy(m.at(p + "attn.k_sconv.weight_proj"));
        y.wvd = dev_copy(m.at(p + "attn.v_sconv.weight_proj"));
        y.n_kside = y.n_vside = KD + NDYN;
      }
      if (l >= 6 && l < 18) {
        const char* ck[4] = {"attn", "k", "v", "mlp"};
        for (int g = 0; g < 4; ++g) y.cdown[g] = dev_copy(m.at(p + "controllers." + ck[g] + ".down_weight"));
      }
      // fused copy (device to device, same bits)
      const size_t n = (size_t)y.n_qside + y.n_kside + y.n_vside;
      CK(cudaMalloc(&y.fused, n * H * 2));
      size_t off = 0;
      auto put = [&](const bf16* s, int rows) { CK(cudaMemcpy(y.fused + off * H, s, (size_t)rows * H * 2, cudaMemcpyDeviceToDevice)); off += rows; };
      put(y.wq, QD); put(y.wg, QD); put(y.wr, NRS); put(y.ws, NSINK);
      put(y.wk, KD); if (y.wkd) put(y.wkd, NDYN);
      put(y.wv, KD); if (y.wvd) put(y.wvd, NDYN);
    }
    unembed = dev_copy(m.at("model.llm.unembed.weight"));
    int v = 0;
    for (int l = 0; l < 6; ++l) { vlayer[v] = l; vmsgs[v++] = false; }
    for (int p = 0; p < 3; ++p)
      for (int l = 6; l < 18; ++l) { vlayer[v] = l; vmsgs[v++] = p > 0; }
    for (int l = 18; l < 24; ++l) { vlayer[v] = l; vmsgs[v++] = false; }
    for (int i = 0; i < NV; ++i) {
      const Layer& y = L[vlayer[i]];
      proj_bytes_step += ((size_t)y.n_qside + y.n_kside + y.n_vside) * H * 2;
      wo_bytes_step += (size_t)H * QD * 2;
      router_bytes_step += (size_t)NEXP * H * 2;
      if (vmsgs[i]) gdown_bytes_step += (size_t)4 * GR * 2 * H * 2;
    }
  }
  bool separate(int v) const { return is_full(vlayer[v]) || vmsgs[v]; }
};

// ------------------------------------------------------------------------------------------------ buffers
struct Bufs {
  bf16 *xq, *xk, *xv, *xo, *xg, *xm;          // inputs [64 x K] (xg: [64 x 4608])
  bf16 *q, *g, *k, *v, *r, *s, *kd, *vd;      // separate outputs
  bf16* f;                                    // fused output [64 x 8664]
  bf16* o;                                    // wo out [64 x H]
  bf16* gd;                                   // gdown out
  float* rl;                                  // router logits [3 x 64 x 128]
  float* lg;                                  // head logits [64 x 201024]
  void* ws;                                   // cublas workspace (gemmEx)
};
static void fill_rand(bf16* d, size_t n, uint64_t seed, float scale) {
  std::vector<uint16_t> h(n);
  std::mt19937_64 g(seed);
  std::normal_distribution<float> nd(0.f, scale);
  for (auto& x : h) x = ink::f2bf(nd(g));
  CK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice));
}
template <class T> static T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); return p; }

// ------------------------------------------------------------------------------------------------ runner
struct Runner {
  Model& md;
  Bufs b;
  ink::LtGemm lt;
  cublasHandle_t cb;
  cudaStream_t st;
  Runner(Model& m, cudaStream_t s) : md(m), st(s) {
    b.xq = dalloc<bf16>(64 * H); b.xk = dalloc<bf16>(2 * 64 * H); b.xv = b.xk + 64 * H;   // xk, xv adjacent (lt_pinb)
    b.xo = dalloc<bf16>(64 * QD); b.xg = dalloc<bf16>(64 * 2 * H); b.xm = dalloc<bf16>(64 * H);
    fill_rand(b.xq, 64 * H, 1, 1.f); fill_rand(b.xk, 64 * H, 2, 1.f); fill_rand(b.xv, 64 * H, 3, 1.f);
    fill_rand(b.xo, 64 * QD, 4, 0.5f); fill_rand(b.xg, 64 * 2 * H, 5, 1.f); fill_rand(b.xm, 64 * H, 6, 1.f);
    b.q = dalloc<bf16>(64 * QD); b.g = dalloc<bf16>(64 * QD); b.k = dalloc<bf16>(64 * (KD + NDYN)); b.v = dalloc<bf16>(64 * (KD + NDYN));
    b.r = dalloc<bf16>(64 * NRS); b.s = dalloc<bf16>(64 * NSINK); b.kd = dalloc<bf16>(64 * NDYN); b.vd = dalloc<bf16>(64 * NDYN);
    b.f = dalloc<bf16>(64 * 8664); b.o = dalloc<bf16>(64 * H); b.gd = dalloc<bf16>(4 * 64 * GR);
    b.rl = dalloc<float>(3 * 64 * NEXP); b.lg = dalloc<float>((size_t)64 * VOCAB_PAD);
    b.ws = dalloc<char>(32 << 20);
    lt.init(st);
    lt.tune_m = 32;
    CBS(cublasCreate(&cb));
    CBS(cublasSetStream(cb, st));
    CBS(cublasSetWorkspace(cb, b.ws, 32 << 20));
    CBS(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));   // torch default: bf16 reduced-precision reduction allowed
    // engine plans (decode class, tuned at M = 32 exactly as Engine::Engine), plus the fused shapes
    const Layer& L0 = md.L[0];
    const Layer& F = md.L[5];
    bf16* x = b.xg;
    void* o = b.lg;
    lt.tune(L0.wq, QD, H, false, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(L0.wr, NRS, H, false, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(L0.wk, KD, H, false, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(L0.ws, NSINK, H, false, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(F.wkd, NDYN, H, false, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(L0.wo, H, QD, false, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(md.L[6].cdown[0], GR, 2 * H, false, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(L0.router, NEXP, 1024, true, CUDA_R_16BF, H, x, o, 0, 32);
    lt.tune(L0.router, NEXP, H - 2048, true, CUDA_R_16BF, H, x, o, 0, 32);
    lt.tune(md.unembed, VOCAB_PAD, H, true, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(L0.fused, 8600, H, false, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(L0.fused, L0.n_qside, H, false, CUDA_R_16BF, 0, x, o, 0, 32);
    lt.tune(F.fused, KD + NDYN, H, false, CUDA_R_16BF, 0, x, o, 0, 32);
    pin_init();
    CK(cudaDeviceSynchronize());
  }
  void ex(const bf16* x, int ldx, const bf16* W, int N, int K, void* out, int ldo, int M, bool f32) {
    const float one = 1.f, zero = 0.f;
    CBS(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16BF, K, x, CUDA_R_16BF, ldx, &zero, out,
                     f32 ? CUDA_R_32F : CUDA_R_16BF, ldo, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  // one op over all 48 visits
  void proj(const std::string& var, int M) {
    for (int v = 0; v < NV; ++v) {
      const Layer& y = md.L[md.vlayer[v]];
      const bool sep = md.separate(v);
      const bf16 *px = b.xq, *pk = sep ? b.xk : b.xq, *pv = sep ? b.xv : b.xq;
      if (var == "skinny" || var == "sk2" || var == "sk2w8") {
        projs::Args a{};
        auto job = [&](const bf16* W, const bf16* x, bf16* out, int N) { a.j[a.njobs++] = {W, x, H, out, N, N, H, 0}; };
        job(y.wq, px, b.q, QD); job(y.wg, px, b.g, QD); job(y.wk, pk, b.k, KD); job(y.wv, pv, b.v, KD);
        job(y.wr, px, b.r, NRS); job(y.ws, px, b.s, NSINK);
        if (y.wkd) { job(y.wkd, pk, b.kd, NDYN); job(y.wvd, pv, b.vd, NDYN); }
        a.M = M;
        if (var == "skinny") projs::launch(a, st);
        else if (var == "sk2") dd::launch_sk2<4>(a, st);
        else dd::launch_sk2<8>(a, st);
      } else if (var == "lt_sep") {
        lt.run(px, H, y.wq, QD, H, b.q, QD, M, false);
        lt.run(px, H, y.wr, NRS, H, b.r, NRS, M, false);
        lt.run(pk, H, y.wk, KD, H, b.k, KD, M, false);
        lt.run(pv, H, y.wv, KD, H, b.v, KD, M, false);
        if (y.wkd) { lt.run(pk, H, y.wkd, NDYN, H, b.kd, NDYN, M, false); lt.run(pv, H, y.wvd, NDYN, H, b.vd, NDYN, M, false); }
        lt.run(px, H, y.ws, NSINK, H, b.s, NSINK, M, false);
        lt.run(px, H, y.wg, QD, H, b.g, QD, M, false);
      } else if (var == "lt_fused" || var == "ex_fused") {
        const bool e = var == "ex_fused";
        auto g = [&](const bf16* x, const bf16* W, int N, bf16* out, int ldo) {
          if (e) ex(x, H, W, N, H, out, ldo, M, false); else lt.run(x, H, W, N, H, out, ldo, M, false);
        };
        const int nt = y.n_qside + y.n_kside + y.n_vside;
        if (!sep) g(px, y.fused, nt, b.f, nt);
        else {
          g(px, y.fused, y.n_qside, b.f, y.n_qside);
          g(pk, y.fused + (size_t)y.n_qside * H, y.n_kside, b.k, y.n_kside);
          g(pv, y.fused + (size_t)(y.n_qside + y.n_kside) * H, y.n_vside, b.v, y.n_vside);
        }
      } else if (var == "lt_pin" || var == "lt_pinb" || var == "lt_pin_nosplit") {
        // the D1 patch: fused GEMMs on ONE pinned algorithm + split copy; lt_pinb: k and v groups as one strided batch
        // of 2 (inputs xk, xv adjacent); lt_pin_nosplit: lt_pinb without the split copy (its cost)
        const int nt = y.n_qside + y.n_kside + y.n_vside;
        if (!sep) pin(px, y.fused, nt, b.f, nt, M);
        else {
          pin(px, y.fused, y.n_qside, b.f, nt, M);
          if (var == "lt_pin") {
            pin(pk, y.fused + (size_t)y.n_qside * H, y.n_kside, b.f + y.n_qside, nt, M);
            pin(pv, y.fused + (size_t)(y.n_qside + y.n_kside) * H, y.n_vside, b.f + y.n_qside + y.n_kside, nt, M);
          } else {
            pin(pk, y.fused + (size_t)y.n_qside * H, y.n_kside, b.f + y.n_qside, nt, M, 2, (long long)(pv - pk));
          }
        }
        if (var == "lt_pin_nosplit") continue;
        dd::ProjSplit a{b.f, nt, {b.q, b.g, b.r, b.s, b.k, b.kd, b.v, b.vd}, y.wkd ? 1 : 0};
        dd::launch_split(a, M, st);
      } else ink::fatal("proj variant " + var);
    }
  }
  // fused projection GEMM on the algorithm cublasLt's heuristic gives first for N = 8600 (split-K 1, no workspace),
  // used for every N and M (one per-element k chain)
  cublasLtHandle_t plh = nullptr;
  cublasLtMatmulDesc_t pop = nullptr;
  cublasLtMatmulAlgo_t palgo{};
  std::map<std::tuple<int, int, int, int, long long>, cublasLtMatrixLayout_t> play;
  cublasLtMatrixLayout_t lay(int r, int c, int ld, int batch = 1, long long stride = 0) {
    auto k = std::make_tuple(r, c, ld, batch, stride);
    auto it = play.find(k);
    if (it != play.end()) return it->second;
    cublasLtMatrixLayout_t l;
    CBS(cublasLtMatrixLayoutCreate(&l, CUDA_R_16BF, r, c, ld));
    if (batch > 1) {
      CBS(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof batch));
      CBS(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof stride));
    }
    return play[k] = l;
  }
  void pin_init() {
    CBS(cublasLtCreate(&plh));
    CBS(cublasLtMatmulDescCreate(&pop, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CBS(cublasLtMatmulDescSetAttribute(pop, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    CBS(cublasLtMatmulDescSetAttribute(pop, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    cublasLtMatmulPreference_t pref;
    CBS(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 0;
    CBS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
    cublasLtMatmulHeuristicResult_t r[8];
    int n = 0;
    CBS(cublasLtMatmulAlgoGetHeuristic(plh, pop, lay(H, 8600, H), lay(H, 64, H), lay(8600, 64, 8600), lay(8600, 64, 8600),
                                       pref, 8, r, &n));
    cublasLtMatmulPreferenceDestroy(pref);
    INK_REQUIRE(n > 0, "no algo");
    palgo = r[0].algo;
    int id = 0, tile = 0, stg = 0, sk = 0;
    size_t w = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&palgo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&palgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&palgo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stg, sizeof stg, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&palgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof sk, &w);
    std::fprintf(stderr, "[pin] N=8600 algo %d tile %d stages %d splitk %d\n", id, tile, stg, sk);
    INK_REQUIRE(sk <= 1, "pinned algo must be split-K 1");
  }
  // batch 2: weights W, W + N*H; inputs x, x + xstride; outputs out, out + N (same ldo)
  void pin(const bf16* x, const bf16* W, int N, bf16* out, int ldo, int M, int batch = 1, long long xstride = 0) {
    const float one = 1.f, zero = 0.f;
    auto a = lay(H, N, H, batch, (long long)N * H), bl = lay(H, M, H, batch, xstride), c = lay(N, M, ldo, batch, N);
    cublasLtMatmulHeuristicResult_t r{};
    INK_REQUIRE(cublasLtMatmulAlgoCheck(plh, pop, a, bl, c, c, &palgo, &r) == CUBLAS_STATUS_SUCCESS && r.workspaceSize == 0,
                "pinned algo refused N=" + std::to_string(N));
    CBS(cublasLtMatmul(plh, pop, &one, W, a, x, bl, &zero, out, c, out, c, &palgo, nullptr, 0, st));
  }
  void wo(const std::string& var, int M) {
    for (int v = 0; v < NV; ++v) {
      const Layer& y = md.L[md.vlayer[v]];
      if (var == "lt") lt.run(b.xo, QD, y.wo, H, QD, b.o, H, M, false);
      else if (var == "ex") ex(b.xo, QD, y.wo, H, QD, b.o, H, M, false);
      else ink::fatal("wo variant " + var);
    }
  }
  void gdown(const std::string& var, int M) {
    for (int v = 0; v < NV; ++v) {
      if (!md.vmsgs[v]) continue;
      const Layer& y = md.L[md.vlayer[v]];
      for (int gi = 0; gi < 4; ++gi) {
        if (var == "lt") lt.run(b.xg, 2 * H, y.cdown[gi], GR, 2 * H, b.gd + gi * 64 * GR, GR, M, false);
        else if (var == "ex") ex(b.xg, 2 * H, y.cdown[gi], GR, 2 * H, b.gd + gi * 64 * GR, GR, M, false);
        else if (var == "lt_sk") sk_gemm(b.xg, y.cdown[gi], b.gd + gi * 64 * GR, M);
        else ink::fatal("gdown variant " + var);
      }
    }
  }
  // gate down projection on ONE pinned split-K algorithm (heuristic at M = 32 with workspace, first candidate with
  // split-K > 1 and a non-atomic reduction): fixed split count and reduction order for every M (batch-invariant if
  // the reduction is deterministic; checked in --mode bits); bits differ from the split-K-1 plan
  cublasLtMatmulAlgo_t skalgo{};
  size_t skws = 0;
  bool sk_ready = false;
  void sk_init() {
    cublasLtMatmulPreference_t pref;
    CBS(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 32 << 20;
    CBS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
    cublasLtMatmulHeuristicResult_t r[32];
    int n = 0;
    auto a = lay(2 * H, GR, 2 * H), bl = lay(2 * H, 32, 2 * H), c = lay(GR, 32, GR);
    CBS(cublasLtMatmulAlgoGetHeuristic(plh, pop, a, bl, c, c, pref, 32, r, &n));
    cublasLtMatmulPreferenceDestroy(pref);
    for (int i = 0; i < n && !sk_ready; ++i) {
      int sk = 0, red = 0, id = 0, tile = 0;
      size_t w = 0;
      cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof sk, &w);
      cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red, &w);
      cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w);
      cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &w);
      if (sk > 1 && red != CUBLASLT_REDUCTION_SCHEME_INPLACE) {
        skalgo = r[i].algo;
        skws = r[i].workspaceSize;
        sk_ready = true;
        std::fprintf(stderr, "[sk] gdown algo %d tile %d splitk %d reduction %d ws %zu\n", id, tile, sk, red, skws);
      }
    }
    INK_REQUIRE(sk_ready, "no split-K candidate");
  }
  void sk_gemm(const bf16* x, const bf16* W, bf16* out, int M) {
    if (!sk_ready) sk_init();
    const float one = 1.f, zero = 0.f;
    auto a = lay(2 * H, GR, 2 * H), bl = lay(2 * H, M, 2 * H), c = lay(GR, M, GR);
    cublasLtMatmulHeuristicResult_t r{};
    INK_REQUIRE(cublasLtMatmulAlgoCheck(plh, pop, a, bl, c, c, &skalgo, &r) == CUBLAS_STATUS_SUCCESS && r.workspaceSize <= (32u << 20),
                "split-K algo refused at M=" + std::to_string(M));
    CBS(cublasLtMatmul(plh, pop, &one, W, a, x, bl, &zero, out, c, out, c, &skalgo, b.ws, 32 << 20, st));
  }
  void router(const std::string& var, int M) {
    for (int v = 0; v < NV; ++v) {
      const Layer& y = md.L[md.vlayer[v]];
      for (int g = 0; g < 3; ++g) {
        INK_REQUIRE(var == "lt", "router variant");
        lt.run(b.xm + g * 1024, H, y.router + g * 1024, NEXP, g < 2 ? 1024 : H - 2048, b.rl + (size_t)g * 64 * NEXP, NEXP, M,
               true, CUDA_R_16BF, H);
      }
    }
  }
  void head(const std::string& var, int M) {   // one call per step (not per visit)
    if (var == "lt") lt.run(b.xm, H, md.unembed, VOCAB_PAD, H, b.lg, VOCAB_PAD, M, true);
    else if (var == "ex") ex(b.xm, H, md.unembed, VOCAB_PAD, H, b.lg, VOCAB_PAD, M, true);
    else ink::fatal("head variant " + var);
  }
  void op(const std::string& o, const std::string& var, int M) {
    if (o == "proj") proj(var, M);
    else if (o == "wo") wo(var, M);
    else if (o == "gdown") gdown(var, M);
    else if (o == "router") router(var, M);
    else if (o == "head") head(var, M);
    else ink::fatal("op " + o);
  }
  size_t bytes(const std::string& o) const {
    if (o == "proj") return md.proj_bytes_step;
    if (o == "wo") return md.wo_bytes_step;
    if (o == "gdown") return md.gdown_bytes_step;
    if (o == "router") return md.router_bytes_step;
    return (size_t)VOCAB_PAD * H * 2;
  }
  cudaGraphExec_t graph(const std::string& o, const std::string& var, int M) {
    op(o, var, M);   // eager warm-up (plan checks happen outside the capture)
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    op(o, var, M);
    CK(cudaStreamEndCapture(st, &g));
    cudaGraphExec_t ex_;
    CK(cudaGraphInstantiate(&ex_, g, 0));
    cudaGraphDestroy(g);
    return ex_;
  }
};

static std::vector<std::string> variants_of(const std::string& o) {
  if (o == "proj") return {"skinny", "lt_sep", "lt_fused", "ex_fused", "sk2", "sk2w8", "lt_pin", "lt_pinb", "lt_pin_nosplit"};
  if (o == "gdown") return {"lt", "ex", "lt_sk"};
  if (o == "wo" || o == "head") return {"lt", "ex"};
  return {"lt"};
}

// ------------------------------------------------------------------------------------------------ bits
static std::vector<uint16_t> host16(const void* d, size_t n) {
  std::vector<uint16_t> h(n);
  CK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
  return h;
}
// proj outputs of the LAST visit (layer 23, full: separate inputs) and of visit 46 (layer 22, shared input), gathered
// in the separate layout [q | gate | r | sink | k | v (| kd | vd)] per row
static std::vector<uint16_t> proj_out(Runner& R, const std::string& var, int M, int vis) {
  const Layer& y = R.md.L[R.md.vlayer[vis]];
  const bool sep = R.md.separate(vis);
  // run only this visit: temporarily one-visit model view
  int keep_l[NV]; bool keep_m[NV];
  std::memcpy(keep_l, R.md.vlayer, sizeof keep_l); std::memcpy(keep_m, R.md.vmsgs, sizeof keep_m);
  for (int v = 0; v < NV; ++v) { R.md.vlayer[v] = keep_l[vis]; R.md.vmsgs[v] = keep_m[vis]; }
  R.proj(var, M);
  CK(cudaStreamSynchronize(R.st));
  std::memcpy(R.md.vlayer, keep_l, sizeof keep_l); std::memcpy(R.md.vmsgs, keep_m, sizeof keep_m);
  const int nd = y.wkd ? NDYN : 0;
  const int W = QD + QD + NRS + NSINK + KD + KD + 2 * nd;
  std::vector<uint16_t> out((size_t)M * W);
  if (var == "lt_fused" || var == "ex_fused") {
    const int nt = y.n_qside + y.n_kside + y.n_vside;
    if (!sep) {
      auto f = host16(R.b.f, (size_t)M * nt);
      for (int r = 0; r < M; ++r) std::memcpy(&out[(size_t)r * W], &f[(size_t)r * nt], (size_t)nt * 2);
    } else {
      auto f = host16(R.b.f, (size_t)M * y.n_qside), k = host16(R.b.k, (size_t)M * y.n_kside), v = host16(R.b.v, (size_t)M * y.n_vside);
      for (int r = 0; r < M; ++r) {
        uint16_t* o = &out[(size_t)r * W];
        std::memcpy(o, &f[(size_t)r * y.n_qside], (size_t)y.n_qside * 2);
        std::memcpy(o + y.n_qside, &k[(size_t)r * y.n_kside], KD * 2);
        std::memcpy(o + y.n_qside + KD, &v[(size_t)r * y.n_vside], KD * 2);
        if (nd) { std::memcpy(o + y.n_qside + 2 * KD, &k[(size_t)r * y.n_kside + KD], nd * 2);
                  std::memcpy(o + y.n_qside + 2 * KD + nd, &v[(size_t)r * y.n_vside + KD], nd * 2); }
      }
    }
  } else {
    auto q = host16(R.b.q, (size_t)M * QD), g = host16(R.b.g, (size_t)M * QD), rr = host16(R.b.r, (size_t)M * NRS),
         s = host16(R.b.s, (size_t)M * NSINK), k = host16(R.b.k, (size_t)M * KD), v = host16(R.b.v, (size_t)M * KD),
         kd = host16(R.b.kd, (size_t)M * NDYN), vd = host16(R.b.vd, (size_t)M * NDYN);
    for (int r = 0; r < M; ++r) {
      uint16_t* o = &out[(size_t)r * W];
      std::memcpy(o, &q[(size_t)r * QD], QD * 2); o += QD;
      std::memcpy(o, &g[(size_t)r * QD], QD * 2); o += QD;
      std::memcpy(o, &rr[(size_t)r * NRS], NRS * 2); o += NRS;
      std::memcpy(o, &s[(size_t)r * NSINK], NSINK * 2); o += NSINK;
      std::memcpy(o, &k[(size_t)r * KD], KD * 2); o += KD;
      std::memcpy(o, &v[(size_t)r * KD], KD * 2); o += KD;
      if (nd) { std::memcpy(o, &kd[(size_t)r * NDYN], nd * 2); o += nd; std::memcpy(o, &vd[(size_t)r * NDYN], nd * 2); }
    }
  }
  return out;
}

static void mode_bits(Runner& R, const std::vector<long long>& rows) {
  for (int vis : {46, 47, 20}) {
    for (long long M : rows) {
      auto ref = proj_out(R, "skinny", (int)M, vis);
      for (auto& var : variants_of("proj")) {
        auto o = proj_out(R, var, (int)M, vis);
        size_t diff = 0;
        for (size_t i = 0; i < o.size(); ++i) diff += o[i] != ref[i];
        // invariance: first 8 rows at M vs the same rows computed at M = 8
        auto o8 = proj_out(R, var, 8, vis);
        size_t W = o.size() / M, inv = 0;
        for (size_t i = 0; i < 8 * W; ++i) inv += o[i] != o8[i];
        p6::Line(kProbe, "bits").kv("op", "proj").kv("visit", vis).kv("layer", R.md.vlayer[vis])
            .kv("separate_inputs", R.md.separate(vis)).kv("variant", var).kv("rows", (int)M)
            .kv("elements", (unsigned long long)o.size()).kv("differ_vs_skinny", (unsigned long long)diff)
            .kv("rows0_7_differ_vs_M8", (unsigned long long)inv).emit();
      }
    }
  }
  for (const char* var : {"ex", "lt_sk"}) {   // gate down projections (last message visit): vs lt, and invariance vs M = 8
    R.gdown(var, 8);
    CK(cudaStreamSynchronize(R.st));
    auto g8 = host16(R.b.gd, 4 * 64 * GR);
    for (long long M : rows) {
      R.gdown("lt", (int)M);
      CK(cudaStreamSynchronize(R.st));
      auto a = host16(R.b.gd, 4 * 64 * GR);
      R.gdown(var, (int)M);
      CK(cudaStreamSynchronize(R.st));
      auto e = host16(R.b.gd, 4 * 64 * GR);
      size_t d = 0, inv = 0, n = 0;
      for (int g = 0; g < 4; ++g)
        for (int i = 0; i < M * GR; ++i) {
          const size_t k = (size_t)g * 64 * GR + i;
          d += a[k] != e[k];
          ++n;
          if (i < 8 * GR) inv += e[k] != g8[k];
        }
      p6::Line(kProbe, "bits").kv("op", "gdown").kv("variant", var).kv("rows", (int)M).kv("elements", (unsigned long long)n)
          .kv("differ_vs_lt", (unsigned long long)d).kv("rows0_7_differ_vs_M8", (unsigned long long)inv).emit();
    }
  }
  for (long long M : rows) {   // wo: ex vs lt
    R.wo("lt", (int)M);
    CK(cudaStreamSynchronize(R.st));
    auto a = host16(R.b.o, (size_t)M * H);
    R.wo("ex", (int)M);
    CK(cudaStreamSynchronize(R.st));
    auto e = host16(R.b.o, (size_t)M * H);
    size_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += a[i] != e[i];
    p6::Line(kProbe, "bits").kv("op", "wo").kv("variant", "ex").kv("rows", (int)M).kv("elements", (unsigned long long)a.size())
        .kv("differ_vs_lt", (unsigned long long)d).emit();
  }
}

// ------------------------------------------------------------------------------------------------ algos
struct Shape { const char* name; int N, K, ldw; bool f32; std::vector<const bf16*> W; };
static void mode_algos(p6::Ctx& c, Runner& R, const std::vector<long long>& rows) {
  std::vector<Shape> sh;
  auto per_visit = [&](auto f) { std::vector<const bf16*> w; for (int v = 0; v < NV; ++v) w.push_back(f(R.md.L[R.md.vlayer[v]])); return w; };
  sh.push_back({"q", QD, H, H, false, per_visit([](const Layer& y) { return (const bf16*)y.wq; })});
  sh.push_back({"gate", QD, H, H, false, per_visit([](const Layer& y) { return (const bf16*)y.wg; })});
  sh.push_back({"k", KD, H, H, false, per_visit([](const Layer& y) { return (const bf16*)y.wk; })});
  sh.push_back({"r", NRS, H, H, false, per_visit([](const Layer& y) { return (const bf16*)y.wr; })});
  sh.push_back({"sink", NSINK, H, H, false, per_visit([](const Layer& y) { return (const bf16*)y.ws; })});
  sh.push_back({"fused8600", 8600, H, H, false, per_visit([](const Layer& y) { return (const bf16*)y.fused; })});
  sh.push_back({"qside6552", 6552, H, H, false, per_visit([](const Layer& y) { return (const bf16*)y.fused; })});
  sh.push_back({"wo", H, QD, QD, false, per_visit([](const Layer& y) { return (const bf16*)y.wo; })});
  sh.push_back({"router1024", NEXP, 1024, H, true, per_visit([](const Layer& y) { return (const bf16*)y.router; })});
  {
    std::vector<const bf16*> w;
    for (int v = 0; v < NV; ++v) if (R.md.vmsgs[v]) for (int g = 0; g < 4; ++g) w.push_back(R.md.L[R.md.vlayer[v]].cdown[g]);
    w.resize(48);
    sh.push_back({"gdown", GR, 2 * H, 2 * H, false, w});
  }
  sh.push_back({"head", VOCAB_PAD, H, H, true, {R.md.unembed}});
  cublasLtHandle_t lh;
  CBS(cublasLtCreate(&lh));
  const auto filt = c.a.slist("--shapes", "");
  for (auto& s : sh) {
    if (!filt.empty() && std::find(filt.begin(), filt.end(), std::string(s.name)) == filt.end()) continue;
    for (long long M : rows) {
      cublasLtMatmulDesc_t op;
      CBS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
      CBS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
      CBS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
      cublasLtMatrixLayout_t la, lb, lc;
      CBS(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, s.K, s.N, s.ldw));
      const bf16* x = s.K == 2 * H ? R.b.xg : s.K == QD ? R.b.xo : R.b.xm;
      const int ldx = s.K == 2 * H ? 2 * H : s.K == QD ? QD : (s.ldw == H && s.K != H ? H : s.K);
      CBS(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, s.K, (int)M, ldx));
      CBS(cublasLtMatrixLayoutCreate(&lc, s.f32 ? CUDA_R_32F : CUDA_R_16BF, s.N, (int)M, s.N));
      void* out = s.f32 ? (void*)R.b.lg : (void*)R.b.lg;
      cublasLtMatmulPreference_t pref;
      CBS(cublasLtMatmulPreferenceCreate(&pref));
      size_t wsz = 32 << 20;
      CBS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
      cublasLtMatmulHeuristicResult_t res[64];
      int n = 0;
      CBS(cublasLtMatmulAlgoGetHeuristic(lh, op, la, lb, lc, lc, pref, 64, res, &n));
      cublasLtMatmulPreferenceDestroy(pref);
      const float one = 1.f, zero = 0.f;
      auto time_it = [&](const char* kind, int idx, std::function<void()> launch_all, int id, int tile, int stages, int sk,
                         size_t wsb, int swz) {
        launch_all();
        if (cudaStreamSynchronize(R.st) != cudaSuccess) { cudaGetLastError(); return; }
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(R.st, cudaStreamCaptureModeThreadLocal));
        launch_all();
        if (cudaStreamEndCapture(R.st, &g) != cudaSuccess) { cudaGetLastError(); return; }
        cudaGraphExec_t ge;
        CK(cudaGraphInstantiate(&ge, g, 0));
        p6::Line l(kProbe, "algo");
        l.kv("shape", s.name).kv("N", s.N).kv("K", s.K).kv("rows", (int)M).kv("kind", kind).kv("cand", idx).kv("algo_id", id)
            .kv("tile", tile).kv("stages", stages).kv("splitk", sk).kv("ws_bytes", (unsigned long long)wsb).kv("swizzle", swz)
            .kv("calls_per_graph", (int)s.W.size());
        const size_t by = (size_t)s.N * s.K * 2 * s.W.size();
        p6::sustained(c, l, [&](cudaStream_t) { cudaGraphLaunch(ge, R.st); }, by / 1e9, 4);
        l.emit();
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
      };
      for (int i = 0; i < n; ++i) {
        int id = 0, tile = 0, stg = 0, sk = 0, swz = 0;
        size_t w = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w);
        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &w);
        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stg, sizeof stg, &w);
        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof sk, &w);
        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof swz, &w);
        const cublasLtMatmulAlgo_t al = res[i].algo;
        const size_t wsb = res[i].workspaceSize;
        time_it("lt", i, [&]() {
          for (auto* W : s.W) cublasLtMatmul(lh, op, &one, W, la, x, lb, &zero, out, lc, out, lc, &al, R.b.ws, wsb, R.st);
        }, id, tile, stg, sk, wsb, swz);
      }
      time_it("gemmex_default", -1, [&]() { for (auto* W : s.W) R.ex(x, ldx, W, s.N, s.K, out, s.N, (int)M, s.f32); }, -1, -1, -1, -1, 0, 0);
      time_it("engine_lt", -1, [&]() {
        for (auto* W : s.W) R.lt.run(x, ldx, W, s.N, s.K, out, s.N, (int)M, s.f32, CUDA_R_16BF, s.ldw);
      }, -1, -1, -1, -1, 0, 0);
      if (!s.f32 && s.K == H && M <= 64) {
        for (int k2 = 0; k2 < 3; ++k2)
          time_it(k2 == 2 ? "sk2w8" : k2 ? "sk2" : "skinny", -1, [&]() {
            for (auto* W : s.W) {
              projs::Args a{};
              a.j[a.njobs++] = {W, x, H, (bf16*)out, s.N, s.N, H, 0};
              a.M = (int)M;
              if (k2 == 1) dd::launch_sk2<4>(a, R.st); else if (k2 == 2) dd::launch_sk2<8>(a, R.st); else projs::launch(a, R.st);
            }
          }, -1, -1, -1, -1, 0, 0);
      }
      cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
      cublasLtMatmulDescDestroy(op);
    }
  }
}

// ------------------------------------------------------------------------------------------------ sweep
static void mode_sweep(p6::Ctx& c, Runner& R, const std::vector<long long>& rows) {
  const auto ops = c.a.slist("--ops", "proj,wo,gdown,router,head");
  const auto vfilt = c.a.slist("--variants", "");
  for (auto& o : ops) {
    std::vector<std::string> vars;
    for (auto& v : variants_of(o))
      if (vfilt.empty() || std::find(vfilt.begin(), vfilt.end(), v) != vfilt.end()) vars.push_back(v);
    for (long long M : rows) {
      std::vector<cudaGraphExec_t> gs;
      for (auto& v : vars) gs.push_back(R.graph(o, v, (int)M));
      for (auto& sl : p6::sandwich((int)vars.size(), c.rounds)) {
        p6::Line l(kProbe, "sweep");
        l.kv("op", o).kv("variant", vars[sl.arm]).kv("rows", (int)M).kv("round", sl.round).kv("pos", sl.pos)
            .kv("bytes_per_call", (unsigned long long)R.bytes(o));
        auto ge = gs[sl.arm];
        p6::sustained(c, l, [&](cudaStream_t) { cudaGraphLaunch(ge, R.st); }, R.bytes(o) / 1e9, 4);
        l.emit();
      }
      for (auto g : gs) cudaGraphExecDestroy(g);
    }
  }
}

int main(int argc, char** argv) {
  p6::Ctx c;
  c.probe = kProbe;
  c.a = {argc, argv};
  p6::open_device(c);
  Model md;
  md.load(c.a.get("--model", "/model"));
  Runner R(md, c.s);
  p6::Line(kProbe, "setup").kv("proj_gb_step", md.proj_bytes_step / 1e9).kv("wo_gb_step", md.wo_bytes_step / 1e9)
      .kv("gdown_gb_step", md.gdown_bytes_step / 1e9).kv("router_gb_step", md.router_bytes_step / 1e9)
      .kv("lt_fallbacks", R.lt.fallbacks).emit();
  const auto rows = c.a.list("--rows", "8,32,64");
  const std::string mode = c.a.get("--mode", "sweep");
  if (mode == "bits") mode_bits(R, rows);
  else if (mode == "algos") mode_algos(c, R, rows);
  else mode_sweep(c, R, rows);
  p6::Line(kProbe, "end").kv("lt_fallbacks", R.lt.fallbacks).emit();
  c.tele->stop();
  return 0;
}
