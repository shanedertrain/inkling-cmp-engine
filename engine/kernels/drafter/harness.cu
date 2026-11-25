// inkling-cmp-engine S2f: standalone drafter harness (our code, 2026-11-23). Loads the real drafter weights
// (mtp.safetensors), the LM head / embedding (model safetensors) and config B's fold (fold.bin from
// probes/s2f/cfgb_drafter.py dumpw), runs one complete DSpark draft step per call for n sessions:
// commit (context K/V of the n*8 verified rows) + 3 layers over synthetic paged context K/V + Markov head (exact
// argmax, bf16 head or int8 head with error bound). Modes:
//   time   : per cell (n, L) and head variant, the whole step in one CUDA graph, sustained (K0 discipline, NVML)
//   eqhead : Markov head on config B's captured inputs (cap.bin), drafts compared with config B's drafts
// Harness pieces (JSON lines, NVML, sustained, safetensors) from probes/p6-attn/{common,safetensors}.hpp (ours).
#include "../../../probes/p6-attn/common.hpp"
#include "../../../probes/p6-attn/safetensors.hpp"
#include "drafter_kernels.cuh"

#include <map>
#include <memory>

using namespace s2f;
using bf16 = __nv_bfloat16;
static const char* PROBE = "s2f_drafter";
#define CK(x) P6_CK(PROBE, x)

static std::vector<void*>* g_track = nullptr;  // when set, allocations are recorded (freed by their owner)
template <class T>
static T* dalloc(size_t n) {
  T* p = nullptr;
  CK(cudaMalloc(&p, n * sizeof(T)));
  if (g_track) g_track->push_back(p);
  return p;
}
static bf16* load_bf16(const st::File& f, const std::string& name, size_t expect) {
  st::Tensor t = f.find(name);
  if (!t.ok || t.dtype != "BF16" || t.bytes() != expect * 2) p6::harness_fail(PROBE, "tensor", name.c_str());
  std::vector<char> h(t.bytes());
  if (!f.read(t, h.data())) p6::harness_fail(PROBE, "read", name.c_str());
  bf16* d = dalloc<bf16>(expect);
  CK(cudaMemcpy(d, h.data(), t.bytes(), cudaMemcpyHostToDevice));
  return d;
}
static bf16* cat_rows(const std::vector<bf16*>& parts, const std::vector<size_t>& rows, size_t cols) {
  size_t tot = 0;
  for (auto r : rows) tot += r;
  bf16* d = dalloc<bf16>(tot * cols);
  size_t off = 0;
  for (size_t i = 0; i < parts.size(); ++i) {
    CK(cudaMemcpy(d + off * cols, parts[i], rows[i] * cols * 2, cudaMemcpyDeviceToDevice));
    off += rows[i];
  }
  return d;
}
static bf16* to_frag(const bf16* w, int N, int K) {
  bf16* o = dalloc<bf16>((size_t)N * K);
  size_t th = (size_t)N * K / 8;
  permute_frag_bf16<<<(unsigned)((th + 255) / 256), 256>>>(w, o, N, K);
  CK(cudaGetLastError());
  return o;
}
__global__ void k_fill_bf16(bf16* p, size_t n, uint64_t seed, float scale) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint64_t h = p6::splitmix64(seed ^ (i * 0x9E3779B97F4A7C15ull));
    float u1 = p6::u01(h), u2 = p6::u01(p6::splitmix64(h));
    float g = sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);  // normal
    p[i] = __float2bfloat16_rn(g * scale);
  }
}
static void fill(bf16* p, size_t n, uint64_t seed, float scale) {
  k_fill_bf16<<<(unsigned)((n + 255) / 256), 256>>>(p, n, seed, scale);
  CK(cudaGetLastError());
}
__global__ void k_fill_e4m3(uint8_t* p, size_t n, uint64_t seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint64_t h = p6::splitmix64(seed ^ (i * 0x9E3779B97F4A7C15ull));
    float u1 = p6::u01(h), u2 = p6::u01(p6::splitmix64(h));
    p[i] = f2e4m3(sqrtf(-2.f * logf(u1)) * cospif(2.f * u2));
  }
}
__global__ void k_gather_cols(const bf16* head, const int* ochan, bf16* Wo) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)VP * NOUT) Wo[i] = head[(i / NOUT) * H + ochan[i % NOUT]];
}
__global__ void k_copy_cols(const bf16* src, int lds, int c0, int ncols, bf16* dst, int rows) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)rows * ncols) dst[i] = src[(i / ncols) * lds + c0 + i % ncols];
}
__global__ void k_cos_sin(float* cs, int maxpos) {  // vLLM _compute_cos_sin_cache, fp32
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)maxpos * 64) return;
  int pos = i / 64, d = i % 64;
  float inv = 1.0f / powf(1e6f, (float)(2 * d) / 128.0f);
  float f = (float)pos * inv;
  cs[(size_t)pos * 128 + d] = cosf(f);
  cs[(size_t)pos * 128 + 64 + d] = sinf(f);
}
// head inputs from captured h rows (eqhead mode): hout copy, f16 copy, x7 = [h, f_rnn], ||h||
__global__ void k_prep_head(const bf16* h, const bf16* f_rnn, int n, bf16* hout, __half* hf16, bf16* x7, float* hnorm) {
  int hr = blockIdx.x, r = hr / 7;
  float ss = 0.f;
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    bf16 v = h[(size_t)hr * H + k];
    float f = __bfloat162float(v);
    hout[(size_t)hr * H + k] = v;
    hf16[(size_t)hr * H + k] = __float2half_rn(f);
    x7[(size_t)hr * 2 * H + k] = v;
    x7[(size_t)hr * 2 * H + H + k] = f_rnn[(size_t)r * H + k];
    ss += f * f;
  }
  __shared__ float red[32];
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0;
    for (int i = 0; i < (int)blockDim.x / 32; ++i) t += red[i];
    hnorm[hr] = sqrtf(t) * (1.0f + 1e-5f) + 1e-6f;
  }
}

// ------------------------------------------------------------------------------------------------ weights
struct Layer {
  bf16 *in_norm, *post_norm, *ctx_norm, *q_norm, *k_norm, *k_norm_ctx;
  bf16 *qkv_f, *wo_f, *gu_f, *down_f, *kvctx_f;  // fragment layout
};
struct Weights {
  Layer L[3];
  bf16 *final_norm, *mask_embed, *embed, *embed_norm, *f_norm;
  bf16 *head, *head_f;  // row-major (exact candidate rows), fragment
  int8_t* head_q8f;     // int8 fragment (m16n8k16, f16 activations)
  int8_t* head_q8k32;   // int8 fragment (m16n8k32 IMMA, int8 activations)
  float* head_qn;       // ||q_v||_2 of the int8 head rows
  int* ochan = nullptr;     // outlier channels of the head input (exact path in the IMMA head), NOUT of them
  uint8_t* omask = nullptr; // [H] 1 on those channels
  bf16* Wo = nullptr;       // [VP][NOUT] head columns of those channels
  float *head_scale, *head_enorm, *head_rn;
  bf16 *w1, *w_state, *wctx_f, *fold_w, *fold_b;
  int8_t* fold_q;
  float *fold_s, *efold, *lrV, *lrP, *lrQ, *qnorm;
  bf16* lrU;
  float *rn, *rnn_b, *alpha;
  float* cos_sin;
};

static Weights load_weights(const std::string& model, const std::string& foldbin, bool want_int8) {
  Weights w{};
  st::File mtp, mdl;
  if (!mtp.open(model + "/mtp.safetensors")) p6::harness_fail(PROBE, "open", "mtp");
  if (!mdl.open(model + "/model-00001-of-00001.safetensors")) p6::harness_fail(PROBE, "open", "model");
  for (int i = 0; i < 3; ++i) {
    std::string p = "layers." + std::to_string(i) + ".", a = p + "self_attn.";
    Layer& L = w.L[i];
    L.in_norm = load_bf16(mtp, p + "input_layernorm.weight", H);
    L.post_norm = load_bf16(mtp, p + "post_attention_layernorm.weight", H);
    L.ctx_norm = load_bf16(mtp, p + "dflare_norm.weight", H);
    L.q_norm = load_bf16(mtp, a + "q_norm.weight", HD);
    L.k_norm = load_bf16(mtp, a + "k_norm.weight", HD);
    L.k_norm_ctx = load_bf16(mtp, a + "k_norm_ctx.weight", HD);
    bf16* q = load_bf16(mtp, a + "q_proj.weight", (size_t)NH * HD * H);
    bf16* k = load_bf16(mtp, a + "k_proj.weight", (size_t)KVH * HD * H);
    bf16* v = load_bf16(mtp, a + "v_proj.weight", (size_t)KVH * HD * H);
    bf16* qkv = cat_rows({q, k, v}, {(size_t)NH * HD, (size_t)KVH * HD, (size_t)KVH * HD}, H);
    L.qkv_f = to_frag(qkv, (NH + 2 * KVH) * HD, H);
    bf16* kc = load_bf16(mtp, a + "k_proj_ctx.weight", (size_t)KVH * HD * H);
    bf16* vc = load_bf16(mtp, a + "v_proj_ctx.weight", (size_t)KVH * HD * H);
    bf16* kv = cat_rows({kc, vc}, {(size_t)KVH * HD, (size_t)KVH * HD}, H);
    L.kvctx_f = to_frag(kv, 2 * KVH * HD, H);
    bf16* wo = load_bf16(mtp, a + "o_proj.weight", (size_t)H * NH * HD);
    L.wo_f = to_frag(wo, H, NH * HD);
    bf16* g = load_bf16(mtp, p + "mlp.gate_proj.weight", (size_t)INTER * H);
    bf16* u = load_bf16(mtp, p + "mlp.up_proj.weight", (size_t)INTER * H);
    bf16* gu = cat_rows({g, u}, {(size_t)INTER, (size_t)INTER}, H);
    L.gu_f = to_frag(gu, 2 * INTER, H);
    bf16* dn = load_bf16(mtp, p + "mlp.down_proj.weight", (size_t)H * INTER);
    L.down_f = to_frag(dn, H, INTER);
    CK(cudaDeviceSynchronize());
    for (bf16* x : {q, k, v, qkv, kc, vc, kv, wo, g, u, gu, dn}) cudaFree(x);
  }
  w.final_norm = load_bf16(mtp, "norm.weight", H);
  w.mask_embed = load_bf16(mtp, "mask_embed", H);
  w.f_norm = load_bf16(mtp, "markov_head.dflare_norm.weight", H);
  w.w1 = load_bf16(mtp, "markov_head.markov_w1.weight", (size_t)VP * RANK);
  bf16* rnn_w = load_bf16(mtp, "markov_head.rnn_proj.weight", (size_t)768 * 5120);
  w.w_state = dalloc<bf16>((size_t)768 * 512);
  bf16* wctx = dalloc<bf16>((size_t)768 * 4608);
  k_copy_cols<<<(768 * 512 + 255) / 256, 256>>>(rnn_w, 5120, 0, 512, w.w_state, 768);
  k_copy_cols<<<(768 * 4608 + 255) / 256, 256>>>(rnn_w, 5120, 512, 4608, wctx, 768);
  w.wctx_f = to_frag(wctx, 768, 4608);
  {
    bf16* b = load_bf16(mtp, "markov_head.rnn_proj.bias", 768);
    std::vector<uint16_t> hb(768);
    CK(cudaMemcpy(hb.data(), b, 768 * 2, cudaMemcpyDeviceToHost));
    std::vector<float> f(768);
    for (int i = 0; i < 768; ++i) f[i] = p6::bf16_to_f32(hb[i]);
    w.rnn_b = dalloc<float>(768);
    CK(cudaMemcpy(w.rnn_b, f.data(), 768 * 4, cudaMemcpyHostToDevice));
    // fuse_alpha = softmax(fuse_weight.float(), -1) (fp32, torch); rows 0..2 layers, row 3 f_rnn
    bf16* fw = load_bf16(mtp, "fuse_weight", 16);
    std::vector<uint16_t> hf(16);
    CK(cudaMemcpy(hf.data(), fw, 32, cudaMemcpyDeviceToHost));
    std::vector<float> al(16);
    for (int r = 0; r < 4; ++r) {
      float mx = -1e30f, s = 0.f;
      for (int c = 0; c < 4; ++c) mx = std::max(mx, p6::bf16_to_f32(hf[r * 4 + c]));
      for (int c = 0; c < 4; ++c) s += (al[r * 4 + c] = std::exp(p6::bf16_to_f32(hf[r * 4 + c]) - mx));
      for (int c = 0; c < 4; ++c) al[r * 4 + c] /= s;
    }
    w.alpha = dalloc<float>(16);
    CK(cudaMemcpy(w.alpha, al.data(), 64, cudaMemcpyHostToDevice));
  }
  w.head = load_bf16(mdl, "model.llm.unembed.weight", (size_t)VP * H);
  w.embed = load_bf16(mdl, "model.llm.embed.weight", (size_t)VP * H);
  w.embed_norm = load_bf16(mdl, "model.llm.embed_norm.weight", H);
  w.head_f = to_frag(w.head, VP, H);
  w.head_rn = dalloc<float>(VP);
  row_norms<<<VP / 8, 256>>>(w.head, w.head_rn, VP, H);
  if (want_int8) {
    int8_t* q = dalloc<int8_t>((size_t)VP * H);
    w.head_scale = dalloc<float>(VP);
    w.head_enorm = dalloc<float>(VP);
    quant_rows_i8<<<VP, 256>>>(w.head, q, w.head_scale, w.head_enorm, H);
    w.head_q8f = dalloc<int8_t>((size_t)VP * H);
    size_t th = (size_t)VP * H / 8;
    permute_frag_i8<<<(unsigned)((th + 255) / 256), 256>>>(q, w.head_q8f, VP, H);
    w.head_q8k32 = dalloc<int8_t>((size_t)VP * H);
    size_t th2 = (size_t)VP * H / 16;
    permute_frag_i8k32<<<(unsigned)((th2 + 255) / 256), 256>>>(q, w.head_q8k32, VP, H);
    w.head_qn = dalloc<float>(VP);
    row_norms_i8<<<VP / 8, 256>>>(q, w.head_qn, VP, H);
    CK(cudaDeviceSynchronize());
    cudaFree(q);
  }
  // fold (config B's own fp64 fold, dumped by cfgb_drafter.py dumpw)
  {
    FILE* f = std::fopen(foldbin.c_str(), "rb");
    if (!f) p6::harness_fail(PROBE, "open", foldbin.c_str());
    std::vector<uint16_t> fw((size_t)VP * RANK), fb(VP);
    if (std::fread(fw.data(), 2, fw.size(), f) != fw.size() || std::fread(fb.data(), 2, fb.size(), f) != fb.size())
      p6::harness_fail(PROBE, "read", "fold.bin");
    std::fclose(f);
    w.fold_w = dalloc<bf16>(fw.size());
    w.fold_b = dalloc<bf16>(VP);
    CK(cudaMemcpy(w.fold_w, fw.data(), fw.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w.fold_b, fb.data(), fb.size() * 2, cudaMemcpyHostToDevice));
    w.rn = dalloc<float>(VP);
    row_norms<<<VP / 8, 256>>>(w.fold_w, w.rn, VP, RANK);
    w.fold_q = dalloc<int8_t>((size_t)VP * RANK);
    w.fold_s = dalloc<float>(VP);
    w.efold = dalloc<float>(VP);
    quant_rows_i8<<<VP, 256>>>(w.fold_w, w.fold_q, w.fold_s, w.efold, RANK);
    w.qnorm = dalloc<float>(VP);
    row_norms_i8<<<VP / 8, 256>>>(w.fold_q, w.qnorm, VP, RANK);
    std::string lrbin = foldbin.substr(0, foldbin.rfind('/') + 1) + "lowrank16.bin";
    FILE* g = std::fopen(lrbin.c_str(), "rb");
    if (!g) p6::harness_fail(PROBE, "open", lrbin.c_str());
    std::vector<float> V(16 * 256), P(VP), Q(VP);
    std::vector<uint16_t> U((size_t)VP * 16);
    if (std::fread(V.data(), 4, V.size(), g) != V.size() || std::fread(U.data(), 2, U.size(), g) != U.size() ||
        std::fread(P.data(), 4, VP, g) != (size_t)VP || std::fread(Q.data(), 4, VP, g) != (size_t)VP)
      p6::harness_fail(PROBE, "read", "lowrank16.bin");
    std::fclose(g);
    w.lrV = dalloc<float>(V.size()); w.lrU = dalloc<bf16>(U.size()); w.lrP = dalloc<float>(VP); w.lrQ = dalloc<float>(VP);
    CK(cudaMemcpy(w.lrV, V.data(), V.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w.lrU, U.data(), U.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w.lrP, P.data(), VP * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w.lrQ, Q.data(), VP * 4, cudaMemcpyHostToDevice));
  }
  w.cos_sin = dalloc<float>((size_t)131072 * 128);
  k_cos_sin<<<131072 * 64 / 256, 256>>>(w.cos_sin, 131072);
  CK(cudaDeviceSynchronize());
  cudaFree(rnn_w);
  cudaFree(wctx);
  return w;
}

// ------------------------------------------------------------------------------------------------ GEMM launch
static size_t gemm_smem(int MT) { return (size_t)2 * MT * 8 * XS * 2; }
static int g_head_warps = 8;
static int g_sk_wo = 6, g_sk_down = 6, g_attn_ctas = 384;  // --sk-wo, --sk-down, --attn-ctas  // --head-warps: warps (16-row tiles) per CTA for the vocabulary GEMM
template <int MT, bool I8, int EPI, int NW>
static void gemm_w(const GemmArgs& a, int sk, cudaStream_t s) {
  static bool attr = false;
  if (!attr) {
    CK(cudaFuncSetAttribute(gemm_frag<MT, I8, EPI, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gemm_smem(MT)));
    attr = true;
  }
  dim3 grid((a.N / 16 + NW - 1) / NW, sk);
  gemm_frag<MT, I8, EPI, NW><<<grid, NW * 32, gemm_smem(MT), s>>>(a);
}
static int g_head_rt2 = 0;  // --head-rt2 0|1: vocabulary GEMM with two row tiles per warp
static int g_gemm3 = 0;      // --gemm3 0 (gemm_frag) | 1 (gemm_frag3) | 4 (gemm_frag4)
static int g_argmax2 = 3;   // 0: markov_argmax, 1: markov_argmax2, 3: markov_argmax3
static int g_dupcheck = 0;  // --dupcheck 1 (replay): rerun each draft and compare h bit for bit
static int g_overlap = 1;   // --overlap 0|1: graph branches for commit and rnn context (default on)
static int g_coop = 0;      // --coop 0|1: Markov loop as one cooperative kernel (measured slower: default off)   // --argmax2 0|1     // --gemm3 0|1: software-pipelined GEMM (all shapes)
template <int MT, bool I8, int EPI>
static void gemm_t(const GemmArgs& a, int sk, cudaStream_t s) {
  if (g_gemm3 == 1) {
    static bool attr3 = false;
    if (!attr3) {
      CK(cudaFuncSetAttribute(gemm_frag3<MT, I8, EPI, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gemm_smem(MT)));
      attr3 = true;
    }
    dim3 grid((a.N / 16 + 7) / 8, sk);
    gemm_frag3<MT, I8, EPI, 8><<<grid, 256, gemm_smem(MT), s>>>(a);
    return;
  }
  if (g_gemm3 == 4) {
    static bool attr4 = false;
    if (!attr4) {
      CK(cudaFuncSetAttribute(gemm_frag4<MT, I8, EPI, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gemm_smem(MT)));
      attr4 = true;
    }
    dim3 grid((a.N / 16 + 7) / 8, sk);
    gemm_frag4<MT, I8, EPI, 8><<<grid, 256, gemm_smem(MT), s>>>(a);
    return;
  }
  if (EPI != EPI_F32 && g_head_rt2) {
    static bool attr = false;
    if (!attr) {
      CK(cudaFuncSetAttribute(gemm_frag2<MT, I8, EPI, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gemm_smem(MT)));
      attr = true;
    }
    dim3 grid((a.N / 16 + 15) / 16, sk);
    gemm_frag2<MT, I8, EPI, 8><<<grid, 256, gemm_smem(MT), s>>>(a);
    return;
  }
  if (EPI != EPI_F32 && g_head_warps == 16) gemm_w<MT, I8, EPI, 16>(a, sk, s);
  else gemm_w<MT, I8, EPI, 8>(a, sk, s);
}
template <bool I8, int EPI>
static void gemm(GemmArgs a, int sk, cudaStream_t s) {
  a.kslice = a.K / sk;
  if (a.kslice % 16 || a.kslice * sk != a.K) p6::harness_fail(PROBE, "gemm", "bad split");
  int MT = (a.M + 7) / 8;
  switch (MT) {
    case 1: gemm_t<1, I8, EPI>(a, sk, s); break;
    case 2: gemm_t<2, I8, EPI>(a, sk, s); break;
    case 3: gemm_t<3, I8, EPI>(a, sk, s); break;
    case 4: gemm_t<4, I8, EPI>(a, sk, s); break;
    case 5: gemm_t<5, I8, EPI>(a, sk, s); break;
    case 6: gemm_t<6, I8, EPI>(a, sk, s); break;
    case 7: gemm_t<7, I8, EPI>(a, sk, s); break;
    case 8: gemm_t<8, I8, EPI>(a, sk, s); break;
    default: p6::harness_fail(PROBE, "gemm", "M > 64");
  }
}
template <int MT>
static void imma_t(const GemmI8Args& a, cudaStream_t s) {
  static bool attr = false;
  size_t sm = (size_t)2 * MT * 8 * XS8 + (size_t)MT * 8 * 32 * 4;
  if (!attr) {
    CK(cudaFuncSetAttribute(gemm_imma<MT, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    attr = true;
  }
  gemm_imma<MT, 8><<<(a.N / 16 + 7) / 8, 256, sm, s>>>(a);
}
static void gemm_imma_launch(const GemmI8Args& a, cudaStream_t s) {
  switch ((a.M + 7) / 8) {
    case 1: imma_t<1>(a, s); break;
    case 2: imma_t<2>(a, s); break;
    case 3: imma_t<3>(a, s); break;
    case 4: imma_t<4>(a, s); break;
    case 5: imma_t<5>(a, s); break;
    case 6: imma_t<6>(a, s); break;
    case 7: imma_t<7>(a, s); break;
    case 8: imma_t<8>(a, s); break;
    default: p6::harness_fail(PROBE, "imma", "M > 64");
  }
}
static GemmArgs ga(const void* W, const void* x, int ldx, int M, int N, int K, float* part, bf16* out = nullptr,
                   const float* scale = nullptr) {
  GemmArgs a{};
  a.W = W; a.x = (const bf16*)x; a.ldx = ldx; a.M = M; a.N = N; a.K = K; a.part = part; a.out = out; a.scale = scale;
  a.ldo = N;
  return a;
}

// ------------------------------------------------------------------------------------------------ real head inputs
// Captured config B head inputs (cap.bin) for time/kbench: NREC sets of n sessions (records concatenated row-wise).
struct RealHead {
  int nrec = 0;
  bf16 *h = nullptr, *f = nullptr;
  int* a = nullptr;
  int* counter = nullptr;
};
static RealHead load_real(const std::string& capbin, int n, int nrec_want) {
  RealHead rh;
  FILE* f = std::fopen(capbin.c_str(), "rb");
  if (!f) return rh;
  char magic[8];
  if (std::fread(magic, 1, 8, f) != 8) { std::fclose(f); return rh; }
  std::vector<uint16_t> H_, F_;
  std::vector<int> A_;
  int have = 0;  // sessions collected
  while ((int)A_.size() < n * nrec_want) {
    int m;
    if (std::fread(&m, 4, 1, f) != 1) break;
    std::vector<int> hdr(3 * m);
    std::vector<uint16_t> hh((size_t)m * 7 * H), fr((size_t)m * H);
    std::vector<int> ref(m * 7);
    if (std::fread(hdr.data(), 4, 3 * m, f) != (size_t)(3 * m) || std::fread(hh.data(), 2, hh.size(), f) != hh.size() ||
        std::fread(fr.data(), 2, fr.size(), f) != fr.size() || std::fread(ref.data(), 4, ref.size(), f) != ref.size())
      break;
    for (int r = 0; r < m && (int)A_.size() < n * nrec_want; ++r) {
      H_.insert(H_.end(), hh.begin() + (size_t)r * 7 * H, hh.begin() + (size_t)(r + 1) * 7 * H);
      F_.insert(F_.end(), fr.begin() + (size_t)r * H, fr.begin() + (size_t)(r + 1) * H);
      A_.push_back(hdr[m + r]);
      ++have;
    }
  }
  std::fclose(f);
  rh.nrec = have / n;
  if (!rh.nrec) return rh;
  rh.h = dalloc<bf16>((size_t)rh.nrec * n * 7 * H);
  rh.f = dalloc<bf16>((size_t)rh.nrec * n * H);
  rh.a = dalloc<int>(rh.nrec * n);
  rh.counter = dalloc<int>(1);
  CK(cudaMemcpy(rh.h, H_.data(), (size_t)rh.nrec * n * 7 * H * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rh.f, F_.data(), (size_t)rh.nrec * n * H * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rh.a, A_.data(), rh.nrec * n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(rh.counter, 0, 4));
  return rh;
}
// overwrite the head inputs with captured record (counter % nrec): hout, hf16, x7, hnorm, f_rnn, anchor
__global__ void k_real_head(RealHead rh, int n, bf16* hout, __half* hf16, bf16* x7, float* hnorm, bf16* f_rnn,
                            int* anchor) {
  int rec = *rh.counter % rh.nrec;
  int hr = blockIdx.x, r = hr / 7;
  const bf16* h = rh.h + ((size_t)rec * n * 7 + hr) * H;
  const bf16* fr = rh.f + ((size_t)rec * n + r) * H;
  float ss = 0.f;
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    bf16 v = h[k];
    float fv = __bfloat162float(v);
    hout[(size_t)hr * H + k] = v;
    hf16[(size_t)hr * H + k] = __float2half_rn(fv);
    x7[(size_t)hr * 2 * H + k] = v;
    x7[(size_t)hr * 2 * H + H + k] = fr[k];
    if (hr % 7 == 0) f_rnn[(size_t)r * H + k] = fr[k];
    ss += fv * fv;
  }
  __shared__ float red[32];
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0;
    for (int i = 0; i < (int)blockDim.x / 32; ++i) t += red[i];
    hnorm[hr] = sqrtf(t) * (1.0f + 1e-5f) + 1e-6f;
    if (hr % 7 == 0) anchor[r] = rh.a[rec * n + r];
  }
}
__global__ void k_bump(int* c) { atomicAdd(c, 1); }

// ------------------------------------------------------------------------------------------------ drafter state
struct Drafter {
  const Weights& w;
  int n, L, T, max_blocks;
  bool int8;
  int* dbg = nullptr;
  float* proj = nullptr;
  int8_t* latq = nullptr;
  bool imma = false;   // int8 head via IMMA with int8 activations (exact drafts through the combined bound)
  bool outl = false;   // + exact outlier channels
  int8_t* hq = nullptr;
  float *hsx = nullptr, *hdn = nullptr, *hn_in = nullptr;
  int ksteps = 7;  // drafts per round (verify depth); the layers always run the full 8-row block
  int vlim = VU;  // shortlist variant: first vlim rows of the head only (acceptance-changing, not output-changing)
  int *anchor, *P, *bt, *drafts, *nsurv;
  bf16 *Kc[3], *Vc[3];
  uint8_t *K8 = nullptr, *V8 = nullptr;  // kv8: E4M3 global-layer draft KV (acceptance-changing option)
  bool kv8 = false;
  bf16 *feats, *cmix, *f_rnn, *residual, *xn, *qbuf, *attn_out, *inter, *hout, *x7;
  __half* hf16;
  float *part, *opart, *ml, *ctx, *Bq, *hnorm, *lat2;
  bf16 *B, *state, *latent;
  unsigned long long *keys, *seedkeys;
  int split_attn[3];
  std::vector<void*> allocs;
  ~Drafter() {
    cudaDeviceSynchronize();
    for (void* p : allocs) cudaFree(p);
  }
  Drafter(const Weights& w_, int n_, int L_, bool i8, bool k8 = false) : w(w_), n(n_), L(L_), T(n_ * BLK), int8(i8), kv8(k8) {
    g_track = &allocs;
    int seqlen = L + BLK;
    max_blocks = (seqlen + PAGE - 1) / PAGE;
    std::vector<int> hbt(n * max_blocks), hP(n, L), ha(n);
    for (int i = 0; i < n * max_blocks; ++i) hbt[i] = 1 + i;
    for (int r = 0; r < n; ++r) ha[r] = 1000 + 7919 * r;
    anchor = dalloc<int>(n); P = dalloc<int>(n); bt = dalloc<int>(n * max_blocks);
    CK(cudaMemcpy(anchor, ha.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(P, hP.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bt, hbt.data(), hbt.size() * 4, cudaMemcpyHostToDevice));
    size_t cache_elems = (size_t)(n * max_blocks + 1) * KVH * PAGE * HD;
    for (int l = 0; l < 3; ++l) {
      Kc[l] = dalloc<bf16>(cache_elems); Vc[l] = dalloc<bf16>(cache_elems);
      fill(Kc[l], cache_elems, 11 + l, 1.0f);
      fill(Vc[l], cache_elems, 21 + l, 1.0f);
    }
    if (kv8) {
      K8 = dalloc<uint8_t>(cache_elems); V8 = dalloc<uint8_t>(cache_elems);
      k_fill_e4m3<<<(unsigned)((cache_elems + 255) / 256), 256>>>(K8, cache_elems, 41);
      k_fill_e4m3<<<(unsigned)((cache_elems + 255) / 256), 256>>>(V8, cache_elems, 42);
    }
    feats = dalloc<bf16>((size_t)T * 4 * H); fill(feats, (size_t)T * 4 * H, 31, 0.5f);
    cmix = dalloc<bf16>((size_t)3 * std::max(T, 64) * H); f_rnn = dalloc<bf16>((size_t)n * H);
    residual = dalloc<bf16>((size_t)T * H); xn = dalloc<bf16>((size_t)T * H);
    qbuf = dalloc<bf16>((size_t)T * NH * HD); attn_out = dalloc<bf16>((size_t)T * NH * HD);
    inter = dalloc<bf16>((size_t)T * INTER);
    hout = dalloc<bf16>((size_t)n * 7 * H); hf16 = dalloc<__half>((size_t)n * 7 * H);
    x7 = dalloc<bf16>((size_t)n * 7 * 2 * H);
    part = dalloc<float>((size_t)8 << 20);  // 32 MB split-K workspace
    for (int l = 0; l < 3; ++l) {
      int window = (l == 1) ? 0 : WIN + BLK;
      int keys = window ? std::min(window, seqlen) : seqlen;
      int tiles = (keys + AT_TILE - 1) / AT_TILE;
      split_attn[l] = std::max(1, std::min(tiles, (g_attn_ctas + 8 * n - 1) / (8 * n)));
      split_attn[l] = std::max(split_attn[l], (tiles + 1 + 255) / 256);  // attn_block holds <= 256 pages per CTA
    }
    int smax = std::max(split_attn[0], std::max(split_attn[1], split_attn[2]));
    opart = dalloc<float>((size_t)n * KVH * smax * 32 * HD);
    ml = dalloc<float>((size_t)n * KVH * smax * 32 * 2);
    ctx = dalloc<float>((size_t)n * 7 * 768);
    B = dalloc<bf16>((size_t)n * 7 * VP);
    Bq = int8 ? dalloc<float>((size_t)n * 7 * VP) : nullptr;
    hnorm = dalloc<float>(n * 7);
    state = dalloc<bf16>((size_t)2 * n * RANK); latent = dalloc<bf16>((size_t)n * RANK);
    lat2 = dalloc<float>(7 * n);
    keys = dalloc<unsigned long long>(7 * n); seedkeys = dalloc<unsigned long long>((size_t)7 * n * MK_NCH);
    drafts = dalloc<int>(7 * n); nsurv = dalloc<int>(7 * n);
    proj = dalloc<float>((size_t)n * 20);
    hq = dalloc<int8_t>((size_t)n * 7 * H);
    hsx = dalloc<float>(n * 7 * (H / KC));
    hdn = dalloc<float>(n * 7);
    hn_in = dalloc<float>(n * 7);
    latq = dalloc<int8_t>((size_t)n * RANK);
    CK(cudaDeviceSynchronize());
    g_track = nullptr;
  }
  Step step() const { Step s{n, anchor, P, bt, max_blocks}; return s; }
  CacheL cache(int l) const {
    CacheL c{Kc[l], Vc[l]};
    if (kv8 && l == 1) { c.k8 = K8; c.v8 = V8; }
    return c;
  }
  Markov markov() const {
    Markov m{};
    m.n = n; m.anchor = anchor; m.ctx = ctx; m.w_state = w.w_state; m.w1 = w.w1; m.fold_w = w.fold_w;
    m.fold_b = w.fold_b; m.rn = w.rn; m.B = B; m.Bq = Bq; m.enorm = w.head_enorm; m.hrn = w.head_rn;
    m.hnorm = hnorm; m.head = w.head; m.h = hout; m.state = state; m.latent = latent; m.lat2 = lat2;
    m.keys = keys; m.seedkeys = seedkeys; m.drafts = drafts; m.nsurv = nsurv;
    m.fold_q = w.fold_q; m.fold_s = w.fold_s; m.efold = w.efold;
    m.lrV = w.lrV; m.lrU = w.lrU; m.lrP = w.lrP; m.lrQ = w.lrQ;
    m.vlim = vlim;
    m.ksteps = ksteps;
    m.dbg = dbg;
    m.proj = proj;
    if (imma) { m.hdn = hdn; m.qhn = w.head_qn; m.hsc = w.head_scale; if (outl) m.hn_in = hn_in; }
    m.latq = latq;
    m.qnorm = w.qnorm;
    return m;
  }
  // concurrency (graph branches): the commit runs beside embed + layer-0 qkv; the rnn context GEMM beside the
  // vocabulary GEMM. g_overlap = 0 keeps everything on one stream.
  cudaStream_t s2 = nullptr;
  cudaEvent_t ev[4] = {};
  float* partC = nullptr;
  bool commit_pending = false;
  void commit(cudaStream_t s) {
    const Step st = step();
    float* part = partC ? partC : this->part;
    k_commit_mix<<<dim3(T, 4), 256, 0, s>>>(feats, w.alpha, w.L[0].ctx_norm, w.L[1].ctx_norm, w.L[2].ctx_norm,
                                            w.f_norm, cmix, f_rnn, T);
    const int sk = 9;
    for (int l = 0; l < 3; ++l)
      gemm<false, EPI_F32>(ga(w.L[l].kvctx_f, cmix + (size_t)l * T * H, H, T, 2 * KVH * HD, H,
                              part + (size_t)l * sk * T * 2048), sk, s);
    Caches cc;
    for (int l = 0; l < 3; ++l) cc.l[l] = cache(l);
    epi_kvctx<<<dim3(T, 3), 512, 0, s>>>(part, sk, T, st, w.L[0].k_norm_ctx, w.L[1].k_norm_ctx, w.L[2].k_norm_ctx,
                                         w.cos_sin, cc);
  }
  // Commit R rows of committed target features (feats [R][4][H] bf16 at positions pos[R], sessions sess[R] or
  // session 0) into the three layers' draft K/V: the engine's general commit (prefill chunks, verified rows).
  void commit_rows(const bf16* feats_d, const int* pos_d, const int* sess_d, int R, cudaStream_t s) {
    const Step st = step();
    Caches cc;
    for (int l = 0; l < 3; ++l) cc.l[l] = cache(l);
    const int sk = 9;
    for (int r0 = 0; r0 < R; r0 += 64) {
      const int Rc = std::min(64, R - r0);
      k_commit_mix<<<dim3(Rc, 3), 256, 0, s>>>(feats_d + (size_t)r0 * 4 * H, w.alpha, w.L[0].ctx_norm, w.L[1].ctx_norm,
                                              w.L[2].ctx_norm, w.f_norm, cmix, f_rnn, Rc);
      for (int l = 0; l < 3; ++l)
        gemm<false, EPI_F32>(ga(w.L[l].kvctx_f, cmix + (size_t)l * Rc * H, H, Rc, 2 * KVH * HD, H,
                                part + (size_t)l * sk * Rc * 2048), sk, s);
      epi_kvctx<<<dim3(Rc, 3), 512, 0, s>>>(part, sk, Rc, st, w.L[0].k_norm_ctx, w.L[1].k_norm_ctx, w.L[2].k_norm_ctx,
                                            w.cos_sin, cc, pos_d + r0, sess_d ? sess_d + r0 : nullptr);
    }
  }
  void layers(cudaStream_t s) {
    const Step st = step();
    k_embed_block<<<T, 256, 0, s>>>(st, w.embed, w.embed_norm, w.mask_embed, w.L[0].in_norm, residual, xn);
    for (int l = 0; l < 3; ++l) {
      const Layer& Lw = w.L[l];
      gemm<false, EPI_F32>(ga(Lw.qkv_f, xn, H, T, (NH + 2 * KVH) * HD, H, part), 4, s);
      epi_qkv<<<T, 1024, 0, s>>>(part, 4, T, st, Lw.q_norm, Lw.k_norm, w.cos_sin, cache(l), qbuf);
      int window = (l == 1) ? 0 : WIN + BLK;
      if (l == 0 && commit_pending) { CK(cudaStreamWaitEvent(s, ev[1], 0)); commit_pending = false; }
      if (kv8 && l == 1)
        attn_block8<<<dim3(split_attn[l], KVH, n), 128, ATTN_SMEM, s>>>(qbuf, st, cache(l), window, split_attn[l], opart, ml);
      else
        attn_block<<<dim3(split_attn[l], KVH, n), 128, ATTN_SMEM, s>>>(qbuf, st, cache(l), window, split_attn[l], opart, ml);
      attn_combine<<<n * KVH * 24, 128, 0, s>>>(opart, ml, split_attn[l], attn_out);
      gemm<false, EPI_F32>(ga(Lw.wo_f, attn_out, NH * HD, T, H, NH * HD, part), g_sk_wo, s);
      epi_addnorm<<<T, 768, 0, s>>>(part, g_sk_wo, T, residual, Lw.post_norm, xn, nullptr, nullptr, nullptr, nullptr,
                                    nullptr);
      gemm<false, EPI_F32>(ga(Lw.gu_f, xn, H, T, 2 * INTER, H, part), 2, s);
      epi_silu<<<(T * INTER + 255) / 256, 256, 0, s>>>(part, 2, T, inter);
      gemm<false, EPI_F32>(ga(Lw.down_f, inter, INTER, T, H, INTER, part), g_sk_down, s);
      if (l < 2)
        epi_addnorm<<<T, 768, 0, s>>>(part, g_sk_down, T, residual, w.L[l + 1].in_norm, xn, nullptr, nullptr, nullptr,
                                      nullptr, nullptr);
      else
        epi_addnorm<<<T, 768, 0, s>>>(part, g_sk_down, T, residual, w.final_norm, nullptr, hout, hf16, x7, f_rnn, hnorm);
    }
  }
  void head(cudaStream_t s) { head_gemm(s); head_rest(s); }
  void head_gemm(cudaStream_t s) {
    const int M = n * 7;
    CK(cudaMemsetAsync(keys, 0, 7 * n * 8, s));
    CK(cudaMemsetAsync(seedkeys, 0, (size_t)7 * n * MK_NCH * 8, s));
    CK(cudaMemsetAsync(lat2, 0, 7 * n * 4, s));
    CK(cudaMemsetAsync(nsurv, 0, 7 * n * 4, s));
    // shortlist: the GEMM covers the first NV rows only (the fragment layout stores 16-row tiles in row order, so the
    // prefix of the buffer is the prefix of the rows); outputs keep the VP row stride (ldo).
    const int NV = vlim == VU ? VP : vlim;
    if (imma) {
      k_quant_h<<<M, 288, 0, s>>>(hout, hq, hsx, hdn, M, outl ? w.omask : nullptr, outl ? hn_in : nullptr);
      GemmI8Args ia{(const uint4*)w.head_q8k32, hq, hsx, w.head_scale, Bq, M, NV, H, VP};
      if (outl) { ia.h = hout; ia.ochan = w.ochan; ia.Wo = w.Wo; }
      gemm_imma_launch(ia, s);
    } else {
      GemmArgs hg = int8 ? ga(w.head_q8f, hf16, H, M, NV, H, Bq, nullptr, w.head_scale) : ga(w.head_f, hout, H, M, NV, H, nullptr, B);
      hg.ldo = VP;
      if (int8) gemm<true, EPI_HEADQ>(hg, 1, s); else gemm<false, EPI_BF16>(hg, 1, s);
    }
  }
  void head_rest(cudaStream_t s) {
    const int M = n * 7;
    if (g_overlap && s2) {  // rnn context beside the vocabulary GEMM (its own workspace)
      CK(cudaEventRecord(ev[2], s));
      CK(cudaStreamWaitEvent(s2, ev[2], 0));
      gemm<false, EPI_F32>(ga(w.wctx_f, x7, 2 * H, M, 768, 2 * H, partC), 24, s2);
      epi_ctx<<<(M * 768 + 255) / 256, 256, 0, s2>>>(partC, 24, M, w.rnn_b, ctx);
      CK(cudaEventRecord(ev[3], s2));
    } else {
      gemm<false, EPI_F32>(ga(w.wctx_f, x7, 2 * H, M, 768, 2 * H, part), 24, s);
      epi_ctx<<<(M * 768 + 255) / 256, 256, 0, s>>>(part, 24, M, w.rnn_b, ctx);
    }
    if (g_overlap && s2) CK(cudaStreamWaitEvent(s, ev[3], 0));
    Markov mk = markov();
    dim3 sg((VU + 2047) / 2048, M);
    if (int8) markov_seed<true><<<sg, 256, 0, s>>>(mk); else markov_seed<false><<<sg, 256, 0, s>>>(mk);
    dim3 ag((VU + MK_CHUNK - 1) / MK_CHUNK, n);
    if (g_coop) {
      static int G8 = 0, G0 = 0;
      if (!G0) {
        int per8 = 0, per0 = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per8, markov_coop<true>, 256, 0));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per0, markov_coop<false>, 256, 0));
        int sms = 0;
        CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        G8 = std::min(per8, 2) * sms;
        G0 = std::min(per0, 2) * sms;
      }
      void* args[] = {(void*)&mk};
      if (int8) CK(cudaLaunchCooperativeKernel((void*)markov_coop<true>, G8, 256, args, 0, s));
      else CK(cudaLaunchCooperativeKernel((void*)markov_coop<false>, G0, 256, args, 0, s));
      return;
    }
    for (int i = 0; i < ksteps; ++i) {
      markov_recur<<<32, 256, 0, s>>>(mk, i);
      dim3 zg((MK_NCH + 7) / 8, n);
      if (int8) markov_zref<true><<<zg, 256, 0, s>>>(mk, i); else markov_zref<false><<<zg, 256, 0, s>>>(mk, i);
      if (g_argmax2 == 3 || imma) {
        dim3 a3(MK3_NCH, n);
        if (int8) markov_argmax3<true><<<a3, 256, 0, s>>>(mk, i); else markov_argmax3<false><<<a3, 256, 0, s>>>(mk, i);
      } else if (g_argmax2) {
        if (int8) markov_argmax2<true><<<ag, 256, 0, s>>>(mk, i); else markov_argmax2<false><<<ag, 256, 0, s>>>(mk, i);
      } else {
        if (int8) markov_argmax<true><<<ag, 256, 0, s>>>(mk, i); else markov_argmax<false><<<ag, 256, 0, s>>>(mk, i);
      }
    }
    markov_finish<<<1, 64, 0, s>>>(mk);
  }
  const RealHead* real = nullptr;
  void real_inputs(cudaStream_t s) {
    if (real && real->nrec) k_real_head<<<n * 7, 256, 0, s>>>(*real, n, hout, hf16, x7, hnorm, f_rnn, anchor);
  }
  void real_bump(cudaStream_t s) {
    if (real && real->nrec) k_bump<<<1, 1, 0, s>>>(real->counter);
  }
  void full(cudaStream_t s) {
    if (g_overlap) {
      if (!s2) {
        CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        partC = dalloc<float>((size_t)3 * 9 * 64 * 2048);
      }
      CK(cudaEventRecord(ev[0], s));
      CK(cudaStreamWaitEvent(s2, ev[0], 0));
      commit(s2);
      CK(cudaEventRecord(ev[1], s2));
      commit_pending = true;
      layers(s);
      real_inputs(s);
      head(s);
      real_bump(s);
      return;
    }
    commit(s); layers(s); real_inputs(s); head(s); real_bump(s);
  }
  double gbytes() const {  // bytes the step must read (weights, KV, head), not counting partials/activations
    double dense = 3.0 * ((double)(NH + 2 * KVH) * HD * H + (double)H * NH * HD + 2.0 * INTER * H + (double)H * INTER) * 2;
    double kvctx = 3.0 * 2 * KVH * HD * H * 2;
    double attn = 0;
    for (int l = 0; l < 3; ++l) {
      int keys = (l == 1) ? L + BLK : std::min(L + BLK, WIN + BLK);
      attn += (double)n * keys * KVH * HD * 2 * 2;
    }
    double head = ((int8 || imma) ? (double)H : (double)H * 2) * (vlim == VU ? VP : vlim);
    if (kv8) attn -= (double)n * (L + BLK) * KVH * HD * 2;  // global layer K/V at 1 byte
    return (dense + kvctx + attn + head + 768.0 * 4608 * 2) / 1e9;
  }
};

static cudaGraphExec_t capture(cudaStream_t s, const std::function<void(cudaStream_t)>& f) {
  f(s);
  CK(cudaStreamSynchronize(s));
  cudaGraph_t g;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  f(s);
  CK(cudaStreamEndCapture(s, &g));
  cudaGraphExec_t e;
  CK(cudaGraphInstantiate(&e, g, 0));
  cudaGraphDestroy(g);
  return e;
}

// ------------------------------------------------------------------------------------------------ eqhead
static void eqhead(p6::Ctx& c, const Weights& w, const std::string& capbin, const std::string& dump) {
  FILE* f = std::fopen(capbin.c_str(), "rb");
  if (!f) p6::harness_fail(PROBE, "open", capbin.c_str());
  char magic[8];
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "S2FCAP1", 7)) p6::harness_fail(PROBE, "magic", "cap");
  FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
  std::map<int, std::unique_ptr<Drafter>> D8, DX, DI;
  long eq_i = 0, calls_eq_i = 0;
  long long surv_i = 0;
  long calls = 0, tok = 0, eq_x = 0, eq_8 = 0, calls_eq_x = 0, calls_eq_8 = 0;
  long long surv_x = 0, surv_8 = 0;
  std::vector<long> hist_x(8, 0);
  while (true) {
    int n;
    if (std::fread(&n, 4, 1, f) != 1) break;
    std::vector<int> hdr(3 * n);
    if (std::fread(hdr.data(), 4, 3 * n, f) != (size_t)(3 * n)) break;
    std::vector<uint16_t> hh((size_t)n * 7 * H), fr((size_t)n * H);
    std::vector<int> ref(n * 7);
    if (std::fread(hh.data(), 2, hh.size(), f) != hh.size() || std::fread(fr.data(), 2, fr.size(), f) != fr.size() ||
        std::fread(ref.data(), 4, ref.size(), f) != ref.size())
      break;
    std::vector<int> out_x(n * 7), out_8(n * 7), sv_x(n * 7), sv_8(n * 7), out_i(n * 7), sv_i(n * 7);
    for (int variant = 0; variant < 3; ++variant) {
      auto& mp = variant == 2 ? DI : variant ? D8 : DX;
      if (!mp.count(n)) {
        mp[n] = std::make_unique<Drafter>(w, n, 64, variant >= 1);
        mp[n]->imma = variant == 2;
        mp[n]->outl = variant == 2 && c.a.has("--eq-outl");
        mp[n]->dbg = dalloc<int>(4);
        CK(cudaMemset(mp[n]->dbg, 0, 16));
      }
      Drafter& d = *mp[n];
      CK(cudaMemcpyAsync(d.anchor, hdr.data() + n, n * 4, cudaMemcpyHostToDevice, c.s));
      bf16* hraw = d.inter;  // scratch (>= n*7*H)
      CK(cudaMemcpyAsync(hraw, hh.data(), hh.size() * 2, cudaMemcpyHostToDevice, c.s));
      CK(cudaMemcpyAsync(d.f_rnn, fr.data(), fr.size() * 2, cudaMemcpyHostToDevice, c.s));
      k_prep_head<<<n * 7, 256, 0, c.s>>>(hraw, d.f_rnn, n, d.hout, d.hf16, d.x7, d.hnorm);
      d.head(c.s);
      std::vector<int>& o = variant == 2 ? out_i : variant ? out_8 : out_x;
      std::vector<int>& sv = variant == 2 ? sv_i : variant ? sv_8 : sv_x;
      CK(cudaMemcpyAsync(o.data(), d.drafts, n * 7 * 4, cudaMemcpyDeviceToHost, c.s));
      std::vector<int> ns(7 * n);
      CK(cudaMemcpyAsync(ns.data(), d.nsurv, 7 * n * 4, cudaMemcpyDeviceToHost, c.s));
      CK(cudaStreamSynchronize(c.s));
      for (int r = 0; r < n; ++r)
        for (int i = 0; i < 7; ++i) sv[r * 7 + i] = ns[i * n + r];
    }
    calls++;
    bool ax = true, a8 = true;
    for (int e = 0; e < n * 7; ++e) {
      tok++;
      eq_x += out_x[e] == ref[e];
      eq_8 += out_8[e] == ref[e];
      ax &= out_x[e] == ref[e];
      a8 &= out_8[e] == ref[e];
      surv_x += sv_x[e];
      surv_8 += sv_8[e];
      int b = 0;
      for (int s = sv_x[e]; s > 1 && b < 7; s >>= 2) ++b;
      hist_x[b]++;
    }
    calls_eq_x += ax;
    calls_eq_8 += a8;
    bool ai = true;
    for (int e = 0; e < n * 7; ++e) { eq_i += out_i[e] == ref[e]; ai &= out_i[e] == ref[e]; surv_i += sv_i[e]; }
    calls_eq_i += ai;
    if (df) {
      std::fwrite(&n, 4, 1, df);
      std::fwrite(hdr.data(), 4, 3 * n, df);
      std::fwrite(ref.data(), 4, n * 7, df);
      std::fwrite(out_x.data(), 4, n * 7, df);
      std::fwrite(out_8.data(), 4, n * 7, df);
      std::fwrite(sv_x.data(), 4, n * 7, df);
      std::fwrite(sv_8.data(), 4, n * 7, df);
    }
  }
  std::fclose(f);
  if (df) std::fclose(df);
  for (int variant = 0; variant < 2; ++variant) {
    long long tot[4] = {0, 0, 0, 0};
    for (auto& kv : (variant ? D8 : DX)) {
      int hd[4];
      CK(cudaMemcpy(hd, kv.second->dbg, 16, cudaMemcpyDeviceToHost));
      for (int i = 0; i < 4; ++i) tot[i] += hd[i];
    }
    p6::Line("s2f", "eqhead_stages").kv("variant", variant ? "int8" : "exact").kv("stage1_rows", tot[0])
        .kv("stage1b_rows", tot[1]).kv("survivors", tot[2]).kv("per_token_stage1", tok ? (double)tot[0] / tok : 0.0)
        .kv("per_token_stage1b", tok ? (double)tot[1] / tok : 0.0).kv("survivor_list_overflows", tot[3]).emit();
  }
  std::vector<double> hx(hist_x.begin(), hist_x.end());
  p6::Line("s2f", "eqhead").kv("calls", calls).kv("tokens", tok).kv("tokens_equal_exact", eq_x)
      .kv("tokens_equal_int8", eq_8).kv("calls_all_equal_exact", calls_eq_x).kv("calls_all_equal_int8", calls_eq_8)
      .kv("mean_survivors_exact", tok ? (double)surv_x / tok : 0.0).kv("mean_survivors_int8", tok ? (double)surv_8 / tok : 0.0)
      .raw("survivor_hist_log4_exact", p6::json_array(hx)).emit();
  p6::Line("s2f", "eqhead_imma").kv("tokens", tok).kv("tokens_equal_imma", eq_i).kv("calls_all_equal_imma", calls_eq_i)
      .kv("mean_survivors_imma", tok ? (double)surv_i / tok : 0.0).emit();
}

static double time_graph(cudaStream_t s, const std::function<void(cudaStream_t)>& f, int reps) {
  cudaGraphExec_t g = capture(s, f);
  for (int i = 0; i < 10; ++i) CK(cudaGraphLaunch(g, s));
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, s);
  for (int i = 0; i < reps; ++i) CK(cudaGraphLaunch(g, s));
  cudaEventRecord(b, s);
  CK(cudaEventSynchronize(b));
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaGraphExecDestroy(g);
  return ms / reps;
}

// ------------------------------------------------------------------------------------------------ replay
// Real-K/V replay of capture boot 2 (probes/s2f/cap2bin.py): every committed target row is committed into a per-request
// drafter (n = 1, paged K/V, bf16 + E4M3 copies), and at every captured draft call the full drafter (layers + exact
// head) runs twice, bf16 K/V and E4M3 global-layer K/V; h (rows 1-7) and f_rnn are compared with config B's served
// values, drafts with config B's drafts. One JSON line per draft call to out_jsonl.
static void replay(p6::Ctx& c, const Weights& w, const std::string& path, const std::string& outp, int max_events) {
  struct Ev { char k; int req, T, anchor, P; long off; };
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) p6::harness_fail(PROBE, "open", path.c_str());
  char magic[8];
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "S2FREP1", 7)) p6::harness_fail(PROBE, "magic", "rep");
  std::vector<Ev> evs;
  std::map<int, int> maxpos, last;
  while (true) {
    char k;
    if (std::fread(&k, 1, 1, f) != 1) break;
    Ev e{};
    e.k = k;
    if (k == 'F') {
      int hd[2];
      if (std::fread(hd, 4, 2, f) != 2) break;
      e.req = hd[0]; e.T = hd[1]; e.off = std::ftell(f);
      std::vector<int> pos(e.T);
      if (std::fread(pos.data(), 4, e.T, f) != (size_t)e.T) break;
      for (int p : pos) maxpos[e.req] = std::max(maxpos[e.req], p);
      std::fseek(f, (long)e.T * 4 * H * 2, SEEK_CUR);
    } else if (k == 'D') {
      int hd[3];
      if (std::fread(hd, 4, 3, f) != 3) break;
      e.req = hd[0]; e.anchor = hd[1]; e.P = hd[2]; e.off = std::ftell(f);
      maxpos[e.req] = std::max(maxpos[e.req], e.P + 8);
      std::fseek(f, (long)8 * H * 2 + 7 * 4, SEEK_CUR);
    } else p6::harness_fail(PROBE, "replay", "bad event");
    last[e.req] = (int)evs.size();
    evs.push_back(e);
  }
  FILE* out = std::fopen(outp.c_str(), "w");
  std::map<int, std::unique_ptr<Drafter>> D;
  std::map<int, std::vector<std::pair<int, std::vector<uint16_t>>>> lastrows;  // req -> (pos, feat row) of recent F
  bf16* feats_d = dalloc<bf16>((size_t)2048 * 4 * H);
  int* pos_d = dalloc<int>(2048);
  bf16* row_d = dalloc<bf16>((size_t)4 * H);
  long nd = 0, tok = 0, eq_ours = 0, eq_kv8 = 0, eq_ours_kv8 = 0;
  double h_eq_sum = 0, frnn_eq_sum = 0, hmax = 0;
  for (size_t ei = 0; ei < evs.size() && nd < max_events; ++ei) {
    const Ev& e = evs[ei];
    if (!D.count(e.req)) {
      D[e.req] = std::make_unique<Drafter>(w, 1, maxpos[e.req] + 8, false, true);
    }
    Drafter& d = *D[e.req];
    if (e.k == 'F') {
      std::vector<int> pos(e.T);
      std::vector<uint16_t> ft((size_t)e.T * 4 * H);
      std::fseek(f, e.off, SEEK_SET);
      if (std::fread(pos.data(), 4, e.T, f) != (size_t)e.T || std::fread(ft.data(), 2, ft.size(), f) != ft.size())
        p6::harness_fail(PROBE, "replay", "read F");
      {  // config B pads rejected rows of a decode step with position 0 (not committed): keep real rows only
        int w = 0;
        for (int r = 0; r < e.T; ++r)
          if (pos[r] != 0 || r == 0) {
            if (w != r) {
              pos[w] = pos[r];
              std::copy(ft.begin() + (size_t)r * 4 * H, ft.begin() + (size_t)(r + 1) * 4 * H, ft.begin() + (size_t)w * 4 * H);
            }
            ++w;
          }
        const_cast<Ev&>(e).T = w;
        pos.resize(w);
        ft.resize((size_t)w * 4 * H);
      }
      for (int r0 = 0; r0 < e.T; r0 += 2048) {
        int Rc = std::min(2048, e.T - r0);
        CK(cudaMemcpyAsync(pos_d, pos.data() + r0, Rc * 4, cudaMemcpyHostToDevice, c.s));
        CK(cudaMemcpyAsync(feats_d, ft.data() + (size_t)r0 * 4 * H, (size_t)Rc * 4 * H * 2, cudaMemcpyHostToDevice, c.s));
        d.commit_rows(feats_d, pos_d, nullptr, Rc, c.s);
        CK(cudaStreamSynchronize(c.s));
      }
      auto& lr = lastrows[e.req];
      int keep0 = std::max(0, e.T - 16);
      for (int r = keep0; r < e.T; ++r)
        lr.push_back({pos[r], std::vector<uint16_t>(ft.begin() + (size_t)r * 4 * H, ft.begin() + (size_t)(r + 1) * 4 * H)});
      if (lr.size() > 64) lr.erase(lr.begin(), lr.begin() + (lr.size() - 64));
    } else {
      std::vector<uint16_t> hid((size_t)7 * H), fr(H);
      std::vector<int> ref(7);
      std::fseek(f, e.off, SEEK_SET);
      if (std::fread(hid.data(), 2, hid.size(), f) != hid.size() || std::fread(fr.data(), 2, H, f) != (size_t)H ||
          std::fread(ref.data(), 4, 7, f) != 7)
        p6::harness_fail(PROBE, "replay", "read D");
      // f_rnn from the latest committed row at P-1
      auto& lr = lastrows[e.req];
      int found = -1;
      for (int i = (int)lr.size() - 1; i >= 0; --i)
        if (lr[i].first == e.P - 1) { found = i; break; }
      if (found < 0) { std::fprintf(out, "{\"req\":%d,\"P\":%d,\"skip\":\"no row at P-1\"}\n", e.req, e.P); continue; }
      CK(cudaMemcpyAsync(row_d, lr[found].second.data(), (size_t)4 * H * 2, cudaMemcpyHostToDevice, c.s));
      k_frnn<<<1, 256, 0, c.s>>>(row_d, w.alpha, w.f_norm, d.f_rnn);
      CK(cudaMemcpyAsync(d.anchor, &e.anchor, 4, cudaMemcpyHostToDevice, c.s));
      CK(cudaMemcpyAsync(d.P, &e.P, 4, cudaMemcpyHostToDevice, c.s));
      std::vector<uint16_t> frn(H);
      CK(cudaMemcpyAsync(frn.data(), d.f_rnn, H * 2, cudaMemcpyDeviceToHost, c.s));
      int dr[2][7];
      std::vector<uint16_t> hh[2];
      for (int v = 0; v < 2; ++v) {
        d.kv8 = v == 1;
        d.layers(c.s);
        d.head(c.s);
        hh[v].resize((size_t)7 * H);
        CK(cudaMemcpyAsync(hh[v].data(), d.hout, (size_t)7 * H * 2, cudaMemcpyDeviceToHost, c.s));
        CK(cudaMemcpyAsync(dr[v], d.drafts, 28, cudaMemcpyDeviceToHost, c.s));
        CK(cudaStreamSynchronize(c.s));
      }
      d.kv8 = true;
      int dup_same = -1;
      if (g_dupcheck) {  // determinism check: rerun the bf16 draft, compare h bit for bit
        d.kv8 = false;
        d.layers(c.s);
        std::vector<uint16_t> h2((size_t)7 * H);
        CK(cudaMemcpyAsync(h2.data(), d.hout, (size_t)7 * H * 2, cudaMemcpyDeviceToHost, c.s));
        CK(cudaStreamSynchronize(c.s));
        dup_same = h2 == hh[0];
        d.kv8 = true;
      }
      long heq = 0, feq = 0;
      double hm[2] = {0, 0}, hrel = 0;
      for (size_t i = 0; i < hid.size(); ++i) {
        float a = p6::bf16_to_f32(hid[i]);
        heq += hh[0][i] == hid[i];
        for (int v = 0; v < 2; ++v) hm[v] = std::max(hm[v], (double)std::fabs(p6::bf16_to_f32(hh[v][i]) - a));
        hrel = std::max(hrel, (double)std::fabs(p6::bf16_to_f32(hh[0][i]) - a) / (std::fabs(a) + 1e-2));
      }
      for (int i = 0; i < H; ++i) feq += frn[i] == fr[i];
      int eo = 0, e8 = 0, e88 = 0;
      for (int i = 0; i < 7; ++i) { eo += dr[0][i] == ref[i]; e8 += dr[1][i] == ref[i]; e88 += dr[1][i] == dr[0][i]; }
      nd++; tok += 7; eq_ours += eo; eq_kv8 += e8; eq_ours_kv8 += e88;
      h_eq_sum += (double)heq / hid.size(); frnn_eq_sum += (double)feq / H; hmax = std::max(hmax, hm[0]);
      std::fprintf(out, "{\"req\":%d,\"P\":%d,\"anchor\":%d,\"cfgb\":[%d,%d,%d,%d,%d,%d,%d],\"ours\":[%d,%d,%d,%d,%d,%d,%d],"
                   "\"kv8\":[%d,%d,%d,%d,%d,%d,%d],\"h_eq\":%.5f,\"h_maxabs\":%.5g,\"h_maxrel\":%.5g,\"h8_maxabs\":%.5g,\"frnn_eq\":%.5f,\"dup_same\":%d}\n",
                   e.req, e.P, e.anchor, ref[0], ref[1], ref[2], ref[3], ref[4], ref[5], ref[6], dr[0][0], dr[0][1], dr[0][2],
                   dr[0][3], dr[0][4], dr[0][5], dr[0][6], dr[1][0], dr[1][1], dr[1][2], dr[1][3], dr[1][4], dr[1][5], dr[1][6],
                   (double)heq / hid.size(), hm[0], hrel, hm[1], (double)feq / H, dup_same);
    }
    if (last[e.req] == (int)ei) { D.erase(e.req); lastrows.erase(e.req); }
  }
  std::fclose(out);
  std::fclose(f);
  p6::Line("s2f", "replay").kv("draft_calls", nd).kv("tokens", tok).kv("ours_eq_cfgb", eq_ours).kv("kv8_eq_cfgb", eq_kv8)
      .kv("kv8_eq_ours", eq_ours_kv8).kv("h_bitequal_frac_mean", nd ? h_eq_sum / nd : 0.0)
      .kv("frnn_bitequal_frac_mean", nd ? frnn_eq_sum / nd : 0.0).kv("h_maxabs_max", hmax).emit();
}

// ------------------------------------------------------------------------------------------------ main
int main(int argc, char** argv) {
  p6::Ctx c;
  c.probe = PROBE;
  c.a = p6::Args{argc, argv};
  p6::open_device(c);
  std::string mode = c.a.get("--mode", "time");
  std::string model = c.a.get("--model", "/model");
  std::string foldbin = c.a.get("--fold", "/data/fold.bin");
  CK(cudaFuncSetAttribute(attn_block, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM));
  CK(cudaFuncSetAttribute(attn_block8, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM));
  g_sk_wo = (int)c.a.num("--sk-wo", 6);
  g_sk_down = (int)c.a.num("--sk-down", 6);
  g_attn_ctas = (int)c.a.num("--attn-ctas", 384);
  g_head_warps = (int)c.a.num("--head-warps", 8);
  g_head_rt2 = (int)c.a.num("--head-rt2", 0);
  g_gemm3 = (int)c.a.num("--gemm3", 0);
  g_argmax2 = (int)c.a.num("--argmax2", 3);
  g_coop = (int)c.a.num("--coop", 0);
  g_overlap = (int)c.a.num("--overlap", 1);
  g_dupcheck = (int)c.a.num("--dupcheck", 0);
  Weights w = load_weights(model, foldbin, true);
  {
    std::string capbin = c.a.get("--cap", "/data/cap.bin");
    FILE* f = std::fopen(capbin.c_str(), "rb");
    std::vector<float> mx(H, 0.f);
    int rows = 0;
    if (f) {
      char magic[8];
      if (std::fread(magic, 1, 8, f) == 8) {
        while (rows < 256) {
          int m;
          if (std::fread(&m, 4, 1, f) != 1) break;
          std::vector<int> hdr(3 * m);
          std::vector<uint16_t> hh((size_t)m * 7 * H), fr((size_t)m * H);
          std::vector<int> ref(m * 7);
          if (std::fread(hdr.data(), 4, 3 * m, f) != (size_t)(3 * m) || std::fread(hh.data(), 2, hh.size(), f) != hh.size() ||
              std::fread(fr.data(), 2, fr.size(), f) != fr.size() || std::fread(ref.data(), 4, ref.size(), f) != ref.size())
            break;
          for (size_t i = 0; i < hh.size(); ++i) mx[i % H] = std::max(mx[i % H], std::fabs(p6::bf16_to_f32(hh[i])));
          rows += m * 7;
        }
      }
      std::fclose(f);
    }
    std::vector<int> idx(H);
    for (int i = 0; i < H; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + NOUT, idx.end(), [&](int a2, int b2) { return mx[a2] > mx[b2]; });
    std::vector<int> oc(idx.begin(), idx.begin() + NOUT);
    std::vector<uint8_t> om(H, 0);
    for (int k : oc) om[k] = 1;
    w.ochan = dalloc<int>(NOUT);
    w.omask = dalloc<uint8_t>(H);
    CK(cudaMemcpy(w.ochan, oc.data(), NOUT * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w.omask, om.data(), H, cudaMemcpyHostToDevice));
    w.Wo = dalloc<bf16>((size_t)VP * NOUT);
    k_gather_cols<<<(VP * NOUT + 255) / 256, 256>>>(w.head, w.ochan, w.Wo);
    CK(cudaDeviceSynchronize());
    p6::Line("s2f", "outliers").kv("calib_rows", rows).kv("max_abs_top", (double)mx[oc[0]])
        .kv("max_abs_33rd", (double)mx[idx[NOUT]]).emit();
  }
  p6::Line("s2f", "weights").kv("ok", true).emit();
  if (mode == "replay") {
    replay(c, w, c.a.get("--rep", "/data/replay.bin"), c.a.get("--out-jsonl", "/out/replay.jsonl"),
           (int)c.a.num("--max-events", 1e9));
    c.tele->stop();
    return 0;
  }
  if (mode == "eqhead") {
    eqhead(c, w, c.a.get("--cap", "/data/cap.bin"), c.a.get("--dump", ""));
    c.tele->stop();
    return 0;
  }
  if (mode == "kbench") {
    for (auto& cs : c.a.slist("--cells", "4:2048,4:44729")) {
      int n = std::atoi(cs.c_str()), L = std::atoi(cs.c_str() + cs.find(':') + 1);
      for (int v8 = 0; v8 < 2; ++v8) {
        Drafter d(w, n, L, v8 == 1);
        RealHead rh = load_real(c.a.get("--cap", "/data/cap.bin"), n, 64);
        d.real = &rh;
        const int T = d.T, M = n * 7;
        const Step st = d.step();
        auto rep = [&](const char* name, double gb, const std::function<void(cudaStream_t)>& f) {
          double ms = time_graph(c.s, f, 200);
          p6::Line("s2f", "kbench").kv("op", name).kv("n", n).kv("L", L).kv("int8", v8).kv("ms", ms)
              .kv("gb", gb).kv("gbs", gb / (ms * 1e-3)).emit();
        };
        const Layer& Lw = w.L[0];
        rep("full", d.gbytes(), [&](cudaStream_t s) { d.full(s); });
        rep("commit", 0.03, [&](cudaStream_t s) { d.commit(s); });
        rep("layers", 0, [&](cudaStream_t s) { d.layers(s); });
        rep("head", 0, [&](cudaStream_t s) { d.real_inputs(s); d.head(s); d.real_bump(s); });
        {
          int keep = g_coop;
          g_coop = 0;
          rep("head_coop0", 0, [&](cudaStream_t s) { d.real_inputs(s); d.head(s); d.real_bump(s); });
          g_coop = 1;
          rep("head_coop1", 0, [&](cudaStream_t s) { d.real_inputs(s); d.head(s); d.real_bump(s); });
          for (int k : {1, 3, 5}) {
            d.ksteps = k;
            char nm[32];
            std::snprintf(nm, sizeof nm, "head_coop1_k%d", k);
            rep(nm, 0, [&](cudaStream_t s) { d.real_inputs(s); d.head(s); d.real_bump(s); });
          }
          d.ksteps = 7;
          g_coop = keep;
        }
        {  // Markov pieces on real inputs (state after one full head pass)
          d.real_inputs(c.s); d.head(c.s); CK(cudaStreamSynchronize(c.s));
          Markov mk = d.markov();
          dim3 sg((VU + 2047) / 2048, M), zg((MK_NCH + 7) / 8, n), ag((VU + MK_CHUNK - 1) / MK_CHUNK, n);
          rep("mk_seed", 0, [&](cudaStream_t s) { if (v8) markov_seed<true><<<sg, 256, 0, s>>>(mk); else markov_seed<false><<<sg, 256, 0, s>>>(mk); });
          rep("mk_recur", 0, [&](cudaStream_t s) { markov_recur<<<32, 256, 0, s>>>(mk, 3); });
          rep("mk_zref", 0, [&](cudaStream_t s) { if (v8) markov_zref<true><<<zg, 256, 0, s>>>(mk, 3); else markov_zref<false><<<zg, 256, 0, s>>>(mk, 3); });
          rep("mk_argmax", 0, [&](cudaStream_t s) { if (v8) markov_argmax<true><<<ag, 256, 0, s>>>(mk, 3); else markov_argmax<false><<<ag, 256, 0, s>>>(mk, 3); });
          rep("epi_ctx", 0, [&](cudaStream_t s) { epi_ctx<<<(M * 768 + 255) / 256, 256, 0, s>>>(d.part, 24, M, w.rnn_b, d.ctx); });
          {
            Markov m3 = mk; m3.vlim = 2048;
            rep("mk_argmax_vlim2048", 0, [&](cudaStream_t s) { if (v8) markov_argmax<true><<<ag, 256, 0, s>>>(m3, 3); else markov_argmax<false><<<ag, 256, 0, s>>>(m3, 3); });
            dim3 a1(1, n);
            rep("mk_argmax_grid1", 0, [&](cudaStream_t s) { if (v8) markov_argmax<true><<<a1, 256, 0, s>>>(mk, 3); else markov_argmax<false><<<a1, 256, 0, s>>>(mk, 3); });
            dim3 a3(MK3_NCH, n);
            rep("mk_argmax3", 0, [&](cudaStream_t s) { if (v8) markov_argmax3<true><<<a3, 256, 0, s>>>(mk, 3); else markov_argmax3<false><<<a3, 256, 0, s>>>(mk, 3); });
            rep("mk_argmax2", 0, [&](cudaStream_t s) { if (v8) markov_argmax2<true><<<ag, 256, 0, s>>>(mk, 3); else markov_argmax2<false><<<ag, 256, 0, s>>>(mk, 3); });
            rep("mk_finish", 0, [&](cudaStream_t s) { markov_finish<<<1, 64, 0, s>>>(mk); });
          }
          {  // stage pass counts of one fresh step-3 argmax (keys reset to the zref seeds first)
            int* dbg = dalloc<int>(3);
            CK(cudaMemset(dbg, 0, 12));
            Markov m2 = mk;
            m2.dbg = dbg;
            CK(cudaMemsetAsync(d.keys + 3 * n, 0, n * 8, c.s));
            if (v8) markov_zref<true><<<zg, 256, 0, c.s>>>(m2, 3); else markov_zref<false><<<zg, 256, 0, c.s>>>(m2, 3);
            if (v8) markov_argmax<true><<<ag, 256, 0, c.s>>>(m2, 3); else markov_argmax<false><<<ag, 256, 0, c.s>>>(m2, 3);
            int hd[3];
            CK(cudaMemcpy(hd, dbg, 12, cudaMemcpyDeviceToHost));
            p6::Line("s2f", "kbench").kv("op", "mk_stage_counts").kv("n", n).kv("int8", v8).kv("stage1", hd[0])
                .kv("stage1b", hd[1]).kv("stage2", hd[2]).kv("ms", 0.0).kv("gbs", 0.0).kv("L", L).emit();
            cudaFree(dbg);
          }
        }
        if (v8) continue;
        float* qbufM = dalloc<float>((size_t)M * VP);  // fp32 [M][VP] output for the int8 head GEMMs
        for (int g3 : {0, 4, 0, 4}) {
          g_gemm3 = g3;
          std::string sfx = g3 ? "_g4" : "_g1";
          if (!g3) {
            k_quant_h<<<M, 288, 0, c.s>>>(d.hout, d.hq, d.hsx, d.hdn, M);
            GemmI8Args ia{(const uint4*)w.head_q8k32, d.hq, d.hsx, w.head_scale, qbufM, M, VP, H, VP};
            rep(("gemm_head_imma" + sfx).c_str(), (double)VP * H / 1e9, [&](cudaStream_t s) { gemm_imma_launch(ia, s); });
          }
          rep(("gemm_head_i8" + sfx).c_str(), (double)VP * H / 1e9, [&](cudaStream_t s) { gemm<true, EPI_HEADQ>(ga(w.head_q8f, d.hf16, H, M, VP, H, qbufM, nullptr, w.head_scale), 1, s); });
          rep(("gemm_head_bf16" + sfx).c_str(), (double)VP * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_BF16>(ga(w.head_f, d.hout, H, M, VP, H, nullptr, d.B), 1, s); });
          rep(("gemm_gu_sk2" + sfx).c_str(), 18432.0 * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.gu_f, d.xn, H, T, 2 * INTER, H, d.part), 2, s); });
          rep(("gemm_down_sk6" + sfx).c_str(), 2304.0 * INTER * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.down_f, d.inter, INTER, T, H, INTER, d.part), 6, s); });
          rep(("gemm_qkv_sk4" + sfx).c_str(), 5120.0 * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.qkv_f, d.xn, H, T, 5120, H, d.part), 4, s); });
          rep(("gemm_wo_sk6" + sfx).c_str(), 2304.0 * 3072 * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.wo_f, d.attn_out, 3072, T, H, 3072, d.part), 6, s); });
        }
        g_gemm3 = (int)c.a.num("--gemm3", 0);
        rep("gemm_qkv_sk4", 5120.0 * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.qkv_f, d.xn, H, T, 5120, H, d.part), 4, s); });
        rep("gemm_qkv_sk2", 5120.0 * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.qkv_f, d.xn, H, T, 5120, H, d.part), 2, s); });
        rep("gemm_wo_sk12", 2304.0 * 3072 * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.wo_f, d.attn_out, 3072, T, H, 3072, d.part), 12, s); });
        rep("gemm_wo_sk6", 2304.0 * 3072 * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.wo_f, d.attn_out, 3072, T, H, 3072, d.part), 6, s); });
        rep("gemm_gu_sk2", 18432.0 * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.gu_f, d.xn, H, T, 2 * INTER, H, d.part), 2, s); });
        rep("gemm_gu_sk1", 18432.0 * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.gu_f, d.xn, H, T, 2 * INTER, H, d.part), 1, s); });
        rep("gemm_down_sk12", 2304.0 * INTER * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.down_f, d.inter, INTER, T, H, INTER, d.part), 12, s); });
        rep("gemm_down_sk6", 2304.0 * INTER * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(Lw.down_f, d.inter, INTER, T, H, INTER, d.part), 6, s); });
        g_head_rt2 = 1;
        rep("gemm_head_i8_rt2", (double)VP * H / 1e9, [&](cudaStream_t s) { gemm<true, EPI_HEADQ>(ga(w.head_q8f, d.hf16, H, M, VP, H, qbufM, nullptr, w.head_scale), 1, s); });
        rep("gemm_head_bf16_rt2", (double)VP * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_BF16>(ga(w.head_f, d.hout, H, M, VP, H, nullptr, d.B), 1, s); });
        g_head_rt2 = 0;
        rep("gemm_head_bf16", (double)VP * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_BF16>(ga(w.head_f, d.hout, H, M, VP, H, nullptr, d.B), 1, s); });
        g_head_rt2 = 0;
        g_head_warps = 16;
        rep("gemm_head_bf16_w16", (double)VP * H * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_BF16>(ga(w.head_f, d.hout, H, M, VP, H, nullptr, d.B), 1, s); });
        rep("gemm_head_i8_w16", (double)VP * H / 1e9, [&](cudaStream_t s) { gemm<true, EPI_HEADQ>(ga(w.head_q8f, d.hf16, H, M, VP, H, qbufM, nullptr, w.head_scale), 1, s); });
        g_head_warps = 8;
        rep("gemm_head_i8_rt1", (double)VP * H / 1e9, [&](cudaStream_t s) { gemm<true, EPI_HEADQ>(ga(w.head_q8f, d.hf16, H, M, VP, H, qbufM, nullptr, w.head_scale), 1, s); });
        g_head_rt2 = 1;
        rep("gemm_ctx_sk24", 768.0 * 4608 * 2 / 1e9, [&](cudaStream_t s) { gemm<false, EPI_F32>(ga(w.wctx_f, d.x7, 2 * H, M, 768, 2 * H, d.part), 24, s); });
        rep("epi_qkv", 0, [&](cudaStream_t s) { epi_qkv<<<T, 1024, 0, s>>>(d.part, 4, T, st, Lw.q_norm, Lw.k_norm, w.cos_sin, CacheL{d.Kc[0], d.Vc[0]}, d.qbuf); });
        rep("epi_addnorm", 0, [&](cudaStream_t s) { epi_addnorm<<<T, 768, 0, s>>>(d.part, 6, T, d.residual, Lw.post_norm, d.xn, nullptr, nullptr, nullptr, nullptr, nullptr); });
        rep("epi_silu", 0, [&](cudaStream_t s) { epi_silu<<<(T * INTER + 255) / 256, 256, 0, s>>>(d.part, 2, T, d.inter); });
        for (int l = 0; l < 3; l += 1) {
          int window = (l == 1) ? 0 : WIN + BLK;
          int keys = window ? std::min(window, L + BLK) : L + BLK;
          char nm[64];
          std::snprintf(nm, sizeof nm, "attn_l%d_s%d", l, d.split_attn[l]);
          rep(nm, (double)n * keys * KVH * HD * 4 / 1e9, [&](cudaStream_t s) {
            attn_block<<<dim3(d.split_attn[l], KVH, n), 128, ATTN_SMEM, s>>>(d.qbuf, st, CacheL{d.Kc[l], d.Vc[l]}, window, d.split_attn[l], d.opart, d.ml);
          });
          rep("attn_combine", 0, [&](cudaStream_t s) { attn_combine<<<n * KVH * 24, 128, 0, s>>>(d.opart, d.ml, d.split_attn[l], d.attn_out); });
        }
        if (rh.nrec) { cudaFree(rh.h); cudaFree(rh.f); cudaFree(rh.a); cudaFree(rh.counter); }
        cudaFree(qbufM);
      }
      {  // FP8 (E4M3) global-layer draft K/V: attention kernel alone and the full step (exact head)
        Drafter dk(w, n, L, true, true);
        RealHead rh = load_real(c.a.get("--cap", "/data/cap.bin"), n, 64);
        dk.real = &rh;
        const Step st = dk.step();
        double ms = time_graph(c.s, [&](cudaStream_t s) {
          attn_block8<<<dim3(dk.split_attn[1], KVH, n), 128, ATTN_SMEM, s>>>(dk.qbuf, st, dk.cache(1), 0, dk.split_attn[1], dk.opart, dk.ml);
        }, 200);
        double gb = (double)n * (L + BLK) * KVH * HD * 2 / 1e9;
        p6::Line("s2f", "kbench").kv("op", "attn_l1_kv8").kv("n", n).kv("L", L).kv("int8", 1).kv("ms", ms).kv("gb", gb)
            .kv("gbs", gb / (ms * 1e-3)).emit();
        double ms2 = time_graph(c.s, [&](cudaStream_t s) { dk.full(s); }, 100);
        p6::Line("s2f", "kbench").kv("op", "full_int8_kv8").kv("n", n).kv("L", L).kv("int8", 1).kv("ms", ms2).kv("gb", 0.0)
            .kv("gbs", 0.0).emit();
        if (rh.nrec) { cudaFree(rh.h); cudaFree(rh.f); cudaFree(rh.a); cudaFree(rh.counter); }
      }
    }
    c.tele->stop();
    return 0;
  }
  auto cells = c.a.slist("--cells", "1:2048,4:2048,8:2048,1:44729,4:44729,8:44729");
  auto variants = c.a.slist("--variants", "exact,int8");
  bool parts = c.a.has("--parts");
  for (auto& cs : cells) {
    int n = std::atoi(cs.c_str()), L = std::atoi(cs.c_str() + cs.find(':') + 1);
    for (int round = 0; round < c.rounds; ++round) {
      std::vector<std::string> vv = variants;
      if (round & 1) std::reverse(vv.begin(), vv.end());
      for (auto& v : vv) {
        const bool vim = v.find("imma") != std::string::npos;
        Drafter d(w, n, L, vim || v.find("int8") != std::string::npos || v.rfind("sl49k8", 0) == 0,
                  v.find("kv8") != std::string::npos);
        d.imma = vim;
        d.outl = v.find("immaO") != std::string::npos;
        if (v.rfind("sl", 0) == 0) d.vlim = 49152;
        d.ksteps = (int)c.a.num("--ksteps", 7);
        RealHead rh = load_real(c.a.get("--cap", "/data/cap.bin"), n, 64);
        d.real = &rh;
        std::vector<std::pair<std::string, cudaGraphExec_t>> gs;
        gs.push_back({"full", capture(c.s, [&](cudaStream_t s) { d.full(s); })});
        if (parts && round == 0) {
          gs.push_back({"commit", capture(c.s, [&](cudaStream_t s) { d.commit(s); })});
          gs.push_back({"layers", capture(c.s, [&](cudaStream_t s) { d.layers(s); })});
          gs.push_back({"head", capture(c.s, [&](cudaStream_t s) { d.real_inputs(s); d.head(s); d.real_bump(s); })});
          gs.push_back({"hgemm", capture(c.s, [&](cudaStream_t s) { d.real_inputs(s); d.head_gemm(s); d.real_bump(s); })});
        }
        for (auto& [name, ge] : gs) {
          p6::Line l("s2f", "step");
          l.kv("part", name).kv("variant", v).kv("n", n).kv("L", L).kv("round", round).kv("gb_min", d.gbytes());
          double secs = c.secs;
          if (name != "full") c.secs = c.a.num("--part-secs", 10.0);
          p6::sustained(c, l, [&](cudaStream_t s) { CK(cudaGraphLaunch(ge, s)); }, d.gbytes(), 4);
          c.secs = secs;
          l.emit();
          cudaGraphExecDestroy(ge);
        }
        CK(cudaDeviceSynchronize());
        if (rh.nrec) { cudaFree(rh.h); cudaFree(rh.f); cudaFree(rh.a); cudaFree(rh.counter); }
        p6::Line("s2f", "realhead").kv("n", n).kv("nrec", rh.nrec).emit();
      }
    }
  }
  c.tele->stop();
  return 0;
}
