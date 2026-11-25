// attnres_bench (own code): per-launch time of the glue row kernels at decode row counts, standalone. Compiled against a
// glue.cuh given by -I (so two engine revisions can be compared with the same harness). Each case: a CUDA graph of 200
// launches of one kernel on T rows (sources: 4 bf16 blocks + the fp32 or bf16 last source, as a loop visit), replayed
// 50 times; us per launch = event time / 10,000. L2-warm (the same rows every launch), like the engine's back-to-back
// visits whose sources were just written.
//   attnres_bench [T]   (default 8)
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "glue.cuh"

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(2);                                                                            \
    }                                                                                          \
  } while (0)

using glue::bf16;
constexpr int H = glue::H;

template <class F>
float time_graph(cudaStream_t st, F launch) {
  cudaGraph_t g;
  cudaGraphExec_t ex;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < 200; ++i) launch();
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ex, g, 0));
  for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ex, st));
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a, st));
  for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(ex, st));
  CK(cudaEventRecord(b, st));
  CK(cudaEventSynchronize(b));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  CK(cudaGraphExecDestroy(ex));
  CK(cudaGraphDestroy(g));
  return ms * 1e3f / (200 * 50);
}

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 8;
  std::vector<uint16_t> hb((size_t)T * H);
  std::vector<float> hf((size_t)4 * T * H);
  srand(1);
  for (auto& x : hb) x = 0x3f80 ^ (rand() & 0x40ff);   // bf16 values around +-1
  for (auto& x : hf) x = (rand() / (float)RAND_MAX - 0.5f);
  bf16* blk[4];
  for (auto& p : blk) {
    CK(cudaMalloc(&p, hb.size() * 2));
    CK(cudaMemcpy(p, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
  }
  float *last32, *queries, *out, *msg, *w;
  bf16 *o0, *o1, *o2, *gin;
  CK(cudaMalloc(&last32, hf.size() * 4));
  CK(cudaMalloc(&queries, (size_t)4 * H * 4));
  CK(cudaMalloc(&out, (size_t)4 * T * H * 4));
  CK(cudaMalloc(&msg, (size_t)4 * T * H * 4));
  CK(cudaMalloc(&w, (size_t)H * 4));
  for (auto** p : {&o0, &o1, &o2}) CK(cudaMalloc(p, (size_t)T * H * 2));
  CK(cudaMalloc(&gin, (size_t)3 * T * 2 * H * 2));
  CK(cudaMemcpy(last32, hf.data(), hf.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(queries, hf.data(), (size_t)4 * H * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(msg, hf.data(), (size_t)4 * T * H * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(w, hf.data(), (size_t)H * 4, cudaMemcpyHostToDevice));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const float eps = 1e-6f;
  for (int nb = 1; nb <= 4; ++nb) {
    glue::ResSrc s{};
    for (int i = 0; i < nb; ++i) s.blk[i] = blk[i];
    s.nb = nb;
    s.last32 = last32;
    s.ld_last = H;
    glue::ResSrc s16 = s;
    s16.last32 = nullptr;
    s16.last16 = blk[3];
    const float t1 = time_graph(st, [&] { glue::attn_res_kernel<1><<<T, 256, 0, st>>>(s, queries, out, eps); });
    const float t3 = time_graph(st, [&] { glue::attn_res_kernel<3><<<T, 256, 0, st>>>(s, queries, out, eps); });
    const float t4 = time_graph(st, [&] { glue::attn_res_kernel<4><<<T, 256, 0, st>>>(s, queries, out, eps); });
    const float n1 = time_graph(st, [&] {
      glue::attn_res_norm_kernel<1><<<T, 256, 0, st>>>(s, queries, w, nullptr, nullptr, o0, nullptr, nullptr, eps);
    });
    const float n3 = time_graph(st, [&] {
      glue::attn_res_norm_kernel<3><<<T, 256, 0, st>>>(s16, queries, w, w, w, o0, o1, o2, eps);
    });
    std::printf("{\"T\":%d,\"nb\":%d,\"attn_res_q1_us\":%.2f,\"attn_res_q3_us\":%.2f,\"attn_res_q4_us\":%.2f,"
                "\"attn_res_norm_q1_us\":%.2f,\"attn_res_norm_q3_us\":%.2f}\n", T, nb, t1, t3, t4, n1, n3);
  }
  const float rn = time_graph(st, [&] { glue::rmsnorm_kernel<float><<<T, 256, 0, st>>>(last32, H, w, o0, H, eps); });
  const float mg = time_graph(st, [&] { glue::mg_in_kernel<<<dim3(T, 3), 256, 0, st>>>(last32, H, msg, H, gin, eps, H, H, (long long)T * 2 * H); });
  std::printf("{\"T\":%d,\"rmsnorm_us\":%.2f,\"mg_in_g3_us\":%.2f}\n", T, rn, mg);
  return 0;
}
