// step_trace floor_bench (own code): the launch floor of a captured decode step. Reads the node list of one real step
// (nodes.csv from analyze.py: kind,gx,gy,gz,bx,by,bz,smem,bytes in graph order), captures a linear CUDA graph with the
// same node count where every kernel node is an EMPTY kernel with the same grid, block and shared memory (and every
// memcpy / memset node the same bytes), then replays it back to back and reports ms per replay, J per replay (NVML
// energy counter) and the SM clock.
//   floor_bench NODES.csv [--mode exact|tiny|nosmem] [--secs 10] [--rep 1]
//     exact : same grid, block, dynamic + static shared memory as the real node (as dynamic shared memory)
//     nosmem: same grid and block, no shared memory
//     tiny  : every kernel <<<1, 32>>> (pure per-node launch cost, no CTA distribution)
//   --rep R : the node list repeated R times in one graph (R steps per replay)
//   --sync 1: the engine's loop (launch, cudaStreamSynchronize, next launch): host launch time and wall per replay
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(2);                                                                            \
    }                                                                                          \
  } while (0)

__global__ void empty_k() {}

struct Node {
  char kind;
  int g[3], b[3], smem;
  long long bytes;
};

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: floor_bench NODES.csv [--mode exact|tiny|nosmem] [--secs S] [--rep R]\n"); return 2; }
  std::string mode = "exact";
  double secs = 10;
  int rep = 1, sync = 0;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--mode") mode = argv[i + 1];
    else if (k == "--sync") sync = std::atoi(argv[i + 1]);
    else if (k == "--secs") secs = std::atof(argv[i + 1]);
    else if (k == "--rep") rep = std::atoi(argv[i + 1]);
  }
  std::vector<Node> nodes;
  {
    std::ifstream f(argv[1]);
    std::string line;
    std::getline(f, line);   // header
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string c;
      Node n{};
      std::getline(ss, c, ','); n.kind = c[0];
      for (int j = 0; j < 3; ++j) { std::getline(ss, c, ','); n.g[j] = std::atoi(c.c_str()); }
      for (int j = 0; j < 3; ++j) { std::getline(ss, c, ','); n.b[j] = std::atoi(c.c_str()); }
      std::getline(ss, c, ','); n.smem = std::atoi(c.c_str());
      std::getline(ss, c, ','); n.bytes = std::atoll(c.c_str());
      nodes.push_back(n);
    }
  }
  int maxsm = 0;
  long long maxb = 1;
  for (auto& n : nodes) { maxsm = std::max(maxsm, n.smem); maxb = std::max(maxb, n.bytes); }
  CK(cudaFuncSetAttribute(empty_k, cudaFuncAttributeMaxDynamicSharedMemorySize, std::min(maxsm, 163 * 1024)));
  void *src, *dst;
  CK(cudaMalloc(&src, maxb));
  CK(cudaMalloc(&dst, maxb));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  int nk = 0, nc = 0, ns = 0;
  for (int r = 0; r < rep; ++r)
    for (auto& n : nodes) {
      if (n.kind == 'K') {
        dim3 g(n.g[0], n.g[1], n.g[2]), b(n.b[0], n.b[1], n.b[2]);
        int sm = std::min(n.smem, 163 * 1024);
        if (mode == "tiny") { g = dim3(1); b = dim3(32); sm = 0; }
        if (mode == "nosmem") sm = 0;
        empty_k<<<g, b, sm, st>>>();
        ++nk;
      } else if (n.kind == 'C') {
        CK(cudaMemcpyAsync(dst, src, n.bytes, cudaMemcpyDeviceToDevice, st));
        ++nc;
      } else if (n.kind == 'S') {
        CK(cudaMemsetAsync(dst, 0, n.bytes, st));
        ++ns;
      }
    }
  CK(cudaGetLastError());
  cudaGraph_t g;
  CK(cudaStreamEndCapture(st, &g));
  cudaGraphExec_t ex;
  CK(cudaGraphInstantiate(&ex, g, 0));
  for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(ex, st));
  CK(cudaStreamSynchronize(st));
  nvmlInit_v2();
  nvmlDevice_t dev;
  nvmlDeviceGetHandleByIndex_v2(0, &dev);
  unsigned long long e0 = 0, e1 = 0;
  std::vector<double> per;   // ms per replay, batches of 20
  std::vector<unsigned> clk;
  unsigned long long pw_sum = 0;
  int pw_n = 0;
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  nvmlDeviceGetTotalEnergyConsumption(dev, &e0);
  const double t0 = now();
  long long launches = 0;
  double last_sample = 0;
  std::vector<double> launch_us, wall_us;
  while (sync && now() - t0 < secs) {   // the engine's loop: launch, synchronise, next launch
    const double h0 = now();
    CK(cudaGraphLaunch(ex, st));
    const double h1 = now();
    CK(cudaStreamSynchronize(st));
    const double h2 = now();
    launch_us.push_back((h1 - h0) * 1e6);
    wall_us.push_back((h2 - h0) * 1e6);
    ++launches;
    if (now() - last_sample > 0.1) {
      unsigned c = 0, p = 0;
      nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &c);
      nvmlDeviceGetPowerUsage(dev, &p);
      clk.push_back(c);
      pw_sum += p;
      ++pw_n;
      last_sample = now();
    }
  }
  if (sync) {
    const double t1 = now();
    nvmlDeviceGetTotalEnergyConsumption(dev, &e1);
    std::sort(launch_us.begin(), launch_us.end());
    std::sort(wall_us.begin(), wall_us.end());
    std::sort(clk.begin(), clk.end());
    std::printf("{\"tool\":\"floor_bench\",\"mode\":\"%s\",\"sync\":1,\"rep\":%d,\"kernels\":%d,\"memcpys\":%d,\"memsets\":%d,"
                "\"launch_api_us_median\":%.2f,\"wall_us_median\":%.2f,\"wall_us_p10\":%.2f,\"wall_us_p90\":%.2f,"
                "\"wall_ms_mean\":%.5f,\"j_per_replay\":%.6f,\"mean_w\":%.2f,\"sm_mhz_median\":%u,\"replays\":%lld}\n",
                mode.c_str(), rep, nk, nc, ns, launch_us[launch_us.size() / 2], wall_us[wall_us.size() / 2],
                wall_us[wall_us.size() / 10], wall_us[wall_us.size() * 9 / 10], (t1 - t0) * 1e3 / launches,
                (e1 - e0) / 1e3 / launches, pw_n ? pw_sum / 1e3 / pw_n : 0.0, clk.empty() ? 0 : clk[clk.size() / 2], launches);
    return 0;
  }
  while (now() - t0 < secs) {
    CK(cudaEventRecord(a, st));
    for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ex, st));
    CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    per.push_back(ms / 20);
    launches += 20;
    if (now() - last_sample > 0.1) {
      unsigned c = 0, p = 0;
      nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &c);
      nvmlDeviceGetPowerUsage(dev, &p);
      clk.push_back(c);
      pw_sum += p;
      ++pw_n;
      last_sample = now();
    }
  }
  const double t1 = now();
  nvmlDeviceGetTotalEnergyConsumption(dev, &e1);
  std::sort(per.begin(), per.end());
  std::sort(clk.begin(), clk.end());
  const double med = per[per.size() / 2];
  const int nodes_per = nk + nc + ns;
  std::printf("{\"tool\":\"floor_bench\",\"mode\":\"%s\",\"rep\":%d,\"kernels\":%d,\"memcpys\":%d,\"memsets\":%d,"
              "\"ms_per_replay_median\":%.5f,\"ms_per_replay_p10\":%.5f,\"ms_per_replay_p90\":%.5f,\"us_per_node\":%.4f,"
              "\"wall_ms_per_replay\":%.5f,\"j_per_replay\":%.6f,\"mean_w\":%.2f,\"sm_mhz_median\":%u,\"replays\":%lld}\n",
              mode.c_str(), rep, nk, nc, ns, med, per[per.size() / 10], per[per.size() * 9 / 10], med * 1e3 / nodes_per,
              (t1 - t0) * 1e3 / launches, (e1 - e0) / 1e3 / launches, pw_n ? pw_sum / 1e3 / pw_n : 0.0,
              clk.empty() ? 0 : clk[clk.size() / 2], launches);
  return 0;
}
