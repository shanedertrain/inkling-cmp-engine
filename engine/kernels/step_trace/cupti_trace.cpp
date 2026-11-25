// step_trace: CUPTI activity-trace injection library (own code). Loaded by the CUDA driver through
// CUDA_INJECTION64_PATH=<this .so>; records every kernel (incl. CUDA-graph nodes), memcpy, memset and runtime-API call
// with GPU / host timestamps on CUPTI's common clock. Activity tracing only (no hardware counters, no profiler API), so it
// runs as an ordinary user without SYS_ADMIN.
//
// Environment:
//   INK_TRACE_OUT   output prefix (default /out/trace); writes <prefix>.<pid>.<n>.csv (one file per dump)
//   INK_TRACE_ARM   optional path: records are kept only while this file exists (polled every 50 ms); each time the
//                   file disappears the window's records are written. Without it every record is kept and written at
//                   process exit.
// CSV columns: kind,start_ns,end_ns,corr,graph_id,node_id,stream,gx,gy,gz,bx,by,bz,smem_static,smem_dyn,regs,bytes,name
//   kind: K kernel, C memcpy (name = copy kind), S memset, R runtime API (name = API function)
#include <cupti.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

struct Rec {
  char kind;
  uint64_t start, end;
  uint32_t corr, graph, stream;
  uint64_t node;
  int32_t g[3], b[3], ss, sd, regs;
  uint64_t bytes;
  int name;
};

std::mutex mu;
std::vector<Rec> recs;
std::vector<std::string> names;
std::unordered_map<std::string, int> name_ix;
std::string out_prefix = "/out/trace";
std::string arm_path;
std::atomic<uint64_t> win_lo{0}, win_hi{~0ull};   // keep records with start in [lo, hi)
std::atomic<bool> armed{false};
int dumps = 0;

int intern(const char* s) {
  std::string k = s ? s : "?";
  auto it = name_ix.find(k);
  if (it != name_ix.end()) return it->second;
  names.push_back(k);
  return name_ix[k] = (int)names.size() - 1;
}

#define CU(x)                                                                                 \
  do {                                                                                        \
    CUptiResult r_ = (x);                                                                     \
    if (r_ != CUPTI_SUCCESS) {                                                                \
      const char* m_ = nullptr;                                                               \
      cuptiGetResultString(r_, &m_);                                                          \
      std::fprintf(stderr, "[step_trace] %s failed: %s\n", #x, m_ ? m_ : "?");               \
    }                                                                                         \
  } while (0)

void CUPTIAPI buf_req(uint8_t** buf, size_t* size, size_t* maxrec) {
  *size = 8 << 20;
  *buf = static_cast<uint8_t*>(aligned_alloc(8, *size));
  *maxrec = 0;
}

void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
  CUpti_Activity* a = nullptr;
  std::lock_guard<std::mutex> lk(mu);
  const uint64_t lo = win_lo.load(), hi = win_hi.load();
  const bool use_win = !arm_path.empty();
  while (cuptiActivityGetNextRecord(buf, valid, &a) == CUPTI_SUCCESS) {
    Rec r{};
    switch (a->kind) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      case CUPTI_ACTIVITY_KIND_KERNEL: {
        auto* k = reinterpret_cast<CUpti_ActivityKernel12*>(a);
        r = {'K', k->start, k->end, k->correlationId, k->graphId, k->streamId, k->graphNodeId,
             {k->gridX, k->gridY, k->gridZ}, {k->blockX, k->blockY, k->blockZ}, k->staticSharedMemory,
             k->dynamicSharedMemory, (int32_t)k->registersPerThread, 0, 0};
        r.name = intern(k->name);
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        auto* m = reinterpret_cast<CUpti_ActivityMemcpy6*>(a);
        r = {'C', m->start, m->end, m->correlationId, m->graphId, m->streamId, m->graphNodeId, {0, 0, 0}, {0, 0, 0}, 0, 0, 0,
             m->bytes, 0};
        static const char* kinds[] = {"unknown", "HtoD", "DtoH", "HtoA", "AtoH", "AtoA", "AtoD", "DtoA", "DtoD", "HtoH", "PtoP"};
        r.name = intern(m->copyKind < 11 ? kinds[m->copyKind] : "other");
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMSET: {
        auto* m = reinterpret_cast<CUpti_ActivityMemset4*>(a);
        r = {'S', m->start, m->end, m->correlationId, m->graphId, m->streamId, m->graphNodeId, {0, 0, 0}, {0, 0, 0}, 0, 0, 0,
             m->bytes, 0};
        r.name = intern("memset");
        break;
      }
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        auto* p = reinterpret_cast<CUpti_ActivityAPI*>(a);
        const char* nm = nullptr;
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, p->cbid, &nm);
        r = {'R', p->start, p->end, p->correlationId, 0, p->threadId, 0, {0, 0, 0}, {0, 0, 0}, 0, 0, 0, 0, 0};
        r.name = intern(nm);
        break;
      }
      default:
        continue;
    }
    if (use_win && (r.start < lo || r.start >= hi)) continue;
    recs.push_back(r);
  }
  free(buf);
}

void dump() {
  CU(cuptiActivityFlushAll(1));
  std::lock_guard<std::mutex> lk(mu);
  if (recs.empty()) return;
  char fn[512];
  std::snprintf(fn, sizeof fn, "%s.%d.%d.csv", out_prefix.c_str(), (int)getpid(), dumps++);
  FILE* f = std::fopen(fn, "w");
  if (!f) { std::fprintf(stderr, "[step_trace] cannot write %s\n", fn); return; }
  std::fprintf(f, "kind,start_ns,end_ns,corr,graph_id,node_id,stream,gx,gy,gz,bx,by,bz,smem_static,smem_dyn,regs,bytes,name\n");
  for (auto& r : recs)
    std::fprintf(f, "%c,%llu,%llu,%u,%u,%llu,%u,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,\"%s\"\n", r.kind, (unsigned long long)r.start,
                 (unsigned long long)r.end, r.corr, r.graph, (unsigned long long)r.node, r.stream, r.g[0], r.g[1], r.g[2], r.b[0],
                 r.b[1], r.b[2], r.ss, r.sd, r.regs, (unsigned long long)r.bytes, names[r.name].c_str());
  std::fclose(f);
  std::fprintf(stderr, "[step_trace] %zu records -> %s\n", recs.size(), fn);
  recs.clear();
}

void poll_arm() {
  for (;;) {
    struct stat sb;
    const bool on = stat(arm_path.c_str(), &sb) == 0;
    if (on && !armed) {
      uint64_t t = 0;
      cuptiGetTimestamp(&t);
      win_lo = t;
      win_hi = ~0ull;
      armed = true;
      std::fprintf(stderr, "[step_trace] armed\n");
    } else if (!on && armed) {
      uint64_t t = 0;
      cuptiGetTimestamp(&t);
      win_hi = t;
      armed = false;
      dump();
      win_lo = ~0ull;   // nothing kept until armed again
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

}  // namespace

extern "C" int InitializeInjection(void) {
  if (const char* o = std::getenv("INK_TRACE_OUT")) out_prefix = o;
  if (const char* a = std::getenv("INK_TRACE_ARM")) arm_path = a;
  if (!arm_path.empty()) win_lo = ~0ull;
  CU(cuptiActivityRegisterCallbacks(buf_req, buf_done));
  // INK_TRACE_KINDS: letters of the kinds to record (default KCSR: kernels, memcpy, memset, runtime API). "R" alone
  // measures host API durations without the per-node cost kernel tracing adds to every graph launch.
  std::string kinds = std::getenv("INK_TRACE_KINDS") ? std::getenv("INK_TRACE_KINDS") : "KCSR";
  if (kinds.find('K') != std::string::npos) CU(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  if (kinds.find('C') != std::string::npos) CU(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  if (kinds.find('S') != std::string::npos) CU(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  if (kinds.find('R') != std::string::npos) CU(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CU(cuptiActivityFlushPeriod(500));
  if (!arm_path.empty()) std::thread(poll_arm).detach();
  else std::atexit(dump);
  std::fprintf(stderr, "[step_trace] injection active (out %s, arm %s)\n", out_prefix.c_str(), arm_path.empty() ? "-" : arm_path.c_str());
  return 1;
}
