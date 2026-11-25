// S4 prefill MoE bench: shared harness, copied unchanged (except this line) from engine/kernels/prefill_attn/harness.hpp, itself from probes/p6-attn/common.hpp (our code): JSON lines, NVML 10 Hz
// telemetry, sustained timing, sandwich A/B. Ported there from the job K0/W Python harness. No third-party parts.
//
// Contract for every P6 binary:
//   - stdout: one JSON object per line; the first is {"probe":"meta",...} (device, UUID, vBIOS, power limit, tag).
//   - Exit 2 on a harness error (setup that must work); a failed *check* is a result line with "ok":false.
//   - Timing discipline (K0/W): every timed arm runs back to back for --secs (default 24 s, >= 20 s), chunks of
//     launches between CUDA events, never more than 3 chunks queued; chunks that start in the first --discard s
//     (default 2 s, GPU time) are dropped; NVML sampled at 10 Hz in a side thread (power, SM/mem clock, clock-event
//     reasons incl. the sw power-cap bit, temperature, total energy counter). Arms run in sandwich order per round
//     (baseline, arms..., baseline), --rounds rounds (default 2), arm order reversed in odd rounds.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace p6 {

// ---------------------------------------------------------------- JSON line builder
class Line {
 public:
  Line(const char* probe, const char* test) {
    s_ = "{\"probe\":";
    str_(probe);
    s_ += ",\"test\":";
    str_(test);
  }
  Line& kv(const char* k, const char* v) { key_(k); str_(v ? v : ""); return *this; }
  Line& kv(const char* k, const std::string& v) { return kv(k, v.c_str()); }
  Line& kv(const char* k, bool v) { key_(k); s_ += v ? "true" : "false"; return *this; }
  Line& kv(const char* k, int v) { return kv(k, (long long)v); }
  Line& kv(const char* k, unsigned v) { return kv(k, (long long)v); }
  Line& kv(const char* k, long v) { return kv(k, (long long)v); }
  Line& kv(const char* k, unsigned long v) { return kv(k, (unsigned long long)v); }
  Line& kv(const char* k, long long v) { key_(k); s_ += std::to_string(v); return *this; }
  Line& kv(const char* k, unsigned long long v) { key_(k); s_ += std::to_string(v); return *this; }
  Line& kv(const char* k, double v) {
    key_(k);
    if (!std::isfinite(v)) { s_ += "null"; return *this; }
    char b[64];
    std::snprintf(b, sizeof b, "%.7g", v);
    s_ += b;
    return *this;
  }
  Line& raw(const char* k, const std::string& json) { key_(k); s_ += json; return *this; }
  Line& rt(cudaError_t e) {
    kv("ok", e == cudaSuccess);
    if (e != cudaSuccess) { kv("err", cudaGetErrorName(e)); kv("err_str", cudaGetErrorString(e)); }
    return *this;
  }
  void emit() {
    s_ += "}";
    std::fputs(s_.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  }

 private:
  void key_(const char* k) { s_ += ","; str_(k); s_ += ":"; }
  void str_(const char* v) {
    s_ += '"';
    for (const char* p = v; *p; ++p) {
      unsigned char c = (unsigned char)*p;
      if (c == '"' || c == '\\') { s_ += '\\'; s_ += (char)c; }
      else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); s_ += b; }
      else s_ += (char)c;
    }
    s_ += '"';
  }
  std::string s_;
};

inline std::string json_array(const std::vector<double>& v) {
  std::string s = "[";
  char b[48];
  for (size_t i = 0; i < v.size(); ++i) {
    std::snprintf(b, sizeof b, "%s%.7g", i ? "," : "", v[i]);
    s += b;
  }
  return s + "]";
}

[[noreturn]] inline void harness_fail(const char* probe, const char* what, const char* err) {
  Line(probe, "harness_error").kv("ok", false).kv("what", what).kv("err", err).emit();
  std::exit(2);
}
#define P6_CK(probe, call)                                                         \
  do {                                                                             \
    cudaError_t e_ = (call);                                                       \
    if (e_ != cudaSuccess) ::p6::harness_fail(probe, #call, cudaGetErrorName(e_)); \
  } while (0)

// ---------------------------------------------------------------- arguments
struct Args {
  int argc;
  char** argv;
  const char* get(const char* name, const char* def) const {
    for (int i = 1; i + 1 < argc; ++i)
      if (!std::strcmp(argv[i], name)) return argv[i + 1];
    return def;
  }
  bool has(const char* name) const {
    for (int i = 1; i < argc; ++i)
      if (!std::strcmp(argv[i], name)) return true;
    return false;
  }
  double num(const char* name, double def) const {
    const char* v = get(name, nullptr);
    return v ? std::atof(v) : def;
  }
  std::vector<long long> list(const char* name, const char* def) const {
    std::vector<long long> out;
    std::string s = get(name, def);
    size_t p = 0;
    while (p < s.size()) {
      size_t q = s.find(',', p);
      if (q == std::string::npos) q = s.size();
      if (q > p) out.push_back(std::atoll(s.substr(p, q - p).c_str()));
      p = q + 1;
    }
    return out;
  }
  std::vector<std::string> slist(const char* name, const char* def) const {
    std::vector<std::string> out;
    std::string s = get(name, def);
    size_t p = 0;
    while (p < s.size()) {
      size_t q = s.find(',', p);
      if (q == std::string::npos) q = s.size();
      if (q > p) out.push_back(s.substr(p, q - p));
      p = q + 1;
    }
    return out;
  }
};

// ---------------------------------------------------------------- NVML through dlopen (no header/link dependency;
// the NVIDIA container runtime injects libnvidia-ml.so.1). Values are NVML's documented enum constants.
struct Nvml {
  typedef int (*fn_init)();
  typedef int (*fn_by_pci)(const char*, void**);
  typedef int (*fn_by_idx)(unsigned, void**);
  typedef int (*fn_u)(void*, unsigned*);
  typedef int (*fn_clk)(void*, int, unsigned*);
  typedef int (*fn_ull)(void*, unsigned long long*);
  typedef int (*fn_temp)(void*, int, unsigned*);
  typedef int (*fn_str)(void*, char*, unsigned);
  void* lib = nullptr;
  void* dev = nullptr;
  fn_u power = nullptr, limit = nullptr;
  fn_clk clock = nullptr;
  fn_ull reasons = nullptr, energy = nullptr;
  fn_temp temp = nullptr;
  fn_str vbios = nullptr, uuid = nullptr;
  bool ok = false;
  std::string err;
  bool open(int cuda_dev) {
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) { err = "dlopen libnvidia-ml.so.1"; return false; }
    auto init = (fn_init)dlsym(lib, "nvmlInit_v2");
    auto by_pci = (fn_by_pci)dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2");
    power = (fn_u)dlsym(lib, "nvmlDeviceGetPowerUsage");
    limit = (fn_u)dlsym(lib, "nvmlDeviceGetPowerManagementLimit");
    clock = (fn_clk)dlsym(lib, "nvmlDeviceGetClockInfo");
    reasons = (fn_ull)dlsym(lib, "nvmlDeviceGetCurrentClocksEventReasons");
    if (!reasons) reasons = (fn_ull)dlsym(lib, "nvmlDeviceGetCurrentClocksThrottleReasons");
    energy = (fn_ull)dlsym(lib, "nvmlDeviceGetTotalEnergyConsumption");
    temp = (fn_temp)dlsym(lib, "nvmlDeviceGetTemperature");
    vbios = (fn_str)dlsym(lib, "nvmlDeviceGetVbiosVersion");
    uuid = (fn_str)dlsym(lib, "nvmlDeviceGetUUID");
    if (!init || !by_pci || !power || !clock || !reasons || !energy || !temp) { err = "dlsym"; return false; }
    if (init() != 0) { err = "nvmlInit_v2"; return false; }
    char bus[32];
    cudaDeviceGetPCIBusId(bus, sizeof bus, cuda_dev);
    if (by_pci(bus, &dev) != 0) { err = std::string("nvmlDeviceGetHandleByPciBusId_v2 ") + bus; return false; }
    ok = true;
    return true;
  }
};

struct Sample {
  double t;  // seconds, steady clock
  double power_w;
  unsigned sm_mhz, mem_mhz, temp_c;
  unsigned long long reasons, energy_mj;
};

class Tele {
 public:
  explicit Tele(Nvml* n, double hz = 10.0) : n_(n), dt_(1.0 / hz) {}
  void start() {
    if (!n_->ok) return;
    th_ = std::thread([this] { loop(); });
  }
  void stop() {
    stop_ = true;
    if (th_.joinable()) th_.join();
  }
  ~Tele() { stop(); }
  static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
  }
  // Statistics over [t0, t1] (K0 Tele.window, same fields).
  void window(Line& l, double t0, double t1) {
    std::vector<Sample> s;
    {
      std::lock_guard<std::mutex> g(mu_);
      for (auto& x : v_)
        if (x.t >= t0 && x.t <= t1) s.push_back(x);
    }
    l.kv("tele_n", (unsigned long long)s.size());
    if (s.size() < 3) return;
    double dt = s.back().t - s.front().t, pw = 0, pmax = 0, sm = 0;
    unsigned smin = ~0u, smax = 0, tmax = 0;
    size_t cap = 0, hw = 0, therm = 0, brake = 0;
    std::vector<double> smv;
    for (auto& x : s) {
      pw += x.power_w;
      pmax = std::max(pmax, x.power_w);
      sm += x.sm_mhz;
      smv.push_back(x.sm_mhz);
      smin = std::min(smin, x.sm_mhz);
      smax = std::max(smax, x.sm_mhz);
      tmax = std::max(tmax, x.temp_c);
      cap += (x.reasons & 0x4) != 0;      // SwPowerCap
      hw += (x.reasons & 0x8) != 0;       // HwSlowdown
      therm += (x.reasons & 0x60) != 0;   // Sw/Hw thermal
      brake += (x.reasons & 0x80) != 0;   // HwPowerBrake
    }
    std::sort(smv.begin(), smv.end());
    double e_j = (double)(s.back().energy_mj - s.front().energy_mj) / 1000.0;
    l.kv("tele_hz", (s.size() - 1) / dt)
        .kv("power_w_mean", pw / s.size())
        .kv("power_w_max", pmax)
        .kv("energy_j", e_j)
        .kv("energy_dt_s", dt)
        .kv("energy_w_mean", e_j / dt)
        .kv("sm_mhz_mean", sm / s.size())
        .kv("sm_mhz_median", smv[smv.size() / 2])
        .kv("sm_mhz_min", smin)
        .kv("sm_mhz_max", smax)
        .kv("mem_mhz", s.front().mem_mhz)
        .kv("sw_power_cap_frac", (double)cap / s.size())
        .kv("hw_slowdown_n", (unsigned long long)hw)
        .kv("thermal_n", (unsigned long long)therm)
        .kv("power_brake_n", (unsigned long long)brake)
        .kv("temp_c_start", s.front().temp_c)
        .kv("temp_c_max", tmax);
  }
  double mean_energy_w(double t0, double t1) {
    std::lock_guard<std::mutex> g(mu_);
    const Sample* a = nullptr;
    const Sample* b = nullptr;
    for (auto& x : v_)
      if (x.t >= t0 && x.t <= t1) { if (!a) a = &x; b = &x; }
    if (!a || a == b) return NAN;
    return (double)(b->energy_mj - a->energy_mj) / 1000.0 / (b->t - a->t);
  }

 private:
  void loop() {
    double next = now();
    while (!stop_) {
      Sample x{};
      x.t = now();
      unsigned mw = 0;
      if (n_->power(n_->dev, &mw) == 0 && n_->clock(n_->dev, 1, &x.sm_mhz) == 0 && n_->clock(n_->dev, 2, &x.mem_mhz) == 0 &&
          n_->reasons(n_->dev, &x.reasons) == 0 && n_->temp(n_->dev, 0, &x.temp_c) == 0 && n_->energy(n_->dev, &x.energy_mj) == 0) {
        x.power_w = mw / 1000.0;
        std::lock_guard<std::mutex> g(mu_);
        v_.push_back(x);
      }
      next += dt_;
      double sl = next - now();
      if (sl > 0) std::this_thread::sleep_for(std::chrono::duration<double>(sl));
    }
  }
  Nvml* n_;
  double dt_;
  std::atomic<bool> stop_{false};
  std::thread th_;
  std::mutex mu_;
  std::vector<Sample> v_;
};

// ---------------------------------------------------------------- device open + meta line
struct Ctx {
  const char* probe;
  Args a;
  cudaDeviceProp prop{};
  Nvml nvml;
  Tele* tele = nullptr;
  double secs = 24.0, discard = 2.0;
  int rounds = 2;
  cudaStream_t s = nullptr;
};

inline void open_device(Ctx& c) {
  P6_CK(c.probe, cudaSetDevice(0));
  P6_CK(c.probe, cudaFree(nullptr));
  P6_CK(c.probe, cudaGetDeviceProperties(&c.prop, 0));
  P6_CK(c.probe, cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking));
  c.secs = c.a.num("--secs", 24.0);
  c.discard = c.a.num("--discard", 2.0);
  c.rounds = (int)c.a.num("--rounds", 2);
  bool nv = c.nvml.open(0);
  int drv = 0, rtv = 0;
  cudaDriverGetVersion(&drv);
  cudaRuntimeGetVersion(&rtv);
  char uuid[96] = "", vb[96] = "";
  unsigned lim = 0;
  if (nv) {
    if (c.nvml.uuid) c.nvml.uuid(c.nvml.dev, uuid, sizeof uuid);
    if (c.nvml.vbios) c.nvml.vbios(c.nvml.dev, vb, sizeof vb);
    if (c.nvml.limit) c.nvml.limit(c.nvml.dev, &lim);
  }
  char bus[32];
  cudaDeviceGetPCIBusId(bus, sizeof bus, 0);
  const char* tag = std::getenv("P6_TAG");
  const char* vbt = std::getenv("P6_VBIOS");
  Line l("meta", c.probe);
  l.kv("kind", "meta").kv("device", c.prop.name).kv("uuid", uuid).kv("vbios", vb).kv("pci", bus)
      .kv("power_limit_w", lim / 1000.0).kv("nvml_ok", nv).kv("nvml_err", c.nvml.err)
      .kv("cc", std::to_string(c.prop.major) + "." + std::to_string(c.prop.minor))
      .kv("sms", c.prop.multiProcessorCount).kv("l2_bytes", c.prop.l2CacheSize)
      .kv("smem_optin", (unsigned long long)c.prop.sharedMemPerBlockOptin)
      .kv("cuda_driver_api", drv).kv("cuda_runtime", rtv).kv("compiled_cudart", CUDART_VERSION)
      .kv("tag", tag ? tag : "").kv("vbios_type", vbt ? vbt : "")
      .kv("secs", c.secs).kv("discard_s", c.discard).kv("rounds", c.rounds)
      .kv("host_unix_s", (long long)std::time(nullptr));
  std::string argv;
  for (int i = 1; i < c.a.argc; ++i) { argv += c.a.argv[i]; argv += ' '; }
  l.kv("argv", argv);
  l.emit();
  c.tele = new Tele(&c.nvml, c.a.num("--hz", 10.0));
  c.tele->start();
}

// ---------------------------------------------------------------- sustained timing (k0bench.sustained, in C++)
// launch(stream) enqueues ONE call. units_per_call = bytes (or any work unit) of one call, for the rate fields.
struct Result {
  double ms_per_call_median = NAN, ms_per_call_mean = NAN, rate_span = NAN, rate_busy = NAN, gap_frac = NAN;
  double energy_w = NAN, j_per_unit = NAN, j_per_call = NAN;
  size_t chunks = 0;
};

inline Result sustained(Ctx& c, Line& l, const std::function<void(cudaStream_t)>& launch, double units_per_call,
                        int calls_per_chunk, const char* unit = "GB") {
  Result r;
  struct Ev { cudaEvent_t a, b; };
  std::vector<Ev> evs;
  evs.reserve(4096);
  P6_CK(c.probe, cudaStreamSynchronize(c.s));
  double t0 = Tele::now();
  while (Tele::now() - t0 < c.secs) {
    Ev e;
    cudaEventCreate(&e.a);
    cudaEventCreate(&e.b);
    cudaEventRecord(e.a, c.s);
    for (int i = 0; i < calls_per_chunk; ++i) launch(c.s);
    cudaEventRecord(e.b, c.s);
    evs.push_back(e);
    if (evs.size() > 3) cudaEventSynchronize(evs[evs.size() - 4].b);
  }
  cudaError_t ke = cudaStreamSynchronize(c.s);
  double t1 = Tele::now();
  if (ke != cudaSuccess) { l.rt(ke); return r; }
  std::vector<double> starts, durs;
  for (auto& e : evs) {
    float st = 0, du = 0;
    cudaEventElapsedTime(&st, evs[0].a, e.a);
    cudaEventElapsedTime(&du, e.a, e.b);
    starts.push_back(st);
    durs.push_back(du);
  }
  std::vector<size_t> keep;
  for (size_t i = 0; i < starts.size(); ++i)
    if (starts[i] >= c.discard * 1000.0) keep.push_back(i);
  if (keep.size() < 3) { l.kv("ok", false).kv("err", "too few chunks kept"); return r; }
  double span = starts[keep.back()] + durs[keep.back()] - starts[keep.front()], busy = 0;
  std::vector<double> kd;
  for (size_t i : keep) { busy += durs[i]; kd.push_back(durs[i] / calls_per_chunk); }
  std::sort(kd.begin(), kd.end());
  double calls = (double)calls_per_chunk * keep.size();
  r.chunks = keep.size();
  r.ms_per_call_median = kd[kd.size() / 2];
  r.ms_per_call_mean = busy / calls;
  r.rate_span = units_per_call * calls / (span * 1e-3);
  r.rate_busy = units_per_call * calls / (busy * 1e-3);
  r.gap_frac = 1.0 - busy / span;
  // drift: first kept 5 s vs last 5 s
  auto rate = [&](bool first) {
    double b = 0; size_t n = 0;
    for (size_t i : keep) {
      bool in = first ? starts[i] < (c.discard + 5) * 1000.0 : starts[i] >= starts[keep.back()] - 5000.0;
      if (in) { b += durs[i]; ++n; }
    }
    return n ? units_per_call * calls_per_chunk * n / (b * 1e-3) : NAN;
  };
  l.kv("ok", true).kv("chunks_kept", (unsigned long long)keep.size()).kv("calls_per_chunk", calls_per_chunk)
      .kv("unit", unit).kv("units_per_call", units_per_call)
      .kv("ms_per_call_median", r.ms_per_call_median).kv("ms_per_call_mean", r.ms_per_call_mean)
      .kv("rate_span", r.rate_span).kv("rate_busy", r.rate_busy).kv("gap_frac", r.gap_frac)
      .kv("rate_first5s", rate(true)).kv("rate_last5s", rate(false));
  double ta = t1 - (span + (starts[keep.front()])) * 1e-3;  // wall time of the first kept chunk (approx.)
  (void)ta;
  double w0 = t0 + c.discard, w1 = t1;
  c.tele->window(l, w0, w1);
  r.energy_w = c.tele->mean_energy_w(w0, w1);
  if (std::isfinite(r.energy_w)) {
    r.j_per_unit = r.energy_w / r.rate_span;
    r.j_per_call = r.energy_w * r.ms_per_call_mean * 1e-3 * (span / busy);
    l.kv("j_per_unit_energy", r.j_per_unit).kv("j_per_call_energy", r.j_per_call);
  }
  for (auto& e : evs) { cudaEventDestroy(e.a); cudaEventDestroy(e.b); }
  return r;
}

// Sandwich A/B order: per round, baseline first and last, arms in between, reversed in odd rounds.
// Returns a list of (round, position, arm index).
struct Slot { int round, pos, arm; bool sandwich_tail; };
inline std::vector<Slot> sandwich(int n_arms, int rounds, int baseline = 0) {
  std::vector<Slot> out;
  for (int r = 0; r < rounds; ++r) {
    std::vector<int> mid;
    for (int i = 0; i < n_arms; ++i)
      if (i != baseline) mid.push_back(i);
    if (r & 1) std::reverse(mid.begin(), mid.end());
    int p = 0;
    out.push_back({r, p++, baseline, false});
    for (int i : mid) out.push_back({r, p++, i, false});
    if (n_arms > 1) out.push_back({r, p++, baseline, true});
  }
  return out;
}

// ---------------------------------------------------------------- small device helpers shared by the probes
__host__ __device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
// uniform in (0, 1)
__host__ __device__ inline float u01(uint64_t h) { return ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f); }

// Host bf16/f16 helpers (bit-exact round-to-nearest-even)
inline uint16_t f32_to_bf16_rne(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
inline float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline double f16_to_f64(uint16_t h) {
  int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  double v = e == 0 ? std::ldexp((double)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp((double)(m | 1024), e - 25);
  return s ? -v : v;
}
// exact value of an E4M3 (fn) code; 0x7f/0xff are NaN in the format
inline double e4m3_to_f64(uint8_t c) {
  int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
  if (e == 15 && m == 7) return NAN;
  double v = e == 0 ? std::ldexp((double)m, -9) : std::ldexp((double)(8 + m), e - 10);
  return s ? -v : v;
}
// round to E4M3 nearest-even, saturating to +-448 (torch clamp(-448, 448).to(float8_e4m3fn))
inline uint8_t f64_to_e4m3_sat(double x) {
  if (std::isnan(x)) return 0x7f;
  uint8_t sign = x < 0 ? 0x80 : 0;
  double a = std::fabs(x);
  if (a >= 448.0) return sign | 0x7e;
  int best = 0;
  double bd = INFINITY;
  for (int c = 0; c < 0x7f; ++c) {  // all finite non-negative codes, monotone
    double d = std::fabs(e4m3_to_f64((uint8_t)c) - a);
    if (d < bd || (d == bd && (c & 1) == 0)) { bd = d; best = c; }
  }
  return sign | (uint8_t)best;
}

}  // namespace p6
