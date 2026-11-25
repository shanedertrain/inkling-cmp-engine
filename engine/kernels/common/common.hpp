// inkling-cmp-engine native runtime: common helpers (own code).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ink {

[[noreturn]] inline void fatal(const std::string& m) {
  std::fprintf(stderr, "FATAL: %s\n", m.c_str());
  std::fflush(stderr);
  std::exit(2);
}
#define INK_REQUIRE(c, m)                                                                            \
  do {                                                                                               \
    if (!(c)) ::ink::fatal(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + (m));     \
  } while (0)
#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess)                                                                           \
      ::ink::fatal(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " #x ": " + cudaGetErrorString(e_)); \
  } while (0)
using bf16 = __nv_bfloat16;

inline float bf2f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f2bf(float f) {  // round to nearest even (finite inputs)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

// One device allocation at start (PLAN 2.1: never free GBs at run time). Bump allocator, 256-byte aligned.
class Arena {
 public:
  void init(size_t bytes) {
    CK(cudaMalloc(&base_, bytes));
    CK(cudaMemset(base_, 0, bytes));
    cap_ = bytes;
    off_ = 0;
  }
  void* take(size_t bytes, const char* what) {
    size_t a = (off_ + 255) & ~size_t(255);
    if (a + bytes > cap_)
      fatal(std::string("arena exhausted at ") + what + ": need " + std::to_string(a + bytes) + " of " + std::to_string(cap_));
    off_ = a + bytes;
    takes_.push_back({what, a, bytes});
    return (char*)base_ + a;
  }
  template <class T>
  T* get(size_t n, const char* what) { return reinterpret_cast<T*>(take(n * sizeof(T), what)); }
  size_t used() const { return off_; }
  size_t cap() const { return cap_; }
  void* base() const { return base_; }
  // every take() in order: name, aligned offset, requested bytes (before the next take's alignment); the arena map of
  // the oracle record set (engine/native/capi/README.md, E1)
  struct Take {
    std::string what;
    size_t off, bytes;
  };
  const std::vector<Take>& takes() const { return takes_; }

 private:
  void* base_ = nullptr;
  size_t cap_ = 0, off_ = 0;
  std::vector<Take> takes_;
};

}  // namespace ink
