// inkling-cmp-engine native runtime: device sampling parameters (own code). Kernel: sample_kernels.cuh; the definition
// of the distribution it samples and the speculative rule are in bench/raw/2026-11-23-spec-sampling/README.md.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

namespace ink {

// Sampling controls of one segment as the kernel reads them (sample_param() builds them from the request's values).
struct SampleParam {
  unsigned long long seed;   // Philox4x32-10 key
  unsigned long long wmin;   // min-p as the smallest kept weight (weights: floor(e * 2^40), e = exp((x - max) / T))
  float temperature;
  unsigned int p24;          // top-p as round(top_p * 2^24); 2^24: off
  int top_k;                 // 0: off
  int greedy;                // 1: the first index of the largest logit (temperature below 1e-5, as vLLM)
};

// Pointers of one sample_rows_kernel launch (one block per sampled row).
struct SampleArgs {
  const float* lg;           // logit rows [rows][ld], the first `vocab` valid
  int ld, vocab;
  const SampleParam* par;
  // engine mapping (lrow != nullptr): row r = logit row r of the last call, call row c = lrow[r], parameters
  // par[call_seg[c]], sampled position call_pos[c] + 1
  const int *lrow, *call_pos, *call_seg;
  // test mapping (lrow == nullptr): row r reads logit row t_lrow[r], parameters par[t_par[r]], position t_pos[r]
  const int *t_lrow, *t_par, *t_pos;
  int* out;                  // [rows] sampled token ids
  unsigned long long* stats; // optional [rows][8]: max | argmax << 32, threshold key, W, kept, hash, R, token, flags
  unsigned int* keys;        // optional [rows][vocab]: each entry's key (float bits of e) if it is a candidate, else 0
};

// Pointers of one sample_leviathan_kernel launch (exact Leviathan rejection + residual sampling over a sparse
// top-16 drafter proposal distribution with integer weights summing to 2^22).
struct LeviathanArgs {
  const float* lg;            // logit rows [rows][ld], the first `vocab` valid
  int ld, vocab;
  const SampleParam* par;     // [nseg] sampling controls
  const int* lrow;            // [rows] call row index per logit row
  const int* call_pos;        // [T] position of each call row
  const int* call_seg;        // [T] segment index of each call row
  const int* call_row0;       // [nseg] first call row of each segment
  const int* call_slot;       // [nseg] engine slot of each segment
  int nseg, T;
  const int* drafts;          // [max_slots][7] drafted tokens
  const int* prop_ids;        // [max_slots][7][16] sparse proposal token ids
  const unsigned int* prop_w; // [max_slots][7][16] proposal weights summing to 2^22
  int* out;                   // [rows] accepted draft token or residual/plain sample
};

// sample_rows_kernel over `rows` blocks on stream st (engine.cu)
void launch_sample_rows(const SampleArgs& a, int rows, cudaStream_t st);
void launch_sample_leviathan(const LeviathanArgs& a, int rows, cudaStream_t st);

constexpr unsigned kSampFlagGreedy = 1, kSampFlagOverflow = 2, kSampFlagLump = 4, kSampFlagMismatch = 8;

inline SampleParam sample_param(uint64_t seed, float temperature, float top_p, float min_p, int top_k) {
  SampleParam p{};
  p.seed = seed;
  p.temperature = temperature;
  p.greedy = !(temperature >= 1e-5f);   // NaN too
  const double tp = top_p;
  p.p24 = !(tp < 1.0) ? (1u << 24) : tp <= 0.0 ? 0u : (unsigned)std::llround(tp * 16777216.0);
  const double mp = !(min_p > 0.0f) ? 0.0 : min_p >= 1.0f ? 1.0 : (double)min_p;
  p.wmin = std::max<unsigned long long>(1ull, (unsigned long long)std::ceil(mp * 1099511627776.0));
  p.top_k = top_k > 0 ? top_k : 0;
  return p;
}

}  // namespace ink
