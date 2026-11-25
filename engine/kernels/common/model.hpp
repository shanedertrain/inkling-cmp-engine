// inkling-cmp-engine native runtime: Inkling-2-Mini dimensions and the device weight set (own code; dimensions from
// engine/ref/config.py, tensor names from engine/ref/weights.py).
#pragma once
#include <array>
#include <string>
#include <vector>

#include "common.hpp"

namespace ink {

constexpr int H = 2304, NH = 24, KVH = 8, HD = 128, DREL = 16, REL_EXT = 1024, WINDOW = 1024, INTER = 1152;
constexpr int NEXP = 128, TOPK = 8, CONVK = 4, NLAYERS = 24, BLOCK = 6, LOOP_START = 6, LOOP_END = 18, PASSES = 3;
constexpr int VOCAB_PAD = 201024, VOCAB = 200058, GATE_RANK = 64;
constexpr int QD = NH * HD, KD = KVH * HD;  // 3072, 1024
constexpr int NVISITS = LOOP_START + PASSES * (LOOP_END - LOOP_START) + (NLAYERS - LOOP_END);  // 48
constexpr float EPS = 1e-6f, QK_CLIP = 22.627416997969522f, REL_CLIP = 8.0f, LOG_ALPHA = 0.1f, LOG_FLOOR = 8192.0f;
constexpr float LOGIT_SOFTCAP = 30.0f, ROUTE_SCALE = 8.0f;
inline bool is_full_layer(int l) { return l == 5 || l == 11 || l == 17 || l == 23; }

struct Visit {
  int layer, pass;
};
inline std::vector<Visit> visit_plan() {
  std::vector<Visit> v;
  for (int l = 0; l < LOOP_START; ++l) v.push_back({l, 0});
  for (int p = 0; p < PASSES; ++p)
    for (int l = LOOP_START; l < LOOP_END; ++l) v.push_back({l, p});
  for (int l = LOOP_END; l < NLAYERS; ++l) v.push_back({l, 0});
  return v;
}

enum ExpertFmt { EXP_NVFP4 = 0, EXP_MXFP8 = 1 };

struct LayerW {
  int idx = 0;
  bool full = false, loop = false;
  // small parameters as fp32 (bf16 in the checkpoint unless noted; exact)
  float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr, *mlp_norm = nullptr;
  float *q_scale = nullptr, *r_scale = nullptr;            // [24, 128], [24, 16]
  bf16* rel_proj = nullptr;                                 // [16, 1024] bf16
  float *k_static = nullptr, *v_static = nullptr;          // full: [4, 1024] by lag
  float *k_conv = nullptr, *v_conv = nullptr;              // local: [1024, 4]
  float *attn_conv = nullptr, *mlp_conv = nullptr;         // [2304, 4]
  bf16* router = nullptr;                                   // [128, 2304] bf16
  float* router_bias = nullptr;                             // [128] (F32 in the checkpoint)
  float* attn_query = nullptr;                              // [1 or 3, 2304] (F32), layer > 0
  int n_attn_query = 0;
  float* mlp_query = nullptr;                               // [2304] (F32)
  // dense bf16 [out, in]
  bf16 *wq = nullptr, *wk = nullptr, *wv = nullptr, *wr = nullptr, *wo = nullptr, *wgate = nullptr, *wsink = nullptr;
  bf16 *wk_dyn = nullptr, *wv_dyn = nullptr;                // full: [32, 2304]
  // loop controllers, order q, k, v, mlp: down bf16 [64, 4608]; forget / gain fp32 [2304, 64]
  std::array<bf16*, 4> c_down{};
  std::array<float*, 4> c_forget{}, c_gain{};
  std::array<float*, 4> c_forgetT{}, c_gainT{};           // [64, 2304] (glue v2: coalesced fused gate)
  // experts (prefill_moe layout, see weights.cpp): per projection (0 = w13, 1 = w2)
  int fmt = EXP_NVFP4;
  std::array<uint8_t*, 2> eW{}, eS{};
  std::array<float*, 2> eF{};
};

struct Weights {
  std::vector<LayerW> layers;
  bf16* embed = nullptr;           // [201024, 2304]
  float* embed_norm = nullptr;
  float* final_query = nullptr;    // [2304]
  float* final_norm = nullptr;
  bf16* unembed = nullptr;         // [201024, 2304] (padded rows)
  float* msg_queries = nullptr;    // [12, 4, 2304]: q, k, v, mlp (engine/ref/weights.py order)
  size_t bytes = 0;
};

class Arena;
void load_weights(Weights& w, Arena& ar, const std::string& model_dir, int threads);
size_t weights_bytes_estimate();

}  // namespace ink
