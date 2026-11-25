// inkling-cmp-engine native runtime: the drafter's round batch (own code). A round drafts only the sessions that
// speculate in it: draft_gather copies their state (anchor, P, block-table row, f_rnn) into compact rows 0..k-1, the
// layers and the head run on those k rows (graph per k), draft_scatter writes the k rows' outputs back to the
// sessions' rows; the other sessions' drafts, survivors and conf are left untouched. Every kernel of the draft
// computes each row from that row's inputs only (GEMMs per element, attention per session with the drafter's fixed
// split count), so a session's drafts do not depend on which other sessions draft with it.
#pragma once

namespace s2f {

// grid k (one CTA per drafted session), block 256
__global__ void draft_gather(const int* sel, const int* anchor, const int* P, const int* bt, int max_blocks,
                             const __nv_bfloat16* f_rnn, int* anchor_c, int* P_c, int* bt_c, __nv_bfloat16* f_rnn_c) {
  const int i = blockIdx.x, r = sel[i];
  if (threadIdx.x == 0) {
    anchor_c[i] = anchor[r];
    P_c[i] = P[r];
  }
  for (int j = threadIdx.x; j < max_blocks; j += blockDim.x) bt_c[(size_t)i * max_blocks + j] = bt[(size_t)r * max_blocks + j];
  for (int h = threadIdx.x; h < H; h += blockDim.x) f_rnn_c[(size_t)i * H + h] = f_rnn[(size_t)r * H + h];
}

// grid 1, block 64 (7 k <= 56 threads used): drafts / conf [n][7], survivors [7][n] from the compact [k] rows
__global__ void draft_scatter(const int* sel, int k, int n, const int* drafts_c, const int* nsurv_c,
                              const float* conf_c, int* drafts, int* nsurv, float* conf) {
  const int e = threadIdx.x;
  if (e >= 7 * k) return;
  const int i = e / 7, j = e % 7, r = sel[i];
  drafts[r * 7 + j] = drafts_c[e];
  conf[r * 7 + j] = conf_c[e];
  nsurv[j * n + r] = nsurv_c[j * k + i];
}

}  // namespace s2f
