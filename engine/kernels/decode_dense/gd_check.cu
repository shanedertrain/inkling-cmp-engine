// gd_check (own code): which (call rows T, group rows m, batch) cuBLASLt's AlgoCheck accepts for the decode gate-down
// shape (n 64, k 4608) on a pinned algorithm, with the engine's strides (x: T * 4608, out: T * 64, w: --wstride).
//   gd_check --pin ID,TILE,STAGES,SWIZZLE,SPLITK,RED [--wstride N] [--tmax 64]
// One line per refused (T, m, batch) and a summary; no kernel runs.
#include <cublasLt.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/common.hpp"
#include "../../native/src/gemm_pins.hpp"

int main(int argc, char** argv) {
  std::vector<long long> pin;
  long long wstride = 64LL * 4608 + 4096;
  int tmax = 64;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--pin") {
      std::string s = argv[i + 1];
      size_t p = 0;
      while (p <= s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        pin.push_back(std::atoll(s.substr(p, q - p).c_str()));
        p = q + 1;
      }
    } else if (a == "--wstride") wstride = std::atoll(argv[i + 1]);
    else if (a == "--tmax") tmax = std::atoi(argv[i + 1]);
  }
  INK_REQUIRE(pin.size() == 6, "--pin ID,TILE,STAGES,SWIZZLE,SPLITK,RED");
  cublasLtHandle_t h;
  INK_REQUIRE(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS, "create");
  ink::gemm_pins::Shape s;
  s.n = 64, s.k = 4608, s.ldw = 4608;
  ink::gemm_pins::AlgoPin ap;
  ap.id = (int)pin[0], ap.tile = (uint32_t)pin[1], ap.stages = (uint32_t)pin[2], ap.swizzle = (uint32_t)pin[3];
  ap.splitk = (int)pin[4], ap.reduction = (uint32_t)pin[5];
  cublasLtMatmulAlgo_t algo = ink::gemm_pins::build_algo(h, s, ap);
  cublasLtMatmulDesc_t op;
  cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
  auto lay = [](int rows, int cols, int ld, int count, long long stride) {
    cublasLtMatrixLayout_t l;
    cublasLtMatrixLayoutCreate(&l, CUDA_R_16BF, rows, cols, ld);
    if (count > 1 || stride) {
      cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof count);
      cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof stride);
    }
    return l;
  };
  int refused = 0, total = 0;
  for (int batch : {1, 3}) {
    for (int T = 1; T <= tmax; ++T)
      for (int m = 1; m <= T; ++m) {
        const bool strided = batch > 1;
        cublasLtMatrixLayout_t a = strided ? lay(4608, 64, 4608, batch, wstride) : lay(4608, 64, 4608, 1, 0);
        cublasLtMatrixLayout_t b = strided ? lay(4608, m, 4608, batch, (long long)T * 4608) : lay(4608, m, 4608, 1, 0);
        cublasLtMatrixLayout_t c = strided ? lay(64, m, 64, batch, (long long)T * 64) : lay(64, m, 64, 1, 0);
        cublasLtMatmulHeuristicResult_t r{};
        const cublasStatus_t st = cublasLtMatmulAlgoCheck(h, op, a, b, c, c, &algo, &r);
        ++total;
        if (st != CUBLAS_STATUS_SUCCESS) {
          ++refused;
          std::printf("refused batch %d T %d m %d status %d\n", batch, T, m, (int)st);
        } else if (m == T && (T == 1 || T == 4 || T == 8 || T == 32 || T == 64))
          std::printf("ok batch %d T %d m %d workspace %zu\n", batch, T, m, r.workspaceSize);
        cublasLtMatrixLayoutDestroy(a);
        cublasLtMatrixLayoutDestroy(b);
        cublasLtMatrixLayoutDestroy(c);
      }
  }
  std::printf("summary: %d of %d refused\n", refused, total);
  return 0;
}
