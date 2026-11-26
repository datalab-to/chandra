// Weight-streaming BF16 GEMM for decode steps: y[M][N] = x[M][K] * W[N][K]^T, M <= 64.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace chandra {

// The K split for an (N, K) weight on a GPU with `sms` SMs, and the number of
// rows (16 * warps) each CTA owns. Depends only on the weight's shape and the
// GPU, never on M, so a token's output is the same at every batch size.
struct DecodeGemmPlan {
  int row_groups;  // 16-row groups per CTA; its 8 warps are row_groups x (8 / row_groups) K slices
  int splits;      // CTAs along K per row tile (1 = no partial sums in global memory)
};
DecodeGemmPlan decode_gemm_plan(int N, int K, int sms);

// Bytes of FP32 workspace and number of uint32 counters `decode_gemm` needs.
size_t decode_gemm_workspace_bytes(int N, int K, int sms);
int decode_gemm_counters(int N, int K, int sms);

// x: [M][K] BF16 row-major. wp: W packed by `pack_weight` (see the Python side).
// y: [M][N] BF16 row-major. workspace/counters: from the sizes above; counters
// must be zero before the first call and are left at zero after each call.
cudaError_t decode_gemm(const void* x, const void* wp, void* y, float* workspace,
                        unsigned* counters, int M, int N, int K, int sms,
                        cudaStream_t stream);

}  // namespace chandra
