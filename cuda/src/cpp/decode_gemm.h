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
  int splits;      // CTAs along K per row tile (1 = no partial sums in global memory); negative:
                   // stream-K with -splits CTAs sharing all (tile, K chunk) units equally
};
DecodeGemmPlan decode_gemm_plan(int N, int K, int sms);
// For measurements: force the plan of (N, K) (rw <= 0 removes the override). Not thread-safe.
void set_decode_gemm_plan(int N, int K, int rw, int splits);

// Bytes of FP32 workspace and number of uint32 counters `decode_gemm` needs.
size_t decode_gemm_workspace_bytes(int N, int K, int sms);
int decode_gemm_counters(int N, int K, int sms);

// x: [M][K] BF16 row-major. wp: W packed by `pack_weight` (see the Python side). epilogue 0:
// y = x W^T; 1: W's rows are interleave_gate_up(gate; up) and y = silu(gate) * up, [M][N / 2].
// y: [M][N] BF16 row-major. workspace/counters: from the sizes above; counters
// must be zero before the first call and are left at zero after each call.
cudaError_t decode_gemm(const void* x, const void* wp, void* y, float* workspace,
                        unsigned* counters, int M, int N, int K, int sms,
                        cudaStream_t stream, int epilogue = 0);

// The same GEMM with FP8 e4m3 weights (packed by pack_weight_fp8) and one FP32 scale per
// output row: y = x * (scale[n] * W8[n])^T. Same plan, workspace and counters.
cudaError_t decode_gemm_fp8(const void* x, const void* wp, const float* wscale, void* y, float* workspace,
                            unsigned* counters, int M, int N, int K, int sms, cudaStream_t stream, int epilogue = 0);

// The same GEMM with MXFP4 weights (OCP MX: e2m1 codes, one E8M0 scale per 32 K), packed by
// pack_weight_fp4: 4.25 bits per weight. N % 64 == 0. Same plan, workspace and counters.
cudaError_t decode_gemm_fp4(const void* x, const void* wp, const void* scales, void* y, float* workspace,
                            unsigned* counters, int M, int N, int K, int sms, cudaStream_t stream, int epilogue = 0);

// The same GEMM with INT4 weights, one BF16 scale per 128 along K (symmetric, stored as q + 8),
// packed by pack_weight_int4: the W4A16 g128 format. N % 64 == 0, K % 128 == 0.
cudaError_t decode_gemm_int4(const void* x, const void* wp, const void* scales, void* y, float* workspace,
                             unsigned* counters, int M, int N, int K, int sms, cudaStream_t stream, int epilogue = 0);

// The exact LM head from INT8 weights. lm_head_bound_i8: INT8 sign-magnitude weights in the
// FP8 layout with per-row scales; xnorm [M][K / 256] and rowc [N][K / 256] are per 256-wide K
// block (the bound is sum_b ||x_b|| c_n,b); per 128-row tile and token an upper bound of the BF16 head's
// logits into tile_ub [N / 128][64], and the best lower bound per token into lb_key [M]
// (zeroed first; order-preserving float bits). lm_head_recheck: the BF16 argmax kernel on the
// tiles whose bound does not exclude them; keys as decode_gemm_argmax. Together: the same pick
// as decode_gemm_argmax, reading ~half the bytes.
cudaError_t lm_head_bound_i8(const void* x, const void* wp, const float* wscale, const float* rowc,
                             const float* xnorm, float* tile_ub, unsigned* lb_key, int M, int N, int K,
                             cudaStream_t stream);
cudaError_t lm_head_recheck(const void* x, const void* wp, const float* tile_ub, const unsigned* lb_key,
                            unsigned long long* keys, int M, int N, int K, cudaStream_t stream);

// LM head with the greedy pick fused: for each token m, keys[m] (zeroed before the call)
// becomes the max over rows n of key(bf16(y[m][n]), n); the picked row is
// 0xffffffff - (keys[m] & 0xffffffff). Ties go to the lowest row. y is never written.
cudaError_t decode_gemm_argmax(const void* x, const void* wp, unsigned long long* keys, int M, int N, int K,
                               int sms, cudaStream_t stream);

// The same with FP8 e4m3 weights and per-row scales (decode_gemm_fp8's layout): half the
// bytes of an LM head, for draft tokens that a verify step checks anyway.
cudaError_t decode_gemm_argmax_fp8(const void* x, const void* wp, const float* wscale, unsigned long long* keys,
                                   int M, int N, int K, cudaStream_t stream);

}  // namespace chandra
