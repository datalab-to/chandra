// The small per-step ops of Chandra 2's language model around the GEMMs.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace chandra {

constexpr int kHidden = 2560;
constexpr int kMlp = 9216;

// residual_out = bf16(x + residual); out = bf16(rmsnorm(residual_out) * (1 + w)), eps 1e-6.
// x may be null (then residual_out = residual). Rows of kHidden, contiguous.
cudaError_t rmsnorm_add(const void* x, const void* residual, void* residual_out, void* out, const void* w, int M,
                        cudaStream_t stream);

// out[m][j] = bf16(silu(gu[m][j]) * gu[m][kMlp + j]) for gate_up rows [M][2 * kMlp].
cudaError_t silu_mul(const void* gate_up, void* out, int M, cudaStream_t stream);

// Full-attention layer, after the merged qkv GEMM: row m of qkv is
// [16 heads x (q 256 | gate 256)] [k 4 x 256] [v 4 x 256] (10,240 BF16).
// q heads: RMSNorm (1 + q_norm_w) and partial interleaved mRoPE (first 64 dims), into
// q_out [M][16][256]. k heads: RMSNorm (1 + k_norm_w) and the same RoPE; k and v are
// written into the paged cache [pages][4][64][256] at (page[m], slot[m]).
// pos: [M][3] int32 (t, h, w) positions; frequency f of the rotary half uses axis f % 3.
struct QkNormRopeArgs {
  const void* qkv;
  int64_t qkv_stride;
  const void* q_norm_w;
  const void* k_norm_w;
  const int* pos;
  float rope_theta;
  void* q_out;
  void* k_cache;
  void* v_cache;
  const int* page;
  const int* slot;
  int M;
  float* k_scale;  // INT8 cache: [pages][4][64] per (token, head) scales (absmax / 127); null for BF16
  float* v_scale;
};
cudaError_t qk_norm_rope_kv(const QkNormRopeArgs& a, cudaStream_t stream);

}  // namespace chandra
