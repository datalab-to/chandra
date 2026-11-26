// Non-causal attention for the vision tower: 16 heads, head_dim 64, whole images
// (tens of thousands of patches), several images per call (cu_seqlens).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace chandra {

struct VitAttnArgs {
  // Row i of q, k, v, out is patch i of the packed images; head h at column h * 64.
  const void* q;
  const void* k;
  const void* v;
  int64_t qkv_stride;  // elements between rows (3,072 for a fused [q | k | v] GEMM output)
  void* out;
  int64_t out_stride;
  const int* cu_seqlens;  // [n_seq + 1] row offsets of the images
  int n_seq;
  int max_len;
};

cudaError_t vit_attn(const VitAttnArgs& a, cudaStream_t stream);

}  // namespace chandra
