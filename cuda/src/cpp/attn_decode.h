// Decode / verify attention for Chandra 2's full-attention layers: 16 query heads,
// 4 KV heads (GQA 4:1), head_dim 256, paged BF16 KV cache, sigmoid output gate.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace chandra {

constexpr int kAttnHeads = 16;
constexpr int kAttnKvHeads = 4;
constexpr int kAttnHeadDim = 256;
constexpr int kAttnPageTokens = 64;  // KV cache page: [page][kv_head][64][256]

struct AttnDecodeArgs {
  // Queries, gates and outputs: token i = b * n_q + t; head h at column offset h * head_stride.
  const void* q;
  int64_t q_token_stride;  // elements
  int q_head_stride;
  const void* gate;
  int64_t gate_token_stride;
  int gate_head_stride;
  void* out;
  int64_t out_token_stride;
  int out_head_stride;
  const void* k_cache;  // [pages][4][64][256] BF16
  const void* v_cache;
  const int* block_table;  // [B][block_table_stride] page ids
  int block_table_stride;
  const int* ctx_lens;  // [B] tokens in the cache, including this step's n_q
  float* partial_o;     // [B][4][max_splits][16][256]
  float* partial_ml;    // [B][4][max_splits][16][2]
  unsigned* tickets;    // [B][4], zero before the first call, left at zero
  int B;
  int n_q;               // 1 (decode) to 4 (verify of 3 draft tokens)
  int stages_per_split;  // 32-key stages per CTA
  int max_splits;
};

// Splits for a batch whose longest context is max_ctx, on `sms` SMs.
void attn_decode_split(int B, int max_ctx, int sms, int* stages_per_split, int* max_splits);

cudaError_t attn_decode(const AttnDecodeArgs& a, cudaStream_t stream);

}  // namespace chandra
