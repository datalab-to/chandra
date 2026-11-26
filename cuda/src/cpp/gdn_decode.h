// One Gated DeltaNet decode step for one layer of Chandra 2 (16 key heads, 32 value
// heads, head dims 128, causal conv1d of width 4 on the 8,192 q/k/v channels).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace chandra {

constexpr int kGdnKeyHeads = 16;
constexpr int kGdnValueHeads = 32;
constexpr int kGdnDim = 128;
constexpr int kGdnChannels = 8192;  // q 2048 | k 2048 | v 4096
constexpr int kGdnConv = 4;

// Floats of replay history per (sequence slot, token, value head): k (128), v' (128), decay, pad.
constexpr int kGdnHistFloats = 260;

struct GdnDecodeArgs {
  // Token rows: sequence n, token t (0 <= t < T) is row n * T + t of qkv, z, b, a and out.
  const void* qkv;  // [B * T][>= 8192] BF16, in_proj_qkv output before the conv (row stride qkv_stride)
  int64_t qkv_stride;
  const void* z;  // [B][>= 4096] BF16 gate of the norm
  int64_t z_stride;
  const void* b;  // [B][32] BF16 beta logits (row stride ab_stride)
  const void* a;  // [B][32] BF16 decay logits
  int64_t ab_stride;
  const float* A_log;     // [32]
  const void* dt_bias;    // [32] BF16
  const void* conv_w;     // [8192][4] BF16
  const void* norm_w;     // [128] BF16
  void* state;            // [slots][32][128][128], FP32 or FP16 (rows: v, columns: k)
  bool state_fp16;
  const int* slot_in;     // [B] state slot to read
  const int* slot_out;    // [B] state slot to write (may equal slot_in)
  void* conv_ring;        // [slots][ring][8192] BF16: raw x of earlier positions, row = position % ring
  int ring;               // >= 3 + T
  const int* conv_slot;   // [B] ring (and history) slot of each sequence
  const int* pos;         // [B] tokens of the sequence before this step's first token
  void* out;              // [B * T][4096] BF16: RMSNorm(o) * w * SiLU(z), the out_proj input
  int64_t out_stride;
  int B;
  int T;  // tokens per sequence in this step: 1 (decode) to 4 (verify of 3 draft tokens)
  // Speculative verification with one state read and one write per step. Before this step's
  // tokens, n_replay[n] updates S = decay_i S + v'_i k_i^T are replayed from hist_in (the
  // previous step's history): the state after the tokens that were accepted, bit for bit, when
  // slot_in holds the state before them. hist_out gets this step's history. Either may be null.
  const int* n_replay;  // [B]
  const float* hist_in;   // [slots][hist_tmax][32][kGdnHistFloats]
  float* hist_out;        // same layout, a different buffer
  int hist_tmax;
};

cudaError_t gdn_decode(const GdnDecodeArgs& a, cudaStream_t stream);

}  // namespace chandra
