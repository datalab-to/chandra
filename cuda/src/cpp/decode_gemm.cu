// Weight-streaming BF16 GEMM for decode steps: y[M][N] = x[M][K] * W[N][K]^T, M <= 64.
//
// Decode is bound by reading the weights once per step, so the kernel is built
// around one stream of weight bytes per warp:
//   * MMA operands swapped: m16n8k16 with A = 16 weight rows, B = 8 tokens, so a
//     batch of M tokens costs ceil(M/8) n8 tiles and no padding to 64.
//   * W is pre-shuffled (pack_weight) so each lane's A fragment for a k16 block
//     is 16 contiguous bytes: one 128-bit streaming load (ld.global.cs.v4), a
//     warp reads 512 contiguous bytes, no shared memory for W.
//   * Each warp keeps kDepth k16 blocks of W in flight in registers.
//   * x is staged per 128-wide K chunk through a double-buffered cp.async ring in
//     shared memory and read with ldmatrix.x4 (two n8 tiles per instruction).
//   * The 8 warps of a CTA are RW row groups x KW K slices. K slices are reduced
//     through shared memory in a fixed order; only GEMMs too narrow to fill the
//     GPU also split K across CTAs, with a fixed-order fix-up by the last CTA.
//     The split depends only on (N, K, #SMs), so results do not depend on M.
#include "decode_gemm.h"

#include <cuda_bf16.h>

#include <algorithm>

namespace chandra {
namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kKC = 128;                   // K per x chunk
constexpr int kBlocksPerChunk = kKC / 16;  // k16 blocks per chunk
constexpr int kDepth = 4;                  // W k16 blocks in flight per warp
// One token's row of an x chunk in shared memory. The extra 16 bytes put the 8
// rows that one ldmatrix 8x8 reads into 8 different 16-byte bank groups.
constexpr int kRowBytes = kKC * 2 + 16;

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int bytes = valid ? 16 : 0;  // 0: zero-fill the 16 bytes
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            const void* smem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(s));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// NP: pairs of n8 token tiles (M <= 16 * NP). RW x KW = kWarps.
template <int NP, int RW, int KW>
__global__ void __launch_bounds__(kThreads, 2)
    decode_gemm_kernel(const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ wp,
                       __nv_bfloat16* __restrict__ y, float* __restrict__ partial,
                       unsigned* __restrict__ counters, int M, int N, int K, int splits) {
  static_assert(RW * KW == kWarps, "8 warps per CTA");
  constexpr int MP = NP * 16;
  constexpr int kBufBytes = MP * kRowBytes;
  constexpr int kAcc = 2 * NP * 4;  // accumulator floats per lane
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ int s_last;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rw = warp / KW, kq = warp % KW;
  const int tile = blockIdx.x, split = blockIdx.y;
  const int KB = K >> 4;
  const int kb_split = KB / splits;
  const int kb0 = split * kb_split;
  const int nchunks = kb_split / kBlocksPerChunk;
  const int nblk = kb_split / KW;  // this warp's k16 blocks
  const int g = tile * RW + rw;    // 16-row group
  // Block i of this warp is k16 block kb0 + i * KW + kq of row group g.
  const uint4* w = wp + (static_cast<size_t>(g) * KB + kb0 + kq) * 32 + lane;
  const size_t wstep = static_cast<size_t>(KW) * 32;

  auto load_x = [&](int c, int buf) {
    unsigned char* s = smem + buf * kBufBytes;
    const int k0 = (kb0 + c * kBlocksPerChunk) * 16;
#pragma unroll
    for (int i = threadIdx.x; i < MP * (kKC / 8); i += kThreads) {
      const int r = i / (kKC / 8), cc = i % (kKC / 8);
      const bool valid = r < M;
      const __nv_bfloat16* src = x + static_cast<size_t>(valid ? r : 0) * K + k0 + cc * 8;
      cp_async16(s + r * kRowBytes + cc * 16, src, valid);
    }
  };

  // The W ring: w0 is the next block to use.
  uint4 w0 = make_uint4(0, 0, 0, 0), w1 = w0, w2 = w0, w3 = w0;
  if (0 < nblk) w0 = ld_stream(w);
  if (1 < nblk) w1 = ld_stream(w + wstep);
  if (2 < nblk) w2 = ld_stream(w + 2 * wstep);
  if (3 < nblk) w3 = ld_stream(w + 3 * wstep);

  float acc[2 * NP][4];
#pragma unroll
  for (int t = 0; t < 2 * NP; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;

  load_x(0, 0);
  cp_async_commit();
  // ldmatrix.x4 row address of this lane: matrix q = lane / 8 is n8 tile (q / 2)
  // of a pair and k half (q % 2); its row lane % 8 is a token.
  const int lq = lane >> 3, lr = lane & 7;
  const int lane_off = ((lq >> 1) * 8 + lr) * kRowBytes + (lq & 1) * 16;

  for (int c = 0; c < nchunks; ++c) {
    if (c + 1 < nchunks) load_x(c + 1, (c + 1) & 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const unsigned char* s = smem + (c & 1) * kBufBytes + lane_off;
#pragma unroll
    for (int jj = 0; jj < kBlocksPerChunk / KW; ++jj) {
      const int j = jj * KW + kq;                       // k16 block within the chunk
      const int i = c * (kBlocksPerChunk / KW) + jj;    // this warp's block index
      const uint4 a = w0;
      w0 = w1;
      w1 = w2;
      w2 = w3;
      if (i + kDepth < nblk) w3 = ld_stream(w + (i + kDepth) * wstep);
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4(b0, b1, b2, b3, s + p * 16 * kRowBytes + j * 32);
        mma_bf16(acc[2 * p], a, b0, b1);
        mma_bf16(acc[2 * p + 1], a, b2, b3);
      }
    }
    __syncthreads();
  }

  // Reduce the KW K slices of each row group through shared memory, in order.
  if constexpr (KW > 1) {
    float* red = reinterpret_cast<float*>(smem);  // [warp][lane][kAcc], fits in the x buffers
    float* mine = red + (warp * 32 + lane) * kAcc;
#pragma unroll
    for (int t = 0; t < 2 * NP; ++t)
      *reinterpret_cast<float4*>(mine + t * 4) = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
    __syncthreads();
    if (kq != 0) return;  // no later __syncthreads below on this path for these warps
#pragma unroll
    for (int t = 0; t < 2 * NP; ++t) {
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int q = 0; q < KW; ++q) {
        const float4 u = *reinterpret_cast<const float4*>(red + ((rw * KW + q) * 32 + lane) * kAcc + t * 4);
        v.x += u.x;
        v.y += u.y;
        v.z += u.z;
        v.w += u.w;
      }
      acc[t][0] = v.x;
      acc[t][1] = v.y;
      acc[t][2] = v.z;
      acc[t][3] = v.w;
    }
  }

  // Accumulator layout: c0, c1 = row gid, tokens 2*tig, 2*tig+1; c2, c3 = row gid + 8.
  const int gid = lane >> 2, tig = lane & 3;
  const int row = g * 16 + gid;
  if (splits == 1) {
#pragma unroll
    for (int t = 0; t < 2 * NP; ++t) {
      const int tok = t * 8 + tig * 2;
      if (tok < M) {
        y[static_cast<size_t>(tok) * N + row] = __float2bfloat16_rn(acc[t][0]);
        y[static_cast<size_t>(tok) * N + row + 8] = __float2bfloat16_rn(acc[t][2]);
      }
      if (tok + 1 < M) {
        y[static_cast<size_t>(tok + 1) * N + row] = __float2bfloat16_rn(acc[t][1]);
        y[static_cast<size_t>(tok + 1) * N + row + 8] = __float2bfloat16_rn(acc[t][3]);
      }
    }
    return;
  }

  // Split across CTAs: write this split's partial [split][N][MP], then the last
  // CTA of the row tile sums the splits in order and writes y.
  float* p0 = partial + (static_cast<size_t>(split) * N + row) * MP;
#pragma unroll
  for (int t = 0; t < 2 * NP; ++t) {
    const int tok = t * 8 + tig * 2;
    *reinterpret_cast<float2*>(p0 + tok) = make_float2(acc[t][0], acc[t][1]);
    *reinterpret_cast<float2*>(p0 + 8 * MP + tok) = make_float2(acc[t][2], acc[t][3]);
  }
  __threadfence();
  // With KW > 1 the kq != 0 warps have returned; the RW writer warps meet at a
  // named barrier (id 1, RW * 32 threads) instead of __syncthreads.
  constexpr int kWriters = RW;
  asm volatile("bar.sync 1, %0;" ::"n"(kWriters * 32));
  if (rw == 0 && lane == 0)
    s_last = atomicAdd(&counters[tile], 1u) == static_cast<unsigned>(splits - 1);
  asm volatile("bar.sync 1, %0;" ::"n"(kWriters * 32));
  if (!s_last) return;
  __threadfence();
  const int rows = RW * 16, r0 = tile * rows;
  const int tid = rw * 32 + lane;
  for (int i = tid; i < rows * M; i += kWriters * 32) {
    const int tok = i / rows, r = r0 + i % rows;
    float sum = 0.f;
    for (int s2 = 0; s2 < splits; ++s2)
      sum += __ldcg(partial + (static_cast<size_t>(s2) * N + r) * MP + tok);
    y[static_cast<size_t>(tok) * N + r] = __float2bfloat16_rn(sum);
  }
  if (tid == 0) counters[tile] = 0;
}

template <int NP, int RW, int KW>
cudaError_t launch(const void* x, const void* wp, void* y, float* ws, unsigned* counters, int M, int N,
                   int K, int splits, cudaStream_t stream) {
  constexpr int smem = 2 * NP * 16 * kRowBytes;
  const dim3 grid(N / (16 * RW), splits);
  decode_gemm_kernel<NP, RW, KW><<<grid, kThreads, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const uint4*>(wp), static_cast<__nv_bfloat16*>(y),
      ws, counters, M, N, K, splits);
  return cudaGetLastError();
}

template <int NP>
cudaError_t launch_rw(int rw, const void* x, const void* wp, void* y, float* ws, unsigned* counters,
                      int M, int N, int K, int splits, cudaStream_t stream) {
  switch (rw) {
    case 8: return launch<NP, 8, 1>(x, wp, y, ws, counters, M, N, K, splits, stream);
    case 4: return launch<NP, 4, 2>(x, wp, y, ws, counters, M, N, K, splits, stream);
    case 2: return launch<NP, 2, 4>(x, wp, y, ws, counters, M, N, K, splits, stream);
    default: return launch<NP, 1, 8>(x, wp, y, ws, counters, M, N, K, splits, stream);
  }
}

}  // namespace

DecodeGemmPlan decode_gemm_plan(int N, int K, int sms) {
  // A CTA's x chunk is re-read from L2 by every CTA, so x traffic per weight byte is
  // (padded tokens) / (rows per CTA): take the widest row tile (128 rows). GEMMs with
  // fewer tiles than SMs split K across CTAs, using the largest split (dividing K's
  // chunks) that does not exceed one CTA per SM, so the FP32 partials stay small.
  // One CTA per SM keeps 8 warps x kDepth x 512 B = 16 KB of weights in flight per SM,
  // about 3x what DRAM latency needs.
  const int chunks = K / kKC;
  const int rw = N % 128 == 0 ? 8 : N % 64 == 0 ? 4 : N % 32 == 0 ? 2 : 1;
  const int tiles = N / (16 * rw);
  int splits = 1;
  if (tiles < sms) {
    const int want = std::max(1, sms / tiles);
    for (int s = std::min(want, chunks); s >= 1; --s) {
      if (chunks % s == 0) {
        splits = s;
        break;
      }
    }
  }
  return {rw, splits};
}

size_t decode_gemm_workspace_bytes(int N, int K, int sms) {
  const DecodeGemmPlan p = decode_gemm_plan(N, K, sms);
  return p.splits > 1 ? static_cast<size_t>(p.splits) * N * 64 * sizeof(float) : 0;
}

int decode_gemm_counters(int N, int K, int sms) {
  const DecodeGemmPlan p = decode_gemm_plan(N, K, sms);
  return N / (16 * p.row_groups);
}

cudaError_t decode_gemm(const void* x, const void* wp, void* y, float* workspace, unsigned* counters,
                        int M, int N, int K, int sms, cudaStream_t stream) {
  if (M < 1 || M > 64 || N % 16 != 0 || K % kKC != 0) return cudaErrorInvalidValue;
  const DecodeGemmPlan p = decode_gemm_plan(N, K, sms);
  const int np = (M + 15) / 16;
  switch (np) {
    case 1: return launch_rw<1>(p.row_groups, x, wp, y, workspace, counters, M, N, K, p.splits, stream);
    case 2: return launch_rw<2>(p.row_groups, x, wp, y, workspace, counters, M, N, K, p.splits, stream);
    case 3: return launch_rw<3>(p.row_groups, x, wp, y, workspace, counters, M, N, K, p.splits, stream);
    default: return launch_rw<4>(p.row_groups, x, wp, y, workspace, counters, M, N, K, p.splits, stream);
  }
}

}  // namespace chandra
