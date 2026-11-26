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

// Greedy-pick key of a logit: the order-preserving bits of its BF16 value, then the
// inverted row, so a max picks the largest value and, among equal ones, the lowest row
// (the first index, as torch.argmax does).
__device__ __forceinline__ unsigned long long argmax_key(float v, int row) {
  const unsigned u = __float_as_uint(__bfloat162float(__float2bfloat16_rn(v)));
  const unsigned ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (static_cast<unsigned long long>(ord) << 32) | (0xffffffffu - static_cast<unsigned>(row));
}

// Four FP8 e4m3 weights -> two BF16 pairs, exactly, without FP8 hardware (sm_80/86):
// PRMT puts each byte in the high byte of a 16-bit lane, one shift and mask/or makes
// BF16 bits whose exponent bias is 127 instead of 7, and one HMUL2 by 2^120 fixes the
// bias (the products are representable, subnormals included).
__device__ __forceinline__ void fp8x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t v01 = __byte_perm(w, 0u, 0x1404);  // b1 << 24 | b0 << 8
  const uint32_t v23 = __byte_perm(w, 0u, 0x3424);  // b3 << 24 | b2 << 8
  const uint32_t t01 = ((v01 >> 4) & 0x07F007F0u) | (v01 & 0x80008000u);
  const uint32_t t23 = ((v23 >> 4) & 0x07F007F0u) | (v23 & 0x80008000u);
  const uint32_t k = 0x7B807B80u;  // 2^120 in both BF16 halves
  const __nv_bfloat162 r01 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&t01),
                                     *reinterpret_cast<const __nv_bfloat162*>(&k));
  const __nv_bfloat162 r23 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&t23),
                                     *reinterpret_cast<const __nv_bfloat162*>(&k));
  lo = *reinterpret_cast<const uint32_t*>(&r01);
  hi = *reinterpret_cast<const uint32_t*>(&r23);
}

// Eight MXFP4 (e2m1) weights of one k16 block -> the four BF16 A registers, exactly.
// pack_weight_fp4 places each code's magnitude bits (e1 e0 m) where BF16's exponent LSBs
// and mantissa MSB are (bits 8:6 of a half; already there for a0, a shift away for the
// others) and its sign at bit 15: those BF16 bits are e2m1 * 2^-126, subnormal 0.5 included.
// One HMUL2 by 2^(E - 1) then gives e2m1 * 2^(E - 127), the block's scale applied.
__device__ __forceinline__ void fp4x8_to_bf16(uint32_t w, uint32_t m_lo, uint32_t m_hi, uint4& a) {
  constexpr uint32_t kMag = 0x01C001C0u, kSign = 0x80008000u;
  const uint32_t a0 = w & (kMag | kSign);
  const uint32_t a1 = ((w >> 3) & kMag) | ((w << 1) & kSign);
  const uint32_t a2 = ((w << 6) & kMag) | ((w << 2) & kSign);
  const uint32_t a3 = (w << 3) & (kMag | kSign);
  const __nv_bfloat162 r0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a0),
                                    *reinterpret_cast<const __nv_bfloat162*>(&m_lo));
  const __nv_bfloat162 r1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a1),
                                    *reinterpret_cast<const __nv_bfloat162*>(&m_hi));
  const __nv_bfloat162 r2 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a2),
                                    *reinterpret_cast<const __nv_bfloat162*>(&m_lo));
  const __nv_bfloat162 r3 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a3),
                                    *reinterpret_cast<const __nv_bfloat162*>(&m_hi));
  a.x = *reinterpret_cast<const uint32_t*>(&r0);
  a.y = *reinterpret_cast<const uint32_t*>(&r1);
  a.z = *reinterpret_cast<const uint32_t*>(&r2);
  a.w = *reinterpret_cast<const uint32_t*>(&r3);
}

// Byte b (E + 126) of a scale word -> 2^(E - 1) in both BF16 halves: (b << 7) | (b << 23).
template <int B>
__device__ __forceinline__ uint32_t e8_to_bf16x2(uint32_t s) {
  return __byte_perm(s, 0u, 0x0404 | (B << 4) | (B << 12)) >> 1;
}

// Four INT8 weights in sign-magnitude form (s << 7 | |q|, |q| <= 127) -> two BF16 pairs,
// exactly: PRMT puts each byte in the low byte of a 16-bit lane and its sign (replicated) in
// the high byte; with 0x4300 or'ed in a lane is +-(128 + |q|), and subtracting its own
// sign-and-exponent (+-128) leaves +-|q|.
__device__ __forceinline__ void i8x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t t01 = (__byte_perm(w, 0u, 0x9180) & 0x807F807Fu) | 0x43004300u;
  const uint32_t t23 = (__byte_perm(w, 0u, 0xB3A2) & 0x807F807Fu) | 0x43004300u;
  const uint32_t c01 = t01 & 0xFF80FF80u, c23 = t23 & 0xFF80FF80u;
  const __nv_bfloat162 r01 = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&t01),
                                     *reinterpret_cast<const __nv_bfloat162*>(&c01));
  const __nv_bfloat162 r23 = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&t23),
                                     *reinterpret_cast<const __nv_bfloat162*>(&c23));
  lo = *reinterpret_cast<const uint32_t*>(&r01);
  hi = *reinterpret_cast<const uint32_t*>(&r23);
}

// Order-preserving float <-> uint (for atomicMax on floats).
__device__ __forceinline__ unsigned f2ord(float v) {
  const unsigned u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ord2f(unsigned o) {
  return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o);
}
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// NP: pairs of n8 token tiles (M <= 16 * NP). RW x KW = kWarps. ARGMAX: instead of
// writing y, reduce each token's greedy pick into keys[token] (an LM head that never
// writes its logits); needs splits == 1. FP8: W holds e4m3 bytes (one 16-byte load per
// lane covers two k16 blocks) and wscale[row] scales each output row.
// I8: INT8 sign-magnitude weights with per-row scales (the FP8 layout and scaling).
// BOUND (with I8, the exact LM head's first pass): instead of y, per token the max over this
// tile's rows of an upper bound of the BF16 head's logit goes to bnd.tile_ub[tile][token],
// and the max lower bound over the vocabulary to bnd.lb_key[token] (f2ord). ARGMAX with
// bnd.tile_ub set (the re-check): a tile whose upper bound rounds (BF16) below the best
// lower bound cannot hold the pick (not even a tie) and returns at once; the others run
// exactly as the plain BF16 argmax kernel does.
struct BoundArgs {
  const float* xnorm;  // [M] ||x_m||_2, rounded up
  const float* rowc;   // [N] c_n: |x . (w_n - w8_n)| <= ||x|| c_n, accumulation error included
  float* tile_ub;      // [N / 128][64]
  unsigned* lb_key;    // [M]
};

template <int NP, int RW, int KW, bool ARGMAX = false, bool FP8 = false, bool FP4 = false, bool I8 = false,
          bool BOUND = false>
__global__ void __launch_bounds__(kThreads, 2)
    decode_gemm_kernel(const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ wp,
                       __nv_bfloat16* __restrict__ y, float* __restrict__ partial,
                       unsigned* __restrict__ counters, int M, int N, int K, int splits,
                       unsigned long long* __restrict__ keys, const float* __restrict__ wscale,
                       const BoundArgs bnd) {
  constexpr int BPL = (FP8 || I8) ? 2 : FP4 ? 4 : 1;  // k16 blocks per 16-byte W load
  static_assert(!I8 || !(FP8 || FP4), "one weight format");
  static_assert(!BOUND || (I8 && !ARGMAX), "the bound pass is INT8, without the argmax epilogue");
  static_assert(!FP8 || KW <= 4, "FP8 needs at least one W load per warp per chunk");
  static_assert(!FP4 || KW <= 2, "FP4 needs at least one W load per warp per chunk");
  static_assert(!(FP8 && FP4), "one weight format");
  static_assert(RW * KW == kWarps, "8 warps per CTA");
  constexpr int MP = NP * 16;
  constexpr int kBufBytes = MP * kRowBytes;
  constexpr int kAcc = 2 * NP * 4;  // accumulator floats per lane
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ int s_last;
  __shared__ unsigned long long s_best[64];
  __shared__ unsigned s_ub[64], s_lb[64];
  __shared__ bool s_cand[64];
  if constexpr (ARGMAX) {
    if (threadIdx.x < 64) s_best[threadIdx.x] = 0ull;  // the main loop's barriers order this
    if (bnd.tile_ub != nullptr) {  // re-check: only tiles that can hold some token's pick
      bool c = false;
      if (threadIdx.x < M)
        c = !(bf16r(bnd.tile_ub[blockIdx.x * 64 + threadIdx.x]) < bf16r(ord2f(bnd.lb_key[threadIdx.x])));
      if (threadIdx.x < 64) s_cand[threadIdx.x] = c;
      if (!__syncthreads_or(c)) return;
    }
  }
  if constexpr (BOUND) {
    if (threadIdx.x < 64) s_ub[threadIdx.x] = s_lb[threadIdx.x] = 0u;
  }

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rw = warp / KW, kq = warp % KW;
  const int tile = blockIdx.x, split = blockIdx.y;
  const int KB = K >> 4;
  const int kb_split = KB / splits;
  const int kb0 = split * kb_split;
  const int nchunks = kb_split / kBlocksPerChunk;
  const int nblk = kb_split / BPL / KW;  // this warp's W loads
  const int g = tile * RW + rw;    // 16-row group
  // Block i of this warp is k16 block kb0 + i * KW + kq of row group g.
  const uint4* w = wp + (static_cast<size_t>(g) * (KB / BPL) + kb0 / BPL + kq) * 32 + lane;
  const size_t wstep = static_cast<size_t>(KW) * 32;
  // FP4: one scale word per (row group, W load, gid): E + 126 of rows gid and gid + 8 for the
  // load's two 32-wide K blocks; the 4 lanes of a quad read the same word.
  const uint32_t* sc = FP4 ? reinterpret_cast<const uint32_t*>(wscale) +
                                 (static_cast<size_t>(g) * (KB / 4) + kb0 / 4 + kq) * 8 + (lane >> 2)
                           : nullptr;
  constexpr size_t sstep = static_cast<size_t>(KW) * 8;

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
  uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;  // FP4 scale words, in step with w0..w3
  if constexpr (FP4) {
    if (0 < nblk) s0 = __ldg(sc);
    if (1 < nblk) s1 = __ldg(sc + sstep);
    if (2 < nblk) s2 = __ldg(sc + 2 * sstep);
    if (3 < nblk) s3 = __ldg(sc + 3 * sstep);
  }

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
    for (int jj = 0; jj < kBlocksPerChunk / BPL / KW; ++jj) {
      const int u = jj * KW + kq;                             // W load within the chunk
      const int i = c * (kBlocksPerChunk / BPL / KW) + jj;    // this warp's load index
      const uint4 wv = w0;
      w0 = w1;
      w1 = w2;
      w2 = w3;
      if (i + kDepth < nblk) w3 = ld_stream(w + (i + kDepth) * wstep);
      uint4 af[BPL];
      if constexpr (FP4) {
        const uint32_t sv = s0;
        s0 = s1;
        s1 = s2;
        s2 = s3;
        if (i + kDepth < nblk) s3 = __ldg(sc + (i + kDepth) * sstep);
        const uint32_t m00 = e8_to_bf16x2<0>(sv), m01 = e8_to_bf16x2<1>(sv);  // row gid, K blocks 0 / 1
        const uint32_t m80 = e8_to_bf16x2<2>(sv), m81 = e8_to_bf16x2<3>(sv);  // row gid + 8
        fp4x8_to_bf16(wv.x, m00, m80, af[0]);
        fp4x8_to_bf16(wv.y, m00, m80, af[1]);
        fp4x8_to_bf16(wv.z, m01, m81, af[2]);
        fp4x8_to_bf16(wv.w, m01, m81, af[3]);
      } else if constexpr (I8) {
        i8x4_to_bf16(wv.x, af[0].x, af[0].y);
        i8x4_to_bf16(wv.y, af[0].z, af[0].w);
        i8x4_to_bf16(wv.z, af[1].x, af[1].y);
        i8x4_to_bf16(wv.w, af[1].z, af[1].w);
      } else if constexpr (FP8) {
        fp8x4_to_bf16(wv.x, af[0].x, af[0].y);
        fp8x4_to_bf16(wv.y, af[0].z, af[0].w);
        fp8x4_to_bf16(wv.z, af[1].x, af[1].y);
        fp8x4_to_bf16(wv.w, af[1].z, af[1].w);
      } else {
        af[0] = wv;
      }
#pragma unroll
      for (int bb = 0; bb < BPL; ++bb) {
        const int j = u * BPL + bb;  // k16 block within the chunk
#pragma unroll
        for (int p = 0; p < NP; ++p) {
          uint32_t b0, b1, b2, b3;
          ldmatrix_x4(b0, b1, b2, b3, s + p * 16 * kRowBytes + j * 32);
          mma_bf16(acc[2 * p], af[bb], b0, b1);
          mma_bf16(acc[2 * p + 1], af[bb], b2, b3);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (FP8 || I8) {  // per-row weight scales (linear, so before the K-slice sum)
    const float s0 = wscale[g * 16 + (lane >> 2)], s8 = wscale[g * 16 + (lane >> 2) + 8];
#pragma unroll
    for (int t = 0; t < 2 * NP; ++t) {
      acc[t][0] *= s0;
      acc[t][1] *= s0;
      acc[t][2] *= s8;
      acc[t][3] *= s8;
    }
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
  if constexpr (ARGMAX) {
    static_assert(KW == 1, "the argmax epilogue needs every warp");
#pragma unroll
    for (int t = 0; t < 2 * NP; ++t) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int tok = t * 8 + tig * 2 + c;
        unsigned long long kk = argmax_key(acc[t][c], row);
        const unsigned long long k8 = argmax_key(acc[t][c + 2], row + 8);
        kk = kk > k8 ? kk : k8;
        // Max over the 8 row lanes (gid) that hold this token.
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
          const unsigned long long o = __shfl_xor_sync(0xffffffffu, kk, off);
          kk = kk > o ? kk : o;
        }
        if (gid == 0 && tok < M && (bnd.tile_ub == nullptr || s_cand[tok])) atomicMax(&s_best[tok], kk);
      }
    }
    __syncthreads();
    if (threadIdx.x < M) atomicMax(keys + threadIdx.x, s_best[threadIdx.x]);
    return;
  }

  if constexpr (BOUND) {
    static_assert(KW == 1, "the bound epilogue needs every warp");
    const float c0 = bnd.rowc[row], c8 = bnd.rowc[row + 8];
#pragma unroll
    for (int t = 0; t < 2 * NP; ++t) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int tok = t * 8 + tig * 2 + c;
        const float xn = tok < M ? bnd.xnorm[tok] : 0.f;
        const float l0 = acc[t][c], l8 = acc[t][c + 2];
        // |l| * 2^-20 covers the rounding of l, l * scale and l +- b (a few ulps each).
        const float b0 = fmaf(xn, c0, fmaf(fabsf(l0), 1e-6f, 1e-30f));
        const float b8 = fmaf(xn, c8, fmaf(fabsf(l8), 1e-6f, 1e-30f));
        float ub = fmaxf(l0 + b0, l8 + b8), lb = fmaxf(l0 - b0, l8 - b8);
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
          ub = fmaxf(ub, __shfl_xor_sync(0xffffffffu, ub, off));
          lb = fmaxf(lb, __shfl_xor_sync(0xffffffffu, lb, off));
        }
        if (gid == 0 && tok < M) {
          atomicMax(&s_ub[tok], f2ord(ub));
          atomicMax(&s_lb[tok], f2ord(lb));
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < M) {
      bnd.tile_ub[tile * 64 + threadIdx.x] = ord2f(s_ub[threadIdx.x]);
      atomicMax(bnd.lb_key + threadIdx.x, s_lb[threadIdx.x]);
    }
    return;
  }

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

template <int NP, int RW, int KW, bool ARGMAX = false, bool FP8 = false, bool FP4 = false, bool I8 = false,
          bool BOUND = false>
cudaError_t launch(const void* x, const void* wp, void* y, float* ws, unsigned* counters, int M, int N,
                   int K, int splits, cudaStream_t stream, unsigned long long* keys = nullptr,
                   const float* wscale = nullptr, BoundArgs bnd = {}) {
  constexpr int smem = 2 * NP * 16 * kRowBytes;
  const dim3 grid(N / (16 * RW), splits);
  decode_gemm_kernel<NP, RW, KW, ARGMAX, FP8, FP4, I8, BOUND><<<grid, kThreads, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const uint4*>(wp), static_cast<__nv_bfloat16*>(y),
      ws, counters, M, N, K, splits, keys, wscale, bnd);
  return cudaGetLastError();
}

template <int NP>
cudaError_t launch_fp8(int rw, const void* x, const void* wp, const float* wscale, void* y, float* ws,
                       unsigned* counters, int M, int N, int K, int splits, cudaStream_t stream) {
  switch (rw) {
    case 8: return launch<NP, 8, 1, false, true>(x, wp, y, ws, counters, M, N, K, splits, stream, nullptr, wscale);
    case 4: return launch<NP, 4, 2, false, true>(x, wp, y, ws, counters, M, N, K, splits, stream, nullptr, wscale);
    case 2: return launch<NP, 2, 4, false, true>(x, wp, y, ws, counters, M, N, K, splits, stream, nullptr, wscale);
    default: return cudaErrorInvalidValue;  // FP8 needs N % 32 == 0
  }
}

template <int NP>
cudaError_t launch_fp4(int rw, const void* x, const void* wp, const void* sc, void* y, float* ws,
                       unsigned* counters, int M, int N, int K, int splits, cudaStream_t stream) {
  const float* s = static_cast<const float*>(sc);
  switch (rw) {
    case 8: return launch<NP, 8, 1, false, false, true>(x, wp, y, ws, counters, M, N, K, splits, stream, nullptr, s);
    case 4: return launch<NP, 4, 2, false, false, true>(x, wp, y, ws, counters, M, N, K, splits, stream, nullptr, s);
    default: return cudaErrorInvalidValue;  // FP4 needs N % 64 == 0
  }
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

cudaError_t decode_gemm_argmax(const void* x, const void* wp, unsigned long long* keys, int M, int N, int K,
                               int sms, cudaStream_t stream) {
  // One CTA per 128-row tile and no K split (a max cannot combine partial sums). A
  // vocabulary-sized N gives thousands of tiles, so the GPU is full anyway.
  (void)sms;
  if (M < 1 || M > 64 || N % 128 != 0 || K % kKC != 0) return cudaErrorInvalidValue;
  switch ((M + 15) / 16) {
    case 1: return launch<1, 8, 1, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys);
    case 2: return launch<2, 8, 1, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys);
    case 3: return launch<3, 8, 1, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys);
    default: return launch<4, 8, 1, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys);
  }
}

cudaError_t lm_head_bound_i8(const void* x, const void* wp, const float* wscale, const float* rowc,
                             const float* xnorm, float* tile_ub, unsigned* lb_key, int M, int N, int K,
                             cudaStream_t stream) {
  if (M < 1 || M > 64 || N % 128 != 0 || K % kKC != 0) return cudaErrorInvalidValue;
  const BoundArgs b{xnorm, rowc, tile_ub, lb_key};
  switch ((M + 15) / 16) {
    case 1: return launch<1, 8, 1, false, false, false, true, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, nullptr, wscale, b);
    case 2: return launch<2, 8, 1, false, false, false, true, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, nullptr, wscale, b);
    case 3: return launch<3, 8, 1, false, false, false, true, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, nullptr, wscale, b);
    default: return launch<4, 8, 1, false, false, false, true, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, nullptr, wscale, b);
  }
}

cudaError_t lm_head_recheck(const void* x, const void* wp, const float* tile_ub, const unsigned* lb_key,
                            unsigned long long* keys, int M, int N, int K, cudaStream_t stream) {
  if (M < 1 || M > 64 || N % 128 != 0 || K % kKC != 0) return cudaErrorInvalidValue;
  const BoundArgs b{nullptr, nullptr, const_cast<float*>(tile_ub), const_cast<unsigned*>(lb_key)};
  switch ((M + 15) / 16) {
    case 1: return launch<1, 8, 1, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys, nullptr, b);
    case 2: return launch<2, 8, 1, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys, nullptr, b);
    case 3: return launch<3, 8, 1, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys, nullptr, b);
    default: return launch<4, 8, 1, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys, nullptr, b);
  }
}

cudaError_t decode_gemm_argmax_fp8(const void* x, const void* wp, const float* wscale, unsigned long long* keys,
                                   int M, int N, int K, cudaStream_t stream) {
  if (M < 1 || M > 64 || N % 128 != 0 || K % kKC != 0) return cudaErrorInvalidValue;
  switch ((M + 15) / 16) {
    case 1: return launch<1, 8, 1, true, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys, wscale);
    case 2: return launch<2, 8, 1, true, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys, wscale);
    case 3: return launch<3, 8, 1, true, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys, wscale);
    default: return launch<4, 8, 1, true, true>(x, wp, nullptr, nullptr, nullptr, M, N, K, 1, stream, keys, wscale);
  }
}

cudaError_t decode_gemm_fp4(const void* x, const void* wp, const void* scales, void* y, float* workspace,
                            unsigned* counters, int M, int N, int K, int sms, cudaStream_t stream) {
  if (M < 1 || M > 64 || N % 64 != 0 || K % kKC != 0) return cudaErrorInvalidValue;
  const DecodeGemmPlan p = decode_gemm_plan(N, K, sms);
  switch ((M + 15) / 16) {
    case 1: return launch_fp4<1>(p.row_groups, x, wp, scales, y, workspace, counters, M, N, K, p.splits, stream);
    case 2: return launch_fp4<2>(p.row_groups, x, wp, scales, y, workspace, counters, M, N, K, p.splits, stream);
    case 3: return launch_fp4<3>(p.row_groups, x, wp, scales, y, workspace, counters, M, N, K, p.splits, stream);
    default: return launch_fp4<4>(p.row_groups, x, wp, scales, y, workspace, counters, M, N, K, p.splits, stream);
  }
}

cudaError_t decode_gemm_fp8(const void* x, const void* wp, const float* wscale, void* y, float* workspace,
                            unsigned* counters, int M, int N, int K, int sms, cudaStream_t stream) {
  if (M < 1 || M > 64 || N % 32 != 0 || K % kKC != 0) return cudaErrorInvalidValue;
  const DecodeGemmPlan p = decode_gemm_plan(N, K, sms);
  switch ((M + 15) / 16) {
    case 1: return launch_fp8<1>(p.row_groups, x, wp, wscale, y, workspace, counters, M, N, K, p.splits, stream);
    case 2: return launch_fp8<2>(p.row_groups, x, wp, wscale, y, workspace, counters, M, N, K, p.splits, stream);
    case 3: return launch_fp8<3>(p.row_groups, x, wp, wscale, y, workspace, counters, M, N, K, p.splits, stream);
    default: return launch_fp8<4>(p.row_groups, x, wp, wscale, y, workspace, counters, M, N, K, p.splits, stream);
  }
}

}  // namespace chandra
