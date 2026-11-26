// Non-causal attention for the vision tower: head_dim 64, very long sequences.
//
// One CTA per (128-query tile, head, image), 8 warps of 16 query rows each.
//   * Q lives in registers as m16n8k16 A fragments (4 k-steps x 4 regs).
//   * K and V stream through a 3-stage cp.async ring of 64-key stages (8 KB each),
//     16-byte chunks XOR-swizzled by key so every ldmatrix phase is conflict-free.
//   * Per warp and stage: S = Q K^T with 32 mma.m16n8k16 (ldmatrix.x4 gives the B
//     fragments of two key tiles), online softmax in base 2, then O += P V with 32
//     more (the S accumulators repack in registers as P's A fragments; V's B
//     fragments come from ldmatrix.x4.trans).
//   * ~130 registers and 48 KB of shared memory: two CTAs (16 warps) per SM.
#include "vit_attn.h"

#include <cuda_bf16.h>

#include <cmath>

namespace chandra {
namespace {

constexpr int kThreads = 256;
constexpr int kQTile = 128;   // query rows per CTA
constexpr int kKTile = 64;    // keys per stage
constexpr int kStages = 3;
constexpr int kDim = 64;
constexpr int kRowBytes = kDim * 2;              // 128 bytes per key row
constexpr int kHalf = kKTile * kRowBytes;        // 8 KB: K (or V) of one stage
constexpr int kStage = 2 * kHalf;                // 16 KB
constexpr int kSmem = kStages * kStage;          // 48 KB
constexpr float kScaleLog2 = 0.125f * 1.4426950408889634f;  // 1/sqrt(64) * log2(e)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// Byte offset of 16-byte chunk c (0..7) of key row k inside a K or V stage.
__device__ __forceinline__ int swz(int k, int c) { return k * kRowBytes + ((c ^ (k & 7)) << 4); }

__global__ void __launch_bounds__(kThreads, 2) vit_attn_kernel(const VitAttnArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int tile = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
  const int start = a.cu_seqlens[seq], len = a.cu_seqlens[seq + 1] - start;
  if (tile * kQTile >= len) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const __nv_bfloat16* kg = static_cast<const __nv_bfloat16*>(a.k) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
  const __nv_bfloat16* vg = static_cast<const __nv_bfloat16*>(a.v) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
  const int n_tiles = (len + kKTile - 1) / kKTile;

  auto load = [&](int t, int buf) {
    unsigned char* sk = smem + buf * kStage;
    unsigned char* sv = sk + kHalf;
#pragma unroll
    for (int i = threadIdx.x; i < kKTile * 8; i += kThreads) {
      const int key = i >> 3, c = i & 7, row = t * kKTile + key;
      const bool valid = row < len;
      const int64_t off = static_cast<int64_t>(valid ? row : 0) * a.qkv_stride + c * 8;
      cp_async16(sk + swz(key, c), kg + off, valid);
      cp_async16(sv + swz(key, c), vg + off, valid);
    }
  };
  load(0, 0);
  cp_async_commit();
  if (n_tiles > 1) load(1, 1);
  cp_async_commit();

  // Q fragments of this warp's 16 rows (rows past the image are zero).
  uint32_t qf[4][4];
  {
    const int r0 = tile * kQTile + warp * 16 + gid;
    const __nv_bfloat16* qg = static_cast<const __nv_bfloat16*>(a.q) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = r0 + half * 8;
      const bool live = r < len;
      const __nv_bfloat16* qr = qg + static_cast<int64_t>(live ? r : 0) * a.qkv_stride + tig * 2;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        qf[kk][half] = live ? *reinterpret_cast<const uint32_t*>(qr + kk * 16) : 0u;
        qf[kk][half + 2] = live ? *reinterpret_cast<const uint32_t*>(qr + kk * 16 + 8) : 0u;
      }
    }
  }

  float o[8][4];
#pragma unroll
  for (int d = 0; d < 8; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int lq = lane >> 3, lr = lane & 7;

  for (int t = 0; t < n_tiles; ++t) {
    if (t + 2 < n_tiles) load(t + 2, (t + 2) % kStages);
    cp_async_commit();
    cp_async_wait<2>();
    __syncthreads();
    const unsigned char* sk = smem + (t % kStages) * kStage;
    const unsigned char* sv = sk + kHalf;

    // S = Q K^T over 64 keys: 8 n8 tiles.
    float s[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int np = 0; np < 4; ++np) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t b[4];
        // matrices: (keys 16np + 8(q>>1) .. +7) x (dims 16kk + 8(q&1) .. +7)
        ldsm_x4(b, sk + swz(16 * np + 8 * (lq >> 1) + lr, 2 * kk + (lq & 1)));
        mma(s[2 * np], qf[kk], b[0], b[1]);
        mma(s[2 * np + 1], qf[kk], b[2], b[3]);
      }
    }

    // Online softmax (base 2); keys past the image are -inf.
    const int key0 = t * kKTile + tig * 2;
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const bool v0 = key0 + n * 8 < len, v1 = key0 + n * 8 + 1 < len;
      s[n][0] = v0 ? s[n][0] * kScaleLog2 : -INFINITY;
      s[n][1] = v1 ? s[n][1] * kScaleLog2 : -INFINITY;
      s[n][2] = v0 ? s[n][2] * kScaleLog2 : -INFINITY;
      s[n][3] = v1 ? s[n][3] * kScaleLog2 : -INFINITY;
      mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
      mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float al0 = ex2(m0 - n0), al1 = ex2(m1 - n1);  // m = -inf only before the first tile: al = 0
    m0 = n0;
    m1 = n1;
    float rs0 = 0.f, rs1 = 0.f;
    uint32_t pf[4][4];  // P as A fragments, one per k16 step over the 64 keys
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const float e0 = ex2(s[n][0] - n0), e1 = ex2(s[n][1] - n0);
      const float e2 = ex2(s[n][2] - n1), e3 = ex2(s[n][3] - n1);
      rs0 += e0 + e1;
      rs1 += e2 + e3;
      pf[n >> 1][(n & 1) * 2] = pack_bf16(e0, e1);      // a0 / a2: row gid
      pf[n >> 1][(n & 1) * 2 + 1] = pack_bf16(e2, e3);  // a1 / a3: row gid + 8
    }
    l0 = l0 * al0 + rs0;
    l1 = l1 * al1 + rs1;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      o[d][0] *= al0;
      o[d][1] *= al0;
      o[d][2] *= al1;
      o[d][3] *= al1;
    }

    // O += P V: 8 n8 dim tiles, 4 k16 steps over keys.
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b[4];
        // matrices: (keys 16kk + 8(q&1) .. +7) x (dims 8(2dp + (q>>1)) .. +7), transposed
        ldsm_x4_t(b, sv + swz(16 * kk + 8 * (lq & 1) + lr, 2 * dp + (lq >> 1)));
        mma(o[2 * dp], pf[kk], b[0], b[1]);
        mma(o[2 * dp + 1], pf[kk], b[2], b[3]);
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();

  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  const int r0 = tile * kQTile + warp * 16 + gid;
  __nv_bfloat16* og = static_cast<__nv_bfloat16*>(a.out) + static_cast<int64_t>(start) * a.out_stride + h * kDim;
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    const int col = d * 8 + tig * 2;
    if (r0 < len)
      *reinterpret_cast<uint32_t*>(og + static_cast<int64_t>(r0) * a.out_stride + col) = pack_bf16(o[d][0] * i0, o[d][1] * i0);
    if (r0 + 8 < len)
      *reinterpret_cast<uint32_t*>(og + static_cast<int64_t>(r0 + 8) * a.out_stride + col) =
          pack_bf16(o[d][2] * i1, o[d][3] * i1);
  }
}

// v2: the next tile's S = Q K^T is issued before this tile's softmax, so the tensor
// pipe has HMMA work while the FP32/SFU units run the softmax (one basic block, no
// dependency between them). Tiles t and t+1 are resident, t+2 and t+3 in flight.
// Lazy rescale: O and l are rescaled only when a row's max grows by more than 8 (log2),
// exact after the final 1/l; P values then stay below 2^8, fine for BF16 and FP32 O.
__device__ __forceinline__ void qk_tile(float (&s)[8][4], const uint32_t (&qf)[4][4], const unsigned char* sk,
                                        int lq, int lr) {
#pragma unroll
  for (int n = 0; n < 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
  for (int np = 0; np < 4; ++np) {
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t b[4];
      ldsm_x4(b, sk + swz(16 * np + 8 * (lq >> 1) + lr, 2 * kk + (lq & 1)));
      mma(s[2 * np], qf[kk], b[0], b[1]);
      mma(s[2 * np + 1], qf[kk], b[2], b[3]);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1) vit_attn_kernel_v2(const VitAttnArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int tile = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
  const int start = a.cu_seqlens[seq], len = a.cu_seqlens[seq + 1] - start;
  if (tile * kQTile >= len) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const __nv_bfloat16* kg = static_cast<const __nv_bfloat16*>(a.k) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
  const __nv_bfloat16* vg = static_cast<const __nv_bfloat16*>(a.v) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
  const int n_tiles = (len + kKTile - 1) / kKTile;

  auto load = [&](int t) {
    unsigned char* sk = smem + (t % kStages) * kStage;
    unsigned char* sv = sk + kHalf;
#pragma unroll
    for (int i = threadIdx.x; i < kKTile * 8; i += kThreads) {
      const int key = i >> 3, c = i & 7, row = t * kKTile + key;
      const bool valid = row < len;
      const int64_t off = static_cast<int64_t>(valid ? row : 0) * a.qkv_stride + c * 8;
      cp_async16(sk + swz(key, c), kg + off, valid);
      cp_async16(sv + swz(key, c), vg + off, valid);
    }
  };
  // Groups: tile 0, 1, 2 in the prologue, then tile t + 3 at the end of iteration t
  // (always committed, possibly empty), so "tile j landed" is wait_group(<groups after j>).
  load(0);
  cp_async_commit();
  if (n_tiles > 1) load(1);
  cp_async_commit();
  if (n_tiles > 2) load(2);
  cp_async_commit();

  uint32_t qf[4][4];
  {
    const int r0 = tile * kQTile + warp * 16 + gid;
    const __nv_bfloat16* qg = static_cast<const __nv_bfloat16*>(a.q) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = r0 + half * 8;
      const bool live = r < len;
      const __nv_bfloat16* qr = qg + static_cast<int64_t>(live ? r : 0) * a.qkv_stride + tig * 2;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        qf[kk][half] = live ? *reinterpret_cast<const uint32_t*>(qr + kk * 16) : 0u;
        qf[kk][half + 2] = live ? *reinterpret_cast<const uint32_t*>(qr + kk * 16 + 8) : 0u;
      }
    }
  }
  const int lq = lane >> 3, lr = lane & 7;

  float o[8][4];
#pragma unroll
  for (int d = 0; d < 8; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  float s[8][4], sn[8][4];
  cp_async_wait<2>();  // tile 0
  __syncthreads();
  qk_tile(s, qf, smem, lq, lr);

  for (int t = 0; t < n_tiles; ++t) {
    if (t + 1 < n_tiles) {
      cp_async_wait<1>();  // tile t + 1 (its group is the second newest)
      __syncthreads();
      qk_tile(sn, qf, smem + ((t + 1) % kStages) * kStage, lq, lr);
    }
    const unsigned char* sv = smem + (t % kStages) * kStage + kHalf;

    // Softmax of tile t (independent of sn: the compiler interleaves it with the HMMAs).
    const int key0 = t * kKTile + tig * 2;
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const bool v0 = key0 + n * 8 < len, v1 = key0 + n * 8 + 1 < len;
      s[n][0] = v0 ? s[n][0] * kScaleLog2 : -INFINITY;
      s[n][1] = v1 ? s[n][1] * kScaleLog2 : -INFINITY;
      s[n][2] = v0 ? s[n][2] * kScaleLog2 : -INFINITY;
      s[n][3] = v1 ? s[n][3] * kScaleLog2 : -INFINITY;
      mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
      mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    // Lazy rescale: move a row's reference max only when this tile exceeds it by > 8.
    if (mx0 > m0 + 8.f) {
      const float al = ex2(m0 - mx0);  // 0 on the first tile (m0 = -inf)
      l0 *= al;
#pragma unroll
      for (int d = 0; d < 8; ++d) o[d][0] *= al, o[d][1] *= al;
      m0 = mx0;
    }
    if (mx1 > m1 + 8.f) {
      const float al = ex2(m1 - mx1);
      l1 *= al;
#pragma unroll
      for (int d = 0; d < 8; ++d) o[d][2] *= al, o[d][3] *= al;
      m1 = mx1;
    }
    uint32_t pf[4][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const float e0 = ex2(s[n][0] - m0), e1 = ex2(s[n][1] - m0);
      const float e2 = ex2(s[n][2] - m1), e3 = ex2(s[n][3] - m1);
      l0 += e0 + e1;
      l1 += e2 + e3;
      pf[n >> 1][(n & 1) * 2] = pack_bf16(e0, e1);
      pf[n >> 1][(n & 1) * 2 + 1] = pack_bf16(e2, e3);
    }

    // O += P V for tile t.
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b[4];
        ldsm_x4_t(b, sv + swz(16 * kk + 8 * (lq & 1) + lr, 2 * dp + (lq >> 1)));
        mma(o[2 * dp], pf[kk], b[0], b[1]);
        mma(o[2 * dp + 1], pf[kk], b[2], b[3]);
      }
    }
    __syncthreads();  // every warp is done with tile t's buffer
    if (t + 3 < n_tiles) load(t + 3);
    cp_async_commit();
#pragma unroll
    for (int n = 0; n < 8; ++n) s[n][0] = sn[n][0], s[n][1] = sn[n][1], s[n][2] = sn[n][2], s[n][3] = sn[n][3];
  }
  cp_async_wait<0>();

  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  const int r0 = tile * kQTile + warp * 16 + gid;
  __nv_bfloat16* og = static_cast<__nv_bfloat16*>(a.out) + static_cast<int64_t>(start) * a.out_stride + h * kDim;
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    const int col = d * 8 + tig * 2;
    if (r0 < len)
      *reinterpret_cast<uint32_t*>(og + static_cast<int64_t>(r0) * a.out_stride + col) = pack_bf16(o[d][0] * i0, o[d][1] * i0);
    if (r0 + 8 < len)
      *reinterpret_cast<uint32_t*>(og + static_cast<int64_t>(r0 + 8) * a.out_stride + col) =
          pack_bf16(o[d][2] * i1, o[d][3] * i1);
  }
}

// ---------------------------------------------------------------------------------
// INT8 QK^T (class G). Softmax is invariant to subtracting a per-channel K mean over
// the image (each score of a query row moves by the same q . mean), which centres K
// before it is quantized per 64-key tile. Q is quantized per row in registers. S is
// computed by mma.m16n8k32.s32.s8.s8.s32 (twice the BF16 rate) on 4 KB K tiles, and
// int32 -> float uses IADD3 + FADD with 1.5 * 2^23 (exact for |acc| < 2^22; here
// |acc| <= 64 * 127^2), so the SFU stays free for the softmax's EX2. PV stays BF16.

__global__ void __launch_bounds__(256) vit_k_mean_kernel(const VitAttnArgs a) {
  const int h = blockIdx.x, seq = blockIdx.y;
  const int start = a.cu_seqlens[seq], len = a.cu_seqlens[seq + 1] - start;
  const int c = threadIdx.x & 63, part = threadIdx.x >> 6;
  const __nv_bfloat16* kg = static_cast<const __nv_bfloat16*>(a.k) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim + c;
  float sum = 0.f;
  for (int r = part; r < len; r += 4) sum += __bfloat162float(kg[static_cast<int64_t>(r) * a.qkv_stride]);
  __shared__ float red[4][64];
  red[part][c] = sum;
  __syncthreads();
  if (part == 0)
    a.k_mean[(seq * 16 + h) * kDim + c] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) / fmaxf(1.f, len);
}

// One CTA per (64-key tile, head, image): thread = (key tid / 4, 16 dims).
__global__ void __launch_bounds__(256) vit_k_quant_kernel(const VitAttnArgs a) {
  const int t = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
  const int start = a.cu_seqlens[seq], len = a.cu_seqlens[seq + 1] - start;
  if (t * kKTile >= len) return;
  const int key = threadIdx.x >> 2, d0 = (threadIdx.x & 3) * 16, row = t * kKTile + key;
  const bool live = row < len;
  float x[16];
  float mx = 0.f;
  const float* mean = a.k_mean + (seq * 16 + h) * kDim + d0;
  if (live) {
    const __nv_bfloat16* src = static_cast<const __nv_bfloat16*>(a.k) + static_cast<int64_t>(start + row) * a.qkv_stride + h * kDim + d0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      x[j] = __bfloat162float(src[j]) - mean[j];
      mx = fmaxf(mx, fabsf(x[j]));
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  __shared__ float wmax[8];
  if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = mx;
  __syncthreads();
  float tmax = 0.f;
#pragma unroll
  for (int w = 0; w < 8; ++w) tmax = fmaxf(tmax, wmax[w]);
  const float scale = tmax > 0.f ? tmax / 127.f : 1.f, inv = 1.f / scale;
  if (threadIdx.x == 0) a.k_scale[(static_cast<size_t>(seq) * a.max_tiles + t) * 16 + h] = scale;
  if (!live) return;
  unsigned w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int q = max(-127, min(127, __float2int_rn(x[j] * inv)));
    w[j >> 2] |= (static_cast<unsigned>(q) & 0xffu) << (8 * (j & 3));
  }
  *reinterpret_cast<uint4*>(a.k8 + static_cast<int64_t>(start + row) * 1024 + h * kDim + d0) = make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ void imma(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float i2f_exact(int x) { return __int_as_float(x + 0x4B400000) - 12582912.f; }
// INT8 K tile: 64 keys x 64 bytes; chunk c (0..3) of key k at c ^ ((k >> 1) & 3) so the 8
// rows an ldmatrix reads land in 8 different 16-byte bank groups.
__device__ __forceinline__ int swz8(int k, int c) { return k * 64 + ((c ^ ((k >> 1) & 3)) << 4); }

constexpr int kHalf8 = kKTile * 64;          // 4 KB of INT8 K per stage
constexpr int kStage8 = kHalf8 + kHalf;      // + 8 KB of BF16 V
constexpr int kSmem8 = kStages * kStage8;    // 36 KB

__global__ void __launch_bounds__(kThreads, 2) vit_attn_kernel_i8(const VitAttnArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int tile = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
  const int start = a.cu_seqlens[seq], len = a.cu_seqlens[seq + 1] - start;
  if (tile * kQTile >= len) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int8_t* k8 = a.k8 + static_cast<int64_t>(start) * 1024 + h * kDim;
  const __nv_bfloat16* vg = static_cast<const __nv_bfloat16*>(a.v) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
  const float* ksc = a.k_scale + static_cast<size_t>(seq) * a.max_tiles * 16 + h;
  const int n_tiles = (len + kKTile - 1) / kKTile;

  auto load = [&](int t, int buf) {
    unsigned char* sk = smem + buf * kStage8;
    unsigned char* sv = sk + kHalf8;
    for (int i = threadIdx.x; i < kKTile * 12; i += kThreads) {
      if (i < kKTile * 4) {
        const int key = i >> 2, c = i & 3, row = t * kKTile + key;
        const bool valid = row < len;
        cp_async16(sk + swz8(key, c), k8 + static_cast<int64_t>(valid ? row : 0) * 1024 + c * 16, valid);
      } else {
        const int j = i - kKTile * 4, key = j >> 3, c = j & 7, row = t * kKTile + key;
        const bool valid = row < len;
        cp_async16(sv + swz(key, c), vg + static_cast<int64_t>(valid ? row : 0) * a.qkv_stride + c * 8, valid);
      }
    }
  };
  load(0, 0);
  cp_async_commit();
  if (n_tiles > 1) load(1, 1);
  cp_async_commit();

  // Q -> INT8 per row. The m16n8k32 A fragment: a0 = (row gid, dims 32ks + 4tig .. +3),
  // a1 = row gid + 8, a2 = dims + 16, a3 = both. A row's 64 dims sit in the 4 lanes of a quad.
  uint32_t qa[2][4];
  float sq0, sq1;
  {
    const int r0 = tile * kQTile + warp * 16 + gid;
    const __nv_bfloat16* qg = static_cast<const __nv_bfloat16*>(a.q) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
    float qv[2][2][2][4];  // [row half][ks][lo/hi 16][4 dims]
    float mx[2] = {0.f, 0.f};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = r0 + half * 8;
      const bool live = r < len;
      const __nv_bfloat16* qr = qg + static_cast<int64_t>(live ? r : 0) * a.qkv_stride;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int hi = 0; hi < 2; ++hi)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float v = live ? __bfloat162float(qr[32 * ks + 16 * hi + 4 * tig + j]) : 0.f;
            qv[half][ks][hi][j] = v;
            mx[half] = fmaxf(mx[half], fabsf(v));
          }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      mx[half] = fmaxf(mx[half], __shfl_xor_sync(0xffffffffu, mx[half], 1));
      mx[half] = fmaxf(mx[half], __shfl_xor_sync(0xffffffffu, mx[half], 2));
    }
    sq0 = mx[0] > 0.f ? mx[0] / 127.f : 1.f;
    sq1 = mx[1] > 0.f ? mx[1] / 127.f : 1.f;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks)
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
          const float inv = 1.f / (half ? sq1 : sq0);
          unsigned w = 0u;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int q = max(-127, min(127, __float2int_rn(qv[half][ks][hi][j] * inv)));
            w |= (static_cast<unsigned>(q) & 0xffu) << (8 * j);
          }
          qa[ks][half + 2 * hi] = w;  // a0: (gid, lo), a1: (gid+8, lo), a2: (gid, hi), a3: (gid+8, hi)
        }
  }

  float o[8][4];
#pragma unroll
  for (int d = 0; d < 8; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int lq = lane >> 3, lr = lane & 7;

  for (int t = 0; t < n_tiles; ++t) {
    if (t + 2 < n_tiles) load(t + 2, (t + 2) % kStages);
    cp_async_commit();
    cp_async_wait<2>();
    __syncthreads();
    const unsigned char* sk = smem + (t % kStages) * kStage8;
    const unsigned char* sv = sk + kHalf8;

    int acc[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0;
#pragma unroll
    for (int np = 0; np < 4; ++np) {
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t b[4];
        // matrices: (keys 16np + 8(q>>1) .. +7) x (16-byte chunk 2ks + (q&1)) of INT8 K
        ldsm_x4(b, sk + swz8(16 * np + 8 * (lq >> 1) + lr, 2 * ks + (lq & 1)));
        imma(acc[2 * np], qa[ks], b[0], b[1]);
        imma(acc[2 * np + 1], qa[ks], b[2], b[3]);
      }
    }
    const float sk_t = ksc[static_cast<size_t>(t) * 16];
    const float f0 = sq0 * sk_t * kScaleLog2, f1 = sq1 * sk_t * kScaleLog2;
    float s[8][4];
    const int key0 = t * kKTile + tig * 2;
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const bool v0 = key0 + n * 8 < len, v1 = key0 + n * 8 + 1 < len;
      s[n][0] = v0 ? i2f_exact(acc[n][0]) * f0 : -INFINITY;
      s[n][1] = v1 ? i2f_exact(acc[n][1]) * f0 : -INFINITY;
      s[n][2] = v0 ? i2f_exact(acc[n][2]) * f1 : -INFINITY;
      s[n][3] = v1 ? i2f_exact(acc[n][3]) * f1 : -INFINITY;
      mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
      mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float al0 = ex2(m0 - n0), al1 = ex2(m1 - n1);
    m0 = n0;
    m1 = n1;
    float rs0 = 0.f, rs1 = 0.f;
    uint32_t pf[4][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const float e0 = ex2(s[n][0] - n0), e1 = ex2(s[n][1] - n0);
      const float e2 = ex2(s[n][2] - n1), e3 = ex2(s[n][3] - n1);
      rs0 += e0 + e1;
      rs1 += e2 + e3;
      pf[n >> 1][(n & 1) * 2] = pack_bf16(e0, e1);
      pf[n >> 1][(n & 1) * 2 + 1] = pack_bf16(e2, e3);
    }
    l0 = l0 * al0 + rs0;
    l1 = l1 * al1 + rs1;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      o[d][0] *= al0;
      o[d][1] *= al0;
      o[d][2] *= al1;
      o[d][3] *= al1;
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b[4];
        ldsm_x4_t(b, sv + swz(16 * kk + 8 * (lq & 1) + lr, 2 * dp + (lq >> 1)));
        mma(o[2 * dp], pf[kk], b[0], b[1]);
        mma(o[2 * dp + 1], pf[kk], b[2], b[3]);
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();

  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  const int r0 = tile * kQTile + warp * 16 + gid;
  __nv_bfloat16* og = static_cast<__nv_bfloat16*>(a.out) + static_cast<int64_t>(start) * a.out_stride + h * kDim;
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    const int col = d * 8 + tig * 2;
    if (r0 < len)
      *reinterpret_cast<uint32_t*>(og + static_cast<int64_t>(r0) * a.out_stride + col) = pack_bf16(o[d][0] * i0, o[d][1] * i0);
    if (r0 + 8 < len)
      *reinterpret_cast<uint32_t*>(og + static_cast<int64_t>(r0 + 8) * a.out_stride + col) =
          pack_bf16(o[d][2] * i1, o[d][3] * i1);
  }
}

// ---------------------------------------------------------------------------------
// v4: 4 warps x 32 query rows. Each K (ldmatrix.x4) and V (ldmatrix.x4.trans) fragment feeds
// the MMAs of two m16 tiles, halving shared-memory reads per MMA against v1, and each warp has
// twice the independent MMA chains. Same 128-query CTA, 64-key stages and 3-stage ring.
constexpr int kThreads4 = 128;

__global__ void __launch_bounds__(kThreads4, 2) vit_attn_kernel_v4(const VitAttnArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int tile = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
  const int start = a.cu_seqlens[seq], len = a.cu_seqlens[seq + 1] - start;
  if (tile * kQTile >= len) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const __nv_bfloat16* kg = static_cast<const __nv_bfloat16*>(a.k) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
  const __nv_bfloat16* vg = static_cast<const __nv_bfloat16*>(a.v) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
  const int n_tiles = (len + kKTile - 1) / kKTile;

  auto load = [&](int t, int buf) {
    unsigned char* sk = smem + buf * kStage;
    unsigned char* sv = sk + kHalf;
#pragma unroll
    for (int i = threadIdx.x; i < kKTile * 8; i += kThreads4) {
      const int key = i >> 3, c = i & 7, row = t * kKTile + key;
      const bool valid = row < len;
      const int64_t off = static_cast<int64_t>(valid ? row : 0) * a.qkv_stride + c * 8;
      cp_async16(sk + swz(key, c), kg + off, valid);
      cp_async16(sv + swz(key, c), vg + off, valid);
    }
  };
  load(0, 0);
  cp_async_commit();
  if (n_tiles > 1) load(1, 1);
  cp_async_commit();

  uint32_t qf[2][4][4];
  const __nv_bfloat16* qg = static_cast<const __nv_bfloat16*>(a.q) + static_cast<int64_t>(start) * a.qkv_stride + h * kDim;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int r0 = tile * kQTile + warp * 32 + mt * 16 + gid;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = r0 + half * 8;
      const bool live = r < len;
      const __nv_bfloat16* qr = qg + static_cast<int64_t>(live ? r : 0) * a.qkv_stride + tig * 2;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        qf[mt][kk][half] = live ? *reinterpret_cast<const uint32_t*>(qr + kk * 16) : 0u;
        qf[mt][kk][half + 2] = live ? *reinterpret_cast<const uint32_t*>(qr + kk * 16 + 8) : 0u;
      }
    }
  }

  float o[2][8][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int d = 0; d < 8; ++d) o[mt][d][0] = o[mt][d][1] = o[mt][d][2] = o[mt][d][3] = 0.f;
  float m[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}}, l[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  const int lq = lane >> 3, lr = lane & 7;

  for (int t = 0; t < n_tiles; ++t) {
    if (t + 2 < n_tiles) load(t + 2, (t + 2) % kStages);
    cp_async_commit();
    cp_async_wait<2>();
    __syncthreads();
    const unsigned char* sk = smem + (t % kStages) * kStage;
    const unsigned char* sv = sk + kHalf;

    float sc[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int n = 0; n < 8; ++n) sc[mt][n][0] = sc[mt][n][1] = sc[mt][n][2] = sc[mt][n][3] = 0.f;
#pragma unroll
    for (int np = 0; np < 4; ++np) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t b[4];
        ldsm_x4(b, sk + swz(16 * np + 8 * (lq >> 1) + lr, 2 * kk + (lq & 1)));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma(sc[mt][2 * np], qf[mt][kk], b[0], b[1]);
          mma(sc[mt][2 * np + 1], qf[mt][kk], b[2], b[3]);
        }
      }
    }

    const int key0 = t * kKTile + tig * 2;
    uint32_t pf[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const bool v0 = key0 + n * 8 < len, v1 = key0 + n * 8 + 1 < len;
        sc[mt][n][0] = v0 ? sc[mt][n][0] * kScaleLog2 : -INFINITY;
        sc[mt][n][1] = v1 ? sc[mt][n][1] * kScaleLog2 : -INFINITY;
        sc[mt][n][2] = v0 ? sc[mt][n][2] * kScaleLog2 : -INFINITY;
        sc[mt][n][3] = v1 ? sc[mt][n][3] * kScaleLog2 : -INFINITY;
        mx0 = fmaxf(mx0, fmaxf(sc[mt][n][0], sc[mt][n][1]));
        mx1 = fmaxf(mx1, fmaxf(sc[mt][n][2], sc[mt][n][3]));
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      const float n0 = fmaxf(m[mt][0], mx0), n1 = fmaxf(m[mt][1], mx1);
      const float al0 = ex2(m[mt][0] - n0), al1 = ex2(m[mt][1] - n1);
      m[mt][0] = n0;
      m[mt][1] = n1;
      float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const float e0 = ex2(sc[mt][n][0] - n0), e1 = ex2(sc[mt][n][1] - n0);
        const float e2 = ex2(sc[mt][n][2] - n1), e3 = ex2(sc[mt][n][3] - n1);
        rs0 += e0 + e1;
        rs1 += e2 + e3;
        pf[mt][n >> 1][(n & 1) * 2] = pack_bf16(e0, e1);
        pf[mt][n >> 1][(n & 1) * 2 + 1] = pack_bf16(e2, e3);
      }
      l[mt][0] = l[mt][0] * al0 + rs0;
      l[mt][1] = l[mt][1] * al1 + rs1;
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        o[mt][d][0] *= al0;
        o[mt][d][1] *= al0;
        o[mt][d][2] *= al1;
        o[mt][d][3] *= al1;
      }
    }

#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b[4];
        ldsm_x4_t(b, sv + swz(16 * kk + 8 * (lq & 1) + lr, 2 * dp + (lq >> 1)));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma(o[mt][2 * dp], pf[mt][kk], b[0], b[1]);
          mma(o[mt][2 * dp + 1], pf[mt][kk], b[2], b[3]);
        }
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();

  __nv_bfloat16* og = static_cast<__nv_bfloat16*>(a.out) + static_cast<int64_t>(start) * a.out_stride + h * kDim;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    float l0 = l[mt][0], l1 = l[mt][1];
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    const int r0 = tile * kQTile + warp * 32 + mt * 16 + gid;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      const int col = d * 8 + tig * 2;
      if (r0 < len)
        *reinterpret_cast<uint32_t*>(og + static_cast<int64_t>(r0) * a.out_stride + col) =
            pack_bf16(o[mt][d][0] * i0, o[mt][d][1] * i0);
      if (r0 + 8 < len)
        *reinterpret_cast<uint32_t*>(og + static_cast<int64_t>(r0 + 8) * a.out_stride + col) =
            pack_bf16(o[mt][d][2] * i1, o[mt][d][3] * i1);
    }
  }
}

}  // namespace

cudaError_t vit_attn(const VitAttnArgs& a, cudaStream_t stream) {
  static bool configured = false;
  if (!configured) {
    cudaError_t e = cudaFuncSetAttribute(vit_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (e == cudaSuccess)
      e = cudaFuncSetAttribute(vit_attn_kernel_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (e == cudaSuccess)
      e = cudaFuncSetAttribute(vit_attn_kernel_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((a.max_len + kQTile - 1) / kQTile, 16, a.n_seq);
  if (a.version == 3) {  // INT8 QK^T: k mean, K quantization, then the attention
    static bool configured8 = false;
    if (!configured8) {
      const cudaError_t e =
          cudaFuncSetAttribute(vit_attn_kernel_i8, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem8);
      if (e != cudaSuccess) return e;
      configured8 = true;
    }
    if (!a.k8 || !a.k_scale || !a.k_mean) return cudaErrorInvalidValue;
    vit_k_mean_kernel<<<dim3(16, a.n_seq), 256, 0, stream>>>(a);
    vit_k_quant_kernel<<<dim3(a.max_tiles, 16, a.n_seq), 256, 0, stream>>>(a);
    vit_attn_kernel_i8<<<grid, kThreads, kSmem8, stream>>>(a);
    return cudaGetLastError();
  }
  if (a.version == 4) {
    vit_attn_kernel_v4<<<grid, kThreads4, kSmem, stream>>>(a);
    return cudaGetLastError();
  }
  if (a.version == 1)
    vit_attn_kernel<<<grid, kThreads, kSmem, stream>>>(a);
  else
    vit_attn_kernel_v2<<<grid, kThreads, kSmem, stream>>>(a);
  return cudaGetLastError();
}

}  // namespace chandra
