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

}  // namespace

cudaError_t vit_attn(const VitAttnArgs& a, cudaStream_t stream) {
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(vit_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((a.max_len + kQTile - 1) / kQTile, 16, a.n_seq);
  vit_attn_kernel<<<grid, kThreads, kSmem, stream>>>(a);
  return cudaGetLastError();
}

}  // namespace chandra
