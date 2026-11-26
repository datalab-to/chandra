// Decode / verify attention, head_dim 256, GQA 4:1, paged BF16 KV, output gate.
//
// One CTA per (sequence, KV head, context split), 4 warps. The CTA's rows are
// the 4 query heads of its KV head times the n_q query tokens (row = t * 4 + h,
// at most 16), so a verify step over several tokens reads each KV byte once.
//   * Q (16 x 256) lives in registers as mma.m16n8k16 A fragments (64 regs).
//   * KV streams through a 3-stage cp.async ring of 32-key stages (K 16 KB +
//     V 16 KB), rows XOR-swizzled at 16 bytes so ldmatrix is conflict-free.
//     Keys past the context are zero-filled, so stale pages cannot inject NaN.
//   * Warp w owns keys 8w..8w+7 of a stage: S = Q K^T with 16 m16n8k16, online
//     softmax in base 2, then O += P V with 32 m16n8k8 (the S accumulators are
//     P's A fragments; V B fragments come from ldmatrix.trans).
//   * Warps merge in a fixed order through shared memory; splits merge through
//     a ticket, the last CTA in a fixed order. out = bf16(O / l) * sigmoid(gate).
#include "attn_decode.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>

namespace chandra {
namespace {

constexpr int kThreads = 128;
constexpr int kStageKeys = 32;
constexpr int kStages = 3;
constexpr int kRowBytes = kAttnHeadDim * 2;              // 512: one key's K or V row
constexpr int kHalfBytes = kStageKeys * kRowBytes;       // 16 KB of K (or V) per stage
constexpr int kStageBytes = 2 * kHalfBytes;              // 32 KB
constexpr int kSmemBytes = kStages * kStageBytes;        // 96 KB
constexpr int kRows = 16;
constexpr float kScaleLog2 = 0.0625f * 1.4426950408889634f;  // 1/sqrt(256) * log2(e)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                        const void* p) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(s));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                          const void* p) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(s));
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma1688(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Byte offset of 16-byte chunk c (0..31) of key row k inside a K or V half.
__device__ __forceinline__ int swz(int k, int c) { return k * kRowBytes + ((c ^ (k & 7)) << 4); }

template <int NQ>
__global__ void __launch_bounds__(kThreads, 1) attn_decode_kernel(const AttnDecodeArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ float s_m[4][kRows], s_l[4][kRows];
  __shared__ int s_last;

  const int split = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
  const int ctx = a.ctx_lens[b];
  const int n_stages = (ctx + kStageKeys - 1) / kStageKeys;
  const int splits = (n_stages + a.stages_per_split - 1) / a.stages_per_split;
  if (split >= splits) return;
  const int st0 = split * a.stages_per_split;
  const int st1 = min(n_stages, st0 + a.stages_per_split);

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gid = lane >> 2, tig = lane & 3;
  const __nv_bfloat16* kc = static_cast<const __nv_bfloat16*>(a.k_cache);
  const __nv_bfloat16* vc = static_cast<const __nv_bfloat16*>(a.v_cache);
  const int* bt = a.block_table + static_cast<size_t>(b) * a.block_table_stride;

  auto load_stage = [&](int st, int buf) {
    const int key0 = st * kStageKeys;
    const int page = bt[key0 / kAttnPageTokens];
    const size_t base = ((static_cast<size_t>(page) * kAttnKvHeads + kvh) * kAttnPageTokens +
                         key0 % kAttnPageTokens) * kAttnHeadDim;
    unsigned char* sk = smem + buf * kStageBytes;
    unsigned char* sv = sk + kHalfBytes;
#pragma unroll
    for (int i = threadIdx.x; i < kStageKeys * 32; i += kThreads) {
      const int k = i >> 5, c = i & 31;
      const bool valid = key0 + k < ctx;
      const size_t off = base + static_cast<size_t>(valid ? k : 0) * kAttnHeadDim + c * 8;
      cp_async16(sk + swz(k, c), kc + off, valid);
      cp_async16(sv + swz(k, c), vc + off, valid);
    }
  };

  // Start the KV stream before reading Q.
  load_stage(st0, 0);
  cp_async_commit();
  if (st0 + 1 < st1) load_stage(st0 + 1, 1);
  cp_async_commit();

  // Q A fragments: rows gid and gid + 8, columns kk*16 + 2*tig (+8).
  uint32_t qf[16][4];
  {
    const __nv_bfloat16* q = static_cast<const __nv_bfloat16*>(a.q);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = gid + half * 8, t = r >> 2, h = kvh * 4 + (r & 3);
      const bool live = t < NQ;
      const __nv_bfloat16* qr =
          q + (static_cast<int64_t>(b) * NQ + (live ? t : 0)) * a.q_token_stride + h * a.q_head_stride + tig * 2;
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        qf[kk][half] = live ? *reinterpret_cast<const uint32_t*>(qr + kk * 16) : 0u;
        qf[kk][half + 2] = live ? *reinterpret_cast<const uint32_t*>(qr + kk * 16 + 8) : 0u;
      }
    }
  }

  float o[32][4];
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  // Last key position each of this lane's rows may see (verify steps are causal).
  const int t0 = gid >> 2, t1 = (gid + 8) >> 2;
  const int lim0 = ctx - NQ + t0, lim1 = ctx - NQ + t1;

  const int my_key = warp * 8 + (lane & 7);  // ldmatrix row of this lane
  const int lq = lane >> 3;                  // ldmatrix matrix of this lane

  for (int st = st0; st < st1; ++st) {
    if (st + 2 < st1) load_stage(st + 2, (st - st0 + 2) % kStages);
    cp_async_commit();
    cp_async_wait<2>();
    __syncthreads();
    const unsigned char* sk = smem + ((st - st0) % kStages) * kStageBytes;
    const unsigned char* sv = sk + kHalfBytes;

    // S = Q K^T for this warp's 8 keys.
    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kk2 = 0; kk2 < 8; ++kk2) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4(b0, b1, b2, b3, sk + swz(my_key, kk2 * 4 + lq));
      mma16816(s, qf[2 * kk2], b0, b1);
      mma16816(s, qf[2 * kk2 + 1], b2, b3);
    }

    // Online softmax (base 2). Lane holds rows gid (s0, s1) and gid+8 (s2, s3).
    const int p0 = st * kStageKeys + warp * 8 + tig * 2;
    const float x0 = p0 <= lim0 ? s[0] * kScaleLog2 : -INFINITY;
    const float x1 = p0 + 1 <= lim0 ? s[1] * kScaleLog2 : -INFINITY;
    const float x2 = p0 <= lim1 ? s[2] * kScaleLog2 : -INFINITY;
    const float x3 = p0 + 1 <= lim1 ? s[3] * kScaleLog2 : -INFINITY;
    float mx0 = fmaxf(x0, x1), mx1 = fmaxf(x2, x3);
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    // A row that has seen no visible key yet keeps m = -inf; use 0 as its base.
    const float b0f = n0 == -INFINITY ? 0.f : n0, b1f = n1 == -INFINITY ? 0.f : n1;
    const float al0 = fast_exp2(m0 - b0f), al1 = fast_exp2(m1 - b1f);
    const float e0 = fast_exp2(x0 - b0f), e1 = fast_exp2(x1 - b0f);
    const float e2 = fast_exp2(x2 - b1f), e3 = fast_exp2(x3 - b1f);
    l0 = l0 * al0 + e0 + e1;
    l1 = l1 * al1 + e2 + e3;
    m0 = n0;
    m1 = n1;
    if (al0 != 1.f || al1 != 1.f) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        o[i][0] *= al0;
        o[i][1] *= al0;
        o[i][2] *= al1;
        o[i][3] *= al1;
      }
    }
    const uint32_t pa0 = pack_bf16(e0, e1), pa1 = pack_bf16(e2, e3);

    // O += P V: V B fragments for 4 n8 dim tiles per ldmatrix.x4.trans.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t v0, v1, v2, v3;
      ldsm_x4_t(v0, v1, v2, v3, sv + swz(my_key, j * 4 + lq));
      mma1688(o[4 * j + 0], pa0, pa1, v0);
      mma1688(o[4 * j + 1], pa0, pa1, v1);
      mma1688(o[4 * j + 2], pa0, pa1, v2);
      mma1688(o[4 * j + 3], pa0, pa1, v3);
    }
    __syncthreads();
  }
  cp_async_wait<0>();

  // Row sums over the quad.
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  if (tig == 0) {
    s_m[warp][gid] = m0;
    s_m[warp][gid + 8] = m1;
    s_l[warp][gid] = l0;
    s_l[warp][gid + 8] = l1;
  }
  __syncthreads();

  // Merge the 4 warps: scale each warp's O to the rows' common max, sum in order.
  float* so = reinterpret_cast<float*>(smem);  // [4 warps][16 rows][256] = 64 KB
  {
    float mr0 = -INFINITY, mr1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      mr0 = fmaxf(mr0, s_m[w][gid]);
      mr1 = fmaxf(mr1, s_m[w][gid + 8]);
    }
    const float f0 = m0 == -INFINITY ? 0.f : fast_exp2(m0 - mr0);
    const float f1 = m1 == -INFINITY ? 0.f : fast_exp2(m1 - mr1);
    float* mine = so + warp * kRows * kAttnHeadDim;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int d = i * 8 + tig * 2;
      *reinterpret_cast<float2*>(mine + gid * kAttnHeadDim + d) = make_float2(o[i][0] * f0, o[i][1] * f0);
      *reinterpret_cast<float2*>(mine + (gid + 8) * kAttnHeadDim + d) = make_float2(o[i][2] * f1, o[i][3] * f1);
    }
  }
  __syncthreads();

  // This CTA's (O, m, l) per row: thread handles 32 consecutive dims of one row.
  constexpr int kLive = 4 * NQ;  // rows that are real queries
  const int r = threadIdx.x >> 3, d0 = (threadIdx.x & 7) * 32;
  float mr = -INFINITY;
#pragma unroll
  for (int w = 0; w < 4; ++w) mr = fmaxf(mr, s_m[w][r]);
  float lr = 0.f;
#pragma unroll
  for (int w = 0; w < 4; ++w) lr += s_m[w][r] == -INFINITY ? 0.f : s_l[w][r] * fast_exp2(s_m[w][r] - mr);
  float acc[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) v += so[(w * kRows + r) * kAttnHeadDim + d0 + i];
    acc[i] = v;
  }

  const int t = r >> 2, h = kvh * 4 + (r & 3);
  auto finish = [&](const float (&ov)[32], float l) {
    if (r >= kLive) return;
    const int64_t tok = static_cast<int64_t>(b) * NQ + t;
    const __nv_bfloat16* g =
        static_cast<const __nv_bfloat16*>(a.gate) + tok * a.gate_token_stride + h * a.gate_head_stride + d0;
    __nv_bfloat16* out = static_cast<__nv_bfloat16*>(a.out) + tok * a.out_token_stride + h * a.out_head_stride + d0;
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
      const uint4 gv = *reinterpret_cast<const uint4*>(g + i);
      const __nv_bfloat16* gb = reinterpret_cast<const __nv_bfloat16*>(&gv);
      uint4 res;
      uint32_t* rp = reinterpret_cast<uint32_t*>(&res);
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        // vLLM rounds the attention output to BF16 before the gate.
        const float o0 = __bfloat162float(__float2bfloat16_rn(ov[i + j] * inv));
        const float o1 = __bfloat162float(__float2bfloat16_rn(ov[i + j + 1] * inv));
        const float s0 = 1.f / (1.f + __expf(-__bfloat162float(gb[j])));
        const float s1 = 1.f / (1.f + __expf(-__bfloat162float(gb[j + 1])));
        rp[j / 2] = pack_bf16(o0 * s0, o1 * s1);
      }
      *reinterpret_cast<uint4*>(out + i) = res;
    }
  };

  if (splits == 1) {
    finish(acc, lr);
    return;
  }

  // Several splits: write this split's partial, the last CTA merges in order.
  const size_t unit = (static_cast<size_t>(b) * kAttnKvHeads + kvh) * a.max_splits;
  float* po = a.partial_o + ((unit + split) * kRows + r) * kAttnHeadDim + d0;
#pragma unroll
  for (int i = 0; i < 32; i += 4) *reinterpret_cast<float4*>(po + i) = make_float4(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
  if ((threadIdx.x & 7) == 0) {
    float* pml = a.partial_ml + ((unit + split) * kRows + r) * 2;
    pml[0] = mr;
    pml[1] = lr;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned* tk = a.tickets + static_cast<size_t>(b) * kAttnKvHeads + kvh;
    s_last = atomicAdd(tk, 1u) == static_cast<unsigned>(splits - 1);
    if (s_last) *tk = 0;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  float mall = -INFINITY;
  for (int sp = 0; sp < splits; ++sp)
    mall = fmaxf(mall, __ldcg(a.partial_ml + ((unit + sp) * kRows + r) * 2));
  float lall = 0.f;
  float tot[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) tot[i] = 0.f;
  for (int sp = 0; sp < splits; ++sp) {
    const float ms = __ldcg(a.partial_ml + ((unit + sp) * kRows + r) * 2);
    const float f = ms == -INFINITY ? 0.f : fast_exp2(ms - mall);
    lall += f * __ldcg(a.partial_ml + ((unit + sp) * kRows + r) * 2 + 1);
    const float* ps = a.partial_o + ((unit + sp) * kRows + r) * kAttnHeadDim + d0;
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
      const float4 v = __ldcg(reinterpret_cast<const float4*>(ps + i));
      tot[i] += f * v.x;
      tot[i + 1] += f * v.y;
      tot[i + 2] += f * v.z;
      tot[i + 3] += f * v.w;
    }
  }
  finish(tot, lall);
}

template <int NQ>
cudaError_t launch(const AttnDecodeArgs& a, cudaStream_t stream) {
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(attn_decode_kernel<NQ>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               kSmemBytes);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid(a.max_splits, kAttnKvHeads, a.B);
  attn_decode_kernel<NQ><<<grid, kThreads, kSmemBytes, stream>>>(a);
  return cudaGetLastError();
}

}  // namespace

void attn_decode_split(int B, int max_ctx, int sms, int* stages_per_split, int* max_splits) {
  // About two CTAs' worth of work per SM, never fewer than 4 stages per CTA.
  const int stages = (max_ctx + kStageKeys - 1) / kStageKeys;
  const int units = B * kAttnKvHeads;
  const int want = std::max(1, (2 * sms + units - 1) / units);  // splits per (seq, head)
  const int sps = std::max(4, (stages + want - 1) / want);
  *stages_per_split = sps;
  *max_splits = std::max(1, (stages + sps - 1) / sps);
}

cudaError_t attn_decode(const AttnDecodeArgs& a, cudaStream_t stream) {
  switch (a.n_q) {
    case 1: return launch<1>(a, stream);
    case 2: return launch<2>(a, stream);
    case 3: return launch<3>(a, stream);
    case 4: return launch<4>(a, stream);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace chandra
