// Small per-step ops around the GEMMs: residual add + RMSNorm, SiLU-and-mul, and
// q/k RMSNorm + partial mRoPE + paged KV write for the full-attention layers.
// All are bandwidth-trivial at decode batch sizes; they exist to keep a step's data
// on the GPU in the engine's layouts with the model's exact rounding points.
#include "small_ops.h"

#include <cuda_bf16.h>

#include <algorithm>

namespace chandra {
namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ uint4 pack8(const float* f) {
  unsigned w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    w[i] = *reinterpret_cast<const unsigned*>(&p);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}
__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// One warp per row: 2,560 = 32 lanes x 10 chunks of 8.
__global__ void __launch_bounds__(32) rmsnorm_add_kernel(const bf16* __restrict__ x, const bf16* __restrict__ res,
                                                         bf16* __restrict__ res_out, bf16* __restrict__ out,
                                                         const bf16* __restrict__ w) {
  constexpr int kChunks = kHidden / 8 / 32;
  const int row = blockIdx.x, lane = threadIdx.x;
  const size_t base = static_cast<size_t>(row) * kHidden;
  float v[kChunks][8];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < kChunks; ++i) {
    const int c = (lane + 32 * i) * 8;
    float r[8];
    unpack8(*reinterpret_cast<const uint4*>(res + base + c), r);
    if (x) {
      float xx[8];
      unpack8(*reinterpret_cast<const uint4*>(x + base + c), xx);
#pragma unroll
      for (int j = 0; j < 8; ++j) r[j] = round_bf16(r[j] + xx[j]);  // the residual stream is BF16
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[i][j] = r[j];
      ss = fmaf(r[j], r[j], ss);
    }
    if (res_out) *reinterpret_cast<uint4*>(res_out + base + c) = pack8(r);
  }
  const float rstd = rsqrtf(wsum(ss) * (1.f / kHidden) + 1e-6f);
#pragma unroll
  for (int i = 0; i < kChunks; ++i) {
    const int c = (lane + 32 * i) * 8;
    float ww[8], o[8];
    unpack8(*reinterpret_cast<const uint4*>(w + c), ww);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = v[i][j] * rstd * (1.f + ww[j]);
    *reinterpret_cast<uint4*>(out + base + c) = pack8(o);
  }
}

__global__ void silu_mul_kernel(const bf16* __restrict__ gu, bf16* __restrict__ out, int M, int F) {
  const size_t n8 = static_cast<size_t>(M) * F / 8;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n8;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t m = i / (F / 8), c = (i % (F / 8)) * 8;
    float g[8], u[8], o[8];
    unpack8(*reinterpret_cast<const uint4*>(gu + m * 2 * F + c), g);
    unpack8(*reinterpret_cast<const uint4*>(gu + m * 2 * F + F + c), u);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = g[j] / (1.f + expf(-g[j])) * u[j];
    *reinterpret_cast<uint4*>(out + m * F + c) = pack8(o);
  }
}

// grid (M, 3), 8 warps: y = 0, 1: q heads 8y..8y+7; y = 2: warps 0-3 k heads, 4-7 v heads.
// A warp holds one head: lane l has dims 8l..8l+7. The rotary half (dims 0..63) is lanes
// 0..7, and dim d's rotate-half partner d +- 32 is the same element of lane l ^ 4.
__global__ void __launch_bounds__(256) qk_norm_rope_kernel(const QkNormRopeArgs a) {
  const int m = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bf16* row = static_cast<const bf16*>(a.qkv) + static_cast<int64_t>(m) * a.qkv_stride;
  const bool is_q = blockIdx.y < 2;
  const bool is_v = !is_q && warp >= 4;
  const int h = is_q ? blockIdx.y * 8 + warp : (warp & 3);
  const bf16* src = is_q ? row + h * 512 : is_v ? row + 9216 + h * 256 : row + 8192 + h * 256;
  float x[8];
  unpack8(*reinterpret_cast<const uint4*>(src + lane * 8), x);
  if (!is_v) {
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) ss = fmaf(x[j], x[j], ss);
    const float rstd = rsqrtf(wsum(ss) * (1.f / 256.f) + 1e-6f);
    float w[8];
    unpack8(*reinterpret_cast<const uint4*>(static_cast<const bf16*>(is_q ? a.q_norm_w : a.k_norm_w) + lane * 8), w);
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = round_bf16(x[j] * rstd * (1.f + w[j]));  // the norm's output is BF16
    // Partial rotary on dims 0..63, rotate-half pairs (d, d + 32), frequency f = d % 32
    // taking its position from axis f % 3 (t, h, w).
    float partner[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) partner[j] = __shfl_xor_sync(0xffffffffu, x[j], 4);
    if (lane < 8) {
      const int* p = a.pos + m * 3;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int d = lane * 8 + j, f = d & 31;
        const float inv_freq = 1.f / powf(a.rope_theta, static_cast<float>(2 * f) / 64.f);
        const float ang = static_cast<float>(p[f % 3]) * inv_freq;
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float rot = d < 32 ? -partner[j] : partner[j];
        x[j] = x[j] * cs + rot * sn;
      }
    }
  }
  if (!is_q && a.k_scale) {  // INT8 cache: symmetric per (token, head)
    float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fabsf(x[j]));
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float scale = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / scale;
    unsigned w[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int q = max(-127, min(127, __float2int_rn(x[j] * inv)));
      w[j >> 2] |= (static_cast<unsigned>(q) & 0xffu) << (8 * (j & 3));
    }
    const size_t tok = (static_cast<size_t>(a.page[m]) * 4 + h) * 64 + a.slot[m];
    unsigned char* cache = static_cast<unsigned char*>(is_v ? a.v_cache : a.k_cache);
    *reinterpret_cast<uint2*>(cache + tok * 256 + lane * 8) = make_uint2(w[0], w[1]);
    if (lane == 0) (is_v ? a.v_scale : a.k_scale)[tok] = scale;
    return;
  }
  const uint4 packed = pack8(x);
  if (is_q) {
    *reinterpret_cast<uint4*>(static_cast<bf16*>(a.q_out) + (static_cast<size_t>(m) * 16 + h) * 256 + lane * 8) = packed;
  } else {
    bf16* cache = static_cast<bf16*>(is_v ? a.v_cache : a.k_cache);
    const size_t off = ((static_cast<size_t>(a.page[m]) * 4 + h) * 64 + a.slot[m]) * 256 + lane * 8;
    *reinterpret_cast<uint4*>(cache + off) = packed;
  }
}

__device__ __forceinline__ float bf2f(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float silu_fast(float x) { return x / (1.f + __expf(-x)); }

__global__ void __launch_bounds__(256) conv_silu_prefill_kernel(const bf16* __restrict__ x, int64_t xs,
                                                                 const bf16* __restrict__ w, bf16* __restrict__ out,
                                                                 int n) {
  const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
  if (c >= 8192) return;
  float acc = bf2f(w + c * 4 + 3) * bf2f(x + static_cast<int64_t>(t) * xs + c);
#pragma unroll
  for (int j = 1; j <= 3; ++j) {
    const float xv = t - j >= 0 ? bf2f(x + static_cast<int64_t>(t - j) * xs + c) : 0.f;
    acc += bf2f(w + c * 4 + 3 - j) * xv;
  }
  out[static_cast<int64_t>(t) * 8192 + c] = __float2bfloat16_rn(silu_fast(acc));
}

// One warp per (row, head): lane l holds dims 4l .. 4l + 3.
__global__ void __launch_bounds__(256) gated_rmsnorm_kernel(const bf16* __restrict__ o, const bf16* __restrict__ z,
                                                             int64_t zs, const bf16* __restrict__ w,
                                                             bf16* __restrict__ out, int n) {
  const int lane = threadIdx.x & 31, unit = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (unit >= n * 32) return;
  const int row = unit >> 5, h = unit & 31;
  const bf16* op = o + static_cast<int64_t>(unit) * 128 + lane * 4;
  float ov[4], ss = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    ov[j] = bf2f(op + j);
    ss = fmaf(ov[j], ov[j], ss);
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float rstd = rsqrtf(ss * (1.f / 128.f) + 1e-6f);
  const bf16* zr = z + static_cast<int64_t>(row) * zs + h * 128 + lane * 4;
  const bf16* wr = w + lane * 4;
  bf16* dst = out + static_cast<int64_t>(row) * 4096 + h * 128 + lane * 4;
  const __nv_bfloat162 r01 = __floats2bfloat162_rn(ov[0] * rstd * bf2f(wr) * silu_fast(bf2f(zr)),
                                                   ov[1] * rstd * bf2f(wr + 1) * silu_fast(bf2f(zr + 1)));
  const __nv_bfloat162 r23 = __floats2bfloat162_rn(ov[2] * rstd * bf2f(wr + 2) * silu_fast(bf2f(zr + 2)),
                                                   ov[3] * rstd * bf2f(wr + 3) * silu_fast(bf2f(zr + 3)));
  uint2 st;
  st.x = *reinterpret_cast<const unsigned*>(&r01);
  st.y = *reinterpret_cast<const unsigned*>(&r23);
  *reinterpret_cast<uint2*>(dst) = st;
}

}  // namespace

cudaError_t conv_silu_prefill(const void* x, int64_t x_stride, const void* w, void* out, int n,
                              cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  conv_silu_prefill_kernel<<<dim3(32, n), 256, 0, stream>>>(static_cast<const bf16*>(x), x_stride,
                                                            static_cast<const bf16*>(w), static_cast<bf16*>(out), n);
  return cudaGetLastError();
}

cudaError_t gated_rmsnorm(const void* o, const void* z, int64_t z_stride, const void* w, void* out, int n,
                          cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  gated_rmsnorm_kernel<<<(n * 32 + 7) / 8, 256, 0, stream>>>(static_cast<const bf16*>(o), static_cast<const bf16*>(z),
                                                             z_stride, static_cast<const bf16*>(w),
                                                             static_cast<bf16*>(out), n);
  return cudaGetLastError();
}

cudaError_t rmsnorm_add(const void* x, const void* residual, void* residual_out, void* out, const void* w, int M,
                        cudaStream_t stream) {
  rmsnorm_add_kernel<<<M, 32, 0, stream>>>(static_cast<const bf16*>(x), static_cast<const bf16*>(residual),
                                           static_cast<bf16*>(residual_out), static_cast<bf16*>(out),
                                           static_cast<const bf16*>(w));
  return cudaGetLastError();
}

cudaError_t silu_mul(const void* gate_up, void* out, int M, cudaStream_t stream, int F) {
  if (F % 8) return cudaErrorInvalidValue;
  const int blocks = std::min(1024, (M * F / 8 + 255) / 256);
  silu_mul_kernel<<<blocks, 256, 0, stream>>>(static_cast<const bf16*>(gate_up), static_cast<bf16*>(out), M, F);
  return cudaGetLastError();
}

cudaError_t qk_norm_rope_kv(const QkNormRopeArgs& a, cudaStream_t stream) {
  qk_norm_rope_kernel<<<dim3(a.M, 3), 256, 0, stream>>>(a);
  return cudaGetLastError();
}

}  // namespace chandra
