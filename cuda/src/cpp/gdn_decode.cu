// One fused Gated DeltaNet decode step per layer: causal conv (ring state) + SiLU,
// l2norm of q and k, the gates, the delta rule on the recurrent state, o, and the
// gated RMSNorm. The state moves once: read once and written once per step.
//
// One CTA (8 warps) per (sequence, value head) owns the head's 128 x 128 state.
// Each lane owns one 128-bit vector of a state row (4 FP32 or 8 FP16 elements);
// every state load of a thread is issued first (ld.global.cs.v4 = LDG.E.EF.128),
// then per row two dot products reduced with xor shuffles, v' = beta (v - decay S k),
// o = decay S q + v' (k . q) and S' = decay S + v' k^T from the same registers, and
// one 128-bit streaming store per row.
#include "gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace chandra {
namespace {

using bf16 = __nv_bfloat16;

template <typename ST>
struct SV;
template <>
struct SV<float> {
  static constexpr int E = 4;
  __device__ static void unpack(const uint4& u, float* f) {
    f[0] = __uint_as_float(u.x);
    f[1] = __uint_as_float(u.y);
    f[2] = __uint_as_float(u.z);
    f[3] = __uint_as_float(u.w);
  }
  __device__ static uint4 pack(const float* f) {
    return make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]), __float_as_uint(f[3]));
  }
};
template <>
struct SV<__half> {
  static constexpr int E = 8;
  __device__ static void unpack(const uint4& u, float* f) {
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 t = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
      f[2 * i] = t.x;
      f[2 * i + 1] = t.y;
    }
  }
  __device__ static uint4 pack(const float* f) {
    unsigned w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __half2 p = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
      w[i] = *reinterpret_cast<const unsigned*>(&p);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
  }
};

template <int LPR>
__device__ __forceinline__ float rsum(float v) {
#pragma unroll
  for (int off = LPR / 2; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float bf(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

template <typename ST>
__global__ void __launch_bounds__(256, 2) gdn_decode_kernel(const GdnDecodeArgs a) {
  constexpr int E = SV<ST>::E;
  constexpr int LPR = kGdnDim / E;  // lanes per state row
  constexpr int RPI = 32 / LPR;     // rows per warp instruction
  constexpr int RPW = kGdnDim / 8;  // rows per warp (16)
  constexpr int IT = RPW / RPI;     // 128-bit state vectors per thread
  __shared__ float xs[3 * kGdnDim];  // conv + SiLU output: q | k | v of this head
  __shared__ float o_s[kGdnDim];

  const int hv = blockIdx.x, n = blockIdx.y, h = hv >> 1;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int sub = lane / LPR, kl = lane % LPR;
  const int rbase = warp * RPW + sub;
  constexpr size_t kHead = static_cast<size_t>(kGdnDim) * kGdnDim;
  const ST* s_in = static_cast<const ST*>(a.state) + (static_cast<size_t>(a.slot_in[n]) * kGdnValueHeads + hv) * kHead;
  ST* s_out = static_cast<ST*>(a.state) + (static_cast<size_t>(a.slot_out[n]) * kGdnValueHeads + hv) * kHead;

  // 1. All state loads in flight before anything else.
  uint4 sv[IT];
#pragma unroll
  for (int i = 0; i < IT; ++i)
    sv[i] = __ldcs(reinterpret_cast<const uint4*>(s_in + static_cast<size_t>(rbase + i * RPI) * kGdnDim + kl * E));

  // 2. Causal conv (width 4) + SiLU for q, k of key head h and v of value head hv.
  //    out[t] = sum_j w[j] x[t - 3 + j]; earlier x come from the ring, row = position % ring.
  {
    const bf16* x = static_cast<const bf16*>(a.qkv) + static_cast<int64_t>(n) * a.qkv_stride;
    bf16* ring = static_cast<bf16*>(a.conv_ring) + static_cast<size_t>(a.conv_slot[n]) * a.ring * kGdnChannels;
    const bf16* cw = static_cast<const bf16*>(a.conv_w);
    const int p = a.pos[n];
    for (int i = threadIdx.x; i < 3 * kGdnDim; i += blockDim.x) {
      const int ch = i < kGdnDim ? h * kGdnDim + i
                     : i < 2 * kGdnDim ? 2048 + h * kGdnDim + (i - kGdnDim)
                                       : 4096 + hv * kGdnDim + (i - 2 * kGdnDim);
      const bf16 xt = x[ch];
      float acc = __bfloat162float(cw[ch * 4 + 3]) * __bfloat162float(xt);
#pragma unroll
      for (int j = 1; j <= 3; ++j)
        if (p - j >= 0) acc += __bfloat162float(cw[ch * 4 + 3 - j]) * bf(ring + static_cast<size_t>((p - j) % a.ring) * kGdnChannels + ch);
      xs[i] = round_bf16(silu(acc));  // the conv output is BF16 in vLLM
      // q and k channels are shared by the two value heads of a key head: the even one writes them.
      if (i >= 2 * kGdnDim || (hv & 1) == 0) ring[static_cast<size_t>(p % a.ring) * kGdnChannels + ch] = xt;
    }
  }
  __syncthreads();

  // 3. l2norm of q and k (q also scaled by 128^-1/2), k . q, and the gates.
  float q[E], k[E];
#pragma unroll
  for (int e = 0; e < E; ++e) {
    q[e] = xs[kl * E + e];
    k[e] = xs[kGdnDim + kl * E + e];
  }
  float sq = 0.f, sk = 0.f;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    sq = fmaf(q[e], q[e], sq);
    sk = fmaf(k[e], k[e], sk);
  }
  sq = rsum<LPR>(sq);
  sk = rsum<LPR>(sk);
  const float rq = rsqrtf(sq + 1e-6f) * 0.08838834764831845f, rk = rsqrtf(sk + 1e-6f);
  float kq = 0.f;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    q[e] *= rq;
    k[e] *= rk;
    kq = fmaf(q[e], k[e], kq);
  }
  kq = rsum<LPR>(kq);
  const bf16* ap = static_cast<const bf16*>(a.a) + static_cast<int64_t>(n) * a.ab_stride;
  const bf16* bp = static_cast<const bf16*>(a.b) + static_cast<int64_t>(n) * a.ab_stride;
  const float xa = bf(ap + hv) + bf(static_cast<const bf16*>(a.dt_bias) + hv);
  const float sp = xa <= 20.f ? log1pf(__expf(xa)) : xa;
  const float decay = __expf(-__expf(a.A_log[hv]) * sp);
  const float beta = 1.f / (1.f + __expf(-bf(bp + hv)));

  // 4. The delta rule, row by row, from the registers loaded in step 1.
#pragma unroll
  for (int i = 0; i < IT; ++i) {
    const int r = rbase + i * RPI;
    float s[E];
    SV<ST>::unpack(sv[i], s);
    float dk = 0.f, dq = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
      dk = fmaf(s[e], k[e], dk);
      dq = fmaf(s[e], q[e], dq);
    }
    dk = rsum<LPR>(dk);
    dq = rsum<LPR>(dq);
    const float vp = beta * (xs[2 * kGdnDim + r] - decay * dk);
    const float o = fmaf(vp, kq, decay * dq);
#pragma unroll
    for (int e = 0; e < E; ++e) s[e] = fmaf(decay, s[e], vp * k[e]);
    __stcs(reinterpret_cast<uint4*>(s_out + static_cast<size_t>(r) * kGdnDim + kl * E), SV<ST>::pack(s));
    if (kl == 0) o_s[r] = round_bf16(o);  // vLLM rounds o to BF16 before the norm
  }
  __syncthreads();

  // 5. Gated RMSNorm over the head: out = o * rsqrt(mean(o^2) + eps) * w * SiLU(z).
  if (warp == 0) {
    float ov[4], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ov[j] = o_s[lane * 4 + j];
      ss = fmaf(ov[j], ov[j], ss);
    }
    ss = rsum<32>(ss);
    const float rstd = rsqrtf(ss * (1.f / kGdnDim) + 1e-6f);
    const bf16* zr = static_cast<const bf16*>(a.z) + static_cast<int64_t>(n) * a.z_stride + hv * kGdnDim + lane * 4;
    const bf16* w = static_cast<const bf16*>(a.norm_w) + lane * 4;
    bf16* out = static_cast<bf16*>(a.out) + static_cast<int64_t>(n) * a.out_stride + hv * kGdnDim + lane * 4;
    __nv_bfloat162 r01 = __floats2bfloat162_rn(ov[0] * rstd * bf(w) * silu(bf(zr)),
                                               ov[1] * rstd * bf(w + 1) * silu(bf(zr + 1)));
    __nv_bfloat162 r23 = __floats2bfloat162_rn(ov[2] * rstd * bf(w + 2) * silu(bf(zr + 2)),
                                               ov[3] * rstd * bf(w + 3) * silu(bf(zr + 3)));
    uint2 st;
    st.x = *reinterpret_cast<const unsigned*>(&r01);
    st.y = *reinterpret_cast<const unsigned*>(&r23);
    *reinterpret_cast<uint2*>(out) = st;
  }
}

}  // namespace

cudaError_t gdn_decode(const GdnDecodeArgs& a, cudaStream_t stream) {
  if (a.ring < 4) return cudaErrorInvalidValue;  // the slot written must differ from the 3 read
  const dim3 grid(kGdnValueHeads, a.B);
  if (a.state_fp16)
    gdn_decode_kernel<__half><<<grid, 256, 0, stream>>>(a);
  else
    gdn_decode_kernel<float><<<grid, 256, 0, stream>>>(a);
  return cudaGetLastError();
}

}  // namespace chandra
