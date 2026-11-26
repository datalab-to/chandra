#include <cstdint>
#include <stdexcept>

#include <nanobind/nanobind.h>

#include "attn_decode.h"
#include "decode_gemm.h"
#include "gdn_decode.h"
#include "small_ops.h"
#include "vit_attn.h"

namespace nb = nanobind;

namespace {

void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "CUDA kernels for Chandra OCR. Pointers are device addresses (torch data_ptr()).";

  m.def(
      "decode_gemm",
      [](uintptr_t x, uintptr_t wp, uintptr_t y, uintptr_t workspace, uintptr_t counters, int M, int N,
         int K, int sms, uintptr_t stream, int epilogue) {
        check(chandra::decode_gemm(reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(wp),
                                   reinterpret_cast<void*>(y), reinterpret_cast<float*>(workspace),
                                   reinterpret_cast<unsigned*>(counters), M, N, K, sms,
                                   reinterpret_cast<cudaStream_t>(stream), epilogue));
      },
      nb::arg("x"), nb::arg("wp"), nb::arg("y"), nb::arg("workspace"), nb::arg("counters"), nb::arg("M"),
      nb::arg("N"), nb::arg("K"), nb::arg("sms"), nb::arg("stream"), nb::arg("epilogue") = 0);

  m.def(
      "decode_gemm_plan",
      [](int N, int K, int sms) {
        const chandra::DecodeGemmPlan p = chandra::decode_gemm_plan(N, K, sms);
        return nb::make_tuple(p.row_groups, p.splits);
      },
      nb::arg("N"), nb::arg("K"), nb::arg("sms"));
  m.def("decode_gemm_fp8", [](uintptr_t x, uintptr_t wp, uintptr_t wscale, uintptr_t y, uintptr_t workspace,
                              uintptr_t counters, int M, int N, int K, int sms, uintptr_t stream, int epilogue) {
    check(chandra::decode_gemm_fp8(reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(wp),
                                   reinterpret_cast<const float*>(wscale), reinterpret_cast<void*>(y),
                                   reinterpret_cast<float*>(workspace), reinterpret_cast<unsigned*>(counters), M, N,
                                   K, sms, reinterpret_cast<cudaStream_t>(stream), epilogue));
  }, nb::arg("x"), nb::arg("wp"), nb::arg("wscale"), nb::arg("y"), nb::arg("workspace"), nb::arg("counters"),
     nb::arg("M"), nb::arg("N"), nb::arg("K"), nb::arg("sms"), nb::arg("stream"), nb::arg("epilogue") = 0);
  m.def(
      "decode_gemm_argmax",
      [](uintptr_t x, uintptr_t wp, uintptr_t keys, int M, int N, int K, int sms, uintptr_t stream) {
        check(chandra::decode_gemm_argmax(reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(wp),
                                          reinterpret_cast<unsigned long long*>(keys), M, N, K, sms,
                                          reinterpret_cast<cudaStream_t>(stream)));
      });
  m.def("decode_gemm_fp4", [](uintptr_t x, uintptr_t wp, uintptr_t sc, uintptr_t y, uintptr_t workspace,
                              uintptr_t counters, int M, int N, int K, int sms, uintptr_t stream, int epilogue) {
    check(chandra::decode_gemm_fp4(reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(wp),
                                   reinterpret_cast<const void*>(sc), reinterpret_cast<void*>(y),
                                   reinterpret_cast<float*>(workspace), reinterpret_cast<unsigned*>(counters), M, N,
                                   K, sms, reinterpret_cast<cudaStream_t>(stream), epilogue));
  }, nb::arg("x"), nb::arg("wp"), nb::arg("sc"), nb::arg("y"), nb::arg("workspace"), nb::arg("counters"),
     nb::arg("M"), nb::arg("N"), nb::arg("K"), nb::arg("sms"), nb::arg("stream"), nb::arg("epilogue") = 0);
  m.def("lm_head_bound_i8", [](uintptr_t x, uintptr_t wp, uintptr_t wscale, uintptr_t rowc, uintptr_t xnorm,
                               uintptr_t tile_ub, uintptr_t lb_key, int M, int N, int K, uintptr_t stream) {
    check(chandra::lm_head_bound_i8(reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(wp),
                                    reinterpret_cast<const float*>(wscale), reinterpret_cast<const float*>(rowc),
                                    reinterpret_cast<const float*>(xnorm), reinterpret_cast<float*>(tile_ub),
                                    reinterpret_cast<unsigned*>(lb_key), M, N, K,
                                    reinterpret_cast<cudaStream_t>(stream)));
  });
  m.def("lm_head_recheck", [](uintptr_t x, uintptr_t wp, uintptr_t tile_ub, uintptr_t lb_key, uintptr_t keys, int M,
                              int N, int K, uintptr_t stream) {
    check(chandra::lm_head_recheck(reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(wp),
                                   reinterpret_cast<const float*>(tile_ub), reinterpret_cast<const unsigned*>(lb_key),
                                   reinterpret_cast<unsigned long long*>(keys), M, N, K,
                                   reinterpret_cast<cudaStream_t>(stream)));
  });
  m.def(
      "decode_gemm_argmax_fp8",
      [](uintptr_t x, uintptr_t wp, uintptr_t wscale, uintptr_t keys, int M, int N, int K, uintptr_t stream) {
        check(chandra::decode_gemm_argmax_fp8(reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(wp),
                                              reinterpret_cast<const float*>(wscale),
                                              reinterpret_cast<unsigned long long*>(keys), M, N, K,
                                              reinterpret_cast<cudaStream_t>(stream)));
      });
  m.def("decode_gemm_workspace_bytes", &chandra::decode_gemm_workspace_bytes, nb::arg("N"), nb::arg("K"),
        nb::arg("sms"));
  m.def("decode_gemm_counters", &chandra::decode_gemm_counters, nb::arg("N"), nb::arg("K"), nb::arg("sms"));

  m.def(
      "attn_decode",
      [](uintptr_t q, int64_t q_ts, int q_hs, uintptr_t gate, int64_t g_ts, int g_hs, uintptr_t out,
         int64_t o_ts, int o_hs, uintptr_t k_cache, uintptr_t v_cache, uintptr_t block_table, int bt_stride,
         uintptr_t ctx_lens, uintptr_t partial_o, uintptr_t partial_ml, uintptr_t tickets, int B, int n_q,
         int stages_per_split, int max_splits, uintptr_t k_scale, uintptr_t v_scale, uintptr_t stream) {
        chandra::AttnDecodeArgs a{};
        a.q = reinterpret_cast<const void*>(q);
        a.q_token_stride = q_ts;
        a.q_head_stride = q_hs;
        a.gate = reinterpret_cast<const void*>(gate);
        a.gate_token_stride = g_ts;
        a.gate_head_stride = g_hs;
        a.out = reinterpret_cast<void*>(out);
        a.out_token_stride = o_ts;
        a.out_head_stride = o_hs;
        a.k_cache = reinterpret_cast<const void*>(k_cache);
        a.v_cache = reinterpret_cast<const void*>(v_cache);
        a.block_table = reinterpret_cast<const int*>(block_table);
        a.block_table_stride = bt_stride;
        a.ctx_lens = reinterpret_cast<const int*>(ctx_lens);
        a.partial_o = reinterpret_cast<float*>(partial_o);
        a.partial_ml = reinterpret_cast<float*>(partial_ml);
        a.tickets = reinterpret_cast<unsigned*>(tickets);
        a.B = B;
        a.n_q = n_q;
        a.stages_per_split = stages_per_split;
        a.max_splits = max_splits;
        a.k_scale = reinterpret_cast<const float*>(k_scale);
        a.v_scale = reinterpret_cast<const float*>(v_scale);
        check(chandra::attn_decode(a, reinterpret_cast<cudaStream_t>(stream)));
      });
  m.attr("gdn_hist_floats") = chandra::kGdnHistFloats;
  m.def("attn_decode_split", [](int B, int max_ctx, int sms) {
    int sps = 0, ms = 0;
    chandra::attn_decode_split(B, max_ctx, sms, &sps, &ms);
    return nb::make_tuple(sps, ms);
  });

  m.def(
      "gdn_decode",
      [](uintptr_t qkv, int64_t qkv_stride, uintptr_t z, int64_t z_stride, uintptr_t b, uintptr_t a,
         int64_t ab_stride, uintptr_t A_log, uintptr_t dt_bias, uintptr_t conv_w, uintptr_t norm_w,
         uintptr_t state, bool state_fp16, uintptr_t slot_in, uintptr_t slot_out, uintptr_t conv_ring, int ring,
         uintptr_t conv_slot, uintptr_t pos, uintptr_t out, int64_t out_stride, int B, int T, uintptr_t n_replay,
         uintptr_t hist_in, uintptr_t hist_out, int hist_tmax, bool commit, uintptr_t stream) {
        chandra::GdnDecodeArgs g{};
        g.qkv = reinterpret_cast<const void*>(qkv);
        g.qkv_stride = qkv_stride;
        g.z = reinterpret_cast<const void*>(z);
        g.z_stride = z_stride;
        g.b = reinterpret_cast<const void*>(b);
        g.a = reinterpret_cast<const void*>(a);
        g.ab_stride = ab_stride;
        g.A_log = reinterpret_cast<const float*>(A_log);
        g.dt_bias = reinterpret_cast<const void*>(dt_bias);
        g.conv_w = reinterpret_cast<const void*>(conv_w);
        g.norm_w = reinterpret_cast<const void*>(norm_w);
        g.state = reinterpret_cast<void*>(state);
        g.state_fp16 = state_fp16;
        g.slot_in = reinterpret_cast<const int*>(slot_in);
        g.slot_out = reinterpret_cast<const int*>(slot_out);
        g.conv_ring = reinterpret_cast<void*>(conv_ring);
        g.ring = ring;
        g.conv_slot = reinterpret_cast<const int*>(conv_slot);
        g.pos = reinterpret_cast<const int*>(pos);
        g.out = reinterpret_cast<void*>(out);
        g.out_stride = out_stride;
        g.B = B;
        g.T = T;
        g.n_replay = reinterpret_cast<const int*>(n_replay);
        g.hist_in = reinterpret_cast<const float*>(hist_in);
        g.hist_out = reinterpret_cast<float*>(hist_out);
        g.hist_tmax = hist_tmax;
        g.commit = commit;
        check(chandra::gdn_decode(g, reinterpret_cast<cudaStream_t>(stream)));
      });

  m.def("vit_attn", [](uintptr_t q, uintptr_t k, uintptr_t v, int64_t qkv_stride, uintptr_t out,
                       int64_t out_stride, uintptr_t cu_seqlens, int n_seq, int max_len, int version,
                       uintptr_t k8, uintptr_t k_scale, uintptr_t k_mean, uintptr_t stream) {
    chandra::VitAttnArgs a{};
    a.q = reinterpret_cast<const void*>(q);
    a.k = reinterpret_cast<const void*>(k);
    a.v = reinterpret_cast<const void*>(v);
    a.qkv_stride = qkv_stride;
    a.out = reinterpret_cast<void*>(out);
    a.out_stride = out_stride;
    a.cu_seqlens = reinterpret_cast<const int*>(cu_seqlens);
    a.n_seq = n_seq;
    a.max_len = max_len;
    a.version = version;
    a.k8 = reinterpret_cast<int8_t*>(k8);
    a.k_scale = reinterpret_cast<float*>(k_scale);
    a.k_mean = reinterpret_cast<float*>(k_mean);
    a.max_tiles = (max_len + 63) / 64;
    check(chandra::vit_attn(a, reinterpret_cast<cudaStream_t>(stream)));
  });
  m.def("rmsnorm_add", [](uintptr_t x, uintptr_t res, uintptr_t res_out, uintptr_t out, uintptr_t w, int M,
                          uintptr_t stream) {
    check(chandra::rmsnorm_add(reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(res),
                               reinterpret_cast<void*>(res_out), reinterpret_cast<void*>(out),
                               reinterpret_cast<const void*>(w), M, reinterpret_cast<cudaStream_t>(stream)));
  });
  m.def("silu_mul", [](uintptr_t gu, uintptr_t out, int M, uintptr_t stream, int F) {
    check(chandra::silu_mul(reinterpret_cast<const void*>(gu), reinterpret_cast<void*>(out), M,
                            reinterpret_cast<cudaStream_t>(stream), F));
  });
  m.def("qk_norm_rope_kv", [](uintptr_t qkv, int64_t qkv_stride, uintptr_t qw, uintptr_t kw, uintptr_t pos,
                              float theta, uintptr_t q_out, uintptr_t kc, uintptr_t vc, uintptr_t page,
                              uintptr_t slot, int M, uintptr_t k_scale, uintptr_t v_scale, uintptr_t stream) {
    chandra::QkNormRopeArgs a{};
    a.qkv = reinterpret_cast<const void*>(qkv);
    a.qkv_stride = qkv_stride;
    a.q_norm_w = reinterpret_cast<const void*>(qw);
    a.k_norm_w = reinterpret_cast<const void*>(kw);
    a.pos = reinterpret_cast<const int*>(pos);
    a.rope_theta = theta;
    a.q_out = reinterpret_cast<void*>(q_out);
    a.k_cache = reinterpret_cast<void*>(kc);
    a.v_cache = reinterpret_cast<void*>(vc);
    a.page = reinterpret_cast<const int*>(page);
    a.slot = reinterpret_cast<const int*>(slot);
    a.M = M;
    a.k_scale = reinterpret_cast<float*>(k_scale);
    a.v_scale = reinterpret_cast<float*>(v_scale);
    check(chandra::qk_norm_rope_kv(a, reinterpret_cast<cudaStream_t>(stream)));
  });
}
