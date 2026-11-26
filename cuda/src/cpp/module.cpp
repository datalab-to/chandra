#include <cstdint>
#include <stdexcept>

#include <nanobind/nanobind.h>

#include "attn_decode.h"
#include "decode_gemm.h"

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
         int K, int sms, uintptr_t stream) {
        check(chandra::decode_gemm(reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(wp),
                                   reinterpret_cast<void*>(y), reinterpret_cast<float*>(workspace),
                                   reinterpret_cast<unsigned*>(counters), M, N, K, sms,
                                   reinterpret_cast<cudaStream_t>(stream)));
      },
      nb::arg("x"), nb::arg("wp"), nb::arg("y"), nb::arg("workspace"), nb::arg("counters"), nb::arg("M"),
      nb::arg("N"), nb::arg("K"), nb::arg("sms"), nb::arg("stream"));

  m.def(
      "decode_gemm_plan",
      [](int N, int K, int sms) {
        const chandra::DecodeGemmPlan p = chandra::decode_gemm_plan(N, K, sms);
        return nb::make_tuple(p.row_groups, p.splits);
      },
      nb::arg("N"), nb::arg("K"), nb::arg("sms"));
  m.def("decode_gemm_workspace_bytes", &chandra::decode_gemm_workspace_bytes, nb::arg("N"), nb::arg("K"),
        nb::arg("sms"));
  m.def("decode_gemm_counters", &chandra::decode_gemm_counters, nb::arg("N"), nb::arg("K"), nb::arg("sms"));

  m.def(
      "attn_decode",
      [](uintptr_t q, int64_t q_ts, int q_hs, uintptr_t gate, int64_t g_ts, int g_hs, uintptr_t out,
         int64_t o_ts, int o_hs, uintptr_t k_cache, uintptr_t v_cache, uintptr_t block_table, int bt_stride,
         uintptr_t ctx_lens, uintptr_t partial_o, uintptr_t partial_ml, uintptr_t tickets, int B, int n_q,
         int stages_per_split, int max_splits, uintptr_t stream) {
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
        check(chandra::attn_decode(a, reinterpret_cast<cudaStream_t>(stream)));
      });
  m.def("attn_decode_split", [](int B, int max_ctx, int sms) {
    int sps = 0, ms = 0;
    chandra::attn_decode_split(B, max_ctx, sms, &sps, &ms);
    return nb::make_tuple(sps, ms);
  });
}
