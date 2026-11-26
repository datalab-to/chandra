#include <cstdint>
#include <stdexcept>

#include <nanobind/nanobind.h>

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
}
