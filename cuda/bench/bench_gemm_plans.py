"""Decode GEMM plans (rows per CTA, K splits) per shape and batch size, BF16 and MXFP4, to choose
the plan rule. Plans are forced with _core.set_decode_gemm_plan (measurement only).

usage: python bench_gemm_plans.py [--rows 1 8 32 64]
"""

import argparse

import torch

import chandra_cuda
from chandra_cuda import _core

SHAPES = [("gdn qkvzba", 12416, 2560), ("gdn out", 2560, 4096), ("attn qkv", 10240, 2560),
          ("mlp gate_up", 18432, 2560), ("mlp down", 2560, 9216)]


def time_us(fn, reps=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 64])
    a = ap.parse_args()
    torch.manual_seed(0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for name, n, k in SHAPES:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        default = _core.decode_gemm_plan(n, k, sms)
        chunks = k // 128
        plans = [(rw, s) for rw in (8, 4) if n % (16 * rw) == 0 for s in (1, 2, 3, 4, 6, 8, 12, 16)
                 if chunks % s == 0 and (n // (16 * rw)) * s <= 6 * sms]
        plans += [(8, -p) for p in (sms, 2 * sms, 3 * sms)]  # stream-K with P CTAs
        print(f"{name} N={n} K={k} default plan {default} ({sms} SMs)")
        for fmt, cls in (("bf16", chandra_cuda.DecodeLinear), ("mxfp4", chandra_cuda.DecodeLinearFP4)):
            for rw, s in plans:
                _core.set_decode_gemm_plan(n, k, rw, s)
                lin = cls(w)
                row = []
                for m in a.rows:
                    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
                    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                    row.append(time_us(lambda: lin(x, out)))
                ctas = (n // (16 * rw)) * s if s > 0 else -s
                print(f"  {fmt:5s} rw {rw} splits {s:2d} ({ctas:4d} CTAs): " +
                      "  ".join(f"M={m} {t:6.1f}" for m, t in zip(a.rows, row)), flush=True)
                del lin
            _core.set_decode_gemm_plan(n, k, 0, 0)
        del w
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
