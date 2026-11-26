"""Time DecodeLinear against cuBLAS (torch F.linear) on Chandra 2's decode GEMMs.

Prints µs per call and weight GB/s per shape and batch size, and the sum over one
decode step (24 GDN layers, 8 full-attention layers, 32 MLPs, the LM head).

usage: python bench_decode_gemm.py [--rows 1 8 16 32 64] [--reps 50]
"""

import argparse

import torch
import torch.nn.functional as F

import chandra_cuda

# name, N, K, uses per decode step
SHAPES = [
    ("gdn qkvz", 12288, 2560, 24),
    ("gdn ba", 64, 2560, 24),
    ("gdn out", 2560, 4096, 24),
    ("attn qkv", 10240, 2560, 8),
    ("attn o", 2560, 4096, 8),
    ("mlp gate_up", 18432, 2560, 32),
    ("mlp down", 2560, 9216, 32),
    ("lm head", 248320, 2560, 1),
]


def time_us(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    torch.manual_seed(0)
    step = {m: [0.0, 0.0, 0.0, 0.0] for m in a.rows}
    print(f"{'shape':12s} {'N':>6s} {'K':>5s} {'M':>3s} {'plan':>8s} {'cuBLAS us':>9s} {'ours us':>8s} "
          f"{'cuBLAS GB/s':>11s} {'ours GB/s':>9s} {'x':>5s} {'fp8 us':>7s} {'fp4 us':>7s}")
    for name, n, k, uses in SHAPES:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        lin = chandra_cuda.DecodeLinear(w)
        lin8 = chandra_cuda.DecodeLinearFP8(w) if n % 32 == 0 and name != "lm head" else None
        lin4 = chandra_cuda.DecodeLinearFP4(w) if n % 64 == 0 and name != "lm head" else None
        gb = n * k * 2 / 1e9
        for m in a.rows:
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            t_ref = time_us(lambda: F.linear(x, w), a.reps)
            t_ours = time_us(lambda: lin(x, out), a.reps)
            t_fp8 = time_us(lambda: lin8(x, out), a.reps) if lin8 is not None else float("nan")
            t_fp4 = time_us(lambda: lin4(x, out), a.reps) if lin4 is not None else float("nan")
            step[m][2] += uses * (t_fp8 if lin8 is not None else t_ours)  # LM head stays BF16
            step[m][3] += uses * (t_fp4 if lin4 is not None else t_ours)
            step[m][0] += uses * t_ref
            step[m][1] += uses * t_ours
            print(f"{name:12s} {n:6d} {k:5d} {m:3d} {str(lin.plan):>8s} {t_ref:9.1f} {t_ours:8.1f} "
                  f"{gb / t_ref * 1e6:11.0f} {gb / t_ours * 1e6:9.0f} {t_ref / t_ours:5.2f} {t_fp8:7.1f} {t_fp4:7.1f}")
        del w, lin, lin8, lin4
        torch.cuda.empty_cache()
    print("one decode step, all GEMMs (ms):")
    for m in a.rows:
        print(f"  M={m:2d}: cuBLAS {step[m][0] / 1e3:6.2f}  ours {step[m][1] / 1e3:6.2f}  "
              f"({step[m][0] / step[m][1]:.2f}x)  ours, FP8 weights (LM head BF16) "
              f"{step[m][2] / 1e3:6.2f} ({step[m][0] / step[m][2]:.2f}x)  MXFP4 {step[m][3] / 1e3:6.2f} "
              f"({step[m][0] / step[m][3]:.2f}x)")


if __name__ == "__main__":
    main()
