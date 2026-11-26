"""Prefill and ViT GEMMs: cuBLAS BF16 (F.linear) against cuBLASLt INT8 (torch._int_mm, int32 out),
plus the torch cost of the per-token activation quantization and the int32 -> BF16 dequant that
an INT8 path needs (what a fused kernel would remove). Tells how much W8A8 can give at these shapes.

usage: python bench_prefill_gemm.py [--reps 20]
"""

import argparse

import torch
import torch.nn.functional as F

# name, N, K, uses per page, M (tokens per page: ~6.7K LLM prefill rows, ~24.5K ViT patches)
SHAPES = [
    ("llm gdn qkvz", 12288, 2560, 24, 6700),
    ("llm gdn out", 2560, 4096, 24, 6700),
    ("llm attn qkv", 10240, 2560, 8, 6700),
    ("llm attn o", 2560, 4096, 8, 6700),
    ("llm gate_up", 18432, 2560, 32, 6700),
    ("llm down", 2560, 9216, 32, 6700),
    ("vit qkv", 3072, 1024, 24, 24576),
    ("vit proj", 1024, 1024, 24, 24576),
    ("vit fc1", 4096, 1024, 24, 24576),
    ("vit fc2", 1024, 4096, 24, 24576),
]


def time_ms(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(0)
    tot = [0.0, 0.0, 0.0]
    print(f"{'shape':14s} {'M':>6s} {'N':>6s} {'K':>5s} {'bf16 ms':>8s} {'TFLOPS':>7s} {'int8 ms':>8s} {'TOPS':>6s} "
          f"{'quant+deq ms':>12s} {'x (gemm)':>8s} {'x (all)':>7s}")
    for name, n, k, uses, m in SHAPES:
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        x8 = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        w8 = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        ws = torch.rand(n, device="cuda")
        flops = 2 * m * n * k
        t_bf = time_ms(lambda: F.linear(x, w), a.reps)
        t_i8 = time_ms(lambda: torch._int_mm(x8, w8.t()), a.reps)
        acc = torch._int_mm(x8, w8.t())

        def qdq():
            s = x.float().abs().amax(dim=1, keepdim=True) / 127
            q = torch.round(x.float() / s).to(torch.int8)
            y = (acc.float() * s * ws).to(torch.bfloat16)
            return q, y

        t_q = time_ms(qdq, a.reps)
        tot[0] += uses * t_bf
        tot[1] += uses * t_i8
        tot[2] += uses * (t_i8 + t_q)
        print(f"{name:14s} {m:6d} {n:6d} {k:5d} {t_bf:8.3f} {flops / t_bf / 1e9:7.1f} {t_i8:8.3f} "
              f"{flops / t_i8 / 1e9:6.1f} {t_q:12.3f} {t_bf / t_i8:8.2f} {t_bf / (t_i8 + t_q):7.2f}")
        del x, w, x8, w8, acc
        torch.cuda.empty_cache()
    print(f"per page (LLM prefill + ViT GEMMs): BF16 {tot[0] / 1e3:.3f} s, INT8 GEMMs alone {tot[1] / 1e3:.3f} s "
          f"({tot[0] / tot[1]:.2f}x), INT8 with unfused torch quant/dequant {tot[2] / 1e3:.3f} s")


if __name__ == "__main__":
    main()
