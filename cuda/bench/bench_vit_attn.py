"""Time vit_attn against FA2 (vLLM's flash_attn_varlen_func, non-causal) on one page image.

usage: python bench_vit_attn.py [--len 24552] [--reps 10]
"""

import argparse

import torch

import chandra_cuda


def time_ms(fn, reps):
    for _ in range(2):
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
    ap.add_argument("--len", type=int, default=24552)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    n = a.len
    qkv = torch.randn(n, 3, 16, 64, device="cuda").to(torch.bfloat16)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    out = torch.empty(n, 16, 64, device="cuda", dtype=torch.bfloat16)
    flops = 4 * n * n * 16 * 64
    t1 = time_ms(lambda: chandra_cuda.vit_attn(q, k, v, cu, n, out, version=1), a.reps)
    print(f"ours v1: {t1:.2f} ms, {flops / t1 / 1e9:.1f} TFLOPS")
    t = time_ms(lambda: chandra_cuda.vit_attn(q, k, v, cu, n, out, version=2), a.reps)
    print(f"ours v2: {t:.2f} ms, {flops / t / 1e9:.1f} TFLOPS")
    t4 = time_ms(lambda: chandra_cuda.vit_attn(q, k, v, cu, n, out, version=4), a.reps)
    print(f"ours v4: {t4:.2f} ms, {flops / t4 / 1e9:.1f} TFLOPS")
    ref1 = chandra_cuda.vit_attn(q, k, v, cu, n, version=1).float()
    t3 = time_ms(lambda: chandra_cuda.vit_attn(q, k, v, cu, n, out, version=3), a.reps)
    rel = ((out.float() - ref1).abs().sum() / ref1.abs().sum()).item()
    print(f"ours INT8 QK (incl. K mean + quant): {t3:.2f} ms, {flops / t3 / 1e9:.1f} TFLOPS-equiv, "
          f"rel L1 vs BF16 {rel:.2e}")
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
        of = torch.empty_like(qc)
        tf = time_ms(lambda: flash_attn_varlen_func(q=qc, k=kc, v=vc, out=of, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                                    max_seqlen_q=n, max_seqlen_k=n, softmax_scale=0.125,
                                                    causal=False, fa_version=2), a.reps)
        print(f"FA2:     {tf:.2f} ms, {flops / tf / 1e9:.1f} TFLOPS  (v2 {tf / t:.2f}x, v1 {tf / t1:.2f}x)")
        print(f"max |ours - FA2| = {(out.float() - of.float()).abs().max().item():.2e}")
    except Exception as e:  # noqa: BLE001
        print("FA2 unavailable:", e)


if __name__ == "__main__":
    main()
