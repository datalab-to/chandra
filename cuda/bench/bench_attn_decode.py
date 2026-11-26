"""Time attn_decode against vLLM's FA2 decode kernel at Chandra 2's decode shapes.

KV bytes per call = B x ctx x 4 KV heads x 256 x 2 (K, V) x 2 bytes. Reports µs and
GB/s per batch size and query tokens per sequence (1 = decode, 4 = verify of 3
draft tokens). FA2 is called as vLLM calls it for this model: flash_attn_varlen_func
on a paged cache of 576-token blocks, causal. Its time does not include the sigmoid
gate; ours does.

usage: python bench_attn_decode.py [--ctx 7995] [--batches 1 8 32 64] [--reps 20]
"""

import argparse
import math

import torch

import chandra_cuda

H, KVH, D, PAGE = 16, 4, 256, 64


def time_us(fn, reps):
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
    ap.add_argument("--ctx", type=int, default=7995)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except Exception as e:  # noqa: BLE001
        flash_attn_varlen_func = None
        print("FA2 unavailable:", e)
    print(f"{'B':>3s} {'n_q':>3s} {'ctx':>5s} {'MB':>7s} {'ours us':>8s} {'GB/s':>5s} {'FA2 us':>8s} {'GB/s':>5s} {'x':>5s} "
          f"{'1K-split us':>11s} {'INT8 KV us':>10s}")
    for b in a.batches:
        pages = math.ceil(a.ctx / PAGE)
        kc = torch.randn(b * pages, KVH, PAGE, D, device="cuda").to(torch.bfloat16)
        vc = torch.randn_like(kc)
        k8 = torch.randint(-127, 128, kc.shape, device="cuda", dtype=torch.int8)  # INT8 KV: half the bytes
        v8 = torch.randint(-127, 128, kc.shape, device="cuda", dtype=torch.int8)
        ks = torch.rand(kc.shape[:3], device="cuda") * 0.02
        vs = torch.rand(kc.shape[:3], device="cuda") * 0.02
        bt = torch.arange(b * pages, device="cuda", dtype=torch.int32).reshape(b, pages)
        ctx = torch.full((b,), a.ctx, dtype=torch.int32, device="cuda")
        # vLLM's layout for FA2: [blocks, 576, 4, 256], K and V interleaved per block.
        fbs = 576
        fblocks = math.ceil(a.ctx / fbs)
        phys = torch.randn(b * fblocks, fbs, KVH, 2 * D, device="cuda").to(torch.bfloat16)
        kf, vf = phys[..., :D], phys[..., D:]
        fbt = torch.arange(b * fblocks, device="cuda", dtype=torch.int32).reshape(b, fblocks)
        mb = b * a.ctx * KVH * D * 2 * 2 / 1e6
        for n_q in (1, 4):
            qg = torch.randn(b * n_q, H, 2 * D, device="cuda").to(torch.bfloat16)
            q, gate = qg[..., :D], qg[..., D:]
            out = torch.empty(b * n_q, H, D, device="cuda", dtype=torch.bfloat16)
            t_ours = time_us(
                lambda: chandra_cuda.attn_decode(q, gate, kc, vc, bt, ctx, n_q=n_q, max_ctx=a.ctx, out=out), a.reps
            )
            t_fix = time_us(  # fixed 1,024-key chunks: batch-invariant, the engine's default
                lambda: chandra_cuda.attn_decode(q, gate, kc, vc, bt, ctx, n_q=n_q, max_ctx=a.ctx, out=out,
                                                 split_keys=1024), a.reps
            )
            t_i8 = time_us(
                lambda: chandra_cuda.attn_decode(q, gate, k8, v8, bt, ctx, n_q=n_q, max_ctx=a.ctx, out=out,
                                                 k_scale=ks, v_scale=vs, split_keys=1024), a.reps
            )
            t_fa = float("nan")
            if flash_attn_varlen_func is not None:
                qf = q.contiguous()
                of = torch.empty_like(qf)
                cu = torch.arange(0, b * n_q + 1, n_q, dtype=torch.int32, device="cuda")
                t_fa = time_us(
                    lambda: flash_attn_varlen_func(
                        q=qf, k=kf, v=vf, out=of, cu_seqlens_q=cu, max_seqlen_q=n_q, seqused_k=ctx,
                        max_seqlen_k=a.ctx, softmax_scale=1 / 16.0, causal=True, block_table=fbt, fa_version=2,
                    ),
                    a.reps,
                )
            print(f"{b:3d} {n_q:3d} {a.ctx:5d} {mb:7.0f} {t_ours:8.1f} {mb / t_ours * 1e3:5.0f} "
                  f"{t_fa:8.1f} {mb / t_fa * 1e3:5.0f} {t_fa / t_ours:5.2f} {t_fix:11.1f} {t_i8:10.1f}")
        del kc, vc, phys, kf, vf, k8, v8
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
