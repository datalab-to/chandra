"""Time gdn_decode (one fused kernel per layer) against vLLM's decode path for the same step:
causal_conv1d_update + fused_recurrent_gated_delta_rule_packed_decode + fused_gdn_gated_rms_norm.

24 layers per CUDA graph (each with its own state pool), as in a decode step. State bytes per
layer = 2 (read + write) x B x 32 x 128 x 128 x element size.

usage: python bench_gdn_decode.py [--batches 1 8 32 64]
"""

import argparse

import torch

import chandra_cuda

HV, D, C, NL = 32, 128, 8192, 24


def graph_us(fn, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64])
    a = ap.parse_args()
    try:
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
        from vllm.models.qwen3_5.nvidia.ops.fused_gdn_gated_norm import fused_gdn_gated_rms_norm
        from vllm.third_party.flash_linear_attention.ops import fused_recurrent_gated_delta_rule_packed_decode
        have_vllm = True
    except Exception as e:  # noqa: BLE001
        print("vLLM kernels unavailable:", e)
        have_vllm = False
    A_log = torch.randn(HV, device="cuda") * 0.5
    dt_bias = (torch.randn(HV, device="cuda") * 0.5).to(torch.bfloat16)
    conv_w = (torch.randn(C, 4, device="cuda") * 0.4).to(torch.bfloat16)
    norm_w = torch.ones(D, device="cuda", dtype=torch.bfloat16)
    print(f"{'B':>3s} {'state':>6s} {'ours us/layer':>13s} {'GB/s':>5s} {'vLLM us/layer':>13s} {'GB/s':>5s} {'x':>5s}")
    for b in a.batches:
        qkv = torch.randn(b, C, device="cuda").to(torch.bfloat16)
        z = torch.randn(b, HV * D, device="cuda").to(torch.bfloat16)
        ba = torch.randn(b, 2 * HV, device="cuda").to(torch.bfloat16)
        bb, aa = ba[:, :HV], ba[:, HV:]
        slot = torch.arange(1, b + 1, dtype=torch.int32, device="cuda")
        pos = torch.full((b,), 100, dtype=torch.int32, device="cuda")
        out = torch.empty(b, HV * D, device="cuda", dtype=torch.bfloat16)
        for sdt in (torch.float32, torch.float16):
            pools = [(torch.randn(b + 1, HV, D, D, device="cuda") * 0.05).to(sdt) for _ in range(NL)]
            rings = [torch.zeros(b + 1, 4, C, device="cuda", dtype=torch.bfloat16) for _ in range(NL)]
            ours = graph_us(lambda: [chandra_cuda.gdn_decode(qkv, z, bb, aa, A_log, dt_bias, conv_w, norm_w, pools[i],
                                                             slot, slot, rings[i], slot, pos, out) for i in range(NL)])
            sb = 2 * b * HV * D * D * pools[0].element_size()
            t_v = float("nan")
            if have_vllm:
                cstates = [torch.zeros(b + 1, C, 3, device="cuda", dtype=torch.bfloat16) for _ in range(NL)]
                mixed = torch.empty(b, C, device="cuda", dtype=torch.bfloat16)
                o4 = torch.empty(b, 1, HV, D, device="cuda", dtype=torch.bfloat16)
                bc, ac = bb.contiguous(), aa.contiguous()

                def vllm_step():
                    for i in range(NL):
                        mixed.copy_(qkv)
                        causal_conv1d_update(mixed, cstates[i], conv_w, None, "silu", conv_state_indices=slot)
                        fused_recurrent_gated_delta_rule_packed_decode(
                            mixed_qkv=mixed, a=ac, b=bc, A_log=A_log, dt_bias=dt_bias, scale=D**-0.5,
                            initial_state=pools[i], out=o4, ssm_state_indices=slot, use_qk_l2norm_in_kernel=True)
                        on = o4.view(b, -1)
                        fused_gdn_gated_rms_norm(x=on, z=z, weight=norm_w, out=on, eps=1e-6, num_v_heads=HV,
                                                 head_v_dim=D, norm_before_gate=True, activation="silu")

                t_v = graph_us(vllm_step)
            print(f"{b:3d} {str(sdt).split('.')[-1]:>6s} {ours / NL:13.1f} {sb / (ours / NL) / 1e3:5.0f} "
                  f"{t_v / NL:13.1f} {sb / (t_v / NL) / 1e3:5.0f} {t_v / ours:5.2f}")
            del pools, rings
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
