"""Write a compressed-tensors copy of Chandra OCR 2 for vLLM: the language model's linear layers
quantized, everything else (vision tower, embeddings, LM head, MTP layer, norms, conv) unchanged.

  w8a8:  INT8 weights, one scale per output channel (symmetric, round to nearest), activations
         quantized per token at run time (dynamic). vLLM runs it with its INT8 scaled-mm kernels
         (CUTLASS on sm_75+).
  w4a16: INT4 weights, one BF16 scale per 128 along K (symmetric, amax / 7.5, round to nearest,
         as compressed-tensors' minmax observer), BF16 activations. vLLM runs it with Marlin.

usage: python make_ct_checkpoint.py SRC_SNAPSHOT_DIR OUT_DIR [--scheme w8a8|w4a16]
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LINEAR_SUFFIXES = (
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight", "linear_attn.out_proj.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)


def is_target(name: str) -> bool:
    return name.startswith("model.language_model.layers.") and name.endswith(LINEAR_SUFFIXES)


def quant_w8(w: torch.Tensor):
    wf = w.float()
    scale = (wf.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    q = torch.round(wf / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def quant_w4_g128(w: torch.Tensor):
    n, k = w.shape
    wf = w.float().reshape(n, k // 128, 128)
    scale = (wf.abs().amax(dim=2) / 7.5).clamp_min(1e-12).to(torch.bfloat16)
    q = torch.round(wf / scale.float()[..., None]).clamp(-8, 7).to(torch.int8).reshape(n, k)
    try:
        from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    except ImportError:
        pack_to_int32 = None
    if pack_to_int32 is not None:
        packed = pack_to_int32(q, 4)
    else:  # the same layout: 8 consecutive K values per int32, low nibble first, stored as q + 8
        u = (q.to(torch.int32) + 8).reshape(n, k // 8, 8)
        packed = (u << (torch.arange(8, device=w.device, dtype=torch.int32) * 4)).sum(dim=2, dtype=torch.int32)
    return packed, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--scheme", default="w8a8", choices=["w8a8", "w4a16"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f = safe_open(os.path.join(a.src, "model.safetensors"), "pt", device="cuda")
    out = {}
    n_q = 0
    for name in f.keys():
        t = f.get_tensor(name)
        if is_target(name) and a.scheme == "w8a8":
            q, s = quant_w8(t)
            out[name] = q.cpu()
            out[name[: -len("weight")] + "weight_scale"] = s.to(torch.bfloat16).cpu()
            n_q += 1
        elif is_target(name):
            packed, s = quant_w4_g128(t)
            base = name[: -len("weight")]
            out[base + "weight_packed"] = packed.cpu()
            out[base + "weight_scale"] = s.cpu()
            out[base + "weight_shape"] = torch.tensor(list(t.shape), dtype=torch.int64)
            n_q += 1
        else:
            out[name] = t.cpu()
        del t
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    for fn in os.listdir(a.src):
        if fn != "model.safetensors" and not fn.endswith(".png"):
            src = os.path.realpath(os.path.join(a.src, fn))
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(a.out, fn))
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    if a.scheme == "w8a8":
        fmt = "int-quantized"
        weights = {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False,
                   "group_size": None, "block_structure": None, "actorder": None, "observer": "minmax"}
        acts = {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True,
                "group_size": None, "block_structure": None, "actorder": None, "observer": None}
    else:
        fmt = "pack-quantized"
        weights = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "dynamic": False,
                   "group_size": 128, "block_structure": None, "actorder": None, "observer": "minmax"}
        acts = None
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": fmt,
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {"targets": ["Linear"], "weights": weights, "input_activations": acts,
                        "output_activations": None}
        },
        "ignore": ["lm_head", "re:.*visual.*", "re:.*mtp.*"],
        "kv_cache_scheme": None,
    }
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"{n_q} linear weights quantized ({a.scheme}); wrote {a.out}")


if __name__ == "__main__":
    main()
