"""Optional CUDA kernels for Chandra OCR.

The kernels run on torch's current CUDA stream and take torch tensors; the
compiled module (`_core`) only sees device pointers.
"""

from __future__ import annotations

import torch

from . import _core

__all__ = ["pack_weight", "DecodeLinear"]


def pack_weight(w: torch.Tensor) -> torch.Tensor:
    """Reorder a [N, K] BF16 weight for `DecodeLinear`.

    For each 16-row group and k16 block, lane l of a warp gets its four
    registers of the mma.m16n8k16 A fragment as 16 contiguous bytes:
    a0 = (row l/4, k 2(l%4)..+1), a1 = row + 8, a2 = k + 8, a3 = row + 8 and k + 8.
    Groups are stored k16 block by k16 block, so a warp streams contiguous memory.
    """
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise ValueError("expected a 2-D bfloat16 weight")
    n, k = w.shape
    if n % 16 or k % 16:
        raise ValueError(f"weight shape {tuple(w.shape)} must be multiples of 16")
    # w[g*16 + rh*8 + gid][kb*16 + kh*8 + tig*2 + e] -> [g][kb][gid][tig][kh][rh][e]
    return (
        w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2)
        .permute(0, 3, 2, 5, 4, 1, 6)
        .contiguous()
        .reshape(-1)
    )


class DecodeLinear:
    """y = x @ weight.T for decode batches (1 to 64 rows), BF16, FP32 accumulation.

    The weight is read once per call from its packed layout. The K split depends
    only on the weight's shape and the GPU, so each row's result is the same for
    any number of rows in the batch. One instance must not run on two streams at
    once (its workspace is shared).
    """

    def __init__(self, weight: torch.Tensor):
        if not weight.is_cuda:
            raise ValueError("weight must be on a CUDA device")
        self.n, self.k = weight.shape
        self.device = weight.device
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        if self.k % 128:
            raise ValueError("K must be a multiple of 128")
        self.packed = pack_weight(weight)
        ws = _core.decode_gemm_workspace_bytes(self.n, self.k, self.sms)
        self.workspace = torch.empty(max(ws // 4, 1), dtype=torch.float32, device=self.device)
        self.counters = torch.zeros(
            _core.decode_gemm_counters(self.n, self.k, self.sms), dtype=torch.int32, device=self.device
        )
        self.plan = _core.decode_gemm_plan(self.n, self.k, self.sms)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != self.k:
            raise ValueError(f"expected a [M, {self.k}] bfloat16 input")
        m = x.shape[0]
        if not 1 <= m <= 64:
            raise ValueError("M must be between 1 and 64")
        x = x.contiguous()
        if out is None:
            out = torch.empty(m, self.n, dtype=torch.bfloat16, device=self.device)
        _core.decode_gemm(
            x.data_ptr(),
            self.packed.data_ptr(),
            out.data_ptr(),
            self.workspace.data_ptr(),
            self.counters.data_ptr(),
            m,
            self.n,
            self.k,
            self.sms,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        return out
