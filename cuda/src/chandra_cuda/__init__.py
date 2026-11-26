"""Optional CUDA kernels for Chandra OCR.

The kernels run on torch's current CUDA stream and take torch tensors; the
compiled module (`_core`) only sees device pointers.
"""

from __future__ import annotations

import torch

from . import _core

__all__ = ["pack_weight", "DecodeLinear", "attn_decode"]

HEADS, KV_HEADS, HEAD_DIM, PAGE_TOKENS = 16, 4, 256, 64

_attn_ws: dict = {}


def attn_decode(
    q: torch.Tensor,
    gate: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    ctx_lens: torch.Tensor,
    n_q: int = 1,
    max_ctx: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gated decode/verify attention for Chandra 2's full-attention layers.

    q, gate: [B * n_q, 16, 256] BF16 views (token, head, dim; the last dim
    contiguous), e.g. the two halves of q_proj's per-head [q | gate] output.
    k_cache, v_cache: [pages, 4, 64, 256] BF16. block_table: [B, max_pages]
    int32. ctx_lens: [B] int32, tokens in the cache including this step's n_q.
    Query token t of a sequence sees keys up to position ctx - n_q + t.
    Returns [B * n_q, 16, 256] BF16: softmax(q k^T / 16) v * sigmoid(gate).
    """
    b = ctx_lens.shape[0]
    if q.shape != (b * n_q, HEADS, HEAD_DIM) or gate.shape != q.shape:
        raise ValueError("q and gate must be [B * n_q, 16, 256]")
    if q.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("the head dimension must be contiguous")
    dev = q.device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    if max_ctx is None:
        max_ctx = int(ctx_lens.max())
    sps, max_splits = _core.attn_decode_split(b, max_ctx, sms)
    key = (dev, b, max_splits)
    ws = _attn_ws.get(key)
    if ws is None:
        ws = (
            torch.empty(b * KV_HEADS * max_splits * 16 * HEAD_DIM, dtype=torch.float32, device=dev),
            torch.empty(b * KV_HEADS * max_splits * 16 * 2, dtype=torch.float32, device=dev),
            torch.zeros(b * KV_HEADS, dtype=torch.int32, device=dev),
        )
        _attn_ws[key] = ws
    if out is None:
        out = torch.empty(b * n_q, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    _core.attn_decode(
        q.data_ptr(), q.stride(0), q.stride(1),
        gate.data_ptr(), gate.stride(0), gate.stride(1),
        out.data_ptr(), out.stride(0), out.stride(1),
        k_cache.data_ptr(), v_cache.data_ptr(),
        block_table.data_ptr(), block_table.stride(0), ctx_lens.data_ptr(),
        ws[0].data_ptr(), ws[1].data_ptr(), ws[2].data_ptr(),
        b, n_q, sps, max_splits, torch.cuda.current_stream(dev).cuda_stream,
    )
    return out


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
