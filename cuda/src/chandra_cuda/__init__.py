"""Optional CUDA kernels for Chandra OCR.

The kernels run on torch's current CUDA stream and take torch tensors; the
compiled module (`_core`) only sees device pointers.
"""

from __future__ import annotations

import torch

from . import _core

__all__ = ["pack_weight", "pack_weight_fp8", "DecodeLinear", "DecodeLinearFP8", "LMHead", "attn_decode", "gdn_decode", "rmsnorm_add", "silu_mul",
           "qk_norm_rope_kv"]


def gdn_decode(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_w: torch.Tensor,
    norm_w: torch.Tensor,
    state: torch.Tensor,
    slot_in: torch.Tensor,
    slot_out: torch.Tensor,
    conv_ring: torch.Tensor,
    conv_slot: torch.Tensor,
    pos: torch.Tensor,
    out: torch.Tensor | None = None,
    T: int = 1,
    n_replay: torch.Tensor | None = None,
    hist_in: torch.Tensor | None = None,
    hist_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One Gated DeltaNet step of one layer, for B sequences of T tokens each (T = 1: decode;
    T = 2..4: verification of draft tokens; row n * T + t is token t of sequence n).

    qkv: [B, 8192] BF16 (in_proj_qkv output, before the conv); z: [B, 4096] BF16;
    b, a: [B, 32] BF16 views (same row stride); A_log: [32] FP32; dt_bias: [32] BF16;
    conv_w: [8192, 4] BF16; norm_w: [128] BF16.
    state: [slots, 32, 128, 128] FP32 or FP16, read from slot_in and written to slot_out ([B] int32).
    conv_ring: [slots, ring >= 4, 8192] BF16 raw inputs by position % ring; conv_slot, pos: [B] int32,
    pos = tokens of the sequence before this step. Returns [B * T, 4096] BF16, the out_proj input.

    Verification with one state read and one write: hist_out ([slots, tmax, 32, gdn_hist_floats]
    FP32, indexed by conv_slot) receives each token's k, v' and decay. If the next step starts from
    the same input state with n_replay[n] = j and hist_in = this hist_out, it first replays the j
    accepted updates, which gives exactly the state after token j.
    """
    if qkv.shape[0] % T:
        raise ValueError("qkv rows must be B * T")
    n = qkv.shape[0] // T
    if (n_replay is None) != (hist_in is None):
        raise ValueError("n_replay and hist_in go together")
    tmax = next((h.shape[1] for h in (hist_in, hist_out) if h is not None), 0)
    if b.stride(0) != a.stride(0):
        raise ValueError("a and b must share a row stride")
    if state.dtype not in (torch.float32, torch.float16):
        raise ValueError("state must be float32 or float16")
    for t in (qkv, z, b, a):
        if t.stride(-1) != 1:
            raise ValueError("the last dimension must be contiguous")
    if out is None:
        out = torch.empty(n * T, 4096, dtype=torch.bfloat16, device=qkv.device)
    _core.gdn_decode(
        qkv.data_ptr(), qkv.stride(0), z.data_ptr(), z.stride(0), b.data_ptr(), a.data_ptr(), a.stride(0),
        A_log.data_ptr(), dt_bias.data_ptr(), conv_w.data_ptr(), norm_w.data_ptr(),
        state.data_ptr(), state.dtype == torch.float16, slot_in.data_ptr(), slot_out.data_ptr(),
        conv_ring.data_ptr(), conv_ring.shape[1], conv_slot.data_ptr(), pos.data_ptr(),
        out.data_ptr(), out.stride(0), n, T,
        0 if n_replay is None else n_replay.data_ptr(),
        0 if hist_in is None else hist_in.data_ptr(),
        0 if hist_out is None else hist_out.data_ptr(),
        tmax, torch.cuda.current_stream(qkv.device).cuda_stream,
    )
    return out

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


class LMHead:
    """Greedy next tokens for 1 to 64 rows: argmax_n of bf16(x @ weight.T)[m, n], computed
    while the weight streams, without writing the [M, vocab] logits. Ties go to the lowest
    index, as torch.argmax. weight: [vocab, K] BF16 with vocab % 128 == 0."""

    def __init__(self, weight: torch.Tensor):
        self.n, self.k = weight.shape
        if self.n % 128 or self.k % 128:
            raise ValueError("vocabulary and K must be multiples of 128")
        self.device = weight.device
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.packed = pack_weight(weight)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        if x.dtype != torch.bfloat16 or x.shape[1] != self.k or not 1 <= m <= 64:
            raise ValueError(f"expected a [1..64, {self.k}] bfloat16 input")
        x = x.contiguous()
        keys = torch.zeros(m, dtype=torch.int64, device=self.device)
        _core.decode_gemm_argmax(
            x.data_ptr(), self.packed.data_ptr(), keys.data_ptr(), m, self.n, self.k, self.sms,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        return 0xFFFFFFFF - (keys & 0xFFFFFFFF)


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


def rmsnorm_add(residual: torch.Tensor, w: torch.Tensor, x: torch.Tensor | None = None,
                residual_out: torch.Tensor | None = None, out: torch.Tensor | None = None):
    """Qwen3.5's decoder-layer norm on [M, 2560] BF16 rows: r = bf16(x + residual) (r = residual
    when x is None), out = bf16(rmsnorm(r) * (1 + w)). Returns (out, r)."""
    m = residual.shape[0]
    if out is None:
        out = torch.empty_like(residual)
    if x is not None and residual_out is None:
        residual_out = torch.empty_like(residual)
    _core.rmsnorm_add(0 if x is None else x.data_ptr(), residual.data_ptr(),
                      0 if residual_out is None else residual_out.data_ptr(), out.data_ptr(), w.data_ptr(), m,
                      _stream(residual))
    return out, (residual if x is None else residual_out)


def silu_mul(gate_up: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """[M, 2 * 9216] gate | up -> [M, 9216] bf16(silu(gate) * up)."""
    m = gate_up.shape[0]
    if out is None:
        out = torch.empty(m, 9216, dtype=torch.bfloat16, device=gate_up.device)
    _core.silu_mul(gate_up.data_ptr(), out.data_ptr(), m, _stream(gate_up))
    return out


def qk_norm_rope_kv(qkv: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor, pos: torch.Tensor,
                    k_cache: torch.Tensor, v_cache: torch.Tensor, page: torch.Tensor, slot: torch.Tensor,
                    rope_theta: float = 1e7, q_out: torch.Tensor | None = None) -> torch.Tensor:
    """After the merged qkv GEMM of a full-attention layer (rows [16 x (q | gate)] [k] [v], 10,240 BF16):
    q -> RMSNorm(1 + w) + partial mRoPE -> q_out [M, 16, 256]; k likewise and v as is into the paged
    cache [pages, 4, 64, 256] at (page[m], slot[m]). pos: [M, 3] int32 (t, h, w) positions."""
    m = qkv.shape[0]
    if qkv.stride(-1) != 1:
        raise ValueError("qkv rows must be contiguous")
    if q_out is None:
        q_out = torch.empty(m, 16, 256, dtype=torch.bfloat16, device=qkv.device)
    _core.qk_norm_rope_kv(qkv.data_ptr(), qkv.stride(0), q_norm_w.data_ptr(), k_norm_w.data_ptr(),
                          pos.contiguous().data_ptr(), float(rope_theta), q_out.data_ptr(), k_cache.data_ptr(),
                          v_cache.data_ptr(), page.data_ptr(), slot.data_ptr(), m, _stream(qkv))
    return q_out


def pack_weight_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] BF16 weight to FP8 e4m3 with one scale per output row (absmax / 448)
    and pack it for DecodeLinearFP8: per 16-row group and pair of k16 blocks, each lane's
    16 bytes are its mma A fragment (pack_weight's element order) for block 2j, then 2j + 1."""
    n, k = w.shape
    if n % 16 or k % 32:
        raise ValueError(f"weight shape {tuple(w.shape)}: N must be a multiple of 16 and K of 32")
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
    q = (wf / scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    # q[g*16 + rh*8 + gid][kb2*32 + kbp*16 + kh*8 + tig*2 + e] -> [g][kb2][gid][tig][kbp][kh][rh][e]
    packed = (
        q.reshape(n // 16, 2, 8, k // 32, 2, 2, 4, 2)
        .permute(0, 3, 2, 6, 4, 5, 1, 7)
        .contiguous()
        .reshape(-1)
    )
    return packed, scale.contiguous()


class DecodeLinearFP8(DecodeLinear):
    """DecodeLinear with FP8 e4m3 weights (per-row scales): half the weight bytes per step.
    Approximate (class G): y = x @ (scale[:, None] * fp8(weight / scale[:, None])).T."""

    def __init__(self, weight: torch.Tensor):
        if weight.shape[0] % 32:
            raise ValueError("N must be a multiple of 32")
        super().__init__(weight)
        self.packed, self.scale = pack_weight_fp8(weight)

    def dequantized(self) -> torch.Tensor:
        """The weight this layer actually multiplies by, as FP32 (for checks)."""
        n, k = self.n, self.k
        q = (
            self.packed.view(n // 16, k // 32, 8, 4, 2, 2, 2, 2)
            .permute(0, 6, 2, 1, 4, 5, 3, 7)
            .reshape(n, k)
            .view(torch.float8_e4m3fn)
            .float()
        )
        return q * self.scale[:, None]

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != self.k:
            raise ValueError(f"expected a [M, {self.k}] bfloat16 input")
        m = x.shape[0]
        if not 1 <= m <= 64:
            raise ValueError("M must be between 1 and 64")
        x = x.contiguous()
        if out is None:
            out = torch.empty(m, self.n, dtype=torch.bfloat16, device=self.device)
        _core.decode_gemm_fp8(
            x.data_ptr(), self.packed.data_ptr(), self.scale.data_ptr(), out.data_ptr(),
            self.workspace.data_ptr(), self.counters.data_ptr(), m, self.n, self.k, self.sms,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        return out
