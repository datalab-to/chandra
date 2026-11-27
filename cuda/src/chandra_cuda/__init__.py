"""Optional CUDA kernels for Chandra OCR.

The kernels run on torch's current CUDA stream and take torch tensors; the
compiled module (`_core`) only sees device pointers.
"""

from __future__ import annotations

import torch

from . import _core

__all__ = ["pack_weight", "pack_weight_fp8", "DecodeLinear", "DecodeLinearFP8", "DecodeLinearFP4", "DecodeLinearINT4", "mxfp4_quantize", "int4_g128_quantize",
           "pack_weight_fp4", "pack_weight_i8", "interleave_gate_up", "autotune_plans", "LMHead", "LMHeadFP8", "LMHeadExact", "attn_decode", "gdn_decode", "rmsnorm_add", "silu_mul",
           "qk_norm_rope_kv", "vit_attn"]


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
    commit: bool = False,
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

    commit=True (speculative decoding with one state slot per sequence): slot_out gets the state
    after the replay, the committed one, and not the state after this step's tokens; hist_out may
    be hist_in. Each step then replays the tokens accepted from the previous one.
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
        tmax, bool(commit), torch.cuda.current_stream(qkv.device).cuda_stream,
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
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    split_keys: int | None = None,
) -> torch.Tensor:
    """Gated decode/verify attention for Chandra 2's full-attention layers.

    split_keys (a multiple of 32): every sequence's keys are split into chunks of this size, so a
    query's result depends only on its own context, not on the batch or max_ctx (batch-invariant,
    and a verify step rounds as plain steps do). None: chunks sized to fill the GPU for this batch.

    With k_scale, v_scale ([pages, 4, 64] FP32), k_cache and v_cache hold INT8: x = int8 * scale
    per (token, head); the kernel dequantizes to FP16 per stage and runs FP16 MMAs.

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
    if split_keys is None:
        sps, max_splits = _core.attn_decode_split(b, max_ctx, sms)
    else:
        if split_keys % 32:
            raise ValueError("split_keys must be a multiple of 32")
        sps = split_keys // 32
        max_splits = max(1, -(-max_ctx // split_keys))
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
        b, n_q, sps, max_splits,
        0 if k_scale is None else k_scale.data_ptr(), 0 if v_scale is None else v_scale.data_ptr(),
        torch.cuda.current_stream(dev).cuda_stream,
    )
    return out


def deinterleave_gate_up(w: torch.Tensor) -> torch.Tensor:
    """Inverse of interleave_gate_up."""
    f = w.shape[0] // 2
    v = w.reshape(f // 8, 2, 8, -1)
    return torch.cat([v[:, 0].reshape(f, -1), v[:, 1].reshape(f, -1)])


def interleave_gate_up(w: torch.Tensor) -> torch.Tensor:
    """[gate; up] (2F rows) -> 16-row groups of 8 gate rows then the same 8 up rows, the row
    order of the decode GEMM's silu_mul epilogue."""
    f = w.shape[0] // 2
    return torch.stack([w[:f].reshape(f // 8, 8, -1), w[f:].reshape(f // 8, 8, -1)], dim=1).reshape(2 * f, -1)


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

    def __init__(self, weight: torch.Tensor, _pack: bool = True, epilogue: str | None = None):
        """epilogue="silu_mul": weight is [gate; up] (2F rows) and the call returns
        silu(x @ gate.T) * (x @ up.T), [M, F], exactly as silu_mul of the two BF16 outputs."""
        if not weight.is_cuda:
            raise ValueError("weight must be on a CUDA device")
        self.n, self.k = weight.shape
        self.device = weight.device
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        if self.k % 128:
            raise ValueError("K must be a multiple of 128")
        if epilogue not in (None, "silu_mul"):
            raise ValueError(f"unknown epilogue {epilogue!r}")
        self.epilogue = 1 if epilogue == "silu_mul" else 0
        if self.epilogue:
            if self.n % 16:
                raise ValueError("silu_mul needs gate and up of a multiple of 8 rows each")
            weight = interleave_gate_up(weight)
        self.n_out = self.n // 2 if self.epilogue else self.n
        self._prep = None if _pack else weight  # for the quantized subclasses
        self.packed = pack_weight(weight) if _pack else None
        self._replan()

    FMT = 0  # the plan registry's key for this weight format

    def unpacked(self) -> torch.Tensor:
        """The [N, K] BF16 weight this layer multiplies by, in the original row order (gate before
        up for silu_mul), for prefill GEMMs. Inverse of pack_weight."""
        n, k = self.n, self.k
        w = (
            self.packed.view(n // 16, k // 16, 8, 4, 2, 2, 2)  # g kb gid tig kh rh e
            .permute(0, 5, 2, 1, 4, 3, 6)
            .reshape(n, k)
        )
        return deinterleave_gate_up(w) if self.epilogue else w

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """x [M, K] -> [M, N] BF16 for any M (prefill), cuBLAS on the unpacked weight (original row
        order, so gate and up stay separate under silu_mul)."""
        return x @ self.unpacked().T

    def _replan(self):
        """Size the workspace and counters for the current plan of (N, K, format)."""
        ws = _core.decode_gemm_workspace_bytes(self.n, self.k, self.sms, self.FMT)
        self.workspace = torch.empty(max(ws // 4, 1), dtype=torch.float32, device=self.device)
        self.counters = torch.zeros(
            _core.decode_gemm_counters(self.n, self.k, self.sms, self.FMT), dtype=torch.int32, device=self.device
        )
        self.plan = _core.decode_gemm_plan_fmt(self.n, self.k, self.sms, self.FMT)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != self.k:
            raise ValueError(f"expected a [M, {self.k}] bfloat16 input")
        m = x.shape[0]
        if not 1 <= m <= 64:
            raise ValueError("M must be between 1 and 64")
        x = x.contiguous()
        if out is None:
            out = torch.empty(m, self.n_out, dtype=torch.bfloat16, device=self.device)
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
            self.epilogue,
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
    """[M, 2F] gate | up (contiguous) -> [M, F] bf16(silu(gate) * up); F = 9,216 in the MLP."""
    m, f2 = gate_up.shape
    if f2 % 16 or not gate_up.is_contiguous():
        raise ValueError("gate_up must be contiguous [M, 2F] with F a multiple of 8")
    if out is None:
        out = torch.empty(m, f2 // 2, dtype=torch.bfloat16, device=gate_up.device)
    _core.silu_mul(gate_up.data_ptr(), out.data_ptr(), m, _stream(gate_up), f2 // 2)
    return out


def qk_norm_rope_kv(qkv: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor, pos: torch.Tensor,
                    k_cache: torch.Tensor, v_cache: torch.Tensor, page: torch.Tensor, slot: torch.Tensor,
                    rope_theta: float = 1e7, q_out: torch.Tensor | None = None,
                    k_scale: torch.Tensor | None = None, v_scale: torch.Tensor | None = None) -> torch.Tensor:
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
                          v_cache.data_ptr(), page.data_ptr(), slot.data_ptr(), m,
                          0 if k_scale is None else k_scale.data_ptr(), 0 if v_scale is None else v_scale.data_ptr(),
                          _stream(qkv))
    return q_out


def pack_weight_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] BF16 weight to FP8 e4m3 with one scale per output row (absmax / 448)
    and pack it for DecodeLinearFP8: per 16-row group and pair of k16 blocks, each lane's
    16 bytes are its mma A fragment (pack_weight's element order) for block 2j, then 2j + 1."""
    n, k = w.shape
    if n % 16 or k % 32:
        raise ValueError(f"weight shape {tuple(w.shape)}: N must be a multiple of 16 and K of 32")
    scale = torch.empty(n, dtype=torch.float32, device=w.device)
    q = torch.empty(n, k, dtype=torch.uint8, device=w.device)
    for i in range(0, n, 16384):  # in row blocks: an FP32 copy of an LM head would be 2.5 GB
        wf = w[i : i + 16384].float()
        scale[i : i + 16384] = (wf.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
        q[i : i + 16384] = (wf / scale[i : i + 16384, None]).to(torch.float8_e4m3fn).view(torch.uint8)
        del wf
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

    FMT = 1

    def __init__(self, weight: torch.Tensor, epilogue: str | None = None):
        if weight.shape[0] % 32:
            raise ValueError("N must be a multiple of 32")
        super().__init__(weight, _pack=False, epilogue=epilogue)  # shapes and the split workspace only
        w, self._prep = self._prep, None
        self.packed, self.scale = pack_weight_fp8(w)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """As the decode kernel: x times the e4m3 values (exact in BF16), then the per-row scale."""
        n, k = self.n, self.k
        q = (
            self.packed.view(n // 16, k // 32, 8, 4, 2, 2, 2, 2)
            .permute(0, 6, 2, 1, 4, 5, 3, 7)
            .reshape(n, k)
            .view(torch.float8_e4m3fn)
            .to(torch.bfloat16)
        )
        sc = self.scale
        if self.epilogue:
            q, sc = deinterleave_gate_up(q), deinterleave_gate_up(sc[:, None])[:, 0]
        return ((x @ q.T).float() * sc[None, :]).to(torch.bfloat16)

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
            out = torch.empty(m, self.n_out, dtype=torch.bfloat16, device=self.device)
        _core.decode_gemm_fp8(
            x.data_ptr(), self.packed.data_ptr(), self.scale.data_ptr(), out.data_ptr(),
            self.workspace.data_ptr(), self.counters.data_ptr(), m, self.n, self.k, self.sms,
            torch.cuda.current_stream(self.device).cuda_stream, self.epilogue,
        )
        return out


def vit_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_len: int,
             out: torch.Tensor | None = None, version: int = 1) -> torch.Tensor:
    """Non-causal attention of the vision tower over packed images: q, k, v [N, 16, 64] BF16 views
    sharing a row stride (e.g. the thirds of the fused qkv GEMM output), cu_seqlens [n + 1] int32.
    Returns [N, 16, 64] BF16 (scale 1/8)."""
    n = q.shape[0]
    if q.shape[1:] != (16, 64) or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must be [N, 16, 64]")
    if not (q.stride(0) == k.stride(0) == v.stride(0)) or q.stride(1) != 64 or q.stride(2) != 1:
        raise ValueError("q, k, v must share a row stride with contiguous heads")
    if out is None:
        out = torch.empty(n, 16, 64, dtype=torch.bfloat16, device=q.device)
    k8 = k_scale = k_mean = None
    if version in (3, 5):  # INT8 QK^T (approximate): per-image K mean, per-tile K scales, per-row Q scales
        n_seq = cu_seqlens.shape[0] - 1
        k8 = torch.empty(n, 16, 64, dtype=torch.int8, device=q.device)
        k_scale = torch.empty(n_seq, (int(max_len) + 63) // 64, 16, dtype=torch.float32, device=q.device)
        k_mean = torch.empty(n_seq, 16, 64, dtype=torch.float32, device=q.device)
    _core.vit_attn(q.data_ptr(), k.data_ptr(), v.data_ptr(), q.stride(0), out.data_ptr(), out.stride(0),
                   cu_seqlens.data_ptr(), cu_seqlens.shape[0] - 1, int(max_len), int(version),
                   0 if k8 is None else k8.data_ptr(), 0 if k_scale is None else k_scale.data_ptr(),
                   0 if k_mean is None else k_mean.data_ptr(), _stream(q))
    return out


class LMHeadFP8:
    """Greedy tokens from FP8 e4m3 weights (per-row scales, pack_weight_fp8): half an LM head's
    bytes, for MTP drafts (the verify step decides what is kept, so drafts may be approximate).
    rows: optional token ids to keep (a draft vocabulary); returned ids are always vocabulary ids.
    The kept rows are padded to a multiple of 128 by repeating the first (ties go to the lowest)."""

    def __init__(self, weight: torch.Tensor, rows: torch.Tensor | None = None):
        self.device = weight.device
        if rows is not None:
            rows = rows.to(self.device, torch.int64)
            pad = (-rows.numel()) % 128
            if pad:
                rows = torch.cat([rows, rows[:1].expand(pad)])
            weight = weight[rows]
        self.rows = rows
        self.n, self.k = weight.shape
        if self.n % 128 or self.k % 128:
            raise ValueError("vocabulary and K must be multiples of 128")
        self.packed, self.scale = pack_weight_fp8(weight)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        if x.dtype != torch.bfloat16 or x.shape[1] != self.k or not 1 <= m <= 64:
            raise ValueError(f"expected a [1..64, {self.k}] bfloat16 input")
        x = x.contiguous()
        keys = torch.zeros(m, dtype=torch.int64, device=self.device)
        _core.decode_gemm_argmax_fp8(x.data_ptr(), self.packed.data_ptr(), self.scale.data_ptr(), keys.data_ptr(),
                                     m, self.n, self.k, torch.cuda.current_stream(self.device).cuda_stream)
        idx = 0xFFFFFFFF - (keys & 0xFFFFFFFF)
        return idx if self.rows is None else self.rows[idx]


def mxfp4_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """OCP MX FP4 along K, bit for bit as vLLM's online `mxfp4` (triton_kernels' downcast_to_mxfp):
    per 32 weights E8M0 = floor(log2(amax, its FP32 mantissa rounded off)) - 2, codes
    e2m1 (sign << 3 | e1 e0 m) rounded to nearest even and saturated at 6. Returns (codes [N, K]
    uint8 0..15, E [N, K / 32] uint8, the scale being 2^(E - 127))."""
    n, k = w.shape
    if k % 32:
        raise ValueError("K must be a multiple of 32")
    codes = torch.empty(n, k, dtype=torch.uint8, device=w.device)
    exps = torch.empty(n, k // 32, dtype=torch.uint8, device=w.device)
    for i in range(0, n, 8192):
        f = w[i : i + 8192].float().reshape(-1, k // 32, 32)
        amax = f.abs().amax(dim=-1)
        mb = ((amax.view(torch.int32).to(torch.int64) + 0x200000) & 0x7F800000).to(torch.int32).view(torch.float32)
        unb = torch.floor(torch.log2(mb + torch.where(mb == 0, 2.0**-126, 0.0))) - 2
        scale = torch.exp2(unb.clamp(-127, 127))
        e = (scale.view(torch.int32) >> 23) & 0xFF
        qs = torch.where(scale == 0, 0.0, 1.0 / scale)
        u = (f * qs[..., None]).reshape(-1, k).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        sign = (u >> 31) & 1
        ex = (u >> 23) & 0xFF
        man = u & 0x7FFFFF
        adj = (126 - ex).clamp(0, 40)
        folded = 0x400000 | (man >> 1)
        den = folded >> adj
        den = den | ((den << adj) != folded).to(torch.int64)
        man = torch.where(ex < 127, den, man)
        ex2 = ex.clamp_min(126) - 126
        rm = (man + ((man >> 22) & 1) + 0x1FFFFF) >> 22
        code = torch.minimum((ex2 << 1) + rm, torch.tensor(7, device=w.device)) | (sign << 3)
        codes[i : i + 8192] = code.to(torch.uint8)
        exps[i : i + 8192] = e.to(torch.uint8)
    return codes, exps


def mxfp4_dequantize(codes: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    """FP32 weights of (codes, E) from mxfp4_quantize."""
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
    v = lut[codes.long()]
    return v * torch.exp2(exps.float() - 127).repeat_interleave(32, dim=1)


def pack_weight_fp4(codes: torch.Tensor, exps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out MXFP4 (codes, E) for DecodeLinearFP4. Per 16-row group g, 64-wide K load L and lane
    (gid, tig): four 32-bit words, one per k16 block, each the lane's 8 A-fragment weights with the
    magnitude bits of a0 / a1 / a2 / a3 at bits 8:6 / 11:9 / 2:0 / 5:3 of each half and their
    signs at 15 / 14 / 13 / 12 (the kernel's fp4x8_to_bf16). Scales: per (g, L, gid) one word of
    E + 126 for (gid, K block 0), (gid, 1), (gid + 8, 0), (gid + 8, 1)."""
    n, k = codes.shape
    if n % 16 or k % 64:
        raise ValueError("N must be a multiple of 16 and K of 64")
    if int(exps.max()) > 128:
        raise ValueError("a scale above 2 (E > 128) does not fit the BF16 multiplier 2^(E - 1)")
    c = codes.to(torch.int64).reshape(n // 16, 2, 8, k // 64, 4, 2, 4, 2)  # g rh gid L bb kh tig e
    mag, sgn = c & 7, c >> 3
    mbase = torch.tensor([[6, 0], [9, 3]], device=codes.device)  # [rh][kh]
    sbase = torch.tensor([[15, 13], [14, 12]], device=codes.device)
    e16 = torch.tensor([0, 16], device=codes.device)
    mshift = (mbase[:, None, None, None, :, None, None] + e16).view(1, 2, 1, 1, 1, 2, 1, 2)
    sshift = (sbase[:, None, None, None, :, None, None] + e16).view(1, 2, 1, 1, 1, 2, 1, 2)
    word = ((mag << mshift) | (sgn << sshift)).sum(dim=(1, 5, 7))  # [g, gid, L, bb, tig] (bits are disjoint)
    packed = word.permute(0, 2, 1, 4, 3).contiguous()  # [g, L, gid, tig, bb]
    packed = (packed & 0xFFFFFFFF).to(torch.int64)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32).reshape(-1)
    e = (exps.to(torch.int64) + 126).reshape(n // 16, 2, 8, k // 64, 2)  # g rh gid L blk
    sw = (e[:, 0, :, :, 0] | (e[:, 0, :, :, 1] << 8) | (e[:, 1, :, :, 0] << 16) | (e[:, 1, :, :, 1] << 24))
    sw = sw.permute(0, 2, 1).contiguous()  # [g, L, gid]
    sw = torch.where(sw >= 2**31, sw - 2**32, sw).to(torch.int32).reshape(-1)
    return packed, sw


class DecodeLinearFP4(DecodeLinear):
    """DecodeLinear with MXFP4 weights (e2m1, one power-of-two scale per 32 along K, as vLLM's
    online `mxfp4`): 4.25 bits per weight, a quarter of BF16's bytes. Approximate (class G):
    y = x @ mxfp4_dequantize(mxfp4_quantize(weight)).T. N must be a multiple of 64."""

    FMT = 2

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("prefill from 4-bit layers is not written yet (their accuracy gate is pending)")

    def __init__(self, weight: torch.Tensor, keep_codes: bool = False, epilogue: str | None = None):
        if weight.shape[0] % 64:
            raise ValueError("N must be a multiple of 64")
        super().__init__(weight, _pack=False, epilogue=epilogue)
        w, self._prep = self._prep, None
        codes, exps = mxfp4_quantize(w)
        self.codes, self.exps = (codes, exps) if keep_codes else (None, None)  # for dequantized()
        self.packed, self.scales = pack_weight_fp4(codes, exps)

    def dequantized(self) -> torch.Tensor:
        if self.codes is None:
            raise ValueError("built without keep_codes=True")
        return mxfp4_dequantize(self.codes, self.exps)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != self.k:
            raise ValueError(f"expected a [M, {self.k}] bfloat16 input")
        m = x.shape[0]
        if not 1 <= m <= 64:
            raise ValueError("M must be between 1 and 64")
        x = x.contiguous()
        if out is None:
            out = torch.empty(m, self.n_out, dtype=torch.bfloat16, device=self.device)
        _core.decode_gemm_fp4(x.data_ptr(), self.packed.data_ptr(), self.scales.data_ptr(), out.data_ptr(),
                              self.workspace.data_ptr(), self.counters.data_ptr(), m, self.n, self.k, self.sms,
                              torch.cuda.current_stream(self.device).cuda_stream, self.epilogue)
        return out


def pack_weight_i8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """INT8 per-row weights for the exact LM head: q = round(w / s), s = absmax / 127, stored as
    sign-magnitude bytes in pack_weight_fp8's layout. Also returns, per row and 256-wide K block b,
    c_n,b with |x . w_n - s_n (x . q_n)| + (both FP32 accumulations' error) <= sum_b ||x_b||_2 c_n,b:
    c_n,b = ||w_n,b - s_n q_n,b||_2 (Cauchy-Schwarz per block, from FP64) + 1e-3 ||w_n,b||_2
    (accumulation; 2 K u per sum is 3.1e-4 at K = 2,560, covering truncating accumulators),
    rounded up. K must be a multiple of 256."""
    n, k = w.shape
    if n % 16 or k % 256:
        raise ValueError(f"weight shape {tuple(w.shape)}: N must be a multiple of 16 and K of 256")
    nb = k // 256
    scale = torch.empty(n, dtype=torch.float32, device=w.device)
    rowc = torch.empty(n, nb, dtype=torch.float32, device=w.device)
    q8 = torch.empty(n, k, dtype=torch.uint8, device=w.device)
    for i in range(0, n, 16384):
        wf = w[i : i + 16384].double()
        s = (wf.abs().amax(dim=1) / 127.0).float().clamp_min(1e-30)
        q = torch.round(wf / s.double()[:, None]).clamp(-127, 127)
        r = (wf - q * s.double()[:, None]).reshape(-1, nb, 256)
        c = r.norm(dim=2) * (1 + 1e-6) + 1e-3 * wf.reshape(-1, nb, 256).norm(dim=2) + 1e-30
        scale[i : i + 16384] = s
        rowc[i : i + 16384] = c.float() * (1 + 2**-20)
        qi = q.to(torch.int16)
        q8[i : i + 16384] = torch.where(qi < 0, 128 - qi, qi).to(torch.uint8)
        del wf, q, r, qi
    packed = (
        q8.reshape(n // 16, 2, 8, k // 32, 2, 2, 4, 2)
        .permute(0, 3, 2, 6, 4, 5, 1, 7)
        .contiguous()
        .reshape(-1)
    )
    return packed, scale, rowc


class LMHeadExact:
    """The greedy pick of LMHead (BF16 logits, ties to the lowest index), bit for bit, reading
    INT8 weights for the whole vocabulary and BF16 weights only for the 128-row tiles that a
    rigorous error bound cannot exclude (usually one or a few per token). The BF16 packed
    weight stays in memory for those re-checks."""

    def __init__(self, weight: torch.Tensor, bf16_head: LMHead | None = None):
        self.n, self.k = weight.shape
        if self.n % 128 or self.k % 128:
            raise ValueError("vocabulary and K must be multiples of 128")
        self.device = weight.device
        self.bf16 = bf16_head if bf16_head is not None else LMHead(weight)
        self.packed8, self.scale, self.rowc = pack_weight_i8(weight)
        self.tile_ub = torch.empty(self.n // 128, 64, dtype=torch.float32, device=self.device)

    def bounds(self, x: torch.Tensor):
        """First pass only: (tile_ub [tiles, M], best lower bound [M]) as FP32."""
        m = x.shape[0]
        if x.dtype != torch.bfloat16 or x.shape[1] != self.k or not 1 <= m <= 64:
            raise ValueError(f"expected a [1..64, {self.k}] bfloat16 input")
        x = x.contiguous()
        xnorm = (x.float().reshape(m, self.k // 256, 256).norm(dim=2) * (1 + 1e-3)).contiguous()
        lb = torch.zeros(m, dtype=torch.int32, device=self.device)
        _core.lm_head_bound_i8(x.data_ptr(), self.packed8.data_ptr(), self.scale.data_ptr(), self.rowc.data_ptr(),
                               xnorm.data_ptr(), self.tile_ub.data_ptr(), lb.data_ptr(), m, self.n, self.k,
                               torch.cuda.current_stream(self.device).cuda_stream)
        return x, lb

    def candidate_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Per token, how many tiles the re-check reads (for measurements)."""
        x, lb = self.bounds(x)
        m = x.shape[0]
        lbu = lb.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        bits = torch.where(lbu >= 2**31, lbu - 2**31, 0xFFFFFFFF - lbu)  # ord -> float bits
        lbf = bits.to(torch.int64).to(torch.int32).view(torch.float32)
        ub = self.tile_ub[:, :m]
        return (~(ub.to(torch.bfloat16).float() < lbf.to(torch.bfloat16).float()[None, :])).sum(dim=0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x, lb = self.bounds(x)
        m = x.shape[0]
        keys = torch.zeros(m, dtype=torch.int64, device=self.device)
        _core.lm_head_recheck(x.data_ptr(), self.bf16.packed.data_ptr(), self.tile_ub.data_ptr(), lb.data_ptr(),
                              keys.data_ptr(), m, self.n, self.k, torch.cuda.current_stream(self.device).cuda_stream)
        return 0xFFFFFFFF - (keys & 0xFFFFFFFF)


def int4_g128_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT4 with one BF16 scale per 128 along K, as compressed-tensors' minmax observer
    (and bench/make_ct_checkpoint.py --scheme w4a16): s = bf16(amax / 7.5), q = clamp(round(w / s),
    -8, 7). Returns (q [N, K] int8, s [N, K / 128] BF16); the weight is bf16(q * s)."""
    n, k = w.shape
    if k % 128:
        raise ValueError("K must be a multiple of 128")
    q = torch.empty(n, k, dtype=torch.int8, device=w.device)
    s = torch.empty(n, k // 128, dtype=torch.bfloat16, device=w.device)
    for i in range(0, n, 8192):
        wf = w[i : i + 8192].float().reshape(-1, k // 128, 128)
        sc = (wf.abs().amax(dim=2) / 7.5).clamp_min(1e-12).to(torch.bfloat16)
        q[i : i + 8192] = torch.round(wf / sc.float()[..., None]).clamp(-8, 7).to(torch.int8).reshape(-1, k)
        s[i : i + 8192] = sc
    return q, s


def pack_weight_int4(q: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out INT4 g128 weights for DecodeLinearINT4: per 16-row group, 64-wide K load and lane, four
    words (one per k16 block) with a_j's two codes (q + 8) at bits 4j and 16 + 4j; scales per (group,
    128-wide K group, gid) as one word s(gid) | s(gid + 8) << 16."""
    n, k = q.shape
    if n % 16 or k % 128:
        raise ValueError("N must be a multiple of 16 and K of 128")
    u = (q.to(torch.int64) + 8).reshape(n // 16, 2, 8, k // 64, 4, 2, 4, 2)  # g rh gid L bb kh tig e
    j = torch.tensor([[0, 2], [1, 3]], device=q.device)  # [rh][kh] -> register a_j
    shift = (4 * j[:, None, None, None, :, None, None] + torch.tensor([0, 16], device=q.device)).view(
        1, 2, 1, 1, 1, 2, 1, 2)
    word = (u << shift).sum(dim=(1, 5, 7))  # [g, gid, L, bb, tig]
    packed = word.permute(0, 2, 1, 4, 3).contiguous()  # [g, L, gid, tig, bb]
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32).reshape(-1)
    sb = s.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF  # [n, k / 128]
    sb = sb.reshape(n // 16, 2, 8, k // 128)
    sw = (sb[:, 0] | (sb[:, 1] << 16)).permute(0, 2, 1).contiguous()  # [g, G, gid]
    sw = torch.where(sw >= 2**31, sw - 2**32, sw).to(torch.int32).reshape(-1)
    return packed, sw


class DecodeLinearINT4(DecodeLinear):
    """DecodeLinear with INT4 weights, one BF16 scale per 128 along K (the W4A16 g128 format of
    make_ct_checkpoint.py --scheme w4a16): 4.125 bits per weight. Approximate (class G):
    y = x @ bf16(q * s).T. N must be a multiple of 64."""

    FMT = 3

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("prefill from 4-bit layers is not written yet (their accuracy gate is pending)")

    def __init__(self, weight: torch.Tensor, keep_codes: bool = False, epilogue: str | None = None):
        if weight.shape[0] % 64:
            raise ValueError("N must be a multiple of 64")
        super().__init__(weight, _pack=False, epilogue=epilogue)
        w, self._prep = self._prep, None
        q, s = int4_g128_quantize(w)
        self.q, self.s = (q, s) if keep_codes else (None, None)
        self.packed, self.scales = pack_weight_int4(q, s)

    def dequantized(self) -> torch.Tensor:
        if self.q is None:
            raise ValueError("built without keep_codes=True")
        return (self.q.float() * self.s.float().repeat_interleave(128, dim=1)).to(torch.bfloat16).float()

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != self.k:
            raise ValueError(f"expected a [M, {self.k}] bfloat16 input")
        m = x.shape[0]
        if not 1 <= m <= 64:
            raise ValueError("M must be between 1 and 64")
        x = x.contiguous()
        if out is None:
            out = torch.empty(m, self.n_out, dtype=torch.bfloat16, device=self.device)
        _core.decode_gemm_int4(x.data_ptr(), self.packed.data_ptr(), self.scales.data_ptr(), out.data_ptr(),
                               self.workspace.data_ptr(), self.counters.data_ptr(), m, self.n, self.k, self.sms,
                               torch.cuda.current_stream(self.device).cuda_stream, self.epilogue)
        return out


def _plan_cache_path() -> str:
    import os

    d = os.environ.get("CHANDRA_CUDA_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "chandra_cuda"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "decode_gemm_plans.json")


def autotune_plans(layers, rows=(1, 16, 64), reps=20, cache: bool = True) -> dict:
    """Choose each decode GEMM's plan (rows per CTA, K splits) by timing a fixed candidate set on
    this GPU, once per (GPU, N, K, weight format), and register it before use. The plan depends
    only on the shape, the format and the GPU, never on M, so results stay batch-invariant; the
    choice is cached (CHANDRA_CUDA_CACHE, default ~/.cache/chandra_cuda) so later runs reuse it.
    layers: DecodeLinear objects (any format). Returns {(N, K, fmt): (rw, splits)}."""
    import json

    path = _plan_cache_path()
    table = {}
    if cache:
        try:
            table = json.load(open(path))
        except (OSError, ValueError):
            table = {}
    chosen = {}
    for lin in layers:
        dev = lin.device
        prop = torch.cuda.get_device_properties(dev)
        key = f"{prop.name}|{lin.sms}|{lin.n}|{lin.k}|{lin.FMT}"
        if (lin.n, lin.k, lin.FMT) in chosen:
            lin._replan()
            continue
        if key not in table:
            chunks = lin.k // 128
            cands = [(rw, s) for rw in (8, 4) if lin.n % (16 * rw) == 0 and not (lin.FMT == 2 and rw < 4)
                     for s in (1, 2, 3, 4, 6, 8) if chunks % s == 0 and (lin.n // (16 * rw)) * s <= 4 * lin.sms]
            xs = {m: torch.randn(m, lin.k, device=dev).to(torch.bfloat16) for m in rows}
            out = {m: torch.empty(m, lin.n_out, device=dev, dtype=torch.bfloat16) for m in rows}
            times = {}
            for rw, s in cands:
                _core.set_decode_gemm_plan(lin.n, lin.k, rw, s, lin.FMT)
                lin._replan()
                tot = 0.0
                for m in rows:
                    for _ in range(3):
                        lin(xs[m], out[m])
                    torch.cuda.synchronize(dev)
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(reps):
                        lin(xs[m], out[m])
                    b.record()
                    torch.cuda.synchronize(dev)
                    tot += a.elapsed_time(b) / reps
                times[(rw, s)] = tot
            best = min(times, key=lambda c: (times[c], c))  # ties: the smaller plan tuple
            table[key] = list(best)
        rw, s = table[key]
        _core.set_decode_gemm_plan(lin.n, lin.k, rw, s, lin.FMT)
        lin._replan()
        chosen[(lin.n, lin.k, lin.FMT)] = (rw, s)
    if cache:
        try:
            json.dump(table, open(path, "w"), indent=1, sort_keys=True)
        except OSError:
            pass
    return chosen

