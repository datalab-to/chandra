"""Two SM partitions of one GPU as CUDA streams (green contexts, CUDA 12.4+ driver API).

A memory-bound decode needs few SMs to stream its weights near full bandwidth, while a compute-bound
prefill or vision pass uses every SM it gets. Kernels launched into a partition's stream run only on
that partition's SMs, so the two can run side by side without one waiting behind the other's blocks.
Every engine kernel merges split work with last-arriver tickets (no block waits for another), so they
are correct on any number of SMs, and their plans depend on shapes only, so results are unchanged.

    parts = sm_partitions(torch.device("cuda"), small_sms=16)
    if parts: (small, small_stream), (large, large_stream) = parts
"""

from __future__ import annotations

import ctypes
import sys

import torch

_KEEP: list = []  # green contexts live as long as the process (their streams are in use)


class _Resource(ctypes.Structure):  # CUdevResource: type, 92 reserved bytes, a 48-byte union (sm.smCount first)
    _fields_ = [("type", ctypes.c_int), ("_reserved", ctypes.c_ubyte * 92), ("sm_count", ctypes.c_uint),
                ("_union_rest", ctypes.c_ubyte * 44)]


_CU_DEV_RESOURCE_TYPE_SM = 1
_CU_GREEN_CTX_DEFAULT_STREAM = 1
_CU_STREAM_NON_BLOCKING = 1


def _driver():
    try:
        return ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    except OSError:
        return None


def sm_partitions(device: torch.device, small_sms: int):
    """((SMs, stream) of a partition of at least small_sms SMs, (SMs, stream) of the remaining SMs), or
    None when the driver has no green contexts (before CUDA 12.4) or the split fails."""
    cu = _driver()
    if cu is None:
        return None
    ver = ctypes.c_int()
    if cu.cuDriverGetVersion(ctypes.byref(ver)) != 0 or ver.value < 12040:
        return None
    torch.zeros(1, device=device)  # the primary context exists and is current
    dev = ctypes.c_int()
    full, small, rest, groups = _Resource(), _Resource(), _Resource(), ctypes.c_uint(1)
    idx = device.index if device.index is not None else torch.cuda.current_device()
    try:
        if (cu.cuDeviceGet(ctypes.byref(dev), idx) != 0
                or cu.cuDeviceGetDevResource(dev, ctypes.byref(full), _CU_DEV_RESOURCE_TYPE_SM) != 0
                or cu.cuDevSmResourceSplitByCount(ctypes.byref(small), ctypes.byref(groups), ctypes.byref(full),
                                                  ctypes.byref(rest), 0, small_sms) != 0
                or groups.value != 1 or rest.sm_count == 0):
            return None
        out = []
        for res in (small, rest):
            desc, ctx, stream = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
            if (cu.cuDevResourceGenerateDesc(ctypes.byref(desc), ctypes.byref(res), 1) != 0
                    or cu.cuGreenCtxCreate(ctypes.byref(ctx), desc, dev, _CU_GREEN_CTX_DEFAULT_STREAM) != 0
                    or cu.cuGreenCtxStreamCreate(ctypes.byref(stream), ctx, _CU_STREAM_NON_BLOCKING, 0) != 0):
                return None
            _KEEP.append(ctx)
            out.append((res.sm_count, torch.cuda.ExternalStream(stream.value, device=device)))
        return tuple(out)
    except AttributeError:  # a driver library without the green-context entry points
        return None
