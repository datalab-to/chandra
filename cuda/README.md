# chandra-cuda

Optional CUDA kernels for Chandra OCR: an engine that keeps a page's data on the GPU, built kernel by kernel. Without this package Chandra runs exactly as before (vLLM or HF).

Kernels so far:

- `DecodeLinear`: weight-streaming BF16 GEMM for decode batches of 1 to 64 rows (`mma.sync.m16n8k16`, pre-shuffled weights read with 128-bit streaming loads, x through a `cp.async` ring and `ldmatrix`). Its K split depends only on the weight's shape and the GPU, so results do not change with the batch size.

## Build

Needs the CUDA toolkit (12.8 or newer) and a C++20 compiler.

```bash
pip install scikit-build-core nanobind ninja
CHANDRA_CUDA_ARCHS="80;86;89;90" pip install --no-build-isolation ./cuda   # default: the local GPU
pytest cuda/tests
python cuda/bench/bench_decode_gemm.py
```
