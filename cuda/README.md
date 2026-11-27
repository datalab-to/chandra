# chandra-cuda

An optional CUDA engine for Chandra OCR 2. Without this package Chandra runs exactly as before (vLLM or HF).
With it, `InferenceManager(method="cuda")` (or `chandra --method cuda`) runs the language model on the
engine's kernels in the same Python process. transformers still provides the processor, the vision
tower and the (t, h, w) positions.

## Use

```bash
CHANDRA_CUDA_ARCHS="80;86;89;90" pip install --no-build-isolation ./cuda   # default: the local GPU
chandra input.pdf out/ --method cuda
```

```python
from chandra.model import InferenceManager
mgr = InferenceManager(method="cuda")   # chandra_cuda.ocr.CudaGenerator with the defaults below
```

Requirements:
- The CUDA toolkit (12.8 or newer) and a C++20 compiler to build.
- vLLM in the environment: prefill uses vLLM's Gated DeltaNet chunk kernel, its FlashAttention-2 build and its CUTLASS INT8 GEMM.
- An sm_80 or newer GPU (Ampere and later). So far it has been measured on one sm_86 card.
- Overlap needs a driver with green contexts (CUDA 12.4+ driver API); without them the engine runs prefill and decode in turn.

## Defaults

`CudaGenerator()` uses the configuration that passed the full olmOCR-bench:

| part | format |
|---|---|
| decoder linear layers | INT8 weights, one scale per output row. Prefill is W8A8 (per-token INT8 activations, CUTLASS); decode is W8A16 (BF16 activations) |
| full-attention KV cache | INT8, one scale per (token, head) |
| Gated DeltaNet state | FP16 |
| LM head | exact: an INT8 pass with a rigorous bound, then BF16 for the few 128-row tiles the bound cannot exclude; the picks are the BF16 head's |
| speculation | MTP-3 with the checkpoint's MTP layer (FP8 weights, a 32K-token FP8 draft head). Drafts are verified, so they change only the speed |
| vision tower | transformers' BF16 tower with a fused q/k rotary (bitwise the same features) |

Other options: `weights="bf16" | "fp8" | "mxfp4" | "int4"`, `kv="bf16"`, `state="fp32"`, `mtp=False`,
`lm_head="bf16"`, `overlap=False`, `slots` (pages decoded at a time, default 16). The 4-bit formats did
not all pass the bench; see the pull request for the gates.

Page handling follows chandra's vLLM client:
- a page stops at an end token or after `max_output_tokens`;
- a page whose text repeats is generated again with sampling (temperature 0.2 per attempt, up to 0.8; top-p 0.95; up to `MAX_VLLM_RETRIES` times);
- a generation stops early once it repeats (`stop_on_repeat`), as #130 does for the vLLM client.

## How it works

- **Decode GEMMs** (`decode_gemm.cu`), for 1 to 64 rows:
  - Weights are pre-shuffled and streamed with 128-bit loads through a register ring with static slots. x goes through a `cp.async` double buffer and `ldmatrix`; the MMAs are `mma.sync.m16n8k16`.
  - Formats: BF16, FP8 e4m3, INT8, MXFP4 and INT4 (one scale per 128 along K), dequantized in registers with byte permutes and BF16 multiplies.
  - Fused epilogues: SiLU-and-mul for the MLP, and the LM head's argmax.
  - Plans (rows per CTA, K splits) depend only on the weight's shape. They are autotuned once per GPU and cached, so results do not change with the batch.
- **Attention** (`attn_decode.cu`): gated decode and verify attention for head dim 256 with GQA 16/4, BF16 or INT8 KV. Keys are split into fixed chunks, so a token's result does not depend on the batch.
- **Gated DeltaNet** (`gdn_decode.cu`):
  - One fused step per layer: causal conv, gates, delta rule, gated RMSNorm.
  - 1 to 4 tokens per sequence with a deferred commit, so an MTP verify step keeps the plain step's state.
- **Prefill**: cuBLAS or CUTLASS GEMMs, vLLM's chunked DeltaNet kernel and FlashAttention-2, plus fused kernels for the conv and the gated norm (`small_ops.cu`).
- **Graphs**: each decode step (verify plus drafts) is a CUDA graph. There is one capture per number of decoding pages, re-pointed at the current pages through device buffers.
- **Scheduling** (`ocr.py`):
  - Pages join and leave the decoding set continuously.
  - With overlap (`partition.py`), a worker thread runs vision and prefill on most of the SMs while the decode steps run on about 28% of them, and on the whole GPU when the worker has nothing to do.
  - Prefill uses private copies of the scratch buffers it shares with the decode graphs.

## Tests and benchmarks

```bash
pytest cuda/tests                                    # kernels against references; engine on a random checkpoint
python cuda/bench/bench_decode_gemm.py --rows 16 64  # GEMMs vs cuBLAS, per format
python cuda/bench/profile_step.py SAMPLE_DIR          # one decode step by kernel
python cuda/bench/engine_olmocr.py BENCH_DATA_DIR --candidate engine   # all olmOCR-bench pages
python -m olmocr.bench.benchmark --dir BENCH_DATA_DIR --candidate engine
```

`SAMPLE_DIR` holds page PNGs and a `manifest.json` (`[{"png": ...}, ...]`); `BENCH_DATA_DIR` is
olmOCR-bench's `bench_data` (see `chandra/scripts/olmocr_bench.py`).
