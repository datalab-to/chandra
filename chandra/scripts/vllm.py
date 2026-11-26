import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from chandra.settings import settings

# H100 80GB is the baseline for scaling
BASELINE_VRAM_GB = 80
BASELINE_MAX_BATCHED_TOKENS = 8192
BASELINE_MAX_NUM_SEQS = 64

GPU_MEMORY_UTILIZATION = 0.85
# Weights (10.6 GB in BF16) plus activations and CUDA graphs.
RESERVED_GB = 13.5
# Cache per page in flight: full-attention KV (8 layers x 32 KB per token for
# ~8.5K tokens of image, prompt and output) plus the Gated DeltaNet and conv
# state (~51 MB).
PAGE_CACHE_GB = 0.33

GPU_VRAM_GB = {
    "b200": 180,
    "h200": 141,
    "gh200": 96,
    "h100": 80,
    "a100-80": 80,
    "a100": 40,
    "a100-40": 40,
    "l40s": 48,
    "a6000": 48,
    "rtx5090": 32,
    "5090": 32,
    "a10": 24,
    "l4": 24,
    "rtx4090": 24,
    "4090": 24,
    "3090": 24,
    "t4": 16,
}


def get_gpu_settings(gpu: str):
    vram = GPU_VRAM_GB.get(gpu)
    if vram is None:
        available = ", ".join(sorted(GPU_VRAM_GB.keys()))
        print(f"Unknown GPU '{gpu}'. Available: {available}")
        sys.exit(1)

    ratio = vram / BASELINE_VRAM_GB
    raw_tokens = BASELINE_MAX_BATCHED_TOKENS * ratio
    max_batched_tokens = max(1024, 2 ** math.floor(math.log2(raw_tokens)))
    max_num_seqs = max(8, (int(BASELINE_MAX_NUM_SEQS * ratio) // 8) * 8)

    # Scaling linearly from the baseline undersizes mid-size GPUs: on 48 GB the
    # KV cache holds ~80 pages, but the linear rule allows 32. Allow as many
    # pages as the cache holds, up to the baseline's count; never fewer than
    # the linear rule gives.
    cache_pages = int((vram * GPU_MEMORY_UTILIZATION - RESERVED_GB) / PAGE_CACHE_GB)
    cache_seqs = min(BASELINE_MAX_NUM_SEQS, (cache_pages // 8) * 8)
    max_num_seqs = max(max_num_seqs, cache_seqs)
    # A ViT image is ~6K tokens; chunks smaller than the baseline's only add steps.
    if vram >= 40:
        max_batched_tokens = max(max_batched_tokens, BASELINE_MAX_BATCHED_TOKENS)

    return max_batched_tokens, max_num_seqs


def detect_container_runtime(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    for candidate in ("docker", "podman"):
        if shutil.which(candidate):
            return candidate
    return "docker"


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server for Chandra")
    parser.add_argument(
        "--gpu",
        default="h100",
        choices=sorted(GPU_VRAM_GB.keys()),
        help="GPU type for optimal settings (default: h100)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the container port to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Host port to bind the vLLM server to (default: 8000)",
    )
    parser.add_argument(
        "--runtime",
        default="auto",
        choices=["auto", "docker", "podman"],
        help="Container runtime to use (default: auto-detect docker or podman)",
    )
    parser.add_argument(
        "--sudo",
        action="store_true",
        default=False,
        help="Run the container command via sudo",
    )
    parser.add_argument(
        "--mtp",
        action="store_true",
        default=False,
        help="Enable MTP speculative decoding (disabled by default, unstable with vLLM)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override the pages in flight per GPU (default: sized from --gpu)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Override the tokens per scheduler step (default: sized from --gpu)",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=None,
        help="Model replicas, one per GPU (default: the number of GPUs in VLLM_GPUS)",
    )
    args = parser.parse_args()

    max_batched_tokens, max_num_seqs = get_gpu_settings(args.gpu)
    if args.max_num_seqs is not None:
        max_num_seqs = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        max_batched_tokens = args.max_num_batched_tokens
    gpu_ids = [g.strip() for g in str(settings.VLLM_GPUS).split(",") if g.strip()]
    data_parallel_size = args.data_parallel_size or max(1, len(gpu_ids))
    runtime = detect_container_runtime(args.runtime)
    dtype = "float16" if args.gpu == "t4" else "bfloat16"

    cmd = []
    if args.sudo:
        cmd.append("sudo")
    cmd.extend([runtime, "run"])

    if runtime == "podman":
        for gid in gpu_ids:
            cmd.extend(["--device", f"nvidia.com/gpu={gid}"])
    else:
        cmd.extend(["--runtime", "nvidia", "--gpus", f"device={settings.VLLM_GPUS}"])

    cmd.extend(
        [
            "-v",
            f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
            "-p",
            f"{args.host}:{args.port}:8000",
            "--ipc=host",
            "vllm/vllm-openai:v0.17.0",
            "--model",
            settings.MODEL_CHECKPOINT,
            "--no-enforce-eager",
            "--max-num-seqs",
            str(max_num_seqs),
            "--dtype",
            dtype,
            "--max-model-len",
            "18000",
            "--max_num_batched_tokens",
            str(max_batched_tokens),
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--enable-prefix-caching",
            "--mm-processor-kwargs",
            json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
            "--served-model-name",
            settings.VLLM_MODEL_NAME,
        ]
    )

    # One replica per GPU: pages are independent, so two GPUs give about
    # twice the pages per second without any cross-GPU traffic.
    if data_parallel_size > 1:
        cmd.extend(["--data-parallel-size", str(data_parallel_size)])

    if args.mtp:
        spec_config = json.dumps({"method": "mtp", "num_speculative_tokens": 1})
        cmd.extend(["--speculative-config", spec_config])

    vram = GPU_VRAM_GB[args.gpu]
    print(f"GPU: {args.gpu} ({vram}GB VRAM, dtype={dtype})")
    print(f"max-num-batched-tokens: {max_batched_tokens}, max-num-seqs: {max_num_seqs}")
    print(f"GPUs: {','.join(gpu_ids)} (data-parallel-size {data_parallel_size})")
    print(f"Listening on: http://{args.host}:{args.port}/v1")
    print(f"MTP: {'enabled' if args.mtp else 'disabled'}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nvLLM server exited with error code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
