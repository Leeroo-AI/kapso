# File: `benchmarks/kernels/benchmark_w8a8_block_fp8.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 415 |
| Functions | `w8a8_block_matmul`, `get_configs_compute_bound`, `get_weight_shapes`, `benchmark_config`, `tune`, `save_configs`, `tune_on_gpu`, `distribute_batch_sizes`, `... +1 more` |
| Imports | argparse, datetime, json, multiprocessing, os, time, torch, tqdm, typing, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tunes W8A8 block FP8 GEMM configurations

**Mechanism:** Multi-GPU tuning framework for W8A8 block-quantized FP8 matrix multiplication. Extracts weight shapes from model configs (Llama, Mistral, etc.), generates Triton kernel search space (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages), and benchmarks w8a8_block_matmul across batch sizes. Uses multiprocessing to distribute tuning across GPUs. Tests block sizes 128x128 and 256x256 for weight/activation quantization. Saves optimal kernel configs as JSON with timestamp. Includes CUDA graph support for accurate timing.

**Significance:** Critical for W8A8 FP8 inference optimization. Block quantization (per-block scales instead of per-tensor) improves accuracy while maintaining FP8 speed benefits. Automated tuning finds best Triton kernel config for each model and batch size. Essential for achieving competitive W8A8 performance. Tuned configs used at runtime for optimal FP8 GEMM execution.
