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

**Purpose:** Benchmarks W8A8 block-wise FP8 quantized GEMM operations.

**Mechanism:** Tests FP8 matrix multiplication with block-quantized weights and activations, measuring performance across different block sizes and matrix dimensions.

**Significance:** Block quantization provides better accuracy than per-tensor while maintaining good performance. Important for evaluating FP8 quantization strategies.
