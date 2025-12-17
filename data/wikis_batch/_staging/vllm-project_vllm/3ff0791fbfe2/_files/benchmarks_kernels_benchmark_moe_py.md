# File: `benchmarks/kernels/benchmark_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 790 |
| Classes | `BenchmarkConfig`, `BenchmarkWorker` |
| Functions | `ensure_divisibility`, `benchmark_config`, `get_rocm_tuning_space`, `get_configs_compute_bound`, `prune_rocm_search_space`, `prune_rocm_configs`, `need_split_k`, `merge_unique_dicts`, `... +4 more` |
| Imports | argparse, contextlib, datetime, itertools, json, os, ray, time, torch, typing, ... +1 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tunes and benchmarks MoE kernel configurations

**Mechanism:** Ray-based distributed tuning framework for MoE fused kernels. Extracts MoE parameters (num_experts, topk, intermediate_size) from model configs (Mixtral, DeepSeek, Qwen, Jamba, etc.). Supports FP16, FP8 W8A8, and INT8 W8A16 quantization with optional block quantization. Tunes across large search space of Triton kernel parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages, waves_per_eu for ROCm). Uses CUDA graphs with 10 invocations per replay for accurate timing. Prunes search space based on platform (ROCm vs CUDA) and matrix dimensions. Saves optimal configs as JSON with Triton version info. Supports DeepGEMM mode for block-quantized FP8.

**Significance:** Critical infrastructure for MoE performance optimization. Automatically finds best kernel configuration for each model and batch size. Essential for achieving competitive MoE inference performance across different hardware platforms. The tuned configs are used at runtime for optimal kernel selection. Key enabler for efficient mixture-of-experts inference in vLLM.
