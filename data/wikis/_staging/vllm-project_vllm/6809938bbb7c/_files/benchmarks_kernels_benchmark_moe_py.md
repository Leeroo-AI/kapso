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

**Purpose:** Comprehensive MoE (Mixture of Experts) kernel benchmarking and tuning framework supporting various quantization modes (FP16, FP8 W8A8, INT8 W8A16) and DeepGEMM optimizations.

**Mechanism:** Uses Ray for distributed benchmarking across multiple GPUs. Supports two modes: (1) benchmarking with existing configurations, (2) tuning to find optimal Triton kernel parameters. Captures operations in CUDA graphs and measures latency for different batch sizes and expert configurations extracted from actual model architectures.

**Significance:** Essential infrastructure for MoE kernel optimization used in models like Mixtral, DeepSeek-V2/V3, Qwen-MoE, and others. Enables systematic performance tuning and validation of MoE implementations across different hardware and quantization schemes. Critical for production deployment of MoE models.
