# File: `benchmarks/kernels/benchmark_moe_permute_unpermute.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 428 |
| Classes | `BenchmarkConfig`, `BenchmarkWorker` |
| Functions | `benchmark_permute`, `benchmark_unpermute`, `get_weight_block_size_safety`, `main` |
| Imports | argparse, ray, torch, transformers, typing, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks MoE permute and unpermute operations comparing standard implementation against customized variants with FP8/INT8 quantization support.

**Mechanism:** Uses Ray for distributed benchmarking. Tests permute (token-to-expert routing) and unpermute (expert-to-token aggregation) operations with CUDA graphs across different batch sizes. Supports both standard and customized permute implementations with optional FP8 quantization alignment.

**Significance:** Essential for optimizing MoE data movement operations which are critical for performance. Permute/unpermute overhead can significantly impact overall MoE throughput, especially with FP8 block alignment requirements for DeepGEMM.
