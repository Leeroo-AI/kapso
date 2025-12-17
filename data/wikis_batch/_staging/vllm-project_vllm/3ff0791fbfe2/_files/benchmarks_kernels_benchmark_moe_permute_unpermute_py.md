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

**Purpose:** Benchmarks MoE permute/unpermute operations separately

**Mechanism:** Ray-based distributed benchmark comparing moe_permute vs _moe_permute and moe_unpermute vs _moe_unpermute_and_reduce implementations. Tests both standard and customized permute paths. Supports FP8 W8A8 quantization with 128-aligned blocks for DeepGEMM. Uses CUDA graphs (10 invocations per graph) for accurate timing. Extracts model config (num_experts, topk, hidden_size) from various MoE architectures (Mixtral, DeepSeek, Qwen, Jamba, etc.). Reports separate timing for permute and unpermute operations.

**Significance:** Critical for evaluating MoE permute/unpermute kernel performance in isolation. These operations rearrange tokens before/after expert processing and can be performance bottlenecks. Helps validate optimizations like customized permute path and alignment requirements for quantized MoE. Essential for understanding MoE overhead breakdown.
