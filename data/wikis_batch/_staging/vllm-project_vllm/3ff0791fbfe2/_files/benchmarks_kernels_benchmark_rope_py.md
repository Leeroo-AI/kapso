# File: `benchmarks/kernels/benchmark_rope.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 106 |
| Functions | `get_benchmark` |
| Imports | itertools, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Compares RoPE implementations (PyTorch/FlashInfer/vLLM)

**Mechanism:** Three-way RoPE (Rotary Position Embedding) benchmark using Triton's perf_report framework. Compares forward_native (PyTorch), flashinfer_rotary_embedding, and forward_cuda (vLLM Triton) implementations. Tests across batch sizes (1-64), sequence lengths (64-512), and head counts (32/48). Supports partial rotary embedding via rope_parameters. Measures performance at quantiles (20%, 50%, 80%). Generates performance plots showing relative speeds. Tests both neox-style and non-neox RoPE variants.

**Significance:** RoPE is fundamental to modern transformers (LLaMA, Mistral, etc.) for position encoding. Applied to every query/key tensor in attention, making it performance-critical. Benchmark validates optimized implementations provide consistent speedup over naive PyTorch. Helps select best implementation for different workload characteristics. Essential for attention performance optimization.
