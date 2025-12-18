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

**Purpose:** Benchmarks RoPE (Rotary Position Embedding) across torch, FlashInfer, and vLLM implementations.

**Mechanism:** Uses Triton benchmarking framework to compare performance across different batch sizes, sequence lengths, and head counts. Tests both neox-style and standard RoPE.

**Significance:** RoPE is fundamental to transformer models. Benchmarking helps optimize this frequently-called operation across implementations.
