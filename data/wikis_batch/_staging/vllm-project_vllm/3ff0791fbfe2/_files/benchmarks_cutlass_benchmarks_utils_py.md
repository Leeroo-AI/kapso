# File: `benchmarks/cutlass_benchmarks/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 100 |
| Functions | `to_fp8`, `to_int8`, `to_bf16`, `to_fp16`, `make_rand_tensors`, `prune_to_2_4`, `make_rand_sparse_tensors`, `make_n_rand_sparse_tensors` |
| Imports | collections, torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tensor utilities for CUTLASS benchmarks

**Mechanism:** Provides quantization converters (to_fp8, to_int8, to_bf16, to_fp16) using vLLM custom ops, random tensor generation with proper scaling, 2:4 structured pruning implementation, and batch sparse tensor creation. Handles proper clamping for quantized formats.

**Significance:** Shared utility library for all CUTLASS benchmark scripts. Ensures consistent tensor preparation and quantization methodology across different benchmark scenarios, critical for fair performance comparisons.
