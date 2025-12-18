# File: `benchmarks/kernels/benchmark_lora.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1488 |
| Classes | `OpType`, `BenchmarkContext`, `BenchmarkTensors` |
| Functions | `dtype_to_str`, `make_rand_lora_weight_tensor`, `make_rand_tensors`, `make_prompt_lora_mapping`, `make_token_lora_mapping`, `ref_group_gemm`, `bench_optype`, `bench_torch_mm`, `... +7 more` |
| Imports | argparse, collections, copy, dataclasses, enum, itertools, json, pathlib, pickle, time, ... +5 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive benchmark suite for LoRA (Low-Rank Adaptation) operations including shrink, expand, and fused MoE LoRA kernels with various configurations and optimizations.

**Mechanism:** Benchmarks LoRA matrix operations across different batch sizes, hidden dimensions, LoRA ranks, and number of adapters. Tests both regular LoRA kernels (shrink/expand) and fused MoE LoRA variants (gate_up/down projections) with CUDA graph optimizations. Includes correctness testing against reference implementations.

**Significance:** Essential for evaluating LoRA adapter performance which enables efficient multi-tenant serving with parameter-efficient fine-tuning. Supports both standard dense LoRA and mixture-of-experts LoRA configurations critical for scalable adapter serving.
