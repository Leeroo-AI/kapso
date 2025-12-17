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

**Purpose:** Comprehensive LoRA kernel benchmarking framework

**Mechanism:** Extensive benchmark suite for LoRA operations including lora_shrink, lora_expand, and fused_moe_lora variants (gate_up/down, shrink/expand). Tests multiple configurations: different batch sizes, hidden sizes, LoRA ranks, number of LoRAs, sorted vs unsorted LoRA indices, and sequence lengths. Supports three benchmark modes: list_bench (explicit dimensions), range_bench (dimension ranges), and model_bench (from model configs). Includes correctness testing against reference group GEMM implementation and uses CUDA graphs for accurate timing. Compares against torch.mm baseline as roofline performance.

**Significance:** Critical tool for LoRA kernel optimization and validation. Enables evaluation of Triton LoRA kernels across diverse workloads. Essential for understanding LoRA operation performance characteristics and identifying optimization opportunities. Supports both standard LoRA and MoE-LoRA operations, making it key for multi-adapter and mixture-of-experts LoRA scenarios.
