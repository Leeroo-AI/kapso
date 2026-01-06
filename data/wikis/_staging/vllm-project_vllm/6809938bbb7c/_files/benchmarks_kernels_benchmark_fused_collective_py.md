# File: `benchmarks/kernels/benchmark_fused_collective.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1129 |
| Classes | `FlashInferFusedAllReduceParams`, `VllmFusedAllreduce` |
| Functions | `setup_flashinfer_workspace`, `cleanup_flashinfer_workspace`, `flashinfer_fused_allreduce_rmsnorm`, `flashinfer_fused_allreduce_rmsnorm_fp8_quant`, `flashinfer_fused_allreduce_rmsnorm_fp4_quant`, `create_test_tensors`, `benchmark_operation`, `run_benchmarks`, `... +5 more` |
| Imports | argparse, itertools, os, pandas, time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks FlashInfer's fused collective operations (all-reduce + rmsnorm + quantization) against standard separate operations for tensor-parallel workloads.

**Mechanism:** Compares FlashInfer's trtllm_allreduce_fusion kernel that fuses all-reduce, RMSNorm, and optional FP8/FP4 quantization against the standard approach of separate operations. Supports multiple quantization modes (none, FP8, FP4) and tests with/without residual connections across various token counts.

**Significance:** Critical for evaluating kernel fusion optimizations in distributed inference. Fusing all-reduce with normalization and quantization can significantly reduce memory bandwidth and improve performance in tensor-parallel scenarios, especially for large batch sizes typical in prefill workloads.
