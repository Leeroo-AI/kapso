# File: `benchmarks/kernels/benchmark_device_communicators.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 508 |
| Classes | `CommunicatorBenchmark` |
| Functions | `print_results`, `main` |
| Imports | collections, contextlib, json, os, time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks distributed communication backends for tensor parallelism

**Mechanism:** Tests multiple allreduce implementations across different tensor sizes: CustomAllreduce (oneshot/twoshot), PyNcclCommunicator, PyNccl with symmetric memory, SymmMemCommunicator (multimem/twoshot). Uses CUDA graphs for accurate performance measurement, benchmarks with various sequence lengths (128-8192), and reports latency in milliseconds with speedup calculations vs PyNccl baseline.

**Significance:** Critical for distributed inference performance tuning. Helps select optimal communication backend for tensor parallel deployments. Enables validation of custom allreduce optimizations vs NCCL. Important for multi-GPU inference efficiency, especially on systems with NVLink/NVLS support.
