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

**Purpose:** Benchmarks different device communication implementations for all-reduce operations including CustomAllreduce (one-shot, two-shot), PyNcclCommunicator, and SymmMemCommunicator variants.

**Mechanism:** Uses distributed PyTorch with torchrun to benchmark all-reduce operations across multiple GPUs. Captures operations in CUDA graphs and measures latency for different sequence lengths and tensor sizes. Compares performance across communicator implementations and identifies the fastest approach.

**Significance:** Essential for evaluating and selecting the most efficient communication backend for tensor-parallel and pipeline-parallel operations in distributed inference. Helps optimize inter-GPU communication patterns which are critical bottlenecks in multi-GPU serving.
