# File: `unsloth/kernels/moe/benchmark/benchmark_fused_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 399 |
| Functions | `run_benchmark_forward`, `run_benchmark_backward`, `setup_model`, `run_benchmark` |
| Imports | argparse, contextlib, grouped_gemm, time, torch, transformers, triton, utils |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarking harness for comparing fused MoE (Mixture of Experts) kernel implementations against reference PyTorch implementations from HuggingFace transformers.

**Mechanism:**
- `run_benchmark_forward()`: Compares forward pass performance between reference and Triton-optimized models
- `run_benchmark_backward()`: Benchmarks backward pass with gradient computation
- `setup_model()`: Creates reference (Llama4TextMoe/Qwen3MoeSparseMoeBlock) and optimized (Triton GroupedGEMM) model pairs
- `run_benchmark()`: Orchestrates benchmarking for forward/backward/dW/dX modes with optional autotuning
- Supports command-line configuration of block sizes, warps, stages, TMA loading, and permutation options
- Saves autotuned kernel configurations and timing results to CSV files

**Significance:** Critical performance validation tool that quantifies speedups achieved by the custom Triton kernels over standard implementations. Enables systematic kernel configuration tuning and performance regression testing for Llama4 and Qwen3 MoE models across different sequence lengths and data types.
