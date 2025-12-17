# File: `benchmarks/kernels/bench_per_token_quant_fp8.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 270 |
| Functions | `with_triton_mode`, `with_dyn_arg`, `bench_compile`, `calculate_diff`, `benchmark_quantization`, `compute_geomean_speedups` |
| Imports | collections, itertools, pandas, torch, unittest, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark per-token FP8 quantization

**Mechanism:** Compares three implementations: Torch (compiled), CUDA, and Triton for QuantFP8 layer. Tests various group shapes (per-tensor/token/group with sizes 64/128), column-major scales, batch sizes (1-1024), hidden sizes (896-7168), and dtypes. Includes correctness checking and geometric mean speedup calculations.

**Significance:** Implementation comparison for dynamic FP8 quantization. Per-token quantization provides better accuracy than per-tensor for activations. Essential for validating different backend choices (compiled vs custom kernels) and understanding performance across configurations.
