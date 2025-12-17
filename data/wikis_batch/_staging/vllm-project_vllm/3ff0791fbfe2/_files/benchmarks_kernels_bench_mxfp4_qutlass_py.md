# File: `benchmarks/kernels/bench_mxfp4_qutlass.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 191 |
| Functions | `get_hadamard_matrix`, `build_mxfp4_runner`, `benchmark`, `prepare_shapes` |
| Imports | argparse, compressed_tensors, copy, itertools, torch, vllm, weight_shapes |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark MXFP4 quantized GEMM

**Mechanism:** Tests MX (Microscaling) FP4 format using CUTLASS kernels with Hadamard transforms (32/64/128 sizes). Compares BF16 baseline against MXFP4 with/without activation quantization. Tests Llama-3.3-70B shapes across extended batch range (1-32K). Uses fusedQuantizeMx for weight/activation quantization.

**Significance:** Performance evaluation for extreme quantization (4-bit). MX format combines block exponents with low-precision mantissas for better accuracy than pure INT4. Critical for understanding 4-bit inference feasibility and performance characteristics.
