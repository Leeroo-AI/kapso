# BitBLAS Quantized Kernel Benchmark

**Knowledge Sources:** benchmarks/kernels/benchmark_bitblas.py
**Domains:** Performance, Quantization, Benchmarking, Third-Party Integration
**Last Updated:** 2025-12-17

## Overview

Benchmarking tool for evaluating Microsoft's BitBLAS library performance on low-bit quantized matrix multiplication.

## Description

The BitBLAS benchmark evaluates performance of the BitBLAS library, a TVM-based framework for optimized low-bit quantized matrix operations. This benchmark helps assess BitBLAS as an alternative quantization backend for vLLM.

BitBLAS supports extremely aggressive quantization schemes including:
- **Weight types**: INT4, INT2, INT1, NF4 (4-bit NormalFloat), FP4 (4-bit Float)
- **Activation types**: FP16, FP32, FP64, INT32, INT8
- **Quantization features**: Group quantization, scaling factors, zero points (original/rescale/quantized modes)
- **Matrix layouts**: NT (non-transpose A, transpose B) and NN

Key features:
- Tests comprehensive shape sets covering square matrices and real model dimensions
- Includes BLOOM-176B, OPT-65B, and LLAMA-70B/65B layer shapes
- Tests both small (M=1) and large (M=8192) batch sizes
- Supports per-channel and group-wise quantization
- Enables auto-tuning for optimal kernel selection
- Requires BitBLAS >= 0.0.1.dev14
- Reports latency in milliseconds

The benchmark evaluates whether BitBLAS provides competitive performance for extreme quantization scenarios (sub-4-bit) where native vLLM kernels may not be optimized.

## Usage

Run the benchmark with customizable quantization configuration:

```bash
# Default INT4 benchmark
python benchmarks/kernels/benchmark_bitblas.py

# INT2 quantization with group size 128
python benchmarks/kernels/benchmark_bitblas.py \
  --W_dtype int2 \
  --group_size 128 \
  --with_scaling \
  --with_zeros

# NF4 (NormalFloat4) quantization
python benchmarks/kernels/benchmark_bitblas.py \
  --W_dtype nf4 \
  --A_dtype float16 \
  --group_size 64

# FP4 quantization with custom target
python benchmarks/kernels/benchmark_bitblas.py \
  --W_dtype fp4_e2m1 \
  --target cuda_sm90
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_bitblas.py`
**Import:**
```python
import bitblas
from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target
from vllm.model_executor.layers.quantization.utils.bitblas_utils import (
    MINIMUM_BITBLAS_VERSION
)
```

**Configuration:**
```python
# Example MatmulConfig creation
config = MatmulConfig(
    M=1,                    # Batch size
    N=16384,               # Output dimension
    K=16384,               # Input dimension
    A_dtype="float16",     # Activation type
    W_dtype="int4",        # Weight type
    out_dtype="float16",   # Output type
    accum_dtype="float16", # Accumulation type
    layout="nt",           # Matrix layout
    with_bias=False,       # Include bias
    group_size=128,        # Group quantization size
    with_scaling=True,     # Use scaling factors
    with_zeros=True,       # Use zero points
    zeros_mode="original"  # Zero point mode
)
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `--target` | str | CUDA target (default: auto-detect) |
| `--group_size` | int | Group size for quantization (default: None) |
| `--A_dtype` | str | Activation dtype (default: "float16") |
| `--W_dtype` | str | Weight dtype (default: "int4") |
| `--accum_dtype` | str | Accumulation dtype (default: "float16") |
| `--out_dtype` | str | Output dtype (default: "float16") |
| `--layout` | str | Matrix layout: "nt" or "nn" (default: "nt") |
| `--with_bias` | flag | Include bias term |
| `--with_scaling` | flag | Enable scaling factors |
| `--with_zeros` | flag | Enable zero points |
| `--zeros_mode` | str | Zero point mode: "original", "rescale", "quantized" |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Latency table | str | Formatted table with operation latencies |
| Profile results | dict | Per-configuration timing in milliseconds |
| Console output | str | Formatted benchmark results |

## Usage Examples

### Example 1: Default INT4 Benchmark
```python
# Run with default settings
python benchmarks/kernels/benchmark_bitblas.py
```

### Example 2: INT2 with Zero Points
```python
# Extreme 2-bit quantization
python benchmarks/kernels/benchmark_bitblas.py \
  --W_dtype int2 \
  --group_size 64 \
  --with_scaling \
  --with_zeros \
  --zeros_mode quantized
```

### Example 3: NormalFloat4 (NF4)
```python
# NF4 quantization (used in QLoRA)
python benchmarks/kernels/benchmark_bitblas.py \
  --W_dtype nf4 \
  --group_size 128 \
  --with_scaling
```

### Example 4: Programmatic Usage
```python
from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target

# Configure INT4 matmul
config = MatmulConfig(
    M=1,
    N=16384,
    K=16384,
    A_dtype="float16",
    W_dtype="int4",
    out_dtype="float16",
    accum_dtype="float16",
    layout="nt",
    group_size=128,
    with_scaling=True
)

# Create and tune matmul operator
target = auto_detect_nvidia_target()
matmul = Matmul(config, target=target, enable_tuning=True)

# Profile performance
latency = matmul.profile_latency()
print(f"Latency: {latency:.3f} ms")
```

### Example 5: Multiple Configurations
```python
import bitblas
from bitblas import Matmul, MatmulConfig

# Test matrix for different weight types
weight_types = ["int4", "int2", "nf4", "fp4_e2m1"]
results = {}

for wtype in weight_types:
    config = MatmulConfig(
        M=8192,
        N=16384,
        K=16384,
        A_dtype="float16",
        W_dtype=wtype,
        out_dtype="float16",
        group_size=128,
        with_scaling=True
    )

    matmul = Matmul(config, enable_tuning=True)
    latency = matmul.profile_latency()
    results[wtype] = latency
    print(f"{wtype}: {latency:.3f} ms")
```

## Related Pages

- **File Detail:** [benchmarks_kernels_benchmark_bitblas_py.md](../files/benchmarks_kernels_benchmark_bitblas_py.md)
- **BitBLAS Utilities:** vllm.model_executor.layers.quantization.utils.bitblas_utils
- **Related Benchmark:** vllm-project_vllm_MacheteBenchmark.md (vLLM's native low-bit kernels)
- **External Dependency:** https://github.com/microsoft/BitBLAS
- **Repository:** https://github.com/vllm-project/vllm
