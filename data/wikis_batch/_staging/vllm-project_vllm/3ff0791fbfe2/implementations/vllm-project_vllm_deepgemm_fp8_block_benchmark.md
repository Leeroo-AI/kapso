---
knowledge_sources:
  - vllm-project/vllm repository
  - File: benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py
domains:
  - Performance Benchmarking
  - FP8 Quantization
  - Matrix Multiplication
  - DeepGEMM
last_updated: 2025-12-17
---

# DeepGEMM FP8 Block Dense GEMM Benchmark

## Overview

Comprehensive benchmarking suite comparing DeepGEMM, vLLM Triton, and vLLM CUTLASS implementations of FP8 block-quantized dense matrix multiplication.

## Description

The `benchmark_fp8_block_dense_gemm.py` script provides detailed performance analysis of three different implementations of block-quantized FP8 GEMM operations:

1. **DeepGEMM**: NVIDIA's optimized FP8 GEMM with block quantization
2. **vLLM Triton**: Custom Triton kernel implementation
3. **vLLM CUTLASS**: CUTLASS-based FP8 GEMM with scaled matrix multiplication

Block quantization uses per-block scaling factors (typically 128x128 blocks) instead of per-tensor scales, providing better accuracy while maintaining most of FP8's performance benefits. This is critical for models that require fine-grained quantization like DeepSeekV3.

The benchmark:
- Tests various matrix shapes relevant to transformer models
- Measures execution time, TFLOPS, and memory bandwidth
- Calculates speedup ratios between implementations
- Validates accuracy against BF16 reference
- Generates formatted performance tables

## Usage

### Command Line Execution

```bash
# Run the full benchmark suite
python benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py
```

### Verbose Mode

```python
# Run with detailed output
if __name__ == "__main__":
    run_benchmarks(verbose=True)
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py`

**Main Function:**

```python
def benchmark_shape(
    m: int,                    # Batch size / rows
    n: int,                    # Output columns
    k: int,                    # Inner dimension
    warmup: int = 100,
    repeat: int = 10000,
    verbose: bool = False
) -> dict
```

**Helper Functions:**

```python
def format_table_row(values, widths):
    """Format a row with specified column widths"""

def print_table(headers, rows, title=None):
    """Print a table with headers and rows"""

def format_speedup(value):
    """Format speedup value with indicator"""

def run_benchmarks(verbose: bool = False):
    """Run benchmarks for a set of common shapes"""
```

**Import:**
```python
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)
from vllm.utils.deep_gemm import (
    calc_diff,
    fp8_gemm_nt,
    get_col_major_tma_aligned_tensor,
    per_block_cast_to_fp8,
)
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | `int` | Number of rows (batch size) |
| `n` | `int` | Number of output columns |
| `k` | `int` | Inner dimension (shared) |
| `warmup` | `int` | Number of warmup iterations (default: 100) |
| `repeat` | `int` | Number of timing iterations (default: 10000) |
| `verbose` | `bool` | Print detailed output (default: False) |

### Tensor Setup

```python
# Input matrices in BF16
A = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
B = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

# Block quantization (128x128 blocks)
B_fp8, B_scale = per_block_cast_to_fp8(B, [128, 128], use_ue8m0=True)
A_fp8, A_scale = per_token_group_quant_fp8(A, 128)

# Reference result
C_ref = A @ B.t()  # (m, n)
```

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `shape` | `dict` | Shape info: `{"m": m, "n": n, "k": k}` |
| `implementations` | `dict` | Performance data per implementation |

**Per-Implementation Data:**

```python
{
    "time_ms": float,         # Average execution time (ms)
    "time_us": float,         # Average execution time (us)
    "tflops": float,          # Throughput in TFLOPS
    "gb_s": float,            # Memory bandwidth (GB/s)
    "diff": {
        "DeepGEMM": float,    # Difference vs DeepGEMM
        "Reference": float    # Difference vs BF16 reference
    },
    "speedup_vs_deepgemm": float,  # Speedup ratio (if not baseline)
    "speedup_vs_triton": float,    # CUTLASS vs Triton speedup
}
```

### Performance Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Time (ms) | Average over iterations | Execution latency |
| TFLOPS | `2*m*n*k / time_s / 1e12` | Computational throughput |
| GB/s | `(m*k + k*n + m*n*2) / time_s / 1e9` | Memory bandwidth |
| Diff | `torch.mean(torch.abs(out - ref) / (torch.abs(ref) + 1e-5))` | Accuracy metric |

## Usage Examples

### Running Single Shape

```python
from benchmark_fp8_block_dense_gemm import benchmark_shape

# Test specific matrix shape
m, n, k = 64, 4096, 7168
result = benchmark_shape(m, n, k, warmup=50, repeat=1000, verbose=True)

# Access results
for impl_name, impl_data in result["implementations"].items():
    print(f"{impl_name}:")
    print(f"  Time: {impl_data['time_ms']:.3f} ms")
    print(f"  TFLOPS: {impl_data['tflops']:.2f}")
    print(f"  Accuracy: {impl_data['diff']['Reference']:.6f}")
```

### Comparing Implementations

```python
# Benchmark and compare
m, n, k = 128, 7168, 18432
result = benchmark_shape(m, n, k, repeat=5000)

deepgemm = result["implementations"]["DeepGEMM"]
triton = result["implementations"]["vLLM Triton"]
cutlass = result["implementations"]["vLLM CUTLASS"]

print(f"DeepGEMM: {deepgemm['time_ms']:.3f} ms, {deepgemm['tflops']:.1f} TFLOPS")
print(f"Triton: {triton['time_ms']:.3f} ms, {triton['tflops']:.1f} TFLOPS")
print(f"CUTLASS: {cutlass['time_ms']:.3f} ms, {cutlass['tflops']:.1f} TFLOPS")

print(f"\nCUTLASS speedup vs Triton: {triton['time_ms'] / cutlass['time_ms']:.2f}x")
print(f"DeepGEMM speedup vs CUTLASS: {cutlass['time_ms'] / deepgemm['time_ms']:.2f}x")
```

### Testing Multiple Shapes

```python
# Test suite of shapes
shapes = [
    (64, 4096, 7168),    # Small batch
    (128, 4096, 7168),   # Medium batch
    (1024, 4096, 7168),  # Large batch
    (4096, 4096, 7168),  # Very large batch
]

results = []
for m, n, k in shapes:
    result = benchmark_shape(m, n, k, repeat=2000)
    results.append(result)

    # Print summary
    deepgemm_time = result["implementations"]["DeepGEMM"]["time_ms"]
    cutlass_time = result["implementations"]["vLLM CUTLASS"]["time_ms"]
    speedup = cutlass_time / deepgemm_time

    print(f"Shape ({m}, {n}, {k}): DeepGEMM {speedup:.2f}x faster than CUTLASS")
```

### Accuracy Analysis

```python
# Check accuracy across implementations
m, n, k = 256, 7168, 16384
result = benchmark_shape(m, n, k, verbose=False)

print("Accuracy (mean absolute relative error):")
for impl_name, impl_data in result["implementations"].items():
    diff = impl_data["diff"]["Reference"]
    print(f"{impl_name:20s}: {diff:.6f}")

    # Check if within acceptable threshold
    threshold = 1e-3  # 0.1%
    status = "PASS" if diff < threshold else "FAIL"
    print(f"  Status: {status}")
```

### Performance Profiling

```python
import torch.profiler as profiler

m, n, k = 128, 4096, 7168

# Setup
A = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
B = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
B_fp8, B_scale = per_block_cast_to_fp8(B, [128, 128], use_ue8m0=True)
A_fp8, A_scale = per_token_group_quant_fp8(A, 128)

# Profile DeepGEMM
with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for _ in range(100):
        C = fp8_gemm_nt((A_fp8, A_scale), (B_fp8, B_scale),
                        torch.empty((m, n), device="cuda", dtype=torch.bfloat16))

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Batch Size Scaling

```python
# Test how performance scales with batch size
batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
n, k = 4096, 7168

deepgemm_times = []
cutlass_times = []

for m in batch_sizes:
    result = benchmark_shape(m, n, k, repeat=1000)

    deepgemm_time = result["implementations"]["DeepGEMM"]["time_us"]
    cutlass_time = result["implementations"]["vLLM CUTLASS"]["time_us"]

    deepgemm_times.append(deepgemm_time)
    cutlass_times.append(cutlass_time)

    print(f"Batch {m:4d}: DeepGEMM {deepgemm_time:6.1f}us, "
          f"CUTLASS {cutlass_time:6.1f}us")

# Plot scaling
import matplotlib.pyplot as plt
plt.plot(batch_sizes, deepgemm_times, marker='o', label='DeepGEMM')
plt.plot(batch_sizes, cutlass_times, marker='s', label='CUTLASS')
plt.xlabel('Batch Size')
plt.ylabel('Time (us)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.title('FP8 GEMM Performance Scaling')
plt.show()
```

### Block Size Comparison

```python
# Test different block sizes (requires modifying per_block_cast_to_fp8 calls)
block_sizes = [(64, 64), (128, 128), (256, 256)]
m, n, k = 128, 4096, 7168

for block_size in block_sizes:
    # Setup with specific block size
    A = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    B_fp8, B_scale = per_block_cast_to_fp8(B, list(block_size), use_ue8m0=True)
    A_fp8, A_scale = per_token_group_quant_fp8(A, block_size[1])

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        C = fp8_gemm_nt((A_fp8, A_scale), (B_fp8, B_scale),
                        torch.empty((m, n), device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    elapsed = time.time() - start

    time_ms = elapsed / 1000 * 1000
    print(f"Block {block_size}: {time_ms:.3f} ms")
```

## Related Pages

- [[vllm-project_vllm_deepgemm_fp8_gemm]] - DeepGEMM FP8 implementation
- [[vllm-project_vllm_w8a8_triton_block_scaled_mm]] - Triton block FP8 kernel
- [[vllm-project_vllm_cutlass_scaled_mm]] - CUTLASS scaled matmul
- [[vllm-project_vllm_fp8_block_quantization]] - Block FP8 quantization
- [[vllm-project_vllm_w8a8_block_fp8_tuner]] - W8A8 tuning framework
- [[vllm-project_vllm_per_token_group_quant]] - Per-token-group quantization
