# Per-Token FP8 Quantization Benchmark

**Knowledge Sources:** benchmarks/kernels/bench_per_token_quant_fp8.py
**Domains:** Performance, Quantization, Benchmarking
**Last Updated:** 2025-12-17

## Overview

Comprehensive benchmarking tool for comparing per-token FP8 quantization implementations across Torch (compiled), CUDA, and Triton backends.

## Description

The Per-Token FP8 Quantization benchmark evaluates different implementations of the QuantFP8 layer with dynamic per-token quantization. This benchmark is critical for understanding which backend provides optimal performance for FP8 activation quantization across various configurations.

The script tests three implementation strategies:
1. **Torch (Compiled)**: PyTorch native implementation with torch.compile optimization
2. **CUDA**: Custom CUDA kernel implementation
3. **Triton**: Triton GPU kernel implementation

Key features:
- Supports multiple group shapes: PER_TENSOR, PER_TOKEN, and per-group (64, 128)
- Tests column-major vs row-major scale layouts
- Evaluates batch sizes from 1 to 1024 tokens
- Covers hidden dimensions from 896 to 7168
- Includes correctness validation before performance measurement
- Computes geometric mean speedups across configurations
- Supports multiple data types (half, bfloat16, float)

The benchmark uses torch.compile with dynamic first dimension marking to simulate realistic vLLM inference patterns. Results help guide backend selection decisions for quantized model deployment.

## Usage

Run the benchmark with customizable parameters:

```bash
# Default benchmark with correctness checking
python benchmarks/kernels/bench_per_token_quant_fp8.py --check

# Benchmark specific data type
python benchmarks/kernels/bench_per_token_quant_fp8.py --dtype bfloat16

# Custom dimensions
python benchmarks/kernels/bench_per_token_quant_fp8.py \
  --batch-sizes 1 16 128 512 1024 \
  --hidden-sizes 2048 4096 8192 \
  --group-sizes 0 -1 64 128
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/bench_per_token_quant_fp8.py`
**Function Signature:**
```python
def benchmark_quantization(
    batch_size,
    hidden_size,
    provider,
    group_shape: GroupShape,
    col_major: bool,
    dtype: torch.dtype
):
    """
    Benchmark quantization implementations

    Args:
        batch_size: Number of tokens
        hidden_size: Hidden dimension size
        provider: "torch", "cuda", or "triton"
        group_shape: Quantization grouping strategy
        col_major: Use column-major scale layout
        dtype: Tensor data type

    Returns:
        Tuple of (median_us, max_us, min_us) in microseconds
    """
```

**Import:**
```python
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `--check` | flag | Enable correctness validation mode |
| `--dtype` | str | Data type: "half", "bfloat16", or "float" (default: bfloat16) |
| `--batch-sizes` | List[int] | Batch sizes to test (default: [1, 16, 128, 512, 1024]) |
| `--hidden-sizes` | List[int] | Hidden dimensions (default: [896, 1024, 2048, 4096, 7168]) |
| `--group-sizes` | List[int] | Group sizes: 0=PER_TENSOR, -1=PER_TOKEN, N=per-group |
| `--no-column-major` | flag | Disable column-major scale testing |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Performance table | DataFrame | Timing results across configurations |
| Speedup table | DataFrame | Geometric mean speedups by group shape |
| Correctness status | str | Validation results when --check is used |
| Console output | str | Formatted benchmark results |

## Usage Examples

### Example 1: Basic Benchmark
```python
# Run default benchmark
python benchmarks/kernels/bench_per_token_quant_fp8.py
```

### Example 2: Correctness Validation
```python
# Validate all implementations match
python benchmarks/kernels/bench_per_token_quant_fp8.py --check
```

### Example 3: Custom Configuration
```python
# Test specific hidden sizes with per-token quantization only
python benchmarks/kernels/bench_per_token_quant_fp8.py \
  --dtype float16 \
  --hidden-sizes 4096 8192 \
  --group-sizes -1 \
  --batch-sizes 128 256 512
```

### Example 4: Programmatic Usage
```python
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
import torch

# Create quantization layer
quant_fp8 = QuantFP8(
    False,  # Not static quantization
    GroupShape.PER_TOKEN,
    column_major_scales=False
)

# Apply quantization
x = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
quantized, scales = quant_fp8.forward_cuda(x)
```

### Example 5: Geometric Mean Speedup Calculation
```python
from benchmarks.kernels.bench_per_token_quant_fp8 import compute_geomean_speedups
import pandas as pd

# Compute speedups over baseline
df = pd.DataFrame(results)
speedups = compute_geomean_speedups(
    df,
    baseline_col="Torch (Compiled)",
    speedup_cols=["CUDA", "Triton"],
    groupby_cols=["col_major", "group_shape"]
)
```

## Related Pages

- **File Detail:** [benchmarks_kernels_bench_per_token_quant_fp8_py.md](../files/benchmarks_kernels_bench_per_token_quant_fp8_py.md)
- **Quantization Layer:** vllm.model_executor.layers.quantization.input_quant_fp8.QuantFP8
- **Related Benchmark:** vllm-project_vllm_SiLUMulFP8QuantBenchmark.md
- **Utilities:** vllm.model_executor.layers.quantization.utils.quant_utils
- **Repository:** https://github.com/vllm-project/vllm
