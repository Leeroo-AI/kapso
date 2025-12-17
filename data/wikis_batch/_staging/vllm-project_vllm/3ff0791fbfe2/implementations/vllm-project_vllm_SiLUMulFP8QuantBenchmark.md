# SiLU-Mul-FP8 Quantization Benchmark

**Knowledge Sources:** benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py
**Domains:** Performance, Quantization, Benchmarking, Kernel Fusion
**Last Updated:** 2025-12-17

## Overview

Benchmarking tool that evaluates fused SiLU-multiply-quantize kernel performance against unfused implementations for LLM feedforward layers.

## Description

The SiLU-Mul-FP8 Quantization benchmark compares the performance of a fused kernel that combines three operations (SiLU activation, element-wise multiplication, and FP8 quantization) against a baseline implementation that executes these operations sequentially. This fusion is critical for MLP layer optimization in quantized LLM inference.

The benchmark evaluates two implementations:
1. **Fused Kernel**: `silu_mul_per_token_group_quant_fp8_colmajor` - Single kernel combining all operations
2. **Reference**: Sequential execution of `silu_and_mul` followed by `_per_token_group_quant_fp8_colmajor`

Key features:
- Tests across token counts ranging from 128 to 2048, plus extended range 2K-128K
- Evaluates hidden dimensions: 2048, 4096, and 8192
- Uses fixed group size of 128 for per-token quantization
- Employs CUDA graphs with argument pools (size 8) for realistic benchmarking
- Validates correctness before performance measurement
- Outputs timing comparisons using torch.benchmark utilities
- Supports deep GEMM E8M0 scale format when available

The fused implementation reduces memory traffic by eliminating intermediate buffer writes/reads and enables better instruction-level parallelism. This optimization is essential for high-throughput quantized inference.

## Usage

Execute the benchmark directly to test across predefined token and hidden dimension combinations:

```bash
# Run default benchmark
python benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py
```

The script automatically tests multiple configurations and reports timing results with comparisons between fused and unfused implementations.

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py`
**Function Signature:**
```python
def bench_impl(
    bench_tensors: list[BenchmarkTensors],
    impl_type: ImplType
) -> TMeasurement:
    """
    Benchmark a specific implementation using CUDA graphs

    Args:
        bench_tensors: List of pre-allocated tensor sets
        impl_type: Implementation to benchmark (FUSED or REFERENCE)

    Returns:
        torch.utils.benchmark.Measurement with timing results
    """
```

**Import:**
```python
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _per_token_group_quant_fp8_colmajor,
    silu_mul_per_token_group_quant_fp8_colmajor
)
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `T` (tokens) | List[int] | Token counts: [128-1920 by 128] + [2K-128K by 2K] |
| `N` (hidden) | List[int] | Hidden dimensions: [2048, 4096, 8192] |
| `GROUP_SIZE` | int | Fixed at 128 for quantization groups |
| `FLOAT8_T` | torch.dtype | torch.float8_e4m3fn for quantized output |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Timing table | TMeasurement | Detailed benchmark results per configuration |
| Comparison | TBenchmark.Compare | Fused vs reference performance comparison |
| Correctness status | str | Validation results (pass/fail) |
| Console output | str | Formatted timing and comparison data |

## Usage Examples

### Example 1: Run Default Benchmark
```python
# Execute benchmark with default configurations
python benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py
```

### Example 2: Programmatic Correctness Check
```python
from benchmarks.kernels.benchmark_2d_silu_mul_fp8_quant import test_correctness

# Validate correctness for specific dimensions
test_correctness(T=1024, N=4096)
# Output: "Testing num_tokens=1024, N=4096 ..."
```

### Example 3: Custom Benchmark Run
```python
from benchmarks.kernels.benchmark_2d_silu_mul_fp8_quant import run

# Define custom token and hidden dimension ranges
T = [256, 512, 1024, 2048]
N = [4096, 8192]

# Run with custom argument pool size
timers = run(T, N, arg_pool_size=16)
```

### Example 4: Using Fused Kernel Directly
```python
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_per_token_group_quant_fp8_colmajor
)
import torch

# Create input tensor
input = torch.randn((1024, 8192), dtype=torch.bfloat16, device="cuda")

# Allocate output tensor
output = torch.empty(
    (1024, 4096),
    dtype=torch.float8_e4m3fn,
    device="cuda"
)

# Apply fused operation
silu_mul_per_token_group_quant_fp8_colmajor(
    input=input,
    output=output,
    use_ue8m0=True  # Use deep GEMM scale format if available
)
```

### Example 5: Analyzing Performance Results
```python
import torch.utils.benchmark as TBenchmark

# After running benchmark, analyze results
timers = run([1024], [4096], arg_pool_size=8)

# Print detailed comparison
compare = TBenchmark.Compare(timers)
compare.print()
# Shows median time, IQR, and relative performance
```

## Related Pages

- **File Detail:** [benchmarks_kernels_benchmark_2d_silu_mul_fp8_quant_py.md](../files/benchmarks_kernels_benchmark_2d_silu_mul_fp8_quant_py.md)
- **FP8 Utilities:** vllm.model_executor.layers.quantization.utils.fp8_utils
- **Related Benchmark:** vllm-project_vllm_PerTokenFP8QuantBenchmark.md
- **Activation Functions:** vllm.model_executor.layers.activation
- **Repository:** https://github.com/vllm-project/vllm
