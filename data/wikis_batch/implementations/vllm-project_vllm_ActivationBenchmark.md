# Activation Function Benchmark

**Knowledge Sources:** benchmarks/kernels/benchmark_activation.py
**Domains:** Performance, Benchmarking, Kernel Optimization
**Last Updated:** 2025-12-17

## Overview

Benchmarking tool for comparing custom CUDA activation kernels against torch.compile implementations across various activation functions.

## Description

The Activation Benchmark evaluates performance of custom optimized CUDA kernels versus PyTorch's torch.compile for common LLM activation functions. This benchmark helps determine whether specialized kernels provide meaningful speedups over the compiled native implementations.

Supported activation functions:
- **silu_and_mul / mul_and_silu**: SiLU activation with element-wise multiplication
- **gelu_and_mul**: GELU activation with multiplication (none approximation)
- **gelu_and_mul_tanh**: GELU with tanh approximation
- **fatrelu_and_mul**: FATReLU (threshold-based ReLU) with multiplication
- **swigluoai_and_mul**: SwiGLU variant used by OpenAI
- **gelu_new**: New GELU approximation
- **gelu_fast**: Fast GELU approximation
- **quick_gelu**: Quick GELU variant

Key features:
- Tests batch sizes: 1, 16, 128
- Sequence lengths: 1, 16, 64, 1024, 4096
- Intermediate dimensions: 3072, 9728, 12288 (typical MLP sizes)
- Compares custom CUDA kernels vs torch.compile
- Uses triton.testing.do_bench_cudagraph for precise measurements
- Configurable data types (half, bfloat16, float)
- Leverages CustomOp registry for kernel instantiation

The benchmark is essential for understanding when custom implementations justify their maintenance cost versus relying on compiler optimizations.

## Usage

Run the benchmark for a specific activation function:

```bash
# Benchmark SiLU-and-mul (default)
python benchmarks/kernels/benchmark_activation.py

# Benchmark GELU with tanh approximation
python benchmarks/kernels/benchmark_activation.py \
  --func-name gelu_and_mul_tanh \
  --dtype bfloat16

# Benchmark FATReLU
python benchmarks/kernels/benchmark_activation.py \
  --func-name fatrelu_and_mul \
  --dtype float16
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_activation.py`
**Function Signature:**
```python
def benchmark_activation(
    batch_size: int,
    seq_len: int,
    intermediate_size: int,
    provider: str,
    func_name: str,
    dtype: torch.dtype
):
    """
    Benchmark activation function implementation

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        intermediate_size: MLP intermediate dimension
        provider: "custom" or "compiled"
        func_name: Name of activation function
        dtype: Tensor data type

    Returns:
        Tuple of (median_ms, max_ms, min_ms)
    """
```

**Import:**
```python
import vllm.model_executor.layers.activation
from vllm.model_executor.custom_op import CustomOp
from vllm.triton_utils import triton
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `--func-name` | str | Activation function name (default: "silu_and_mul") |
| `--dtype` | str | Data type: "half", "bfloat16", "float" (default: "bfloat16") |
| `batch_size` | int | Batch size from [1, 16, 128] |
| `seq_len` | int | Sequence length from [1, 16, 64, 1024, 4096] |
| `intermediate_size` | int | Hidden dimension from [3072, 9728, 12288] |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Performance plot | PNG | Visualization of custom vs compiled performance |
| Timing table | DataFrame | Median latency across all configurations |
| Console output | str | Formatted benchmark results |

## Usage Examples

### Example 1: Default Benchmark
```python
# Run SiLU-and-mul benchmark
python benchmarks/kernels/benchmark_activation.py
```

### Example 2: GELU Comparison
```python
# Compare GELU variants
python benchmarks/kernels/benchmark_activation.py --func-name gelu_and_mul
python benchmarks/kernels/benchmark_activation.py --func-name gelu_and_mul_tanh
python benchmarks/kernels/benchmark_activation.py --func-name gelu_fast
```

### Example 3: Custom Configuration
```python
# Benchmark with FP16
python benchmarks/kernels/benchmark_activation.py \
  --func-name swigluoai_and_mul \
  --dtype half
```

### Example 4: Programmatic Usage
```python
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
import torch

# Set up environment
current_platform.seed_everything(42)
device = "cuda"
torch.set_default_device(device)

# Create layer from registry
layer = CustomOp.op_registry["silu_and_mul"]()

# Create input tensor
num_tokens = 128
intermediate_size = 9728
x = torch.randn(num_tokens, intermediate_size, dtype=torch.bfloat16, device=device)

# Run activation
output = layer(x)
```

### Example 5: Using Compiled Version
```python
from vllm.model_executor.custom_op import CustomOp
import torch

# Get native implementation
layer = CustomOp.op_registry["gelu_and_mul"](approximate="none")

# Compile the native forward method
compiled_layer = torch.compile(layer.forward_native)

# Create input
x = torch.randn(128, 9728, dtype=torch.bfloat16, device="cuda")

# Run compiled version
output = compiled_layer(x)
```

### Example 6: FATReLU with Threshold
```python
from vllm.model_executor.custom_op import CustomOp

# Create FATReLU layer with custom threshold
threshold = 0.5
layer = CustomOp.op_registry["fatrelu_and_mul"](threshold)

# Apply to input
x = torch.randn(128, 9728, dtype=torch.bfloat16, device="cuda")
output = layer(x)
```

## Related Pages

- **File Detail:** [benchmarks_kernels_benchmark_activation_py.md](../files/benchmarks_kernels_benchmark_activation_py.md)
- **Activation Layers:** vllm.model_executor.layers.activation
- **Custom Operations:** vllm.model_executor.custom_op.CustomOp
- **Related Benchmark:** vllm-project_vllm_SiLUMulFP8QuantBenchmark.md
- **Repository:** https://github.com/vllm-project/vllm
