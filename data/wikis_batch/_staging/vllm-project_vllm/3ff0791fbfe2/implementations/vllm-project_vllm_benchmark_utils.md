---
knowledge_sources:
  - vllm-project/vllm repository
  - File: benchmarks/kernels/utils.py
domains:
  - Performance Benchmarking
  - Testing Infrastructure
  - CUDA Graphs
  - Kernel Optimization
last_updated: 2025-12-17
---

# Benchmark Utility Classes

## Overview

Infrastructure classes and utilities for consistent and accurate kernel benchmarking across vLLM, including support for CUDA graphs and argument rotation patterns.

## Description

The `utils.py` module provides foundational benchmarking utilities used throughout vLLM's performance testing suite. It addresses common benchmarking challenges:

1. **Cache effects**: ArgPool pattern rotates through different argument values to prevent artificial speedups
2. **CUDA graphs**: CudaGraphBenchParams enables benchmarking kernel performance within CUDA graphs
3. **Measurement consistency**: Standardized timing and measurement interfaces
4. **Parameter management**: Automatic handling of argument iteration and rotation

Key components:
- **CudaGraphBenchParams**: Configuration for CUDA graph benchmarking
- **ArgPool**: Pool of argument values that rotate during benchmarking
- **Bench**: Main benchmarking class with timing and execution logic
- **ArgsIterator**: Iterator for cycling through argument combinations

## Usage

### Importing Utilities

```python
from benchmarks.kernels.utils import (
    Bench,
    ArgPool,
    CudaGraphBenchParams,
)
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/utils.py`

**Classes:**

```python
@dataclasses.dataclass
class CudaGraphBenchParams:
    num_ops_in_cuda_graph: int

@dataclasses.dataclass
class ArgPool:
    values: Iterable[Any]

    def __getitem__(self, index):
        return self.values[index]

class Bench:
    def __init__(
        self,
        cuda_graph_params: CudaGraphBenchParams | None,
        label: str,
        sub_label: str,
        description: str,
        fn: Callable,
        *args,
        **kwargs
    )

    def run(self) -> TMeasurement
```

**Nested Iterator Class:**

```python
class Bench.ArgsIterator:
    def __init__(self, args_list, kwargs_list)
    def __next__(self)
    def reset(self)
    @property
    def n_args(self) -> int
```

**Import:**
```python
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
```

## I/O Contract

### CudaGraphBenchParams

| Field | Type | Description |
|-------|------|-------------|
| `num_ops_in_cuda_graph` | `int` | Number of kernel operations to include in CUDA graph |

### ArgPool

| Field | Type | Description |
|-------|------|-------------|
| `values` | `Iterable[Any]` | Collection of values to rotate through during benchmarking |

**Methods:**
- `__getitem__(index)`: Access value at index

### Bench

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cuda_graph_params` | `CudaGraphBenchParams \| None` | CUDA graph config or None for eager mode |
| `label` | `str` | Primary label for benchmark |
| `sub_label` | `str` | Secondary label for configuration |
| `description` | `str` | Detailed description of benchmark |
| `fn` | `Callable` | Function to benchmark |
| `*args` | `Any` | Positional arguments (can include ArgPool) |
| `**kwargs` | `Any` | Keyword arguments (can include ArgPool) |

**Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `run()` | `TMeasurement` | Execute benchmark and return timing measurement |
| `run_eager()` | `TMeasurement` | Run in eager mode (no CUDA graph) |
| `run_cudagraph()` | `TMeasurement` | Run with CUDA graph |
| `collapse_argpool(*args, **kwargs)` | `tuple[list, list]` | Expand ArgPool into concrete argument lists |
| `get_cuda_graph_runner()` | `torch.cuda.CUDAGraph` | Create CUDA graph for benchmark |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `min_run_time` | `float` | Minimum benchmark run time (default: 1.0s) |
| `use_cuda_graph` | `bool` | Whether CUDA graphs are enabled |
| `args_iterator` | `ArgsIterator` | Iterator over argument combinations |

### ArgsIterator

| Property | Type | Description |
|----------|------|-------------|
| `n_args` | `int` | Number of argument combinations |

**Methods:**

| Method | Description |
|--------|-------------|
| `__next__()` | Generator yielding (args, kwargs) tuples |
| `reset()` | Reset iterator to beginning |

## Usage Examples

### Basic Benchmark

```python
from benchmarks.kernels.utils import Bench
import torch

def my_kernel(x, y):
    return x @ y

# Simple benchmark without argument pooling
with Bench(
    cuda_graph_params=None,
    label="GEMM",
    sub_label="m=128, n=256, k=512",
    description="Simple matrix multiplication",
    fn=my_kernel,
    x=torch.randn(128, 512, device="cuda"),
    y=torch.randn(512, 256, device="cuda")
) as bench:
    result = bench.run()
    print(f"Median time: {result.median * 1000:.3f} ms")
```

### Using ArgPool to Prevent Cache Effects

```python
from benchmarks.kernels.utils import Bench, ArgPool
import torch

# Create pool of different input matrices
x_pool = ArgPool([
    torch.randn(128, 512, device="cuda") for _ in range(4)
])
y_pool = ArgPool([
    torch.randn(512, 256, device="cuda") for _ in range(4)
])

def my_kernel(x, y):
    return x @ y

# Benchmark will rotate through different matrices
with Bench(
    cuda_graph_params=None,
    label="GEMM with ArgPool",
    sub_label="m=128, n=256, k=512",
    description="Matrix multiplication with cache busting",
    fn=my_kernel,
    x=x_pool,
    y=y_pool
) as bench:
    result = bench.run()
    print(f"Median time: {result.median * 1000:.3f} ms")
```

### CUDA Graph Benchmarking

```python
from benchmarks.kernels.utils import Bench, CudaGraphBenchParams
import torch

def fused_kernel(x, w, bias):
    return torch.nn.functional.linear(x, w, bias)

# Benchmark within CUDA graph
cuda_params = CudaGraphBenchParams(num_ops_in_cuda_graph=10)

with Bench(
    cuda_graph_params=cuda_params,
    label="Fused Linear",
    sub_label="CUDA Graph (10 ops)",
    description="Linear layer in CUDA graph",
    fn=fused_kernel,
    x=torch.randn(32, 512, device="cuda"),
    w=torch.randn(256, 512, device="cuda"),
    bias=torch.randn(256, device="cuda")
) as bench:
    result = bench.run()
    print(f"Time per op: {result.median / 10 * 1000:.3f} ms")
```

### Comparing Eager vs CUDA Graph

```python
from benchmarks.kernels.utils import Bench, CudaGraphBenchParams
import torch

def attention_kernel(q, k, v):
    scores = q @ k.transpose(-2, -1)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v

# Setup
q = torch.randn(32, 8, 64, 64, device="cuda")
k = torch.randn(32, 8, 64, 64, device="cuda")
v = torch.randn(32, 8, 64, 64, device="cuda")

# Eager mode
with Bench(
    cuda_graph_params=None,
    label="Attention",
    sub_label="Eager",
    description="Attention without CUDA graph",
    fn=attention_kernel,
    q=q, k=k, v=v
) as bench:
    eager_result = bench.run()

# CUDA graph mode
with Bench(
    cuda_graph_params=CudaGraphBenchParams(num_ops_in_cuda_graph=5),
    label="Attention",
    sub_label="CUDA Graph",
    description="Attention with CUDA graph",
    fn=attention_kernel,
    q=q, k=k, v=v
) as bench:
    graph_result = bench.run()

print(f"Eager: {eager_result.median * 1000:.3f} ms")
print(f"CUDA Graph: {graph_result.median * 1000:.3f} ms")
print(f"Speedup: {eager_result.median / graph_result.median:.2f}x")
```

### Multiple ArgPool Parameters

```python
from benchmarks.kernels.utils import Bench, ArgPool
import torch

# Create pools for multiple parameters
batch_sizes = [32, 64, 128, 256]
x_pool = ArgPool([
    torch.randn(bs, 512, device="cuda") for bs in batch_sizes
])
weights_pool = ArgPool([
    torch.randn(256, 512, device="cuda") for _ in batch_sizes
])

def linear(x, w):
    return x @ w.t()

# Will cycle through (bs=32, w0), (bs=64, w1), (bs=128, w2), (bs=256, w3)
with Bench(
    cuda_graph_params=None,
    label="Linear with varying batch",
    sub_label="Batch sweep",
    description="Linear layer with multiple batch sizes",
    fn=linear,
    x=x_pool,
    w=weights_pool
) as bench:
    result = bench.run()
    print(f"Average time: {result.median * 1000:.3f} ms")
```

### Custom Timing Configuration

```python
from benchmarks.kernels.utils import Bench
import torch

def expensive_kernel(x):
    for _ in range(100):
        x = torch.sin(x) + torch.cos(x)
    return x

with Bench(
    cuda_graph_params=None,
    label="Expensive kernel",
    sub_label="100 iterations",
    description="Trigonometric operations",
    fn=expensive_kernel,
    x=torch.randn(1024, 1024, device="cuda")
) as bench:
    # Customize minimum run time
    bench.min_run_time = 5.0  # 5 seconds minimum

    result = bench.run()
    print(f"Median time: {result.median * 1000:.3f} ms")
    print(f"IQR: {result.iqr * 1000:.3f} ms")
    print(f"Number of runs: {result.number_per_run}")
```

### Analyzing Measurement Results

```python
from benchmarks.kernels.utils import Bench
import torch

def my_kernel(x):
    return torch.matmul(x, x.t())

with Bench(
    cuda_graph_params=None,
    label="Self-matmul",
    sub_label="1024x1024",
    description="X @ X.T",
    fn=my_kernel,
    x=torch.randn(1024, 1024, device="cuda")
) as bench:
    result = bench.run()

    # Access detailed statistics
    print(f"Label: {result.task_spec.sub_label}")
    print(f"Median: {result.median * 1000:.3f} ms")
    print(f"Mean: {result.mean * 1000:.3f} ms")
    print(f"IQR: {result.iqr * 1000:.3f} ms")
    print(f"Min: {result.times[0] * 1000:.3f} ms")
    print(f"Max: {result.times[-1] * 1000:.3f} ms")
    print(f"Runs: {len(result.times)}")
    print(f"Meets confidence: {result.meets_confidence()}")
    print(f"Has warnings: {result.has_warnings}")
```

### Building Benchmark Suites

```python
from benchmarks.kernels.utils import Bench, ArgPool
import torch
import pandas as pd

def create_benchmark_suite():
    """Create suite of benchmarks for different configurations"""
    results = []

    # Test different matrix sizes
    sizes = [256, 512, 1024, 2048]

    for size in sizes:
        # Create fresh tensors for each size
        x = torch.randn(size, size, device="cuda")
        y = torch.randn(size, size, device="cuda")

        with Bench(
            cuda_graph_params=None,
            label="GEMM",
            sub_label=f"{size}x{size}",
            description=f"Matrix multiplication {size}x{size}",
            fn=lambda a, b: a @ b,
            a=x, b=y
        ) as bench:
            result = bench.run()

            results.append({
                "size": size,
                "time_ms": result.median * 1000,
                "tflops": 2 * size**3 / result.median / 1e12
            })

    # Create DataFrame
    df = pd.DataFrame(results)
    print(df)

    return df

results_df = create_benchmark_suite()
```

## Related Pages

- [[vllm-project_vllm_cuda_graphs]] - CUDA graph support in vLLM
- [[vllm-project_vllm_benchmark_silu_mul_fp8_quant]] - SiLU benchmark using these utilities
- [[vllm-project_vllm_benchmark_suite]] - Full benchmark suite
- [[vllm-project_vllm_performance_testing]] - Performance testing framework
- [[vllm-project_vllm_torch_benchmark]] - torch.utils.benchmark integration
