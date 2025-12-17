---
knowledge_sources:
  - vllm-project/vllm repository
  - File: benchmarks/kernels/benchmark_w8a8_block_fp8.py
domains:
  - Kernel Tuning
  - FP8 Quantization
  - Matrix Multiplication
  - Performance Optimization
last_updated: 2025-12-17
---

# W8A8 Block FP8 Kernel Tuner

## Overview

Multi-GPU automated tuning framework for finding optimal Triton kernel configurations for W8A8 block-quantized FP8 matrix multiplication across different models and batch sizes.

## Description

The `benchmark_w8a8_block_fp8.py` script provides comprehensive automated tuning for W8A8 (8-bit weight, 8-bit activation) block-quantized FP8 GEMM operations. Block quantization uses per-block scaling factors instead of per-tensor scales, providing better accuracy while maintaining FP8 performance.

Key features:
- **Multi-GPU parallelization**: Distributes batch size tuning across available GPUs
- **Weight shape extraction**: Automatically derives weight shapes from model configs with tensor parallelism
- **Comprehensive search space**: Tests combinations of BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages
- **Model support**: Pre-configured for DeepSeekV3, Llama, Mistral, Qwen, and other popular models
- **JSON output**: Saves optimal configs indexed by (N, K, batch_size) for runtime use
- **TP-aware**: Accounts for tensor parallelism when computing weight shapes

The tuning process:
1. Generates search space of kernel configurations
2. Distributes batch sizes across GPUs for parallel tuning
3. Benchmarks each configuration for each weight shape
4. Selects best config per (N, K, batch_size) tuple
5. Saves results as JSON files for production use

## Usage

### Command Line Execution

```bash
# Tune for DeepSeekV3 with TP=8
python benchmarks/kernels/benchmark_w8a8_block_fp8.py --tp-size 8 --input-type fp8

# Tune specific batch size on single GPU
python benchmarks/kernels/benchmark_w8a8_block_fp8.py --tp-size 8 --batch-size 64

# Custom block sizes
python benchmarks/kernels/benchmark_w8a8_block_fp8.py --tp-size 4 --block-n 256 --block-k 256

# Custom save path
python benchmarks/kernels/benchmark_w8a8_block_fp8.py --tp-size 8 --save-path ./tuned_configs/
```

### Configuration Options

```bash
--tp-size         Tensor parallelism size (default: 8)
--input-type      Input quantization type: fp8 (default: fp8)
--out-dtype       Output dtype: float16, bfloat16, float32 (default: float16)
--block-n         Block size for N dimension (default: 128)
--block-k         Block size for K dimension (default: 128)
--batch-size      Single batch size to tune (optional, for testing)
--save-path       Directory to save tuned configs (default: ./)
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_w8a8_block_fp8.py`

**Main Functions:**

```python
def w8a8_block_matmul(
    A: torch.Tensor,           # (M, K) FP8 activations
    B: torch.Tensor,           # (N, K) FP8 weights
    As: torch.Tensor,          # (M, K_tiles) FP32 activation scales
    Bs: torch.Tensor,          # (N_tiles, K_tiles) FP32 weight scales
    block_size: list[int],     # [block_n, block_k]
    config: dict[str, Any],    # Triton kernel config
    output_dtype: torch.dtype = torch.float16
) -> torch.Tensor
```

```python
def tune(
    M: int,                     # Batch size / rows
    N: int,                     # Output columns
    K: int,                     # Inner dimension
    block_size: list[int],      # [block_n, block_k]
    out_dtype: torch.dtype,
    search_space: list[dict],
    input_type: str
) -> dict[str, Any]
```

```python
def get_weight_shapes(tp_size: int) -> list[tuple[int, int]]:
    """Extract weight shapes for DeepSeekV3 with tensor parallelism"""
```

**Kernel Config Search Space:**

```python
def get_configs_compute_bound():
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [64, 128]:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 16, 32, 64]:
                            configs.append({
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": group_size,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            })
    return configs
```

**Import:**
```python
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `torch.Tensor` | FP8 activation tensor, shape `(M, K)` |
| `B` | `torch.Tensor` | FP8 weight tensor, shape `(N, K)` |
| `As` | `torch.Tensor` | Activation scales, shape `(M, K_tiles)` |
| `Bs` | `torch.Tensor` | Weight scales, shape `(N_tiles, K_tiles)` |
| `block_size` | `list[int]` | `[block_n, block_k]` quantization block sizes |
| `config` | `dict` | Triton kernel configuration |
| `output_dtype` | `torch.dtype` | Output tensor dtype (default: float16) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `C` | `torch.Tensor` | Result tensor, shape `(M, N)` in `output_dtype` |
| `best_config` | `dict` | Optimal kernel configuration |
| `best_time` | `float` | Best execution time in microseconds |

### Tuned Config JSON Format

Saved as `N={N},K={K},device_name={device},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json`:

```json
{
  "1": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3
  },
  "64": {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 32,
    "num_warps": 8,
    "num_stages": 4
  }
}
```

## Usage Examples

### Basic Tuning Run

```python
# Run tuning for TP=8 configuration
import subprocess

result = subprocess.run([
    "python", "benchmarks/kernels/benchmark_w8a8_block_fp8.py",
    "--tp-size", "8",
    "--input-type", "fp8",
    "--save-path", "./configs/"
], capture_output=True, text=True)

print(result.stdout)
```

### Custom Weight Shapes

```python
from benchmark_w8a8_block_fp8 import tune, get_configs_compute_bound

# Define custom weight shapes
weight_shapes = [
    (4096, 4096),   # Square
    (11008, 4096),  # FFN up
    (4096, 11008),  # FFN down
]

batch_sizes = [1, 8, 32, 128]
block_size = [128, 128]
search_space = get_configs_compute_bound()

# Tune each shape
for N, K in weight_shapes:
    print(f"\nTuning shape N={N}, K={K}")
    for M in batch_sizes:
        best_config = tune(
            M, N, K,
            block_size,
            torch.float16,
            search_space,
            "fp8"
        )
        print(f"  M={M}: {best_config}")
```

### Loading and Using Tuned Configs

```python
import json
import torch
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)

# Load tuned config
device_name = torch.cuda.get_device_name().replace(" ", "_")
config_file = f"N=4096,K=4096,device_name={device_name},dtype=fp8_w8a8,block_shape=[128,128].json"

with open(config_file) as f:
    configs = json.load(f)

# Get config for batch size 64
batch_size = 64
config = configs[str(batch_size)]

# Use for inference
A = torch.randn(64, 4096, dtype=torch.float8_e4m3fn, device="cuda")
B = torch.randn(4096, 4096, dtype=torch.float8_e4m3fn, device="cuda")
As = torch.randn(64, 32, dtype=torch.float32, device="cuda")  # 4096/128 = 32
Bs = torch.randn(32, 32, dtype=torch.float32, device="cuda")

C = _w8a8_triton_block_scaled_mm[lambda meta: (
    triton.cdiv(64, meta["BLOCK_SIZE_M"]) * triton.cdiv(4096, meta["BLOCK_SIZE_N"]),
)](
    A, B, C_output, As, Bs,
    M=64, N=4096, K=4096,
    block_n=128, block_k=128,
    stride_am=A.stride(0), stride_ak=A.stride(1),
    stride_bk=B.stride(1), stride_bn=B.stride(0),
    stride_cm=C.stride(0), stride_cn=C.stride(1),
    stride_as_m=As.stride(0), stride_as_k=As.stride(1),
    stride_bs_n=Bs.stride(0), stride_bs_k=Bs.stride(1),
    **config
)
```

### Analyzing Tuning Results

```python
import json
import glob
import pandas as pd

# Load all tuned configs
config_files = glob.glob("*.json")
results = []

for config_file in config_files:
    # Parse filename: N={N},K={K},device_name={device},dtype=fp8_w8a8,block_shape=[{bn},{bk}].json
    parts = config_file.replace(".json", "").split(",")
    N = int(parts[0].split("=")[1])
    K = int(parts[1].split("=")[1])

    with open(config_file) as f:
        configs = json.load(f)

    for batch_size, config in configs.items():
        results.append({
            "N": N,
            "K": K,
            "batch_size": int(batch_size),
            "BLOCK_M": config["BLOCK_SIZE_M"],
            "BLOCK_N": config["BLOCK_SIZE_N"],
            "BLOCK_K": config["BLOCK_SIZE_K"],
            "GROUP_M": config["GROUP_SIZE_M"],
            "num_warps": config["num_warps"],
            "num_stages": config["num_stages"],
        })

df = pd.DataFrame(results)
print(df.describe())

# Most common configs
print("\nMost common BLOCK_M:")
print(df["BLOCK_M"].value_counts())
```

### Benchmarking Tuned vs Default

```python
import torch
import time

def benchmark_config(A, B, As, Bs, block_size, config, iterations=100):
    """Benchmark a specific config"""
    # Warmup
    for _ in range(10):
        w8a8_block_matmul(A, B, As, Bs, block_size, config)
    torch.cuda.synchronize()

    # Timing
    start = time.time()
    for _ in range(iterations):
        w8a8_block_matmul(A, B, As, Bs, block_size, config)
    torch.cuda.synchronize()

    return (time.time() - start) / iterations * 1e6  # microseconds

# Test tuned vs default config
M, N, K = 64, 4096, 4096
block_size = [128, 128]

# Setup tensors
A = torch.randn(M, K, dtype=torch.float8_e4m3fn, device="cuda")
B = torch.randn(N, K, dtype=torch.float8_e4m3fn, device="cuda")
As = torch.randn(M, K//128, dtype=torch.float32, device="cuda")
Bs = torch.randn(N//128, K//128, dtype=torch.float32, device="cuda")

# Default config
default_config = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}

# Load tuned config
with open("config.json") as f:
    tuned_config = json.load(f)[str(M)]

# Benchmark
default_time = benchmark_config(A, B, As, Bs, block_size, default_config)
tuned_time = benchmark_config(A, B, As, Bs, block_size, tuned_config)

print(f"Default config: {default_time:.2f} us")
print(f"Tuned config: {tuned_time:.2f} us")
print(f"Speedup: {default_time / tuned_time:.2f}x")
```

## Related Pages

- [[vllm-project_vllm_w8a8_triton_block_scaled_mm]] - Core Triton kernel implementation
- [[vllm-project_vllm_fp8_block_quantization]] - Block FP8 quantization
- [[vllm-project_vllm_fp8_utils]] - FP8 utility functions
- [[vllm-project_vllm_deepgemm_fp8_benchmark]] - DeepGEMM FP8 comparison
- [[vllm-project_vllm_triton_autotuning]] - Triton autotuning infrastructure
- [[vllm-project_vllm_tensor_parallelism]] - Tensor parallelism implementation
