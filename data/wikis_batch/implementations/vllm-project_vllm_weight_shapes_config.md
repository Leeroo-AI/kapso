---
knowledge_sources:
  - vllm-project/vllm repository
  - File: benchmarks/kernels/weight_shapes.py
domains:
  - Model Configuration
  - Tensor Parallelism
  - Performance Benchmarking
  - Weight Management
last_updated: 2025-12-17
---

# Weight Shapes Configuration with Tensor Parallelism

## Overview

Centralized configuration defining weight shapes for popular LLM models with tensor parallelism (TP) split dimension information for accurate distributed benchmarking.

## Description

The `weight_shapes.py` module provides a comprehensive dictionary of weight matrix shapes for various transformer models, enhanced with tensor parallelism metadata. Each weight shape entry specifies both the full (TP=1) dimensions and which dimension should be sharded when using tensor parallelism.

Format: `([K, N], TP_SPLIT_DIM)` where:
- `[K, N]`: Weight matrix dimensions at TP=1
- `TP_SPLIT_DIM`: Dimension to shard (0=K, 1=N)

This enables benchmarks to automatically compute correct shapes for different TP configurations, ensuring realistic performance testing under distributed inference conditions.

Supported models:
- Mistral family (7B, Large-Instruct-2407)
- Llama 2 (7B, 13B, 70B)
- Llama 3 (8B, 405B)
- Llama 3.1 (8B, 70B-Instruct)
- Llama 3.3 (70B-Instruct)
- Qwen 2.5 (7B, 32B, 72B-Instruct)
- DeepSeek-Coder-V2-Lite-Instruct
- Cohere Command-A-03

## Usage

### Importing Weight Shapes

```python
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/weight_shapes.py`

**Data Structure:**

```python
WEIGHT_SHAPES = {
    "model_name": [
        ([K, N], TP_SPLIT_DIM),
        # More weight shapes...
    ],
}
```

**Example Entry:**

```python
"meta-llama/Llama-3-8b": [
    ([4096, 6144], 1),      # Attention QKV projection (N-split)
    ([4096, 4096], 0),      # Attention output projection (K-split)
    ([4096, 28672], 1),     # FFN up projection (N-split)
    ([14336, 4096], 0),     # FFN down projection (K-split)
]
```

## I/O Contract

### Dictionary Structure

| Key | Type | Description |
|-----|------|-------------|
| Model name | `str` | Hugging Face model identifier |
| Weight shapes | `list[tuple]` | List of `([K, N], TP_SPLIT_DIM)` tuples |

### Weight Shape Tuple

| Element | Type | Description |
|---------|------|-------------|
| `[K, N]` | `list[int]` | Weight dimensions at TP=1 |
| `TP_SPLIT_DIM` | `int` | Dimension to shard: 0 for K, 1 for N |

### Tensor Parallelism Logic

**TP_SPLIT_DIM = 0 (K-split):**
```python
# Original: (K, N)
# TP=2: (K//2, N)
# TP=4: (K//4, N)
```

**TP_SPLIT_DIM = 1 (N-split):**
```python
# Original: (K, N)
# TP=2: (K, N//2)
# TP=4: (K, N//4)
```

### Model Categories

| Category | Description | Example Shapes |
|----------|-------------|----------------|
| Attention QKV | Query/Key/Value projections | N-split: ([H, 3*H], 1) |
| Attention Output | Output projection | K-split: ([H, H], 0) |
| FFN Up | Feed-forward up projection | N-split: ([H, FFN], 1) |
| FFN Down | Feed-forward down projection | K-split: ([FFN//2, H], 0) |
| MoE Router | Expert routing layer | Usually N-split |
| MoE Expert | Individual expert weights | K or N-split |

## Usage Examples

### Accessing Weight Shapes

```python
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES

# Get shapes for Llama-3-8b
model_name = "meta-llama/Llama-3-8b"
shapes = WEIGHT_SHAPES[model_name]

print(f"Weight shapes for {model_name}:")
for (K, N), split_dim in shapes:
    split_str = "K-split" if split_dim == 0 else "N-split"
    print(f"  ({K:5d}, {N:5d}) - {split_str}")
```

### Computing TP-Sharded Shapes

```python
def get_tp_shape(shape_tuple, tp_size):
    """Compute weight shape for given TP size"""
    (K, N), split_dim = shape_tuple

    if split_dim == 0:
        # K dimension is split
        return (K // tp_size, N)
    else:
        # N dimension is split
        return (K, N // tp_size)

# Example: Llama-3-8b attention QKV with TP=4
model_name = "meta-llama/Llama-3-8b"
qkv_shape = WEIGHT_SHAPES[model_name][0]  # ([4096, 6144], 1)

tp1_shape = get_tp_shape(qkv_shape, 1)   # (4096, 6144)
tp2_shape = get_tp_shape(qkv_shape, 2)   # (4096, 3072)
tp4_shape = get_tp_shape(qkv_shape, 4)   # (4096, 1536)
tp8_shape = get_tp_shape(qkv_shape, 8)   # (4096, 768)

print(f"TP=1: {tp1_shape}")
print(f"TP=2: {tp2_shape}")
print(f"TP=4: {tp4_shape}")
print(f"TP=8: {tp8_shape}")
```

### Benchmarking with TP Shapes

```python
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES
import torch

def benchmark_model_weights(model_name, tp_size, batch_size):
    """Benchmark all weight shapes for a model at given TP size"""
    shapes = WEIGHT_SHAPES[model_name]

    results = []
    for (K, N), split_dim in shapes:
        # Compute TP-sharded shape
        if split_dim == 0:
            k_tp, n_tp = K // tp_size, N
        else:
            k_tp, n_tp = K, N // tp_size

        # Create tensors
        A = torch.randn(batch_size, k_tp, device="cuda", dtype=torch.float16)
        B = torch.randn(n_tp, k_tp, device="cuda", dtype=torch.float16)

        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            C = A @ B.t()
        end.record()
        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end) / 100
        results.append({
            "shape": f"({k_tp}, {n_tp})",
            "time_ms": time_ms,
            "split": "K" if split_dim == 0 else "N"
        })

    return results

# Benchmark Llama-3-8b with TP=4
results = benchmark_model_weights("meta-llama/Llama-3-8b", tp_size=4, batch_size=32)
for r in results:
    print(f"{r['shape']:20s} {r['split']}-split: {r['time_ms']:.3f} ms")
```

### Generating Benchmark Suite

```python
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES

def generate_benchmark_configs(model_name, tp_sizes, batch_sizes):
    """Generate all benchmark configurations for a model"""
    shapes = WEIGHT_SHAPES[model_name]
    configs = []

    for (K, N), split_dim in shapes:
        for tp_size in tp_sizes:
            for batch_size in batch_sizes:
                # Compute TP-sharded dimensions
                if split_dim == 0:
                    k_tp, n_tp = K // tp_size, N
                else:
                    k_tp, n_tp = K, N // tp_size

                configs.append({
                    "model": model_name,
                    "tp_size": tp_size,
                    "batch_size": batch_size,
                    "m": batch_size,
                    "k": k_tp,
                    "n": n_tp,
                    "split_dim": "K" if split_dim == 0 else "N",
                    "original_shape": (K, N)
                })

    return configs

# Generate configs
tp_sizes = [1, 2, 4, 8]
batch_sizes = [1, 8, 32, 128]
configs = generate_benchmark_configs(
    "meta-llama/Llama-3-8b",
    tp_sizes,
    batch_sizes
)

print(f"Generated {len(configs)} benchmark configurations")
print("\nExample configs:")
for config in configs[:5]:
    print(f"  TP={config['tp_size']}, BS={config['batch_size']}, "
          f"M={config['m']}, K={config['k']}, N={config['n']}, "
          f"{config['split_dim']}-split")
```

### Comparing Models

```python
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES

def compare_model_sizes():
    """Compare total parameter counts across models"""
    results = []

    for model_name, shapes in WEIGHT_SHAPES.items():
        total_params = 0
        for (K, N), _ in shapes:
            total_params += K * N

        results.append({
            "model": model_name,
            "num_shapes": len(shapes),
            "total_params_M": total_params / 1e6,
        })

    # Sort by parameter count
    results.sort(key=lambda x: x["total_params_M"], reverse=True)

    print("Model Parameter Counts (benchmark weights only):")
    for r in results:
        print(f"  {r['model']:50s}: {r['total_params_M']:8.1f}M "
              f"({r['num_shapes']} weight matrices)")

compare_model_sizes()
```

### Validating TP Divisibility

```python
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES

def validate_tp_divisibility(model_name, tp_sizes):
    """Check if all shapes are divisible by given TP sizes"""
    shapes = WEIGHT_SHAPES[model_name]

    print(f"Validating {model_name} for TP sizes: {tp_sizes}")

    for (K, N), split_dim in shapes:
        split_name = "K" if split_dim == 0 else "N"
        split_value = K if split_dim == 0 else N

        valid = []
        invalid = []

        for tp_size in tp_sizes:
            if split_value % tp_size == 0:
                valid.append(tp_size)
            else:
                invalid.append(tp_size)

        if invalid:
            print(f"  Shape ({K}, {N}) {split_name}-split:")
            print(f"    Valid TP: {valid}")
            print(f"    Invalid TP: {invalid} (not divisible by {split_value})")

# Validate for common TP sizes
validate_tp_divisibility("meta-llama/Llama-3-8b", [1, 2, 4, 8, 16])
```

### Memory Estimation

```python
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES

def estimate_memory_usage(model_name, tp_size, dtype_bytes=2):
    """Estimate memory usage for weights at given TP size"""
    shapes = WEIGHT_SHAPES[model_name]

    total_params = 0
    for (K, N), split_dim in shapes:
        # Compute TP-sharded dimensions
        if split_dim == 0:
            k_tp, n_tp = K // tp_size, N
        else:
            k_tp, n_tp = K, N // tp_size

        total_params += k_tp * n_tp

    # Memory in GB (assuming fp16/bf16)
    memory_gb = total_params * dtype_bytes / 1024**3

    return {
        "model": model_name,
        "tp_size": tp_size,
        "params": total_params,
        "memory_gb": memory_gb,
    }

# Estimate for different TP sizes
for tp in [1, 2, 4, 8]:
    result = estimate_memory_usage("meta-llama/Llama-3-8b", tp)
    print(f"TP={tp}: {result['params']/1e6:.1f}M params, "
          f"{result['memory_gb']:.2f} GB")
```

### Finding Optimal TP Size

```python
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES

def recommend_tp_size(model_name, gpu_memory_gb, dtype_bytes=2):
    """Recommend TP size based on available GPU memory"""
    shapes = WEIGHT_SHAPES[model_name]

    # Calculate memory for TP=1
    total_params = sum(K * N for (K, N), _ in shapes)
    memory_gb_tp1 = total_params * dtype_bytes / 1024**3

    # Find minimum TP size that fits
    tp_candidates = [1, 2, 4, 8, 16]
    for tp in tp_candidates:
        memory_per_gpu = memory_gb_tp1 / tp
        if memory_per_gpu <= gpu_memory_gb * 0.7:  # 70% utilization
            return {
                "recommended_tp": tp,
                "memory_per_gpu": memory_per_gpu,
                "utilization": memory_per_gpu / gpu_memory_gb,
            }

    return None

# Example: fit on A100 80GB
result = recommend_tp_size("meta-llama/Llama-3-8b", gpu_memory_gb=80)
if result:
    print(f"Recommended TP: {result['recommended_tp']}")
    print(f"Memory per GPU: {result['memory_per_gpu']:.2f} GB")
    print(f"Utilization: {result['utilization']:.1%}")
```

## Related Pages

- [[vllm-project_vllm_tensor_parallelism]] - Tensor parallelism implementation
- [[vllm-project_vllm_model_config]] - Model configuration system
- [[vllm-project_vllm_distributed_inference]] - Distributed inference setup
- [[vllm-project_vllm_weight_loading]] - Weight loading and initialization
- [[vllm-project_vllm_benchmark_shapes]] - Legacy benchmark shapes (simpler)
- [[vllm-project_vllm_tp_sharding]] - TP sharding strategies
