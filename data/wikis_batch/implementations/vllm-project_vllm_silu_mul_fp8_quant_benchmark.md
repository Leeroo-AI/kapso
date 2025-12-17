---
knowledge_sources:
  - vllm-project/vllm repository
  - File: benchmarks/kernels/benchmark_silu_mul_fp8_quant.py
domains:
  - Performance Benchmarking
  - FP8 Quantization
  - MoE Operations
  - Kernel Optimization
last_updated: 2025-12-17
---

# SiLU+Mul+FP8 Quantization Benchmark

## Overview

A comprehensive benchmarking suite comparing CUDA and Triton implementations of the fused SiLU activation, element-wise multiply, and FP8 quantization operation for Mixture of Experts (MoE) inference.

## Description

The `benchmark_silu_mul_fp8_quant.py` script provides an extensive performance analysis framework for evaluating different implementations of the critical SiLU+mul+quant operation used in MoE models. This fused operation combines:

1. SiLU activation: `silu(x) = x * sigmoid(x)`
2. Element-wise multiplication with gate values
3. FP8 quantization with per-token-group scales

The benchmark suite tests implementations across:
- Multiple expert configurations (E=8, 32, 256)
- Various token distributions (uniform, imbalanced, even assignment)
- Different hidden dimensions (H=7168 for DeepSeekV3)
- Token count variations (256-131072 tokens)

Key implementations compared:
- **SiLU V2 (CUDA)**: Optimized CUDA kernel `persistent_masked_m_silu_mul_quant`
- **Triton Kernel**: Custom Triton implementation `_silu_mul_fp8_quant_deep_gemm`

The benchmark generates comprehensive visualizations including memory bandwidth utilization, speedup ratios, and performance across different workload patterns.

## Usage

### Command Line Execution

```bash
# Run the full benchmark suite
python benchmarks/kernels/benchmark_silu_mul_fp8_quant.py
```

### Configuration

Key configuration variables in the script:

```python
# Expert configurations to test
configs = [
    (8, 1024, 7168),    # (E, T, H)
    (32, 1024, 7168),
    (256, 1024, 7168),
]

# Number of benchmark runs
runs = 100
num_warmups = 20

# Token distribution strategies
strategies = ["random_imbalanced", "uniform", "max_t"]
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_silu_mul_fp8_quant.py`

**Key Functions:**

```python
def silu_mul_fp8_quant_deep_gemm_triton(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,)
    num_parallel_tokens: int,
    group_size: int = 128,
    eps: float = 1e-10,
    expert_offsets: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]
```

```python
def benchmark(
    kernel: Callable,
    E: int,  # Number of experts
    T: int,  # Max tokens per expert
    H: int,  # Hidden dimension
    total_tokens: int,
    num_parallel_tokens: int = 64,
    G: int = 128,  # Group size
    runs: int = 200,
    num_warmups: int = 20,
    gen_strategy: str = "default",
    iterations_per_run: int = 20
)
```

**Triton Kernel:**

```python
@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    input_ptr, y_q_ptr, y_s_ptr, counts_ptr,
    H: tl.constexpr, GROUP_SIZE: tl.constexpr,
    stride_i_e, stride_i_t, stride_i_h,
    stride_yq_e, stride_yq_t, stride_yq_h,
    stride_ys_e, stride_ys_t, stride_ys_g,
    stride_counts_e,
    eps: tl.constexpr, fp8_min: tl.constexpr, fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr
)
```

**Import:**
```python
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    persistent_masked_m_silu_mul_quant,
)
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `torch.Tensor` | Input activations of shape `(E, T, 2*H)` in bfloat16 |
| `tokens_per_expert` | `torch.Tensor` | Number of valid tokens per expert, shape `(E,)` |
| `num_parallel_tokens` | `int` | Parallelization parameter (default: 64) |
| `group_size` | `int` | Quantization group size (default: 128) |
| `eps` | `float` | Epsilon for numerical stability (default: 1e-10) |
| `E` | `int` | Number of experts |
| `T` | `int` | Maximum tokens per expert |
| `H` | `int` | Hidden dimension per output |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `y_q` | `torch.Tensor` | Quantized output in FP8 (float8_e4m3fn), shape `(E, T, H)` |
| `y_s` | `torch.Tensor` | Per-group quantization scales in FP32, shape `(E, T, G)` where G=H/group_size |
| `median_time_ms` | `float` | Median execution time in milliseconds |
| `gflops` | `float` | Performance in GFLOPS |
| `memory_bw` | `float` | Memory bandwidth in GB/s |
| `perc` | `float` | Percentage of peak Hopper bandwidth (3.35 TB/s) |

### Benchmark Metrics

| Metric | Description |
|--------|-------------|
| Time (ms) | Median execution time across runs |
| GFLOPS | Computational throughput (8 ops per element) |
| GB/s | Memory bandwidth utilization |
| % Utilization | Percentage of peak memory bandwidth |
| Speedup | Relative performance (baseline/implementation) |

## Usage Examples

### Running Full Benchmark Suite

```python
import torch
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    persistent_masked_m_silu_mul_quant,
)

# Configuration
E, T, H = 32, 1024, 7168
total_tokens = 8 * E
runs = 100
num_warmups = 20

# Run benchmark for CUDA implementation
time_ms, gflops, gbps, perc = benchmark(
    persistent_masked_m_silu_mul_quant,
    E, T, H, total_tokens,
    runs=runs,
    num_warmups=num_warmups,
    gen_strategy="uniform"
)

print(f"CUDA SiLU V2: {time_ms:.3f}ms, {gflops:.2f} GFLOPS, {perc:.2f}% BW")
```

### Custom Token Distribution

```python
# Test with imbalanced token distribution
def generate_expert_loads(n_e, total_tokens, ratio=0.7, device="cuda"):
    mean = total_tokens // n_e
    min_max = mean // ratio
    e = torch.ones(size=(E,), dtype=torch.int64, device=device) * mean
    e[0] = min_max
    r = torch.rand(size=(E - 1,))
    r /= r.sum()
    r *= total_tokens - min_max
    r = r.round().long()
    e[1:] = r.to(device=device)
    return e

tokens_per_expert = generate_expert_loads(32, 16384)

# Run with custom distribution
y = torch.rand((E, T, 2 * H), dtype=torch.bfloat16, device="cuda")
y_q, y_s = silu_mul_fp8_quant_deep_gemm_triton(
    y, tokens_per_expert, num_parallel_tokens=64, group_size=128
)
```

### Analyzing Results

```python
# The benchmark generates plots automatically
# Results stored in all_results list with structure:
# (strategy_name, all_ratios, all_silu_v2_results,
#  all_triton_results, config_labels, config_x_axis)

# Access performance data
for strategy_name, ratios, v2_results, triton_results, labels, _ in all_results:
    print(f"\nStrategy: {strategy_name}")
    for i, label in enumerate(labels):
        v2_time, v2_gflops, v2_gbps, v2_perc = v2_results[i]
        triton_time, triton_gflops, triton_gbps, triton_perc = triton_results[i]
        speedup = triton_time / v2_time
        print(f"{label}: CUDA is {speedup:.2f}x faster")
```

### Testing with Different Group Sizes

```python
# Compare different quantization group sizes
group_sizes = [64, 128, 256]
for group_size in group_sizes:
    time_ms, _, _, perc = benchmark(
        persistent_masked_m_silu_mul_quant,
        E=32, T=1024, H=7168,
        total_tokens=16384,
        G=group_size,
        runs=50
    )
    print(f"Group size {group_size}: {time_ms:.3f}ms ({perc:.1f}% BW)")
```

## Related Pages

- [[vllm-project_vllm_persistent_masked_silu_mul_quant]] - CUDA implementation
- [[vllm-project_vllm_batched_deep_gemm_moe]] - DeepGEMM MoE layer
- [[vllm-project_vllm_fp8_quantization]] - FP8 quantization utilities
- [[vllm-project_vllm_w8a8_block_fp8_tuner]] - W8A8 block FP8 tuning
- [[vllm-project_vllm_benchmark_utils]] - Benchmark utility classes
- [[vllm-project_vllm_triton_kernels]] - Triton kernel implementations
