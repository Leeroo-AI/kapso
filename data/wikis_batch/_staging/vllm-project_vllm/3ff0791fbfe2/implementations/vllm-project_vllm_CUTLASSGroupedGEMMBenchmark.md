# CUTLASS Grouped GEMM Benchmark

**Knowledge Sources:** benchmarks/kernels/benchmark_grouped_gemm_cutlass.py
**Domains:** Performance, Quantization, Benchmarking, Mixture-of-Experts
**Last Updated:** 2025-12-17

## Overview

Benchmarking tool comparing CUTLASS grouped GEMM implementation against Triton MOE kernels for FP8 quantized mixture-of-experts workloads.

## Description

The CUTLASS Grouped GEMM benchmark evaluates the performance of CUTLASS-based grouped matrix multiplication kernels specifically designed for FP8 quantized MOE layers. This benchmark helps determine when CUTLASS grouped GEMM provides advantages over Triton's fused experts implementation.

The benchmark compares:
1. **CUTLASS Grouped GEMM**: FP8 grouped matrix multiplication with stride configuration
2. **Triton MOE**: Triton-based fused experts with FP8 quantization

Both implementations are tested with and without CUDA graphs to measure both raw kernel performance and graph-optimized execution.

Key features:
- Tests multiple model configurations: Mixtral-8x7B, DeepSeek-V2-Lite, Granite-3.0
- Supports batch sizes from 1 to 512
- Uses per-tensor FP8 quantization for weights and activations
- Benchmarks both direct kernel calls and CUDA graph execution
- Captures 10 operations per graph for amortized timing
- Uses torch.utils.benchmark for consistent measurements
- Reports comparative timing across implementations

The benchmark is essential for understanding when grouped GEMM provides performance advantages for MOE workloads, particularly at different batch sizes and model scales.

## Usage

Run the benchmark with configurable model and batch size settings:

```bash
# Default benchmark (Mixtral, DeepSeek-V2-Lite, Granite models)
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py

# Specific model with custom batch sizes
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py \
  --models mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --batch-sizes 1 4 8 16 32 64 128 256 512

# Test with tensor parallelism
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py \
  --models deepseek-ai/DeepSeek-V2-Lite \
  --tp-sizes 1 2 4 \
  --batch-sizes 64 128 256

# Limit to specific dimensions
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py \
  --limit-k 5120 \
  --limit-n 12288
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_grouped_gemm_cutlass.py`
**Function Signature:**
```python
def bench_run(
    results: list[benchmark.Measurement],
    model: str,
    num_experts: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    mkn: tuple[int, int, int]
):
    """
    Benchmark grouped GEMM vs Triton MOE

    Args:
        results: List to collect timing measurements
        model: Model name
        num_experts: Number of experts
        topk: Top-k expert selection
        per_act_token: Per-token activation quantization
        per_out_ch: Per-output-channel quantization
        mkn: (M, K, N) matrix dimensions

    Returns:
        None (appends results to list)
    """
```

**Import:**
```python
import torch.utils.benchmark as benchmark
from benchmark_shapes import WEIGHT_SHAPES_MOE
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `--models` | List[str] | Model names from WEIGHT_SHAPES_MOE |
| `--tp-sizes` | List[int] | Tensor parallelism sizes (default: [1]) |
| `--batch-sizes` | List[int] | Batch sizes (default: [1,4,8,...,512]) |
| `--limit-k` | List[int] | Filter to specific K dimensions |
| `--limit-n` | List[int] | Filter to specific N dimensions |
| `--limit-num-groups` | List[int] | Limit number of expert groups |
| `--limit-per-act-token` | List[int] | Filter per-act-token configs |
| `--limit-per-out-ch` | List[int] | Filter per-out-ch configs |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Timing comparison | benchmark.Compare | Comparative results across implementations |
| Per-config measurements | benchmark.Measurement | Individual timing data |
| Console output | str | Formatted benchmark results |

## Usage Examples

### Example 1: Default Benchmark
```python
# Run with default models and batch sizes
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py
```

### Example 2: Mixtral-Specific Testing
```python
# Focus on Mixtral-8x7B
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py \
  --models mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --batch-sizes 16 32 64 128 256
```

### Example 3: Tensor Parallelism Evaluation
```python
# Test TP scaling
python benchmarks/kernels/benchmark_grouped_gemm_cutlass.py \
  --models deepseek-ai/DeepSeek-V2-Lite \
  --tp-sizes 1 2 4 8 \
  --batch-sizes 128 256
```

### Example 4: Programmatic Usage
```python
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
import torch

# Configuration
num_experts = 8
topk = 2
m, k, n = 128, 4096, 14336
device = "cuda"
dtype = torch.half

# Create and quantize weights
w1 = torch.randn((num_experts, 2*n, k), device=device, dtype=dtype) / 10
w2 = torch.randn((num_experts, k, n), device=device, dtype=dtype) / 10

w1_q = torch.empty((num_experts, 2*n, k), device=device, dtype=torch.float8_e4m3fn)
w2_q = torch.empty((num_experts, k, n), device=device, dtype=torch.float8_e4m3fn)
w1_scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)
w2_scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)

for expert in range(num_experts):
    w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(w1[expert])
    w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(w2[expert])

# Create inputs
a = torch.randn((m, k), device=device, dtype=dtype) / 10
score = torch.randn((m, num_experts), device=device, dtype=dtype)
topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

# Get activation scale
_, a_scale = ops.scaled_fp8_quant(a)

# Create stride tensors for grouped GEMM
ab_strides1 = torch.full((num_experts,), k, device=device, dtype=torch.int64)
ab_strides2 = torch.full((num_experts,), n, device=device, dtype=torch.int64)
c_strides1 = torch.full((num_experts,), 2*n, device=device, dtype=torch.int64)
c_strides2 = torch.full((num_experts,), k, device=device, dtype=torch.int64)

# Create quantization config
quant_config = fp8_w8a8_moe_quant_config(
    w1_scale=w1_scale,
    w2_scale=w2_scale,
    per_act_token_quant=False
)

# Run grouped GEMM
output = cutlass_moe_fp8(
    a, w1_q, w2_q,
    topk_weights, topk_ids,
    ab_strides1, ab_strides2,
    c_strides1, c_strides2,
    quant_config=quant_config
)
```

### Example 5: CUDA Graph Execution
```python
from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
import torch

# Setup for graph capture
with set_current_vllm_config(
    VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
):
    # Create CUDA graph
    cutlass_stream = torch.cuda.Stream()
    cutlass_graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(cutlass_graph, stream=cutlass_stream):
        output = cutlass_moe_fp8(
            a, w1_q, w2_q,
            topk_weights, topk_ids,
            ab_strides1, ab_strides2,
            c_strides1, c_strides2,
            quant_config=quant_config
        )

    torch.cuda.synchronize()

    # Benchmark graph replay
    for _ in range(100):
        cutlass_graph.replay()
    torch.cuda.synchronize()
```

### Example 6: Triton MOE Comparison
```python
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

# Run Triton implementation for comparison
triton_output = fused_experts(
    a,
    w1_q,
    w2_q,
    topk_weights,
    topk_ids,
    quant_config=quant_config
)

# Compare outputs
torch.testing.assert_close(output, triton_output, rtol=1e-2, atol=1e-2)
```

## Related Pages

- **File Detail:** [benchmarks_kernels_benchmark_grouped_gemm_cutlass_py.md](../files/benchmarks_kernels_benchmark_grouped_gemm_cutlass_py.md)
- **CUTLASS MOE:** vllm.model_executor.layers.fused_moe.cutlass_moe
- **Related Benchmarks:**
  - vllm-project_vllm_CUTLASSFP8MOEBenchmark.md
  - vllm-project_vllm_NVFP4MOEBenchmark.md
- **Model Shapes:** benchmark_shapes.WEIGHT_SHAPES_MOE
- **Repository:** https://github.com/vllm-project/vllm
