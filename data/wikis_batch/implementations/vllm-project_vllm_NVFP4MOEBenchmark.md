# NVFP4 MOE Kernel Benchmark

**Knowledge Sources:** benchmarks/kernels/benchmark_cutlass_fp4_moe.py
**Domains:** Performance, Quantization, Benchmarking, Mixture-of-Experts
**Last Updated:** 2025-12-17

## Overview

Benchmarking tool comparing NVFP4 block-scaled CUTLASS MOE kernels against FP8 tensor-scaled Triton MOE kernels for extreme model compression.

## Description

The NVFP4 MOE benchmark evaluates performance of CUTLASS-based FP4 quantized mixture-of-experts kernels against FP8 Triton implementations. This benchmark is crucial for validating 4-bit MOE deployment on Blackwell architecture GPUs.

The benchmark compares:
1. **NVFP4 CUTLASS MOE**: Block-scaled FP4 quantization (16-element blocks) with CUTLASS grouped GEMM
2. **FP8 Triton MOE**: Tensor-scaled FP8 quantization with Triton fused experts

Key features:
- Tests DeepSeek-R1-FP4 model configuration (256 experts, top-8 routing)
- Evaluates batch sizes from 4 to 2048 tokens
- Uses block-level scales (16-element blocks) for FP4 weights
- Supports both raw kernel calls and CUDA graph execution
- Measures performance with and without CUDA graphs
- Uses NVTX annotations for detailed profiling
- Reports latency per invocation (accounts for multiple operations per graph)

The FP4 format enables 2x compression over FP8, critical for massive expert counts in large MOE models. Block scaling provides better accuracy than tensor scaling at extreme quantization levels.

## Usage

Run the benchmark with configurable model and batch size settings:

```bash
# Default benchmark (DeepSeek-R1-FP4, various batch sizes)
python benchmarks/kernels/benchmark_cutlass_fp4_moe.py

# Custom batch sizes
python benchmarks/kernels/benchmark_cutlass_fp4_moe.py \
  --models nvidia/DeepSeek-R1-FP4 \
  --batch-sizes 16 32 64 128 \
  --tp-sizes 1

# Limit dimensions for faster testing
python benchmarks/kernels/benchmark_cutlass_fp4_moe.py \
  --limit-k 2048 \
  --limit-n 7168
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_cutlass_fp4_moe.py`
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
    Run benchmark comparing FP4 and FP8 MOE implementations

    Args:
        results: List to store timing measurements
        model: Model name
        num_experts: Number of experts
        topk: Number of experts to route to
        per_act_token: Per-token activation quantization
        per_out_ch: Per-output-channel quantization
        mkn: (M, K, N) matrix dimensions

    Returns:
        None (appends results to list)
    """
```

**Import:**
```python
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import (
    fp8_w8a8_moe_quant_config,
    nvfp4_moe_quant_config
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `--models` | List[str] | Model names (default: nvidia/DeepSeek-R1-FP4) |
| `--tp-sizes` | List[int] | Tensor parallelism sizes (default: [1]) |
| `--batch-sizes` | List[int] | Batch sizes (default: [4,8,16,...,2048]) |
| `--limit-k` | List[int] | Limit to specific K dimensions |
| `--limit-n` | List[int] | Limit to specific N dimensions |
| `--limit-num-groups` | List[int] | Limit number of expert groups |
| `--limit-per-act-token` | List[int] | Filter per-act-token configs |
| `--limit-per-out-ch` | List[int] | Filter per-out-ch configs |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Timing table | benchmark.Compare | Comparison of all implementations |
| Per-config results | benchmark.Measurement | Individual timing measurements |
| Console output | str | Formatted benchmark results |

## Usage Examples

### Example 1: Default Benchmark
```python
# Run with default settings
python benchmarks/kernels/benchmark_cutlass_fp4_moe.py
```

### Example 2: Custom Batch Sizes
```python
# Test specific batch sizes
python benchmarks/kernels/benchmark_cutlass_fp4_moe.py \
  --batch-sizes 32 64 128 256
```

### Example 3: Tensor Parallelism Scaling
```python
# Test TP scaling
python benchmarks/kernels/benchmark_cutlass_fp4_moe.py \
  --tp-sizes 1 2 4 8 \
  --batch-sizes 128 512 1024
```

### Example 4: Programmatic Usage
```python
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4
from vllm import _custom_ops as ops
import torch

# Create FP4 quantized weights
num_experts = 256
k = 2048
n = 7168
w1_fp4 = torch.randint(0, 256, (num_experts, 2*n, k//2), dtype=torch.uint8, device="cuda")
w2_fp4 = torch.randint(0, 256, (num_experts, k, n//2), dtype=torch.uint8, device="cuda")

# Create block scales
w1_blockscale = torch.randn((num_experts, 2*n, k//16), dtype=torch.float8_e4m3fn, device="cuda")
w2_blockscale = torch.randn((num_experts, k, n//16), dtype=torch.float8_e4m3fn, device="cuda")

# Global scales
w1_gs = torch.ones((num_experts,), dtype=torch.float32, device="cuda")
w2_gs = torch.ones((num_experts,), dtype=torch.float32, device="cuda")
a1_gs = torch.ones((num_experts,), dtype=torch.float32, device="cuda")
a2_gs = torch.ones((num_experts,), dtype=torch.float32, device="cuda")

# Create quantization config
quant_config = nvfp4_moe_quant_config(
    a1_gscale=a1_gs,
    a2_gscale=a2_gs,
    w1_scale=w1_blockscale,
    w2_scale=w2_blockscale,
    g1_alphas=w1_gs,
    g2_alphas=w2_gs
)

# Run MOE forward pass
m = 128
a = torch.randn((m, k), dtype=torch.half, device="cuda")
topk_weights = torch.randn((m, 8), dtype=torch.half, device="cuda")
topk_ids = torch.randint(0, num_experts, (m, 8), dtype=torch.int32, device="cuda")

output = cutlass_moe_fp4(
    a=a,
    w1_fp4=w1_fp4,
    w2_fp4=w2_fp4,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    m=m,
    n=n,
    k=k,
    e=num_experts,
    quant_config=quant_config
)
```

### Example 5: CUDA Graph Execution
```python
import torch
from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config

# Setup for CUDA graph capture
with set_current_vllm_config(
    VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
):
    # Create graph
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()

    with torch.cuda.graph(graph, stream=stream):
        output = cutlass_moe_fp4(
            a=a,
            w1_fp4=w1_fp4,
            w2_fp4=w2_fp4,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            m=m, n=n, k=k, e=num_experts,
            quant_config=quant_config
        )

    # Replay graph for performance
    torch.cuda.synchronize()
    graph.replay()
```

## Related Pages

- **File Detail:** [benchmarks_kernels_benchmark_cutlass_fp4_moe_py.md](../files/benchmarks_kernels_benchmark_cutlass_fp4_moe_py.md)
- **CUTLASS MOE:** vllm.model_executor.layers.fused_moe.cutlass_moe
- **Related Benchmark:** vllm-project_vllm_CUTLASSGroupedGEMMBenchmark.md
- **FP4 Quantization:** vllm-project_vllm_NVFP4HadamardBenchmark.md
- **Repository:** https://github.com/vllm-project/vllm
