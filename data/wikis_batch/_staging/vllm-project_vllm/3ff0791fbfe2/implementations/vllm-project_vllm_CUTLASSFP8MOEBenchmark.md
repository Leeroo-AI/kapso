# CUTLASS FP8 MOE Benchmark

**Knowledge Sources:** benchmarks/kernels/benchmark_cutlass_moe_fp8.py
**Domains:** Performance, Quantization, Benchmarking, Mixture-of-Experts
**Last Updated:** 2025-12-17

## Overview

Benchmarking tool comparing CUTLASS FP8 MOE kernels against Triton FP8 MOE implementations for quantized mixture-of-experts inference.

## Description

The CUTLASS FP8 MOE benchmark evaluates two FP8-quantized mixture-of-experts implementations to determine the optimal kernel for different model configurations and batch sizes. Both implementations use FP8 quantization but differ in their execution backend and optimization strategies.

The benchmark compares:
1. **CUTLASS MOE FP8**: CUTLASS-based grouped GEMM with FP8 quantization
2. **Triton MOE FP8**: Triton-based fused experts with FP8 quantization

Key features:
- Tests multiple model architectures: Mixtral-8x7B, DeepSeek-V2, GLM-4-5, Llama-4-Maverick
- Supports configurable per-activation-token and per-output-channel quantization
- Evaluates batch sizes from 4 to 2048 tokens
- Uses CUDA graphs for precise timing measurements (10 operations per graph)
- Measures performance using CUDA events
- Reports latency in microseconds per operation
- Provides detailed configuration tables for each model

The benchmark forces per-tensor quantization as a workaround for known issues with per-token quantization in CUTLASS MoE FP8, matching the working end-to-end setup.

## Usage

Run the benchmark with model and configuration options:

```bash
# Default benchmark (Mixtral-8x7B)
python benchmarks/kernels/benchmark_cutlass_moe_fp8.py

# Specific model with custom settings
python benchmarks/kernels/benchmark_cutlass_moe_fp8.py \
  --models "Llama-4-Maverick-17B-128E-Instruct-FP8" \
  --tp-sizes 8 \
  --batch-sizes 2 4 8 \
  --per-act-token-opts false \
  --per-out-ch-opts false

# DeepSeek-V2 evaluation
python benchmarks/kernels/benchmark_cutlass_moe_fp8.py \
  --models deepseek-v2 \
  --batch-sizes 16 32 64 128
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_cutlass_moe_fp8.py`
**Function Signature:**
```python
def bench_run(
    results: list,
    model: str,
    num_experts: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    mkn: tuple[int, int, int]
) -> dict:
    """
    Benchmark CUTLASS and Triton MOE implementations

    Args:
        results: List to collect results (deprecated)
        model: Model name
        num_experts: Number of experts
        topk: Top-k expert selection
        per_act_token: Per-token activation quantization
        per_out_ch: Per-output-channel quantization
        mkn: (M, K, N) matrix dimensions

    Returns:
        Dictionary with batch_size, triton_time_us, cutlass_time_us
    """
```

**Import:**
```python
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `--models` | List[str] | Model names from predefined configs |
| `--tp-sizes` | List[int] | Tensor parallelism sizes (default: [1]) |
| `--batch-sizes` | List[int] | Batch sizes (default: [4,8,16,...,2048]) |
| `--limit-k` | List[int] | Filter to specific K dimensions |
| `--limit-n` | List[int] | Filter to specific N dimensions |
| `--per-act-token-opts` | List[bool] | Per-activation quantization flags |
| `--per-out-ch-opts` | List[bool] | Per-output-channel quantization flags |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Results table | str | Formatted table with Triton and CUTLASS timings |
| Per-config timing | dict | Batch size, Triton time (us), CUTLASS time (us) |
| Console output | str | Configuration and timing information |

## Usage Examples

### Example 1: Default Benchmark
```python
# Run Mixtral-8x7B benchmark
python benchmarks/kernels/benchmark_cutlass_moe_fp8.py
```

### Example 2: Llama Maverick with TP=8
```python
# Benchmark large-scale MOE with tensor parallelism
python benchmarks/kernels/benchmark_cutlass_moe_fp8.py \
  --models "Llama-4-Maverick-17B-128E-Instruct-FP8" \
  --tp-sizes 8 \
  --batch-sizes 2 4 8 16
```

### Example 3: Limited Dimension Testing
```python
# Test specific dimensions quickly
python benchmarks/kernels/benchmark_cutlass_moe_fp8.py \
  --models mixtral-8x7b \
  --limit-k 4096 \
  --limit-n 14336 \
  --batch-sizes 64 128
```

### Example 4: Programmatic Usage
```python
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8
from vllm import _custom_ops as ops
import torch

# Create FP8 quantized weights
num_experts = 8
m, k, n = 128, 4096, 14336
device = "cuda"
dtype = torch.half

# Quantize weights
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
a = torch.randn((m, k), device=device, dtype=dtype)
score = torch.randn((m, num_experts), device=device, dtype=dtype)
topk_weights, topk_ids, _ = fused_topk(a, score, topk=2, renormalize=False)

# Create stride tensors
ab_strides1 = torch.full((num_experts,), k, dtype=torch.int64, device=device)
ab_strides2 = torch.full((num_experts,), n, dtype=torch.int64, device=device)
c_strides1 = torch.full((num_experts,), 2*n, dtype=torch.int64, device=device)
c_strides2 = torch.full((num_experts,), k, dtype=torch.int64, device=device)

# Setup quantization config
a1_scale = torch.full((), 1e-2, device=device, dtype=torch.float32)
a2_scale = torch.full((), 1e-2, device=device, dtype=torch.float32)

quant_config = fp8_w8a8_moe_quant_config(
    w1_scale=w1_scale,
    w2_scale=w2_scale,
    a1_scale=a1_scale,
    a2_scale=a2_scale,
    per_act_token_quant=False,
    per_out_ch_quant=False
)

# Run CUTLASS MOE
output = cutlass_moe_fp8(
    a=a,
    w1_q=w1_q,
    w2_q=w2_q,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    ab_strides1=ab_strides1,
    ab_strides2=ab_strides2,
    c_strides1=c_strides1,
    c_strides2=c_strides2,
    quant_config=quant_config,
    activation="silu",
    global_num_experts=num_experts
)
```

### Example 5: CUDA Graph Benchmarking
```python
import torch

# Create CUDA graph for CUTLASS
cutlass_stream = torch.cuda.Stream()
cutlass_graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(cutlass_graph, stream=cutlass_stream):
    for _ in range(10):  # Capture 10 invocations
        cutlass_moe_fp8(
            a=a, w1_q=w1_q, w2_q=w2_q,
            topk_weights=topk_weights, topk_ids=topk_ids,
            ab_strides1=ab_strides1, ab_strides2=ab_strides2,
            c_strides1=c_strides1, c_strides2=c_strides2,
            quant_config=quant_config,
            activation="silu",
            global_num_experts=num_experts
        )

torch.cuda.synchronize()

# Benchmark with timing
start_event = torch.Event(enable_timing=True)
end_event = torch.Event(enable_timing=True)

start_event.record()
cutlass_graph.replay()
end_event.record()
end_event.synchronize()

# Time is for 10 operations
time_per_op_ms = start_event.elapsed_time(end_event) / 10
```

## Related Pages

- **File Detail:** [benchmarks_kernels_benchmark_cutlass_moe_fp8_py.md](../files/benchmarks_kernels_benchmark_cutlass_moe_fp8_py.md)
- **CUTLASS MOE:** vllm.model_executor.layers.fused_moe.cutlass_moe
- **Related Benchmark:** vllm-project_vllm_CUTLASSGroupedGEMMBenchmark.md
- **FP4 Comparison:** vllm-project_vllm_NVFP4MOEBenchmark.md
- **Repository:** https://github.com/vllm-project/vllm
