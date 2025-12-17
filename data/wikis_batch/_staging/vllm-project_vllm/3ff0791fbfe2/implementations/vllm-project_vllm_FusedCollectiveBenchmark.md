# Fused Collective Operations Benchmark

**Knowledge Sources:** benchmarks/kernels/benchmark_fused_collective.py
**Domains:** Performance, Distributed Computing, Benchmarking, Kernel Fusion
**Last Updated:** 2025-12-17

## Overview

Comprehensive benchmarking tool for evaluating FlashInfer's fused allreduce+rmsnorm+quantization operations against standard sequential implementations in tensor parallel setups.

## Description

The Fused Collective Operations benchmark compares FlashInfer's `trtllm_allreduce_fusion` kernels against vLLM's standard implementations that execute allreduce, RMSNorm, and quantization as separate operations. This benchmark is critical for validating performance benefits of kernel fusion in distributed transformer inference.

The benchmark evaluates:
1. **FlashInfer Fused Operations**:
   - Allreduce + RMSNorm
   - Allreduce + RMSNorm + FP8 Quantization
   - Allreduce + RMSNorm + FP4 Quantization
   - Both oneshot and twoshot execution modes

2. **Standard vLLM Operations**:
   - Sequential: tensor_model_parallel_all_reduce → RMSNorm → Quantization
   - Custom CUDA ops vs native torch ops
   - Torch.compile variants

Key features:
- Requires torchrun with world_size > 1 for collective operations
- Tests multiple token counts (default: 128, 512, 1024, 2048)
- Configurable hidden dimensions (default: 8192)
- Supports bfloat16, float16, and float32 data types
- Tests with/without residual connections
- Evaluates oneshot vs twoshot modes for FlashInfer
- Uses CUDA graphs for precise timing measurements
- Outputs results as markdown tables with speedup calculations
- Supports workspace sizes up to 64MB for large inputs

The benchmark helps justify using FlashInfer's optimized kernels by demonstrating reduced latency through operation fusion in distributed inference scenarios.

## Usage

Execute with torchrun for distributed benchmarking:

```bash
# Default benchmark with 2 GPUs
torchrun --nproc_per_node=2 benchmarks/kernels/benchmark_fused_collective.py

# Custom token counts and hidden dimension
torchrun --nproc_per_node=4 benchmarks/kernels/benchmark_fused_collective.py \
  --num-tokens 256 512 1024 2048 4096 \
  --hidden-dim 8192 \
  --dtypes bfloat16

# Test specific quantization modes
torchrun --nproc_per_node=2 benchmarks/kernels/benchmark_fused_collective.py \
  --quant-modes fp8,fp4 \
  --output-file results.md

# Skip residual tests and oneshot mode
torchrun --nproc_per_node=2 benchmarks/kernels/benchmark_fused_collective.py \
  --no-residual \
  --no-oneshot
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_fused_collective.py`
**Function Signature:**
```python
def run_benchmarks(
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    use_residual: bool,
    allreduce_params: FlashInferFusedAllReduceParams | None,
    quant_modes: set[str],
    no_oneshot: bool
) -> dict:
    """
    Run all benchmarks for given configuration

    Args:
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension size
        dtype: Tensor data type
        use_residual: Test with residual connections
        allreduce_params: FlashInfer parameters (None if unavailable)
        quant_modes: Set of {"none", "fp8", "fp4"}
        no_oneshot: Skip oneshot benchmarks

    Returns:
        Dictionary of operation names to timing results (ms)
    """
```

**Import:**
```python
import flashinfer.comm as flashinfer_comm
from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_all_reduce
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `--num-tokens` | List[int] | Token counts to test (default: [128,512,1024,2048]) |
| `--hidden-dim` | int | Hidden dimension size (default: 8192) |
| `--dtypes` | List[str] | Data types to test (default: ["bfloat16"]) |
| `--no-residual` | flag | Skip residual connection tests |
| `--quant-modes` | str | Comma-separated modes: none,fp8,fp4 (default: all) |
| `--warmup` | int | Warmup iterations (default: 5) |
| `--trials` | int | Benchmark trials (default: 20) |
| `--output-file` | str | Output markdown file path |
| `--no-oneshot` | flag | Skip oneshot benchmarks |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Markdown results | file | Formatted benchmark results with speedup tables |
| Console output | str | Real-time results for each configuration |
| Timing data | dict | Operation names mapped to latency (ms) |
| Speedup calculations | DataFrame | Speedups over fastest baseline |

## Usage Examples

### Example 1: Basic 2-GPU Benchmark
```bash
# Run with 2 GPUs
torchrun --nproc_per_node=2 benchmarks/kernels/benchmark_fused_collective.py
```

### Example 2: Custom Configuration
```bash
# 4 GPUs with custom settings
torchrun --nproc_per_node=4 benchmarks/kernels/benchmark_fused_collective.py \
  --num-tokens 512 1024 2048 \
  --hidden-dim 4096 \
  --dtypes float16 bfloat16 \
  --output-file my_results.md
```

### Example 3: FP8-Only Testing
```bash
# Test only FP8 quantization
torchrun --nproc_per_node=2 benchmarks/kernels/benchmark_fused_collective.py \
  --quant-modes fp8 \
  --no-residual
```

### Example 4: Programmatic FlashInfer Usage
```python
import flashinfer.comm as flashinfer_comm
import torch
from vllm.distributed import get_tp_group

# Setup workspace
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
hidden_dim = 8192
max_token_num = 1024

ipc_handles, workspace_tensor = (
    flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
        tp_rank=rank,
        tp_size=world_size,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        group=get_tp_group().device_group,
        use_fp32_lamport=False
    )
)

# Create tensors
num_tokens = 512
input_tensor = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
residual = torch.randn_like(input_tensor)
rms_gamma = torch.ones(hidden_dim, dtype=torch.bfloat16, device="cuda")
rms_eps = 1e-6

# Run fused allreduce + rmsnorm
flashinfer_comm.trtllm_allreduce_fusion(
    allreduce_in=input_tensor,
    token_num=num_tokens,
    residual_in=residual,
    residual_out=input_tensor,  # In-place update
    norm_out=input_tensor,
    rms_gamma=rms_gamma,
    rms_eps=rms_eps,
    hidden_dim=hidden_dim,
    workspace_ptrs=workspace_tensor,
    pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
    allreduce_out=None,
    quant_out=None,
    scale_out=None,
    layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
    scale_factor=None,
    use_oneshot=True,
    world_rank=rank,
    world_size=world_size,
    launch_with_pdl=True,
    trigger_completion_at_end=True,
    fp32_acc=True
)
```

### Example 5: Standard vLLM Operations
```python
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
import torch

# Create RMSNorm layer
hidden_dim = 8192
rms_norm = RMSNorm(hidden_dim, eps=1e-6, dtype=torch.bfloat16)

# Input tensors
input_tensor = torch.randn((512, hidden_dim), dtype=torch.bfloat16, device="cuda")
residual = torch.randn_like(input_tensor)

# Sequential operations
allreduce_out = tensor_model_parallel_all_reduce(input_tensor)
norm_out, residual_out = rms_norm(allreduce_out, residual)
```

### Example 6: FP8 Quantization with Fusion
```python
import flashinfer.comm as flashinfer_comm
import torch

# Create FP8 output tensors
quant_out = torch.empty((num_tokens, hidden_dim), dtype=torch.float8_e4m3fn, device="cuda")
scale_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")

# Fused allreduce + rmsnorm + FP8 quantization
flashinfer_comm.trtllm_allreduce_fusion(
    allreduce_in=input_tensor,
    token_num=num_tokens,
    residual_in=residual,
    residual_out=input_tensor,
    norm_out=input_tensor,
    rms_gamma=rms_gamma,
    rms_eps=rms_eps,
    hidden_dim=hidden_dim,
    workspace_ptrs=workspace_tensor,
    pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
    allreduce_out=None,
    quant_out=quant_out,
    scale_out=None,
    layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
    scale_factor=scale_factor,
    use_oneshot=True,
    world_rank=rank,
    world_size=world_size,
    launch_with_pdl=True,
    trigger_completion_at_end=True,
    fp32_acc=True
)
```

## Related Pages

- **File Detail:** [benchmarks_kernels_benchmark_fused_collective_py.md](../files/benchmarks_kernels_benchmark_fused_collective_py.md)
- **Distributed Operations:** vllm.distributed
- **RMSNorm Layer:** vllm.model_executor.layers.layernorm.RMSNorm
- **FP8 Quantization:** vllm.model_executor.layers.quantization.input_quant_fp8.QuantFP8
- **FlashInfer:** https://github.com/flashinfer-ai/flashinfer
- **Repository:** https://github.com/vllm-project/vllm
