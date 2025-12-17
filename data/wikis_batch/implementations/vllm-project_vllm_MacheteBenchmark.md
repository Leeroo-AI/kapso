# Machete Quantized GEMM Benchmark

**Knowledge Sources:** benchmarks/kernels/benchmark_machete.py
**Domains:** Performance, Quantization, Benchmarking, Kernel Optimization
**Last Updated:** 2025-12-17

## Overview

Comprehensive benchmarking tool for evaluating Machete's custom quantized GEMM kernels against established implementations including Marlin, CUTLASS, and PyTorch.

## Description

The Machete benchmark provides extensive performance evaluation of vLLM's Machete quantized GEMM implementation across diverse quantization configurations and matrix shapes. This benchmark is critical for validating Machete's competitiveness against proven quantization backends.

The benchmark compares:
1. **Machete**: Custom quantized GEMM with schedule optimization
2. **Marlin**: Established GPTQ/AWQ quantization kernel
3. **CUTLASS Scaled MM**: NVIDIA's optimized scaled matrix multiply
4. **PyTorch FP16**: Baseline floating-point matmul

Key features:
- **Quantization schemes**: uint4b8, uint4 with group scales, zero points, channel scales, token scales
- **Activation types**: FP16, BF16, INT8, FP8
- **Test modes**:
  - `square_bench`: Square matrices with configurable dimensions
  - `range_bench`: Custom M/K/N ranges
  - `model_bench`: Real model layer shapes (Llama-2, Llama-3, etc.)
- **Schedule sweeping**: Automatically finds optimal kernel configuration
- **Cache considerations**: Creates weights exceeding L2 cache (>100MB) for realistic measurements
- **Output formats**: Pickle files with raw measurements, CSV for schedule results

The benchmark validates when Machete provides performance advantages over established solutions across different quantization configurations, batch sizes, and hardware targets.

## Usage

The script supports three benchmarking modes:

```bash
# Square matrices
python benchmarks/kernels/benchmark_machete.py \
  --act-type float16 \
  --group-scale-type float16 \
  --group-size 128 \
  square_bench \
  --dim-start 128 \
  --dim-end 512 \
  --dim-increment 64

# Custom dimension ranges
python benchmarks/kernels/benchmark_machete.py \
  --act-type float16 \
  --group-scale-type float16 \
  --group-size 128 \
  range_bench \
  --dim-start 128,16384,16384 \
  --dim-end 512,16384,16384 \
  --dim-increment 64,0,0

# Model-specific benchmarks
python benchmarks/kernels/benchmark_machete.py \
  --act-type float16 \
  --group-scale-type float16 \
  --group-size 128 \
  --sweep-schedules \
  model_bench \
  --models meta-llama/Llama-2-7b-hf \
  --batch-sizes 16 32 64 \
  --tp-sizes 1

# FP8 activation with channel/token scales
python benchmarks/kernels/benchmark_machete.py \
  --act-type float8_e4m3fn \
  --group-scale-type float16 \
  --group-size 128 \
  --channel-scale-type float \
  --token-scale-type float \
  model_bench \
  --models meta-llama/Llama-3-8b \
  --batch-sizes 128 256
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_machete.py`
**Function Signature:**
```python
def bench(
    types: TypeConfig,
    group_size: int,
    m: int,
    k: int,
    n: int,
    label: str,
    sub_label: str,
    sweep_schedules: bool = True
) -> list[TMeasurement]:
    """
    Benchmark implementations for given configuration

    Args:
        types: Type configuration for activations, weights, scales
        group_size: Quantization group size
        m, k, n: Matrix dimensions
        label: Benchmark label
        sub_label: Configuration description
        sweep_schedules: Find optimal Machete schedule

    Returns:
        List of timing measurements
    """
```

**Import:**
```python
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_permute_scales,
    marlin_zero_points
)
from vllm.scalar_type import ScalarType, scalar_types
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `--act-type` | str | Activation dtype: bfloat16, float16, int8, float8_e4m3fn |
| `--group-scale-type` | str | Group scale dtype: bfloat16, float16 |
| `--group-zero-type` | str | Group zero point dtype: bfloat16, float16 |
| `--channel-scale-type` | str | Channel scale dtype: float |
| `--token-scale-type` | str | Token scale dtype: float |
| `--out-type` | str | Output dtype: bfloat16, float16 |
| `--group-size` | int | Quantization group size (default: 128) |
| `--sweep-schedules` | flag | Run schedule optimization sweep |
| `--sweep-csv-out` | str | CSV output for schedule results |

**Square Bench Mode:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `--dim-start` | int | Starting dimension |
| `--dim-end` | int | Ending dimension (inclusive) |
| `--dim-increment` | int | Dimension increment |

**Range Bench Mode:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `--dim-start` | str | Start M,K,N (comma-separated) |
| `--dim-end` | str | End M,K,N (comma-separated) |
| `--dim-increment` | str | Increment M,K,N (comma-separated) |

**Model Bench Mode:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `--models` | List[str] | Model names from WEIGHT_SHAPES |
| `--batch-sizes` | List[int] | Batch sizes (default: [1,16,32,...,1024]) |
| `--tp-sizes` | List[int] | Tensor parallelism sizes (default: [1]) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Pickle file | .pkl | Raw benchmark.Measurement objects |
| CSV file | .csv | Schedule sweep results (if enabled) |
| Console output | str | Formatted timing comparisons |
| Comparison table | benchmark.Compare | Performance across implementations |

## Usage Examples

### Example 1: Basic Square Benchmark
```bash
# Test square matrices from 128 to 512
python benchmarks/kernels/benchmark_machete.py \
  --act-type float16 \
  --group-scale-type float16 \
  --group-size 128 \
  square_bench \
  --dim-start 128 \
  --dim-end 512 \
  --dim-increment 64
```

### Example 2: Model Shapes with Schedule Sweep
```bash
# Benchmark Llama-2 with optimal schedule finding
python benchmarks/kernels/benchmark_machete.py \
  --act-type bfloat16 \
  --group-scale-type bfloat16 \
  --group-size 128 \
  --sweep-schedules \
  --sweep-csv-out schedules.csv \
  model_bench \
  --models meta-llama/Llama-2-70b-hf \
  --batch-sizes 1 16 128 \
  --tp-sizes 1
```

### Example 3: FP8 with Channel and Token Scales
```bash
# Advanced FP8 quantization with per-channel and per-token scales
python benchmarks/kernels/benchmark_machete.py \
  --act-type float8_e4m3fn \
  --group-scale-type float16 \
  --group-size 128 \
  --channel-scale-type float \
  --token-scale-type float \
  --out-type float16 \
  model_bench \
  --models meta-llama/Llama-3-8b \
  --batch-sizes 128
```

### Example 4: Programmatic Machete Usage
```python
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
import torch

# Create quantized weights
m, k, n = 128, 4096, 4096
wtype = scalar_types.uint4b8
a = torch.randn((m, k), dtype=torch.float16, device="cuda")
w = torch.randn((k, n), dtype=torch.float16, device="cuda")

# Quantize and pack weights
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights, pack_rows
)
_, w_q_packed, w_s, w_zp = quantize_weights(
    w, wtype, group_size=128, zero_points=False
)
w_q_packed = pack_rows(w_q_packed, wtype.size_bits, *w_q_packed.shape)

# Prepack for Machete
w_q = w_q_packed.t().contiguous().t()  # Make column-major
w_q_prepacked = ops.machete_prepack_B(
    w_q,
    a.dtype,
    wtype,
    w_s.dtype if w_s is not None else None
)

# Run Machete GEMM
output = ops.machete_mm(
    a=a,
    b_q=w_q_prepacked,
    b_type=wtype,
    b_group_scales=w_s,
    b_group_zeros=None,
    b_group_size=128,
    b_channel_scales=None,
    a_token_scales=None,
    out_type=torch.float16
)
```

### Example 5: Schedule Sweeping
```python
from vllm import _custom_ops as ops

# Get supported schedules for configuration
schedules = ops.machete_supported_schedules(
    a_type=torch.float16,
    b_type=scalar_types.uint4b8,
    group_scales_type=torch.float16,
    group_zeros_type=None,
    token_scales_type=None,
    channel_scales_type=None,
    out_type=torch.float16
)

print(f"Found {len(schedules)} supported schedules")

# Benchmark with specific schedule
best_schedule = None
best_time = float('inf')

for schedule in schedules:
    # Run with specific schedule
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    output = ops.machete_mm(
        a=a, b_q=w_q_prepacked, b_type=wtype,
        b_group_scales=w_s, b_group_size=128,
        out_type=torch.float16,
        schedule=schedule
    )
    end.record()
    end.synchronize()

    time_ms = start.elapsed_time(end)
    if time_ms < best_time:
        best_time = time_ms
        best_schedule = schedule

print(f"Best schedule: {best_schedule}, time: {best_time:.3f} ms")
```

### Example 6: Comparing Implementations
```python
import torch.utils.benchmark as TBenchmark

# Create benchmark timers for each implementation
timers = []

# PyTorch baseline
timers.append(TBenchmark.Timer(
    stmt="torch.matmul(a, w)",
    globals={"a": a, "w": w, "torch": torch}
))

# Machete
timers.append(TBenchmark.Timer(
    stmt="ops.machete_mm(a, w_q, wtype, w_s, None, 128, None, None, torch.float16)",
    globals={
        "ops": ops, "a": a, "w_q": w_q_prepacked,
        "wtype": wtype, "w_s": w_s, "torch": torch
    }
))

# Run and compare
results = [timer.blocked_autorange(min_run_time=1) for timer in timers]
compare = TBenchmark.Compare(results)
compare.print()
```

## Related Pages

- **File Detail:** [benchmarks_kernels_benchmark_machete_py.md](../files/benchmarks_kernels_benchmark_machete_py.md)
- **Machete Operations:** vllm._custom_ops (machete_mm, machete_prepack_B)
- **Quantization Utilities:** vllm.model_executor.layers.quantization.utils.quant_utils
- **Scalar Types:** vllm.scalar_type
- **Related Benchmarks:**
  - vllm-project_vllm_BitBLASBenchmark.md (alternative low-bit backend)
- **Repository:** https://github.com/vllm-project/vllm
