# NVFP4 Hadamard Benchmark

**Knowledge Sources:** benchmarks/kernels/bench_nvfp4_qutlass.py
**Domains:** Performance, Quantization, Benchmarking
**Last Updated:** 2025-12-17

## Overview

Benchmarking tool that evaluates NVFP4 quantized GEMM performance with Hadamard transform preprocessing for improved quantization quality.

## Description

The NVFP4 Hadamard benchmark script compares BF16 matrix multiplication against NVFP4 (NVIDIA FP4) quantized operations enhanced with Hadamard transforms. This benchmark evaluates performance using CUTLASS kernels with block-scaled quantization.

The implementation applies deterministic Hadamard matrices of varying sizes (16, 32, 64, 128) to redistribute outliers in activation tensors before quantization. This preprocessing step improves quantization accuracy for 4-bit formats. The benchmark tests two modes: one with pre-quantized activations and another with on-the-fly quantization.

Key features:
- Tests across extensive batch sizes from 1 to 32,768 tokens
- Supports multiple Hadamard transform sizes for quality vs. performance tradeoffs
- Uses CUTLASS scaled FP4 matrix multiplication with block quantization (16-element blocks)
- Evaluates real model shapes from meta-llama/Llama-3.3-70B-Instruct
- Reports throughput in TFLOP/s for comparison

The benchmark is designed specifically for Blackwell architecture GPUs that have hardware support for FP4 computation, enabling extreme model compression while maintaining inference quality.

## Usage

The script can be executed directly with configurable model and tensor parallelism settings:

```bash
python benchmarks/kernels/bench_nvfp4_qutlass.py \
  --models meta-llama/Llama-3.3-70B-Instruct \
  --tp-sizes 1
```

Multiple models and TP sizes can be specified to test different configurations. Results are displayed as interactive plots showing throughput across batch sizes.

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/bench_nvfp4_qutlass.py`
**Function Signature:**
```python
def benchmark(batch_size, provider, N, K, had_size):
    """
    Benchmark function decorated with triton.testing.perf_report

    Args:
        batch_size: Number of tokens (M dimension)
        provider: "torch-bf16", "nvfp4", or "nvfp4-noquant"
        N: Output dimension
        K: Inner dimension
        had_size: Hadamard matrix size (16, 32, 64, or 128)

    Returns:
        Tuple of (median_tflops, max_tflops, min_tflops)
    """
```

**Import:**
```python
from vllm import _custom_ops as ops
from vllm._custom_ops import fusedQuantizeNv
from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `--models` | List[str] | Model names to benchmark (default: Llama-3.3-70B-Instruct) |
| `--tp-sizes` | List[int] | Tensor parallelism sizes (default: [1]) |
| `batch_size` | int | Number of tokens in batch (1 to 32768) |
| `had_size` | int | Hadamard matrix dimension (16, 32, 64, 128) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| TFLOP/s metrics | float | Throughput in teraflops per second |
| Plots | PNG files | Performance visualization by batch size |
| Console output | str | Formatted benchmark results table |

## Usage Examples

### Example 1: Default Benchmark
```python
# Run default benchmark for Llama-3.3-70B
python benchmarks/kernels/bench_nvfp4_qutlass.py
```

### Example 2: Custom Model with Multiple TP Sizes
```python
# Benchmark with tensor parallelism
python benchmarks/kernels/bench_nvfp4_qutlass.py \
  --models meta-llama/Llama-3.3-70B-Instruct \
  --tp-sizes 1 2 4 8
```

### Example 3: Programmatic Usage
```python
from benchmarks.kernels.bench_nvfp4_qutlass import benchmark, prepare_shapes

# Configure benchmark
args = argparse.Namespace(
    models=["meta-llama/Llama-3.3-70B-Instruct"],
    tp_sizes=[1]
)

# Run benchmarks for each shape
for K, N, model in prepare_shapes(args):
    for had_size in [16, 32, 64, 128]:
        benchmark.run(
            print_data=True,
            show_plots=True,
            save_path=f"bench_nvfp4_res_n{N}_k{K}",
            N=N,
            K=K,
            had_size=had_size
        )
```

## Related Pages

- **File Detail:** [benchmarks_kernels_bench_nvfp4_qutlass_py.md](../files/benchmarks_kernels_bench_nvfp4_qutlass_py.md)
- **Related Benchmark:** vllm-project_vllm_NVFP4MOEBenchmark.md (FP4 MoE evaluation)
- **Quantization Utilities:** vllm.model_executor.layers.quantization.qutlass_utils
- **CUTLASS Operations:** vllm._custom_ops.cutlass_scaled_fp4_mm
- **Repository:** https://github.com/vllm-project/vllm
