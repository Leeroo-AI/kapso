# Benchmark Marlin Quantized GEMM

**Knowledge Sources:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_marlin.py`
**Domains:** Performance Testing, Kernel Benchmarking, Quantization, GEMM Operations
**Last Updated:** 2025-12-17

## Overview

A comprehensive benchmarking script for Marlin quantized GEMM kernels that compares performance across different quantization types, group sizes, activation ordering, and batch sizes.

## Description

The `benchmark_marlin.py` script provides extensive performance testing capabilities for Marlin quantized matrix multiplication kernels. It supports multiple quantization schemes including GPTQ Marlin, Marlin 24 (2:4 sparsity), FP4 Marlin, FP8 Marlin, and AllSpark W8A16 implementations. The benchmark compares quantized GEMM performance against PyTorch's native FP16 matmul baseline across various configurations.

The script uses a parameterized testing approach that sweeps through model architectures (defined in WEIGHT_SHAPES), batch sizes, quantization types (uint4, uint8, int4, int8, float4_e2m1f, float8_e4m3fn), group sizes, activation ordering modes, and k_full flags. It measures kernel execution time using CUDA events and presents comparative performance data.

Key capabilities include:
- Benchmarking GPTQ Marlin GEMM (FP16 and FP32 output variants)
- Testing Marlin 24 2:4 structured sparse GEMM
- Evaluating GPTQ Marlin repack operations
- Comparing AllSpark W8A16 quantized GEMM
- Support for act_order (activation ordering) optimizations
- Testing k_full vs k_partial scenarios
- Configurable search space with limit filters for focused benchmarking

The implementation generates appropriate quantized weights for each scheme, handles workspace allocation for different Marlin variants, and uses PyTorch's benchmark.Timer with blocked_autorange for accurate timing measurements.

## Usage

The script is designed to run as a standalone benchmarking tool with command-line configuration options.

**Basic Usage:**

```bash
# Run full benchmark suite on default models with default batch sizes
python benchmarks/kernels/benchmark_marlin.py

# Quick benchmark with limited configuration
python benchmarks/kernels/benchmark_marlin.py \
    --batch-sizes 1 16 32 \
    --limit-k 4096 \
    --limit-n 4096 \
    --limit-group-size 128 \
    --limit-num-bits 4 \
    --limit-act-order 0 \
    --limit-k-full 1

# Benchmark specific models
python benchmarks/kernels/benchmark_marlin.py \
    --models meta-llama/Llama-2-7b-hf/TP1 \
    --batch-sizes 256 512 1024
```

**Command-line Arguments:**

- `--models`: List of model names from WEIGHT_SHAPES (default: meta-llama/Llama-2-7b-hf/TP1)
- `--batch-sizes`: Batch sizes to test (default: 1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
- `--limit-k`: Filter by hidden dimension K
- `--limit-n`: Filter by output dimension N
- `--limit-group-size`: Filter by quantization group sizes
- `--limit-num-bits`: Filter by quantization bit widths
- `--limit-act-order`: Filter by activation ordering (0 or 1)
- `--limit-k-full`: Filter by k_full flag (0 or 1)

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_marlin.py`

**Primary Function:**

```python
def bench_run(
    results: list[benchmark.Measurement],
    model: str,
    act_order: bool,
    is_k_full: bool,
    quant_type: ScalarType,
    group_size: int,
    size_m: int,
    size_k: int,
    size_n: int,
) -> None
```

**Entry Point:**

```python
def main(args) -> None
```

**Import:**

```python
# Run as script
python benchmarks/kernels/benchmark_marlin.py [OPTIONS]
```

## I/O Contract

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `models` | list[str] | Model names defining weight shape configurations to benchmark |
| `batch_sizes` | list[int] | Batch sizes (M dimension) to test across |
| `limit_k` | list[int] | Optional filter for K dimension values |
| `limit_n` | list[int] | Optional filter for N dimension values |
| `limit_group_size` | list[int] | Optional filter for quantization group sizes |
| `limit_num_bits` | list[int] | Optional filter for quantization bit widths |
| `limit_act_order` | list[int] | Optional filter for activation ordering flag |
| `limit_k_full` | list[int] | Optional filter for k_full flag |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Console Output | benchmark.Compare | Formatted comparison table showing kernel performance across configurations |
| Measurements | list[benchmark.Measurement] | Timing data for each tested configuration including median/min/max |

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `quant_type` | ScalarType | Quantization data type (uint4, uint8, int4, int8, float4_e2m1f, float8_e4m3fn) |
| `group_size` | int | Number of elements per quantization group (-1 for per-channel) |
| `act_order` | bool | Whether to use activation ordering optimization |
| `is_k_full` | bool | Whether K dimension is fully processed |
| `size_m` | int | Batch size dimension |
| `size_k` | int | Hidden/input dimension |
| `size_n` | int | Output dimension |

## Usage Examples

### Example 1: Quick Performance Check

```python
# Run focused benchmark on specific configuration
# Command:
python benchmarks/kernels/benchmark_marlin.py \
    --batch-sizes 1 16 32 \
    --limit-k 4096 \
    --limit-n 4096 \
    --limit-group-size 128 \
    --limit-num-bits 4 \
    --limit-act-order 0 \
    --limit-k-full 1

# This tests:
# - 3 batch sizes (1, 16, 32)
# - Single K=4096, N=4096 configuration
# - Group size 128 only
# - 4-bit quantization only
# - No activation ordering
# - k_full mode only
```

### Example 2: Comprehensive Model Benchmark

```python
# Test full batch size sweep on Llama-2-7B
python benchmarks/kernels/benchmark_marlin.py \
    --models meta-llama/Llama-2-7b-hf/TP1

# This benchmarks all supported:
# - Quantization types
# - Group sizes (128, 64, -1)
# - Batch sizes (1 to 8192)
# - Act order combinations
# - k_full modes
# Across all layer shapes in the model
```

### Example 3: Comparing Quantization Schemes

```python
# Compare 4-bit vs 8-bit quantization
python benchmarks/kernels/benchmark_marlin.py \
    --batch-sizes 256 512 \
    --limit-num-bits 4 8 \
    --limit-group-size 128 \
    --limit-act-order 0

# Output shows comparative performance:
# - pytorch_gemm (FP16 baseline)
# - gptq_marlin_gemm (optimized quantized)
# - gptq_marlin_gemm_fp32 (FP32 accumulation)
# - gptq_marlin_24_gemm (2:4 sparse if supported)
# - allspark_w8a16_gemm (W8A16 if supported)
```

### Example 4: Testing FP8 Quantization

```python
# Benchmark FP8 variants
python benchmarks/kernels/benchmark_marlin.py \
    --batch-sizes 128 256 512 \
    --limit-group-size -1 128 \
    --limit-act-order 0

# FP8 (float8_e4m3fn) tests will run with:
# - Group sizes: -1 (per-channel) or 128
# - No activation ordering (not supported for FP8)
# - Comparing against FP16 baseline
```

### Example 5: Repack Operation Benchmark

```python
# Focus on GPTQ repack performance
python benchmarks/kernels/benchmark_marlin.py \
    --batch-sizes 1024 \
    --limit-num-bits 4 \
    --limit-group-size 128

# This includes gptq_marlin_repack timings showing:
# - Cost of repacking GPTQ format to Marlin format
# - Important for first-time model loading optimization
```

## Implementation Notes

**Quantization Support:**
- GPTQ Marlin: uint4, uint8, int4, int8 with flexible group sizes
- Marlin 24: 2:4 structured sparsity with uint4, uint8, int4, int8
- FP4 Marlin: float4_e2m1f with group_size=16 only
- FP8 Marlin: float8_e4m3fn with group_size=-1 or 128
- AllSpark W8A16: uint8/int8 per-channel (group_size=-1) on Ampere GPUs

**Performance Characteristics:**
- Marlin kernels typically show 2-4x speedup over PyTorch FP16 at batch size >= 16
- 2:4 sparse variants provide additional 1.5-2x gains when sparsity pattern is available
- FP8 quantization offers best memory bandwidth efficiency
- Activation ordering can improve performance but increases complexity

**Workspace Requirements:**
- GPTQ Marlin workspace: allocated based on GPTQ_MARLIN_MIN_THREAD_N and GPTQ_MARLIN_MAX_PARALLEL
- Marlin 24 workspace: allocated based on GPTQ_MARLIN_24_MIN_THREAD_N and GPTQ_MARLIN_24_MAX_PARALLEL
- Workspace size depends on output dimension N

**Limitations:**
- act_order not supported with has_zp (zero-point) quantization
- act_order requires group_size < size_k
- FP4 only supports group_size=16
- FP8 only supports group_size=-1 or 128
- AllSpark requires Ampere architecture (sm_version 80-89)

## Related Pages

- **vllm Marlin Quantization Utils** - Weight quantization and packing utilities
- **vllm GPTQ Marlin Layers** - Integration of Marlin kernels in model layers
- **vllm Custom Ops** - Custom CUDA operations including Marlin kernels
- **vllm Benchmark Shapes** - Model weight shape definitions for benchmarking
- **vllm Quantization Layers** - High-level quantization layer implementations
- **vllm Scalar Types** - Type system for quantized data formats

**See Also:**
- https://github.com/vllm-project/vllm - vLLM repository
- Marlin quantization paper and documentation
- GPTQ quantization methodology
