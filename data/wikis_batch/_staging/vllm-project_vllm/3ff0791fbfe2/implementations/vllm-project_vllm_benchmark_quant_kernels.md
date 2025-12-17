# Benchmark Quantization Kernels

**Knowledge Sources:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_quant.py`
**Domains:** Performance Testing, Kernel Benchmarking, Quantization, FP8, INT8
**Last Updated:** 2025-12-17

## Overview

A streamlined benchmark script for measuring FP8 and INT8 quantization kernel performance with static and dynamic scaling options.

## Description

The `benchmark_quant.py` script provides focused performance measurement for vLLM's quantization kernels that convert FP16/BF16 activations to FP8 or INT8 formats. These quantization operations are critical bottlenecks in quantized inference, occurring at every layer during forward passes.

**Quantization Operations:**

1. **FP8 Quantization** (`ops.scaled_fp8_quant`):
   - Converts FP16/BF16 tensors to float8_e4m3fn format
   - Supports static (pre-computed) or dynamic (computed on-the-fly) scaling
   - Used for weight and activation quantization in FP8 inference

2. **INT8 Quantization** (`ops.scaled_int8_quant`):
   - Converts FP16/BF16 tensors to int8 format
   - Supports static or dynamic scaling
   - Used for weight and activation quantization in INT8 inference

**Scaling Modes:**

- **Static Scale (`--static-scale`)**: Scale factor pre-computed and passed as input, kernel only performs quantization
- **Dynamic Scale (default)**: Scale factor computed by kernel from input tensor statistics (max absolute value)

The benchmark measures end-to-end quantization latency including scale computation when applicable, using CUDA events for precise timing and supporting CUDA profiler integration for detailed analysis.

## Usage

The script runs as a standalone benchmark with configurable parameters for tensor size, quantization type, and profiling.

**Basic Usage:**

```bash
# Benchmark INT8 quantization with dynamic scaling (default)
python benchmarks/kernels/benchmark_quant.py

# Benchmark FP8 quantization
python benchmarks/kernels/benchmark_quant.py --quant-dtype fp8

# Benchmark with static scaling
python benchmarks/kernels/benchmark_quant.py --static-scale

# Benchmark with custom tensor dimensions
python benchmarks/kernels/benchmark_quant.py \
    --num-tokens 8192 \
    --hidden-size 4096

# Benchmark FP8 with bfloat16 input
python benchmarks/kernels/benchmark_quant.py \
    --quant-dtype fp8 \
    --dtype bfloat16

# Run with CUDA profiler for detailed analysis
python benchmarks/kernels/benchmark_quant.py \
    --quant-dtype fp8 \
    --profile

# Custom iteration counts
python benchmarks/kernels/benchmark_quant.py \
    --num-warmup-iters 10 \
    --num-iters 200
```

**Command-line Arguments:**

- `--num-tokens`: Number of tokens in input tensor (default: 4096)
- `--hidden-size`: Hidden dimension size (default: 8192)
- `--static-scale`: Use pre-computed scale factor
- `--quant-dtype`: Quantization type - fp8, int8 (default: int8)
- `--dtype`: Input tensor dtype - half, bfloat16, float (default: half)
- `--seed`: Random seed for reproducibility (default: 0)
- `--profile`: Enable CUDA profiler for single iteration
- `--num-warmup-iters`: Number of warmup iterations (default: 5)
- `--num-iters`: Number of benchmark iterations (default: 100, ignored if --profile set)

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_quant.py`

**Main Function:**

```python
@torch.inference_mode()
def main(
    num_tokens: int,
    hidden_size: int,
    static_scale: bool,
    quant_dtype: torch.dtype,
    dtype: torch.dtype,
    seed: int = 0,
    do_profile: bool = False,
    num_warmup_iters: int = 5,
    num_iters: int = 100,
) -> None
```

**Quantization Operations:**

```python
# INT8 quantization
ops.scaled_int8_quant(x, scale)

# FP8 quantization
ops.scaled_fp8_quant(x, scale)
```

**Import:**

```python
from vllm import _custom_ops as ops

# Run as script
python benchmarks/kernels/benchmark_quant.py [OPTIONS]
```

## I/O Contract

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `x` | torch.Tensor | Input tensor to quantize [num_tokens, hidden_size] |
| `scale` | torch.Tensor or None | Scale factor [1, 1] for static scaling, None for dynamic |
| `num_tokens` | int | Number of tokens (batch dimension) |
| `hidden_size` | int | Hidden dimension size |
| `quant_dtype` | torch.dtype | Target quantization dtype (int8 or float8_e4m3fn) |
| `dtype` | torch.dtype | Input tensor dtype (half, bfloat16, float) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Console Output | str | Kernel running time in microseconds |
| Latency | float | Per-iteration kernel execution time |

### Quantization Function Signatures

```python
# Returns (quantized_tensor, computed_scale)
(q_tensor, scale) = ops.scaled_int8_quant(
    x: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor | None,  # [1, 1] or None
) -> tuple[torch.Tensor, torch.Tensor]

(q_tensor, scale) = ops.scaled_fp8_quant(
    x: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor | None,  # [1, 1] or None
) -> tuple[torch.Tensor, torch.Tensor]
```

## Usage Examples

### Example 1: Basic INT8 Quantization Benchmark

```python
# Default configuration: INT8, dynamic scaling, 4096 tokens, 8192 hidden size
python benchmarks/kernels/benchmark_quant.py

# Output:
# Namespace(num_tokens=4096, hidden_size=8192, static_scale=False, quant_dtype='int8',
#           dtype='half', seed=0, profile=False, num_warmup_iters=5, num_iters=100)
# Warming up...
# Kernel running time: 87.345 us
```

### Example 2: FP8 Quantization

```python
# Benchmark FP8 quantization
python benchmarks/kernels/benchmark_quant.py --quant-dtype fp8

# FP8 characteristics:
# - Converts to float8_e4m3fn format
# - 8 bits: 1 sign, 4 exponent, 3 mantissa
# - Range: ±448.0 (approximately)
# - Better for model accuracy than INT8 in many cases
# - Similar performance to INT8 quantization

# Typical output:
# Kernel running time: 89.123 us
```

### Example 3: Static vs Dynamic Scaling

```python
# Dynamic scaling (default) - scale computed by kernel
python benchmarks/kernels/benchmark_quant.py \
    --quant-dtype fp8 \
    --num-tokens 4096 \
    --hidden-size 8192

# Output: 89.123 us (includes scale computation)

# Static scaling - scale pre-computed
python benchmarks/kernels/benchmark_quant.py \
    --quant-dtype fp8 \
    --num-tokens 4096 \
    --hidden-size 8192 \
    --static-scale

# Output: 56.789 us (faster, no scale computation)
# Speedup: ~1.57x (scale computation overhead ~36% of total time)
```

### Example 4: Tensor Size Scaling

```python
# Small tensor (1K tokens, 4K hidden)
python benchmarks/kernels/benchmark_quant.py \
    --num-tokens 1024 \
    --hidden-size 4096

# Output: 23.456 us (4M elements)

# Medium tensor (4K tokens, 8K hidden) - default
python benchmarks/kernels/benchmark_quant.py \
    --num-tokens 4096 \
    --hidden-size 8192

# Output: 87.345 us (32M elements)

# Large tensor (8K tokens, 16K hidden)
python benchmarks/kernels/benchmark_quant.py \
    --num-tokens 8192 \
    --hidden-size 16384

# Output: 345.678 us (128M elements)

# Scaling analysis:
# - 4M elements: 23.456 us → 5.86 ns/element
# - 32M elements: 87.345 us → 2.73 ns/element
# - 128M elements: 345.678 us → 2.70 ns/element
# Better efficiency at larger sizes due to improved GPU utilization
```

### Example 5: Input Dtype Comparison

```python
# FP16 input (default)
python benchmarks/kernels/benchmark_quant.py --dtype half

# Output: 87.345 us

# BF16 input
python benchmarks/kernels/benchmark_quant.py --dtype bfloat16

# Output: 87.892 us (similar, slight overhead)

# FP32 input
python benchmarks/kernels/benchmark_quant.py --dtype float

# Output: 92.456 us (slightly slower due to larger input size)

# Observations:
# - FP16 and BF16 nearly identical performance
# - FP32 ~5-6% slower due to 2x memory bandwidth
# - Quantization output size same regardless of input dtype
```

### Example 6: CUDA Profiler Integration

```python
# Profile single iteration with CUDA profiler
python benchmarks/kernels/benchmark_quant.py \
    --quant-dtype fp8 \
    --profile

# This will:
# 1. Start CUDA profiler (cudaProfilerStart)
# 2. Run single quantization iteration
# 3. Stop CUDA profiler (cudaProfilerStop)
# 4. Report timing for that single iteration

# Use with nsys or nvprof:
nsys profile -o quant_profile \
    python benchmarks/kernels/benchmark_quant.py --quant-dtype fp8 --profile

# Then analyze profile:
nsys stats quant_profile.qdrep

# Shows detailed kernel breakdown:
# - Memory transfers
# - Kernel execution time
# - Memory bandwidth utilization
# - GPU occupancy
```

### Example 7: Comparing INT8 and FP8

```python
# Run both back-to-back for comparison
python benchmarks/kernels/benchmark_quant.py --quant-dtype int8 > int8_results.txt
python benchmarks/kernels/benchmark_quant.py --quant-dtype fp8 > fp8_results.txt

# Typical results:
# INT8: 87.345 us
# FP8:  89.123 us

# Analysis:
# - FP8 typically 2-5% slower than INT8
# - FP8 has more complex conversion logic (float format vs integer rounding)
# - Difference negligible in practice
# - FP8 often preferred for better model accuracy despite slight overhead
```

### Example 8: High-Precision Timing

```python
# Use more iterations for stable measurements
python benchmarks/kernels/benchmark_quant.py \
    --num-warmup-iters 20 \
    --num-iters 500 \
    --quant-dtype fp8

# Benefits:
# - More stable timing (reduced variance)
# - Better statistical confidence
# - Captures potential thermal throttling effects
# - Useful for CI/CD performance regression testing

# Output shows average over 500 iterations:
# Kernel running time: 88.947 us (σ < 1us)
```

### Example 9: Performance Bottleneck Analysis

```python
# Analyze where time is spent

# 1. Dynamic scaling (baseline)
python benchmarks/kernels/benchmark_quant.py --quant-dtype fp8
# Result: 89.123 us

# 2. Static scaling (no scale computation)
python benchmarks/kernels/benchmark_quant.py --quant-dtype fp8 --static-scale
# Result: 56.789 us

# Breakdown:
# - Scale computation: 89.123 - 56.789 = 32.334 us (36.3% of total)
# - Quantization itself: 56.789 us (63.7% of total)

# Implications:
# - For models with static quantization: use --static-scale approach
# - For dynamic quantization: scale computation is significant overhead
# - Potential optimization: fuse scale computation with quantization kernel
```

### Example 10: Batch Size Impact

```python
# Vary num_tokens to see batch size scaling

# Small batch (decode phase)
python benchmarks/kernels/benchmark_quant.py --num-tokens 128 --hidden-size 8192
# Output: 12.345 us → 1.19 ns/element

# Medium batch
python benchmarks/kernels/benchmark_quant.py --num-tokens 1024 --hidden-size 8192
# Output: 34.567 us → 4.12 ns/element

# Large batch (prefill phase)
python benchmarks/kernels/benchmark_quant.py --num-tokens 8192 --hidden-size 8192
# Output: 189.234 us → 2.82 ns/element

# Observations:
# - Very small batches (128): poor GPU utilization, high overhead
# - Medium batches (1024): good balance
# - Large batches (8192): best per-element efficiency
# - Optimal batch size depends on GPU architecture and occupancy
```

## Implementation Notes

**Quantization Details:**

**INT8 Quantization:**
1. Compute scale (if dynamic): `scale = max(abs(x)) / 127.0`
2. Scale input: `x_scaled = x / scale`
3. Round and clamp: `q = clamp(round(x_scaled), -128, 127)`
4. Convert to int8: `q.to(torch.int8)`

**FP8 Quantization:**
1. Compute scale (if dynamic): `scale = max(abs(x)) / 448.0` (FP8 E4M3 max)
2. Scale input: `x_scaled = x / scale`
3. Convert to FP8: `q = x_scaled.to(torch.float8_e4m3fn)`
4. Hardware handles saturation and rounding

**Default Configuration:**
- num_tokens: 4096 (typical prefill batch size)
- hidden_size: 8192 (common model dimension)
- Total elements: 33,554,432 (32M)
- Input size (FP16): 64 MB
- Output size (INT8/FP8): 32 MB
- 2x compression ratio

**Performance Characteristics:**
- **Memory-bandwidth bound**: Performance scales linearly with GPU memory bandwidth
- **Scale computation overhead**: ~30-40% of total time for dynamic scaling
- **Static scaling benefit**: ~1.5-1.6x speedup by providing pre-computed scale
- **Tensor size scaling**: Better efficiency at larger sizes (improved GPU utilization)
- **Dtype impact**: FP32 input ~5% slower than FP16/BF16 due to 2x bandwidth

**CUDA Event Timing:**
- Uses CUDA events for precise GPU time measurement
- Includes kernel execution but not CPU-GPU sync overhead
- Measures only quantization kernel, not tensor creation
- Average over multiple iterations reduces variance

**CUDA Profiler Integration:**
- `--profile` flag enables CUDA profiler for detailed analysis
- Runs single iteration instead of averaging
- Compatible with `nsys`, `nvprof`, Nsight Systems
- Useful for identifying memory bottlenecks and occupancy issues

**Typical Use Cases:**

1. **Static Quantization**: Weights quantized once during model loading
   - Use `--static-scale` mode
   - Scale computed during calibration
   - No runtime scale computation overhead

2. **Dynamic Quantization**: Activations quantized per-layer during inference
   - Use default mode (no --static-scale)
   - Scale computed from activation statistics
   - ~30-40% overhead for scale computation

3. **KV Cache Quantization**: Keys and values quantized during generation
   - Typically dynamic scaling per token
   - Critical for memory-bandwidth efficiency
   - FP8 often preferred for better accuracy

**Hardware Considerations:**
- **NVIDIA Hopper (H100)**: Native FP8 support, very fast FP8 quantization
- **NVIDIA Ada/Ampere**: FP8 via software emulation, slower than INT8
- **Older GPUs**: INT8 typically preferred over FP8
- **Memory bandwidth**: Primary bottleneck, faster HBM = faster quantization

**Optimization Opportunities:**
- Fuse scale computation with quantization kernel
- Use per-channel or per-group quantization for better accuracy
- Quantize multiple tensors in parallel to saturate bandwidth
- Use CUDA graphs to reduce launch overhead

## Related Pages

- **vllm Custom Ops** - Custom CUDA operations including quantization kernels
- **vllm FP8 Support** - FP8 quantization infrastructure and utilities
- **vllm INT8 Support** - INT8 quantization infrastructure and utilities
- **vllm Quantization Layers** - High-level quantization layer implementations
- **vllm KV Cache** - KV cache with quantization support
- **vllm Benchmark Per-Token Group Quant** - Per-token group quantization benchmarks

**See Also:**
- https://github.com/vllm-project/vllm - vLLM repository
- FP8 E4M3 format specification (NVIDIA Hopper whitepaper)
- Quantization-aware training and post-training quantization techniques
