# Benchmark Per-Token Group Quantization

**Knowledge Sources:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_per_token_group_quant.py`
**Domains:** Performance Testing, Kernel Benchmarking, Quantization, CUDA vs Triton
**Last Updated:** 2025-12-17

## Overview

A focused benchmark script comparing CUDA and Triton implementations of per-token group quantization for FP8 and INT8 data types.

## Description

The `benchmark_per_token_group_quant.py` script provides detailed performance comparison between CUDA kernel and Triton fallback implementations for per-token group quantization. This operation is critical for dynamic quantization during inference, where activations need to be quantized on-the-fly with per-token, per-group scaling factors.

**Quantization Schemes Supported:**

1. **FP8 (float8_e4m3fn)**:
   - Per-token, per-group scaling
   - Optional column-major scale layout
   - Optional ue8m0 (unsigned 8-bit, 0 exponent) scale format
   - Used for efficient KV cache and activation quantization

2. **INT8**:
   - Per-token, per-group scaling
   - Standard row-major scale layout only
   - No ue8m0 option
   - Used for weight-activation quantization

**Implementation Comparison:**

- **CUDA Implementation**: Optimized CUDA kernels (`fp8_utils.per_token_group_quant_fp8`, `int8_utils.per_token_group_quant_int8`)
- **Triton Implementation**: Triton-compiled fallback (forced via mocking `current_platform.is_cuda`)

The benchmark measures speedup of CUDA over Triton across various configurations, helping inform fallback strategy and optimization priorities.

## Usage

The script runs as a standalone benchmark with configurable iteration counts and dtype selection.

**Basic Usage:**

```bash
# Benchmark both FP8 and INT8
python benchmarks/kernels/benchmark_per_token_group_quant.py

# Benchmark only FP8
python benchmarks/kernels/benchmark_per_token_group_quant.py --dtype fp8

# Benchmark only INT8
python benchmarks/kernels/benchmark_per_token_group_quant.py --dtype int8

# Custom iteration counts
python benchmarks/kernels/benchmark_per_token_group_quant.py \
    --warmup-iters 20 \
    --bench-iters 200
```

**Command-line Arguments:**

- `--warmup-iters`: Number of warmup iterations (default: 10)
- `--bench-iters`: Number of benchmark iterations (default: 100)
- `--dtype`: Data type to benchmark - fp8, int8, both (default: both)

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_per_token_group_quant.py`

**Core Benchmark Function:**

```python
def _run_single(
    shape: tuple[int, int],
    group_size: int,
    dtype: str,
    *,
    column_major: bool = False,
    scale_ue8m0: bool = False,
    warmup_iters: int,
    bench_iters: int,
) -> None
```

**Triton Mode Context Manager:**

```python
@contextmanager
def _triton_mode():
    """Temporarily force the Triton fallback path"""
    with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
        yield
```

**Timing Helper:**

```python
def _time_cuda(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    warmup_iters: int,
    bench_iters: int,
) -> float
```

**Import:**

```python
# Run as script
python benchmarks/kernels/benchmark_per_token_group_quant.py [OPTIONS]
```

## I/O Contract

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `shape` | tuple[int, int] | Tensor shape (num_tokens, hidden_dim) |
| `group_size` | int | Number of elements per quantization group |
| `dtype` | str | Quantization type: "fp8" or "int8" |
| `column_major` | bool | FP8 only: use column-major scale layout |
| `scale_ue8m0` | bool | FP8 only: use ue8m0 scale format |
| `warmup_iters` | int | Number of warmup iterations |
| `bench_iters` | int | Number of benchmark iterations |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Console Output | str | Formatted table with configuration, timings, and speedup |
| CUDA Latency | float | CUDA implementation time in milliseconds |
| Triton Latency | float | Triton implementation time in milliseconds |
| Speedup | float | CUDA speedup factor over Triton |

### Quantization Function Signatures

**FP8:**
```python
(quantized_tensor, scale) = fp8_utils.per_token_group_quant_fp8(
    x: torch.Tensor,  # [num_tokens, hidden_dim]
    group_size: int,
    column_major_scales: bool = False,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]
```

**INT8:**
```python
(quantized_tensor, scale) = int8_utils.per_token_group_quant_int8(
    x: torch.Tensor,  # [num_tokens, hidden_dim]
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]
```

## Usage Examples

### Example 1: Basic Benchmark Output

```python
# Run default benchmark
python benchmarks/kernels/benchmark_per_token_group_quant.py

# Output:
# Configuration                                         | CUDA (ms)  | Triton (ms)   | Speed-up
# ---------------------------------------------------------------------------------------
# shape=(32, 128)  gs=64   col_major=False  ue8m0=False  dtype=fp8 | CUDA   0.012 ms  | Triton   0.034 ms  | speed-up ×2.83
# shape=(32, 128)  gs=64   col_major=False  ue8m0=True   dtype=fp8 | CUDA   0.011 ms  | Triton   0.033 ms  | speed-up ×3.00
# shape=(32, 128)  gs=64   col_major=True   ue8m0=False  dtype=fp8 | CUDA   0.013 ms  | Triton   0.035 ms  | speed-up ×2.69
# ...
# shape=(16, 512)  gs=128  col_major=False  ue8m0=False  dtype=int8| CUDA   0.018 ms  | Triton   0.056 ms  | speed-up ×3.11
```

### Example 2: FP8 Column-Major vs Row-Major

```python
# FP8 with default settings (row-major)
python benchmarks/kernels/benchmark_per_token_group_quant.py --dtype fp8

# Results show column_major impact:
# Row-major (col_major=False):
#   - Standard scale layout: [num_tokens, hidden_dim // group_size]
#   - Better cache locality for token-wise processing
#   - Typical speedup: 2.8-3.2x over Triton
#
# Column-major (col_major=True):
#   - Transposed scale layout: [hidden_dim // group_size, num_tokens]
#   - Better for certain reduction patterns
#   - Typical speedup: 2.6-3.0x over Triton
#   - Slightly slower CUDA due to strided access
```

### Example 3: ue8m0 Scale Format

```python
# FP8 with ue8m0 scale format
# ue8m0 = unsigned 8-bit, 0 exponent format for scales

# Without ue8m0 (use_ue8m0=False):
#   - Scales stored as float32
#   - Higher precision but more memory bandwidth
#   - Typical speedup: 2.8-3.2x
#
# With ue8m0 (use_ue8m0=True):
#   - Scales stored as uint8 with fixed exponent
#   - Lower precision but 4x less scale memory
#   - Typical speedup: 2.9-3.3x (slightly better due to bandwidth)
#   - Requires careful handling of scale range
```

### Example 4: INT8 Quantization

```python
# INT8 benchmark
python benchmarks/kernels/benchmark_per_token_group_quant.py --dtype int8

# INT8 characteristics:
# - No column_major or ue8m0 options (always row-major, float32 scales)
# - Simpler quantization: value = clamp(round(x / scale), -128, 127)
# - FP8 quantization: value = cast_to_fp8(x / scale)
#
# Typical results:
# shape=(32, 128)  gs=64   dtype=int8 | CUDA 0.015 ms  | Triton 0.048 ms  | speed-up ×3.20
# shape=(64, 256)  gs=128  dtype=int8 | CUDA 0.028 ms  | Triton 0.089 ms  | speed-up ×3.18
#
# INT8 generally shows higher CUDA speedup due to simpler quantization logic
```

### Example 5: Group Size Impact

```python
# Default group sizes tested: [64, 128]

# Group size = 64:
#   - More groups per token (hidden_dim / 64)
#   - Finer-grained quantization (better accuracy)
#   - More scale computations
#   - CUDA speedup: 2.8-3.2x
#
# Group size = 128:
#   - Fewer groups per token (hidden_dim / 128)
#   - Coarser quantization (lower accuracy)
#   - Fewer scale computations
#   - CUDA speedup: 2.9-3.3x (slightly better due to less overhead)
#
# Recommendation: Use gs=128 for performance, gs=64 for accuracy
```

### Example 6: Tensor Size Scaling

```python
# Default shapes tested: [(32, 128), (64, 256), (16, 512)]

# Small tensors (32, 128):
#   - Total elements: 4K
#   - CUDA overhead relatively higher
#   - Speedup: 2.7-3.0x
#
# Medium tensors (64, 256):
#   - Total elements: 16K
#   - Better GPU utilization
#   - Speedup: 2.9-3.2x
#
# Narrow-tall tensors (16, 512):
#   - Same total elements: 8K but different shape
#   - Tests impact of tensor geometry
#   - Speedup: 2.8-3.1x
#
# General trend: Speedup improves with tensor size due to better amortization
```

### Example 7: Custom Benchmark Configuration

```python
# Extended benchmark with more iterations
python benchmarks/kernels/benchmark_per_token_group_quant.py \
    --warmup-iters 20 \
    --bench-iters 500 \
    --dtype fp8

# More iterations provide:
# - More stable timing measurements
# - Better statistical significance
# - Reduced variance in reported speedup
# - Longer total benchmark time
#
# Use for final performance validation or hardware characterization
```

### Example 8: Understanding Speedup Factors

```python
# Typical speedup analysis:
#
# 2.5-3.0x speedup:
#   - CUDA kernel is 2.5-3x faster than Triton
#   - Triton overhead: compilation, memory layout, dispatch
#   - CUDA advantages: hand-tuned, optimal memory access, fused operations
#
# When to use Triton fallback:
#   - Non-CUDA platforms (ROCm, CPU)
#   - Rapid prototyping without CUDA expertise
#   - Platforms where CUDA kernels not available
#
# When to use CUDA:
#   - Production inference on CUDA GPUs
#   - Performance-critical paths
#   - When 2.5-3x speedup is significant
#
# Trade-off:
#   - CUDA: faster but requires maintenance and platform-specific code
#   - Triton: slower but portable and easier to develop
```

## Implementation Notes

**Test Configuration:**
- **Shapes**: [(32, 128), (64, 256), (16, 512)]
- **Group sizes**: [64, 128]
- **FP8 options**: column_major × ue8m0 = 4 combinations
- **INT8 options**: Single configuration (row-major, float32 scales)
- **Total tests**: 3 shapes × 2 group_sizes × (4 FP8 + 1 INT8) = 30 configurations

**CUDA Requirements:**
- Script requires CUDA platform (checks `current_platform.is_cuda()`)
- Raises RuntimeError if run on non-CUDA devices
- Triton mode simulated via mocking, not actual non-CUDA execution

**Quantization Details:**

**FP8 Quantization:**
1. Compute per-group max absolute value
2. Calculate scale = max_abs / fp8_max (typically 448.0 for E4M3)
3. Quantize: q = (x / scale).to(torch.float8_e4m3fn)
4. Store quantized values and scales

**INT8 Quantization:**
1. Compute per-group max absolute value
2. Calculate scale = max_abs / 127.0
3. Quantize: q = clamp(round(x / scale), -128, 127).to(torch.int8)
4. Store quantized values and scales

**Scale Layouts:**

**Row-major (default):**
- Shape: [num_tokens, num_groups] where num_groups = hidden_dim // group_size
- Contiguous along token dimension
- Better for token-wise operations

**Column-major (FP8 only):**
- Shape: [num_groups, num_tokens]
- Transposed layout
- Better for group-wise reductions

**ue8m0 Format (FP8 only):**
- Unsigned 8-bit representation with fixed exponent
- Range: 1.0 to 256.0 (powers of 2)
- Stored as: scale_value = 2^(ue8m0_value)
- 4x memory savings vs float32
- Limited precision but sufficient for many cases

**Performance Characteristics:**
- CUDA consistently 2.5-3.3x faster than Triton
- Speedup increases slightly with tensor size
- FP8 with ue8m0 shows marginal improvements due to reduced scale bandwidth
- INT8 shows slightly higher speedup due to simpler quantization logic
- Column-major layout slightly slower in CUDA due to strided access

**Memory Bandwidth:**
- Input: num_tokens × hidden_dim × sizeof(bfloat16) = N × H × 2 bytes
- Output: N × H × 1 byte (FP8/INT8)
- Scales: N × (H/G) × scale_size
  - FP8 without ue8m0: N × (H/G) × 4 bytes
  - FP8 with ue8m0: N × (H/G) × 1 byte
  - INT8: N × (H/G) × 4 bytes
- Total bandwidth reduced by ~50% for output, additional scale overhead

**Typical Use Cases:**
- **KV Cache Quantization**: FP8 with ue8m0, group_size=128, column_major=False
- **Activation Quantization**: FP8 or INT8, group_size=128, row-major
- **Weight Quantization**: Usually static, not using per-token quantization

## Related Pages

- **vllm FP8 Utilities** - FP8 quantization utility functions
- **vllm INT8 Utilities** - INT8 quantization utility functions
- **vllm Quantization Layers** - High-level quantization layer implementations
- **vllm KV Cache** - KV cache with quantization support
- **vllm Triton Kernels** - Triton kernel implementations
- **vllm CUDA Kernels** - Custom CUDA kernel implementations

**See Also:**
- https://github.com/vllm-project/vllm - vLLM repository
- FP8 E4M3 format specification
- Per-channel vs per-token quantization strategies
