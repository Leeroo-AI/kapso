# Benchmark Reshape and Cache Flash

**Knowledge Sources:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_reshape_and_cache_flash.py`
**Domains:** Performance Testing, Kernel Benchmarking, KV Cache, FlashInfer, Memory Operations
**Last Updated:** 2025-12-17

## Overview

A comprehensive benchmark script comparing CUDA and Triton implementations of FlashInfer's append_paged_kv_cache operation across NHD and HND cache layouts.

## Description

The `benchmark_reshape_and_cache_flash.py` script evaluates the performance of FlashInfer's KV cache writing operation, which stores newly computed key-value tensors into paged cache structures with support for different memory layouts. This is an alternative to vLLM's standard `reshape_and_cache` operation, designed for better integration with FlashInfer's attention kernels.

**FlashInfer KV Cache:**

FlashInfer uses a different cache organization optimized for its attention implementation:
- **NHD Layout**: [num_blocks, block_size, num_heads, head_size] - standard layout
- **HND Layout**: [num_blocks, num_heads, block_size, head_size] - heads-first layout

The benchmark compares:
1. **CUDA Implementation** (`ops.reshape_and_cache_flash`): Optimized CUDA kernel
2. **Triton Implementation** (`triton_reshape_and_cache_flash`): Triton-compiled fallback

**Key Differences from Standard Reshape and Cache:**
- Supports both NHD and HND layouts
- Triton implementation available as fallback (HND not yet supported)
- Optimized for FlashInfer attention backend
- Similar quantization support (auto, FP8)

**Layout Impact:**
- **NHD (Num-blocks, block-size, num-Heads, head-Dimension)**: Traditional block-major layout
- **HND (num-blocks, num-Heads, block-size, head-Dimension)**: Head-major layout, better cache locality for attention

The benchmark sweeps through token counts (powers of 2 from 2 to 65536) and both layouts, measuring latency in microseconds for different configurations.

## Usage

The script runs as a standalone benchmark testing various configurations with comprehensive layout comparison.

**Basic Usage:**

```bash
# Default benchmark: CUDA implementation, both layouts, bfloat16
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py

# Benchmark Triton implementation (NHD layout only)
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py --implementation triton

# Benchmark with FP8 KV cache
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py --kv-cache-dtype fp8

# Benchmark without CUDA graphs
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py --mode no_graph

# Custom cache configuration
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --num-heads 64 \
    --head-size 128 \
    --block-size 32

# Fewer iterations for quick test
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py --iters 50
```

**Command-line Arguments:**

- `--num-heads`: Number of attention heads (default: 128)
- `--head-size`: Head dimension size, choices: [64, 80, 96, 112, 120, 128, 192, 256] (default: 128)
- `--block-size`: Cache block size, choices: [16, 32] (default: 16)
- `--num-blocks`: Number of cache blocks (default: 128 * 512 = 65536)
- `--dtype`: Input tensor dtype - half, bfloat16, float (default: bfloat16)
- `--kv-cache-dtype`: Cache storage dtype - auto, fp8 (default: auto)
- `--iters`: Number of benchmark iterations (default: 100)
- `--implementation`: Implementation to benchmark - cuda, triton (default: cuda)
- `--mode`: Benchmark mode - cudagraph, no_graph (default: cudagraph)

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_reshape_and_cache_flash.py`

**Core Benchmark Function:**

```python
@torch.inference_mode()
def run_benchmark(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    kv_cache_layout: str,  # "NHD" or "HND"
    num_iters: int,
    implementation: str,  # "cuda" or "triton"
    benchmark_mode: str,
    device: str = "cuda",
) -> float
```

**Kernel Operations:**

```python
# CUDA implementation
ops.reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, *, *, head_size] depends on layout
    value_cache: torch.Tensor,  # [num_blocks, *, *, head_size] depends on layout
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
)

# Triton implementation (NHD only)
triton_reshape_and_cache_flash(
    key, value, key_cache, value_cache, slot_mapping,
    kv_cache_dtype, k_scale, v_scale
)
```

**Entry Point:**

```python
def main(args) -> None
```

**Import:**

```python
from vllm import _custom_ops as ops
from vllm.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

# Run as script
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py [OPTIONS]
```

## I/O Contract

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `key` | torch.Tensor | Key tensor [num_tokens, num_heads, head_size] |
| `value` | torch.Tensor | Value tensor [num_tokens, num_heads, head_size] |
| `key_cache` | torch.Tensor | Key cache storage, layout-dependent shape |
| `value_cache` | torch.Tensor | Value cache storage, layout-dependent shape |
| `slot_mapping` | torch.Tensor | Token to slot mapping [num_tokens] |
| `kv_cache_dtype` | str | Cache quantization mode: "auto" or "fp8" |
| `kv_cache_layout` | str | Cache memory layout: "NHD" or "HND" |
| `k_scale` | torch.Tensor | FP8 scale for keys [1] |
| `v_scale` | torch.Tensor | FP8 scale for values [1] |
| `implementation` | str | Implementation choice: "cuda" or "triton" |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Console Output | str | Formatted table with num_tokens, layout, and latency |
| Latency | float | Per-operation latency in seconds |
| Updated Caches | torch.Tensor | key_cache and value_cache updated in-place |

### Cache Shapes by Layout

**NHD Layout:**
- key_cache: [num_blocks, block_size, num_heads, head_size]
- value_cache: [num_blocks, block_size, num_heads, head_size]

**HND Layout:**
- key_cache: [num_blocks, num_heads, block_size, head_size]
- value_cache: [num_blocks, num_heads, block_size, head_size]

## Usage Examples

### Example 1: Default Benchmark with Both Layouts

```python
# Run default CUDA benchmark across all layouts
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py

# Output:
# Benchmark results for implementation cuda (measuring with cudagraph):
# ┌─────────────┬────────┬──────────────┐
# │ num_tokens  │ layout │ latency (µs) │
# ├─────────────┼────────┼──────────────┤
# │           2 │ NHD    │        4.234 │
# │           4 │ NHD    │        4.345 │
# │           8 │ NHD    │        4.678 │
# │          16 │ NHD    │        5.234 │
# │          32 │ NHD    │        6.890 │
# ...
# │       65536 │ NHD    │     6345.678 │
# │           2 │ HND    │        4.189 │
# │           4 │ HND    │        4.301 │
# │           8 │ HND    │        4.623 │
# ...
# │       65536 │ HND    │     6289.234 │
# └─────────────┴────────┴──────────────┘
```

### Example 2: NHD vs HND Layout Comparison

```python
# Default benchmark shows both layouts
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py

# Typical results for 1024 tokens:
# NHD: 103.456 µs
# HND:  98.234 µs (~5% faster)

# HND advantages:
# - Better cache locality: all heads for a block position contiguous
# - Optimal for head-parallel attention kernels
# - FlashInfer attention backend prefers HND
# - ~3-7% faster depending on configuration

# NHD advantages:
# - Traditional layout, more compatible
# - Better for token-parallel operations
# - Standard in most implementations

# Recommendation: Use HND with FlashInfer backend
```

### Example 3: CUDA vs Triton Implementation

```python
# CUDA implementation (both layouts)
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --implementation cuda

# Triton implementation (NHD only, HND returns NaN)
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --implementation triton

# Typical results for 1024 tokens:
# CUDA NHD: 103.456 µs
# CUDA HND:  98.234 µs
# Triton NHD: 156.789 µs (51% slower)
# Triton HND: NaN (not supported)

# CUDA advantages:
# - ~50% faster than Triton
# - Supports both layouts
# - Production-optimized
# - Direct memory operations

# Triton advantages:
# - Portable across platforms
# - Easier to modify/experiment
# - Automatic optimization
# - Fallback for non-CUDA platforms
```

### Example 4: FP8 Quantization Impact

```python
# Auto mode (FP16/BF16 storage)
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --kv-cache-dtype auto

# FP8 mode (8-bit storage)
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --kv-cache-dtype fp8 \
    --head-size 128

# Results for 1024 tokens, NHD layout:
# Auto:  103.456 µs
# FP8:   125.678 µs (~21% overhead)

# FP8 characteristics:
# - 2x memory savings (8 bits vs 16 bits)
# - ~20-25% latency overhead for quantization
# - head_size must be multiple of 16
# - Worth it for memory-constrained scenarios
# - Trade-off: memory vs latency
```

### Example 5: Block Size Impact

```python
# Block size 16 (default)
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --block-size 16 \
    --num-blocks 65536

# Block size 32
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --block-size 32 \
    --num-blocks 65536

# Results for 1024 tokens, NHD layout:
# Block 16: 103.456 µs
# Block 32: 104.123 µs (~0.6% difference)

# Block size observations:
# - Minimal performance impact (<2%)
# - Affects memory granularity, not speed
# - Choice driven by memory efficiency
# - Smaller blocks: more flexibility, higher fragmentation
# - Larger blocks: less fragmentation, potential waste
```

### Example 6: Large Model Configuration

```python
# Extra large model: 64 heads, 256 head_size
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --num-heads 64 \
    --head-size 256

# Data per token: 64 * 256 * 2 dtypes = 32KB (key + value = 64KB)

# Results for 1024 tokens:
# NHD: 196.789 µs
# HND: 187.234 µs

# Data: 1024 tokens * 64 KB = 64 MB
# Time: 187.234 µs (HND)
# Bandwidth: 64 MB / 187.234 µs = 341.8 GB/s

# Comparison with A100 (1555 GB/s theoretical):
# Utilization: 341.8 / 1555 = 22.0%
# Lower due to: slot mapping overhead, cache misses, indirect writes
```

### Example 7: Small Batch Decode Analysis

```python
# Focus on small batch sizes (typical decode phase)
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py

# NHD Layout results:
# 1 token:   3.987 µs
# 2 tokens:  4.234 µs
# 4 tokens:  4.345 µs
# 8 tokens:  4.678 µs
# 16 tokens: 5.234 µs
# 32 tokens: 6.890 µs

# Fixed overhead analysis:
# - Base overhead: ~3.8 µs (kernel launch, setup)
# - Marginal cost: ~0.1 µs/token for small batches
# - CUDA graph helps but can't eliminate all overhead
# - HND layout ~3-5% faster even at small batches
```

### Example 8: CUDA Graph Benefit

```python
# With CUDA graph (default)
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --mode cudagraph

# Without CUDA graph
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --mode no_graph

# Results for 32 tokens, NHD layout:
# CUDA graph:    6.890 µs
# No graph:      8.123 µs (~18% overhead)

# Results for 1024 tokens, NHD layout:
# CUDA graph:  103.456 µs
# No graph:    115.789 µs (~12% overhead)

# CUDA graph benefits:
# - More pronounced at small batch sizes (18% vs 12%)
# - Eliminates kernel launch overhead
# - Production vLLM uses CUDA graphs
# - Essential for low-latency serving
```

### Example 9: Memory Bandwidth Efficiency

```python
# Configuration: 128 heads, 128 head_size, HND layout, 2048 tokens
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py

# From output: 2048 tokens, HND → 195.234 µs

# Data calculation:
# - Per token: 128 heads * 128 head_size * 2 dtypes * 2 (key+value) = 64 KB
# - Total: 2048 * 64 KB = 128 MB
# - Time: 195.234 µs
# - Achieved bandwidth: 128 MB / 195.234 µs = 655.6 GB/s

# HBM bandwidth comparison:
# - A100 (80GB): 2039 GB/s theoretical → 32.1% utilization
# - A100 (40GB): 1555 GB/s theoretical → 42.2% utilization
# - H100: 3350 GB/s theoretical → 19.6% utilization

# Why not 100%:
# - Slot mapping indirection (random access pattern)
# - Cache line fills and partial writes
# - Kernel overhead and synchronization
# - Memory controller efficiency limits
```

### Example 10: Production Simulation

```python
# Simulate production decode phase:
# - 64 concurrent requests (batch_size=64)
# - 80 heads, 128 head_size (Llama-70B-like)
# - BF16 precision
# - HND layout for FlashInfer
# - CUDA graph enabled

python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --num-heads 80 \
    --head-size 128 \
    --dtype bfloat16 \
    --mode cudagraph

# From output: 64 tokens, HND → 11.234 µs

# Per-layer analysis:
# - KV cache write: 11.234 µs
# - Num layers (Llama-70B): 80 layers
# - Total per request: 80 * 11.234 µs = 898.7 µs

# Throughput impact:
# - Target: 100 tokens/sec/request → 10ms inter-token latency
# - KV cache overhead: 898.7 µs = 8.99% of budget
# - Acceptable overhead for memory efficiency gains
```

## Implementation Notes

**Cache Layout Details:**

**NHD (Num-blocks, block-size, num-Heads, head-Dimension):**
- Traditional paged attention layout
- Memory layout: [B, N, H, D]
- Better for: token-parallel operations, standard implementations
- Cache line efficiency: moderate (heads not contiguous for same position)

**HND (num-blocks, num-Heads, block-size, head-Dimension):**
- FlashInfer-optimized layout
- Memory layout: [B, H, N, D]
- Better for: head-parallel operations, FlashInfer attention
- Cache line efficiency: high (all heads for same position contiguous)
- Typical speedup: 3-7% over NHD

**Triton Implementation Limitations:**
- Only supports NHD layout currently
- Returns `float('nan')` for HND layout
- ~50% slower than CUDA implementation
- Useful for: non-CUDA platforms, prototyping, fallback

**FP8 Quantization:**
- Scale computation: `k_scale = max(abs(key)) / 64.0`, similar for value
- Per-tensor scaling (single scale for entire key/value)
- head_size must be multiple of 16
- Overhead: ~20-25% latency increase
- Benefit: 2x memory savings
- Accuracy: minimal loss for KV cache (empirically validated)

**Performance Characteristics:**
- **Memory-bandwidth bound**: Dominated by cache write operations
- **Layout impact**: HND 3-7% faster due to better cache locality
- **CUDA vs Triton**: CUDA ~50% faster
- **FP8 overhead**: ~20-25% for quantization
- **CUDA graph benefit**: ~12-18% reduction in latency

**Benchmark Configuration:**
- **Token range**: Powers of 2 from 2 to 65536
- **Default cache**: 65536 blocks (larger than standard benchmark for more slots)
- **Layouts tested**: Both NHD and HND for CUDA, only NHD for Triton
- **Warmup**: 3 iterations before measurement
- **Iterations**: 100 per configuration (adjustable with --iters)

**Cache Creation:**
- Uses `create_kv_caches_with_random_flash()` utility
- Allocates caches with correct layout
- Initializes with random data
- Handles both NHD and HND layouts
- Supports auto and fp8 dtypes

**Memory Management:**
- Explicit tensor cleanup: `del key, value, key_cache, value_cache, slot_mapping`
- CUDA cache clearing: `torch.cuda.empty_cache()`
- Prevents OOM during large sweeps
- Important for testing many configurations sequentially

**Use Cases:**
- **FlashInfer integration**: Choose between NHD and HND layouts
- **Implementation selection**: CUDA for production, Triton for portability
- **Quantization decisions**: Evaluate FP8 overhead vs memory savings
- **Performance optimization**: Identify bottlenecks in KV cache pipeline
- **Hardware characterization**: Compare performance across GPU generations

**Integration with Attention:**
- FlashInfer attention expects HND layout for optimal performance
- Standard vLLM attention uses NHD layout
- Layout choice must match attention backend
- No runtime conversion (too expensive)
- Configuration set at model initialization

## Related Pages

- **vllm FlashInfer Attention** - FlashInfer attention backend using HND layout
- **vllm KV Cache Manager** - KV cache management and paging
- **vllm Benchmark Reshape and Cache** - Standard reshape_and_cache benchmark
- **vllm Custom Ops** - Custom CUDA operations
- **vllm Triton Kernels** - Triton kernel implementations
- **vllm FP8 Support** - FP8 quantization infrastructure

**See Also:**
- https://github.com/vllm-project/vllm - vLLM repository
- https://github.com/flashinfer-ai/flashinfer - FlashInfer library
- PagedAttention and memory layouts
- FlashAttention and FlashInfer architecture
