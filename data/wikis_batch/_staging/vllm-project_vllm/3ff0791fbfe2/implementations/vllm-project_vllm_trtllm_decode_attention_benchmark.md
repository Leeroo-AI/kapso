---
knowledge_sources:
  - vllm-project/vllm repository
  - File: benchmarks/kernels/benchmark_trtllm_decode_attention.py
domains:
  - Performance Benchmarking
  - Attention Mechanisms
  - TensorRT-LLM Integration
  - FP8 Inference
last_updated: 2025-12-17
---

# TensorRT-LLM Decode Attention Benchmark

## Overview

Benchmarks FlashInfer's TensorRT-LLM compatible decode attention implementation against baseline FlashInfer for generation phase performance evaluation.

## Description

The `benchmark_trtllm_decode_attention.py` script provides comprehensive performance analysis of FlashInfer's decode attention kernels with TensorRT-LLM compatibility. Decode attention is critical during the generation phase where the model processes one token at a time to generate the next token.

Key features:
- **Baseline**: Standard FlashInfer `BatchDecodeWithPagedKVCacheWrapper`
- **TRT-LLM**: FlashInfer's `trtllm_batch_decode_with_kv_cache` with FP8/FP4 support
- **Quantization support**: FP8 for Q/K/V/O tensors and FP4 for output
- **Paged KV cache**: Memory-efficient block-based KV storage
- **CSV output**: Timestamped results with configuration and performance metrics

The benchmark sweeps across:
- Batch sizes: 1, 4, 8, 16, 32, 64, 128, 256
- Sequence lengths: 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
- Quantization modes: BF16 baseline, FP8 KV, full FP8, FP8+FP4 output

## Usage

### Command Line Execution

```bash
# Run the full benchmark suite
python benchmarks/kernels/benchmark_trtllm_decode_attention.py
```

### Configuration

The benchmark iterates through predefined configurations:

```python
batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
max_seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# Quantization configurations tested
quant_dtypes = [
    (None, None, None),              # BF16 baseline
    (None, FP8_DTYPE, None),          # FP8 KV cache only
    (FP8_DTYPE, FP8_DTYPE, None),     # FP8 Q+KV
    (FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),# Full FP8
    (FP8_DTYPE, FP8_DTYPE, FP4_DTYPE),# FP8 + FP4 output
]
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_trtllm_decode_attention.py`

**Main Function:**

```python
@torch.no_grad()
def benchmark_decode(
    dtype: torch.dtype,
    quant_dtypes: tuple[torch.dtype | None, torch.dtype | None, torch.dtype | None],
    batch_size: int,
    max_seq_len: int,
    num_heads: tuple[int, int] = (64, 8),
    head_size: int = 128,
    kv_layout: str = "HND",
    block_size: int = 16,
    warmup: int = 10,
    trials: int = 20
)
```

**Quantization Helper:**

```python
def to_float8(x, dtype=torch.float8_e4m3fn):
    """Convert tensor to FP8 with scale factor"""
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()
```

**Import:**
```python
import flashinfer
from vllm.utils.math_utils import round_up
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `torch.dtype` | Base dtype for computation (bfloat16) |
| `quant_dtypes` | `tuple[dtype, dtype, dtype]` | Q, KV, O quantization dtypes |
| `batch_size` | `int` | Number of sequences in batch |
| `max_seq_len` | `int` | Maximum sequence length |
| `num_heads` | `tuple[int, int]` | (num_qo_heads, num_kv_heads) for GQA |
| `head_size` | `int` | Dimension of each attention head (default: 128) |
| `kv_layout` | `str` | KV cache layout: "HND" or "NHD" (default: "HND") |
| `block_size` | `int` | Size of paged KV cache blocks (default: 16) |
| `warmup` | `int` | Number of warmup iterations (default: 10) |
| `trials` | `int` | Number of timing trials (default: 20) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `batch_size` | `int` | Tested batch size |
| `trtllm_mean` | `float` | Mean TRT-LLM execution time (ms) |
| `trtllm_std` | `float` | Standard deviation for TRT-LLM (ms) |
| `baseline_mean` | `float` | Mean baseline execution time (ms) |
| `baseline_std` | `float` | Standard deviation for baseline (ms) |
| `speedup_percent` | `float` | Performance improvement ratio |
| `q_dtype` | `str` | Query tensor dtype |
| `kv_cache_dtype` | `str` | KV cache tensor dtype |
| `output_dtype` | `str` | Output tensor dtype |
| `block_size` | `int` | KV cache block size |
| `num_kv_heads` | `int` | Number of KV heads |
| `head_size` | `int` | Head dimension |
| `max_seq_len` | `int` | Maximum sequence length |

### CSV Output Format

Results written to `flashinfer_trtllm_benchmark_YYYYMMDD_HHMMSS.csv`:

```csv
batch_size,trtllm_mean,trtllm_std,baseline_mean,baseline_std,speedup_percent,q_dtype,kv_cache_dtype,output_dtype,block_size,num_kv_heads,head_size,max_seq_len
1,0.123,0.005,0.145,0.006,0.152,torch.bfloat16,torch.bfloat16,torch.bfloat16,16,8,128,1024
```

## Usage Examples

### Running Single Configuration

```python
import torch
import flashinfer
from benchmark_trtllm_decode_attention import benchmark_decode

# Test FP8 KV cache with BF16 baseline
dtype = torch.bfloat16
FP8_DTYPE = torch.float8_e4m3fn
quant_dtypes = (None, FP8_DTYPE, None)  # FP8 KV cache only

result = benchmark_decode(
    dtype=dtype,
    quant_dtypes=quant_dtypes,
    batch_size=32,
    max_seq_len=4096,
    num_heads=(64, 8),
    head_size=128,
    kv_layout="HND",
    block_size=16,
    warmup=10,
    trials=20
)

print(f"TRT-LLM: {result['trtllm_mean']:.3f}ms")
print(f"Baseline: {result['baseline_mean']:.3f}ms")
print(f"Speedup: {result['speedup_percent']:.1%}")
```

### Custom Quantization Testing

```python
# Test full FP8 pipeline
quant_dtypes = (FP8_DTYPE, FP8_DTYPE, FP8_DTYPE)

# Run benchmark
result = benchmark_decode(
    dtype=torch.bfloat16,
    quant_dtypes=quant_dtypes,
    batch_size=64,
    max_seq_len=8192,
    warmup=20,
    trials=50
)
```

### Analyzing Results

```python
import pandas as pd

# Load benchmark results
df = pd.read_csv("flashinfer_trtllm_benchmark_20241217_143022.csv")

# Filter for specific configuration
fp8_results = df[df['kv_cache_dtype'] == 'torch.float8_e4m3fn']

# Compute statistics
print(f"Average speedup: {fp8_results['speedup_percent'].mean():.1%}")
print(f"Best speedup: {fp8_results['speedup_percent'].max():.1%}")

# Plot speedup vs batch size
import matplotlib.pyplot as plt
for seq_len in [1024, 4096, 16384]:
    subset = fp8_results[fp8_results['max_seq_len'] == seq_len]
    plt.plot(subset['batch_size'], subset['speedup_percent'],
             marker='o', label=f'seq_len={seq_len}')
plt.xlabel('Batch Size')
plt.ylabel('Speedup (%)')
plt.legend()
plt.show()
```

### Testing Different Block Sizes

```python
# Compare different block sizes for paged KV cache
block_sizes = [8, 16, 32, 64]
results = []

for block_size in block_sizes:
    result = benchmark_decode(
        dtype=torch.bfloat16,
        quant_dtypes=(FP8_DTYPE, FP8_DTYPE, None),
        batch_size=32,
        max_seq_len=8192,
        block_size=block_size,
        trials=50
    )
    results.append((block_size, result['trtllm_mean']))
    print(f"Block size {block_size}: {result['trtllm_mean']:.3f}ms")
```

### Memory Usage Analysis

```python
import torch

# Setup for memory tracking
torch.cuda.reset_peak_memory_stats()

# Run decode benchmark
result = benchmark_decode(
    dtype=torch.bfloat16,
    quant_dtypes=(FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),
    batch_size=128,
    max_seq_len=32768,
    block_size=16
)

# Report memory usage
peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
print(f"Peak memory: {peak_mem:.2f} GB")
print(f"Latency: {result['trtllm_mean']:.3f}ms")
```

## Related Pages

- [[vllm-project_vllm_flashinfer_decode_attention]] - FlashInfer decode implementation
- [[vllm-project_vllm_trtllm_prefill_attention_benchmark]] - TRT-LLM prefill benchmark
- [[vllm-project_vllm_paged_kv_cache]] - Paged KV cache implementation
- [[vllm-project_vllm_fp8_kv_cache]] - FP8 KV cache support
- [[vllm-project_vllm_attention_backends]] - Attention backend interface
- [[vllm-project_vllm_gqa_mqa_attention]] - Grouped/Multi-query attention
