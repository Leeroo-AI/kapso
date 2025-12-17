---
knowledge_sources:
  - vllm-project/vllm repository
  - File: benchmarks/kernels/benchmark_trtllm_prefill_attention.py
domains:
  - Performance Benchmarking
  - Attention Mechanisms
  - TensorRT-LLM Integration
  - FP8 Inference
last_updated: 2025-12-17
---

# TensorRT-LLM Prefill Attention Benchmark

## Overview

Benchmarks FlashInfer's TensorRT-LLM compatible prefill attention implementation for evaluating prompt processing performance during the initial token generation phase.

## Description

The `benchmark_trtllm_prefill_attention.py` script provides comprehensive performance analysis of FlashInfer's prefill attention kernels with TensorRT-LLM compatibility. Prefill attention is critical during the prompt processing phase where the model computes attention for all input tokens simultaneously.

Key features:
- **Baseline**: Standard FlashInfer `BatchPrefillWithPagedKVCacheWrapper`
- **TRT-LLM**: FlashInfer's `trtllm_batch_context_with_kv_cache` with FP8/FP4 support
- **Quantization support**: FP8 for Q/K/V/O tensors and FP4 for output
- **Causal masking**: Supports causal attention for autoregressive models
- **Paged KV cache**: Memory-efficient block-based KV storage
- **CSV output**: Timestamped results with configuration and performance metrics

The benchmark sweeps across:
- Batch sizes: 1, 4, 8, 16, 32, 64, 128, 256
- Sequence lengths: 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
- Quantization modes: BF16 baseline, full FP8, FP8+FP4 output

## Usage

### Command Line Execution

```bash
# Run the full benchmark suite
python benchmarks/kernels/benchmark_trtllm_prefill_attention.py
```

### Configuration

The benchmark iterates through predefined configurations:

```python
batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
max_seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# Quantization configurations tested
quant_dtypes = [
    (None, None, None),              # BF16 baseline
    (FP8_DTYPE, FP8_DTYPE, None),     # FP8 Q+KV
    (FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),# Full FP8
    (FP8_DTYPE, FP8_DTYPE, FP4_DTYPE),# FP8 + FP4 output
]
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_trtllm_prefill_attention.py`

**Main Function:**

```python
@torch.no_grad()
def benchmark_prefill(
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

### Query Tensor Setup

For prefill, queries are variable length per sequence:

```python
# Query lengths sampled randomly
q_lens = torch.randint(1, max_q_len, (batch_size,), dtype=torch.int32)
q_indptr = torch.cumsum(torch.cat([torch.tensor([0]), q_lens]), dim=0)

# Query tensor shape: (sum(q_lens), num_qo_heads, head_size)
total_q_tokens = torch.sum(q_lens).item()
query = torch.randn(total_q_tokens, num_qo_heads, head_size, dtype=dtype)
```

### CSV Output Format

Results written to `flashinfer_trtllm_benchmark_YYYYMMDD_HHMMSS.csv`:

```csv
batch_size,trtllm_mean,trtllm_std,baseline_mean,baseline_std,speedup_percent,q_dtype,kv_cache_dtype,output_dtype,block_size,num_kv_heads,head_size,max_seq_len
1,1.234,0.045,1.345,0.056,0.082,torch.float8_e4m3fn,torch.float8_e4m3fn,torch.bfloat16,16,8,128,4096
```

## Usage Examples

### Running Single Configuration

```python
import torch
import flashinfer
from benchmark_trtllm_prefill_attention import benchmark_prefill

# Test full FP8 prefill
dtype = torch.bfloat16
FP8_DTYPE = torch.float8_e4m3fn
quant_dtypes = (FP8_DTYPE, FP8_DTYPE, None)

result = benchmark_prefill(
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

### Evaluating TTFT (Time To First Token)

```python
# TTFT is dominated by prefill latency for long prompts
prompt_lengths = [512, 1024, 2048, 4096, 8192]
ttft_results = []

for prompt_len in prompt_lengths:
    result = benchmark_prefill(
        dtype=torch.bfloat16,
        quant_dtypes=(FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),
        batch_size=1,  # Single user
        max_seq_len=prompt_len,
        trials=100
    )
    ttft_ms = result['trtllm_mean']
    ttft_results.append((prompt_len, ttft_ms))
    print(f"Prompt length {prompt_len}: TTFT = {ttft_ms:.2f}ms")

# Plot TTFT vs prompt length
import matplotlib.pyplot as plt
lengths, ttfts = zip(*ttft_results)
plt.plot(lengths, ttfts, marker='o')
plt.xlabel('Prompt Length (tokens)')
plt.ylabel('Time To First Token (ms)')
plt.title('TTFT vs Prompt Length')
plt.show()
```

### Batch Prefill Throughput

```python
# Measure throughput for batch prefill scenarios
batch_sizes = [1, 4, 8, 16, 32, 64]
seq_len = 2048
throughput_results = []

for bs in batch_sizes:
    result = benchmark_prefill(
        dtype=torch.bfloat16,
        quant_dtypes=(FP8_DTYPE, FP8_DTYPE, None),
        batch_size=bs,
        max_seq_len=seq_len,
        trials=50
    )

    # Calculate tokens per second
    total_tokens = bs * seq_len
    time_s = result['trtllm_mean'] / 1000
    throughput = total_tokens / time_s

    throughput_results.append((bs, throughput))
    print(f"Batch size {bs}: {throughput:.0f} tokens/s")
```

### Comparing Quantization Schemes

```python
quant_configs = {
    "BF16 baseline": (None, None, None),
    "FP8 Q+KV": (FP8_DTYPE, FP8_DTYPE, None),
    "Full FP8": (FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),
    "FP8 + FP4 output": (FP8_DTYPE, FP8_DTYPE, torch.uint8),
}

results = {}
for name, quant_dtypes in quant_configs.items():
    result = benchmark_prefill(
        dtype=torch.bfloat16,
        quant_dtypes=quant_dtypes,
        batch_size=16,
        max_seq_len=4096,
        trials=50
    )
    results[name] = result['trtllm_mean']
    print(f"{name}: {result['trtllm_mean']:.3f}ms")

# Calculate relative speedups
baseline = results["BF16 baseline"]
for name, time_ms in results.items():
    speedup = baseline / time_ms
    print(f"{name}: {speedup:.2f}x vs baseline")
```

### Memory-Throughput Analysis

```python
import torch

# Test different sequence lengths to find memory/compute bound transition
seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
batch_size = 32

for seq_len in seq_lengths:
    torch.cuda.reset_peak_memory_stats()

    result = benchmark_prefill(
        dtype=torch.bfloat16,
        quant_dtypes=(FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),
        batch_size=batch_size,
        max_seq_len=seq_len,
        trials=20
    )

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    time_ms = result['trtllm_mean']

    # Estimate FLOPs for attention: 2 * B * N^2 * D
    flops = 2 * batch_size * seq_len * seq_len * 128
    tflops = flops / (time_ms / 1000) / 1e12

    print(f"seq_len={seq_len}: {time_ms:.2f}ms, "
          f"{peak_mem_gb:.2f}GB, {tflops:.2f} TFLOPS")
```

### Testing GQA Configurations

```python
# Test different GQA (Grouped Query Attention) configurations
gqa_configs = [
    (64, 8),   # Standard GQA (8x groups)
    (64, 16),  # 4x groups
    (64, 32),  # 2x groups
    (64, 64),  # MHA (no grouping)
]

for num_qo_heads, num_kv_heads in gqa_configs:
    result = benchmark_prefill(
        dtype=torch.bfloat16,
        quant_dtypes=(FP8_DTYPE, FP8_DTYPE, None),
        batch_size=16,
        max_seq_len=4096,
        num_heads=(num_qo_heads, num_kv_heads),
        trials=50
    )

    ratio = num_qo_heads // num_kv_heads
    print(f"QO/KV heads: {num_qo_heads}/{num_kv_heads} ({ratio}x grouping): "
          f"{result['trtllm_mean']:.3f}ms")
```

## Related Pages

- [[vllm-project_vllm_flashinfer_prefill_attention]] - FlashInfer prefill implementation
- [[vllm-project_vllm_trtllm_decode_attention_benchmark]] - TRT-LLM decode benchmark
- [[vllm-project_vllm_paged_kv_cache]] - Paged KV cache implementation
- [[vllm-project_vllm_fp8_kv_cache]] - FP8 KV cache support
- [[vllm-project_vllm_attention_backends]] - Attention backend interface
- [[vllm-project_vllm_causal_attention]] - Causal attention masking
- [[vllm-project_vllm_gqa_mqa_attention]] - Grouped/Multi-query attention
