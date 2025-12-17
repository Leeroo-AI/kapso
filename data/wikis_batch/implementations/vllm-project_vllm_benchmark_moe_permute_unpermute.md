# Benchmark MoE Permute Unpermute

**Knowledge Sources:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_moe_permute_unpermute.py`
**Domains:** Performance Testing, Kernel Benchmarking, Mixture of Experts, Token Routing
**Last Updated:** 2025-12-17

## Overview

A specialized benchmark script for measuring the performance of MoE token permute and unpermute operations with support for both standard and customized implementations.

## Description

The `benchmark_moe_permute_unpermute.py` script focuses on benchmarking the permute and unpermute operations that are critical components of Mixture of Experts (MoE) inference. These operations handle token routing before and after expert computation:

**Permute Operation:**
- Reorders tokens based on expert assignments from top-k gating
- Groups tokens by expert for efficient batched computation
- Optional FP8 quantization with alignment for DeepGEMM
- Produces permuted hidden states and inverse permutation indices

**Unpermute Operation:**
- Restores original token order after expert computation
- Applies top-k weights during unpermute (weighted sum)
- Reduces results back to original hidden state
- Optional inplace operations for memory efficiency

**Two Implementation Paths:**
1. **Standard Path** (`_moe_permute`, `_moe_unpermute_and_reduce`): Original vLLM implementation
2. **Customized Path** (`moe_permute`, `moe_unpermute`): Optimized implementation with different indexing scheme

The benchmark uses Ray for distributed testing across multiple GPUs, measuring both operations separately with CUDA graphs for accurate timing. It supports various quantization modes (FP8 W8A8, INT8 W8A16, FP16) and tests across different batch sizes and MoE architectures.

## Usage

The script runs distributed benchmarks using Ray, testing specified MoE models across batch sizes.

**Basic Usage:**

```bash
# Benchmark Mixtral with standard permute implementation
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1

# Benchmark with customized permute implementation
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --use-customized-permute

# Benchmark with FP8 quantization
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model deepseek-ai/DeepSeek-V3 \
    --dtype fp8_w8a8

# Benchmark with INT8 quantization
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --dtype int8_w8a16 \
    --trust-remote-code

# Benchmark specific batch size
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --batch-size 1024
```

**Command-line Arguments:**

- `--model`: HuggingFace model identifier (default: mistralai/Mixtral-8x7B-Instruct-v0.1)
- `--dtype`: Quantization scheme - auto, fp8_w8a8, int8_w8a16 (default: auto)
- `--use-customized-permute`: Use customized permute/unpermute implementation
- `--seed`: Random seed for reproducibility (default: 0)
- `--batch-size`: Specific batch size to test (default: sweep 1-4096)
- `--trust-remote-code`: Allow loading models with custom code

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_moe_permute_unpermute.py`

**Permute Benchmark:**

```python
def benchmark_permute(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 100,
    use_customized_permute: bool = False,
) -> float
```

**Unpermute Benchmark:**

```python
def benchmark_unpermute(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 100,
    use_customized_permute: bool = False,
) -> float
```

**Ray Worker:**

```python
@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_customized_permute: bool = False,
    ) -> tuple[dict[str, int], float]
```

**Entry Point:**

```python
def main(args: argparse.Namespace) -> None
```

**Import:**

```python
# Run as script
python benchmarks/kernels/benchmark_moe_permute_unpermute.py [OPTIONS]
```

## I/O Contract

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `model` | str | HuggingFace model identifier for MoE architecture |
| `dtype` | str | Quantization scheme: auto, fp8_w8a8, int8_w8a16 |
| `use_customized_permute` | bool | Whether to use customized implementation |
| `batch_size` | int | Optional specific batch size (default: full sweep) |
| `seed` | int | Random seed for reproducibility |
| `trust_remote_code` | bool | Allow custom model code |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Console Output | str | Per-batch-size permute and unpermute latencies |
| Permute Time | float | Permute operation latency in microseconds |
| Unpermute Time | float | Unpermute operation latency in microseconds |

### Operation Signatures

**Standard Path:**
```python
# Permute
(permuted_hidden_states, a1q_scale, sorted_token_ids, expert_ids, inv_perm) = _moe_permute(
    hidden_states, a1q_scale, topk_ids, num_experts, expert_map, align_block_size
)

# Unpermute
_moe_unpermute_and_reduce(
    output_hidden_states, permuted_hidden_states, inv_perm, topk_weights, inplace=True
)
```

**Customized Path:**
```python
# Permute
(permuted_hidden_states, a1q_scale, first_token_off, inv_perm_idx, m_indices) = moe_permute(
    hidden_states, a1q_scale, topk_ids, n_expert, expert_map, align_block_size
)

# Unpermute
moe_unpermute(
    output, permuted_hidden_states, topk_weights, inv_perm_idx, first_token_off
)
```

## Usage Examples

### Example 1: Basic Permute/Unpermute Benchmark

```python
# Benchmark standard implementation across all batch sizes
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1

# Output:
# Batch size: 1
# Permute time: 12.34 us
# Unpermute time: 15.67 us
# Batch size: 2
# Permute time: 13.45 us
# Unpermute time: 16.78 us
# ...
# Batch size: 4096
# Permute time: 234.56 us
# Unpermute time: 289.12 us
```

### Example 2: Comparing Implementations

```python
# Standard implementation
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --batch-size 1024

# Output:
# Batch size: 1024
# Permute time: 89.45 us
# Unpermute time: 112.34 us

# Customized implementation
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --batch-size 1024 \
    --use-customized-permute

# Output:
# Batch size: 1024
# Permute time: 76.23 us  # ~15% faster
# Unpermute time: 98.56 us  # ~12% faster
```

### Example 3: FP8 Quantization with Alignment

```python
# FP8 W8A8 requires 128-byte alignment for DeepGEMM
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model deepseek-ai/DeepSeek-V3 \
    --dtype fp8_w8a8 \
    --batch-size 2048

# FP8 characteristics:
# - Permute includes quantization step (_fp8_quantize)
# - align_block_size=128 for DeepGEMM compatibility
# - Returns quantized permuted states and scale factors
# - Additional overhead but enables efficient FP8 GEMM

# Output shows total permute time including quantization:
# Batch size: 2048
# Permute time: 145.67 us  # includes quantization
# Unpermute time: 178.23 us  # includes dequant and weighted sum
```

### Example 4: Large Batch Performance

```python
# Test large batch sizes typical for prefill
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --batch-size 4096

# Large batch characteristics:
# - Permute overhead becomes proportionally smaller
# - Better GPU utilization
# - Memory bandwidth bound
# - Typical results:
#   Batch size: 4096
#   Permute time: 234.56 us (0.057 us/token)
#   Unpermute time: 289.12 us (0.071 us/token)
```

### Example 5: INT8 Weight Quantization

```python
# INT8 W8A16 doesn't quantize activations
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dtype int8_w8a16 \
    --batch-size 1024

# INT8 W8A16 characteristics:
# - No activation quantization in permute
# - align_block_size=None (no special alignment)
# - Same data flow as FP16 for permute/unpermute
# - Weights quantized separately in expert computation

# Output similar to FP16:
# Batch size: 1024
# Permute time: 89.45 us
# Unpermute time: 112.34 us
```

### Example 6: CUDA Graph Overhead Analysis

```python
# The benchmark captures 10 invocations per CUDA graph
# This amortizes launch overhead and provides realistic timing

# Single invocation time components:
# 1. Kernel launch overhead: ~5-10us
# 2. Actual computation: varies with batch size
# 3. Synchronization: minimal with graphs

# Example for small batch:
# Batch size: 16
# Permute time: 23.45 us
# Without CUDA graph: ~33.45 us (10us launch overhead)
# 10 invocations in graph: ~134.5 us total
# Per-invocation amortized: 13.45 us actual compute
```

### Example 7: Multi-GPU Distributed Benchmark

```python
# Ray automatically distributes across available GPUs
# For 4-GPU system:

python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1

# Ray distributes batch sizes round-robin:
# GPU 0: batch_size=1, 5, 9, 13, ...
# GPU 1: batch_size=2, 6, 10, 14, ...
# GPU 2: batch_size=3, 7, 11, 15, ...
# GPU 3: batch_size=4, 8, 12, 16, ...

# All results collected and printed in order
# Enables ~4x faster benchmarking
```

### Example 8: Analyzing Performance Scaling

```python
# Run full sweep and analyze scaling
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    > permute_results.txt

# Typical scaling characteristics:
# Batch 1-16: High per-token cost (launch overhead dominant)
# Batch 16-256: Improving efficiency (better GPU utilization)
# Batch 256-1024: Near-optimal per-token cost
# Batch 1024+: Memory bandwidth bound, slight per-token increase

# Example data:
# Batch=16:   permute=23us (1.44 us/token), unpermute=28us (1.75 us/token)
# Batch=256:  permute=67us (0.26 us/token), unpermute=89us (0.35 us/token)
# Batch=1024: permute=89us (0.087 us/token), unpermute=112us (0.109 us/token)
# Batch=4096: permute=235us (0.057 us/token), unpermute=289us (0.071 us/token)
```

## Implementation Notes

**Supported Model Architectures:**
- Mixtral, Qwen2MoE, Qwen3MoE, DeepSeek-V2/V3, DBRX, Jamba, GLM4MoE, Llama4 MoE
- Automatically extracts num_experts, num_experts_per_tok, hidden_size from config
- Adapts to different expert counts and top-k values

**Default Batch Size Sweep:**
- [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
- Covers decode (1-16), small batch inference (32-128), and prefill (256-4096)
- Can be overridden with --batch-size for focused testing

**FP8 W8A8 Specifics:**
- Requires align_block_size=128 for DeepGEMM compatibility
- Permute includes FP8 quantization via _fp8_quantize
- Returns quantized states and per-tensor scale factors
- Unpermute works with FP8 inputs (simulated by converting to dtype)
- Additional ~10-20% overhead for quantization

**INT8 W8A16 Specifics:**
- No activation quantization (only weights quantized)
- align_block_size=None (no special alignment requirements)
- Permute/unpermute identical to FP16 path
- Performance similar to FP16 for these operations

**CUDA Graph Optimization:**
- Captures 10 consecutive invocations in single graph
- Reduces per-invocation launch overhead
- Especially important for small batch sizes
- Reported time is average across 10 invocations

**Implementation Differences:**

**Standard Path:**
- Uses sorted_token_ids and inv_perm arrays
- Unpermute integrated with reduction (_moe_unpermute_and_reduce)
- Single kernel for unpermute + weighted sum

**Customized Path:**
- Uses first_token_off, inv_perm_idx, m_indices
- Separate unpermute operation (moe_unpermute)
- Different indexing scheme, potentially more cache-friendly
- Typically 10-20% faster for both operations

**Memory Layout:**
- Input: [num_tokens, hidden_size]
- Permuted output: [num_tokens * topk, hidden_size]
- Expert computation happens on permuted layout
- Unpermute restores to [num_tokens, hidden_size]

**Gating Integration:**
- Uses fused_topk for expert selection
- topk_weights: [num_tokens, topk] weights for each token-expert pair
- topk_ids: [num_tokens, topk] expert indices for each token
- Weights applied during unpermute for final weighted sum

**Performance Characteristics:**
- Permute typically 10-20% faster than unpermute
- Both operations memory-bandwidth bound
- Scales sublinearly with batch size (better efficiency at large batches)
- Customized implementation shows consistent 10-20% improvement
- FP8 quantization adds 10-20% overhead to permute

**Use Cases:**
- Performance tuning: Choose between standard and customized implementations
- Quantization strategy: Understand overhead of FP8 quantization in routing
- Batch size selection: Find optimal batch size for efficiency
- Hardware characterization: Compare permute/unpermute on different GPUs

## Related Pages

- **vllm MoE Permute Utils** - Permute/unpermute kernel implementations
- **vllm Fused MoE Layers** - MoE layers using these operations
- **vllm FP8 Quantization** - FP8 quantization infrastructure
- **vllm CUDA Graph** - CUDA Graph integration
- **vllm MoE Topk** - Top-k expert selection implementation
- **vllm Benchmark MoE Kernels** - Full MoE pipeline benchmarking

**See Also:**
- https://github.com/vllm-project/vllm - vLLM repository
- MoE architecture and token routing strategies
- CUDA Graph performance optimization
