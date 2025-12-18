{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::KV_Cache]], [[domain::Memory]], [[domain::Attention]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks reshape_and_cache kernel for storing key-value tensors into paged KV cache with optional FP8 conversion.

=== Description ===
This benchmark evaluates the reshape_and_cache kernel that stores computed key and value tensors into the paged KV cache structure used by vLLM. The kernel handles reshaping tensors from [num_tokens, num_heads, head_size] to the paged format, writing them to allocated cache blocks according to the slot_mapping, and optionally converting to FP8 format for reduced memory usage.

The benchmark tests performance across exponentially increasing token counts (2^1 to 2^16) with configurable parameters for number of heads, head size, block size, and cache dtype (auto or fp8). It supports both CUDA graph and non-graph execution modes, measuring latency in microseconds. The benchmark creates random slot mappings to simulate realistic cache access patterns and ensures proper cache initialization with random data to avoid cold-start effects.

=== Usage ===
Run this benchmark when evaluating KV cache storage performance, measuring FP8 conversion overhead, comparing CUDA graph benefits, or determining optimal block sizes and cache configurations for different model architectures.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_reshape_and_cache.py benchmarks/kernels/benchmark_reshape_and_cache.py]

=== Signature ===
<syntaxhighlight lang="python">
@torch.inference_mode()
def run_benchmark(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    num_iters: int,
    benchmark_mode: str,
    device: str = "cuda",
) -> float
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Default benchmark (auto dtype, CUDA graph mode)
python benchmarks/kernels/benchmark_reshape_and_cache.py

# FP8 KV cache with custom parameters
python benchmarks/kernels/benchmark_reshape_and_cache.py \
    --kv-cache-dtype fp8 \
    --num-heads 64 \
    --head-size 128 \
    --block-size 16

# No CUDA graph mode
python benchmarks/kernels/benchmark_reshape_and_cache.py \
    --mode no_graph \
    --iters 100

# Custom dtype and blocks
python benchmarks/kernels/benchmark_reshape_and_cache.py \
    --dtype half \
    --num-blocks 32768 \
    --block-size 32
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| num_heads || int || Number of attention heads (default: 128)
|-
| head_size || int || Head dimension (default: 128, choices: 64-256)
|-
| block_size || int || KV cache block size (default: 16, choices: 16, 32)
|-
| num_blocks || int || Total number of cache blocks (default: 128*128)
|-
| dtype || str || Tensor dtype: half, bfloat16, float
|-
| kv_cache_dtype || str || Cache dtype: auto or fp8
|-
| iters || int || Number of iterations (default: 200)
|-
| mode || str || Benchmark mode: cudagraph or no_graph
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| num_tokens || int || Number of tokens tested (powers of 2)
|-
| latency_us || float || Latency in microseconds
|-
| results_table || str || Formatted table with all results
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Create key/value tensors
key = torch.randn(num_tokens, num_heads, head_size,
                 dtype=dtype, device="cuda")
value = torch.randn_like(key)

# Create slot mapping (random slots)
num_slots = block_size * num_blocks
slot_mapping = torch.tensor(
    random.sample(range(num_slots), num_tokens),
    dtype=torch.long, device="cuda"
)

# Create KV caches
key_caches, value_caches = create_kv_caches_with_random(
    num_blocks, block_size, 1, num_heads, head_size,
    kv_cache_dtype, dtype, device="cuda"
)
key_cache, value_cache = key_caches[0], value_caches[0]

# Compute FP8 scaling factors
k_scale = (key.amax() / 64.0).to(torch.float32)
v_scale = (value.amax() / 64.0).to(torch.float32)

# Reshape and cache operation
ops.reshape_and_cache(
    key, value, key_cache, value_cache,
    slot_mapping, kv_cache_dtype,
    k_scale, v_scale
)

# CUDA graph mode
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    ops.reshape_and_cache(
        key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype,
        k_scale, v_scale
    )
g.replay()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_reshape_and_cache_flash]]
* [[Concept:KV_Cache]]
* [[Concept:Paged_Attention]]
* [[Concept:FP8_KV_Cache]]
