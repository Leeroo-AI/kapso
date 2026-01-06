{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::KV_Cache]], [[domain::FlashInfer]], [[domain::Attention]], [[domain::Memory_Layout]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks reshape_and_cache_flash kernel for FlashInfer KV cache storage comparing CUDA and Triton implementations with NHD and HND layouts.

=== Description ===
This benchmark evaluates the reshape_and_cache_flash kernel that stores key-value tensors into FlashInfer's KV cache format. FlashInfer uses a different cache layout optimized for its attention kernels, supporting both NHD (num_heads x head_size x depth) and HND (head_size x num_heads x depth) memory layouts. The benchmark compares CUDA kernel implementation against Triton fallback for both layouts.

The benchmark tests performance across exponentially increasing token counts (2^1 to 2^16) for both NHD and HND cache layouts. It supports optional FP8 cache dtype for reduced memory usage and measures both CUDA graph and non-graph execution modes. The Triton implementation currently only supports NHD layout, so HND benchmarks return NaN for Triton. Results help identify the fastest implementation and optimal layout for different model configurations.

=== Usage ===
Run this benchmark when evaluating FlashInfer KV cache storage, comparing CUDA vs Triton performance, determining optimal cache layouts (NHD vs HND), or measuring FP8 conversion overhead for FlashInfer attention.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_reshape_and_cache_flash.py benchmarks/kernels/benchmark_reshape_and_cache_flash.py]

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
    kv_cache_layout: str,
    num_iters: int,
    implementation: str,
    benchmark_mode: str,
    device: str = "cuda",
) -> float
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Default benchmark (CUDA implementation, both layouts)
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py

# Triton implementation
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --implementation triton

# FP8 KV cache with custom parameters
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --kv-cache-dtype fp8 \
    --num-heads 64 \
    --head-size 128

# No CUDA graph mode
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --mode no_graph \
    --iters 100

# Custom configuration
python benchmarks/kernels/benchmark_reshape_and_cache_flash.py \
    --dtype bfloat16 \
    --num-blocks 65536 \
    --block-size 16 \
    --implementation cuda
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
| num_blocks || int || Total number of cache blocks (default: 128*512)
|-
| dtype || str || Tensor dtype: half, bfloat16, float
|-
| kv_cache_dtype || str || Cache dtype: auto or fp8
|-
| iters || int || Number of iterations (default: 100)
|-
| implementation || str || Kernel implementation: cuda or triton
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
| layout || str || Cache layout: NHD or HND
|-
| latency_us || float || Latency in microseconds (or NaN if unsupported)
|-
| results_table || str || Formatted table with results for both layouts
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Create key/value tensors
key = torch.randn(num_tokens, num_heads, head_size,
                 dtype=dtype, device="cuda")
value = torch.randn_like(key)

# Create slot mapping
slot_mapping = torch.tensor(
    random.sample(range(num_slots), num_tokens),
    dtype=torch.long, device="cuda"
)

# Create FlashInfer KV caches with specific layout
key_caches, value_caches = create_kv_caches_with_random_flash(
    num_blocks, block_size, 1, num_heads, head_size,
    kv_cache_dtype, dtype, device="cuda",
    cache_layout=kv_cache_layout  # "NHD" or "HND"
)
key_cache, value_cache = key_caches[0], value_caches[0]

# Compute FP8 scaling factors
k_scale = (key.amax() / 64.0).to(torch.float32)
v_scale = (value.amax() / 64.0).to(torch.float32)

# CUDA implementation
ops.reshape_and_cache_flash(
    key, value, key_cache, value_cache,
    slot_mapping, kv_cache_dtype,
    k_scale, v_scale
)

# Triton implementation (NHD only)
triton_reshape_and_cache_flash(
    key, value, key_cache, value_cache,
    slot_mapping, kv_cache_dtype,
    k_scale, v_scale
)

# CUDA graph mode
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype,
        k_scale, v_scale
    )
g.replay()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_reshape_and_cache]]
* [[Concept:FlashInfer]]
* [[Concept:KV_Cache]]
* [[Concept:Memory_Layout]]
* [[Concept:NHD_vs_HND]]
