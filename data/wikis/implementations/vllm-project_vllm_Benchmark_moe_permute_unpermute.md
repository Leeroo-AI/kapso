{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::MoE]], [[domain::Token_Routing]], [[domain::Memory_Layout]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks MoE token permutation and unpermutation operations comparing standard and customized implementations for efficient expert routing.

=== Description ===
This benchmark evaluates the performance of token permutation and unpermutation operations required for MoE inference. These operations reorganize tokens so that tokens routed to the same expert are contiguous in memory, enabling efficient batched computation. The benchmark compares two implementations: the standard _moe_permute/_moe_unpermute_and_reduce and the customized moe_permute/moe_unpermute variants.

The benchmark supports FP16/BF16 unquantized, FP8 W8A8 quantized, and INT8 W8A16 quantized modes. For FP8, it includes 128-aligned block allocation required by DeepGEMM. It uses Ray for distributed benchmarking across multiple GPUs and measures both permute and unpermute operations separately with CUDA graphs (10 invocations per graph). Results help identify which implementation is faster for different batch sizes and model configurations.

=== Usage ===
Run this benchmark when optimizing MoE token routing performance, comparing permutation implementations, or evaluating the overhead of token reorganization for different batch sizes and expert configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe_permute_unpermute.py benchmarks/kernels/benchmark_moe_permute_unpermute.py]

=== Signature ===
<syntaxhighlight lang="python">
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
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Benchmark with default implementation
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1

# Benchmark customized implementation
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --use-customized-permute

# Specific batch size with FP8
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --dtype fp8_w8a8 \
    --batch-size 1024

# INT8 W8A16 quantization
python benchmarks/kernels/benchmark_moe_permute_unpermute.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dtype int8_w8a16 \
    --batch-size 512
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || str || Model name (Mixtral, DeepSeek, Qwen2-MoE, etc.)
|-
| dtype || str || Quantization mode: auto, fp8_w8a8, int8_w8a16
|-
| batch_size || int || Batch size to test (optional, defaults to sweep)
|-
| use_customized_permute || bool || Use customized permute/unpermute implementation
|-
| seed || int || Random seed (default: 0)
|-
| trust_remote_code || bool || Trust remote code for model loading
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Number of tokens tested
|-
| permute_time || float || Permute operation time in microseconds
|-
| unpermute_time || float || Unpermute operation time in microseconds
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Standard permutation
def _moe_permute(hidden_states, a1q_scale, topk_ids, n_expert,
                expert_map, align_block_size):
    permuted_hidden_states, a1q_scale, sorted_token_ids, \
        expert_ids, inv_perm = _moe_permute(
            hidden_states, a1q_scale, topk_ids, n_expert,
            expert_map, align_block_size
        )
    return results

# Customized permutation with alignment
def moe_permute(hidden_states, a1q_scale, topk_ids, n_expert,
               expert_map, align_block_size):
    permuted_hidden_states, a1q_scale, first_token_off, \
        inv_perm_idx, m_indices = moe_permute(
            hidden_states, a1q_scale, topk_ids, n_expert,
            expert_map, align_block_size
        )
    return results

# Benchmark structure
def benchmark_permute(num_tokens, num_experts, hidden_size,
                     topk, dtype, use_fp8_w8a8, use_int8_w8a16,
                     num_iters=100, use_customized_permute=False):
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype)

    if use_fp8_w8a8:
        align_block_size = 128  # DeepGEMM requirement
        qhidden_states, scale = _fp8_quantize(hidden_states, None, None)
    else:
        align_block_size = None
        qhidden_states = hidden_states

    # Capture in CUDA graph with 10 invocations
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            if use_customized_permute:
                moe_permute(qhidden_states, None, topk_ids,
                           num_experts, None, align_block_size)
            else:
                _moe_permute(qhidden_states, None, topk_ids,
                            num_experts, None, align_block_size)

    return avg_latency_us
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_moe]]
* [[Concept:Mixture_of_Experts]]
* [[Concept:Token_Routing]]
* [[Concept:Memory_Layout_Optimization]]
