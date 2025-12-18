{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Model Configuration]], [[domain::Tensor Parallelism]], [[domain::Weight Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A configuration module defining weight matrix shapes for popular LLM architectures with tensor parallelism awareness.

=== Description ===
This module provides a centralized registry of weight shapes (GEMM dimensions) for major LLM model families including Llama, Mistral, Qwen, DeepSeek, and others. Each shape is specified as ([K, N], TP_SPLIT_DIM) where K and N are the original dimensions and TP_SPLIT_DIM indicates which dimension is sharded during tensor parallelism (0 for K-sharding, 1 for N-sharding). This enables accurate benchmarking and kernel tuning for specific model architectures across different TP configurations. The shapes cover QKV projections, MLP layers, and MoE configurations.

=== Usage ===
Use this module when benchmarking kernels for specific model architectures, generating auto-tuned kernel configurations for TP deployments, or understanding how weight shapes change with tensor parallelism.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/weight_shapes.py benchmarks/kernels/weight_shapes.py]

=== Signature ===
<syntaxhighlight lang="python">
# Weight shapes are tuples: ([K, N], TP_SPLIT_DIM)
# TP_SPLIT_DIM: 0 = K dimension is sharded, 1 = N dimension is sharded

WEIGHT_SHAPES: dict[str, list[tuple[list[int], int]]]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES

# Get shapes for specific model
llama3_shapes = WEIGHT_SHAPES["meta-llama/Llama-3-8b"]
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model_name || str || HuggingFace model identifier
|-
| tp_size || int || Tensor parallel size (e.g., 1, 2, 4, 8)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| weight_shapes || list[tuple] || List of ([K, N], split_dim) for model layers
|-
| effective_K || int || K dimension after TP sharding
|-
| effective_N || int || N dimension after TP sharding
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from benchmarks.kernels.weight_shapes import WEIGHT_SHAPES

# Get Llama-3-8B weight shapes
shapes = WEIGHT_SHAPES["meta-llama/Llama-3-8b"]
# Returns: [([4096, 6144], 1), ([4096, 4096], 0),
#           ([4096, 28672], 1), ([14336, 4096], 0)]

# Calculate effective shapes for TP=4
tp_size = 4
for shape, split_dim in shapes:
    K, N = shape
    if split_dim == 0:
        effective_K = K // tp_size
        effective_N = N
    else:
        effective_K = K
        effective_N = N // tp_size
    print(f"Original: K={K}, N={N}, Split dim={split_dim}")
    print(f"TP={tp_size}: K={effective_K}, N={effective_N}")

# Example: Llama-3-8B QKV projection with TP=4
# Original: [4096, 6144], split_dim=1 (N-sharded)
# TP=4: K=4096, N=1536 (each rank gets 1536 output dims)

# DeepSeek-Coder-V2 with MoE layers
deepseek_shapes = WEIGHT_SHAPES["deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"]
print(f"DeepSeek shapes: {len(deepseek_shapes)} unique weight matrices")
</syntaxhighlight>

== Supported Models ==

{| class="wikitable"
|-
! Model Family !! Models !! Key Characteristics
|-
| Llama || 2-7b, 2-13b, 2-70b, 3-8b, 3.1-8B, 3.1-405b, 3.3-70B || Standard dense transformers
|-
| Mistral || 7B-v0.1, Large-2407 || Dense with sliding window
|-
| Qwen || 2.5-7B, 2.5-32B, 2.5-72B || Dense with grouped query attention
|-
| DeepSeek || Coder-V2-Lite || MoE architecture with expert routing
|-
| Cohere || c4ai-command-a-03-2025 || Command-R series
|}

== Related Pages ==
* [[Concept:Tensor_Parallelism]]
* [[Concept:Weight_Sharding]]
* [[Tool:Model_Architecture_Registry]]
* [[Implementation:TP_GEMM_Kernels]]
