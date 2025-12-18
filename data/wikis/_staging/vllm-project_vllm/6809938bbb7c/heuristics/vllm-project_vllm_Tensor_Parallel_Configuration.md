# Tensor Parallel Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Distributed Inference|https://docs.vllm.ai/en/latest/serving/distributed_serving.html]]
|-
! Domains
| [[domain::Optimization]], [[domain::Distributed_Computing]], [[domain::Multi_GPU]]
|-
! Last Updated
| [[last_updated::2025-01-15 14:00 GMT]]
|}

== Overview ==

Configuration guidance for distributing model inference across multiple GPUs using tensor parallelism (TP) to handle models larger than single-GPU memory.

=== Description ===

Tensor parallelism splits model layers horizontally across multiple GPUs, allowing models that don't fit in single-GPU memory to be served. Each GPU holds a fraction (1/TP) of the model weights and computes a portion of each layer's output. This requires high-bandwidth GPU interconnects (NVLink preferred) for efficient all-reduce operations between GPUs.

=== Usage ===

Use this heuristic when:
- **Model too large** for single GPU (e.g., 70B model on 24GB GPUs)
- **Reducing per-GPU memory** to enable larger batch sizes
- **Latency-sensitive workloads** where TP has lower latency than pipeline parallelism
- **Multi-GPU server** with fast interconnects (NVLink, NVSwitch)

== The Insight (Rule of Thumb) ==

* **Action:** Set `tensor_parallel_size` in `LLM()` constructor
* **Default Value:** 1 (single GPU)
* **Valid Values:** Power of 2 typically (1, 2, 4, 8)
* **Trade-off:** More GPUs = more memory available but higher communication overhead

{| class="wikitable"
! Model Size !! Recommended TP !! GPU Config (80GB A100)
|-
| 7B parameters || 1 || 1x A100
|-
| 13B parameters || 1 || 1x A100
|-
| 34B parameters || 2 || 2x A100
|-
| 70B parameters || 4 || 4x A100 (or 2x with quantization)
|-
| 180B parameters || 8 || 8x A100
|-
| 405B parameters (Llama 3.1) || 8+ || 8x H100 or split across nodes
|}

{| class="wikitable"
! Scenario !! TP Setting !! Rationale
|-
| Single GPU fits model || 1 || Avoid unnecessary communication overhead
|-
| Model slightly exceeds VRAM || 2 || Minimum split to fit model
|-
| Maximize throughput || Minimum that fits || More TP = more overhead, less throughput per GPU
|-
| Ultra-low latency || 4-8 || More GPUs reduce per-GPU computation
|-
| Mixed GPU types || Not recommended || TP requires identical GPUs
|}

== Reasoning ==

'''Why TP matters:'''
- Each GPU processes 1/TP of the attention heads and FFN dimensions
- Requires all-reduce synchronization after each layer (communication bound)
- NVLink provides 600+ GB/s bandwidth (vs PCIe's ~32 GB/s)
- Memory scales linearly: 70B model uses ~35GB per GPU with TP=4

'''Communication overhead:'''
- TP introduces all-reduce operations between every transformer layer
- On NVLink: ~5-10% overhead for TP=2, ~15-25% for TP=8
- On PCIe: Overhead can exceed 50%, not recommended for TP>2

'''When NOT to use high TP:'''
- If model fits in single GPU memory
- PCIe-only connections (prefer pipeline parallelism)
- Throughput-maximizing scenarios (multiple replicas better than high TP)

== Code Evidence ==

Parameter definition from `vllm/entrypoints/llm.py:118-119`:
<syntaxhighlight lang="python">
tensor_parallel_size: The number of GPUs to use for distributed
    execution with tensor parallelism.
</syntaxhighlight>

Default in constructor from `vllm/entrypoints/llm.py:202`:
<syntaxhighlight lang="python">
def __init__(
    self,
    model: str,
    ...
    tensor_parallel_size: int = 1,  # Default: single GPU
    ...
)
</syntaxhighlight>

CUDAGraph batch size scaling with TP from `vllm/config/vllm.py:1003-1017`:
<syntaxhighlight lang="python">
# Filter sizes that aren't divisible by TP size for sequence parallelism
removed_sizes = [
    size
    for size in possible_sizes
    if size % self.parallel_config.tensor_parallel_size != 0
]
if removed_sizes:
    logger.warning(
        "Removed %s from cudagraph capture batch size because "
        "sequence parallelism is enabled",
        removed_sizes,
        self.parallel_config.tensor_parallel_size,
    )
</syntaxhighlight>

== Usage Examples ==

=== Basic Multi-GPU Setup ===
<syntaxhighlight lang="python">
from vllm import LLM

# Serve Llama 70B on 4 GPUs
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
)
</syntaxhighlight>

=== Server Deployment ===
<syntaxhighlight lang="bash">
# Launch vLLM server with tensor parallelism
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --port 8000
</syntaxhighlight>

=== Combined with Quantization ===
<syntaxhighlight lang="python">
# Run 70B with quantization on 2 GPUs instead of 4
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,
    quantization="awq",  # 4-bit quantization
)
</syntaxhighlight>

== Common Issues ==

{| class="wikitable"
! Issue !! Cause !! Solution
|-
|| OOM even with TP=8 || Model too large or high batch size || Add quantization, reduce batch size
|-
|| Slow inference with high TP || PCIe bottleneck || Use NVLink-connected GPUs
|-
|| Initialization hangs || NCCL communication failure || Check `NCCL_DEBUG=INFO`, verify GPU topology
|-
|| Uneven GPU utilization || Non-divisible attention heads || TP should divide num_attention_heads
|}

== Related Pages ==

* [[uses_heuristic::Implementation:vllm-project_vllm_LLM_init]]
* [[uses_heuristic::Workflow:vllm-project_vllm_Basic_Offline_Inference]]
* [[uses_heuristic::Workflow:vllm-project_vllm_OpenAI_Compatible_Serving]]
* [[uses_heuristic::Principle:vllm-project_vllm_LLM_Class_Initialization]]
