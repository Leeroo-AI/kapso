{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Paper|LoRA: Low-Rank Adaptation|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::NLP]], [[domain::LoRA]], [[domain::Model_Serving]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for configuring the vLLM engine to support serving multiple LoRA adapters dynamically on a single base model.

=== Description ===

When initializing vLLM with LoRA support, the `EngineArgs` configuration determines:
- Whether LoRA serving is enabled
- Maximum number of concurrent adapters in a batch
- Maximum LoRA rank supported
- CPU cache size for adapter swapping
- Additional vocabulary size for adapter-specific tokens

This configuration is set at engine startup and cannot be changed without restarting the server.

=== Usage ===

Configure LoRA engine settings when:
- Deploying multi-tenant LoRA serving
- Running multiple fine-tuned model variants on one base model
- Optimizing memory for adapter caching
- Supporting high-rank LoRA adapters

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/engine/arg_utils.py
* '''Lines:''' L1-300

=== Signature ===
<syntaxhighlight lang="python">
# LoRA-specific EngineArgs parameters
class EngineArgs:
    # ... other parameters ...

    enable_lora: bool = False
    """Enable LoRA adapter serving."""

    max_loras: int = 1
    """Maximum number of LoRA adapters in a single batch."""

    max_lora_rank: int = 16
    """Maximum LoRA rank supported (affects memory allocation)."""

    fully_sharded_loras: bool = False
    """Enable fully sharded LoRAs for tensor parallelism."""

    max_cpu_loras: int | None = None
    """Maximum LoRA adapters cached in CPU memory (default: max_loras)."""

    lora_extra_vocab_size: int = 256
    """Extra vocabulary slots for adapter-specific tokens."""

    lora_dtype: str = "auto"
    """Data type for LoRA weights (auto, float16, bfloat16, float32)."""

    long_lora_scaling_factors: tuple[float, ...] | None = None
    """Scaling factors for long-context LoRA (S2 attention)."""
</syntaxhighlight>

=== Import / Usage ===
<syntaxhighlight lang="python">
from vllm import LLM

# Initialize LLM with LoRA support
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=4,
    max_lora_rank=64,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| enable_lora || bool || Yes || Enable LoRA support (default: False)
|-
| max_loras || int || No || Max concurrent adapters per batch (default: 1)
|-
| max_lora_rank || int || No || Maximum LoRA rank supported (default: 16)
|-
| max_cpu_loras || int || No || CPU cache size for adapters
|-
| lora_extra_vocab_size || int || No || Extra vocab slots (default: 256)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| LLM/Engine || LLM || Engine configured for LoRA adapter serving
|}

== Usage Examples ==

=== Basic LoRA Configuration ===
<syntaxhighlight lang="python">
from vllm import LLM

# Enable LoRA with default settings
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
)
</syntaxhighlight>

=== Multi-Adapter Serving ===
<syntaxhighlight lang="python">
from vllm import LLM

# Support up to 4 concurrent adapters
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_loras=4,           # 4 adapters per batch
    max_lora_rank=64,      # Support rank up to 64
    max_cpu_loras=16,      # Cache 16 adapters in CPU RAM
    lora_extra_vocab_size=512,  # Extra tokens for adapters
)
</syntaxhighlight>

=== High-Rank with Tensor Parallelism ===
<syntaxhighlight lang="python">
from vllm import LLM

# Fully sharded LoRA for multi-GPU
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    enable_lora=True,
    max_loras=2,
    max_lora_rank=128,
    fully_sharded_loras=True,  # Shard LoRA weights across GPUs
)
</syntaxhighlight>

=== CLI Configuration ===
<syntaxhighlight lang="bash">
# Server with LoRA support
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --max-cpu-loras 16 \
    --lora-extra-vocab-size 256
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_LoRA_Engine_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
