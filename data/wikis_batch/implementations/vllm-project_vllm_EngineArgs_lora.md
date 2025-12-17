'''Metadata'''
{| class="wikitable"
|-
! Key !! Value
|-
| Knowledge Sources || vLLM Source Code, Engine Configuration API
|-
| Domains || vLLM Engine Configuration, LoRA Serving
|-
| Last Updated || 2025-12-17
|}

== Overview ==

'''EngineArgs_lora''' provides the command-line and programmatic interface for configuring LoRA support in the vLLM engine through the EngineArgs class. This implementation translates user-specified LoRA parameters into a LoRAConfig object that governs adapter serving capabilities.

== Description ==

The EngineArgs class accepts LoRA-specific parameters that control how the engine allocates resources and manages adapters during inference. These parameters are exposed both as command-line arguments and as constructor parameters for programmatic initialization.

=== Core LoRA Parameters ===

* '''enable_lora''': Boolean flag to activate LoRA support (must be True for LoRA inference)
* '''max_loras''': Integer specifying maximum concurrent LoRA adapters in a batch (default: 1)
* '''max_lora_rank''': Maximum supported rank across all adapters (default: 16, supports: 8, 16, 32, 64, 128, 256, 320, 512)
* '''max_cpu_loras''': CPU cache capacity for adapter swapping (defaults to max_loras if not specified)
* '''lora_dtype''': Data type for LoRA weights ('auto', 'float16', 'bfloat16')
* '''fully_sharded_loras''': Enable full tensor parallelism for LoRA layers (default: False)

When enable_lora=True, the EngineArgs.create_engine_config() method constructs a LoRAConfig object from these parameters, which is then embedded in the VllmConfig passed to the engine initialization.

== Code Reference ==

=== Source Location ===
* '''File''': vllm/engine/arg_utils.py
* '''Class''': EngineArgs
* '''Configuration''': vllm/config/lora.py (LoRAConfig dataclass)

=== Signature ===

<syntaxhighlight lang="python">
@dataclass
class EngineArgs:
    # LoRA-related parameters
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    max_cpu_loras: int | None = None
    lora_dtype: str = "auto"
    fully_sharded_loras: bool = False
    # ... other engine parameters

    def create_engine_config(self, usage_context: UsageContext) -> VllmConfig:
        # Creates LoRAConfig when enable_lora=True
        pass
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from vllm.engine.arg_utils import EngineArgs
from vllm import LLMEngine
</syntaxhighlight>

== I/O Contract ==

=== Input Parameters ===

{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| enable_lora || bool || False || Activates LoRA adapter support in engine
|-
| max_loras || int || 1 || Maximum LoRA adapters active in single batch
|-
| max_lora_rank || int || 16 || Highest rank value supported (affects memory)
|-
| max_cpu_loras || int \| None || None || CPU cache size (defaults to max_loras)
|-
| lora_dtype || str || "auto" || Data type: 'auto', 'float16', 'bfloat16'
|-
| fully_sharded_loras || bool || False || Enable full TP sharding for LoRA layers
|}

=== Output ===

{| class="wikitable"
|-
! Component !! Type !! Description
|-
| VllmConfig.lora_config || LoRAConfig \| None || Configured LoRAConfig object (None if enable_lora=False)
|}

== Usage Examples ==

=== Example 1: Basic LoRA Configuration ===

<syntaxhighlight lang="python">
from vllm import LLMEngine
from vllm.engine.arg_utils import EngineArgs

# Configure engine with LoRA support
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=1,
    max_lora_rank=64,
    max_cpu_loras=2
)

# Create engine from configuration
engine = LLMEngine.from_engine_args(engine_args)
</syntaxhighlight>

=== Example 2: Multi-LoRA Configuration ===

<syntaxhighlight lang="python">
from vllm.engine.arg_utils import EngineArgs
from vllm import LLMEngine

# Support multiple concurrent LoRA adapters
engine_args = EngineArgs(
    model="meta-llama/Llama-3.2-3B-Instruct",
    enable_lora=True,
    max_loras=4,            # Support 4 adapters in same batch
    max_lora_rank=128,      # Support up to rank 128
    max_cpu_loras=8,        # Cache 8 adapters in CPU memory
    lora_dtype="bfloat16"   # Use bfloat16 precision
)

engine = LLMEngine.from_engine_args(engine_args)
</syntaxhighlight>

=== Example 3: High-Performance Configuration ===

<syntaxhighlight lang="python">
from vllm.engine.arg_utils import EngineArgs
from vllm import LLMEngine

# Optimize for high sequence length and tensor parallelism
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-70B",
    enable_lora=True,
    max_loras=2,
    max_lora_rank=64,
    fully_sharded_loras=True,  # Better performance at high TP size
    tensor_parallel_size=4
)

engine = LLMEngine.from_engine_args(engine_args)
</syntaxhighlight>

=== Example 4: Command-Line Configuration ===

<syntaxhighlight lang="bash">
vllm serve meta-llama/Llama-3.1-8B \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 32 \
  --max-cpu-loras 4 \
  --lora-dtype float16
</syntaxhighlight>

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_LoRA_Engine_Configuration]]
* [[related_to::vllm-project_vllm_LLMEngine_from_engine_args]]
* [[related_to::vllm-project_vllm_LoRAConfig]]
