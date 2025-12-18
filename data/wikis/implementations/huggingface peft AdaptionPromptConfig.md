{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|LLaMA-Adapter|https://arxiv.org/abs/2303.16199]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Prompt_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Configuration class for Adaption Prompt (LLaMA-Adapter) that stores parameters for inserting learnable prompt tokens into attention layers.

=== Description ===

AdaptionPromptConfig stores configuration for inserting adapter tokens into attention computations. The adapter_len parameter controls how many tokens to insert, and adapter_layers specifies how many layers from the top receive adapters. Model-specific configurations (target_modules, projection layers) are defined in TRANSFORMERS_MODEL_CONFIG for llama, mistral, and gpt2 model types.

=== Usage ===

Use AdaptionPromptConfig for LLaMA-Adapter style fine-tuning where learnable tokens are prepended to keys and values in attention. Supports LLaMA, Mistral, and GPT-2 model types. Target modules are auto-configured based on model type if not specified.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adaption_prompt/config.py src/peft/tuners/adaption_prompt/config.py]
* '''Lines:''' 1-89

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class AdaptionPromptConfig(PeftConfig):
    """
    Configuration for Adaption Prompt (LLaMA-Adapter).

    Args:
        target_modules: Attention submodule name to insert prompts
        adapter_len: Number of adapter tokens to insert
        adapter_layers: Number of layers from top to adapt
    """
    target_modules: str = None
    adapter_len: int = None
    adapter_layers: int = None

ModelTypeConfig = namedtuple(
    "ModelTypeConfig",
    ["compute_query_states", "target_modules", "k_proj_layer", "v_proj_layer", "o_proj_layer"]
)

TRANSFORMERS_MODEL_CONFIG = {
    "llama": ModelTypeConfig(...),
    "mistral": ModelTypeConfig(...),
    "gpt2": ModelTypeConfig(...),
}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import AdaptionPromptConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| adapter_len || int || Yes || Number of adapter tokens
|-
| adapter_layers || int || Yes || Number of layers to adapt (from top)
|-
| target_modules || str || No || Attention submodule (auto-configured)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| AdaptionPromptConfig || dataclass || Configuration for get_peft_model
|}

== Usage Examples ==

=== Basic Adaption Prompt ===
<syntaxhighlight lang="python">
from peft import AdaptionPromptConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = AdaptionPromptConfig(
    adapter_len=10,         # 10 adapter tokens
    adapter_layers=30,      # Apply to top 30 layers
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
</syntaxhighlight>

=== Supported Model Types ===
<syntaxhighlight lang="python">
# LLaMA/Mistral: uses self_attn with k_proj, v_proj, o_proj
# GPT-2: uses attn with c_attn

# Model type is auto-detected from config
config = AdaptionPromptConfig(
    adapter_len=10,
    adapter_layers=12,
    # target_modules auto-set based on model.config.model_type
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
