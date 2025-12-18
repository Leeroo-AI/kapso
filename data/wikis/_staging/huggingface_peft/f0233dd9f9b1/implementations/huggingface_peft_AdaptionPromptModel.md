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

Model class that applies LLaMA-Adapter style adaption prompts by replacing top L attention modules with AdaptedAttention wrappers.

=== Description ===

AdaptionPromptModel wraps a transformer model and replaces the top adapter_layers attention modules with AdaptedAttention wrappers. It supports multi-adapter management by caching inactive adapters and swapping them in/out. The model freezes all parameters except adaption_prompt and adaption_gate. Different adapters are stored in _cached_adapters dictionary and swapped via set_adapter().

=== Usage ===

Use AdaptionPromptModel for LLaMA-Adapter fine-tuning. Created automatically via get_peft_model with AdaptionPromptConfig. Supports adding multiple adapters, switching between them, and enabling/disabling adapter layers.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adaption_prompt/model.py src/peft/tuners/adaption_prompt/model.py]
* '''Lines:''' 1-170

=== Signature ===
<syntaxhighlight lang="python">
class AdaptionPromptModel(nn.Module):
    """
    Implements adaption prompts (LLaMA-Adapter).

    Attributes:
        model: Wrapped transformer model
        peft_config: Dict of adapter configs by name
        _parents: Module parents for attention replacement
        _cached_adapters: Inactive adapters storage
        _active_adapter: Currently active adapter name
    """

    def __init__(
        self,
        model,
        configs: dict,
        adapter_name: str,
    ):
        """Initialize and add first adapter."""

    def add_adapter(self, adapter_name: str, config: AdaptionPromptConfig):
        """Add a new adapter with given config."""

    def set_adapter(self, adapter_name: str):
        """Switch to a different adapter."""

    def enable_adapter_layers(self):
        """Enable adapter layers."""

    def disable_adapter_layers(self):
        """Disable adapter layers."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.adaption_prompt import AdaptionPromptModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || nn.Module || Yes || Base transformer model
|-
| configs || dict || Yes || Adapter configs by name
|-
| adapter_name || str || Yes || Initial adapter name
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward() || ModelOutput || Wrapped model output
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from peft import AdaptionPromptConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = AdaptionPromptConfig(
    adapter_len=10,
    adapter_layers=30,
)

# Creates AdaptionPromptModel internally
peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Multi-Adapter Support ===
<syntaxhighlight lang="python">
# Add multiple adapters
peft_model.add_adapter("task1", config1)
peft_model.add_adapter("task2", config2)

# Switch between adapters
peft_model.set_adapter("task1")
output1 = peft_model(input_ids)

peft_model.set_adapter("task2")
output2 = peft_model(input_ids)
</syntaxhighlight>

=== Enable/Disable Adapters ===
<syntaxhighlight lang="python">
# Disable adapters for base model inference
peft_model.disable_adapter_layers()
base_output = peft_model(input_ids)

# Re-enable adapters
peft_model.enable_adapter_layers()
adapted_output = peft_model(input_ids)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
