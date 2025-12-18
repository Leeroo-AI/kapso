{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|C3A|https://arxiv.org/abs/2407.19342]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Circulant_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Model class that applies C3A (Circulant Convolution Adaptation) by wrapping Linear layers with FFT-based block circulant transformations.

=== Description ===

C3AModel extends BaseTuner to apply circulant convolution adaptation to transformer models. It creates C3ALinear layers for Linear target modules. Supports per-layer block_size overrides via block_size_pattern using regex matching. Only torch.nn.Linear layers are supported.

=== Usage ===

Use C3AModel for FFT-based circulant adaptation. Created automatically via get_peft_model with C3AConfig. Set block_size_pattern to override block sizes for specific layers.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/c3a/model.py src/peft/tuners/c3a/model.py]
* '''Lines:''' 1-102

=== Signature ===
<syntaxhighlight lang="python">
class C3AModel(BaseTuner):
    """
    Creates C3A model from pretrained transformer.

    Args:
        model: Base transformer model
        config: C3AConfig
        adapter_name: Name for the adapter

    Attributes:
        prefix: "c3a_"
        tuner_layer_cls: C3ALayer
    """
    prefix: str = "c3a_"
    tuner_layer_cls = C3ALayer

    def _create_and_replace(self, c3a_config, adapter_name, target, ...):
        """Create or update C3A layers with per-layer block_size."""

    @staticmethod
    def _create_new_module(c3a_config, adapter_name, target, **kwargs):
        """Create C3ALinear module."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.c3a import C3AModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || nn.Module || Yes || Base transformer model
|-
| config || C3AConfig || Yes || C3A configuration
|-
| adapter_name || str || No || Adapter name (default: "default")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| C3AModel || nn.Module || Model with C3A layers
|}

== Usage Examples ==

=== Basic C3A Model ===
<syntaxhighlight lang="python">
from peft import C3AConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = C3AConfig(
    block_size=256,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Per-Layer Block Size ===
<syntaxhighlight lang="python">
from peft import C3AConfig, get_peft_model

# Use different block sizes for different layers
config = C3AConfig(
    block_size=256,
    block_size_pattern={
        r"layers\.0\..*\.k_proj": 128,  # Regex matching
        "layers.1.self_attn.k_proj": 64,  # Exact match
    },
    target_modules=["q_proj", "k_proj", "v_proj"],
)

model = get_peft_model(model, config)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
