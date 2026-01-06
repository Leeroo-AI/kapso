{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|BOFT|https://arxiv.org/abs/2311.06243]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Orthogonal_Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Model class that applies BOFT (Butterfly Orthogonal Fine-Tuning) by wrapping Linear and Conv2d layers with butterfly-factorized orthogonal transformations.

=== Description ===

BOFTModel extends BaseTuner to apply BOFT to transformer models. It creates Linear or Conv2d BOFT layers based on target module type, passing block_size, block_num, and n_butterfly_factor parameters. Uses TRANSFORMERS_MODELS_TO_BOFT_TARGET_MODULES_MAPPING for default target modules. The _create_and_replace method handles both new module creation and updating existing BOFT layers with additional adapters.

=== Usage ===

Use BOFTModel for orthogonal fine-tuning with butterfly factorization. Created automatically via get_peft_model with BOFTConfig. Supports both Linear and Conv2d layers (e.g., for vision models like DinoV2).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/model.py src/peft/tuners/boft/model.py]
* '''Lines:''' 1-132

=== Signature ===
<syntaxhighlight lang="python">
class BOFTModel(BaseTuner):
    """
    Creates BOFT model from pretrained transformer.

    Args:
        model: Base transformer model
        config: BOFTConfig
        adapter_name: Name for the adapter

    Attributes:
        prefix: "boft_"
        tuner_layer_cls: BOFTLayer
        target_module_mapping: Default target modules per model type
    """
    prefix: str = "boft_"
    tuner_layer_cls = BOFTLayer

    def _create_and_replace(
        self,
        boft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        """Create or update BOFT layers."""

    @staticmethod
    def _create_new_module(boft_config, adapter_name, target, **kwargs):
        """Create Linear or Conv2d BOFT module."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.boft import BOFTModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || nn.Module || Yes || Base transformer model
|-
| config || BOFTConfig || Yes || BOFT configuration
|-
| adapter_name || str || No || Adapter name (default: "default")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| BOFTModel || nn.Module || Model with BOFT layers
|}

== Usage Examples ==

=== Vision Model with BOFT ===
<syntaxhighlight lang="python">
from peft import BOFTConfig, get_peft_model
import transformers

# BOFT works well for vision models
model = transformers.Dinov2ForImageClassification.from_pretrained(
    "facebook/dinov2-large",
    num_labels=100,
)

config = BOFTConfig(
    boft_block_size=8,
    boft_n_butterfly_factor=1,
    target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
    boft_dropout=0.1,
    bias="boft_only",
    modules_to_save=["classifier"],
)

boft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Multi-Adapter BOFT ===
<syntaxhighlight lang="python">
from peft import BOFTConfig, get_peft_model

# Add multiple BOFT adapters
config1 = BOFTConfig(boft_block_size=4, target_modules=["q_proj"])
config2 = BOFTConfig(boft_block_size=8, target_modules=["q_proj"])

model = get_peft_model(base_model, config1, adapter_name="adapter1")
model.add_adapter("adapter2", config2)

# Switch between adapters
model.set_adapter("adapter2")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
