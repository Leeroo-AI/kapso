{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|X-LoRA|https://arxiv.org/abs/2402.07148]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Mixture_of_Experts]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

X-LoRA model class that creates a mixture-of-experts system over multiple LoRA adapters with a learned classifier for dynamic per-token routing.

=== Description ===

XLoraModel implements X-LoRA (Mixture of LoRA experts) by wrapping a LoraModel and adding a classifier network that predicts per-token, per-layer scaling factors for each adapter. The model performs a two-pass forward: first with dummy scalings to compute hidden states, then with the classifier to predict real scalings for the actual forward. This enables dynamic routing where different tokens can leverage different combinations of specialized LoRA experts.

=== Usage ===

Use XLoraModel when you have multiple task-specific LoRA adapters and want to automatically route between them. The classifier learns optimal combinations during training. X-LoRA supports top-k selection for sparse expert activation and scalings logging for analysis.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/xlora/model.py src/peft/tuners/xlora/model.py]
* '''Lines:''' 1-525

=== Signature ===
<syntaxhighlight lang="python">
def convert_layers_to_xlora(base, xloramodel, config) -> tuple[int, torch.device | None]:
    """Convert LoRA layers to X-LoRA layers with routing."""

class XLoraModel(BaseTuner):
    """
    Creates X-LoRA (Mixture of LoRA experts) model.

    Args:
        model: Base model to apply X-LoRA to
        config: XLoraConfig with adapters dict and classifier settings
        adapter_name: Name for the X-LoRA adapter

    Attributes:
        lora_model: Underlying LoraModel with loaded adapters
        internal_xlora_classifier: XLoraClassifier for routing
        internal_xlora_scalings: Latest computed scalings
    """

    def __init__(
        self,
        model: nn.Module,
        config: XLoraConfig,
        adapter_name: str,
        **kwargs,
    ) -> None:
        """Load adapters and initialize classifier."""

    def set_topk_lora(self, value: Optional[int]):
        """Set top-k expert selection."""

    def set_global_scaling_weight(self, weight: float):
        """Set global LoRA weight multiplier."""

    def get_latest_scalings(self) -> Optional[torch.Tensor]:
        """Get most recent scalings [batch, seq, layers, adapters]."""

    def enable_scalings_logging(self):
        """Enable logging of all scalings."""

    def get_scalings_log(self) -> list[torch.Tensor]:
        """Get logged scalings history."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import XLoraModel, XLoraConfig, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || nn.Module || Yes || Base transformer model
|-
| config || XLoraConfig || Yes || Configuration with adapters dict
|-
| adapters || dict[str, str] || Yes || Map of adapter names to paths/IDs
|-
| hidden_size || int || Yes || Model hidden dimension for classifier
|-
| xlora_depth || int || No || Classifier MLP depth (default: 2)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward() || ModelOutput || Model output with dynamic expert routing
|-
| get_latest_scalings() || torch.Tensor || [batch, seq, layers, adapters] scalings
|}

== Usage Examples ==

=== Creating X-LoRA Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AutoConfig
from peft import XLoraConfig, get_peft_model

model_config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

config = XLoraConfig(
    task_type="CAUSAL_LM",
    hidden_size=model_config.hidden_size,
    xlora_depth=4,              # Classifier depth
    adapters={
        "math": "./path/to/math_adapter/",
        "code": "./path/to/code_adapter/",
        "writing": "./path/to/writing_adapter/",
    },
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    use_cache=False,            # Required for X-LoRA
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== X-LoRA with Top-K Selection ===
<syntaxhighlight lang="python">
# Enable sparse top-k expert selection
model.set_topk_lora(2)  # Only use top 2 adapters per token

# Adjust global scaling weight
model.set_global_scaling_weight(0.8)

# Generate with dynamic routing
outputs = model.generate(input_ids, max_new_tokens=100)
</syntaxhighlight>

=== Analyzing Scalings ===
<syntaxhighlight lang="python">
# Enable scalings logging
model.enable_scalings_logging()

# Run inference
outputs = model(input_ids)

# Get latest scalings
scalings = model.get_latest_scalings()
# Shape: [batch_size, seq_len, n_layers, n_adapters]

# Get full log
scalings_log = model.get_scalings_log()

# Get bucketed by sequence length
bucketed = model.get_bucketed_scalings_log()

# Disable logging
model.disable_scalings_logging()
</syntaxhighlight>

=== X-LoRA Training ===
<syntaxhighlight lang="python">
# By default, LoRA adapters are frozen
# Only the classifier is trained

# To train adapters too:
config = XLoraConfig(
    use_trainable_adapters=True,  # Unfreeze LoRA weights
    # ...
)

# Or manually:
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
