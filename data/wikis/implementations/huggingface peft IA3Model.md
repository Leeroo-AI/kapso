{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|IA3|https://arxiv.org/abs/2205.05638]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Activation_Scaling]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Model class that creates IA3 adapters from a pretrained transformer, managing activation scaling vectors across key, value, and feedforward modules.

=== Description ===

IA3Model extends BaseTuner to implement the Infused Adapter by Inhibiting and Amplifying Inner Activations method. The model manages learned scaling vectors for attention (key/value) and feedforward layers. It distinguishes between feedforward and attention layers, applying different scaling strategies. The model supports 4-bit and 8-bit quantization via bitsandbytes integration.

=== Usage ===

Use IA3Model when you need the most parameter-efficient adapter method with minimal memory footprint. IA3 is ideal for deploying many adapters simultaneously since each adapter only adds vectors (not matrices), making it extremely lightweight compared to LoRA.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/model.py src/peft/tuners/ia3/model.py]
* '''Lines:''' 1-316

=== Signature ===
<syntaxhighlight lang="python">
class IA3Model(BaseTuner):
    """
    Creates IA3 model from a pretrained transformers model.

    Args:
        model: The model to be adapted (PreTrainedModel)
        config: The configuration of the IA3 model (IA3Config)
        adapter_name: The name of the adapter (default: "default")
        low_cpu_mem_usage: Create empty adapter weights on meta device

    Attributes:
        prefix: "ia3_" - prefix for IA3 parameters
        tuner_layer_cls: IA3Layer class
    """
    prefix = "ia3_"
    tuner_layer_cls = IA3Layer

    @staticmethod
    def _create_new_module(ia3_config, adapter_name, target, **kwargs):
        """Create new IA3 module for the target layer."""

    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
    ) -> None:
        """Merge adapters with given weights into new adapter."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import IA3Model, IA3Config, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || The pretrained model to adapt
|-
| config || IA3Config || Yes || Configuration specifying target and feedforward modules
|-
| adapter_name || str || No || Name for the adapter (default: "default")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward() || ModelOutput || Model output with IA3 scaling applied
|-
| add_weighted_adapter() || None || Creates new merged adapter from existing adapters
|}

== Usage Examples ==

=== Basic IA3 Model ===
<syntaxhighlight lang="python">
from peft import IA3Config, IA3Model, get_peft_model
from transformers import AutoModelForSeq2SeqLM

# Load base model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Configure IA3
config = IA3Config(
    target_modules=["k", "v", "wo"],
    feedforward_modules=["wo"],  # Specify which are feedforward
    init_ia3_weights=True,
)

# Create IA3 model
model = get_peft_model(model, config)

# Check parameter count
model.print_trainable_parameters()
# IA3 typically has ~10x fewer parameters than LoRA
</syntaxhighlight>

=== IA3 with Model Type Auto-Detection ===
<syntaxhighlight lang="python">
from peft import IA3Config, get_peft_model
from transformers import AutoModelForCausalLM

# For supported models, target/feedforward modules are auto-detected
model = AutoModelForCausalLM.from_pretrained("gpt2")

config = IA3Config(
    # target_modules and feedforward_modules auto-detected for gpt2
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Weighted Adapter Merging ===
<syntaxhighlight lang="python">
# Train multiple IA3 adapters for different tasks
# ...training code...

# Merge adapters with custom weights
model.add_weighted_adapter(
    adapters=["task1", "task2"],
    weights=[0.7, 0.3],
    adapter_name="merged_tasks",
)

# Activate the merged adapter
model.set_adapter("merged_tasks")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
