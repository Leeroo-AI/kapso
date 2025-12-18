{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Fourier Transform]], [[domain::Model Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
FourierFTModel creates a Fourier Fine-Tuning adapted model from a pretrained transformers model by applying frequency-domain transformations to specific layers.

=== Description ===
FourierFTModel is a parameter-efficient fine-tuning (PEFT) method that adapts pretrained models using Fourier frequency transformations. The method modifies selected layers by injecting learnable frequency components, allowing efficient adaptation with minimal trainable parameters. It extends the BaseTuner class and specifically targets linear layers (torch.nn.Linear and Conv1D) for transformation. The approach is based on the research described in https://huggingface.co/papers/2405.03003.

The model supports adapter-based architecture, allowing multiple adapters with different configurations. It uses pattern-based targeting to apply different frequency parameters to different layers based on regex patterns.

=== Usage ===
Use FourierFTModel when you need to fine-tune large language models with minimal parameter overhead using frequency-domain techniques. It's particularly useful when:
* You want to adapt pretrained models efficiently
* You need control over the number of frequency components per layer
* You want to experiment with Fourier-based adaptation methods
* Memory constraints limit full fine-tuning

== Code Reference ==
=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/fourierft/model.py src/peft/tuners/fourierft/model.py]
* '''Lines:''' 31-129

=== Signature ===
<syntaxhighlight lang="python">
class FourierFTModel(BaseTuner):
    def __init__(
        self,
        model: torch.nn.Module,
        config: FourierFTConfig,
        adapter_name: str = "default",
        low_cpu_mem_usage: bool = False
    ):
        """
        Args:
            model: The model to be adapted
            config: The configuration of the FourierFT model
            adapter_name: The name of the adapter, defaults to "default"
            low_cpu_mem_usage: Create empty adapter weights on meta device
        """

    def _create_and_replace(
        self,
        fourierft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        """Create and replace target modules with FourierFT layers"""

    @staticmethod
    def _create_new_module(fourierft_config, adapter_name, target, **kwargs):
        """Create new FourierFT module based on target type"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import FourierFTModel, FourierFTConfig
from peft import get_peft_model
</syntaxhighlight>

== I/O Contract ==
=== Input Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description !! Default
|-
| model || torch.nn.Module || The pretrained model to be adapted || Required
|-
| config || FourierFTConfig || Configuration object with FourierFT parameters || Required
|-
| adapter_name || str || Name identifier for the adapter || "default"
|-
| low_cpu_mem_usage || bool || Whether to create empty weights on meta device || False
|}

=== Configuration Parameters (FourierFTConfig) ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| n_frequency || int || Number of frequency components
|-
| n_frequency_pattern || dict || Pattern-based frequency configuration for different layers
|-
| scaling || float || Scaling factor for frequency components
|-
| random_loc_seed || int || Random seed for frequency location initialization
|-
| fan_in_fan_out || bool || Whether target layer stores weights as (fan_in, fan_out)
|-
| init_weights || bool || Whether to initialize FourierFT weights
|}

=== Output ===
{| class="wikitable"
! Return Type !! Description
|-
| torch.nn.Module || The adapted model with FourierFT layers injected
|}

=== Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| prefix || str || "fourierft_" - prefix for FourierFT parameters
|-
| tuner_layer_cls || type || FourierFTLayer class
|-
| target_module_mapping || dict || Mapping of model architectures to default target modules
|}

== Usage Examples ==
=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import FourierFTConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure FourierFT
config = FourierFTConfig(
    n_frequency=32,
    target_modules=["q_proj", "v_proj"],
    scaling=1.0,
    init_weights=True
)

# Create FourierFT model
model = get_peft_model(base_model, config)

# Check trainable parameters
model.print_trainable_parameters()
</syntaxhighlight>

=== Advanced Configuration with Pattern-Based Frequencies ===
<syntaxhighlight lang="python">
from peft import FourierFTConfig, get_peft_model

# Configure different frequencies for different layer patterns
config = FourierFTConfig(
    n_frequency=32,  # default
    n_frequency_pattern={
        "q_proj": 64,  # use 64 frequencies for query projections
        "v_proj": 64,  # use 64 frequencies for value projections
        "mlp": 16      # use 16 frequencies for MLP layers
    },
    target_modules=["q_proj", "v_proj", "k_proj", "mlp"],
    scaling=2.0,
    random_loc_seed=42
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Working with Conv1D Layers ===
<syntaxhighlight lang="python">
# For models like GPT-2 that use Conv1D
config = FourierFTConfig(
    n_frequency=32,
    target_modules=["c_attn", "c_proj"],
    fan_in_fan_out=True,  # Required for Conv1D layers
    init_weights=True
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Multiple Adapters ===
<syntaxhighlight lang="python">
# Add first adapter
config1 = FourierFTConfig(n_frequency=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config1, adapter_name="adapter1")

# Add second adapter
config2 = FourierFTConfig(n_frequency=64, target_modules=["q_proj", "v_proj"])
model.add_adapter("adapter2", config2)

# Switch between adapters
model.set_adapter("adapter1")
# ... training/inference
model.set_adapter("adapter2")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
