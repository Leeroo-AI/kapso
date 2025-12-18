== PrefixTuningConfig ==

=== Knowledge Sources ===
* [https://github.com/huggingface/peft PEFT Repository]
* [https://arxiv.org/abs/2101.00190 Prefix-Tuning Paper]

=== Domains ===
[[Category:NLP]]
[[Category:PEFT]]
[[Category:Prompt_Tuning]]
[[Category:Prefix_Tuning]]
[[Category:Configuration]]

=== Overview ===

==== Description ====
'''PrefixTuningConfig''' is the configuration class for prefix tuning, a parameter-efficient fine-tuning method that prepends trainable continuous vectors (prefixes) to the hidden states at each transformer layer. This approach, introduced in the paper "Prefix-Tuning: Optimizing Continuous Prompts for Generation", enables task-specific adaptation while keeping the base model parameters frozen.

The configuration manages two key aspects:
* '''encoder_hidden_size''': The dimensionality of the hidden layer in the optional prefix encoder
* '''prefix_projection''': Whether to use an MLP to project prefix embeddings (recommended for improved training stability)

When prefix_projection is enabled, a two-layer MLP with tanh activation transforms the prefix embeddings, providing better reparameterization and more stable optimization.

==== Usage ====
Used to configure prefix tuning when adapting pre-trained language models to downstream tasks. Prefix tuning is particularly effective for generation tasks and provides a flexible alternative to full fine-tuning or adapter methods.

=== Code Reference ===

==== Source Location ====
<code>src/peft/tuners/prefix_tuning/config.py</code>

==== Signature ====
<syntaxhighlight lang="python">
@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """

    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )
</syntaxhighlight>

==== Import ====
<syntaxhighlight lang="python">
from peft.tuners.prefix_tuning import PrefixTuningConfig
</syntaxhighlight>

=== I/O Contract ===

==== Configuration Parameters ====
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| encoder_hidden_size || int || None || Hidden size of the MLP encoder used when prefix_projection is True
|-
| prefix_projection || bool || False || Whether to use an MLP to reparameterize prefix embeddings
|-
| colspan="4" | ''Inherits all parameters from PromptLearningConfig''
|}

==== Key Inherited Parameters ====
{| class="wikitable"
! Parameter !! Type !! Description
|-
| num_virtual_tokens || int || Number of virtual tokens (prefix length)
|-
| token_dim || int || Dimension of the token embeddings
|-
| num_layers || int || Number of transformer layers
|-
| task_type || str/TaskType || Type of task (e.g., "CAUSAL_LM", "SEQ_2_SEQ_LM")
|}

==== Returns ====
Configuration object ready to be used with PrefixEncoder model.

==== Side Effects ====
* Sets peft_type to PeftType.PREFIX_TUNING in __post_init__

=== Usage Examples ===

==== Basic Configuration Without Projection ====
<syntaxhighlight lang="python">
from peft import PrefixTuningConfig

# Simple prefix tuning without MLP projection
config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=30,
    token_dim=768,
    num_layers=12,
    prefix_projection=False
)
</syntaxhighlight>

==== Configuration With Projection (Recommended) ====
<syntaxhighlight lang="python">
from peft import PrefixTuningConfig

# Prefix tuning with MLP projection for better stability
config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=30,
    token_dim=768,
    num_layers=12,
    encoder_hidden_size=512,
    prefix_projection=True
)
</syntaxhighlight>

==== Seq2Seq Model Configuration ====
<syntaxhighlight lang="python">
from peft import PrefixTuningConfig

# Configuration for encoder-decoder models
config = PrefixTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=2,  # Encoder + Decoder
    num_attention_heads=12,
    num_layers=12,
    encoder_hidden_size=768,
    prefix_projection=True
)
</syntaxhighlight>

==== Complete Example with Model ====
<syntaxhighlight lang="python">
from peft import PrefixTuningConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Create configuration
config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    encoder_hidden_size=1024,
    prefix_projection=True
)

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Apply prefix tuning
peft_model = get_peft_model(model, config)

# Check trainable parameters
peft_model.print_trainable_parameters()
# Output shows only prefix parameters are trainable
</syntaxhighlight>

==== Comparing Projection vs No Projection ====
<syntaxhighlight lang="python">
from peft import PrefixTuningConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Without projection
config_no_proj = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prefix_projection=False
)
model1 = get_peft_model(model, config_no_proj)

# With projection (more parameters but better training stability)
config_with_proj = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    encoder_hidden_size=512,
    prefix_projection=True
)
model2 = get_peft_model(model, config_with_proj)

print("Without projection:")
model1.print_trainable_parameters()

print("\nWith projection:")
model2.print_trainable_parameters()
</syntaxhighlight>

=== Related Pages ===
* [[huggingface_peft_PrefixEncoder|PrefixEncoder]] - The model implementation using this configuration
* [[huggingface_peft_PromptLearningConfig|PromptLearningConfig]] - Parent configuration class
* [[huggingface_peft_PromptEncoderConfig|PromptEncoderConfig]] - Related P-tuning configuration
* [[Prefix_Tuning|Prefix Tuning]]
* [[PEFT|Parameter-Efficient Fine-Tuning]]
