== PromptEncoderConfig ==

=== Knowledge Sources ===
* [https://github.com/huggingface/peft PEFT Repository]
* [https://arxiv.org/abs/2103.10385 GPT Understands, Too (P-tuning Paper)]

=== Domains ===
[[Category:NLP]]
[[Category:PEFT]]
[[Category:Prompt_Tuning]]
[[Category:P_Tuning]]
[[Category:Configuration]]

=== Overview ===

==== Description ====
'''PromptEncoderConfig''' is the configuration class for the PromptEncoder used in P-tuning. P-tuning is a parameter-efficient fine-tuning method that uses a trainable encoder (MLP or LSTM) to generate continuous prompt embeddings, rather than using discrete tokens or simple embedding layers.

The configuration manages the architecture of the prompt encoder, including:
* The type of reparameterization (MLP or LSTM)
* Hidden layer dimensions
* Number of encoder layers
* Dropout probability

This configuration extends PromptLearningConfig and is specifically designed for the P-tuning approach, which showed that using a prompt encoder can significantly improve the effectiveness of prompt-based learning, especially for smaller models.

==== Usage ====
Used to configure P-tuning models when adapting pre-trained language models with learnable continuous prompts. The encoder network generates more expressive virtual token embeddings compared to direct embedding approaches.

=== Code Reference ===

==== Source Location ====
<code>src/peft/tuners/p_tuning/config.py</code>

==== Signature ====
<syntaxhighlight lang="python">
@dataclass
class PromptEncoderConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEncoder`].

    Args:
        encoder_reparameterization_type (Union[[`PromptEncoderReparameterizationType`], `str`]):
            The type of reparameterization to use.
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        encoder_num_layers (`int`): The number of layers of the prompt encoder.
        encoder_dropout (`float`): The dropout probability of the prompt encoder.
    """

    encoder_reparameterization_type: Union[str, PromptEncoderReparameterizationType] = field(
        default=PromptEncoderReparameterizationType.MLP,
        metadata={"help": "How to reparameterize the prompt encoder"},
    )
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the prompt encoder"},
    )
    encoder_num_layers: int = field(
        default=2,
        metadata={"help": "The number of layers of the prompt encoder"},
    )
    encoder_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout of the prompt encoder"},
    )
</syntaxhighlight>

==== Import ====
<syntaxhighlight lang="python">
from peft.tuners.p_tuning import PromptEncoderConfig
</syntaxhighlight>

=== I/O Contract ===

==== Configuration Parameters ====
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| encoder_reparameterization_type || Union[str, PromptEncoderReparameterizationType] || MLP || Type of encoder: "MLP" (recommended) or "LSTM"
|-
| encoder_hidden_size || int || None || Hidden size of the prompt encoder network
|-
| encoder_num_layers || int || 2 || Number of layers in the encoder (note: MLP always uses 2 layers regardless)
|-
| encoder_dropout || float || 0.0 || Dropout probability for LSTM encoder (not used for MLP)
|-
| colspan="4" | ''Inherits all parameters from PromptLearningConfig''
|}

==== Encoder Type Enum ====
{| class="wikitable"
! Value !! Description
|-
| MLP || Multi-layer perceptron encoder (recommended, always 2 layers)
|-
| LSTM || Bidirectional LSTM encoder with configurable layers
|}

==== Returns ====
Configuration object ready to be used with PromptEncoder model.

==== Side Effects ====
* Sets peft_type to PeftType.P_TUNING in __post_init__

=== Usage Examples ===

==== Basic MLP Configuration ====
<syntaxhighlight lang="python">
from peft import PromptEncoderConfig

# Create P-tuning configuration with MLP encoder
config = PromptEncoderConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=768
)
</syntaxhighlight>

==== LSTM Encoder Configuration ====
<syntaxhighlight lang="python">
from peft import PromptEncoderConfig

# Create P-tuning configuration with LSTM encoder
config = PromptEncoderConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,
    token_dim=1024,
    encoder_reparameterization_type="LSTM",
    encoder_hidden_size=512,
    encoder_num_layers=3,
    encoder_dropout=0.1
)
</syntaxhighlight>

==== Using Enum for Encoder Type ====
<syntaxhighlight lang="python">
from peft import PromptEncoderConfig
from peft.tuners.p_tuning import PromptEncoderReparameterizationType

# Use enum for type safety
config = PromptEncoderConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=15,
    token_dim=768,
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    encoder_hidden_size=1024
)
</syntaxhighlight>

==== Complete Configuration for Sequence Classification ====
<syntaxhighlight lang="python">
from peft import PromptEncoderConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

# Create configuration
config = PromptEncoderConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=20,
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=768
)

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Apply P-tuning
peft_model = get_peft_model(model, config)
print(peft_model.print_trainable_parameters())
</syntaxhighlight>

=== Related Pages ===
* [[huggingface_peft_PromptEncoder|PromptEncoder]] - The model implementation using this configuration
* [[huggingface_peft_PromptLearningConfig|PromptLearningConfig]] - Parent configuration class
* [[huggingface_peft_PrefixTuningConfig|PrefixTuningConfig]] - Related prefix tuning configuration
* [[P_Tuning|P-Tuning]]
* [[PEFT|Parameter-Efficient Fine-Tuning]]
