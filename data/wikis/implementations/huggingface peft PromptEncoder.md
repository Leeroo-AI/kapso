== PromptEncoder ==

=== Knowledge Sources ===
* [https://github.com/huggingface/peft PEFT Repository]
* [https://github.com/NVIDIA/NeMo NeMo Framework (Original Implementation)]
* [https://arxiv.org/abs/2103.10385 GPT Understands, Too (P-tuning Paper)]

=== Domains ===
[[Category:NLP]]
[[Category:PEFT]]
[[Category:Prompt_Tuning]]
[[Category:P_Tuning]]
[[Category:Neural_Networks]]

=== Overview ===

==== Description ====
'''PromptEncoder''' is a PyTorch neural network module that generates virtual token embeddings for P-tuning. Unlike simple embedding layers, the PromptEncoder uses a trainable neural network (MLP or LSTM) to reparameterize the prompt embeddings, making them more expressive and effective.

The architecture consists of:
* An embedding layer that creates initial token representations
* A reparameterization network (MLP or bidirectional LSTM) that transforms embeddings
* For LSTM: A bidirectional LSTM followed by a 2-layer MLP
* For MLP: A 2-layer feed-forward network with ReLU activations

This approach, based on NVIDIA's NeMo implementation, has been shown to significantly improve prompt-based learning, especially for smaller language models. The encoder transforms fixed-size virtual token indices into continuous embeddings that can be prepended to the input sequence.

==== Usage ====
Used as the prompt generation component in P-tuning PEFT models. It creates learnable continuous prompts that guide model behavior without modifying the base model parameters.

=== Code Reference ===

==== Source Location ====
<code>src/peft/tuners/p_tuning/model.py</code>

==== Signature ====
<syntaxhighlight lang="python">
class PromptEncoder(torch.nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for p-tuning.

    Args:
        config ([`PromptEncoderConfig`]): The configuration of the prompt encoder.

    Input shape: (`batch_size`, `total_virtual_tokens`)

    Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config):
        """
        Initialize the prompt encoder.

        Args:
            config: PromptEncoderConfig object
        """

    def forward(self, indices):
        """
        Generate continuous prompt embeddings.

        Args:
            indices: Virtual token indices

        Returns:
            Continuous prompt embeddings
        """
</syntaxhighlight>

==== Import ====
<syntaxhighlight lang="python">
from peft import PromptEncoder, PromptEncoderConfig
</syntaxhighlight>

=== I/O Contract ===

==== Constructor Parameters ====
{| class="wikitable"
! Parameter !! Type !! Description
|-
| config || PromptEncoderConfig || Configuration object specifying encoder architecture
|}

==== Forward Method ====

===== Inputs =====
{| class="wikitable"
! Parameter !! Type !! Shape !! Description
|-
| indices || torch.Tensor || (batch_size, total_virtual_tokens) || Indices for virtual token embeddings
|}

===== Returns =====
{| class="wikitable"
! Type !! Shape !! Description
|-
| torch.Tensor || (batch_size, total_virtual_tokens, token_dim) || Continuous prompt embeddings
|}

==== Model Attributes ====
{| class="wikitable"
! Attribute !! Type !! Description
|-
| token_dim || int || Hidden embedding dimension of the base transformer
|-
| input_size || int || Input size to the encoder (equals token_dim)
|-
| output_size || int || Output size from the encoder (equals token_dim)
|-
| hidden_size || int || Hidden size of the encoder network
|-
| total_virtual_tokens || int || Total number of virtual tokens (num_virtual_tokens × num_transformer_submodules)
|-
| encoder_type || PromptEncoderReparameterizationType || Type of encoder (MLP or LSTM)
|-
| embedding || torch.nn.Embedding || Embedding layer for virtual tokens
|-
| mlp_head || torch.nn.Sequential || MLP transformation network (always present when not in inference mode)
|-
| lstm_head || torch.nn.LSTM || Bidirectional LSTM (only present when encoder_type is LSTM)
|}

==== Side Effects ====
* Creates trainable embedding and encoder parameters
* Warns if encoder_num_layers is specified for MLP (always uses 2 layers)

==== Exceptions ====
* '''ValueError''': Raised if encoder_reparameterization_type is not MLP or LSTM

=== Usage Examples ===

==== Basic MLP Encoder ====
<syntaxhighlight lang="python">
import torch
from peft import PromptEncoder, PromptEncoderConfig

# Create configuration with MLP encoder
config = PromptEncoderConfig(
    peft_type="P_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=768,
)

# Create prompt encoder
prompt_encoder = PromptEncoder(config)

# Generate prompt embeddings
batch_size = 4
indices = torch.arange(20).unsqueeze(0).expand(batch_size, -1)
embeddings = prompt_encoder(indices)

print(embeddings.shape)  # torch.Size([4, 20, 768])
</syntaxhighlight>

==== LSTM Encoder ====
<syntaxhighlight lang="python">
import torch
from peft import PromptEncoder, PromptEncoderConfig

# Create configuration with LSTM encoder
config = PromptEncoderConfig(
    peft_type="P_TUNING",
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,
    token_dim=1024,
    num_transformer_submodules=1,
    encoder_reparameterization_type="LSTM",
    encoder_hidden_size=512,
    encoder_num_layers=2,
    encoder_dropout=0.1,
)

# Create prompt encoder
prompt_encoder = PromptEncoder(config)

# Forward pass
batch_size = 8
indices = torch.arange(10).unsqueeze(0).expand(batch_size, -1)
embeddings = prompt_encoder(indices)

print(embeddings.shape)  # torch.Size([8, 10, 1024])
</syntaxhighlight>

==== Integration with PEFT Model ====
<syntaxhighlight lang="python">
from peft import get_peft_model, PromptEncoderConfig
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Create P-tuning configuration
config = PromptEncoderConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=768
)

# Apply P-tuning (PromptEncoder is created internally)
peft_model = get_peft_model(model, config)

# Check trainable parameters
peft_model.print_trainable_parameters()
# Output: trainable params: X || all params: Y || trainable%: Z
</syntaxhighlight>

==== Encoder-Decoder Model (Seq2Seq) ====
<syntaxhighlight lang="python">
import torch
from peft import PromptEncoder, PromptEncoderConfig

# For encoder-decoder models, total_virtual_tokens is doubled
config = PromptEncoderConfig(
    peft_type="P_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=2,  # Encoder + Decoder
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=768,
)

prompt_encoder = PromptEncoder(config)

# total_virtual_tokens = 20 * 2 = 40
print(f"Total virtual tokens: {prompt_encoder.total_virtual_tokens}")
# Output: Total virtual tokens: 40

batch_size = 4
indices = torch.arange(40).unsqueeze(0).expand(batch_size, -1)
embeddings = prompt_encoder(indices)

print(embeddings.shape)  # torch.Size([4, 40, 768])
</syntaxhighlight>

==== Inspecting Model Architecture ====
<syntaxhighlight lang="python">
import torch
from peft import PromptEncoder, PromptEncoderConfig

config = PromptEncoderConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,
    token_dim=768,
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=1024,
)

prompt_encoder = PromptEncoder(config)

# Inspect the MLP architecture
print("Embedding layer:", prompt_encoder.embedding)
print("\nMLP Head:")
for i, layer in enumerate(prompt_encoder.mlp_head):
    print(f"  Layer {i}: {layer}")

# Output shows:
# Embedding layer: Embedding(10, 768)
# MLP Head:
#   Layer 0: Linear(in_features=768, out_features=1024, bias=True)
#   Layer 1: ReLU()
#   Layer 2: Linear(in_features=1024, out_features=1024, bias=True)
#   Layer 3: ReLU()
#   Layer 4: Linear(in_features=1024, out_features=768, bias=True)
</syntaxhighlight>

=== Related Pages ===
* [[huggingface_peft_PromptEncoderConfig|PromptEncoderConfig]] - Configuration for this encoder
* [[huggingface_peft_PrefixEncoder|PrefixEncoder]] - Related prefix tuning encoder
* [[huggingface_peft_MultitaskPromptTuningModel|MultitaskPromptEmbedding]] - Related multitask prompt embedding
* [[P_Tuning|P-Tuning]]
* [[PEFT|Parameter-Efficient Fine-Tuning]]
