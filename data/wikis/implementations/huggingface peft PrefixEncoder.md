== PrefixEncoder ==

=== Knowledge Sources ===
* [https://github.com/huggingface/peft PEFT Repository]
* [https://github.com/THUDM/P-tuning-v2 P-tuning v2 (Original Implementation)]
* [https://arxiv.org/abs/2101.00190 Prefix-Tuning Paper]

=== Domains ===
[[Category:NLP]]
[[Category:PEFT]]
[[Category:Prompt_Tuning]]
[[Category:Prefix_Tuning]]
[[Category:Neural_Networks]]

=== Overview ===

==== Description ====
'''PrefixEncoder''' is a PyTorch neural network module that encodes prefix embeddings for prefix tuning. It generates continuous vectors that are prepended to the keys and values at each layer of a transformer model, effectively conditioning the model's behavior without modifying its parameters.

The encoder supports two modes:
* '''Direct embedding''': Maps virtual token indices directly to prefix embeddings
* '''Projected embedding''': Uses a two-layer MLP (with tanh activation) to reparameterize the prefix, providing better training stability and optimization

The architecture is based on the P-tuning v2 implementation and outputs prefix embeddings that contain both key and value vectors for all transformer layers (shape: batch_size × num_virtual_tokens × 2*layers*hidden_dim).

==== Usage ====
Used as the prefix generation component in prefix tuning PEFT models. It creates learnable continuous prefixes that are inserted into the attention mechanism at each transformer layer, allowing task-specific adaptation while keeping the base model frozen.

=== Code Reference ===

==== Source Location ====
<code>src/peft/tuners/prefix_tuning/model.py</code>

==== Signature ====
<syntaxhighlight lang="python">
class PrefixEncoder(torch.nn.Module):
    r"""
    The `torch.nn` model to encode the prefix.

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example:

    ```py
    >>> from peft import PrefixEncoder, PrefixTuningConfig

    >>> config = PrefixTuningConfig(
    ...     peft_type="PREFIX_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_hidden_size=768,
    ... )
    >>> prefix_encoder = PrefixEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The two-layer MLP to transform the prefix embeddings if
          `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (`batch_size`, `num_virtual_tokens`)

    Output shape: (`batch_size`, `num_virtual_tokens`, `2*layers*hidden`)
    """

    def __init__(self, config):
        """
        Initialize the prefix encoder.

        Args:
            config: PrefixTuningConfig object
        """

    def forward(self, prefix: torch.Tensor):
        """
        Generate prefix key-value embeddings for all layers.

        Args:
            prefix: Prefix token indices

        Returns:
            Past key-value embeddings for transformer layers
        """
</syntaxhighlight>

==== Import ====
<syntaxhighlight lang="python">
from peft import PrefixEncoder, PrefixTuningConfig
</syntaxhighlight>

=== I/O Contract ===

==== Constructor Parameters ====
{| class="wikitable"
! Parameter !! Type !! Description
|-
| config || PrefixTuningConfig || Configuration object specifying encoder architecture
|}

==== Forward Method ====

===== Inputs =====
{| class="wikitable"
! Parameter !! Type !! Shape !! Description
|-
| prefix || torch.Tensor || (batch_size, num_virtual_tokens) || Indices for prefix tokens
|}

===== Returns =====
{| class="wikitable"
! Type !! Shape !! Description
|-
| torch.Tensor || (batch_size, num_virtual_tokens, 2*num_layers*token_dim) || Past key-value embeddings for all transformer layers
|}

==== Model Attributes ====
{| class="wikitable"
! Attribute !! Type !! Description
|-
| prefix_projection || bool || Whether to use MLP projection for prefix embeddings
|-
| embedding || torch.nn.Embedding || Embedding layer for prefix tokens
|-
| transform || torch.nn.Sequential || Two-layer MLP for reparameterization (only if prefix_projection=True)
|}

==== Side Effects ====
* Creates trainable embedding parameters
* Creates trainable MLP parameters if prefix_projection is enabled

=== Usage Examples ===

==== Basic Prefix Encoder Without Projection ====
<syntaxhighlight lang="python">
import torch
from peft import PrefixEncoder, PrefixTuningConfig

# Configuration without projection
config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_layers=12,
    prefix_projection=False
)

# Create prefix encoder
prefix_encoder = PrefixEncoder(config)

# Generate prefix embeddings
batch_size = 4
prefix_indices = torch.arange(20).unsqueeze(0).expand(batch_size, -1)
past_key_values = prefix_encoder(prefix_indices)

# Output shape: (batch_size, num_virtual_tokens, 2*num_layers*token_dim)
print(past_key_values.shape)  # torch.Size([4, 20, 18432])
# 18432 = 2 * 12 * 768 (keys and values for 12 layers, each 768-dim)
</syntaxhighlight>

==== Prefix Encoder With MLP Projection ====
<syntaxhighlight lang="python">
import torch
from peft import PrefixEncoder, PrefixTuningConfig

# Configuration with MLP projection (recommended)
config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_layers=12,
    encoder_hidden_size=512,
    prefix_projection=True
)

# Create prefix encoder
prefix_encoder = PrefixEncoder(config)

# Generate prefix embeddings
batch_size = 4
prefix_indices = torch.arange(20).unsqueeze(0).expand(batch_size, -1)
past_key_values = prefix_encoder(prefix_indices)

print(past_key_values.shape)  # torch.Size([4, 20, 18432])
</syntaxhighlight>

==== Encoder-Decoder Model (Seq2Seq) ====
<syntaxhighlight lang="python">
import torch
from peft import PrefixEncoder, PrefixTuningConfig

# Configuration for seq2seq models
config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=2,  # Encoder + Decoder
    num_attention_heads=12,
    num_layers=12,
    encoder_hidden_size=768,
    prefix_projection=True
)

prefix_encoder = PrefixEncoder(config)

# Forward pass
batch_size = 4
prefix_indices = torch.arange(20).unsqueeze(0).expand(batch_size, -1)
past_key_values = prefix_encoder(prefix_indices)

print(past_key_values.shape)  # torch.Size([4, 20, 18432])
</syntaxhighlight>

==== Integration with PEFT Model ====
<syntaxhighlight lang="python">
from peft import get_peft_model, PrefixTuningConfig
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Create prefix tuning configuration
config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=30,
    encoder_hidden_size=512,
    prefix_projection=True
)

# Apply prefix tuning (PrefixEncoder is created internally)
peft_model = get_peft_model(model, config)

# Check trainable parameters
peft_model.print_trainable_parameters()
# Output shows only prefix parameters are trainable
</syntaxhighlight>

==== Inspecting Model Architecture ====
<syntaxhighlight lang="python">
import torch
from peft import PrefixEncoder, PrefixTuningConfig

# Configuration with projection
config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_layers=12,
    encoder_hidden_size=512,
    prefix_projection=True
)

prefix_encoder = PrefixEncoder(config)

print("Prefix projection enabled:", prefix_encoder.prefix_projection)
print("\nEmbedding layer:", prefix_encoder.embedding)
print("\nTransform MLP:")
for i, layer in enumerate(prefix_encoder.transform):
    print(f"  Layer {i}: {layer}")

# Output:
# Prefix projection enabled: True
# Embedding layer: Embedding(20, 768)
# Transform MLP:
#   Layer 0: Linear(in_features=768, out_features=512, bias=True)
#   Layer 1: Tanh()
#   Layer 2: Linear(in_features=512, out_features=18432, bias=True)
</syntaxhighlight>

==== Understanding the Output Shape ====
<syntaxhighlight lang="python">
import torch
from peft import PrefixEncoder, PrefixTuningConfig

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,
    token_dim=768,
    num_layers=6,
    prefix_projection=False
)

prefix_encoder = PrefixEncoder(config)

# Forward pass
batch_size = 2
prefix = torch.arange(10).unsqueeze(0).expand(batch_size, -1)
output = prefix_encoder(prefix)

print(f"Output shape: {output.shape}")
# Output: torch.Size([2, 10, 9216])

# Breaking down the output dimension:
num_layers = 6
token_dim = 768
output_dim = 2 * num_layers * token_dim  # 2 for keys and values
print(f"Output dimension breakdown: 2 × {num_layers} × {token_dim} = {output_dim}")
# Output: Output dimension breakdown: 2 × 6 × 768 = 9216

# This output is then reshaped and split to create past_key_values
# for each of the 6 transformer layers
</syntaxhighlight>

==== Comparing Direct vs Projected Embeddings ====
<syntaxhighlight lang="python">
import torch
from peft import PrefixEncoder, PrefixTuningConfig

# Direct embedding (no projection)
config_direct = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_layers=12,
    prefix_projection=False
)

encoder_direct = PrefixEncoder(config_direct)
print("Direct embedding size:", encoder_direct.embedding.weight.shape)
# Output: torch.Size([20, 18432])
# Directly embeds to final size

# With MLP projection
config_mlp = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_layers=12,
    encoder_hidden_size=512,
    prefix_projection=True
)

encoder_mlp = PrefixEncoder(config_mlp)
print("MLP - Embedding size:", encoder_mlp.embedding.weight.shape)
print("MLP - First linear:", encoder_mlp.transform[0])
print("MLP - Second linear:", encoder_mlp.transform[2])
# Output:
# MLP - Embedding size: torch.Size([20, 768])
# MLP - First linear: Linear(in_features=768, out_features=512, bias=True)
# MLP - Second linear: Linear(in_features=512, out_features=18432, bias=True)
</syntaxhighlight>

=== Related Pages ===
* [[huggingface_peft_PrefixTuningConfig|PrefixTuningConfig]] - Configuration for this encoder
* [[huggingface_peft_PromptEncoder|PromptEncoder]] - Related P-tuning encoder
* [[huggingface_peft_MultitaskPromptTuningModel|MultitaskPromptEmbedding]] - Related multitask prompt embedding
* [[Prefix_Tuning|Prefix Tuning]]
* [[PEFT|Parameter-Efficient Fine-Tuning]]
