= TrainableTokensLayer =

== Knowledge Sources ==
* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* Source: src/peft/tuners/trainable_tokens/layer.py

== Domains ==
* [[Natural Language Processing (NLP)]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Token Embeddings]]
* [[Neural Network Layers]]
* [[Memory Optimization]]

== Overview ==

=== Description ===
TrainableTokensLayer is a layer implementation that wraps embedding layers (nn.Embedding) or linear layers (nn.Linear) to enable selective training of specific token embeddings. Instead of training the entire embedding matrix, this layer maintains trainable delta weights only for specified token indices, significantly reducing memory requirements.

The layer stores both the updated weights (trainable_tokens_delta) and original weights (trainable_tokens_original) for the specified tokens, allowing for efficient merging/unmerging operations. It supports weight tying scenarios where multiple layers share the same embeddings.

=== Usage ===
TrainableTokensLayer is automatically instantiated when applying TrainableTokensConfig to a model. It intercepts forward passes through embedding layers and applies the trainable token updates using index_copy operations. The layer supports both nn.Embedding and nn.Linear base layers to handle tied weights scenarios.

== Code Reference ==

=== Source Location ===
File: src/peft/tuners/trainable_tokens/layer.py
Lines: 30-263

=== Class Signature ===
<syntaxhighlight lang="python">
class TrainableTokensLayer(nn.Module, BaseTunerLayer):
    """
    Layer for training specific tokens efficiently.

    Args:
        base_layer: The embedding or linear layer to wrap
        adapter_name: Name of the adapter
        token_indices: List of token indices to make trainable
        tied_adapter: Optional tied adapter for weight sharing
    """
    adapter_layer_names = ("trainable_tokens_delta",)
    other_param_names = ("token_indices", "trainable_tokens_original")
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| base_layer || nn.Module || The base embedding or linear layer to wrap
|-
| adapter_name || str || Name of the adapter
|-
| token_indices || list[int] || List of token indices to make trainable
|-
| tied_adapter || Optional[TrainableTokensLayer] || Tied adapter for weight sharing (default: None)
|-
| kwargs || dict || Additional keyword arguments
|}

=== Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| trainable_tokens_delta || nn.ParameterDict || Trainable delta weights for specified tokens
|-
| trainable_tokens_original || BufferDict || Original embeddings for specified tokens
|-
| token_indices || dict || Mapping of adapter names to token indices
|-
| merged_adapters || list || List of currently merged adapter names
|-
| tied_adapter || Optional[TrainableTokensLayer] || Reference to tied adapter if applicable
|}

=== Key Methods ===
{| class="wikitable"
! Method !! Parameters !! Returns !! Description
|-
| update_layer || adapter_name, **kwargs || None || Update layer with new adapter configuration
|-
| merge || safe_merge, adapter_names || None || Merge token updates into base weights
|-
| unmerge || None || None || Restore original token embeddings
|-
| get_merged_weights || active_adapters || torch.Tensor || Get weights with adapters applied
|-
| forward || x, *args, **kwargs || torch.Tensor || Forward pass with trainable tokens
|-
| forward_adapters || x, active_adapters, *args, **kwargs || torch.Tensor || Forward with specific adapters
|}

== Usage Examples ==

=== Basic Usage with Model ===
<syntaxhighlight lang="python">
from peft import TrainableTokensConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Add new tokens
new_tokens = ["<task>", "<domain>"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# Configure trainable tokens
token_indices = [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
config = TrainableTokensConfig(token_indices=token_indices)

# Apply configuration (creates TrainableTokensLayer)
model = get_peft_model(model, config)
</syntaxhighlight>

=== Accessing the Layer ===
<syntaxhighlight lang="python">
# Access the trainable tokens layer
embedding_layer = model.get_input_embeddings()

# Check if it's a TrainableTokensLayer
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
if isinstance(embedding_layer, TrainableTokensLayer):
    print("TrainableTokensLayer detected")
    print(f"Token indices: {embedding_layer.token_indices}")
</syntaxhighlight>

=== Merging and Unmerging ===
<syntaxhighlight lang="python">
# Get embedding layer
embedding_layer = model.get_input_embeddings()

# Merge trainable tokens into base embeddings
embedding_layer.merge()

# Check merged status
print(f"Is merged: {embedding_layer.merged}")

# Unmerge to restore original behavior
embedding_layer.unmerge()
</syntaxhighlight>

=== Inspecting Trainable Token Weights ===
<syntaxhighlight lang="python">
# Access trainable token deltas
for adapter_name, delta in embedding_layer.trainable_tokens_delta.items():
    print(f"Adapter: {adapter_name}")
    print(f"Shape: {delta.shape}")  # (num_tokens, embedding_dim)
    print(f"Trainable: {delta.requires_grad}")

# Access original weights
for adapter_name, original in embedding_layer.trainable_tokens_original.items():
    print(f"Original weights for {adapter_name}: {original.shape}")
</syntaxhighlight>

=== Checking for Overlapping Tokens ===
<syntaxhighlight lang="python">
# The layer automatically checks for overlapping token indices
# when multiple adapters are used
try:
    embedding_layer.merge(adapter_names=["adapter1", "adapter2"])
except ValueError as e:
    print(f"Overlap detected: {e}")
</syntaxhighlight>

=== Forward Pass with Multiple Adapters ===
<syntaxhighlight lang="python">
# Set active adapters
model.set_adapter(["adapter1", "adapter2"])

# Forward pass uses all active adapters
input_ids = torch.randint(0, len(tokenizer), (2, 10))
outputs = model(input_ids)
</syntaxhighlight>

=== Safe Merging with NaN Detection ===
<syntaxhighlight lang="python">
# Merge with safety check
try:
    embedding_layer.merge(safe_merge=True)
    print("Merge successful, no NaNs detected")
except ValueError as e:
    print(f"Merge failed: {e}")
</syntaxhighlight>

=== Working with Tied Weights ===
<syntaxhighlight lang="python">
# When model has tied embeddings (input + output)
# The layer automatically handles weight tying

# Check if tied
if embedding_layer.tied_adapter is not None:
    print("This layer is tied to another adapter")
    print(f"Tied to: {embedding_layer.tied_adapter}")
</syntaxhighlight>

=== DeepSpeed Zero3 Support ===
<syntaxhighlight lang="python">
# The layer automatically handles DeepSpeed Zero3 context
# when collecting token weights during initialization

# Check if DeepSpeed is enabled
from peft.utils.integrations import check_deepspeed_zero3_enabled
if check_deepspeed_zero3_enabled():
    print("DeepSpeed Zero3 detected - using distributed weight collection")
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_TrainableTokensConfig|TrainableTokensConfig]] - Configuration class
* [[huggingface_peft_TrainableTokensModel|TrainableTokensModel]] - Model class
* [[Token Embeddings]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Memory Optimization]]

[[Category:Machine Learning]]
[[Category:PEFT]]
[[Category:Neural Network Layers]]
[[Category:NLP]]
[[Category:HuggingFace]]
