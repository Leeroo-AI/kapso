{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Model Loading Documentation|https://huggingface.co/docs/transformers/main_classes/model]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for constructing neural network architectures from configuration objects provided by HuggingFace Transformers.

=== Description ===
PreTrainedModel.__init__ is the base constructor for all transformer models in the HuggingFace Transformers library. It initializes the fundamental model structure by storing the configuration, validating and setting up the attention implementation (checking hardware compatibility for FlashAttention, SDPA, etc.), and preparing the model for subsequent weight loading. The constructor handles critical setup tasks including checking if the model can generate text (and attaching appropriate generation configurations), validating that attention mechanisms are supported on the current hardware, and establishing the foundation for the module hierarchy that specific model classes will build upon.

This implementation is the entry point for all model instantiation in Transformers and is always called by model-specific subclass constructors.

=== Usage ===
PreTrainedModel.__init__ is called automatically when instantiating any transformer model, either directly (MyModel(config)) or indirectly through from_pretrained(). While rarely called directly by end users, understanding this constructor is important for implementing custom model architectures or debugging model initialization issues.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/modeling_utils.py
* '''Lines:''' 1308-1350 (constructor), 1136-1200 (class definition)

=== Signature ===
<syntaxhighlight lang="python">
def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
    super().__init__()
    # Validation and setup logic
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import BertModel, BertConfig
from transformers.modeling_utils import PreTrainedModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| config || PreTrainedConfig || Yes || Configuration object containing all model hyperparameters
|-
| *inputs || tuple || No || Legacy positional arguments (deprecated)
|-
| **kwargs || dict || No || Additional initialization arguments
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Initialized model instance with architecture defined but weights uninitialized
|}

== Usage Examples ==

=== Creating Model from Configuration ===
<syntaxhighlight lang="python">
from transformers import BertConfig, BertModel

# Create configuration
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512
)

# Instantiate model with random weights
model = BertModel(config)

print(f"Model has {model.num_parameters():,} parameters")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Number of layers: {model.config.num_hidden_layers}")
</syntaxhighlight>

=== Creating Custom Architecture ===
<syntaxhighlight lang="python">
from transformers import GPT2Config, GPT2LMHeadModel

# Custom small GPT-2 configuration
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=512,      # Smaller than default
    n_layer=6,       # Fewer layers than default
    n_head=8,
    n_inner=2048
)

# Instantiate with custom config
model = GPT2LMHeadModel(config)

print(f"Custom model size: {model.num_parameters():,} parameters")
</syntaxhighlight>

=== Using Attention Implementations ===
<syntaxhighlight lang="python">
from transformers import LlamaConfig, LlamaForCausalLM
import torch

# Configure for FlashAttention 2
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    attn_implementation="flash_attention_2"  # Requires flash-attn package
)

# Model will validate FlashAttention 2 is available during __init__
model = LlamaForCausalLM(config)

# Alternative: Use PyTorch SDPA (Scaled Dot Product Attention)
config_sdpa = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    attn_implementation="sdpa"  # Uses torch.nn.functional.scaled_dot_product_attention
)

model_sdpa = LlamaForCausalLM(config_sdpa)
</syntaxhighlight>

=== Meta Device Initialization for Large Models ===
<syntaxhighlight lang="python">
from transformers import LlamaConfig, LlamaForCausalLM
import torch

# Configure large model
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=8192,
    num_hidden_layers=80,
    num_attention_heads=64
)

# Initialize on meta device (no memory allocation)
with torch.device("meta"):
    model = LlamaForCausalLM(config)

# Model structure is created but no actual memory used
print(f"Model on meta device: {model.device}")
print(f"Parameters are not materialized yet")

# Later load weights using device_map to materialize tensors
# model = LlamaForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-70b-hf",
#     device_map="auto"
# )
</syntaxhighlight>

=== Checking Attention Implementation Support ===
<syntaxhighlight lang="python">
from transformers import AutoConfig, AutoModel
import torch

config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# Try different attention implementations
for attn_impl in ["eager", "sdpa", "flash_attention_2"]:
    try:
        config.attn_implementation = attn_impl
        model = AutoModel.from_config(config)
        print(f"{attn_impl}: Supported ✓")
        del model
    except (ValueError, ImportError) as e:
        print(f"{attn_impl}: Not supported - {str(e)[:50]}...")
</syntaxhighlight>

=== Creating Model for Training from Scratch ===
<syntaxhighlight lang="python">
from transformers import GPT2Config, GPT2LMHeadModel
import torch

# Create fresh config for training from scratch
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)

# Instantiate model with random initialization
model = GPT2LMHeadModel(config)

# Model is ready for training
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop would go here
print(f"Model ready for training with {model.num_parameters():,} parameters")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Model_Instantiation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Loading_Environment]]
