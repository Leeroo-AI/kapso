{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for instantiating an uninitialized model architecture on the meta device from a configuration object provided by the HuggingFace Transformers library.

=== Description ===

`PreTrainedModel.__init__()` combined with PyTorch's `init_empty_weights()` context manager creates a model instance without allocating actual memory for parameters. This is accomplished using PyTorch's meta device, which tracks tensor shapes and types without storing data. The model's architecture (layers, connections, modules) is fully constructed according to the configuration, but all parameters exist as metadata only. This pattern enables memory-efficient model loading where weights can be loaded incrementally, quantized on-the-fly, or distributed across devices without ever instantiating the full model in standard memory.

=== Usage ===

Use this when you need to:
* Load large models that exceed single-device memory capacity
* Implement custom weight loading logic that operates layer-by-layer
* Support quantization workflows that require uninitialized target tensors
* Create device mapping strategies that need architecture information before weight loading
* Build model inspection tools that analyze architecture without loading weights

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/modeling_utils.py (lines 1308-1383)

=== Signature ===
<syntaxhighlight lang="python">
class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        """
        Initialize a PreTrainedModel from a configuration.

        Args:
            config (PreTrainedConfig): Model configuration object containing
                architecture parameters (hidden_size, num_layers, etc.)
            *inputs: Additional positional arguments (typically unused)
            **kwargs: Additional keyword arguments (typically unused)

        Returns:
            None (constructor modifies self in-place)

        Notes:
            - When used with torch.device("meta") or accelerate.init_empty_weights(),
              creates model structure without allocating parameter memory
            - Sets self.config to the provided configuration
            - Initializes generation_config if model supports generation
            - Calls post_init() which handles weight initialization if not on meta device
        """

# Usage pattern for empty initialization:
from accelerate import init_empty_weights

with init_empty_weights():
    model = AutoModel.from_config(config)
    # model now exists with architecture but no weight data
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoModel, AutoConfig
from accelerate import init_empty_weights
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| config || PreTrainedConfig || Yes || Configuration object specifying model architecture (hidden_size, num_layers, attention_heads, etc.)
|-
| torch_dtype || torch.dtype | None || No || Data type for model parameters (e.g., torch.float16, torch.bfloat16)
|-
| attn_implementation || str | None || No || Attention implementation to use ("eager", "sdpa", "flash_attention_2")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel subclass || Instantiated model architecture with uninitialized parameters (when used with init_empty_weights context)
|}

'''State After Initialization (with init_empty_weights):'''
* All parameters exist as meta tensors (shape and dtype tracked, no data allocated)
* Model architecture fully constructed (layers, attention, embeddings)
* Configuration attached to model.config
* Device map not yet applied
* Weights not yet loaded

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import AutoConfig, AutoModel
from accelerate import init_empty_weights
import torch

# Load configuration
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create empty model structure without allocating memory
with init_empty_weights():
    model = AutoModel.from_config(config)

print(f"Model class: {type(model).__name__}")
print(f"Config: {model.config.hidden_size} hidden size")

# Check that parameters are on meta device
for name, param in model.named_parameters():
    print(f"{name}: {param.device}, {param.shape}")
    break  # Just show first parameter

# Example: bert-base-uncased
config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")

with init_empty_weights():
    bert_model = AutoModel.from_config(config)
    num_params = sum(p.numel() for p in bert_model.parameters())
    print(f"BERT parameters: {num_params:,}")
    # Shows parameter count without allocating ~440MB of memory

# With dtype specification
with init_empty_weights():
    fp16_model = AutoModel.from_config(
        config,
        torch_dtype=torch.float16
    )
    # Model parameters will be float16 when loaded

# With attention implementation
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
with init_empty_weights():
    flash_model = AutoModel.from_config(
        config,
        attn_implementation="flash_attention_2"
    )

# Typical usage in model loading pipeline
from transformers.utils import hub
from accelerate import infer_auto_device_map

config = AutoConfig.from_pretrained("bigscience/bloom-7b1")

# Create empty model
with init_empty_weights():
    model = AutoModel.from_config(config, torch_dtype=torch.float16)

# Infer device map based on architecture (doesn't need actual weights)
device_map = infer_auto_device_map(
    model,
    max_memory={0: "10GB", 1: "10GB", "cpu": "30GB"}
)

print(f"Device map: {device_map}")
# Now can load weights according to device map
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Model_Instantiation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
