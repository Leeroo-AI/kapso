{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Model_Architecture]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Comprehensive Rotary Position Embedding (RoPE) implementation supporting multiple scaling strategies including linear, dynamic NTK, YaRN, LongRoPE, and Llama3-specific variants for extending context length in transformer models.

=== Description ===
The modeling_rope_utils module provides a unified framework for computing RoPE inverse frequencies across different scaling methods. It includes the dynamic_rope_update decorator that automatically recomputes frequencies during forward passes for dynamic RoPE implementations, the RotaryEmbeddingConfigMixin class for validating and standardizing RoPE configuration parameters, and computation functions for six RoPE variants: default, linear scaling, dynamic NTK, YaRN, LongRoPE, and Llama3. Each variant implements a specific frequency scaling strategy to extend context length beyond pre-training limits. The module handles per-layer RoPE configurations in hybrid models, supports partial rotary embeddings, and provides comprehensive validation of RoPE parameters with detailed error messages.

=== Usage ===
Use this module when implementing or configuring transformer models with rotary position embeddings. Apply the @dynamic_rope_update decorator to RoPE forward methods in models that need dynamic frequency updates based on sequence length. Use RotaryEmbeddingConfigMixin in model configuration classes to standardize and validate rope_parameters dictionaries. Call the appropriate ROPE_INIT_FUNCTIONS based on rope_type to compute inverse frequencies and attention scaling factors during model initialization or inference.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' src/transformers/modeling_rope_utils.py

=== Signature ===
<syntaxhighlight lang="python">
def dynamic_rope_update(rope_forward):
    """Decorator to update RoPE parameters in forward pass for dynamic RoPE"""
    @wraps(rope_forward)
    def wrapper(self, x, position_ids, layer_type=None):
        # Update frequencies if dynamic or longrope
        return rope_forward(self, x, position_ids, **kwargs)
    return wrapper

def _compute_linear_scaling_rope_parameters(
    config: "PreTrainedConfig",
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    # Compute inverse frequencies with linear scaling
    pass

def _compute_dynamic_ntk_parameters(
    config: "PreTrainedConfig",
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    # Compute inverse frequencies with dynamic NTK scaling
    pass

def _compute_yarn_parameters(
    config: "PreTrainedConfig",
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    # Compute YaRN inverse frequencies with interpolation/extrapolation
    pass

def _compute_longrope_parameters(
    config: "PreTrainedConfig",
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    # Compute LongRoPE frequencies with short/long factors
    pass

def _compute_llama3_parameters(
    config: "PreTrainedConfig",
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    # Compute Llama 3.1 RoPE with wavelen-based smoothing
    pass

ROPE_INIT_FUNCTIONS = {
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}

class RotaryEmbeddingConfigMixin:
    def convert_rope_params_to_dict(self, **kwargs):
        # Convert rope_scaling to rope_parameters dict
        pass

    def standardize_rope_params(self):
        # Standardize rope_parameters format
        pass

    def validate_rope(self, ignore_keys: Optional[set] = None):
        # Validate rope_parameters based on rope_type
        pass
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.modeling_rope_utils import (
    dynamic_rope_update,
    ROPE_INIT_FUNCTIONS,
    RotaryEmbeddingConfigMixin,
)
</syntaxhighlight>

== I/O Contract ==

=== RoPE Computation Function Inputs ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| config || PreTrainedConfig || Model configuration with rope_parameters, hidden_size, num_attention_heads, max_position_embeddings
|-
| device || torch.device || Device for tensor initialization (e.g., "cpu", "cuda")
|-
| seq_len || int (optional) || Current sequence length for dynamic scaling (used by dynamic, longrope)
|-
| layer_type || str (optional) || Layer type identifier for models with per-layer RoPE configs (e.g., "full_attention", "global_sliding")
|}

=== RoPE Computation Function Outputs ===
{| class="wikitable"
! Return !! Type !! Description
|-
| inv_freq || torch.Tensor || Inverse frequencies tensor for RoPE computation, shape (dim/2,)
|-
| attention_factor || float || Scaling factor applied to attention scores (used by yarn, longrope)
|}

=== RotaryEmbeddingConfigMixin Methods ===
{| class="wikitable"
! Method !! Inputs !! Outputs !! Description
|-
| convert_rope_params_to_dict || kwargs dict || kwargs dict || Extracts rope_scaling/rope_theta from kwargs, sets rope_parameters, validates config
|-
| standardize_rope_params || none || none || Moves rope_theta to rope_parameters, handles per-layer configs, sets rope_type defaults
|-
| validate_rope || ignore_keys set (optional) || none || Validates rope_parameters based on rope_type, raises KeyError on missing required keys
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Example 1: Apply dynamic RoPE update decorator to forward method
import torch
from transformers.modeling_rope_utils import dynamic_rope_update

class MyRoPELayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters.get("rope_type", "default")
        # Initialize inverse frequencies
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type)
        if init_fn:
            self.inv_freq, self.attention_scaling = init_fn(config, device="cpu")

    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type=None):
        # inv_freq is automatically updated based on position_ids if needed
        freqs = position_ids.float() @ self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

# Example 2: Compute linear scaling RoPE parameters
from transformers import AutoConfig
from transformers.modeling_rope_utils import _compute_linear_scaling_rope_parameters

config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config.rope_parameters = {
    "rope_type": "linear",
    "rope_theta": 10000.0,
    "factor": 2.0  # Extend context 2x
}

inv_freq, attention_factor = _compute_linear_scaling_rope_parameters(
    config=config,
    device=torch.device("cuda"),
    seq_len=None  # Not used for linear scaling
)
print(f"Inverse frequencies shape: {inv_freq.shape}")
print(f"Attention factor: {attention_factor}")

# Example 3: Configure YaRN scaling for long context
config.max_position_embeddings = 32768
config.rope_parameters = {
    "rope_type": "yarn",
    "rope_theta": 10000.0,
    "factor": 4.0,
    "original_max_position_embeddings": 8192,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 0.8,
}

from transformers.modeling_rope_utils import _compute_yarn_parameters
inv_freq_yarn, attention_factor_yarn = _compute_yarn_parameters(
    config=config,
    device=torch.device("cuda")
)

# Example 4: Use RotaryEmbeddingConfigMixin in model config
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin

class MyModelConfig(PretrainedConfig, RotaryEmbeddingConfigMixin):
    def __init__(self, rope_theta=10000.0, rope_scaling=None, **kwargs):
        super().__init__(**kwargs)
        self.rope_parameters = None
        # Convert old rope_scaling format to rope_parameters
        kwargs = self.convert_rope_params_to_dict(
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            **kwargs
        )

# Create config with validation
my_config = MyModelConfig(
    hidden_size=4096,
    num_attention_heads=32,
    max_position_embeddings=32768,
    rope_theta=10000.0,
    rope_scaling={
        "type": "linear",
        "factor": 4.0
    }
)

# Example 5: Compute LongRoPE parameters for Phi-3
config_longrope = AutoConfig.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
config_longrope.rope_parameters = {
    "rope_type": "longrope",
    "rope_theta": 10000.0,
    "factor": 16.0,
    "short_factor": [1.0] * 32,  # 32 = head_dim * partial_rotary_factor / 2
    "long_factor": [1.1] * 32,
}

from transformers.modeling_rope_utils import _compute_longrope_parameters
inv_freq_short, attn_factor = _compute_longrope_parameters(
    config=config_longrope,
    device=torch.device("cpu"),
    seq_len=2048  # Short context uses short_factor
)

inv_freq_long, attn_factor_long = _compute_longrope_parameters(
    config=config_longrope,
    device=torch.device("cpu"),
    seq_len=131072  # Long context uses long_factor
)

# Example 6: Compute Llama 3.1 RoPE with frequency smoothing
config_llama3 = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
config_llama3.rope_parameters = {
    "rope_type": "llama3",
    "rope_theta": 500000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

from transformers.modeling_rope_utils import _compute_llama3_parameters
inv_freq_llama3, _ = _compute_llama3_parameters(
    config=config_llama3,
    device=torch.device("cuda")
)

# Example 7: Use ROPE_INIT_FUNCTIONS dispatcher
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

rope_type = config.rope_parameters.get("rope_type", "default")
if rope_type in ROPE_INIT_FUNCTIONS:
    compute_fn = ROPE_INIT_FUNCTIONS[rope_type]
    inv_freq, attn_factor = compute_fn(config, device=torch.device("cuda"))
else:
    # Fall back to default RoPE
    base = config.rope_parameters["rope_theta"]
    dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
</syntaxhighlight>

== Related Pages ==
* (Empty)
