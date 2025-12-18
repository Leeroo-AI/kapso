{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|MiSS|https://arxiv.org/abs/2503.01944]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Sparse_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Mixed-rank Sparse Scaling layer that applies block-wise additive or multiplicative adaptation with minimal parameters for efficient fine-tuning.

=== Description ===

MissLayer implements Multiple Initialization Schemes with Sparsity (MiSS) for parameter-efficient adaptation. The layer supports three initialization modes: standard MiSS adds a sparse block matrix to reshaped inputs, BAT (Block Additive Transformation) applies block-diagonal transformations, and mini uses repeated small matrices. The adapter modifies outputs by reshaping inputs into blocks, applying the learned sparse transformation, and summing back to the original shape.

=== Usage ===

Use MiSS when you need extremely parameter-efficient adaptation. MiSS can be more efficient than LoRA for certain architectures due to its block-sparse structure. The BAT variant provides multiplicative adaptation while standard MiSS is additive. The mini variant is useful when output features should share parameters.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/miss/layer.py src/peft/tuners/miss/layer.py]
* '''Lines:''' 1-394

=== Signature ===
<syntaxhighlight lang="python">
class MissLayer(BaseTunerLayer):
    """
    Mixed-rank Sparse Scaling layer.

    Attributes:
        miss_block: ParameterDict of sparse adaptation matrices
        miss_r: Dict of rank values per adapter
        miss_mini_r: Dict of mini rank values per adapter
        miss_dropout: ModuleDict of dropout layers
    """
    adapter_layer_names = ("miss_block",)
    other_param_names = ("miss_r", "miss_dropout", "miss_mini_r")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        mini_r: int,
        miss_dropout: float,
        init_weights: bool | str,
        **kwargs,
    ) -> None:
        """Create MiSS adapter with specified initialization."""

    def reset_bat_parameters(self, adapter_name: str, r: int):
        """Initialize BAT block-diagonal parameters."""

    def reset_mini_parameters(self, adapter_name: str, r: int, mini_r: int):
        """Initialize mini shared parameters."""

class MissLinear(nn.Module, MissLayer):
    """MiSS implemented in Linear layer."""

    def get_delta_weight(self, adapter, orig_weight, re: bool = False):
        """Compute BAT delta weight transformation."""

    def get_delta_weight_miss(self, adapter, orig_weight, re: bool = False):
        """Compute standard/mini MiSS delta weight."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.miss import MissLayer, MissConfig, MissModel
from peft import MissConfig, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained Linear layer
|-
| adapter_name || str || Yes || Name for the adapter
|-
| r || int || Yes || Rank/block size for adaptation
|-
| mini_r || int || No || Mini rank for shared parameters
|-
| miss_dropout || float || No || Dropout probability
|-
| init_weights || bool/str || Yes || "bat", "mini", or True for standard
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input + sparse block adaptation
|-
| get_delta_weight() || torch.Tensor || Block transformation for merging
|}

== Usage Examples ==

=== Standard MiSS Configuration ===
<syntaxhighlight lang="python">
from peft import MissConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Standard MiSS with additive blocks
config = MissConfig(
    r=8,                        # Block size
    target_modules=["q_proj", "v_proj"],
    miss_dropout=0.0,
    init_weights=True,          # Standard initialization
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
</syntaxhighlight>

=== BAT (Block Additive Transformation) ===
<syntaxhighlight lang="python">
from peft import MissConfig, get_peft_model

# BAT provides multiplicative block adaptation
config = MissConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    init_weights="bat",         # Block Additive Transformation
)

model = get_peft_model(model, config)
# BAT: W' = W @ block + block (block-wise transformation)
</syntaxhighlight>

=== Mini MiSS with Shared Parameters ===
<syntaxhighlight lang="python">
from peft import MissConfig, get_peft_model

# Mini variant shares small matrix across output
config = MissConfig(
    r=16,
    mini_r=64,                  # Small shared matrix size
    target_modules=["q_proj", "v_proj"],
    init_weights="mini",        # Use mini initialization
)

model = get_peft_model(model, config)
# Even fewer parameters than standard MiSS
</syntaxhighlight>

=== MiSS Forward Computation ===
<syntaxhighlight lang="python">
# For standard MiSS:
# 1. Reshape input: x -> [batch, seq, in_features//r, r]
# 2. Sum over blocks: sum(x, dim=-2) -> [batch, seq, r]
# 3. Apply transformation: summed @ miss_block -> [batch, seq, out_features]
# 4. Add to base output: result = base(x) + transformed

# For BAT:
# 1. Reshape weight into r x r blocks
# 2. Apply learned block transformation
# 3. Reconstruct transformed weight
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
