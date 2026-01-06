{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Block_LoRA]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Gradient-aware Low-Rank Adaptation layer that applies block-wise LoRA with information exchange between blocks and optional hybrid vanilla LoRA component.

=== Description ===

GraLoRALayer implements a block-wise variant of LoRA where the weight matrix is divided into k blocks along both input and output dimensions. Each block has its own low-rank adaptation matrices, and GraLoRA enables information exchange between blocks through its specialized attention-like computation. It also supports a hybrid mode where a vanilla LoRA component is added for global information capture alongside the block-wise adaptation.

=== Usage ===

Use GraLoRA when you want block-wise adaptation with information sharing between blocks. The method is particularly useful when you want to capture both local (block-wise) and global (hybrid LoRA) patterns in the weight updates. The k parameter controls the number of blocks, and hybrid_r adds a vanilla LoRA component.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/gralora/layer.py src/peft/tuners/gralora/layer.py]
* '''Lines:''' 1-393

=== Signature ===
<syntaxhighlight lang="python">
class GraloraLayer(BaseTunerLayer):
    """
    Gradient-aware LoRA layer with block-wise adaptation.

    Attributes:
        gralora_A: Block-wise A matrices (nn.ParameterDict)
        gralora_B: Block-wise B matrices (nn.ParameterDict)
        gralora_A_general: Hybrid LoRA A component (nn.ModuleDict)
        gralora_B_general: Hybrid LoRA B component (nn.ModuleDict)
        gralora_k: Number of blocks per adapter (dict)
        hybrid_r: Rank of hybrid LoRA component (dict)
    """
    adapter_layer_names = ("gralora_A", "gralora_B", "gralora_A_general", "gralora_B_general")
    other_param_names = ("r", "hybrid_r", "alpha", "scaling", "gralora_dropout")

    def update_layer(
        self,
        adapter_name: str,
        module_name: str,
        r: int,
        alpha: int,
        gralora_dropout: float,
        gralora_k: int = 2,
        hybrid_r: int = 0,
        init_weights: bool = True,
    ):
        """Update layer with GraLoRA parameters."""

class Linear(nn.Linear, GraloraLayer):
    """GraLoRA implemented in a dense layer."""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        module_name: str,
        r: int = 0,
        alpha: int = 1,
        gralora_dropout: float = 0.0,
        gralora_k: int = 2,
        hybrid_r: int = 0,
        fan_in_fan_out: bool = False,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initialize GraLoRA Linear layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.gralora import GraloraLayer, GraLoRAConfig, GraLoRAModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained linear layer to adapt
|-
| adapter_name || str || Yes || Name identifier for the adapter
|-
| module_name || str || Yes || Name of the module being adapted
|-
| r || int || Yes || Rank for block-wise LoRA (divided among k blocks)
|-
| alpha || int || No || Scaling factor (default: 1)
|-
| gralora_k || int || No || Number of blocks (default: 2)
|-
| hybrid_r || int || No || Rank for vanilla LoRA component (default: 0)
|-
| gralora_dropout || float || No || Dropout probability (default: 0.0)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input transformed by base layer + block-wise LoRA + hybrid LoRA
|-
| get_delta_weight() || torch.Tensor || Combined delta weight from blocks and hybrid component
|}

== Usage Examples ==

=== Basic GraLoRA Configuration ===
<syntaxhighlight lang="python">
from peft import GraLoRAConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure GraLoRA
config = GraLoRAConfig(
    r=32,                  # Total rank (divided among k blocks)
    gralora_k=4,           # Number of blocks
    alpha=64,              # Scaling factor
    target_modules=["q_proj", "v_proj"],
    gralora_dropout=0.05,
)

# Create PEFT model
model = get_peft_model(model, config)
</syntaxhighlight>

=== GraLoRA with Hybrid Component ===
<syntaxhighlight lang="python">
from peft import GraLoRAConfig, get_peft_model

# Add hybrid vanilla LoRA for global patterns
config = GraLoRAConfig(
    r=32,                  # Block-wise rank
    hybrid_r=16,           # Additional vanilla LoRA rank
    gralora_k=4,           # 4 blocks
    alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, config)
# Effective total rank: 32 (blocks) + 16 (hybrid) = 48
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
