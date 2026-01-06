{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|BOFT|https://huggingface.co/papers/2311.06243]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Orthogonal_Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Butterfly Orthogonal Fine-Tuning layer that applies orthogonal transformations via Cayley parameterization and butterfly factorization for efficient parameter updates.

=== Description ===

BOFTLayer implements the BOFT (Butterfly Orthogonal Fine-Tuning) method which uses butterfly factorization to efficiently represent orthogonal transformations. The key insight is that orthogonal matrices can be factorized into products of simpler block-diagonal matrices using butterfly permutations, dramatically reducing parameters while maintaining the orthogonality constraint. The layer uses Cayley parameterization to ensure orthogonality and supports optional CUDA acceleration via a custom kernel for fast block-diagonal operations.

=== Usage ===

Use BOFT when you want to apply orthogonal transformations to model weights with fewer parameters than OFT. BOFT is particularly effective for vision models and when you need to preserve the geometry of the weight space. The butterfly factorization allows scaling to larger models while maintaining orthogonality.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/layer.py src/peft/tuners/boft/layer.py]
* '''Lines:''' 1-1011

=== Signature ===
<syntaxhighlight lang="python">
class BOFTLayer(BaseTunerLayer):
    """
    Implements the BOFT layer.

    Attributes:
        boft_R: Trainable rotation parameters (nn.ParameterDict)
        boft_s: Trainable scaling parameters (nn.ParameterDict)
        boft_block_size: Block size per adapter (dict)
        boft_block_num: Number of blocks per adapter (dict)
        boft_dropout: Multiplicative dropout layers (nn.ModuleDict)
    """
    adapter_layer_names = ("boft_R", "boft_s")
    other_param_names = ("boft_block_size", "boft_block_num", "boft_dropout")

    def update_layer(
        self,
        adapter_name: str,
        boft_block_size: int,
        boft_block_num: int,
        boft_n_butterfly_factor: int,
        boft_dropout: float,
        init_weights: bool,
        inference_mode: bool = False,
        **kwargs,
    ):
        """Update the layer with BOFT adapter parameters."""

    def cayley_batch(self, data: torch.Tensor) -> torch.Tensor:
        """Apply Cayley parameterization to ensure orthogonality."""

class Linear(nn.Module, BOFTLayer):
    """BOFT implemented in a dense layer."""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        boft_block_size: int = 8,
        boft_block_num: int = 0,
        boft_n_butterfly_factor: int = 0,
        boft_dropout: float = 0.1,
        fan_in_fan_out: bool = False,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initialize BOFT Linear layer."""

class Conv2d(nn.Module, BOFTLayer):
    """BOFT implemented in a Conv2d layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.boft import BOFTLayer, BOFTConfig, BOFTModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained layer to adapt (Linear or Conv2d)
|-
| adapter_name || str || Yes || Name identifier for the adapter
|-
| boft_block_size || int || Yes || Size of each orthogonal block (must divide in_features)
|-
| boft_block_num || int || Yes || Number of blocks (alternative to block_size, set one to 0)
|-
| boft_n_butterfly_factor || int || No || Number of butterfly factors (default: 1)
|-
| boft_dropout || float || No || Multiplicative dropout probability (default: 0.1)
|-
| init_weights || bool || No || Whether to initialize weights to identity (default: True)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input transformed by orthogonal rotation and scaling
|-
| get_delta_weight() || tuple || (butterfly_oft_mat, boft_s) transformation components
|}

== Usage Examples ==

=== Basic BOFT Configuration ===
<syntaxhighlight lang="python">
from peft import BOFTConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure BOFT
config = BOFTConfig(
    boft_block_size=8,          # Size of orthogonal blocks
    boft_n_butterfly_factor=1,   # Butterfly factorization depth
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    boft_dropout=0.1,
    bias="none",
)

# Create PEFT model
model = get_peft_model(model, config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")
</syntaxhighlight>

=== BOFT for Vision Models ===
<syntaxhighlight lang="python">
from peft import BOFTConfig, get_peft_model
from transformers import AutoModelForImageClassification

# Load vision model
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

# BOFT works well for vision models
config = BOFTConfig(
    boft_block_num=4,           # Use block_num instead of block_size
    boft_n_butterfly_factor=2,  # More butterfly factors for larger models
    target_modules=["query", "value"],
    boft_dropout=0.05,
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Merging BOFT Weights ===
<syntaxhighlight lang="python">
# After training, merge BOFT into base weights
model.merge_and_unload()

# Or merge with safety check
model.merge_adapter(safe_merge=True)

# Save merged model
model.save_pretrained("./boft_merged_model")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
