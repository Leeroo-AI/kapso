{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Repo|LyCORIS|https://github.com/KohakuBlueleaf/LyCORIS]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::LyCORIS]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Low-rank Kronecker product adaptation layer that decomposes weight updates using Kronecker products of smaller matrices for highly parameter-efficient fine-tuning.

=== Description ===

LoKrLayer implements adaptation using the Kronecker product of two matrices. The weight update is decomposed as W = kron(W1, W2) where W1 and W2 can themselves be low-rank decompositions (W1 = W1a @ W1b). This allows representing large weight updates with very few parameters since kron(A, B) expands dimensions multiplicatively. The factorization helper automatically determines optimal matrix shapes based on the decompose_factor parameter.

=== Usage ===

Use LoKr for extremely parameter-efficient adaptation, especially for large weight matrices. LoKr is part of the LyCORIS family and excels when the weight update has a Kronecker structure. It's particularly effective for Stable Diffusion and vision models. The decompose_both option allows further reduction by low-rank decomposing both Kronecker factors.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/lokr/layer.py src/peft/tuners/lokr/layer.py]
* '''Lines:''' 1-512

=== Signature ===
<syntaxhighlight lang="python">
class LoKrLayer(nn.Module, LycorisLayer):
    """
    Low-rank Kronecker product layer.

    Attributes:
        lokr_w1, lokr_w1_a, lokr_w1_b: First Kronecker factor (full or decomposed)
        lokr_w2, lokr_w2_a, lokr_w2_b: Second Kronecker factor
        lokr_t2: Tensor decomposition for effective Conv2d
    """
    adapter_layer_names = (
        "lokr_w1", "lokr_w1_a", "lokr_w1_b",
        "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2"
    )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        alpha: float,
        rank_dropout: float,
        module_dropout: float,
        init_weights: bool,
        use_effective_conv2d: bool,
        decompose_both: bool,
        decompose_factor: int,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        """Create LoKr adapter parameters."""

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """Compute Kronecker product of factors."""

def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Find optimal factorization of dimension."""

class Linear(LoKrLayer):
    """LoKr implemented in Linear layer."""

class Conv2d(LoKrLayer):
    """LoKr implemented in Conv2d layer."""

class Conv1d(LoKrLayer):
    """LoKr implemented in Conv1d layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lokr import LoKrLayer, LoKrConfig, LoKrModel
from peft import LoKrConfig, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || The pretrained layer (Linear, Conv2d, or Conv1d)
|-
| adapter_name || str || No || Name for the adapter (default: "default")
|-
| r || int || Yes || Rank for low-rank decomposition of Kronecker factors
|-
| alpha || float || No || Scaling factor
|-
| decompose_both || bool || No || Apply low-rank decomposition to both factors
|-
| decompose_factor || int || No || Target factor for dimension factorization (-1 for auto)
|-
| use_effective_conv2d || bool || No || Use tensor decomposition for Conv2d
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input + Kronecker product adaptation
|-
| get_delta_weight() || torch.Tensor || kron(W1, W2) * scaling
|}

== Usage Examples ==

=== Basic LoKr Configuration ===
<syntaxhighlight lang="python">
from peft import LoKrConfig, get_peft_model
from diffusers import UNet2DConditionModel

# Load model
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet"
)

# Configure LoKr
config = LoKrConfig(
    r=8,
    alpha=16,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    decompose_both=False,  # Only decompose second factor
    decompose_factor=-1,   # Auto-detect best factorization
)

unet = get_peft_model(unet, config)
</syntaxhighlight>

=== LoKr with Full Decomposition ===
<syntaxhighlight lang="python">
from peft import LoKrConfig, get_peft_model

# Maximum parameter efficiency with decompose_both
config = LoKrConfig(
    r=4,
    alpha=8,
    target_modules=["to_q", "to_v"],
    decompose_both=True,   # Decompose both Kronecker factors
    decompose_factor=8,    # Target factor for factorization
)

model = get_peft_model(model, config)
# Even fewer parameters than standard LoKr
</syntaxhighlight>

=== LoKr for Vision Transformers ===
<syntaxhighlight lang="python">
from peft import LoKrConfig, get_peft_model
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

config = LoKrConfig(
    r=8,
    alpha=16,
    target_modules=["query", "value"],
    rank_dropout=0.1,
)

model = get_peft_model(model, config)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
