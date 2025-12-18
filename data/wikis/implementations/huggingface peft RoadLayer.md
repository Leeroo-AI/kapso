{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|RoAD|https://arxiv.org/abs/2501.00029]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Rotation_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Rotation and Dimension adaptation layer that applies efficient 2D rotations with learned angles and scales to output vectors for parameter-efficient fine-tuning.

=== Description ===

RoadLayer implements RoAD (Rotation and Dimension) adaptation by splitting output vectors into groups and applying learned 2D rotations within each group. Each pair of elements is transformed as: y0 = x0 * alpha * cos(theta) - xn * alpha * sin(theta). The method supports three variants: road_1 (reuses parameters within groups), road_2 (unique per element), and road_4 (separate parameters for each rotation component). Only angles and scales are stored, enabling efficient elementwise inference.

=== Usage ===

Use RoAD when you want extremely parameter-efficient adaptation with interpretable rotational transformations. RoAD is particularly effective for models where rotational geometry matters. The method merges by applying R @ W where R is the block-diagonal rotation matrix.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/road/layer.py src/peft/tuners/road/layer.py]
* '''Lines:''' 1-419

=== Signature ===
<syntaxhighlight lang="python">
class RoadLayer(BaseTunerLayer):
    """
    Rotation and Dimension adaptation layer.

    Attributes:
        road_theta: ParameterDict of rotation angles
        road_alpha: ParameterDict of scaling factors
        variant: Dict mapping adapter to variant ("road_1", "road_2", "road_4")
        group_size: Dict mapping adapter to group size
    """
    adapter_layer_names = ("road_theta", "road_alpha")
    other_param_names = ("variant", "group_size")

    def update_layer(
        self,
        adapter_name,
        variant,
        group_size,
        init_weights,
        **kwargs,
    ):
        """Create RoAD rotation parameters."""

class Linear(nn.Module, RoadLayer):
    """RoAD implemented in Linear layer."""

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply rotational transformation to output."""

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
        """Merge R @ W into base weights."""

def _apply_road(variant, group_size, road_theta, road_alpha, x):
    """Efficient elementwise rotation application."""

def _get_delta_weight(variant, group_size, road_theta, road_alpha):
    """Construct full rotation matrix for merging."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.road import RoadLayer, RoadConfig, RoadModel
from peft import RoadConfig, get_peft_model
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
| variant || str || Yes || "road_1", "road_2", or "road_4"
|-
| group_size || int || Yes || Size of rotation groups (must divide out_features)
|-
| init_weights || bool || No || Initialize to identity (theta=0, alpha=1)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Base output with rotational transformation
|-
| _get_delta_weight() || torch.Tensor || Full rotation matrix R for merging
|}

== Usage Examples ==

=== Basic RoAD Configuration ===
<syntaxhighlight lang="python">
from peft import RoadConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# RoAD with minimal parameters (road_1)
config = RoadConfig(
    variant="road_1",           # Reuse parameters within groups
    group_size=64,              # Must divide out_features
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# Very few parameters: out_features / 2 per layer
</syntaxhighlight>

=== RoAD Variant Comparison ===
<syntaxhighlight lang="python">
from peft import RoadConfig, get_peft_model

# road_1: out_features // 2 parameters (most efficient)
config_1 = RoadConfig(variant="road_1", group_size=64, target_modules=["q_proj"])

# road_2: out_features parameters (medium)
config_2 = RoadConfig(variant="road_2", group_size=64, target_modules=["q_proj"])

# road_4: out_features * 2 parameters (most expressive)
config_4 = RoadConfig(variant="road_4", group_size=64, target_modules=["q_proj"])
</syntaxhighlight>

=== RoAD Forward Computation ===
<syntaxhighlight lang="python">
# RoAD applies transformation to layer output:
# 1. Split output into groups
# 2. For each group, pair first half with second half elements
# 3. Apply rotation: y = x * cos(theta) * alpha ± x_paired * sin(theta) * alpha
# 4. This is equivalent to block-diagonal rotation matrix multiplication
</syntaxhighlight>

=== Mixed Adapter Batch Inference ===
<syntaxhighlight lang="python">
# RoAD supports mixed adapter batches
outputs = model(
    input_ids,
    adapter_names=["adapter1", "adapter1", "adapter2"],
)
# Each sample uses its specified adapter
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
