{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|Poly|https://arxiv.org/abs/2307.06069]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Multi_Task_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Polytropon adapter layer that uses routable mixtures of low-rank skill modules for multi-task parameter-efficient fine-tuning.

=== Description ===

PolyLayer implements the Polytropon method which combines multiple "skill" LoRA modules through learned routing. The layer maintains n_skills sets of low-rank matrices (A and B) split across n_splits dimensions. A router module dynamically computes mixing weights based on task IDs or input, combining skills via einsum operations. This allows a single model to handle multiple tasks by routing to different skill combinations.

=== Usage ===

Use Poly for multi-task learning scenarios where different tasks may benefit from different combinations of learned skills. It's particularly effective when tasks have overlapping but distinct requirements. The router can be task-ID based (Poly) or input-based (MoPE) for more fine-grained routing decisions.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/poly/layer.py src/peft/tuners/poly/layer.py]
* '''Lines:''' 1-166

=== Signature ===
<syntaxhighlight lang="python">
class PolyLayer(BaseTunerLayer):
    """
    Polytropon layer with routable skill modules.

    Attributes:
        poly_lora_A: ParameterDict of A matrices [n_splits, n_skills, in//n_splits, r]
        poly_lora_B: ParameterDict of B matrices [n_splits, n_skills, r, out//n_splits]
        poly_router: ModuleDict of router modules
        r: Rank for low-rank decomposition
        n_tasks: Number of tasks
        n_skills: Number of skill modules
        n_splits: Number of dimension splits
    """
    adapter_layer_names = ("poly_lora_A", "poly_lora_B", "poly_router")
    other_param_names = ("r", "n_tasks", "n_skills", "n_splits")

    def update_layer(
        self,
        adapter_name,
        poly_config: PolyConfig,
        inference_mode: bool = False,
        **kwargs,
    ):
        """Create Poly adapter parameters."""

    def reset_poly_parameters(self, adapter_name, init_weights):
        """Initialize A/B matrices and router."""

class Linear(nn.Module, PolyLayer):
    """Poly implemented in Linear layer."""
    def forward(
        self,
        x: torch.Tensor,
        *args,
        task_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with task-based routing."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.poly import PolyLayer, PolyConfig, PolyModel
from peft import PolyConfig, get_peft_model
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
| r || int || Yes || Rank for low-rank decomposition
|-
| n_tasks || int || Yes || Number of tasks for routing
|-
| n_skills || int || Yes || Number of skill modules
|-
| n_splits || int || No || Number of dimension splits (default: 1)
|-
| poly_type || str || No || Router type ("poly" or "mope")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Input + routed skill combination
|-
| mixing_weights || torch.Tensor || Router output [batch, n_splits, n_skills]
|}

== Usage Examples ==

=== Basic Poly Configuration ===
<syntaxhighlight lang="python">
from peft import PolyConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure Poly for multi-task learning
config = PolyConfig(
    r=8,
    n_tasks=4,              # Number of tasks
    n_skills=8,             # Skill modules to combine
    n_splits=1,             # Dimension splits
    target_modules=["c_attn", "c_proj"],
    poly_type="poly",       # Task-based routing
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Poly with Task IDs ===
<syntaxhighlight lang="python">
from peft import PolyConfig, get_peft_model

config = PolyConfig(
    r=16,
    n_tasks=3,
    n_skills=6,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config)

# During training/inference, pass task_ids
task_ids = torch.tensor([0, 1, 0, 2])  # Task for each batch item
outputs = model(input_ids, task_ids=task_ids)
</syntaxhighlight>

=== MoPE (Input-Based Routing) ===
<syntaxhighlight lang="python">
from peft import PolyConfig, get_peft_model

# Use input-based routing instead of task IDs
config = PolyConfig(
    r=8,
    n_tasks=1,              # Not used with MoPE
    n_skills=8,
    target_modules=["q_proj", "v_proj"],
    poly_type="mope",       # Mixture of Prompt Experts
)

model = get_peft_model(model, config)
# Routing is determined by input, no task_ids needed
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
