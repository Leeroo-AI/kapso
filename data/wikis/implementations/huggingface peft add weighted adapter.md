{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|TIES|https://arxiv.org/abs/2306.01708]]
* [[source::Paper|DARE|https://arxiv.org/abs/2311.03099]]
|-
! Domains
| [[domain::Adapter]], [[domain::Model_Merging]], [[domain::Multi_Task]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for combining multiple LoRA adapters into a single merged adapter using various merging algorithms.

=== Description ===

`add_weighted_adapter` creates a new adapter by merging multiple existing adapters with specified weights. It supports various merging algorithms including linear interpolation, SVD-based methods, TIES (TrIm, Elect Sign & Merge), and DARE (Drop And REscale). This enables model composition and multi-task learning.

=== Usage ===

Use this after loading multiple adapters to combine their capabilities. Choose the combination type based on your needs: "linear" for simple interpolation, "ties" for conflict resolution, "dare_ties" for sparsification with TIES. Weights can be positive or negative.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/tuners/lora/model.py
* '''Lines:''' L573-708

=== Signature ===
<syntaxhighlight lang="python">
def add_weighted_adapter(
    self,
    adapters: list[str],
    weights: list[float],
    adapter_name: str,
    combination_type: str = "svd",
    svd_rank: int | None = None,
    svd_clamp: int | None = None,
    svd_full_matrices: bool = True,
    svd_driver: str | None = None,
    density: float | None = None,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> None:
    """
    Merge adapters into a new adapter.

    Args:
        adapters: List of adapter names to merge
        weights: Weights for each adapter (can be negative)
        adapter_name: Name for the merged adapter
        combination_type: Merge algorithm (svd, linear, ties, dare_ties, etc.)
        density: Pruning density for TIES/DARE (0.0-1.0)
        majority_sign_method: Sign election for TIES ("total" or "frequency")
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Method on LoraModel (accessed via PeftModel.base_model)
# model.add_weighted_adapter(...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| adapters || list[str] || Yes || Names of adapters to combine
|-
| weights || list[float] || Yes || Weight per adapter (can be negative for subtraction)
|-
| adapter_name || str || Yes || Name for the new merged adapter
|-
| combination_type || str || No || Algorithm: svd, linear, ties, dare_ties, etc.
|-
| density || float || No || Pruning density for TIES/DARE (0.0-1.0)
|-
| majority_sign_method || str || No || "total" or "frequency" for TIES
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || - || Adapter added in-place to model
|}

== Usage Examples ==

=== Linear Combination ===
<syntaxhighlight lang="python">
# Load adapters
model.load_adapter("path/to/math", adapter_name="math")
model.load_adapter("path/to/code", adapter_name="code")

# Linear merge: 70% math + 30% code
model.add_weighted_adapter(
    adapters=["math", "code"],
    weights=[0.7, 0.3],
    adapter_name="math_code",
    combination_type="linear",
)

model.set_adapter("math_code")
</syntaxhighlight>

=== TIES Merging ===
<syntaxhighlight lang="python">
# TIES: Handles sign conflicts between adapters
model.add_weighted_adapter(
    adapters=["adapter1", "adapter2", "adapter3"],
    weights=[1.0, 1.0, 1.0],
    adapter_name="ties_merged",
    combination_type="ties",
    density=0.5,  # Keep top 50% of parameters
    majority_sign_method="total",
)
</syntaxhighlight>

=== DARE-TIES ===
<syntaxhighlight lang="python">
# DARE-TIES: Random pruning + TIES merging
model.add_weighted_adapter(
    adapters=["math", "code", "writing"],
    weights=[0.5, 0.3, 0.2],
    adapter_name="multi_task",
    combination_type="dare_ties",
    density=0.7,  # Random drop 30% of parameters
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Adapter_Merge_Execution]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
