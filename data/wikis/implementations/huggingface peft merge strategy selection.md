{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::File|merge_utils.py|src/peft/utils/merge_utils.py]]
|-
! Domains
| [[domain::Model_Merging]], [[domain::Multi_Task]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for configuring merge algorithm parameters before calling `add_weighted_adapter()`.

=== Description ===

The merge strategy is configured via parameters to `add_weighted_adapter()`. The underlying algorithms are implemented in `src/peft/utils/merge_utils.py` and include:
* `ties()` - TIES merging algorithm
* `dare_ties()` - DARE + TIES combination
* `dare_linear()` - DARE + linear average
* `magnitude_prune()` - Pruning by magnitude

=== Usage ===

Configure merge parameters as arguments to `add_weighted_adapter()`. This is a configuration step, not a separate API call.

== Code Reference ==

=== Source Location ===
* '''File:''' `src/peft/utils/merge_utils.py`
* '''Lines:''' L144-269

=== Available Algorithms ===
<syntaxhighlight lang="python">
# combination_type options for add_weighted_adapter():
COMBINATION_TYPES = [
    "svd",              # SVD-based merge
    "linear",           # Simple weighted average
    "cat",              # Concatenation (increases rank)
    "ties",             # TIES merging
    "ties_svd",         # TIES + SVD
    "dare_ties",        # DARE + TIES
    "dare_linear",      # DARE + linear
    "dare_ties_svd",    # DARE + TIES + SVD
    "dare_linear_svd",  # DARE + linear + SVD
    "magnitude_prune",  # Prune by magnitude
    "magnitude_prune_svd",
]
</syntaxhighlight>

=== Key Functions ===
<syntaxhighlight lang="python">
def ties(task_tensors, weights, density, majority_sign_method="total"):
    """
    TIES: TrIm, Elect Sign, Merge.

    Args:
        task_tensors: List of adapter weight tensors
        weights: Per-adapter weights
        density: Fraction of weights to keep (0.0-1.0)
        majority_sign_method: "total" or "frequency"

    Returns:
        Merged tensor
    """

def dare_ties(task_tensors, weights, density, majority_sign_method="total"):
    """DARE pruning followed by TIES merge."""

def dare_linear(task_tensors, weights, density):
    """DARE pruning followed by linear average."""
</syntaxhighlight>

== Usage Examples ==

=== TIES Merge ===
<syntaxhighlight lang="python">
model.add_weighted_adapter(
    adapters=["task1", "task2", "task3"],
    weights=[0.4, 0.3, 0.3],
    adapter_name="merged_ties",
    combination_type="ties",
    density=0.7,  # Keep 70% of weights
    majority_sign_method="total",
)
</syntaxhighlight>

=== DARE Linear Merge ===
<syntaxhighlight lang="python">
model.add_weighted_adapter(
    adapters=["task1", "task2"],
    weights=[0.5, 0.5],
    adapter_name="merged_dare",
    combination_type="dare_linear",
    density=0.5,  # Random 50% pruning with rescale
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Merge_Strategy_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
