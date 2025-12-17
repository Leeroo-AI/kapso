{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Similarity_Metrics]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Mathematical formula for calculating normalized similarity scores from graph edit distance costs in the n8n workflow comparison system.

=== Description ===

The similarity formula converts raw graph edit distance (GED) costs into normalized similarity scores between 0.0 and 1.0. This normalization enables:
* Intuitive interpretation (1.0 = identical, 0.0 = maximally different)
* Threshold-based pass/fail decisions
* Comparison of results across different workflow pairs
* Configuration-independent similarity metrics

The formula implements a linear normalization with clamping to ensure valid score ranges.

=== Usage ===

Use this formula when you need to:
* Convert edit costs to percentage similarity
* Implement threshold-based workflow validation
* Compare similarity across different workflow pairs
* Report intuitive similarity metrics to users

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py
* '''Lines:''' L132-137

=== Formula ===
<syntaxhighlight lang="python">
similarity_score = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))
</syntaxhighlight>

== Mathematical Specification ==

=== Formula Definition ===

<math>
\text{similarity} = \max(0, \min(1, 1 - \frac{\text{edit\_cost}}{\text{max\_cost}}))
</math>

Where:
* '''edit_cost''': The computed graph edit distance between two workflows
* '''max_cost''': Maximum allowable cost (from configuration)
* '''similarity''': Normalized similarity score in range [0.0, 1.0]

=== Properties ===

{| class="wikitable"
|-
! Property !! Value !! Meaning
|-
| Range || [0.0, 1.0] || Clamped to valid probability range
|-
| Perfect match || edit_cost = 0 → similarity = 1.0 || Identical workflows score 1.0
|-
| Maximum difference || edit_cost ≥ max_cost → similarity = 0.0 || Completely different workflows score 0.0
|-
| Linearity || Linear decrease between extremes || Similarity decreases proportionally with edit cost
|}

=== Normalization Behavior ===

The formula performs three operations:

1. '''Base calculation:''' <code>1.0 - (edit_cost / max_cost)</code>
   * When edit_cost = 0: result = 1.0
   * When edit_cost = max_cost: result = 0.0
   * Linear interpolation between

2. '''Upper clamping:''' <code>min(1.0, ...)</code>
   * Prevents scores above 1.0 (shouldn't occur but ensures safety)

3. '''Lower clamping:''' <code>max(0.0, ...)</code>
   * Prevents negative scores when edit_cost > max_cost
   * Ensures all differences beyond max_cost score as 0.0

== Usage Examples ==

=== Direct Calculation ===
<syntaxhighlight lang="python">
# Example 1: Perfect match
edit_cost = 0.0
max_cost = 100.0
similarity = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))
print(f"Similarity: {similarity:.2f}")  # Output: 1.00

# Example 2: Half similarity
edit_cost = 50.0
max_cost = 100.0
similarity = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))
print(f"Similarity: {similarity:.2f}")  # Output: 0.50

# Example 3: Maximum difference
edit_cost = 100.0
max_cost = 100.0
similarity = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))
print(f"Similarity: {similarity:.2f}")  # Output: 0.00

# Example 4: Beyond maximum (clamped)
edit_cost = 150.0
max_cost = 100.0
similarity = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))
print(f"Similarity: {similarity:.2f}")  # Output: 0.00 (clamped)
</syntaxhighlight>

=== Integration with Comparison ===
<syntaxhighlight lang="python">
from src.similarity import calculate_graph_edit_distance
from src.config_loader import load_config

# Calculate edit distance
config = load_config("preset:balanced")
result = calculate_graph_edit_distance(graph1, graph2, config)

# The similarity score is already calculated using the formula
edit_cost = result['edit_cost']
max_cost = result['max_cost']
similarity = result['similarity_score']

# Verify the formula
expected = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))
assert abs(similarity - expected) < 1e-10, "Similarity calculation mismatch"

print(f"Edit Cost: {edit_cost}")
print(f"Max Cost: {max_cost}")
print(f"Similarity: {similarity:.2%}")
</syntaxhighlight>

=== Threshold-Based Validation ===
<syntaxhighlight lang="python">
def validate_workflow_similarity(graph1, graph2, config, threshold=0.8):
    """Validate if workflows meet similarity threshold."""
    result = calculate_graph_edit_distance(graph1, graph2, config)

    # Similarity calculated using the formula
    similarity = result['similarity_score']
    edit_cost = result['edit_cost']
    max_cost = result['max_cost']

    # Determine pass/fail
    passed = similarity >= threshold

    print(f"Edit Cost: {edit_cost:.1f} / {max_cost:.1f}")
    print(f"Similarity: {similarity:.2%}")
    print(f"Threshold: {threshold:.2%}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    return passed

# Example validation
config = load_config("preset:strict")
passed = validate_workflow_similarity(graph1, graph2, config, threshold=0.85)
</syntaxhighlight>

=== Custom Similarity Function ===
<syntaxhighlight lang="python">
def calculate_similarity(edit_cost: float, max_cost: float) -> float:
    """
    Calculate normalized similarity score.

    Implements the n8n workflow comparison similarity formula:
    similarity = max(0, min(1, 1 - (edit_cost / max_cost)))

    Args:
        edit_cost: Graph edit distance cost
        max_cost: Maximum allowable cost

    Returns:
        Normalized similarity score in [0.0, 1.0]
    """
    if max_cost <= 0:
        raise ValueError("max_cost must be positive")

    return max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))

# Usage
similarity = calculate_similarity(edit_cost=25.0, max_cost=100.0)
print(f"Similarity: {similarity:.2%}")  # Output: 75.00%
</syntaxhighlight>

=== Visualizing Similarity Curve ===
<syntaxhighlight lang="python">
import matplotlib.pyplot as plt
import numpy as np

# Generate edit cost values
max_cost = 100.0
edit_costs = np.linspace(0, 150, 1000)

# Calculate similarities using the formula
similarities = np.array([
    max(0.0, min(1.0, 1.0 - (cost / max_cost)))
    for cost in edit_costs
])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(edit_costs, similarities, linewidth=2)
plt.axhline(y=0.8, color='r', linestyle='--', label='Typical threshold (80%)')
plt.axvline(x=max_cost, color='g', linestyle='--', label=f'Max cost ({max_cost})')
plt.xlabel('Edit Cost')
plt.ylabel('Similarity Score')
plt.title('Workflow Similarity Formula: similarity = max(0, min(1, 1 - cost/max_cost))')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 150)
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('similarity_curve.png', dpi=150)
plt.show()

print("Similarity curve visualization saved to similarity_curve.png")
</syntaxhighlight>

=== Sensitivity Analysis ===
<syntaxhighlight lang="python">
# Analyze how different max_cost values affect similarity
max_costs = [50, 100, 150, 200]
test_edit_cost = 25.0

print(f"Edit cost: {test_edit_cost}")
print("\nSimilarity for different max_cost values:")
print("-" * 40)

for max_cost in max_costs:
    similarity = max(0.0, min(1.0, 1.0 - (test_edit_cost / max_cost)))
    percentage = similarity * 100

    print(f"max_cost = {max_cost:3.0f} → similarity = {similarity:.3f} ({percentage:.1f}%)")

# Output:
# Edit cost: 25.0
# Similarity for different max_cost values:
# ----------------------------------------
# max_cost =  50 → similarity = 0.500 (50.0%)
# max_cost = 100 → similarity = 0.750 (75.0%)
# max_cost = 150 → similarity = 0.833 (83.3%)
# max_cost = 200 → similarity = 0.875 (87.5%)
</syntaxhighlight>

== Mathematical Analysis ==

=== Inverse Relationship ===
The formula implements an inverse linear relationship:
* As edit_cost increases, similarity decreases linearly
* The rate of decrease is determined by max_cost
* Larger max_cost values result in gentler similarity decline

=== Boundary Conditions ===
{| class="wikitable"
|-
! Condition !! Formula Result !! Clamped Result !! Interpretation
|-
| edit_cost = 0 || 1.0 || 1.0 || Perfect match
|-
| edit_cost = max_cost/2 || 0.5 || 0.5 || Half similar
|-
| edit_cost = max_cost || 0.0 || 0.0 || At threshold
|-
| edit_cost > max_cost || < 0.0 || 0.0 || Beyond threshold (clamped)
|}

=== Configuration Impact ===
The max_cost parameter (from configuration) controls the sensitivity:
* '''Lower max_cost:''' More strict, similarity drops quickly
* '''Higher max_cost:''' More lenient, similarity drops slowly

Example comparison:
<syntaxhighlight lang="python">
edit_cost = 30.0

# Strict configuration (max_cost = 50)
strict_similarity = max(0.0, min(1.0, 1.0 - (30.0 / 50.0)))
print(f"Strict: {strict_similarity:.2%}")  # 40%

# Lenient configuration (max_cost = 200)
lenient_similarity = max(0.0, min(1.0, 1.0 - (30.0 / 200.0)))
print(f"Lenient: {lenient_similarity:.2%}")  # 85%
</syntaxhighlight>

== Design Rationale ==

=== Why Linear Normalization? ===
* '''Simple and predictable:''' Easy to understand and explain
* '''Proportional:''' Each unit of edit cost has equal impact
* '''Configurable:''' max_cost parameter allows tuning sensitivity

=== Why Clamping? ===
* '''Lower bound (0.0):''' Prevents confusing negative similarities
* '''Upper bound (1.0):''' Ensures valid probability interpretation
* '''Robustness:''' Handles edge cases and configuration errors gracefully

=== Alternative Approaches Not Used ===
{| class="wikitable"
|-
! Approach !! Formula !! Why Not Used
|-
| Exponential decay || <code>exp(-edit_cost / λ)</code> || Non-linear, harder to configure
|-
| Logarithmic || <code>1 / (1 + log(edit_cost))</code> || Non-intuitive scaling
|-
| Sigmoid || <code>1 / (1 + exp((cost - μ) / σ))</code> || More parameters, complexity
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Similarity_Calculation]]

=== Used By ===
* [[used_by::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]

=== Related Concepts ===
* [[related::Concept:Graph_Edit_Distance]]
* [[related::Concept:Normalization]]
* [[related::Concept:Similarity_Metrics]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Workflow_Comparison]]

[[Category:Implementation]]
[[Category:n8n]]
[[Category:Mathematical_Formula]]
[[Category:Similarity_Metrics]]
