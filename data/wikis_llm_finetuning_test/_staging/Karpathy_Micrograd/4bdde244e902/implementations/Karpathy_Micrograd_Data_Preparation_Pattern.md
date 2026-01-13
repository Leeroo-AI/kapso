# Implementation: Data_Preparation_Pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Micrograd|https://github.com/karpathy/micrograd]]
* [[source::Blog|Micrograd Demo Notebook|https://github.com/karpathy/micrograd/blob/master/demo.ipynb]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Data_Science]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Concrete pattern for preparing training data as Python lists for micrograd's neural network training workflow.

=== Description ===

This is a **Pattern Doc** rather than an API Doc—micrograd does not provide a data loading API. Instead, users define their training data as standard Python lists.

The pattern requires:
1. `xs`: A list of input vectors (each vector is a list of floats)
2. `ys`: A list of target values (floats), one per input vector

This approach is intentionally minimal to keep the focus on the autograd and neural network concepts.

=== Usage ===

Use this pattern when preparing data for training with micrograd. The format integrates directly with the `MLP.__call__` forward pass—each `x` in `xs` is passed through the network independently.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/karpathy/micrograd micrograd]
* '''File:''' User code (no library implementation)

=== Pattern Specification ===
<syntaxhighlight lang="python">
# Data structure type signatures
xs: list[list[float]]  # Input feature vectors
ys: list[float]        # Target values (one per input)

# Constraints:
# - len(xs) == len(ys)  # Same number of inputs and outputs
# - len(xs[i]) == len(xs[j]) for all i, j  # All input vectors same dimension
# - len(xs[0]) == nin  # Must match network input dimension
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# No import needed - uses built-in Python lists
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| raw_data || Any || Yes || Source data to be converted (domain-specific)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| xs || list[list[float]] || Input feature vectors, shape (n_samples, n_features)
|-
| ys || list[float] || Target values, shape (n_samples,)
|}

== Usage Examples ==

=== Binary Classification (Moon Dataset) ===
<syntaxhighlight lang="python">
# Example from micrograd demo: binary classification of 2D points
# Class +1: upper region, Class -1: lower region

xs = [
    [2.0, 3.0],
    [3.0, -1.0],
    [1.0, 0.5],
    [-1.0, -2.0],
]

ys = [1.0, -1.0, 1.0, -1.0]  # Binary labels

# Verify dimensions match network
n = MLP(2, [4, 4, 1])  # 2 inputs matches len(xs[0])
</syntaxhighlight>

=== Regression Example ===
<syntaxhighlight lang="python">
# Simple regression: predict y = 2*x1 + 3*x2
xs = [
    [1.0, 1.0],   # Expected: 5.0
    [2.0, 0.0],   # Expected: 4.0
    [0.0, 2.0],   # Expected: 6.0
    [-1.0, 1.0],  # Expected: 1.0
]

ys = [5.0, 4.0, 6.0, 1.0]

# Network with linear output
n = MLP(2, [8, 1])  # Last layer has nonlin=False automatically
</syntaxhighlight>

=== XOR Problem ===
<syntaxhighlight lang="python">
# Classic XOR: non-linearly separable binary classification
xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]

ys = [-1.0, 1.0, 1.0, -1.0]  # XOR truth table with -1/+1 encoding

# Requires hidden layer for non-linear decision boundary
n = MLP(2, [4, 1])
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Karpathy_Micrograd_Data_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:Karpathy_Micrograd_Python_3]]
