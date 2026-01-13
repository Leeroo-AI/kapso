# Principle: Backward_Pass

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Backpropagation|https://doi.org/10.1038/323533a0]]
* [[source::Paper|Automatic Differentiation Survey|https://arxiv.org/abs/1502.05767]]
* [[source::Blog|Karpathy Backprop Video|https://www.youtube.com/watch?v=VMj-3S1tku0]]
* [[source::Textbook|Deep Learning Book Ch. 6.5|https://www.deeplearningbook.org/contents/mlp.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Automatic_Differentiation]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Principle of computing gradients by traversing the computation graph in reverse topological order, applying the chain rule at each node.

=== Description ===

The Backward Pass (Backpropagation) is the key algorithm enabling neural network training:

1. **Topological Ordering:** First, sort all nodes so each node comes after its dependencies
2. **Gradient Initialization:** Set the output node's gradient to 1.0 (∂L/∂L = 1)
3. **Reverse Traversal:** Process nodes in reverse order, accumulating gradients
4. **Chain Rule Application:** Each node distributes gradient to its inputs via local derivatives

This algorithm has O(N) complexity where N is the number of operations—the same as the forward pass—making deep learning computationally tractable.

=== Usage ===

Apply this principle when:
- Training neural networks (computing parameter gradients)
- Performing gradient-based optimization
- Implementing automatic differentiation systems

The backward pass must be called after:
1. Forward pass (builds the computation graph)
2. Loss computation (creates the root node)

After backward pass, all parameters have their `.grad` attribute populated.

== Theoretical Basis ==

=== Chain Rule ===

The fundamental mathematical principle:

<math>
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
</math>

For a computation y = f(x):
- We receive <math>\frac{\partial L}{\partial y}</math> from downstream (stored in `y.grad`)
- We compute <math>\frac{\partial y}{\partial x}</math> locally (the Jacobian)
- We pass <math>\frac{\partial L}{\partial x}</math> to upstream (accumulated in `x.grad`)

=== Topological Sort ===

Required to ensure we process nodes only after all their consumers are processed.

'''Algorithm:'''
<syntaxhighlight lang="python">
def topological_sort(root):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:      # Visit all inputs first
                build_topo(child)
            topo.append(v)             # Then add self

    build_topo(root)
    return topo  # Now reversed(topo) gives correct backward order
</syntaxhighlight>

=== Local Gradient Rules ===

Each operation defines its local gradient:

{| class="wikitable"
|-
! Operation !! Forward !! Backward (∂out/∂inputs)
|-
| Addition || out = a + b || ∂out/∂a = 1, ∂out/∂b = 1
|-
| Multiplication || out = a * b || ∂out/∂a = b, ∂out/∂b = a
|-
| Power || out = a^n || ∂out/∂a = n * a^(n-1)
|-
| ReLU || out = max(0, a) || ∂out/∂a = 1 if a > 0 else 0
|}

=== Gradient Accumulation ===

When a value is used multiple times, gradients are '''summed''':

<math>
\text{If } z = f(x) \text{ and } w = g(x), \text{ then } \frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}\frac{\partial z}{\partial x} + \frac{\partial L}{\partial w}\frac{\partial w}{\partial x}
</math>

This is why micrograd uses `+=` rather than `=` in backward functions.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Karpathy_Micrograd_Value_Backward]]
