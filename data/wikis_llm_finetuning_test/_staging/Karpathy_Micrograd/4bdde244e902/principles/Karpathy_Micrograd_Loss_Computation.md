# Principle: Loss_Computation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Textbook|Deep Learning Book Ch. 6.2|https://www.deeplearningbook.org/contents/mlp.html]]
* [[source::Paper|Support Vector Machines|https://doi.org/10.1023/A:1022627411411]]
* [[source::Blog|Karpathy Loss Functions|https://www.youtube.com/watch?v=VMj-3S1tku0]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Principle of quantifying prediction error as a single scalar value that guides the optimization process through gradient descent.

=== Description ===

Loss Computation transforms the difference between predictions and ground truth into a differentiable scalar that:

1. **Quantifies Error:** Measures how far predictions are from targets
2. **Enables Optimization:** Provides a single value to minimize
3. **Extends Computation Graph:** Loss becomes the root node for backpropagation
4. **Guides Learning:** Gradients flow from loss back through the network

The loss function choice affects training dynamics:
- **MSE (Mean Squared Error):** Penalizes large errors quadratically; good for regression
- **Hinge Loss:** Margin-based; promotes separation in classification
- **Cross-Entropy:** Information-theoretic; standard for classification probabilities

=== Usage ===

Apply this principle after the forward pass to:
- Convert predictions and targets into a single scalar for optimization
- Complete the computation graph before calling backward()
- Choose appropriate loss functions for your task (regression vs. classification)

The loss value should decrease during successful training, indicating the model is improving.

== Theoretical Basis ==

=== Mean Squared Error ===

<math>
\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
</math>

Properties:
- Differentiable everywhere
- Penalizes large errors heavily (quadratic)
- Output range: [0, ∞)

=== Hinge Loss (SVM-style) ===

<math>
\mathcal{L}_{hinge} = \sum_{i=1}^{n} \max(0, 1 - y_i \cdot \hat{y}_i)
</math>

Properties:
- Margin-based: wants correct class score > incorrect by margin 1
- Creates sparse gradients (zero for correct predictions with margin)
- Requires labels in {-1, +1}

=== Implementation in Micrograd ===

Micrograd computes loss using Value arithmetic operations:

'''MSE Loss Pseudo-code:'''
<syntaxhighlight lang="python">
# Each operation creates a Value node in the graph
loss = sum((pred - target)**2 for pred, target in zip(predictions, targets))
# Graph: pred -> (-target) -> (**2) -> (sum)
</syntaxhighlight>

'''Hinge Loss Pseudo-code:'''
<syntaxhighlight lang="python">
# Using relu() for max(0, x)
loss = sum((1 + -yi*scorei).relu() for yi, scorei in zip(ys, scores))
# Graph: yi -> (*-1) -> (*scorei) -> (+1) -> (relu) -> (sum)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Karpathy_Micrograd_Loss_Operations]]
