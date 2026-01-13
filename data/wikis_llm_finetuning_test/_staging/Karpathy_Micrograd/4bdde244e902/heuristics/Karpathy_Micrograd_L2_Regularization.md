# Heuristic: L2_Regularization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|micrograd demo|https://github.com/karpathy/micrograd/blob/master/demo.ipynb]]
* [[source::Discussion|Regularization|https://cs231n.github.io/neural-networks-2/#reg]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

L2 weight regularization with alpha=1e-4 to prevent overfitting and improve generalization.

=== Description ===

The micrograd demo implements L2 regularization (weight decay) by adding the squared sum of all parameters to the loss function, scaled by a small coefficient alpha. This encourages smaller weights, which typically leads to simpler models with better generalization to unseen data.

L2 regularization is computed as: `reg_loss = alpha * sum(p*p for p in parameters)`. The total loss becomes: `total_loss = data_loss + reg_loss`, and gradients flow through both terms during backpropagation.

=== Usage ===

Apply L2 regularization when training neural networks, especially when the model has more capacity than necessary for the task (risk of overfitting). The coefficient alpha=1e-4 is a common starting point; increase it if overfitting persists, decrease it if the model underfits.

== The Insight (Rule of Thumb) ==

* **Action:** Add L2 penalty to loss: `reg_loss = alpha * sum(p*p for p in model.parameters())`
* **Value:** alpha = 1e-4 (0.0001) as a reasonable default
* **Trade-off:** Reduces overfitting at the cost of slightly higher training loss; too high alpha causes underfitting

== Reasoning ==

L2 regularization penalizes large weights, which often correspond to complex decision boundaries that fit noise in training data. By encouraging smaller weights, the model learns smoother functions that generalize better.

Mathematically, L2 regularization is equivalent to adding a Gaussian prior on weights during Bayesian inference. The gradient of the L2 term with respect to each weight is simply `2 * alpha * weight`, effectively shrinking weights toward zero by a constant proportion each step.

The choice of alpha=1e-4 is a standard starting point in deep learning - small enough to not dominate the data loss, but large enough to have a regularizing effect. For the 337-parameter MLP in the demo, this provides gentle regularization while allowing the model to fit the moons dataset.

== Code Evidence ==

L2 regularization implementation from `demo.ipynb` cell 6:
<syntaxhighlight lang="python">
# loss function
def loss(batch_size=None):
    # ...forward pass...

    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))

    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)
</syntaxhighlight>

Note: The regularization is computed using micrograd's `Value` operations, so gradients automatically flow through `p*p` to update each parameter.

== Related Pages ==

* [[used_by::Implementation:Karpathy_Micrograd_Loss_Operations]]
* [[used_by::Principle:Karpathy_Micrograd_Loss_Computation]]
