# Principle: Parameter_Update

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|SGD Convergence|https://arxiv.org/abs/1606.04838]]
* [[source::Textbook|Deep Learning Book Ch. 8|https://www.deeplearningbook.org/contents/optimization.html]]
* [[source::Blog|Karpathy SGD Explanation|https://www.youtube.com/watch?v=VMj-3S1tku0]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Principle of modifying learnable parameters in the direction that reduces loss, using gradients computed during backpropagation.

=== Description ===

Parameter Update is the step where learning actually occurs:

1. **Gradient Direction:** Gradients point in the direction of steepest loss increase
2. **Negative Step:** Moving opposite to gradients decreases loss
3. **Learning Rate:** Controls step size; balances speed vs. stability
4. **In-Place Modification:** Parameters are updated directly for next forward pass

This is the simplest form of gradient descent—Stochastic Gradient Descent (SGD). More advanced optimizers (Adam, RMSprop) build on this foundation.

=== Usage ===

Apply this principle after the backward pass to:
- Update network weights and biases
- Implement the core learning mechanism
- Progress toward a loss minimum

The learning rate is the most critical hyperparameter:
- Too high: Overshooting, divergence, oscillation
- Too low: Slow convergence, stuck in local minima
- Typical range: 0.001 to 0.1

== Theoretical Basis ==

=== Gradient Descent ===

The update rule:

<math>
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}
</math>

Where:
- <math>\theta</math> are the parameters (weights, biases)
- <math>\eta</math> is the learning rate
- <math>\nabla_\theta \mathcal{L}</math> is the gradient of loss w.r.t. parameters

=== Why Negative Gradient? ===

The gradient <math>\nabla \mathcal{L}</math> points in the direction of steepest '''increase'''. To minimize loss, we move in the opposite direction.

<math>
\mathcal{L}(\theta - \eta \nabla \mathcal{L}) < \mathcal{L}(\theta) \quad \text{(for small enough } \eta \text{)}
</math>

=== Convergence Guarantee ===

For convex functions with Lipschitz gradients, gradient descent converges to global minimum. Neural networks are non-convex, but empirically SGD finds good local minima.

'''Update Pattern Pseudo-code:'''
<syntaxhighlight lang="python">
# Stochastic Gradient Descent
for parameter in model.parameters():
    parameter.data = parameter.data - learning_rate * parameter.grad
</syntaxhighlight>

=== Learning Rate Effects ===

{| class="wikitable"
|-
! Learning Rate !! Effect !! Typical Values
|-
| Too High || Loss oscillates or diverges || > 0.1
|-
| Just Right || Steady decrease, fast convergence || 0.001 - 0.1
|-
| Too Low || Very slow progress || < 0.0001
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Karpathy_Micrograd_Module_Parameters]]
