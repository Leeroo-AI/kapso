# Principle: Training_Loop

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Textbook|Deep Learning Book Ch. 8|https://www.deeplearningbook.org/contents/optimization.html]]
* [[source::Blog|Karpathy Training Loop|https://www.youtube.com/watch?v=VMj-3S1tku0]]
* [[source::Paper|Batch Normalization|https://arxiv.org/abs/1502.03167]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Principle of iteratively refining network parameters through repeated cycles of forward pass, loss computation, backpropagation, and parameter update.

=== Description ===

The Training Loop orchestrates the complete learning process:

1. **Gradient Reset:** Zero all gradients before each iteration (critical!)
2. **Forward Pass:** Compute predictions and build computation graph
3. **Loss Computation:** Calculate scalar error measure
4. **Backward Pass:** Compute gradients via backpropagation
5. **Parameter Update:** Apply gradient descent step
6. **Repeat:** Continue until convergence or max epochs

The loop structure ensures gradients don't accumulate incorrectly across iterations and that the computation graph is fresh for each batch.

=== Usage ===

Apply this principle to:
- Train any neural network
- Structure the optimization process
- Implement convergence monitoring

Key decisions in loop design:
- **Epochs:** How many full passes over the data
- **Batch size:** How many samples per gradient update
- **Learning rate schedule:** Fixed or decaying over time
- **Early stopping:** When to terminate training

== Theoretical Basis ==

=== Epoch vs. Iteration ===

{| class="wikitable"
|-
! Term !! Definition
|-
| Iteration || One forward-backward-update cycle
|-
| Epoch || One complete pass through all training data
|-
| Batch || Subset of data used per iteration
|}

In micrograd (no batching), one iteration = one epoch when processing all data.

=== Why Zero Gradients? ===

Gradients accumulate by default (using `+=`). Without zeroing:
- Iteration 1: grad = g1
- Iteration 2: grad = g1 + g2 (wrong!)
- Iteration 3: grad = g1 + g2 + g3 (very wrong!)

The `zero_grad()` method resets all gradients to 0 before each backward pass.

=== Convergence Monitoring ===

Track loss over epochs to detect:
- **Convergence:** Loss stabilizes at low value
- **Overfitting:** Training loss decreases but validation loss increases
- **Divergence:** Loss increases (learning rate too high)

'''Training Loop Pseudo-code:'''
<syntaxhighlight lang="python">
for epoch in range(max_epochs):
    # 1. Zero gradients
    model.zero_grad()

    # 2. Forward pass
    predictions = [model(x) for x in inputs]

    # 3. Compute loss
    loss = loss_function(predictions, targets)

    # 4. Backward pass
    loss.backward()

    # 5. Update parameters
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    # 6. Monitor progress
    if epoch % log_interval == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
</syntaxhighlight>

=== Common Patterns ===

{| class="wikitable"
|-
! Pattern !! Purpose !! Implementation
|-
| Learning rate decay || Prevent overshooting || `lr = lr * decay_factor` each epoch
|-
| Early stopping || Prevent overfitting || Stop when validation loss increases
|-
| Gradient clipping || Prevent exploding gradients || `grad = min(grad, max_norm)`
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Karpathy_Micrograd_Module_Zero_Grad]]
