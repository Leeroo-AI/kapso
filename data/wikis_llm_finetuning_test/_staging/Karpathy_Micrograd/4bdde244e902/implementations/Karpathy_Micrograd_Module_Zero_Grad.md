# Implementation: Module_Zero_Grad

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Micrograd|https://github.com/karpathy/micrograd]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Concrete tool for resetting all parameter gradients to zero before each training iteration provided by the micrograd library.

=== Description ===

The `Module.zero_grad()` method iterates over all parameters (via `self.parameters()`) and sets each gradient to 0. This is essential because:
- Gradients accumulate by default (using `+=` in backward functions)
- Without zeroing, gradients from previous iterations would corrupt current gradients
- This mirrors PyTorch's `optimizer.zero_grad()` pattern

=== Usage ===

Call `zero_grad()` at the beginning of each training iteration, before `backward()`. The order matters:
1. `model.zero_grad()` - Clear old gradients
2. `loss.backward()` - Compute new gradients
3. Parameter update - Use the fresh gradients

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/karpathy/micrograd micrograd]
* '''File:''' micrograd/nn.py
* '''Lines:''' 6-8

=== Signature ===
<syntaxhighlight lang="python">
class Module:
    def zero_grad(self):
        """Reset all parameter gradients to zero.

        Must be called before backward() to prevent gradient accumulation
        across training iterations.

        This method iterates over self.parameters() and sets each
        Value's .grad attribute to 0.

        Returns:
            None (modifies gradients in-place)
        """
        for p in self.parameters():
            p.grad = 0
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP  # or any Module subclass
# zero_grad() is a method inherited from Module
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| self || Module || Yes || The module whose parameters' gradients will be zeroed
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (in-place) || None || All parameter.grad values set to 0
|}

== Usage Examples ==

=== Correct Training Loop ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [4, 4, 1])
learning_rate = 0.05

xs = [[2.0, 3.0], [-1.0, -1.0], [3.0, -2.0], [0.5, 1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for epoch in range(100):
    # Forward pass
    ypred = [model(x) for x in xs]

    # Loss computation
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # CRITICAL: Zero gradients BEFORE backward!
    model.zero_grad()

    # Backward pass
    loss.backward()

    # Parameter update
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
</syntaxhighlight>

=== What Happens Without zero_grad() ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [2, 1])
x = [1.0, 2.0]
y = 1.0

# First iteration
pred = model(x)
loss = (pred - y) ** 2
loss.backward()

first_grad = model.parameters()[0].grad
print(f"After 1st backward: grad = {first_grad:.4f}")

# Second iteration WITHOUT zero_grad (BUG!)
pred = model(x)
loss = (pred - y) ** 2
loss.backward()  # Gradients ACCUMULATE

second_grad = model.parameters()[0].grad
print(f"After 2nd backward (no zero_grad): grad = {second_grad:.4f}")
# second_grad ≈ 2 * first_grad (WRONG - gradients accumulated!)

# Third iteration WITH zero_grad (CORRECT)
model.zero_grad()  # Reset to 0
pred = model(x)
loss = (pred - y) ** 2
loss.backward()

third_grad = model.parameters()[0].grad
print(f"After 3rd backward (with zero_grad): grad = {third_grad:.4f}")
# third_grad ≈ first_grad (CORRECT - fresh gradient)
</syntaxhighlight>

=== Complete Training Example ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

# Create model
model = MLP(2, [16, 16, 1])

# Training data (XOR-like problem)
xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
ys = [-1.0, 1.0, 1.0, -1.0]

# Hyperparameters
learning_rate = 0.1
epochs = 200

# Training loop
print("Training...")
for epoch in range(epochs):
    # Forward
    predictions = [model(x) for x in xs]

    # Loss (MSE)
    loss = sum((pred - target)**2 for pred, target in zip(predictions, ys))

    # Zero gradients - CRITICAL STEP
    model.zero_grad()

    # Backward
    loss.backward()

    # Update
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    # Logging
    if epoch % 50 == 0:
        acc = sum(1 for p, y in zip(predictions, ys)
                  if (p.data > 0) == (y > 0)) / len(ys)
        print(f"Epoch {epoch:3d} | Loss: {loss.data:.4f} | Accuracy: {acc:.2%}")

# Final predictions
print("\nFinal predictions:")
for x, y in zip(xs, ys):
    pred = model(x)
    print(f"  Input: {x} -> Pred: {pred.data:+.4f}, Target: {y:+.1f}")
</syntaxhighlight>

=== Order of Operations ===
<syntaxhighlight lang="python">
# The correct sequence in each training iteration:

# Step 1: Forward pass (builds computation graph)
predictions = [model(x) for x in xs]

# Step 2: Compute loss (creates root node for graph)
loss = compute_loss(predictions, targets)

# Step 3: Zero gradients (BEFORE backward!)
model.zero_grad()

# Step 4: Backward pass (populates fresh gradients)
loss.backward()

# Step 5: Update parameters (uses the gradients)
for p in model.parameters():
    p.data -= lr * p.grad

# Why this order?
# - zero_grad before backward: ensures clean slate for gradient computation
# - backward before update: gradients must exist before we use them
# - update after backward: uses the just-computed gradients
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Karpathy_Micrograd_Training_Loop]]

=== Requires Environment ===
* [[requires_env::Environment:Karpathy_Micrograd_Python_3]]
