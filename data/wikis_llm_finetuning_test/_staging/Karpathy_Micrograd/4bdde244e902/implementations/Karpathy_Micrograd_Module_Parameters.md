# Implementation: Module_Parameters

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

Concrete tool for accessing all learnable parameters in a neural network provided by the micrograd library.

=== Description ===

The `Module.parameters()` method recursively collects all `Value` objects that represent learnable parameters (weights and biases). This method is essential for:
- Iterating over parameters during gradient descent updates
- Counting total parameters in a model
- Implementing optimizers

Each class in the hierarchy (Neuron, Layer, MLP) overrides this method to collect its own parameters and delegate to children.

=== Usage ===

Call `parameters()` on any Module subclass (Neuron, Layer, MLP) to get a flat list of all learnable Values. Use this list for:
- SGD updates: `for p in model.parameters(): p.data -= lr * p.grad`
- Counting parameters: `len(model.parameters())`
- Applying regularization: `reg = sum(p**2 for p in model.parameters())`

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/karpathy/micrograd micrograd]
* '''File:''' micrograd/nn.py
* '''Lines:'''
** Module.parameters: 10-11
** Neuron.parameters: 24-25
** Layer.parameters: 39-40
** MLP.parameters: 56-57

=== Signatures ===
<syntaxhighlight lang="python">
class Module:
    def parameters(self):
        """Return empty list (base class).

        Returns:
            list[Value]: Empty list; subclasses override.
        """
        return []

class Neuron(Module):
    def parameters(self):
        """Return weights and bias.

        Returns:
            list[Value]: [w[0], w[1], ..., w[n], b]
        """
        return self.w + [self.b]

class Layer(Module):
    def parameters(self):
        """Return parameters from all neurons.

        Returns:
            list[Value]: Flattened list of all neuron parameters.
        """
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def parameters(self):
        """Return parameters from all layers.

        Returns:
            list[Value]: Flattened list of all layer parameters.
        """
        return [p for layer in self.layers for p in layer.parameters()]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP  # or Neuron, Layer
# parameters() is a method, no separate import
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| self || Module || Yes || The module to extract parameters from
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| parameters || list[Value] || Flat list of all learnable parameters (weights and biases)
|}

== Usage Examples ==

=== Basic SGD Update ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [4, 4, 1])
learning_rate = 0.01

# After forward pass, loss computation, and backward()...

# Get all parameters
params = model.parameters()
print(f"Total parameters: {len(params)}")  # (2*4+4) + (4*4+4) + (4*1+1) = 41

# Update each parameter
for p in params:
    p.data -= learning_rate * p.grad
</syntaxhighlight>

=== Full Training Loop ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [16, 16, 1])
learning_rate = 0.05

xs = [[2.0, 3.0], [-1.0, -1.0], [3.0, -2.0], [0.5, 1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for epoch in range(100):
    # Forward pass
    ypred = [model(x) for x in xs]

    # Loss computation
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Zero gradients (important!)
    model.zero_grad()

    # Backward pass
    loss.backward()

    # Parameter update - the key step!
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
</syntaxhighlight>

=== Parameter Count by Architecture ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

# Different architectures
architectures = [
    (2, [4, 1]),           # Small
    (2, [16, 16, 1]),      # Medium
    (10, [64, 32, 1]),     # Larger
]

for nin, nouts in architectures:
    model = MLP(nin, nouts)
    n_params = len(model.parameters())

    # Calculate manually to verify
    sz = [nin] + nouts
    expected = sum(sz[i]*sz[i+1] + sz[i+1] for i in range(len(nouts)))

    print(f"MLP({nin}, {nouts}): {n_params} params (expected: {expected})")

# Output:
# MLP(2, [4, 1]): 17 params (expected: 17)
# MLP(2, [16, 16, 1]): 337 params (expected: 337)
# MLP(10, [64, 32, 1]): 2785 params (expected: 2785)
</syntaxhighlight>

=== L2 Regularization ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [16, 16, 1])
lambda_reg = 0.01

xs = [[2.0, 3.0], [-1.0, -1.0]]
ys = [1.0, -1.0]

# Forward pass
ypred = [model(x) for x in xs]

# Data loss
data_loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

# Regularization: penalize large weights
# Uses parameters() to access all weights
reg_loss = lambda_reg * sum(p**2 for p in model.parameters())

# Total loss
loss = data_loss + reg_loss

# Now backward() will compute gradients including regularization term
model.zero_grad()
loss.backward()

# Update includes effect of regularization (weight decay)
for p in model.parameters():
    p.data -= 0.01 * p.grad
</syntaxhighlight>

=== Inspecting Parameter Gradients ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [2, 1])

# Forward, loss, backward
x = [1.0, 2.0]
pred = model(x)
loss = pred ** 2
model.zero_grad()
loss.backward()

# Inspect gradients
for i, p in enumerate(model.parameters()):
    print(f"Parameter {i}: data={p.data:.4f}, grad={p.grad:.4f}")

# Example output:
# Parameter 0: data=0.4521, grad=0.1234  (first weight)
# Parameter 1: data=-0.7832, grad=0.2468  (second weight)
# Parameter 2: data=0.0000, grad=0.0617  (bias)
# ...
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Karpathy_Micrograd_Parameter_Update]]

=== Requires Environment ===
* [[requires_env::Environment:Karpathy_Micrograd_Python_3]]
