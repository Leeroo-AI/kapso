# Implementation: MLP_Call

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Micrograd|https://github.com/karpathy/micrograd]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Neural_Networks]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Concrete tool for executing forward pass through a neural network provided by the micrograd library.

=== Description ===

The `MLP.__call__` method executes the forward pass by sequentially applying each layer to the input. This method:
1. Takes an input vector (list of floats or Values)
2. Passes it through each layer in sequence
3. Returns the final output (a single Value for single-output networks)

The forward pass simultaneously computes output values and constructs the computation graph via the `_prev` references in each `Value` node.

=== Usage ===

Call the MLP object directly (it acts as a callable) when:
- Making predictions on new data
- Computing predictions during training for loss calculation
- Testing the network on validation data

The input must match the `nin` parameter used when constructing the MLP.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/karpathy/micrograd micrograd]
* '''File:''' micrograd/nn.py
* '''Lines:''' 51-54

=== Signature ===
<syntaxhighlight lang="python">
class MLP(Module):
    def __call__(self, x):
        """Execute forward pass through all layers.

        Args:
            x: list[float] or list[Value]
                Input vector. Length must equal `nin` from __init__.

        Returns:
            Value or list[Value]:
                Network output. Single Value if final layer has 1 neuron,
                otherwise list of Values (one per output neuron).
        """
        for layer in self.layers:
            x = layer(x)
        return x
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| x || list[float] or list[Value] || Yes || Input feature vector; length must match network's input dimension
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || Value || Network prediction (for single-output networks)
|-
| output || list[Value] || Network predictions (for multi-output networks)
|}

== Usage Examples ==

=== Single Prediction ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

# Create network
model = MLP(2, [4, 4, 1])

# Single input
x = [1.0, 2.0]

# Forward pass
output = model(x)  # Returns a single Value

print(output.data)  # The actual prediction value
# Example: -0.234 (random weights, untrained)
</syntaxhighlight>

=== Batch Predictions (Manual Loop) ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [16, 16, 1])

# Training data
xs = [
    [2.0, 3.0],
    [-1.0, -1.0],
    [3.0, -2.0],
]

# Forward pass on all samples (no batching in micrograd)
predictions = [model(x) for x in xs]

# Each prediction is a Value object
for i, pred in enumerate(predictions):
    print(f"Input {xs[i]} -> Prediction: {pred.data:.4f}")
</syntaxhighlight>

=== Forward Pass for Training ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [4, 1])

# Training loop forward pass
xs = [[2.0, 3.0], [-1.0, -1.0]]
ys = [1.0, -1.0]

# Compute predictions (this builds the computation graph)
ypred = [model(x) for x in xs]

# Now compute loss (this extends the graph)
# The graph now connects: inputs -> network -> predictions -> loss
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

print(f"Loss: {loss.data:.4f}")
# Ready for loss.backward() to compute gradients
</syntaxhighlight>

== Supporting Components ==

=== Layer.__call__ ===
<syntaxhighlight lang="python">
def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out
</syntaxhighlight>
* '''Location:''' micrograd/nn.py:35-37
* '''Purpose:''' Apply all neurons in parallel, return single Value or list

=== Neuron.__call__ ===
<syntaxhighlight lang="python">
def __call__(self, x):
    act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
    return act.relu() if self.nonlin else act
</syntaxhighlight>
* '''Location:''' micrograd/nn.py:20-22
* '''Purpose:''' Compute weighted sum + bias, optionally apply ReLU

=== Value Operations Used ===
The forward pass uses these Value operations to build the graph:
- `Value.__mul__` (weight * input)
- `Value.__add__` (sum of products + bias)
- `Value.relu` (activation function)

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Karpathy_Micrograd_Forward_Pass_Computation]]

=== Requires Environment ===
* [[requires_env::Environment:Karpathy_Micrograd_Python_3]]
