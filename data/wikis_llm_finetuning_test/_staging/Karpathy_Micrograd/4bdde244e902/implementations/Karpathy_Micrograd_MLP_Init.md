# Implementation: MLP_Init

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

Concrete tool for constructing Multi-Layer Perceptron networks provided by the micrograd library.

=== Description ===

The `MLP` class constructs a feedforward neural network by chaining multiple `Layer` objects. Each layer (except the last) applies ReLU activation; the final layer is linear to allow unbounded outputs suitable for regression or loss computation.

The class hierarchy is:
- `MLP` contains multiple `Layer`s
- `Layer` contains multiple `Neuron`s
- `Neuron` contains `Value` weights and bias

=== Usage ===

Import and instantiate `MLP` when you need to create a neural network for training. The architecture is specified by:
- `nin`: Number of input features (must match your data dimensionality)
- `nouts`: List of layer output sizes (e.g., `[16, 16, 1]` for two hidden layers of 16 neurons and one output)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/karpathy/micrograd micrograd]
* '''File:''' micrograd/nn.py
* '''Lines:''' 45-60

=== Signature ===
<syntaxhighlight lang="python">
class MLP(Module):
    """Multi-Layer Perceptron neural network.

    Args:
        nin: int
            Number of input features.
        nouts: list[int]
            List of output sizes for each layer.
            Example: [16, 16, 1] creates:
            - Hidden layer 1: nin -> 16 neurons (with ReLU)
            - Hidden layer 2: 16 -> 16 neurons (with ReLU)
            - Output layer: 16 -> 1 neuron (linear, no ReLU)
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
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
| nin || int || Yes || Number of input features (dimensionality of input vectors)
|-
| nouts || list[int] || Yes || Output sizes for each layer; last element is the final output dimension
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| self || MLP || Initialized MLP object with random weights in [-1, 1]
|-
| self.layers || list[Layer] || The sequential layers of the network
|}

== Usage Examples ==

=== Basic Binary Classifier ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

# Create network: 2 inputs -> 16 hidden -> 16 hidden -> 1 output
model = MLP(2, [16, 16, 1])

# Architecture breakdown:
# Layer 0: 2 inputs -> 16 outputs (ReLU activation)
# Layer 1: 16 inputs -> 16 outputs (ReLU activation)
# Layer 2: 16 inputs -> 1 output (Linear, no activation)

print(model)
# Output: MLP of [Layer of [ReLUNeuron(2), ...], Layer of [...], Layer of [LinearNeuron(16)]]
</syntaxhighlight>

=== Multi-Class Classifier ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

# 10 input features, 3-way classification
model = MLP(10, [32, 16, 3])

# This creates 3 output neurons for 3 classes
# Use with appropriate multi-class loss (e.g., softmax cross-entropy)
</syntaxhighlight>

=== Simple XOR Network ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

# Minimal network for XOR problem
model = MLP(2, [4, 1])

# 2 inputs (binary features)
# 4 hidden neurons (enough for non-linear boundary)
# 1 output (binary classification)
</syntaxhighlight>

=== Checking Parameter Count ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(3, [4, 4, 1])
params = model.parameters()

print(f"Total parameters: {len(params)}")
# Layer 0: 3*4 weights + 4 biases = 16
# Layer 1: 4*4 weights + 4 biases = 20
# Layer 2: 4*1 weights + 1 bias = 5
# Total: 41 parameters
</syntaxhighlight>

== Supporting Components ==

The MLP class relies on:

=== Layer Class ===
<syntaxhighlight lang="python">
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
</syntaxhighlight>
* '''Location:''' micrograd/nn.py:30-43
* '''Purpose:''' Groups multiple neurons with the same input dimension

=== Neuron Class ===
<syntaxhighlight lang="python">
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
</syntaxhighlight>
* '''Location:''' micrograd/nn.py:13-28
* '''Purpose:''' Single computational unit with weights, bias, and optional ReLU

=== Value Class ===
<syntaxhighlight lang="python">
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
</syntaxhighlight>
* '''Location:''' micrograd/engine.py:2-11
* '''Purpose:''' Scalar wrapper enabling automatic differentiation

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Karpathy_Micrograd_Network_Architecture_Definition]]

=== Requires Environment ===
* [[requires_env::Environment:Karpathy_Micrograd_Python_3]]
