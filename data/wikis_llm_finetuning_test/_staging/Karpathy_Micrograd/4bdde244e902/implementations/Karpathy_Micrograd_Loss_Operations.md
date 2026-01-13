# Implementation: Loss_Operations

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

Concrete tools for computing loss functions using Value arithmetic operations provided by the micrograd library.

=== Description ===

Micrograd does not provide built-in loss functions. Instead, users compose loss computations using the differentiable arithmetic operations on `Value` objects:

- `Value.__sub__` for (prediction - target)
- `Value.__pow__` for squaring errors
- `Value.__add__` for summing (via Python's `sum()`)
- `Value.relu` for hinge loss margin

This approach is educational: it shows that loss functions are just computational graphs like any other.

=== Usage ===

Compose loss functions using Value operations after the forward pass. The resulting `Value` object becomes the root of the computation graph for backpropagation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/karpathy/micrograd micrograd]
* '''File:''' micrograd/engine.py
* '''Key Methods:'''
** `Value.__sub__`: Lines 78-79
** `Value.__pow__`: Lines 35-43
** `Value.__add__`: Lines 13-22
** `Value.relu`: Lines 45-52

=== Key Signatures ===
<syntaxhighlight lang="python">
class Value:
    def __sub__(self, other):  # self - other
        """Subtraction via addition of negation.

        Implementation: return self + (-other)
        Creates graph: self -> __neg__(other) -> __add__
        """
        return self + (-other)

    def __pow__(self, other):
        """Power operation (exponentiation).

        Args:
            other: int or float (not Value - only constant powers)

        Returns:
            Value: self ** other with backward for gradient

        Gradient: d/dx[x^n] = n * x^(n-1)
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """Rectified Linear Unit activation.

        Returns:
            Value: max(0, self) with backward for gradient

        Gradient: 1 if self > 0 else 0
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from micrograd.engine import Value
# Loss operations are methods on Value, no separate import needed
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| predictions || list[Value] || Yes || Network outputs from forward pass
|-
| targets || list[float] || Yes || Ground truth values
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| loss || Value || Single scalar loss value (root of computation graph)
|}

== Usage Examples ==

=== Mean Squared Error Loss ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [4, 1])

# Training data
xs = [[2.0, 3.0], [-1.0, -1.0], [3.0, -2.0]]
ys = [1.0, -1.0, 1.0]

# Forward pass
ypred = [model(x) for x in xs]

# MSE Loss: sum of squared errors
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

print(f"MSE Loss: {loss.data:.4f}")

# Now loss is the root of the graph
# loss._prev contains the sum operation's children
# which trace back through all predictions to all parameters
</syntaxhighlight>

=== Hinge Loss (SVM-style) ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [4, 1])

# Binary classification with -1/+1 labels
xs = [[2.0, 3.0], [-1.0, -1.0], [3.0, -2.0], [0.5, 1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

# Forward pass
scores = [model(x) for x in xs]

# Hinge Loss: sum(max(0, 1 - y*score))
# Using relu() for max(0, x)
loss = sum((1 + -yi*scorei).relu() for yi, scorei in zip(ys, scores))

print(f"Hinge Loss: {loss.data:.4f}")
</syntaxhighlight>

=== Loss with L2 Regularization ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [16, 16, 1])
alpha = 0.01  # Regularization strength

xs = [[2.0, 3.0], [-1.0, -1.0]]
ys = [1.0, -1.0]

# Data loss
ypred = [model(x) for x in xs]
data_loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

# Regularization loss (L2 penalty on weights)
reg_loss = alpha * sum(p**2 for p in model.parameters())

# Total loss
loss = data_loss + reg_loss

print(f"Data loss: {data_loss.data:.4f}")
print(f"Reg loss: {reg_loss.data:.4f}")
print(f"Total loss: {loss.data:.4f}")
</syntaxhighlight>

=== Understanding the Graph Structure ===
<syntaxhighlight lang="python">
from micrograd.engine import Value

# Simple example: (pred - target)^2
pred = Value(0.8)  # Network prediction
target = 1.0       # Ground truth

# Build loss graph
diff = pred - target           # Creates subtraction node
squared = diff ** 2            # Creates power node

print(f"Loss value: {squared.data}")  # 0.04

# Inspect graph structure
print(f"squared._prev: {squared._prev}")  # {diff}
print(f"squared._op: {squared._op}")      # '**2'
print(f"diff._prev: {diff._prev}")        # {pred, Value(-1.0)}

# The graph traces: squared -> diff -> pred
# backward() will follow this path in reverse
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Karpathy_Micrograd_Loss_Computation]]

=== Requires Environment ===
* [[requires_env::Environment:Karpathy_Micrograd_Python_3]]
