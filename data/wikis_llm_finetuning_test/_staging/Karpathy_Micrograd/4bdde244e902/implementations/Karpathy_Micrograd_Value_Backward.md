# Implementation: Value_Backward

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Micrograd|https://github.com/karpathy/micrograd]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Automatic_Differentiation]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Concrete tool for computing gradients via reverse-mode automatic differentiation provided by the micrograd library.

=== Description ===

The `Value.backward()` method implements backpropagation:
1. Builds topological ordering of the computation graph via DFS
2. Sets the output gradient to 1.0
3. Traverses nodes in reverse order
4. Calls each node's `_backward()` closure to propagate gradients

After calling `backward()`, every `Value` in the graph has its `.grad` attribute populated with ∂loss/∂value.

=== Usage ===

Call `backward()` on the loss value (root of computation graph) after:
1. Forward pass through the network
2. Loss computation

The method populates gradients in-place—no return value. Access gradients via `parameter.grad`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/karpathy/micrograd micrograd]
* '''File:''' micrograd/engine.py
* '''Lines:''' 54-70

=== Signature ===
<syntaxhighlight lang="python">
class Value:
    def backward(self):
        """Compute gradients for all Values in the computation graph.

        This method implements reverse-mode automatic differentiation:
        1. Topologically sorts all nodes reachable from self
        2. Sets self.grad = 1.0 (d(self)/d(self) = 1)
        3. Traverses in reverse order, calling _backward() on each node

        After calling this method, all Values in the graph have their
        .grad attribute populated with d(self)/d(value).

        Returns:
            None (gradients are stored in-place in .grad attributes)

        Note:
            Call zero_grad() before backward() to clear previous gradients,
            otherwise gradients will accumulate across multiple backward passes.
        """
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule
        self.grad = 1
        for v in reversed(topo):
            v._backward()
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from micrograd.engine import Value
# backward() is a method on Value, called on the loss
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| self || Value || Yes || The root node (typically loss value) to backpropagate from
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (in-place) || None || Gradients stored in .grad attribute of all Values in graph
|}

== Usage Examples ==

=== Basic Backward Pass ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [4, 1])

# Forward pass
x = [2.0, 3.0]
y_true = 1.0

pred = model(x)
loss = (pred - y_true) ** 2

print(f"Loss: {loss.data:.4f}")
print(f"Gradients before backward: {model.parameters()[0].grad}")  # 0

# Backward pass
loss.backward()

print(f"Gradients after backward: {model.parameters()[0].grad}")  # non-zero
</syntaxhighlight>

=== Full Training Step ===
<syntaxhighlight lang="python">
from micrograd.nn import MLP

model = MLP(2, [4, 4, 1])
learning_rate = 0.01

xs = [[2.0, 3.0], [-1.0, -1.0], [3.0, -2.0]]
ys = [1.0, -1.0, 1.0]

# Forward pass
ypred = [model(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

# Zero gradients (important if not first iteration!)
model.zero_grad()

# Backward pass - computes all gradients
loss.backward()

# Now update parameters using gradients
for p in model.parameters():
    p.data -= learning_rate * p.grad

print(f"Loss: {loss.data:.4f}")
</syntaxhighlight>

=== Inspecting Gradient Flow ===
<syntaxhighlight lang="python">
from micrograd.engine import Value

# Build a simple computation graph
a = Value(2.0)
b = Value(3.0)
c = a * b      # c = 6.0
d = c + a      # d = 8.0 (note: 'a' is used twice)
e = d ** 2     # e = 64.0

# Backward from e
e.backward()

# Check gradients
print(f"e.grad: {e.grad}")  # 1.0 (root)
print(f"d.grad: {d.grad}")  # 2*d = 16.0 (d/de of e=d^2)
print(f"c.grad: {c.grad}")  # 16.0 (from d = c + a)
print(f"a.grad: {a.grad}")  # 16.0 (from d) + 16.0*b (from c) = 16 + 48 = 64
print(f"b.grad: {b.grad}")  # 16.0*a (from c) = 32
</syntaxhighlight>

=== Understanding _backward Closures ===
<syntaxhighlight lang="python">
from micrograd.engine import Value

# Multiplication example
a = Value(2.0)
b = Value(3.0)
c = a * b  # c.data = 6.0

# The _backward closure captures: self=a, other=b, out=c
# When called, it does:
#   a.grad += b.data * c.grad  (derivative w.r.t. first operand)
#   b.grad += a.data * c.grad  (derivative w.r.t. second operand)

c.grad = 1.0  # Simulate being the output
c._backward()

print(f"a.grad: {a.grad}")  # 3.0 (= b.data * 1.0)
print(f"b.grad: {b.grad}")  # 2.0 (= a.data * 1.0)
</syntaxhighlight>

== Internal Mechanics ==

=== Topological Sort ===
The DFS-based topological sort ensures correct gradient flow:
<syntaxhighlight lang="python">
# Graph: a -> c -> d -> e
#        b -> c
#        a -----> d

# After topological sort: [a, b, c, d, e] or [b, a, c, d, e]
# Reversed: [e, d, c, a, b] or [e, d, c, b, a]
# Either ordering is valid - all dependencies are respected
</syntaxhighlight>

=== Gradient Accumulation ===
The `+=` in `_backward` functions enables gradient accumulation for multi-use values:
<syntaxhighlight lang="python">
# When 'a' is used in both c=a*b and d=c+a:
# a.grad += contribution from c
# a.grad += contribution from d
# Final a.grad = sum of all paths from output to a
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Karpathy_Micrograd_Backward_Pass]]

=== Requires Environment ===
* [[requires_env::Environment:Karpathy_Micrograd_Python_3]]
