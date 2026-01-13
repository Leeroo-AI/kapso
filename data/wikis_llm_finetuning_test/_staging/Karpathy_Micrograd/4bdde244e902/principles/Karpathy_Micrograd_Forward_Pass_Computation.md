# Principle: Forward_Pass_Computation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Textbook|Deep Learning Book Ch. 6.3|https://www.deeplearningbook.org/contents/mlp.html]]
* [[source::Blog|Karpathy Backprop Lecture|https://www.youtube.com/watch?v=VMj-3S1tku0]]
* [[source::Paper|Backpropagation Original|https://doi.org/10.1038/323533a0]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Neural_Networks]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Principle of computing network output by sequentially propagating input through all layers while building a computation graph for gradient computation.

=== Description ===

The Forward Pass is the first phase of neural network computation where:

1. **Input Propagation:** Data flows from input layer through hidden layers to output
2. **Graph Construction:** Each operation creates a node in the computation graph with references to its operands
3. **Activation Application:** Non-linear functions (ReLU) are applied at each layer (except typically the last)
4. **Output Production:** The final layer produces predictions that can be compared to targets

The computation graph built during the forward pass is essential for backpropagation—it records the sequence of operations so gradients can flow backward through the same path.

=== Usage ===

Apply this principle when:
- Making predictions with a neural network
- Computing loss values during training
- Building the computation graph for subsequent backward pass

The forward pass must be executed before every backward pass because:
- The computation graph needs to capture the current parameter values
- Intermediate values are stored in the graph for gradient computation
- The loss value is computed as the root of the graph

== Theoretical Basis ==

=== Layer-wise Computation ===

For a network with L layers, the forward pass computes:

<math>
\mathbf{h}^{(0)} = \mathbf{x}
</math>
<math>
\mathbf{h}^{(l)} = \sigma(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}) \quad \text{for } l = 1, ..., L-1
</math>
<math>
\mathbf{y} = W^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}
</math>

Where:
- <math>\mathbf{h}^{(l)}</math> is the hidden representation at layer l
- <math>W^{(l)}</math> and <math>\mathbf{b}^{(l)}</math> are layer parameters
- <math>\sigma</math> is the activation function (ReLU)

=== Graph Construction ===

Each operation creates a node that remembers:
1. **Result value:** The computed output
2. **Children:** References to input operands (for gradient flow)
3. **Backward function:** A closure that computes local gradients

'''Pseudo-code:'''
<syntaxhighlight lang="python">
# Forward pass through MLP
def forward(x, layers):
    for layer in layers:
        # Each operation builds the graph
        x = layer(x)  # Creates new Value nodes
    return x  # Final output contains full graph via _prev references
</syntaxhighlight>

=== Neuron Computation Graph ===

A single neuron builds this graph:
<syntaxhighlight lang="text">
x[0] --- (*w[0]) ---\
x[1] --- (*w[1]) ----\
...                   (+) --- (+b) --- (ReLU) --- output
x[n] --- (*w[n]) ----/
</syntaxhighlight>

Each arrow represents a `Value` node with `_prev` pointing to its inputs.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Karpathy_Micrograd_MLP_Call]]
