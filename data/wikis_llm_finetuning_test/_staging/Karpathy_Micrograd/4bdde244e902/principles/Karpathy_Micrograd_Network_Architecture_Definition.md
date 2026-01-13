# Principle: Network_Architecture_Definition

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Universal Approximation Theorem|https://doi.org/10.1016/0893-6080(89)90020-8]]
* [[source::Textbook|Deep Learning Book Ch. 6|https://www.deeplearningbook.org/contents/mlp.html]]
* [[source::Blog|Karpathy Zero to Hero|https://karpathy.ai/zero-to-hero.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Neural_Networks]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Principle of constructing Multi-Layer Perceptron (MLP) networks by composing layers of interconnected neurons with learnable parameters.

=== Description ===

Network Architecture Definition establishes the structure of a neural network before training. This involves:

1. **Hierarchical Composition:** Networks are built from neurons → layers → full network (MLP)
2. **Parameter Initialization:** Weights and biases are initialized (typically randomly) to break symmetry
3. **Activation Functions:** Non-linear activations (ReLU) enable learning complex decision boundaries
4. **Architecture Specification:** The topology (input size, hidden layer sizes, output size) determines the network's capacity

The Universal Approximation Theorem guarantees that a sufficiently wide feedforward network with one hidden layer can approximate any continuous function—but deeper networks often achieve this with fewer parameters.

=== Usage ===

Apply this principle when:
- Designing a new neural network for a specific task
- Determining the number and size of hidden layers
- Balancing model capacity (more parameters) against overfitting risk
- Implementing the network structure before the training loop

The architecture choice directly impacts:
- **Capacity:** Larger networks can represent more complex functions
- **Training time:** More parameters require more computation
- **Generalization:** Over-parameterized networks may overfit

== Theoretical Basis ==

=== Neuron Computation ===

A single neuron computes:

<math>
y = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b)
</math>

Where:
- <math>\mathbf{x} \in \mathbb{R}^n</math> is the input vector
- <math>\mathbf{w} \in \mathbb{R}^n</math> are the weights (learnable)
- <math>b \in \mathbb{R}</math> is the bias (learnable)
- <math>\sigma</math> is the activation function (e.g., ReLU)

=== Layer Computation ===

A layer of m neurons computes:

<math>
\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})
</math>

Where <math>W \in \mathbb{R}^{m \times n}</math> is the weight matrix.

=== MLP Computation ===

An MLP chains multiple layers:

<math>
\mathbf{y} = f_L(f_{L-1}(...f_1(\mathbf{x})))
</math>

'''Pseudo-code Architecture:'''
<syntaxhighlight lang="python">
# MLP as composition of layers
class MLP:
    layers = [Layer_1, Layer_2, ..., Layer_L]

    def forward(x):
        for layer in layers:
            x = layer(x)  # Apply each layer sequentially
        return x
</syntaxhighlight>

=== Activation Functions ===

ReLU (Rectified Linear Unit) is the standard choice:

<math>
\text{ReLU}(x) = \max(0, x)
</math>

Properties:
- Non-saturating (no vanishing gradient for positive inputs)
- Computationally efficient
- Sparse activation (neurons output 0 for negative inputs)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Karpathy_Micrograd_MLP_Init]]
