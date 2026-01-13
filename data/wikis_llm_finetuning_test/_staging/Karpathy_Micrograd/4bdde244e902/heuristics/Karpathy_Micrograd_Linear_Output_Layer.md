# Heuristic: Linear_Output_Layer

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|micrograd|https://github.com/karpathy/micrograd]]
* [[source::Discussion|Output Layer Design|https://cs231n.github.io/neural-networks-1/#architectures]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Neural_Network_Design]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Use linear (non-ReLU) activation on the final layer to allow positive and negative output scores for classification.

=== Description ===

The MLP class in micrograd automatically configures the last layer to be linear (no ReLU activation) while all hidden layers use ReLU. This is essential for binary classification where the output score should range from negative to positive values, allowing the decision boundary to be at zero.

If ReLU were applied to the output, all negative scores would be clipped to zero, preventing the model from expressing "confidently negative" predictions. The implementation achieves this through: `nonlin=i!=len(nouts)-1` - setting `nonlin=False` only for the final layer index.

=== Usage ===

Always use a linear output layer for classification tasks where the loss function expects unbounded scores (e.g., hinge loss, cross-entropy). The ReLU non-linearity should only be applied to hidden layers to enable learning complex non-linear patterns.

== The Insight (Rule of Thumb) ==

* **Action:** Set `nonlin=False` for the output layer; use `nonlin=True` for all hidden layers
* **Value:** Hidden layers: ReLU activation; Output layer: Linear (identity) activation
* **Trade-off:** None - this is the correct design pattern for classification

== Reasoning ==

The choice of output activation depends on the loss function:
- **Hinge loss** (SVM): Expects unbounded scores where sign indicates class
- **Cross-entropy**: Typically followed by softmax, so pre-softmax logits should be unbounded
- **MSE regression**: Unbounded outputs match unbounded targets

ReLU in hidden layers is essential for learning non-linear decision boundaries - without it, stacking layers gives no more expressiveness than a single linear layer. But ReLU on output would destroy the ability to express negative confidence.

The micrograd implementation elegantly encodes this pattern: the `nonlin` parameter controls ReLU application, and the MLP constructor automatically sets `nonlin=False` for exactly the last layer.

== Code Evidence ==

Output layer configuration from `nn.py:45-49`:
<syntaxhighlight lang="python">
class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
</syntaxhighlight>

Explanation:
- `i != len(nouts)-1` evaluates to `True` for all hidden layers (0, 1, ..., n-2)
- `i != len(nouts)-1` evaluates to `False` for the output layer (n-1)
- So hidden layers get `nonlin=True` (ReLU), output layer gets `nonlin=False` (linear)

Neuron applying the nonlinearity from `nn.py:20-22`:
<syntaxhighlight lang="python">
def __call__(self, x):
    act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
    return act.relu() if self.nonlin else act
</syntaxhighlight>

== Related Pages ==

* [[used_by::Implementation:Karpathy_Micrograd_MLP_Init]]
* [[used_by::Implementation:Karpathy_Micrograd_MLP_Call]]
* [[used_by::Principle:Karpathy_Micrograd_Network_Architecture_Definition]]
* [[used_by::Principle:Karpathy_Micrograd_Forward_Pass_Computation]]
