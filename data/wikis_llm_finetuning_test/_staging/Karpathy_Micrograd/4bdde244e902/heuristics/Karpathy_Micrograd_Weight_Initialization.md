# Heuristic: Weight_Initialization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|micrograd|https://github.com/karpathy/micrograd]]
* [[source::Discussion|Neural Network Initialization|https://cs231n.github.io/neural-networks-2/#init]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Weight initialization strategy using uniform random values in [-1, 1] with zero-initialized biases for simple networks.

=== Description ===

Micrograd uses a straightforward weight initialization scheme: weights are drawn from a uniform distribution over [-1, 1], and biases are initialized to zero. This simple approach works well for small educational networks but differs from more sophisticated initialization schemes used in production deep learning (e.g., Xavier/Glorot, Kaiming/He initialization).

The uniform [-1, 1] distribution provides sufficient randomness to break symmetry while keeping values bounded. Zero-initialized biases are a common choice as they don't bias the initial output distribution.

=== Usage ===

Apply this initialization pattern when building simple neural networks for educational purposes or prototyping. For deeper networks or production use, consider more sophisticated initialization schemes that account for layer width (Xavier) or activation functions (Kaiming).

== The Insight (Rule of Thumb) ==

* **Action:** Initialize weights uniformly from `random.uniform(-1, 1)`, biases to zero
* **Value:** Weight range: [-1, 1], Bias: 0
* **Trade-off:** Simple and effective for small networks; may cause vanishing/exploding gradients in deep networks

== Reasoning ==

This initialization scheme breaks symmetry (neurons learn different features) while keeping initial activations bounded. For the small 2-3 layer networks typical in micrograd demos, this works well. The simplicity also aids educational understanding - students can reason about initial value ranges directly.

For deeper networks, the variance of activations can grow or shrink exponentially with depth when using fixed-range initialization, leading to vanishing or exploding gradients. Production frameworks address this with width-dependent scaling (Xavier: `1/sqrt(fan_in)`, Kaiming: `sqrt(2/fan_in)`).

== Code Evidence ==

Weight initialization from `nn.py:15-17`:
<syntaxhighlight lang="python">
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
</syntaxhighlight>

Note: Each weight is independently sampled; there's no fan-in/fan-out scaling.

== Related Pages ==

* [[used_by::Implementation:Karpathy_Micrograd_MLP_Init]]
* [[used_by::Principle:Karpathy_Micrograd_Network_Architecture_Definition]]
