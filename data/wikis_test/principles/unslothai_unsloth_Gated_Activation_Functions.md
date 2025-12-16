# Principle: Gated Activation Functions

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GLU Variants Improve Transformer|https://arxiv.org/abs/2002.05202]]
* [[source::Paper|GELU Activation Function|https://arxiv.org/abs/1606.08415]]
* [[source::Paper|Attention Is All You Need|https://arxiv.org/abs/1706.03762]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Activation_Functions]], [[domain::Transformers]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
A family of activation functions that combine element-wise gating with nonlinear transformations to improve gradient flow and model expressivity in transformer architectures.

=== Description ===
Gated Activation Functions represent a class of nonlinear transformations that multiply a gating signal with an activation signal, enabling the network to selectively pass or suppress information. This gating mechanism improves upon standard activations like ReLU by providing smoother gradients and better information flow.

The key variants include:

1. **GLU (Gated Linear Unit)**: `output = gate * sigmoid(gate)` - Original gated activation
2. **SwiGLU**: `output = gate * silu(gate)` - Uses SiLU/Swish activation, popular in LLaMA models
3. **GEGLU**: `output = gate * gelu(gate)` - Uses GELU activation, used in Gemma and GLM models
4. **ReGLU**: `output = gate * relu(gate)` - ReLU-based variant

**Problem Solved:**
Standard ReLU activations suffer from "dying neurons" where gradients become zero for negative inputs. Gated activations address this by:
- Providing non-zero gradients across the input range
- Allowing the model to learn optimal gating behavior
- Improving capacity through the multiplicative gating interaction

**Context:**
Gated activations are used in the FFN (Feed-Forward Network) layers of modern transformers, replacing the traditional `ReLU(Wx + b)` pattern with `activation(Wx) * (Vx)` where the model learns both the gate and value projections.

=== Usage ===
Use gated activation functions when:
- Building transformer FFN layers that require improved gradient flow
- Working with models that specify SwiGLU (LLaMA, Mistral) or GEGLU (Gemma, GLM)
- Seeking better model capacity than standard ReLU/GELU alone

Implementation choice depends on the model architecture:
- **SwiGLU**: Default for most modern LLMs (Llama 2/3, Mistral, Qwen)
- **GEGLU**: Used in Google's Gemma family
- **GLU**: Original formulation, less common in modern models

== Theoretical Basis ==
The gating mechanism combines two linear transformations with element-wise multiplication:

'''General Form:'''
<syntaxhighlight lang="python">
# Standard FFN
output = activation(W1 @ x + b1)

# Gated FFN (GLU family)
gate = activation(W_gate @ x + b_gate)  # Gating signal
up = W_up @ x + b_up  # Value/up projection
output = gate * up  # Element-wise multiplication
</syntaxhighlight>

'''Specific Variants:'''
<math>
\text{GLU}(x) = x \odot \sigma(x)
</math>

<math>
\text{SwiGLU}(x) = x \odot \text{SiLU}(x) = x \odot (x \cdot \sigma(x))
</math>

<math>
\text{GEGLU}(x) = x \odot \text{GELU}(x) = x \odot \frac{x}{2}(1 + \text{erf}(\frac{x}{\sqrt{2}}))
</math>

'''GEGLU Derivative (for backpropagation):'''
<math>
\frac{d\text{GEGLU}}{dx} = \frac{1}{2}(1 + \text{erf}(\frac{x}{\sqrt{2}})) + \frac{x}{\sqrt{2\pi}}e^{-x^2/2}
</math>

'''Trade-offs:'''
- **Parameters**: Gated FFN requires 2 projections (gate + up) vs 1, increasing parameters by ~50%
- **Compute**: Additional element-wise operations, but offset by improved convergence
- **Memory**: Fused kernels (as in Unsloth) eliminate intermediate storage overhead

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_geglu_kernel]]
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Dtype_Selection]]
