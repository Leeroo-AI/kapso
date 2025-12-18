{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Doc|PEFT Quantization|https://huggingface.co/docs/peft/developer_guides/quantization]]
|-
! Domains
| [[domain::Quantization]], [[domain::Training]], [[domain::Memory_Efficiency]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for preparing quantized models for stable training by ensuring proper dtype handling and gradient flow.

=== Description ===

K-bit Training Preparation addresses numerical stability challenges when training with quantized models. Key preparations include:
1. Casting layer norms to float32 (prevents gradient instability)
2. Enabling gradient checkpointing (trades compute for memory)
3. Setting up proper gradient flow through input embeddings
4. Freezing all base model parameters

Without these preparations, QLoRA training often diverges or produces NaN gradients.

=== Usage ===

Apply this immediately after loading a quantized model and before applying PEFT:
* Always call `prepare_model_for_kbit_training()` for quantized models
* Enable gradient checkpointing by default (saves ~30% VRAM)
* Use `use_reentrant=False` for newer PyTorch versions

== Theoretical Basis ==

'''Layer Norm Stability:'''

Layer normalization with quantized inputs can cause instability:
<math>\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta</math>

When <math>x</math> comes from dequantized values, the statistics <math>\mu, \sigma^2</math> can have high variance. Keeping layer norms in float32 ensures:
<syntaxhighlight lang="python">
# Pseudo-code for dtype handling
for module in model.modules():
    if isinstance(module, LayerNorm):
        module.weight = module.weight.to(torch.float32)
        module.bias = module.bias.to(torch.float32)
</syntaxhighlight>

'''Gradient Checkpointing:'''

Standard memory: <math>O(L)</math> activations (L = layers)
Checkpointed: <math>O(\sqrt{L})</math> activations

Trade-off: ~33% extra forward compute for ~50% memory reduction.

'''Input Gradient Flow:'''

QLoRA requires gradients to flow through embeddings:
<syntaxhighlight lang="python">
# Without this hook, gradients don't propagate to LoRA layers
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_prepare_model_for_kbit_training]]
