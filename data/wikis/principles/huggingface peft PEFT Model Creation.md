{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Adapter]], [[domain::Model_Architecture]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for injecting adapter layers into a pre-trained model to create a parameter-efficient fine-tuning setup.

=== Description ===

PEFT Model Creation transforms a standard pre-trained model into an adapter-enabled model. This involves:
1. Freezing all base model parameters
2. Injecting low-rank adapter layers at specified locations
3. Initializing adapter weights (B=0 for no-op initialization)
4. Setting up proper forward pass with adapter contributions

The result is a model where only a small fraction (typically <1%) of parameters are trainable, dramatically reducing memory requirements for fine-tuning.

=== Usage ===

Apply this principle after loading the base model and configuring LoRA:
* **Standard training:** Use default autocast settings for stable training
* **Multi-adapter:** Pass `adapter_name` to create named adapters
* **Mixed adapters:** Use `mixed=True` to combine different PEFT types
* **Memory optimization:** Use `low_cpu_mem_usage=True` for large models

== Theoretical Basis ==

'''Adapter Injection:'''

The model modification replaces original linear layers with adapter-augmented versions:

<syntaxhighlight lang="python">
# Pseudo-code for adapter injection
for layer in find_target_modules(model, config.target_modules):
    # Freeze original weights
    layer.weight.requires_grad = False

    # Add low-rank adapter
    layer.lora_A = nn.Parameter(torch.randn(r, in_features))
    layer.lora_B = nn.Parameter(torch.zeros(out_features, r))
</syntaxhighlight>

'''Forward Pass Modification:'''

The adapted forward pass becomes:
<math>h = W_0 x + \frac{\alpha}{r} \cdot B(Ax)</math>

Where:
* <math>W_0 x</math> is the original (frozen) computation
* <math>B(Ax)</math> is the low-rank adaptation
* <math>\alpha/r</math> is the scaling factor

'''Initialization:'''

Standard LoRA initialization:
* <math>A</math>: Kaiming uniform initialization
* <math>B</math>: Zero initialization

This ensures the adapted model starts as the original model (since <math>BA = 0</math>).

'''Parameter Efficiency:'''

For a linear layer with dimensions <math>d \times k</math>:
* Original parameters: <math>dk</math>
* LoRA parameters: <math>r(d + k)</math>
* Savings ratio: <math>\frac{r(d+k)}{dk} \approx \frac{r}{min(d,k)}</math>

For typical LLM dimensions (d=k=4096) and r=16:
<math>\text{Ratio} = \frac{16 \times 8192}{4096 \times 4096} \approx 0.8\%</math>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_get_peft_model]]
