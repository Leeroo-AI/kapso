# Principle: Model_Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|LLM.int8(): 8-bit Matrix Multiplication for Transformers|https://arxiv.org/abs/2208.07339]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Deep_Learning]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for loading pre-trained language models with optional quantization to reduce memory footprint while preserving model capabilities for fine-tuning.

=== Description ===

Model Loading in the context of QLoRA fine-tuning involves loading a pre-trained transformer model from HuggingFace Hub with 4-bit quantization using the NF4 (Normal Float 4) data type. This approach, introduced in the QLoRA paper, allows training of large language models on consumer hardware by reducing the memory requirements by approximately 75% compared to full-precision models.

The key innovation is using NF4 quantization during inference while performing backpropagation through the quantized weights. The quantized base model remains frozen, and only small LoRA adapter weights are trained in full precision. This combination enables fine-tuning of models like Llama-3.2-70B on a single GPU.

=== Usage ===

Use this principle when:
* Fine-tuning large language models with limited GPU memory
* Training models that would otherwise exceed available VRAM in full precision
* Performing parameter-efficient fine-tuning (PEFT) with LoRA adapters
* Starting a QLoRA training workflow

This is the essential first step in any Unsloth fine-tuning pipeline. The loaded model serves as the frozen base for LoRA adapter injection.

== Theoretical Basis ==

The NF4 quantization scheme maps floating-point weights to 4-bit representations optimized for normally distributed data:

1. **Absmax Quantization**: Weights are normalized by the absolute maximum value in each block
2. **Normal Float Mapping**: The 16 quantization levels are chosen to minimize expected squared error for normally distributed values
3. **Double Quantization**: Quantization constants are themselves quantized to reduce memory overhead

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Abstract algorithm for 4-bit model loading
config = AutoConfig.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="sequential"
)
# Model now uses ~25% of original memory
</syntaxhighlight>

The trade-off is slightly reduced precision in activations, but empirically this has minimal impact on fine-tuning quality when combined with LoRA.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]

