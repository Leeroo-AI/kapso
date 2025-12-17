# Heuristic: huggingface_transformers_Quantized_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|PEFT and Quantization|https://huggingface.co/docs/transformers/peft]]
|-
! Domains
| [[domain::Optimization]], [[domain::Quantization]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Rules and compatibility requirements for fine-tuning quantized models with PEFT adapters.

=== Description ===

Quantized models (INT8/INT4) cannot be directly fine-tuned because the quantized weights are not differentiable in the traditional sense. Instead, trainable adapter layers (LoRA, QLoRA) must be attached on top of the frozen quantized base model. The Trainer validates these constraints and provides clear error messages when misconfigured.

=== Usage ===

Apply this heuristic when:
- Attempting to **fine-tune a quantized model**
- Setting up **QLoRA training** (4-bit base + LoRA adapters)
- Debugging **"cannot fine-tune quantized model"** errors
- Choosing between quantization methods for training vs inference

This heuristic prevents wasted time on incompatible training configurations.

== The Insight (Rule of Thumb) ==

* **Rule 1**: Quantized base models REQUIRE PEFT adapters for training
* **Rule 2**: Not all quantization methods support training—check `is_trainable`
* **Rule 3**: Quantized models CANNOT be combined with `torch.compile()`
* **Value**: Use `peft` >= 0.18.0 with compatible quantization (BnB INT8/INT4)

**Compatible Quantization Methods for Training:**
- BitsAndBytes INT8 (`load_in_8bit=True`)
- BitsAndBytes INT4/NF4 (`load_in_4bit=True`)
- Some QAT (Quantization-Aware Training) methods

**Incompatible with Training:**
- Pre-quantized GPTQ models (inference-only)
- AWQ models (inference-only)
- Most other weight-only quantization methods

== Reasoning ==

**Why PEFT is required:**

1. **Frozen Quantized Weights**: Quantized weights are stored in a compressed format (INT4/INT8) that doesn't support standard backpropagation. The quantization process is not differentiable.

2. **Memory Efficiency**: PEFT adds only ~0.1-1% trainable parameters on top of frozen base model, making training feasible on consumer hardware.

3. **Gradient Flow**: LoRA adapters provide a differentiable path for gradients, while the quantized base model remains frozen.

**Why torch.compile is incompatible:**

`torch.compile()` traces the model graph, but quantized layers have special dispatch logic that breaks during compilation. The combination is explicitly blocked.

== Code Evidence ==

Quantized training validation from `trainer.py:500-529`:

<syntaxhighlight lang="python">
_is_quantized_and_base_model = getattr(model, "is_quantized", False) and not getattr(
    model, "_hf_peft_config_loaded", False
)
_quantization_method_supports_training = (
    getattr(model, "hf_quantizer", None) is not None and model.hf_quantizer.is_trainable
)

_is_model_quantized_and_qat_trainable = getattr(model, "hf_quantizer", None) is not None and getattr(
    model.hf_quantizer, "is_qat_trainable", False
)

# Filter out quantized + compiled models
if _is_quantized_and_base_model and hasattr(model, "_orig_mod"):
    raise ValueError(
        "You cannot fine-tune quantized model with `torch.compile()` make sure to pass a "
        "non-compiled model when fine-tuning a quantized model with PEFT"
    )
</syntaxhighlight>

PEFT requirement enforcement from `trainer.py:518-529`:

<syntaxhighlight lang="python">
# At this stage the model is already loaded
if _is_quantized_and_base_model and not _is_peft_model(model) and not _is_model_quantized_and_qat_trainable:
    raise ValueError(
        "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters "
        "on top of the quantized model to correctly perform fine-tuning. Please see: "
        "https://huggingface.co/docs/transformers/peft for more details"
    )
elif _is_quantized_and_base_model and not _quantization_method_supports_training:
    raise ValueError(
        f"The model you are trying to fine-tune is quantized with "
        f"{model.hf_quantizer.quantization_config.quant_method} but that quantization method "
        "do not support training..."
    )
</syntaxhighlight>

Trainability property from `quantizer_bnb_8bit.py:146-148`:

<syntaxhighlight lang="python">
@property
def is_trainable(self) -> bool:
    return True
</syntaxhighlight>

== Best Practices ==

1. **QLoRA Setup Pattern**:
   ```python
   from transformers import AutoModelForCausalLM, BitsAndBytesConfig
   from peft import get_peft_model, LoraConfig

   # Load quantized base model
   bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
   model = AutoModelForCausalLM.from_pretrained("model_name", quantization_config=bnb_config)

   # Add trainable adapters
   peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
   model = get_peft_model(model, peft_config)

   # Now model can be trained with Trainer
   ```

2. **Check trainability before training**:
   ```python
   if hasattr(model, "hf_quantizer"):
       print(f"Trainable: {model.hf_quantizer.is_trainable}")
   ```

== Related Pages ==

* [[uses_heuristic::Implementation:huggingface_transformers_BitsAndBytesConfig]]
* [[uses_heuristic::Implementation:huggingface_transformers_Trainer_init]]
* [[uses_heuristic::Workflow:huggingface_transformers_Quantization]]
* [[uses_heuristic::Workflow:huggingface_transformers_Training]]
