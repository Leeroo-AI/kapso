# QLoRA Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
* [[source::Paper|QLoRA Paper|https://arxiv.org/abs/2305.14314]]
* [[source::Blog|Fine-tuning LLMs on Consumer Hardware|https://pytorch.org/blog/finetune-llms/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::Quantization]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

End-to-end process for fine-tuning large language models on consumer GPUs using 4-bit quantization (QLoRA), enabling training of 7B+ parameter models on hardware with 16-24GB VRAM.

=== Description ===

QLoRA combines 4-bit NormalFloat quantization with LoRA adapters to dramatically reduce memory requirements during fine-tuning. The base model is loaded in 4-bit precision using bitsandbytes, while LoRA adapter weights remain in full precision (float32) for stable training. This allows training models that would otherwise require 4x more memory, making 7B-13B parameter models accessible on consumer GPUs.

=== Usage ===

Execute this workflow when:
* You have limited GPU memory (8-24GB VRAM)
* You want to fine-tune large models (7B+ parameters) that don't fit in memory at full precision
* You're doing instruction tuning, chat fine-tuning, or domain adaptation
* Training speed is acceptable (4-bit is slower than full precision due to dequantization overhead)

== Execution Steps ==

=== Step 1: Configure Quantization ===
[[step::Principle:huggingface_peft_Quantization_Configuration]]

Create a `BitsAndBytesConfig` specifying 4-bit quantization parameters. NormalFloat4 (NF4) is the recommended quantization type as it's optimized for normally distributed weights.

'''Key parameters:'''
* `load_in_4bit=True`: Enable 4-bit quantization
* `bnb_4bit_quant_type="nf4"`: Use NormalFloat4 quantization
* `bnb_4bit_compute_dtype=torch.bfloat16`: Compute in bfloat16 for speed
* `bnb_4bit_use_double_quant=True`: Enable nested quantization for additional memory savings

=== Step 2: Load Quantized Model ===
[[step::Principle:huggingface_peft_Quantized_Model_Loading]]

Load the base model with the quantization configuration applied. The model weights are automatically quantized during loading, reducing memory footprint by approximately 4x compared to float16.

'''What happens:'''
* Model weights are loaded and quantized to 4-bit on-the-fly
* Memory footprint reduced from ~14GB to ~4GB for a 7B model
* Model is automatically placed on GPU with `device_map="auto"`
* Attention and linear layers use quantized weights

=== Step 3: Prepare for K-bit Training ===
[[step::Principle:huggingface_peft_Kbit_Training_Preparation]]

Apply `prepare_model_for_kbit_training()` to enable gradient computation for quantized models. This enables input embeddings to require gradients (necessary since quantized layers don't propagate gradients directly) and sets up gradient checkpointing.

'''Modifications applied:'''
* Enable `input_require_grads` for the embedding layer
* Disable caching for training mode
* Optionally enable gradient checkpointing for additional memory savings

=== Step 4: Configure LoRA for QLoRA ===
[[step::Principle:huggingface_peft_QLoRA_Configuration]]

Create a `LoraConfig` specifically tuned for QLoRA training. The configuration is similar to standard LoRA but typically uses slightly different hyperparameters optimized for quantized training.

'''QLoRA-specific considerations:'''
* Target all linear layers for maximum effectiveness: `target_modules="all-linear"`
* Use moderate rank (r=16-64) as quantization adds implicit regularization
* Set `task_type` appropriately for your use case
* LoRA weights remain in full precision (float32) despite base model being quantized

=== Step 5: Create QLoRA Model ===
[[step::Principle:huggingface_peft_PEFT_Model_Creation]]

Wrap the quantized model with LoRA configuration using `get_peft_model()`. The LoRA layers are injected in full precision, operating on dequantized outputs from the 4-bit base model.

'''Architecture:'''
* Base model: 4-bit quantized weights (frozen)
* LoRA A/B matrices: Full precision trainable weights
* Forward pass: dequantize → compute → apply LoRA delta

=== Step 6: Execute Training ===
[[step::Principle:huggingface_peft_QLoRA_Training_Execution]]

Run training with appropriate settings for quantized models. Gradient accumulation is often needed to achieve effective batch sizes while staying within memory limits.

'''Training considerations:'''
* Use gradient accumulation to increase effective batch size
* Enable gradient checkpointing if memory is still tight
* Use paged optimizers (e.g., `paged_adamw_8bit`) for additional memory savings
* Monitor for NaN losses which can indicate precision issues

=== Step 7: Save QLoRA Adapter ===
[[step::Principle:huggingface_peft_Adapter_Serialization]]

Save the trained LoRA adapter weights. Only the full-precision LoRA weights are saved (not the quantized base model), resulting in a small checkpoint that can be loaded onto any compatible base model.

'''Output:'''
* Small adapter checkpoint (same size as regular LoRA)
* Can be loaded with quantized or non-quantized base models
* Base model reference stored in config for reproducibility

== Execution Diagram ==

{{#mermaid:graph TD
    A[Configure Quantization] --> B[Load Quantized Model]
    B --> C[Prepare K-bit Training]
    C --> D[Configure QLoRA]
    D --> E[Create QLoRA Model]
    E --> F[Execute Training]
    F --> G[Save Adapter]
}}

== Related Pages ==

* [[step::Principle:huggingface_peft_Quantization_Configuration]]
* [[step::Principle:huggingface_peft_Quantized_Model_Loading]]
* [[step::Principle:huggingface_peft_Kbit_Training_Preparation]]
* [[step::Principle:huggingface_peft_QLoRA_Configuration]]
* [[step::Principle:huggingface_peft_PEFT_Model_Creation]]
* [[step::Principle:huggingface_peft_QLoRA_Training_Execution]]
* [[step::Principle:huggingface_peft_Adapter_Serialization]]
