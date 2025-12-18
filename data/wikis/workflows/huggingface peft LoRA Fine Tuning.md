# LoRA Fine-Tuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
* [[source::Paper|LoRA Paper|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

End-to-end process for parameter-efficient fine-tuning of transformer models using Low-Rank Adaptation (LoRA), training only ~0.1-1% of model parameters while achieving performance comparable to full fine-tuning.

=== Description ===

This workflow outlines the standard procedure for applying LoRA adapters to pretrained transformer models. LoRA injects trainable low-rank decomposition matrices into transformer layers (typically attention projections), keeping the original model weights frozen. The process covers model preparation, adapter configuration, training loop execution, and adapter serialization. Only the small adapter weights (typically a few MB) need to be saved, rather than the entire model.

=== Usage ===

Execute this workflow when you need to adapt a pretrained model to a downstream task (classification, generation, etc.) and want to minimize GPU memory usage and training time. Suitable when:
* You have limited GPU memory (8-24GB)
* You need to train multiple task-specific adapters from the same base model
* You want to preserve the original model capabilities while adding specialized knowledge

== Execution Steps ==

=== Step 1: Load Base Model ===
[[step::Principle:huggingface_peft_Base_Model_Loading]]

Load the pretrained transformer model from HuggingFace Hub or local path. The model should be in evaluation mode with all parameters frozen by default. Consider setting `device_map` for automatic device placement on multi-GPU systems.

'''Key considerations:'''
* Use `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`, or task-specific auto classes
* Set `torch_dtype` appropriately (float16/bfloat16 for memory efficiency)
* Enable `device_map="auto"` for large models spanning multiple GPUs

=== Step 2: Configure LoRA Adapter ===
[[step::Principle:huggingface_peft_LoRA_Configuration]]

Create a `LoraConfig` object specifying the adapter hyperparameters. This defines which layers receive adapter injection, the rank of the decomposition, and training behavior.

'''Core parameters:'''
* `r` (rank): Dimension of the low-rank matrices (commonly 8-64)
* `lora_alpha`: Scaling factor, typically set to `r` or `2*r`
* `target_modules`: List of module names to inject adapters (e.g., `["q_proj", "v_proj"]`)
* `task_type`: Task type for correct forward pass (CAUSAL_LM, SEQ_2_SEQ_LM, etc.)
* `lora_dropout`: Dropout probability for regularization

=== Step 3: Create PEFT Model ===
[[step::Principle:huggingface_peft_PEFT_Model_Creation]]

Wrap the base model with the LoRA configuration using `get_peft_model()`. This injects adapter layers into the specified target modules and freezes all original parameters. The resulting model has trainable parameters only in the LoRA layers.

'''What happens:'''
* Original weight matrix W remains frozen
* Two small matrices A (r x d) and B (d x r) are added: output = W(x) + B(A(x))
* Only A and B are trainable (typically <1% of total parameters)

=== Step 4: Prepare for Training ===
[[step::Principle:huggingface_peft_Training_Preparation]]

Set up the training loop with appropriate optimizer, learning rate scheduler, and data loaders. For gradient checkpointing compatibility with quantized models, call `prepare_model_for_kbit_training()` if needed.

'''Considerations:'''
* Use higher learning rates than full fine-tuning (1e-4 to 3e-4 typical)
* Enable gradient checkpointing for memory efficiency
* Set up appropriate evaluation metrics for your task

=== Step 5: Execute Training Loop ===
[[step::Principle:huggingface_peft_Training_Execution]]

Run the training loop using either a custom loop or HuggingFace Trainer. Only the LoRA parameters receive gradient updates; the base model remains frozen.

'''Training approaches:'''
* Use `transformers.Trainer` with standard training arguments
* Or implement custom loop with `model.train()`, forward pass, backward pass, optimizer step
* Monitor loss convergence - LoRA typically converges faster than full fine-tuning

=== Step 6: Save Adapter Weights ===
[[step::Principle:huggingface_peft_Adapter_Serialization]]

Save only the trained adapter weights using `model.save_pretrained()`. This creates a small checkpoint (typically a few MB) containing the LoRA weights and configuration, separate from the base model.

'''Output artifacts:'''
* `adapter_config.json`: LoRA configuration and base model reference
* `adapter_model.safetensors`: Trained LoRA weights (small file)
* Model card with training metadata

== Execution Diagram ==

{{#mermaid:graph TD
    A[Load Base Model] --> B[Configure LoRA]
    B --> C[Create PEFT Model]
    C --> D[Prepare Training]
    D --> E[Execute Training]
    E --> F[Save Adapter]
}}

== Related Pages ==

* [[step::Principle:huggingface_peft_Base_Model_Loading]]
* [[step::Principle:huggingface_peft_LoRA_Configuration]]
* [[step::Principle:huggingface_peft_PEFT_Model_Creation]]
* [[step::Principle:huggingface_peft_Training_Preparation]]
* [[step::Principle:huggingface_peft_Training_Execution]]
* [[step::Principle:huggingface_peft_Adapter_Serialization]]
