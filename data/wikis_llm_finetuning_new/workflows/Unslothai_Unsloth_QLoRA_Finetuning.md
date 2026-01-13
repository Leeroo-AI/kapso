# Workflow: QLoRA_Finetuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Blog|QLoRA Paper|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::Parameter_Efficient_Training]]
|-
! Last Updated
| [[last_updated::2026-01-12 19:00 GMT]]
|}

== Overview ==

End-to-end process for parameter-efficient fine-tuning of Large Language Models using 4-bit quantization (QLoRA) with Unsloth's optimized training pipeline.

=== Description ===

This workflow outlines the standard procedure for fine-tuning Large Language Models on consumer hardware using Quantized Low-Rank Adaptation (QLoRA). Unsloth achieves 2x faster training and 70% memory reduction compared to standard HuggingFace implementations through custom Triton kernels and optimized memory management.

The process covers:
* Model loading with 4-bit NormalFloat quantization
* LoRA adapter injection into attention and MLP layers
* Data formatting with chat templates
* Supervised fine-tuning using TRL's SFTTrainer
* Model merging and export in multiple formats

=== Usage ===

Execute this workflow when you have an instruction-tuning dataset and need to adapt a base language model (Llama, Mistral, Qwen, Gemma, etc.) for a specific task, but have limited GPU resources (e.g., <24GB VRAM). This is the recommended approach for most fine-tuning scenarios.

== Execution Steps ==

=== Step 1: Model_Loading ===

Initialize the language model with memory-optimized settings. The FastLanguageModel loader automatically detects the model architecture, applies 4-bit quantization using bitsandbytes, and patches attention/MLP layers with optimized Triton kernels.

'''Key considerations:'''
* Set `load_in_4bit=True` for QLoRA (reduces memory by ~75%)
* Configure `max_seq_length` based on your training data requirements
* The loader auto-detects dtype (bfloat16 preferred on Ampere+ GPUs)
* Model architecture is automatically identified from HuggingFace config

=== Step 2: LoRA_Adapter_Injection ===

Inject Low-Rank Adapter matrices into the frozen base model. The `get_peft_model` function wraps specific layers with trainable LoRA parameters while keeping the quantized base weights frozen.

'''What happens:'''
* Target modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) receive LoRA adapters
* Rank `r` determines adapter capacity (typical values: 8, 16, 32, 64)
* `lora_alpha` controls scaling of adapter contributions
* Only adapter parameters (~0.1-1% of total) are marked as trainable

'''Key considerations:'''
* Higher rank = more capacity but more memory and slower training
* Use `use_gradient_checkpointing="unsloth"` for long sequences
* Target "all-linear" to apply LoRA to all linear layers

=== Step 3: Data_Formatting ===

Transform raw training data into the structured prompt format expected by the model. Apply the model's chat template for proper tokenization and special token handling.

'''What happens:'''
* Raw data is converted to message format (system/user/assistant roles)
* Chat template converts messages to model-specific token sequence
* Special tokens mark turn boundaries for proper attention masking
* Optional: Use `train_on_responses_only` to mask instruction tokens from loss

'''Key considerations:'''
* Match chat template to base model (llama-3.1, chatml, mistral, etc.)
* Ensure consistent schema across all training examples
* Consider using dataset packing for efficiency with short sequences

=== Step 4: Training_Configuration ===

Configure the training loop with appropriate hyperparameters. Set up the SFTTrainer with optimized settings for QLoRA training.

'''Key parameters:'''
* Batch size and gradient accumulation (effective batch size = per_device * accumulation)
* Learning rate (typically 1e-4 to 2e-4 for LoRA)
* Number of epochs or max steps
* Optimizer (adamw_8bit recommended for memory efficiency)
* Mixed precision (bf16 on supported hardware)

=== Step 5: SFT_Training ===

Execute the supervised fine-tuning loop. The trainer handles forward/backward passes, gradient accumulation, and checkpointing while Unsloth's optimizations provide speed and memory improvements.

'''What happens:'''
* Model alternates between inference mode (generation) and training mode
* Gradients flow only through LoRA adapter parameters
* Loss is computed on target tokens (assistant responses)
* Optimized Triton kernels accelerate attention and cross-entropy computation

=== Step 6: Model_Saving ===

Save the trained model in the desired format. Options include saving just the LoRA adapter (small files, requires base model), or merging adapters into the base model for standalone deployment.

'''Save methods:'''
* `lora` - Save only adapter weights (~10-100MB)
* `merged_16bit` - Merge and save full model in 16-bit precision
* GGUF export for llama.cpp/Ollama deployment (see GGUF_Export workflow)

'''Key considerations:'''
* LoRA saves are fast and small, ideal for iteration
* Merged models are standalone but larger
* Consider pushing to HuggingFace Hub for sharing

== Execution Diagram ==

{{#mermaid:graph TD
    A[Model_Loading] --> B[LoRA_Adapter_Injection]
    B --> C[Data_Formatting]
    C --> D[Training_Configuration]
    D --> E[SFT_Training]
    E --> F[Model_Saving]
}}

== GitHub URL ==

The executable implementation will be available at:

[[github_url::PENDING_REPO_BUILD]]

<!-- This URL will be populated by the repo builder phase -->
