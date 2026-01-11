# Workflow: QLoRA_Finetuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://unsloth.ai/docs]]
* [[source::Doc|Fine-tuning Guide|https://unsloth.ai/docs/get-started/fine-tuning-llms-guide]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::QLoRA]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==
End-to-end process for parameter-efficient fine-tuning of Large Language Models using QLoRA (Quantized Low-Rank Adaptation) with Unsloth's optimized training pipeline.

=== Description ===
This workflow outlines the standard procedure for fine-tuning Large Language Models (LLMs) on consumer hardware using Unsloth's optimized QLoRA implementation. The process leverages 4-bit NormalFloat quantization to reduce memory requirements by up to 70%, allowing training of 7B+ parameter models on single GPUs with less than 24GB VRAM.

The workflow covers:
1. **Model Loading**: Loading pre-quantized base models with Unsloth's memory-optimized loader
2. **LoRA Configuration**: Injecting low-rank adapter matrices into attention and MLP layers
3. **Data Preparation**: Formatting datasets with appropriate chat templates for instruction tuning
4. **Training**: Supervised fine-tuning using TRL's SFTTrainer with Unsloth optimizations
5. **Model Export**: Saving merged weights or LoRA adapters for deployment

Unsloth achieves 2x faster training through custom Triton kernels for cross-entropy loss, RMS normalization, and fused LoRA operations.

=== Usage ===
Execute this workflow when:
* You have a domain-specific dataset (instruction-tuning, conversational, or task-specific format)
* You need to adapt a base LLM to follow specific instructions or exhibit specialized behavior
* You have limited GPU resources (8GB-24GB VRAM) and cannot afford full fine-tuning
* You want to preserve the base model's general capabilities while adding specialized knowledge

'''Input requirements:'''
* Pre-trained base model (LLaMA, Mistral, Qwen, Gemma, etc.)
* Training dataset in conversational or instruction format
* CUDA-capable GPU with 8GB+ VRAM

'''Expected outputs:'''
* Trained LoRA adapter weights (can be merged with base model)
* Optionally: Merged 16-bit model for deployment
* Optionally: GGUF file for llama.cpp/Ollama deployment

== Execution Steps ==

=== Step 1: Model Loading ===
[[step::Principle:Unslothai_Unsloth_Model_Loading]]

Initialize the language model with memory-optimized settings using `FastLanguageModel.from_pretrained()`. The loader automatically applies 4-bit NormalFloat quantization with double quantization for additional memory savings. The attention layers are patched for efficient training on consumer GPUs.

'''Key considerations:'''
* Set `load_in_4bit=True` for QLoRA training (default)
* Choose `max_seq_length` based on your training data and available VRAM
* Use pre-quantized models from Unsloth Hub for faster loading

=== Step 2: LoRA Configuration ===
[[step::Principle:Unslothai_Unsloth_LoRA_Configuration]]

Configure and inject Low-Rank Adapter matrices into the model using `FastLanguageModel.get_peft_model()`. This sets up trainable adapter weights while keeping the base model frozen, dramatically reducing memory requirements and training time.

'''Key considerations:'''
* Choose rank (r) based on task complexity: 8-16 for simple tasks, 32-128 for complex reasoning
* Target all linear layers for best results: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
* Enable `use_gradient_checkpointing="unsloth"` for 30% VRAM reduction
* Set `lora_alpha` equal to rank for stable training

=== Step 3: Data Formatting ===
[[step::Principle:Unslothai_Unsloth_Data_Formatting]]

Transform raw training data into the structured prompt format expected by the model. Apply the appropriate chat template to ensure proper tokenization boundaries and special token insertion.

'''Key considerations:'''
* Use `get_chat_template()` to apply model-specific formatting (llama-3, chatml, etc.)
* Ensure all examples follow a consistent schema (system/user/assistant roles)
* For instruction tuning, use `train_on_responses_only()` to mask non-response tokens
* Validate that special tokens (BOS, EOS, header tokens) are properly inserted

=== Step 4: Training Configuration ===
[[step::Principle:Unslothai_Unsloth_Training_Configuration]]

Configure the training loop using TRL's SFTTrainer with Unsloth-optimized settings. Set hyperparameters for batch size, learning rate, and gradient accumulation based on available hardware.

'''Key considerations:'''
* Use `adamw_8bit` optimizer for memory efficiency
* Set gradient accumulation to simulate larger batch sizes on limited VRAM
* Enable `bf16` training on Ampere+ GPUs for best performance
* Configure warmup steps and learning rate schedule

=== Step 5: Supervised Fine-Tuning ===
[[step::Principle:Unslothai_Unsloth_Supervised_Finetuning]]

Execute the training loop using SFTTrainer. The trainer handles batching, gradient computation, and optimization while Unsloth's Triton kernels accelerate the forward and backward passes.

'''Key considerations:'''
* Monitor loss curves for convergence
* Use `logging_steps` to track training progress
* Enable checkpointing for long training runs
* Consider sample packing for datasets with short sequences

=== Step 6: Model Saving ===
[[step::Principle:Unslothai_Unsloth_Model_Saving]]

Save the trained model using one of several export methods: LoRA adapter only, merged 16-bit weights, or GGUF format for deployment with llama.cpp/Ollama.

'''Key considerations:'''
* Use `save_pretrained()` for LoRA adapter only (smallest, requires base model for inference)
* Use `save_pretrained_merged(save_method="merged_16bit")` for standalone deployment
* Use `save_pretrained_gguf()` for llama.cpp/Ollama deployment
* Push to Hugging Face Hub with `push_to_hub_merged()` for sharing

== Execution Diagram ==
{{#mermaid:graph TD
    A[Model Loading] --> B[LoRA Configuration]
    B --> C[Data Formatting]
    C --> D[Training Configuration]
    D --> E[Supervised Fine-Tuning]
    E --> F[Model Saving]
    F --> G{Export Format}
    G -->|LoRA Only| H[LoRA Adapter]
    G -->|Merged| I[16-bit Model]
    G -->|GGUF| J[llama.cpp Format]
}}

== Related Pages ==
* [[step::Principle:Unslothai_Unsloth_Model_Loading]]
* [[step::Principle:Unslothai_Unsloth_LoRA_Configuration]]
* [[step::Principle:Unslothai_Unsloth_Data_Formatting]]
* [[step::Principle:Unslothai_Unsloth_Training_Configuration]]
* [[step::Principle:Unslothai_Unsloth_Supervised_Finetuning]]
* [[step::Principle:Unslothai_Unsloth_Model_Saving]]
