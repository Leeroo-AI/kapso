{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|Fine-tuning Guide|https://docs.unsloth.ai/get-started/fine-tuning-guide]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::QLoRA]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 14:00 GMT]]
|}

== Overview ==
End-to-end process for parameter-efficient fine-tuning of Large Language Models using 4-bit QLoRA quantization with Unsloth's optimized kernels for 2-5x speedup and 70% VRAM reduction.

=== Description ===
This workflow outlines the standard procedure for fine-tuning LLMs on consumer hardware using Unsloth's optimization library. It leverages 4-bit NormalFloat (NF4) quantization via bitsandbytes combined with Low-Rank Adapters (LoRA) to minimize memory requirements, enabling training of 7B+ parameter models on single GPUs with limited VRAM (8-24GB).

The process covers:
1. Loading a base model with automatic 4-bit quantization
2. Applying LoRA adapters to trainable projection layers
3. Preparing instruction-format datasets with chat templates
4. Running supervised fine-tuning (SFT) with optimized Triton kernels
5. Saving trained adapters or merging to full-precision weights

Key optimizations include custom cross-entropy loss kernels, fused RMS LayerNorm, and intelligent gradient checkpointing that together deliver significant speedups over standard HuggingFace implementations.

=== Usage ===
Execute this workflow when:
* You have a domain-specific dataset (instruction-tuning, conversational, or completion style)
* You need to adapt a base model (Llama, Qwen, Mistral, Gemma, etc.) to follow specific instructions or exhibit domain expertise
* You have limited GPU resources (8-80GB VRAM) and need memory-efficient training
* You want faster training iteration cycles compared to standard fine-tuning

'''Input requirements:'''
* HuggingFace model name or local model path
* Training dataset in instruction/completion format
* CUDA-capable GPU with compute capability 7.0+

'''Output:'''
* Trained LoRA adapter weights (compact, ~50-200MB)
* Optionally: merged 16-bit model weights for deployment

== Execution Steps ==

=== Step 1: Environment Setup ===
[[step::Principle:unslothai_unsloth_Environment_Initialization]]

Initialize the Unsloth environment by importing the library before other ML frameworks. This ensures all optimization patches are applied to transformers, TRL, and PEFT libraries. The import order is critical - importing Unsloth after transformers/TRL will result in unoptimized execution paths.

'''Key considerations:'''
* Import `unsloth` before `transformers`, `trl`, or `peft`
* CUDA toolkit and bitsandbytes must be properly linked
* Triton kernels are automatically compiled on first use

=== Step 2: Model Loading with Quantization ===
[[step::Principle:unslothai_unsloth_Model_Loading]]

Load the base model using `FastLanguageModel.from_pretrained()` with 4-bit quantization enabled. The loader automatically selects the appropriate model architecture (Llama, Qwen, Mistral, etc.), applies NF4 quantization via bitsandbytes, and patches attention layers with optimized implementations.

'''What happens:'''
* Model architecture is auto-detected from config
* Weights are loaded in 4-bit NormalFloat format
* Tokenizer is loaded and optionally fixed for special token handling
* RoPE scaling is configured for the specified max sequence length
* Memory-efficient attention backend is selected (SDPA, FlashAttention, or xFormers)

'''Key parameters:'''
* `model_name`: HuggingFace model ID or local path
* `max_seq_length`: Maximum context length (RoPE scaling applied automatically)
* `load_in_4bit`: Enable 4-bit QLoRA quantization
* `dtype`: Compute dtype (auto-detected, typically bfloat16)

=== Step 3: LoRA Adapter Injection ===
[[step::Principle:unslothai_unsloth_LoRA_Configuration]]

Inject Low-Rank Adapter matrices into the frozen base model using `FastLanguageModel.get_peft_model()`. Only the small adapter matrices (A and B) are trained, dramatically reducing memory requirements and training time while preserving the base model's capabilities.

'''What happens:'''
* LoRA adapters are injected into specified projection layers
* Original weight matrix W remains frozen
* Two small matrices A and B are added: W' = W + BA
* Only A and B are updated during training (typically <1% of total parameters)
* Gradient checkpointing is optionally enabled for memory savings

'''Key parameters:'''
* `r`: LoRA rank (8, 16, 32, 64 common choices - higher = more capacity)
* `target_modules`: Layers to apply LoRA (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
* `lora_alpha`: Scaling factor (typically equal to r)
* `use_gradient_checkpointing`: "unsloth" for optimized checkpointing

=== Step 4: Dataset Preparation ===
[[step::Principle:unslothai_unsloth_Data_Formatting]]

Transform raw training data into the structured prompt format expected by the model. This involves mapping input fields to a consistent template and applying the model's chat template for proper tokenization with special token boundaries.

'''What happens:'''
* Raw data is formatted into instruction/input/output structure
* Chat template is applied (Llama, ChatML, Alpaca, etc.)
* Special tokens (BOS, EOS, header tokens) are properly inserted
* Dataset is tokenized and prepared for training

'''Key considerations:'''
* Use `get_chat_template()` to apply model-specific formatting
* `train_on_responses_only()` can mask instruction tokens from loss computation
* Sample packing can be enabled to maximize GPU utilization

=== Step 5: Trainer Configuration ===
[[step::Principle:unslothai_unsloth_Training_Configuration]]

Configure the SFTTrainer (from TRL library) with optimized training arguments. Unsloth patches TRL trainers to use optimized kernels for cross-entropy loss, gradient computation, and mixed-precision training.

'''Key parameters:'''
* `per_device_train_batch_size`: Batch size per GPU
* `gradient_accumulation_steps`: Effective batch size multiplier
* `learning_rate`: Typically 2e-4 to 5e-5 for fine-tuning
* `max_steps` or `num_train_epochs`: Training duration
* `optim`: "adamw_8bit" for memory-efficient optimizer
* `bf16`/`fp16`: Mixed precision training

=== Step 6: Training Execution ===
[[step::Principle:unslothai_unsloth_SFT_Training]]

Execute the supervised fine-tuning loop with Unsloth's optimized kernels. During training, custom Triton kernels handle cross-entropy loss computation, RoPE embeddings, and layer normalization with significant speedups over native PyTorch implementations.

'''What happens:'''
* Forward pass uses fused RoPE and attention kernels
* Cross-entropy loss computed with memory-efficient chunked softmax
* Backward pass leverages custom gradient kernels
* Only LoRA adapter weights are updated
* Checkpoints saved based on configuration

=== Step 7: Model Saving ===
[[step::Principle:unslothai_unsloth_Model_Saving]]

Save the trained model using one of several methods: LoRA adapters only, merged 16-bit weights, or quantized GGUF format. The `save_pretrained_merged()` method handles dequantization and LoRA weight merging automatically.

'''Save methods:'''
* `save_pretrained()`: Save LoRA adapter weights only (compact)
* `save_pretrained_merged(..., save_method="merged_16bit")`: Merge and save full model
* `save_pretrained_gguf()`: Convert to GGUF format for llama.cpp/Ollama

== Execution Diagram ==
{{#mermaid:graph TD
    A[Environment Setup] --> B[Model Loading]
    B --> C[LoRA Injection]
    C --> D[Dataset Preparation]
    D --> E[Trainer Configuration]
    E --> F[Training Execution]
    F --> G[Model Saving]
}}

== Related Pages ==
* [[step::Principle:unslothai_unsloth_Environment_Initialization]]
* [[step::Principle:unslothai_unsloth_Model_Loading]]
* [[step::Principle:unslothai_unsloth_LoRA_Configuration]]
* [[step::Principle:unslothai_unsloth_Data_Formatting]]
* [[step::Principle:unslothai_unsloth_Training_Configuration]]
* [[step::Principle:unslothai_unsloth_SFT_Training]]
* [[step::Principle:unslothai_unsloth_Model_Saving]]
