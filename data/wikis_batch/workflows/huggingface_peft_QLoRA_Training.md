# huggingface_peft_QLoRA_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
* [[source::Paper|QLoRA Paper|https://arxiv.org/abs/2305.14314]]
* [[source::Blog|QLoRA Fine-tuning|https://pytorch.org/blog/finetune-llms/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::Quantization]], [[domain::Memory_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Memory-efficient fine-tuning of large language models by combining 4-bit quantization with Low-Rank Adaptation (QLoRA).

=== Description ===

QLoRA enables fine-tuning of models that would otherwise require prohibitive GPU memory by quantizing the base model to 4-bit precision while training LoRA adapters in higher precision. This approach reduces memory requirements by 4-8x compared to standard fine-tuning, allowing training of 7B+ parameter models on a single consumer GPU.

**Key Innovations:**
* 4-bit NormalFloat (NF4) quantization for base model weights
* Double quantization to further reduce memory overhead
* Paged optimizers for handling memory spikes
* Full-precision LoRA adapters for stable training

=== Usage ===

Execute this workflow when you need to:
* Fine-tune large models (7B-70B parameters) on limited VRAM
* Train on consumer GPUs (16-24GB VRAM)
* Achieve near full-precision quality with dramatically reduced memory
* Deploy memory-efficient training pipelines

**Prerequisites:**
* bitsandbytes library installed for quantization
* CUDA-capable GPU with compute capability >= 7.5
* Sufficient CPU RAM for model loading before quantization

== Execution Steps ==

=== Step 1: Configure Quantization ===
[[step::Principle:huggingface_peft_Quantization_Config]]

Set up the BitsAndBytes configuration for 4-bit quantization. This defines how the base model weights will be stored in reduced precision while maintaining computational precision for forward passes.

'''Configuration parameters:'''
* `load_in_4bit`: Enable 4-bit quantization
* `bnb_4bit_quant_type`: Quantization type (nf4 recommended for normal distributions)
* `bnb_4bit_compute_dtype`: Dtype for computations (bfloat16 for stability)
* `bnb_4bit_use_double_quant`: Enable double quantization for extra memory savings

=== Step 2: Load Quantized Model ===
[[step::Principle:huggingface_peft_Model_Loading]]

Load the base model with the quantization configuration applied. The model weights are automatically quantized to 4-bit during loading, dramatically reducing GPU memory consumption.

'''Memory impact:'''
* 7B model: ~4GB VRAM (vs ~14GB in float16)
* 13B model: ~8GB VRAM (vs ~26GB in float16)
* 70B model: ~40GB VRAM (vs ~140GB in float16)

=== Step 3: Configure LoRA for QLoRA ===
[[step::Principle:huggingface_peft_LoRA_Configuration]]

Define the LoRA configuration specifically tuned for QLoRA training. The adapter will be trained in full precision (float32) while the base model remains in 4-bit.

'''QLoRA-specific considerations:'''
* Adapters trained in float32 for numerical stability
* Target all linear layers for best results
* Consider higher rank (r=64) since base is quantized
* Enable gradient checkpointing for additional memory savings

=== Step 4: Apply PEFT with QLoRA ===
[[step::Principle:huggingface_peft_PEFT_Application]]

Inject LoRA adapters into the quantized model. The adapters are created in full precision and attach to the 4-bit quantized base layers via bitsandbytes integration.

'''Technical details:'''
* Adapters wrap quantized Linear4bit layers
* Forward pass: dequantize → compute → add adapter contribution
* Only adapter parameters are trainable
* Memory-efficient gradient computation through quantized base

=== Step 5: Enable Memory Optimizations ===
[[step::Principle:huggingface_peft_Memory_Optimization]]

Configure additional memory optimizations to maximize training batch size and stability on limited hardware.

'''Optimizations:'''
* Gradient checkpointing: Trade compute for memory
* Paged optimizers: Handle memory spikes gracefully
* Mixed precision training: FP16/BF16 for non-adapter computations
* Gradient accumulation: Simulate larger batch sizes

=== Step 6: Train QLoRA Adapter ===
[[step::Principle:huggingface_peft_Adapter_Training]]

Train the model with the QLoRA setup. Training proceeds with the base model frozen in 4-bit and adapters updated in full precision.

'''Training characteristics:'''
* Slower per-step than full-precision (dequantization overhead)
* Much larger effective batch sizes possible
* Quality approaches full-precision fine-tuning
* Stable training despite quantization

=== Step 7: Save QLoRA Adapter ===
[[step::Principle:huggingface_peft_Adapter_Saving]]

Save the trained adapter weights. Note that only the adapter (in full precision) is saved; the quantization config is preserved in the adapter config for loading.

'''Deployment options:'''
* Load adapter onto fresh quantized base model
* Merge adapter into full-precision base for serving
* Convert to other quantization formats (GGUF, GPTQ)

== Execution Diagram ==

{{#mermaid:graph TD
    A[Configure Quantization] --> B[Load Quantized Model]
    B --> C[Configure LoRA]
    C --> D[Apply PEFT]
    D --> E[Enable Memory Opts]
    E --> F[Train Adapter]
    F --> G[Save Adapter]
    G --> H{Deploy}
    H --> I[Serve Quantized]
    H --> J[Merge Full Precision]
}}

== Related Pages ==

* [[step::Principle:huggingface_peft_Quantization_Config]]
* [[step::Principle:huggingface_peft_Model_Loading]]
* [[step::Principle:huggingface_peft_LoRA_Configuration]]
* [[step::Principle:huggingface_peft_PEFT_Application]]
* [[step::Principle:huggingface_peft_Memory_Optimization]]
* [[step::Principle:huggingface_peft_Adapter_Training]]
* [[step::Principle:huggingface_peft_Adapter_Saving]]
