{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Guide|https://huggingface.co/docs/transformers/quantization]]
* [[source::Doc|BitsAndBytes|https://huggingface.co/docs/bitsandbytes]]
|-
! Domains
| [[domain::Quantization]], [[domain::Memory_Optimization]], [[domain::Inference]], [[domain::Model_Compression]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Workflow for loading and running models with reduced precision quantization to minimize memory usage while maintaining model quality.

=== Description ===
The quantization system in HuggingFace Transformers supports 20+ quantization methods for reducing model memory footprint. Key supported methods include:

* **bitsandbytes 4-bit**: NormalFloat4 (NF4) quantization for QLoRA training
* **bitsandbytes 8-bit**: INT8 quantization with dynamic outlier handling
* **GPTQ**: Post-training quantization with calibration data
* **AWQ**: Activation-aware weight quantization
* **FP8**: 8-bit floating point for H100+ GPUs
* **GGUF**: Cross-platform quantized format from llama.cpp

The quantizer system uses a registry pattern with automatic dispatch based on model configuration or user-specified quantization config.

=== Usage ===
Execute this workflow when you need to:
* Load large models on consumer GPUs with limited VRAM
* Reduce memory usage for inference
* Prepare models for QLoRA/PEFT training
* Deploy models efficiently in production
* Convert between quantization formats

Prerequisites:
* Model checkpoint (Hub or local)
* Quantization library installed (bitsandbytes, auto-gptq, etc.)
* GPU with sufficient memory for quantized model

== Execution Steps ==

=== Step 1: Quantization Configuration ===
[[step::Principle:huggingface_transformers_Quantization_Config]]

Create the quantization configuration specifying the method and parameters. Each quantization method has its own config class (BitsAndBytesConfig, GPTQConfig, AwqConfig, etc.).

'''Configuration options:'''
* load_in_4bit / load_in_8bit for bitsandbytes
* Compute dtype (bf16/fp16) for dequantization
* Quantization type (nf4, fp4)
* Double quantization for additional compression
* Module targeting for selective quantization

=== Step 2: Quantizer Selection ===
[[step::Principle:huggingface_transformers_Quantizer_Selection]]

Select the appropriate quantizer implementation based on the configuration. The auto quantizer dispatches to method-specific quantizer classes.

'''Supported quantizers:'''
* Bnb4BitHfQuantizer / Bnb8BitHfQuantizer
* GptqHfQuantizer / AwqHfQuantizer
* FbgemmFp8HfQuantizer / TorchAoHfQuantizer
* HqqHfQuantizer / QuantoHfQuantizer

=== Step 3: Pre-Loading Validation ===
[[step::Principle:huggingface_transformers_Quantization_Validation]]

Validate that the environment supports the requested quantization method. Check for required libraries, GPU capabilities, and compatible configurations.

'''Validation checks:'''
* Required library availability
* CUDA/GPU capability requirements
* Configuration compatibility
* Mutually exclusive options

=== Step 4: Weight Quantization ===
[[step::Principle:huggingface_transformers_Weight_Quantization]]

Apply quantization to model weights during or after loading. Different methods have different application points (during loading, post-loading calibration).

'''Quantization timing:'''
* On-the-fly: bitsandbytes quantizes during loading
* Post-training: GPTQ requires calibration dataset
* Pre-quantized: Load already quantized checkpoints

=== Step 5: Linear Layer Replacement ===
[[step::Principle:huggingface_transformers_Linear_Layer_Replacement]]

Replace standard nn.Linear modules with quantized equivalents. Each quantization method has its own Linear implementation that handles quantized weights and dequantization.

'''Quantized modules:'''
* Linear4bit / Linear8bitLt for bitsandbytes
* QuantLinear for GPTQ
* WQLinear for AWQ
* FP8Linear for FP8 quantization

=== Step 6: Module Targeting ===
[[step::Principle:huggingface_transformers_Module_Targeting]]

Select which modules to quantize based on configuration. Common patterns include quantizing attention and MLP layers while keeping embeddings and normalization in full precision.

'''Targeting strategies:'''
* All linear layers (default)
* Exclude specific modules (lm_head, embed_tokens)
* Include specific patterns (q_proj, k_proj, etc.)
* Skip certain layer indices

=== Step 7: Post-Quantization Setup ===
[[step::Principle:huggingface_transformers_Post_Quantization_Setup]]

Finalize the quantized model configuration. Set up compute hooks, register forward hooks for dynamic dequantization, and prepare for training if applicable.

'''Post-quantization operations:'''
* Register dequantization hooks
* Set trainable parameters for QLoRA
* Configure gradient checkpointing compatibility
* Update model config with quantization info

== Execution Diagram ==
{{#mermaid:graph TD
    A[Quantization Configuration] --> B[Quantizer Selection]
    B --> C[Pre-Loading Validation]
    C --> D[Weight Quantization]
    D --> E[Linear Layer Replacement]
    E --> F[Module Targeting]
    F --> G[Post-Quantization Setup]
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Quantization_Config]]
* [[step::Principle:huggingface_transformers_Quantizer_Selection]]
* [[step::Principle:huggingface_transformers_Quantization_Validation]]
* [[step::Principle:huggingface_transformers_Weight_Quantization]]
* [[step::Principle:huggingface_transformers_Linear_Layer_Replacement]]
* [[step::Principle:huggingface_transformers_Module_Targeting]]
* [[step::Principle:huggingface_transformers_Post_Quantization_Setup]]
