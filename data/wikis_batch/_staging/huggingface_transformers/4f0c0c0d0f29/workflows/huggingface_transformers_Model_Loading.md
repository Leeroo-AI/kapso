{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|HuggingFace Documentation|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Loading]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
End-to-end process for loading pretrained transformer models from the HuggingFace Hub or local checkpoints using the `from_pretrained` API.

=== Description ===
This workflow covers the complete model loading pipeline in the Transformers library. It handles:

1. **Configuration Resolution**: Determining the correct model architecture from config files
2. **Checkpoint Discovery**: Locating and validating weight files (safetensors, bin, or sharded)
3. **Quantization Handling**: Applying optional quantization configurations (BitsAndBytes, GPTQ, AWQ, etc.)
4. **Weight Loading**: Loading state dictionaries with proper device placement and dtype handling
5. **Post-Processing**: Tying weights, applying adapters, and finalizing the model

The workflow supports 100+ model architectures and 20+ quantization methods, with automatic handling of sharded checkpoints, device mapping, and memory optimization.

=== Usage ===
Execute this workflow when you need to:
* Load a pretrained model from the HuggingFace Hub for inference or fine-tuning
* Load a locally saved checkpoint
* Apply quantization to reduce memory footprint
* Use device mapping for multi-GPU inference
* Load custom models with `trust_remote_code=True`

== Execution Steps ==

=== Step 1: Configuration Loading ===
[[step::Principle:huggingface_transformers_Configuration_Loading]]

Load the model's configuration file to determine the architecture and hyperparameters. The configuration contains essential information like hidden size, number of layers, attention heads, and vocabulary size. AutoConfig automatically resolves the correct configuration class based on the model type.

'''Key considerations:'''
* Configs can be loaded from Hub, local files, or passed directly
* Custom configs may require `trust_remote_code=True`
* Config parameters can be overridden via kwargs

=== Step 2: Checkpoint Discovery ===
[[step::Principle:huggingface_transformers_Checkpoint_Discovery]]

Locate and validate the model's weight files. The system supports multiple formats: safetensors (preferred), PyTorch bin files, and sharded checkpoints with index files. Automatic conversion from legacy formats is handled transparently.

'''What happens:'''
* Search for safetensors first, then fall back to bin files
* Handle sharded checkpoints via index.json files
* Validate checkpoint integrity and compatibility
* Support both Hub downloads and local paths

=== Step 3: Quantization Configuration ===
[[step::Principle:huggingface_transformers_Quantization_Configuration]]

If quantization is requested, configure the appropriate quantizer. The library supports 20+ quantization methods including BitsAndBytes (INT4/INT8), GPTQ, AWQ, EETQ, FP8, and more. Each quantizer has specific requirements and optimizations.

'''Options available:'''
* BitsAndBytes: 4-bit and 8-bit quantization with NF4/FP4 options
* GPTQ/AWQ: Weight-only post-training quantization
* FP8: Hardware-accelerated on H100 GPUs
* TorchAO: PyTorch native quantization toolkit

=== Step 4: Model Instantiation ===
[[step::Principle:huggingface_transformers_Model_Instantiation]]

Create the model architecture with empty weights using the meta device. This avoids allocating memory for weights that will be immediately overwritten. The model class is resolved via AutoModel or specified explicitly.

'''Process:'''
* Use `init_empty_weights()` context to avoid memory allocation
* Resolve the correct model class (AutoModelForCausalLM, etc.)
* Initialize all layers with proper configuration
* Handle custom model code if needed

=== Step 5: Weight Loading ===
[[step::Principle:huggingface_transformers_Weight_Loading]]

Load the state dictionary into the model with proper device placement. This step handles dtype conversion, device mapping across multiple GPUs, and disk offloading for large models. Sharded checkpoints are loaded incrementally.

'''Technical details:'''
* Convert weights to target dtype (float16, bfloat16, etc.)
* Apply device_map for multi-GPU distribution
* Handle tied weights and parameter sharing
* Report missing/unexpected keys

=== Step 6: Post-Processing ===
[[step::Principle:huggingface_transformers_Model_Post_Processing]]

Finalize the model by tying shared weights, applying any PEFT adapters, and running quantization post-processing. The model is prepared for the intended use (inference or training).

'''Final steps:'''
* Tie embedding and output weights if configured
* Load and merge PEFT/LoRA adapters if specified
* Run quantizer post-processing hooks
* Set model to eval mode for inference

== Execution Diagram ==
{{#mermaid:graph TD
    A[Configuration Loading] --> B[Checkpoint Discovery]
    B --> C[Quantization Configuration]
    C --> D[Model Instantiation]
    D --> E[Weight Loading]
    E --> F[Post-Processing]
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Configuration_Loading]]
* [[step::Principle:huggingface_transformers_Checkpoint_Discovery]]
* [[step::Principle:huggingface_transformers_Quantization_Configuration]]
* [[step::Principle:huggingface_transformers_Model_Instantiation]]
* [[step::Principle:huggingface_transformers_Weight_Loading]]
* [[step::Principle:huggingface_transformers_Model_Post_Processing]]
