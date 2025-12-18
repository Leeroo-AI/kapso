{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Model Loading Guide|https://huggingface.co/docs/transformers/main_classes/model]]
* [[source::Doc|Quantization|https://huggingface.co/docs/transformers/quantization]]
|-
! Domains
| [[domain::Model_Loading]], [[domain::Quantization]], [[domain::Weight_Management]], [[domain::Hub_Integration]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
End-to-end workflow for loading pretrained models from the HuggingFace Hub or local checkpoints, with support for quantization, device mapping, and weight transformations.

=== Description ===
The model loading system in HuggingFace Transformers handles the complexity of instantiating models from various checkpoint formats. It supports:

* **Hub Integration**: Direct loading from HuggingFace Hub with automatic caching
* **Quantization**: Load models in reduced precision (4-bit, 8-bit) using bitsandbytes, GPTQ, AWQ, and other methods
* **Device Mapping**: Automatic or manual distribution across multiple GPUs and CPU offloading
* **Weight Conversion**: Transform checkpoint weights between different formats (e.g., original → HF format)
* **Sharded Checkpoints**: Load large models split across multiple files

The PreTrainedModel.from_pretrained() method is the primary entry point, handling config loading, weight retrieval, and model instantiation in a unified interface.

=== Usage ===
Execute this workflow when you need to:
* Load a model from the HuggingFace Hub for inference or training
* Load a locally saved checkpoint
* Apply quantization to reduce memory usage
* Distribute a large model across multiple devices
* Convert checkpoints from other frameworks

Prerequisites:
* Model identifier (Hub ID or local path)
* Appropriate compute resources for the model size
* Optional: quantization configuration

== Execution Steps ==

=== Step 1: Configuration Resolution ===
[[step::Principle:huggingface_transformers_Configuration_Resolution]]

Load and validate the model configuration from the checkpoint. The configuration determines the model architecture, hidden sizes, number of layers, and other structural parameters.

'''Configuration sources:'''
* config.json file in checkpoint directory
* AutoConfig for automatic model class detection
* User-provided config overrides
* Remote config from HuggingFace Hub

=== Step 2: Checkpoint File Discovery ===
[[step::Principle:huggingface_transformers_Checkpoint_Discovery]]

Locate and validate checkpoint files to load. Support for single-file checkpoints (pytorch_model.bin, model.safetensors) and sharded checkpoints with index files.

'''Supported formats:'''
* SafeTensors (.safetensors) - preferred for security and speed
* PyTorch (.bin) - legacy format
* Sharded checkpoints with index JSON
* GGUF quantized formats

=== Step 3: Quantization Configuration ===
[[step::Principle:huggingface_transformers_Quantization_Configuration]]

If quantization is requested, configure the appropriate quantizer. The quantizer system supports 20+ methods including bitsandbytes (4-bit NF4, 8-bit INT8), GPTQ, AWQ, and FP8 formats.

'''Quantization dispatch:'''
* Auto-detect quantization from model config
* Apply user-specified BitsAndBytesConfig
* Select quantizer based on method (bnb, gptq, awq, etc.)
* Configure compute dtype and quantization parameters

=== Step 4: Model Instantiation ===
[[step::Principle:huggingface_transformers_Model_Instantiation]]

Create the model instance with the resolved configuration. Handle special initialization modes like meta-device initialization for large models, DeepSpeed Zero-3 initialization, and tensor parallelism.

'''Instantiation modes:'''
* Standard: Allocate weights on target device
* Empty weights: Use accelerate's init_empty_weights for large models
* DeepSpeed Zero-3: Partition parameters across GPUs
* Tensor Parallel: Distribute with device mesh

=== Step 5: State Dict Loading ===
[[step::Principle:huggingface_transformers_State_Dict_Loading]]

Load checkpoint weights into the model. Handle weight name conversions, dtype casting, and sharded loading for large models.

'''Loading features:'''
* Automatic weight name remapping
* Dtype conversion (fp32 → bf16, etc.)
* Low CPU memory mode for large models
* Strict vs. non-strict loading

=== Step 6: Device Placement ===
[[step::Principle:huggingface_transformers_Device_Placement]]

Move the model to the appropriate device(s). Support automatic device mapping with accelerate for multi-GPU and CPU offloading scenarios.

'''Placement strategies:'''
* Single device: Move entire model
* Device map: Distribute layers across GPUs
* Disk offload: Offload to NVMe for very large models
* Sequential: Load layers on-demand

=== Step 7: Post-Loading Hooks ===
[[step::Principle:huggingface_transformers_Post_Loading_Hooks]]

Execute post-loading operations like tying weights, initializing generation config, and applying adapter weights (PEFT/LoRA).

'''Post-loading operations:'''
* Tie input/output embeddings
* Initialize GenerationConfig for generative models
* Load and merge PEFT adapters
* Apply quantization hooks

== Execution Diagram ==
{{#mermaid:graph TD
    A[Configuration Resolution] --> B[Checkpoint File Discovery]
    B --> C[Quantization Configuration]
    C --> D[Model Instantiation]
    D --> E[State Dict Loading]
    E --> F[Device Placement]
    F --> G[Post-Loading Hooks]
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Configuration_Resolution]]
* [[step::Principle:huggingface_transformers_Checkpoint_Discovery]]
* [[step::Principle:huggingface_transformers_Quantization_Configuration]]
* [[step::Principle:huggingface_transformers_Model_Instantiation]]
* [[step::Principle:huggingface_transformers_State_Dict_Loading]]
* [[step::Principle:huggingface_transformers_Device_Placement]]
* [[step::Principle:huggingface_transformers_Post_Loading_Hooks]]
