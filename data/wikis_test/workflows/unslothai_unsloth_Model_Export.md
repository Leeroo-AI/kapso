{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|Saving Guide|https://docs.unsloth.ai/basics/running-and-saving-models]]
* [[source::Doc|GGUF Guide|https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Export]], [[domain::GGUF]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-17 14:00 GMT]]
|}

== Overview ==
End-to-end process for exporting trained Unsloth models to various deployment formats including HuggingFace Hub, GGUF quantization for llama.cpp/Ollama, and vLLM-compatible checkpoints.

=== Description ===
This workflow covers the complete model export pipeline from trained LoRA adapters to deployment-ready formats. Unsloth supports multiple export paths:

1. **LoRA Adapters Only**: Save compact adapter weights (~50-200MB) for later loading
2. **Merged 16-bit**: Dequantize 4-bit weights and merge LoRA adapters into full-precision model
3. **GGUF Quantization**: Convert to llama.cpp format with various quantization levels (Q4_K_M, Q8_0, etc.)
4. **Ollama Modelfile**: Generate complete Ollama-ready package with chat templates
5. **HuggingFace Hub**: Push to Hub as public or private repository

The workflow handles the complex process of:
* Dequantizing 4-bit NF4 weights back to float16/bfloat16
* Merging LoRA adapter matrices (W' = W + s*BA) into base weights
* Converting to GGUF format using llama.cpp tooling
* Applying appropriate quantization schemes
* Generating Ollama Modelfiles with correct chat templates

=== Usage ===
Execute this workflow when:
* You have completed training and need to deploy the model
* You want to share your model on HuggingFace Hub
* You need to run the model locally with llama.cpp or Ollama
* You want to convert between quantization formats

'''Input requirements:'''
* Trained Unsloth model (with LoRA adapters)
* Tokenizer from training
* Target format specification

'''Output formats:'''
* HuggingFace format (safetensors)
* GGUF format (various quantization levels)
* Ollama Modelfile package
* vLLM-compatible checkpoint

== Execution Steps ==

=== Step 1: Training Completion Verification ===
[[step::Principle:unslothai_unsloth_Training_Verification]]

Verify that training has completed successfully and the model produces expected outputs. Sample generation should confirm the model has learned the desired behavior before investing in export.

'''Verification steps:'''
* Run inference on test prompts to verify quality
* Check LoRA weights have been updated (not all zeros)
* Confirm tokenizer handles special tokens correctly

=== Step 2: Save Method Selection ===
[[step::Principle:unslothai_unsloth_Export_Format_Selection]]

Select the appropriate export format based on deployment target. Each format has trade-offs in size, speed, and compatibility:

'''Format options:'''
| Format | Size | Inference Speed | Compatibility |
|--------|------|-----------------|---------------|
| LoRA only | Small (~100MB) | Requires base model | HuggingFace/Unsloth |
| Merged 16-bit | Large (14-140GB) | Native speed | All frameworks |
| GGUF Q4_K_M | Medium (4-40GB) | Fast (CPU/GPU) | llama.cpp, Ollama |
| GGUF Q8_0 | Larger (8-80GB) | Faster | llama.cpp, Ollama |

=== Step 3: LoRA Adapter Export ===
[[step::Principle:unslothai_unsloth_LoRA_Export]]

Save LoRA adapter weights only. This is the most compact option, storing only the trained adapter matrices without the base model weights.

'''What happens:'''
* LoRA A and B matrices saved to safetensors
* adapter_config.json specifies configuration
* Base model reference preserved for reloading

'''Use cases:'''
* Sharing adapters that work with public base models
* Storing checkpoints during training
* Multi-adapter deployment scenarios

=== Step 4: Merged Model Export ===
[[step::Principle:unslothai_unsloth_Merged_Export]]

Dequantize the 4-bit base model and merge LoRA adapters into the weights. This produces a standard HuggingFace model that can be used with any compatible framework.

'''What happens:'''
* 4-bit NF4 weights dequantized to float16/bfloat16
* LoRA weights merged: W_final = W_base + scale * (A @ B)
* Model saved in safetensors format
* Config and tokenizer files included

'''Key parameters:'''
* `save_method = "merged_16bit"`: Full precision merge
* `save_method = "merged_4bit"`: Re-quantize after merge
* `save_method = "lora"`: Adapter-only (no merge)

=== Step 5: GGUF Conversion ===
[[step::Principle:unslothai_unsloth_GGUF_Conversion]]

Convert the merged model to GGUF format for use with llama.cpp ecosystem tools. GGUF is the standard format for local LLM inference with CPU and GPU support.

'''What happens:'''
* llama.cpp tools installed/verified
* Model converted to GGUF with `convert_hf_to_gguf.py`
* Optional quantization applied (Q4_K_M, Q8_0, etc.)
* Tokenizer vocabulary embedded in GGUF file

'''Quantization options:'''
* `q4_k_m`: Recommended balance of size/quality
* `q5_k_m`: Higher quality, larger size
* `q8_0`: Near-lossless, largest quantized size
* `f16`: Full precision, largest size

=== Step 6: Ollama Package Creation ===
[[step::Principle:unslothai_unsloth_Ollama_Export]]

Generate an Ollama-ready package including the GGUF model and appropriate Modelfile with chat template configuration.

'''What happens:'''
* GGUF model prepared (from previous step)
* Modelfile generated with correct chat template
* Template matched to model family (Llama, ChatML, etc.)
* System prompt and parameters configured

'''Modelfile contents:'''
* FROM directive pointing to GGUF file
* TEMPLATE with chat formatting
* PARAMETER settings (temperature, context length)
* SYSTEM prompt if applicable

=== Step 7: HuggingFace Hub Upload ===
[[step::Principle:unslothai_unsloth_Hub_Upload]]

Push the exported model to HuggingFace Hub for sharing or deployment. Supports both public and private repositories.

'''What happens:'''
* Repository created on HuggingFace Hub
* Model files uploaded (safetensors, config, tokenizer)
* Model card generated with training details
* GGUF files optionally included

'''Key parameters:'''
* `push_to_hub()`: Upload to Hub
* `repo_id`: Target repository name
* `private`: Public or private visibility
* `token`: HuggingFace authentication

=== Step 8: Validation ===
[[step::Principle:unslothai_unsloth_Export_Validation]]

Validate the exported model by loading and running inference. Different validation approaches for different formats:

'''Validation approaches:'''
* HuggingFace: Load with transformers and generate
* GGUF: Test with llama-cli or Ollama
* Hub: Download and verify accessibility

== Execution Diagram ==
{{#mermaid:graph TD
    A[Training Complete] --> B[Select Format]
    B --> C{Format Type}
    C -->|LoRA Only| D[Save Adapters]
    C -->|Merged| E[Merge & Save]
    C -->|GGUF| F[Convert to GGUF]
    C -->|Ollama| G[Create Modelfile]
    E --> F
    F --> G
    D --> H{Push to Hub?}
    E --> H
    F --> H
    G --> H
    H -->|Yes| I[Upload to Hub]
    H -->|No| J[Local Validation]
    I --> J
}}

== Related Pages ==
* [[step::Principle:unslothai_unsloth_Training_Verification]]
* [[step::Principle:unslothai_unsloth_Export_Format_Selection]]
* [[step::Principle:unslothai_unsloth_LoRA_Export]]
* [[step::Principle:unslothai_unsloth_Merged_Export]]
* [[step::Principle:unslothai_unsloth_GGUF_Conversion]]
* [[step::Principle:unslothai_unsloth_Ollama_Export]]
* [[step::Principle:unslothai_unsloth_Hub_Upload]]
* [[step::Principle:unslothai_unsloth_Export_Validation]]
