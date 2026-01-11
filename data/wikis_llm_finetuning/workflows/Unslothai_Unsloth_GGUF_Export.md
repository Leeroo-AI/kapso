# Workflow: GGUF_Export

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Saving to GGUF|https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf]]
* [[source::Doc|Deployment Guide|https://unsloth.ai/docs/basics/inference-and-deployment]]
|-
! Domains
| [[domain::LLMs]], [[domain::Deployment]], [[domain::GGUF]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==
End-to-end process for exporting fine-tuned models to GGUF format for deployment with llama.cpp, Ollama, and other inference engines.

=== Description ===
This workflow converts fine-tuned Unsloth models to GGUF (GPT-Generated Unified Format) for efficient CPU and edge deployment. GGUF is the standard format used by llama.cpp, Ollama, LM Studio, and similar inference engines.

The workflow supports:
1. **LoRA Merging**: Merging trained LoRA adapters with base model weights
2. **GGUF Conversion**: Converting merged weights to GGUF format
3. **Quantization**: Applying various quantization methods (Q4_K_M, Q5_K_M, Q8_0, etc.)
4. **Ollama Integration**: Automatic Modelfile generation and local deployment

Unsloth handles the entire pipeline automatically, including:
* Building llama.cpp if not present
* Fixing tokenizer compatibility issues (especially for SentencePiece models)
* Generating appropriate chat templates for Ollama

=== Usage ===
Execute this workflow when:
* You need to deploy a fine-tuned model for CPU inference
* You want to use the model with Ollama, LM Studio, or llama.cpp
* You need a compact, quantized model for edge deployment
* You want to share your model in a widely-compatible format

'''Input requirements:'''
* Fine-tuned model (LoRA adapter or merged weights)
* Base model for merging (if using LoRA)
* Sufficient disk space for conversion (2x model size)

'''Expected outputs:'''
* GGUF file(s) in specified quantization format(s)
* Optionally: Ollama Modelfile for local deployment
* Model ready for llama.cpp/Ollama inference

== Execution Steps ==

=== Step 1: Model Preparation ===
[[step::Principle:Unslothai_Unsloth_Model_Preparation]]

Ensure the model is ready for GGUF export. If you have a LoRA adapter, it will be merged with the base model during the export process.

'''Key considerations:'''
* GGUF export works directly from trained Unsloth models
* LoRA adapters are automatically merged during conversion
* Ensure you have the same base model that was used for training
* Full fine-tuned models can also be exported

=== Step 2: Quantization Method Selection ===
[[step::Principle:Unslothai_Unsloth_Quantization_Selection]]

Choose the appropriate quantization method based on your deployment requirements. Unsloth supports all standard llama.cpp quantization formats.

'''Available quantization methods:'''
| Method | Description | Use Case |
|--------|-------------|----------|
| `f16` | 16-bit float, no quantization | Highest accuracy, large files |
| `q8_0` | 8-bit quantization | Good accuracy, moderate size |
| `q4_k_m` | 4-bit with K-quants (recommended) | Best balance of size/quality |
| `q5_k_m` | 5-bit with K-quants | Better quality than Q4 |
| `q2_k` | 2-bit quantization | Smallest size, lower quality |

'''Key considerations:'''
* `q4_k_m` is recommended for most use cases
* Use `q8_0` when accuracy is critical
* `f16` or `bf16` for maximum quality (no quantization)
* Can export multiple quantizations in one call

=== Step 3: GGUF Export Execution ===
[[step::Principle:Unslothai_Unsloth_GGUF_Export]]

Execute the GGUF export using `model.save_pretrained_gguf()`. This handles LoRA merging, conversion, and quantization automatically.

'''Key considerations:'''
* First export may take longer as llama.cpp is built
* Tokenizer fixes are applied automatically
* Export creates files in the specified directory
* Multiple quantization formats can be specified as a list

=== Step 4: Ollama Modelfile Generation ===
[[step::Principle:Unslothai_Unsloth_Ollama_Modelfile]]

Optionally generate an Ollama Modelfile for easy local deployment. Unsloth automatically detects the appropriate chat template.

'''Key considerations:'''
* Ollama templates are auto-mapped from HuggingFace templates
* System prompts can be customized in the Modelfile
* Use `ollama create` to register the model locally
* Test with `ollama run` before deployment

=== Step 5: Hub Upload ===
[[step::Principle:Unslothai_Unsloth_GGUF_Hub_Upload]]

Optionally upload GGUF files to Hugging Face Hub using `model.push_to_hub_gguf()` for sharing and deployment.

'''Key considerations:'''
* Requires Hugging Face authentication token
* Creates a new repository or updates existing
* Includes quantization method in model card
* Can push multiple quantizations to same repo

=== Step 6: Deployment Verification ===
[[step::Principle:Unslothai_Unsloth_GGUF_Verification]]

Verify the exported GGUF model works correctly with your target inference engine.

'''Verification steps:'''
1. Test with llama-cli for basic inference
2. Test with Ollama for chat capabilities
3. Compare outputs with original model
4. Benchmark inference speed and memory usage

== Execution Diagram ==
{{#mermaid:graph TD
    A[Model Preparation] --> B[Quantization Method Selection]
    B --> C[GGUF Export Execution]
    C --> D{Ollama Deployment?}
    D -->|Yes| E[Modelfile Generation]
    D -->|No| F{Hub Upload?}
    E --> F
    F -->|Yes| G[Push to Hub]
    F -->|No| H[Deployment Verification]
    G --> H
    H --> I[llama.cpp/Ollama Ready]
}}

== Related Pages ==
* [[step::Principle:Unslothai_Unsloth_Model_Preparation]]
* [[step::Principle:Unslothai_Unsloth_Quantization_Selection]]
* [[step::Principle:Unslothai_Unsloth_GGUF_Export]]
* [[step::Principle:Unslothai_Unsloth_Ollama_Modelfile]]
* [[step::Principle:Unslothai_Unsloth_GGUF_Hub_Upload]]
* [[step::Principle:Unslothai_Unsloth_GGUF_Verification]]
