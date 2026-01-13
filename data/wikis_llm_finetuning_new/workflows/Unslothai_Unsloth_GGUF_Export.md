# Workflow: GGUF_Export

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Doc|Ollama Documentation|https://ollama.ai/docs]]
|-
! Domains
| [[domain::Model_Deployment]], [[domain::Quantization]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-01-12 19:00 GMT]]
|}

== Overview ==

Export fine-tuned models to GGUF format for deployment with llama.cpp, Ollama, and other inference engines supporting the GGUF specification.

=== Description ===

This workflow converts fine-tuned language models (with merged LoRA weights) into the GGUF (GPT-Generated Unified Format) binary format. GGUF is the standard format for llama.cpp and Ollama, enabling efficient CPU and GPU inference without Python dependencies.

The workflow supports:
* Multiple quantization levels (q4_k_m, q5_k_m, q8_0, f16, etc.)
* Automatic Ollama Modelfile generation with correct chat templates
* Direct push to HuggingFace Hub
* Validation of converted models

=== Usage ===

Execute this workflow when you have a fine-tuned model and want to:
* Deploy for local inference with Ollama
* Run on CPU-only environments
* Reduce model size for edge deployment
* Share quantized versions on HuggingFace Hub

== Execution Steps ==

=== Step 1: Model_Preparation ===

Ensure the model is ready for GGUF conversion. If starting from a LoRA adapter, merge it into the base model first.

'''Prerequisites:'''
* Model with merged weights (use `save_pretrained_merged` if starting from LoRA)
* Model must be in HuggingFace format (config.json, model files)
* Sufficient disk space for intermediate files (2x model size typical)

'''What happens:'''
* If model has LoRA adapters, they must be merged first
* Base model weights combined with adapter deltas
* Result saved in 16-bit precision as starting point

=== Step 2: GGUF_Conversion ===

Convert the HuggingFace model to GGUF format. Unsloth handles llama.cpp installation and conversion automatically.

'''What happens:'''
* llama.cpp conversion scripts invoked automatically
* Model architecture mapped to GGUF tensor layout
* Tokenizer vocabulary exported to GGUF metadata
* Initial F16 GGUF file created

'''Key considerations:'''
* Conversion requires llama.cpp (auto-installed if missing)
* Some model architectures may have limited GGUF support
* Sentencepiece tokenizers require special handling

=== Step 3: Quantization_Selection ===

Choose the appropriate quantization method based on quality/size tradeoffs. Unsloth supports all standard llama.cpp quantization methods.

'''Common quantization options:'''
* `q4_k_m` - Recommended balance of quality and size
* `q5_k_m` - Higher quality, moderate size increase
* `q8_0` - Near-lossless, larger files
* `f16` - No quantization, maximum quality
* `q2_k` - Maximum compression, quality loss

'''Quantization characteristics:'''
* Lower bits = smaller files, faster inference, lower quality
* K-quant methods (q4_k_m, q5_k_m) use mixed precision for critical tensors
* Quality impact varies by model size (larger models tolerate more compression)

=== Step 4: GGUF_Quantization ===

Apply the selected quantization to produce the final GGUF file. Multiple quantization levels can be produced from a single F16 conversion.

'''What happens:'''
* llama-quantize tool processes F16 GGUF
* Weights converted to target precision
* Quantization parameters stored in GGUF metadata
* Compressed GGUF file produced

'''Output structure:'''
* Single .gguf file containing model and metadata
* File size varies by quantization (7B model: ~4GB for q4_k_m, ~7GB for q8_0)

=== Step 5: Ollama_Template_Generation ===

Generate an Ollama Modelfile with the correct chat template for the fine-tuned model. This ensures proper prompt formatting during inference.

'''What happens:'''
* Chat template extracted from tokenizer configuration
* Template converted to Ollama's Go template format
* System prompt and stop tokens configured
* Modelfile written alongside GGUF

'''Modelfile contents:'''
* FROM directive pointing to GGUF file
* TEMPLATE with model's chat format
* PARAMETER settings (temperature, stop tokens)
* Optional SYSTEM default prompt

=== Step 6: Validation_and_Publishing ===

Test the converted model and optionally publish to HuggingFace Hub.

'''Validation steps:'''
* Load in Ollama: `ollama create model-name -f Modelfile`
* Test inference: `ollama run model-name "test prompt"`
* Verify chat template formatting
* Compare outputs with original model

'''Publishing options:'''
* Push to HuggingFace Hub with `push_to_hub=True`
* Include multiple quantization variants
* Add model card with training details

== Execution Diagram ==

{{#mermaid:graph TD
    A[Model_Preparation] --> B[GGUF_Conversion]
    B --> C[Quantization_Selection]
    C --> D[GGUF_Quantization]
    D --> E[Ollama_Template_Generation]
    E --> F[Validation_and_Publishing]
}}

== GitHub URL ==

The executable implementation will be available at:

[[github_url::PENDING_REPO_BUILD]]

<!-- This URL will be populated by the repo builder phase -->
