{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai/]]
* [[source::Doc|Ollama Documentation|https://ollama.ai/]]
* [[source::Repo|llama.cpp|https://github.com/ggml-org/llama.cpp]]
|-
! Domains
| [[domain::Model_Deployment]], [[domain::MLOps]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Process of making trained language models available for inference in production environments, encompassing format selection, quantization strategy, and serving infrastructure configuration.

=== Description ===
Model deployment is the final step in the LLM fine-tuning pipeline, transforming a trained model into a form suitable for real-world inference. Deployment considerations include:

1. **Format Selection**: Choose between HuggingFace (for vLLM, TRT-LLM) or GGUF (for Ollama, llama.cpp)
2. **Quantization**: Balance model size, inference speed, and output quality
3. **Serving Infrastructure**: Select appropriate inference engine (vLLM, Ollama, etc.)
4. **Resource Allocation**: Configure GPU memory, batch sizes, and concurrency
5. **Distribution**: Push to HuggingFace Hub or deploy to private infrastructure

Deployment targets include:
- **Cloud GPU (vLLM, TRT-LLM)**: Maximum throughput for API serving
- **Local GPU (Ollama, llama.cpp)**: Development and personal use
- **CPU (llama.cpp, Ollama)**: No GPU required, slower inference
- **Edge devices**: Highly quantized models (Q2, Q3) for constrained environments

=== Usage ===
Plan model deployment when:
- Completing fine-tuning and preparing for production
- Evaluating inference speed and resource requirements
- Distributing models to team members or community
- Building applications that consume model outputs

Deployment format decision:
- **HuggingFace 16-bit**: Maximum compatibility, vLLM/SGLang serving
- **GGUF Q4_K_M**: Good balance, Ollama/llama.cpp, 4x smaller
- **GGUF Q8_0**: Near-lossless, llama.cpp, 2x smaller
- **LoRA only**: Minimal storage, requires base model at inference

== Theoretical Basis ==
Deployment optimization balances the inference triangle: latency, throughput, and quality.

'''Deployment Decision Framework:'''
<syntaxhighlight lang="text">
Deployment Decision Tree:
┌─────────────────────────────────────────────────────┐
│ Question 1: What hardware is available?             │
├─────────────────────────────────────────────────────┤
│ Cloud GPU (A100, H100)                              │
│   └─> Format: HuggingFace FP16                      │
│       Engine: vLLM or TensorRT-LLM                  │
│       Throughput: Highest                           │
├─────────────────────────────────────────────────────┤
│ Consumer GPU (RTX 3090, 4090)                       │
│   └─> Format: GGUF Q4_K_M or HF FP16               │
│       Engine: Ollama, llama.cpp, or vLLM            │
│       Throughput: Good                              │
├─────────────────────────────────────────────────────┤
│ CPU Only                                            │
│   └─> Format: GGUF Q4_K_M or Q3_K_M                │
│       Engine: Ollama, llama.cpp                     │
│       Throughput: Limited, higher latency           │
└─────────────────────────────────────────────────────┘
</syntaxhighlight>

'''Quantization Impact on Deployment:'''
<syntaxhighlight lang="python">
# Pseudo-code for deployment sizing
def estimate_deployment_requirements(model_params_b, format_type):
    """
    Estimate resource requirements for deployment.
    """
    requirements = {}

    if format_type == "fp16":
        # 2 bytes per parameter
        requirements["model_size_gb"] = model_params_b * 2
        requirements["vram_needed_gb"] = model_params_b * 2.5  # +overhead
        requirements["quality_retention"] = 1.0

    elif format_type == "q8_0":
        # ~1 byte per parameter
        requirements["model_size_gb"] = model_params_b * 1
        requirements["vram_needed_gb"] = model_params_b * 1.3
        requirements["quality_retention"] = 0.99

    elif format_type == "q4_k_m":
        # ~0.5 bytes per parameter
        requirements["model_size_gb"] = model_params_b * 0.5
        requirements["vram_needed_gb"] = model_params_b * 0.7
        requirements["quality_retention"] = 0.95

    elif format_type == "q2_k":
        # ~0.3 bytes per parameter
        requirements["model_size_gb"] = model_params_b * 0.3
        requirements["vram_needed_gb"] = model_params_b * 0.4
        requirements["quality_retention"] = 0.85

    return requirements
</syntaxhighlight>

'''Serving Engine Comparison:'''
<syntaxhighlight lang="text">
Inference Engine Characteristics:
┌──────────────────────────────────────────────────────────────┐
│ Engine      │ Formats    │ Strengths           │ Best For    │
├──────────────────────────────────────────────────────────────┤
│ vLLM        │ HF, AWQ    │ Highest throughput, │ Production  │
│             │            │ PagedAttention,     │ API serving │
│             │            │ continuous batching │             │
├──────────────────────────────────────────────────────────────┤
│ Ollama      │ GGUF       │ Easy setup, local   │ Development │
│             │            │ deployment, macOS   │ Local use   │
├──────────────────────────────────────────────────────────────┤
│ llama.cpp   │ GGUF       │ CPU efficiency,     │ Edge, CPU   │
│             │            │ minimal deps,       │ deployment  │
│             │            │ broad platform      │             │
├──────────────────────────────────────────────────────────────┤
│ TRT-LLM     │ Engines    │ NVIDIA optimized,   │ Enterprise  │
│             │            │ lowest latency      │ deployment  │
└──────────────────────────────────────────────────────────────┘
</syntaxhighlight>

'''Hub Distribution:'''
<syntaxhighlight lang="python">
# Pseudo-code for HuggingFace Hub upload
def deploy_to_hub(model_path, repo_id, token, format_type="merged_16bit"):
    """
    Push model to HuggingFace Hub for distribution.
    """
    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if needed
    api.create_repo(repo_id, token=token, exist_ok=True)

    # Upload all files
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        token=token,
    )

    # Add model card with usage instructions
    model_card = generate_model_card(repo_id, format_type)
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
    )
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_gguf]]

=== Tips and Tricks ===
