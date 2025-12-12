{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Doc|Unsloth GGUF Guide|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Deployment]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Concrete tool for exporting fine-tuned models to GGUF format for use with llama.cpp, Ollama, and other local inference engines.

=== Description ===
The `save_pretrained_gguf` method exports models to GGUF format, the standard for local LLM inference. GGUF files work with llama.cpp, Ollama, LM Studio, and other popular inference frameworks. Unsloth handles the conversion automatically, including merging LoRA adapters and applying quantization.

=== Usage ===
Call this method when you need to deploy your fine-tuned model for local inference. Choose quantization level based on target hardware and quality requirements. Essential for deploying to edge devices or systems without GPU.

== Code Signature ==
<syntaxhighlight lang="python">
# Save to GGUF
model.save_pretrained_gguf(
    save_directory: str,
    tokenizer: PreTrainedTokenizer,
    quantization_method: str = "q4_k_m",  # Quantization type
)

# Push GGUF to HuggingFace Hub
model.push_to_hub_gguf(
    repo_id: str,
    tokenizer: PreTrainedTokenizer,
    quantization_method: str = "q4_k_m",
    token: str = None,
)
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * Trained model with LoRA adapters
    * Tokenizer for embedding table
    * Quantization method selection
* **Produces:**
    * `.gguf` file for local inference

== Quantization Methods ==
{| class="wikitable"
! Method !! Size (7B) !! Quality !! Speed !! Use Case
|-
|| q4_k_m || ~4GB || Good || Fast || Recommended default
|-
|| q5_k_m || ~5GB || Better || Fast || Better quality, more RAM
|-
|| q8_0 || ~8GB || Best || Medium || Quality-focused
|-
|| f16 || ~14GB || Perfect || Slow || Full precision reference
|-
|| q2_k || ~3GB || Acceptable || Fastest || Extreme compression
|}

== Example Usage ==
<syntaxhighlight lang="python">
# After training...

# Option 1: Save GGUF locally
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method = "q4_k_m",  # Good balance
)

# Option 2: Multiple quantizations
for quant in ["q4_k_m", "q5_k_m", "q8_0"]:
    model.save_pretrained_gguf(f"model_{quant}", tokenizer, quant)

# Option 3: Push to HuggingFace Hub
model.push_to_hub_gguf(
    "your-username/llama-3-finetuned-GGUF",
    tokenizer,
    quantization_method = "q4_k_m",
    token = "hf_...",
)

# Use with Ollama
# 1. Create Modelfile:
#    FROM ./model_gguf/unsloth.Q4_K_M.gguf
# 2. Run: ollama create mymodel -f Modelfile
# 3. Run: ollama run mymodel
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]

=== Tips and Tricks ===
(No specific heuristics - quantization choice depends on deployment target)

