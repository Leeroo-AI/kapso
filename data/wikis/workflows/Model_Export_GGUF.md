{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Doc|Ollama|https://ollama.ai/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Deployment]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
End-to-end process for exporting fine-tuned models to GGUF format for deployment with llama.cpp, Ollama, and local inference engines.

=== Description ===
This workflow converts fine-tuned models from HuggingFace format to GGUF for efficient local inference. GGUF is the standard format for running LLMs on CPUs and consumer hardware. Unsloth handles the full pipeline: merging LoRA adapters, converting to GGUF, and applying quantization.

=== Usage ===
Execute this workflow after fine-tuning when you need to deploy your model for local inference. Use when targeting edge devices, offline applications, or users without GPU access. GGUF enables running 7B models on laptops and smartphones.

== Execution Steps ==
=== Step 1: Load Trained Model ===
Load your fine-tuned model with LoRA adapters.

<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",
    max_seq_length = 2048,
    load_in_4bit = True,
)
</syntaxhighlight>

=== Step 2: Choose Quantization Level ===
Select quantization based on target deployment.

<syntaxhighlight lang="python">
# Quantization options for GGUF:
# - q4_k_m: Best balance (recommended)
# - q5_k_m: Higher quality, larger file
# - q8_0: Best quality, largest file
# - q2_k: Smallest file, lower quality
# - f16: Full precision, reference only

quantization = "q4_k_m"
</syntaxhighlight>

=== Step 3: Export to GGUF ===
Convert and save as GGUF file.

<syntaxhighlight lang="python">
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method = quantization,
)

print(f"Saved to model_gguf/unsloth.{quantization.upper()}.gguf")
</syntaxhighlight>

=== Step 4: Export Multiple Quantizations (Optional) ===
Create variants for different deployment targets.

<syntaxhighlight lang="python">
for quant in ["q4_k_m", "q5_k_m", "q8_0"]:
    model.save_pretrained_gguf(
        f"model_{quant}",
        tokenizer,
        quantization_method = quant,
    )
</syntaxhighlight>

=== Step 5: Deploy to Ollama ===
Create and run with Ollama.

<syntaxhighlight lang="bash">
# Create Modelfile
cat > Modelfile << EOF
FROM ./model_gguf/unsloth.Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"

SYSTEM "You are a helpful assistant."
EOF

# Create Ollama model
ollama create mymodel -f Modelfile

# Run
ollama run mymodel
</syntaxhighlight>

=== Step 6: Push to HuggingFace (Optional) ===
Share GGUF model on HuggingFace Hub.

<syntaxhighlight lang="python">
model.push_to_hub_gguf(
    "your-username/model-name-GGUF",
    tokenizer,
    quantization_method = "q4_k_m",
    token = "hf_...",
)
</syntaxhighlight>

== Execution Diagram ==
{{#mermaid:graph TD
    A[Load Trained Model] --> B[Select Quantization]
    B --> C[Export to GGUF]
    C --> D{Deployment Target}
    D --> E[Ollama]
    D --> F[llama.cpp]
    D --> G[LM Studio]
    D --> H[HuggingFace Hub]
    E --> I[Create Modelfile]
    I --> J[ollama create]
    J --> K[ollama run]
}}

== Size Comparison ==
{| class="wikitable"
! Format !! 7B Model Size !! Quality !! Speed
|-
|| Original (FP16) || 14 GB || 100% || Baseline
|-
|| q8_0 || 7.2 GB || ~99% || 1.3x
|-
|| q5_k_m || 4.8 GB || ~97% || 1.5x
|-
|| q4_k_m || 4.1 GB || ~95% || 1.7x
|-
|| q2_k || 2.8 GB || ~85% || 2x
|}

== Related Pages ==
=== Execution Steps ===
(Export workflow - no theoretical principles, implementation focused)

=== Tips and Tricks ===
(Quantization choice depends on deployment constraints)

