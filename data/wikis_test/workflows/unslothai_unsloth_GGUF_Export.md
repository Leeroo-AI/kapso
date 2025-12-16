{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|GGUF Saving Guide|https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf]]
* [[source::Repo|llama.cpp|https://github.com/ggml-org/llama.cpp]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Export]], [[domain::GGUF]], [[domain::Quantization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

== Overview ==
End-to-end process for exporting fine-tuned models to GGUF format for deployment with llama.cpp, Ollama, and other GGML-based inference engines.

=== Description ===
This workflow covers the complete pipeline for converting trained models (with or without LoRA adapters) into the GGUF format used by llama.cpp and Ollama. The process handles:

1. **LoRA Merging**: Fuse trained adapters back into base model weights
2. **Format Conversion**: Convert HuggingFace models to GGUF using llama.cpp's conversion scripts
3. **Quantization**: Apply various quantization methods (Q4_K_M, Q5_K_M, Q8_0, etc.)
4. **Ollama Integration**: Generate Modelfile templates for direct Ollama deployment

Key features:
- Automatic llama.cpp installation and compilation
- Support for 20+ quantization methods
- Sentencepiece tokenizer fixing for GGUF compatibility
- Model-specific Ollama Modelfile template generation (50+ templates)
- Direct push to HuggingFace Hub as GGUF

=== Usage ===
Execute this workflow when:
- You have a fine-tuned model (LoRA or full) that needs deployment
- You want to run inference on CPU or consumer hardware
- You need to deploy via Ollama for local LLM serving
- You want smaller model files with acceptable accuracy tradeoffs

Input requirements:
- A trained Unsloth model (or any HuggingFace-compatible model)
- Sufficient disk space for temporary files during conversion
- Internet connection for llama.cpp download (first run only)

Output formats:
- `.gguf` files for llama.cpp/Ollama
- Optional `Modelfile` for Ollama deployment

== Execution Steps ==

=== Step 1: Load Trained Model ===
[[step::Principle:unslothai_unsloth_Model_Loading]]
Load your fine-tuned model. This can be a checkpoint with LoRA adapters or a fully merged model.

```python
from unsloth import FastLanguageModel

# Load from saved checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./my_finetuned_model",  # Local path or HF repo
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = None,
)
```

=== Step 2: Merge LoRA Adapters ===
[[step::Principle:unslothai_unsloth_LoRA_Merging]]
If your model has LoRA adapters, they must be merged back into the base weights before GGUF conversion. This step fuses adapter weights using the formula: `W_merged = W_base + (A @ B) * scaling`.

```python
# Merging happens automatically during save_pretrained_gguf
# The save function handles:
# 1. Dequantizing 4-bit weights to float32
# 2. Computing merged weights: W + s * (A @ B)
# 3. Converting back to target precision
```

=== Step 3: Convert to GGUF Format ===
[[step::Principle:unslothai_unsloth_GGUF_Conversion]]
Use `save_pretrained_gguf()` to convert the model. Unsloth automatically handles llama.cpp installation, conversion script execution, and tokenizer fixes.

```python
# Save as GGUF with quantization
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method = "q4_k_m",  # See quantization options below
)
```

**Available Quantization Methods:**
| Method | Description | Use Case |
|--------|-------------|----------|
| `f16` | Float16, retains 100% accuracy | Development, accuracy testing |
| `bf16` | BFloat16, 100% accuracy | Ampere+ GPUs |
| `q8_0` | 8-bit quantization | High quality, moderate size |
| `q4_k_m` | 4-bit with Q6_K for important layers | **Recommended** balance |
| `q5_k_m` | 5-bit with Q6_K for important layers | Higher quality than q4_k_m |
| `q4_k_s` | 4-bit for all tensors | Smaller file size |
| `q3_k_m` | 3-bit mixed quantization | Aggressive compression |
| `q2_k` | 2-bit quantization | Extreme compression |

=== Step 4: Generate Ollama Modelfile ===
[[step::Principle:unslothai_unsloth_Ollama_Integration]]
For Ollama deployment, generate an appropriate Modelfile with chat template and parameters. Unsloth includes 50+ model-specific Ollama templates.

```python
# Save with Ollama Modelfile generation
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method = "q4_k_m",
)

# The Modelfile is automatically generated based on:
# 1. Model architecture detection
# 2. Chat template mapping
# 3. Default parameters (temperature, stop tokens, etc.)
```

Example generated Modelfile:
```
FROM ./model-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
```

=== Step 5: Deploy or Upload ===
[[step::Principle:unslothai_unsloth_Model_Deployment]]
Deploy locally with Ollama or push to HuggingFace Hub for distribution.

```python
# Push GGUF to HuggingFace Hub
model.push_to_hub_gguf(
    "your-username/model-name-GGUF",
    tokenizer,
    quantization_method = "q4_k_m",
    token = "hf_...",  # Your HF token
)

# Or deploy locally with Ollama
# ollama create mymodel -f ./model_gguf/Modelfile
```

== Execution Diagram ==
{{#mermaid:graph TD
    A[Step 1: Load Trained Model] --> B{Has LoRA?}
    B -->|Yes| C[Step 2: Merge LoRA Adapters]
    B -->|No| D[Step 3: GGUF Conversion]
    C --> D
    D --> E[Step 4: Generate Modelfile]
    E --> F{Deployment Target?}
    F -->|Ollama| G[ollama create]
    F -->|llama.cpp| H[llama-cli]
    F -->|HuggingFace| I[push_to_hub_gguf]
}}

== Related Pages ==
* [[step::Principle:unslothai_unsloth_Model_Loading]]
* [[step::Principle:unslothai_unsloth_LoRA_Merging]]
* [[step::Principle:unslothai_unsloth_GGUF_Conversion]]
* [[step::Principle:unslothai_unsloth_Ollama_Integration]]
* [[step::Principle:unslothai_unsloth_Model_Deployment]]
