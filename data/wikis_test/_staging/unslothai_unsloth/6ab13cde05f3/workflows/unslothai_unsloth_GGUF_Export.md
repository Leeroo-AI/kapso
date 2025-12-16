{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|GGUF Export Guide|https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf]]
* [[source::Doc|Ollama Integration|https://docs.unsloth.ai/basics/running-and-saving-models/ollama]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Quantization]], [[domain::Deployment]], [[domain::GGUF]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==
End-to-end process for exporting fine-tuned models to GGUF format for deployment with llama.cpp, Ollama, and other inference engines.

=== Description ===
This workflow covers the conversion of trained Unsloth models to GGUF (GPT-Generated Unified Format), the standard format for efficient CPU/GPU inference using llama.cpp and derivative tools. The workflow includes:

* **Weight Merging**: Combining LoRA adapters with base model weights to 16-bit precision
* **GGUF Conversion**: Using llama.cpp's convert scripts to create GGUF files
* **Quantization**: Applying various quantization schemes (q4_k_m, q5_k_m, q8_0, etc.) to reduce file size
* **Ollama Integration**: Creating Modelfiles for easy Ollama deployment

Supported quantization methods include:
* `f16` / `bf16`: Full precision (largest, most accurate)
* `q8_0`: 8-bit quantization (good balance)
* `q4_k_m` / `q5_k_m`: Recommended mixed precision (small, fast, accurate)
* `q2_k` / `q3_k_s`: Extreme compression (smallest, some quality loss)

=== Usage ===
Execute this workflow when:
* You have a trained model (LoRA or merged) ready for deployment
* You need to run inference on CPU or consumer GPUs without Python dependencies
* You want to deploy via Ollama, llama.cpp CLI, or compatible tools
* You need smaller model files for distribution

Input requirements:
* A trained Unsloth model (LoRA adapters or merged weights)
* Disk space for temporary files (~2x model size)
* llama.cpp (auto-installed by Unsloth if needed)

Output:
* GGUF file(s) at specified quantization level(s)
* Optional Ollama Modelfile for easy deployment

== Execution Steps ==

=== Step 1: Prepare Trained Model ===
[[step::Principle:unslothai_unsloth_Model_Preparation]]
Ensure your model is ready for export. If you have LoRA adapters, they must be merged before GGUF conversion.

```python
from unsloth import FastLanguageModel
import torch

# Load your trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",  # Your trained model path
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = None,
)

# Or if continuing from training:
# model, tokenizer = ... (already loaded from training)
```

=== Step 2: Merge LoRA Weights (if applicable) ===
[[step::Principle:unslothai_unsloth_Weight_Merging]]
Merge LoRA adapter weights into the base model. This is required before GGUF conversion.

```python
# Save merged model to 16-bit (required for GGUF)
model.save_pretrained_merged(
    "merged_model_16bit",
    tokenizer,
    save_method = "merged_16bit",
)
```

=== Step 3: Export to GGUF ===
[[step::Principle:unslothai_unsloth_GGUF_Conversion]]
Convert the merged model to GGUF format. Unsloth handles llama.cpp installation and conversion automatically.

```python
# Export to GGUF with quantization
# This automatically installs llama.cpp if needed
model.save_pretrained_gguf(
    "gguf_model",              # Output directory
    tokenizer,
    quantization_method = "q4_k_m",  # Recommended quantization
)

# Multiple quantization methods at once
model.save_pretrained_gguf(
    "gguf_model",
    tokenizer,
    quantization_method = ["q4_k_m", "q5_k_m", "q8_0"],
)
```

=== Step 4: Verify GGUF Output ===
[[step::Principle:unslothai_unsloth_GGUF_Validation]]
Verify the exported GGUF file is valid and check its properties.

```python
# The output files will be named:
# - gguf_model/unsloth.Q4_K_M.gguf
# - gguf_model/unsloth.Q5_K_M.gguf
# - gguf_model/unsloth.Q8_0.gguf

# Test with llama.cpp (if installed)
import subprocess
result = subprocess.run([
    "llama-cli",
    "-m", "gguf_model/unsloth.Q4_K_M.gguf",
    "-p", "Hello, how are you?",
    "-n", "50"
], capture_output=True, text=True)
print(result.stdout)
```

=== Step 5: Push to HuggingFace Hub (Optional) ===
[[step::Principle:unslothai_unsloth_Hub_Upload]]
Upload GGUF files directly to HuggingFace Hub for distribution.

```python
# Push GGUF to Hub
model.push_to_hub_gguf(
    "your-username/model-name-GGUF",
    tokenizer,
    quantization_method = ["q4_k_m", "q5_k_m"],
    token = "hf_...",
)
```

=== Step 6: Create Ollama Modelfile (Optional) ===
[[step::Principle:unslothai_unsloth_Ollama_Integration]]
Generate an Ollama Modelfile for easy deployment via Ollama.

```python
# Unsloth auto-generates Ollama-compatible templates
# based on the model's chat template

# Manual Modelfile creation:
modelfile_content = '''
FROM ./unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
"""

PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
'''

# Save Modelfile
with open("gguf_model/Modelfile", "w") as f:
    f.write(modelfile_content)

# Deploy with Ollama:
# ollama create mymodel -f Modelfile
# ollama run mymodel
```

== Execution Diagram ==
{{#mermaid:graph TD
    A[Load Trained Model] --> B{Has LoRA Adapters?}
    B -->|Yes| C[Merge LoRA to 16-bit]
    B -->|No| D[Model Ready]
    C --> D
    D --> E[Export to GGUF]
    E --> F[Select Quantization]
    F --> G[q4_k_m: Balanced]
    F --> H[q5_k_m: Higher Quality]
    F --> I[q8_0: Highest Quality]
    G --> J[Verify GGUF Output]
    H --> J
    I --> J
    J --> K{Deploy Target}
    K -->|Hub| L[Push to HuggingFace]
    K -->|Ollama| M[Create Modelfile]
    K -->|llama.cpp| N[Use Directly]
}}

== Quantization Methods Reference ==

| Method | BPW | Quality | Speed | Use Case |
|--------|-----|---------|-------|----------|
| `f16`/`bf16` | 16.0 | Best | Slow | Reference/Testing |
| `q8_0` | 8.0 | Excellent | Good | Quality-focused |
| `q5_k_m` | 5.5 | Very Good | Fast | Recommended |
| `q4_k_m` | 4.5 | Good | Very Fast | Default choice |
| `q3_k_m` | 3.5 | OK | Fastest | Resource-limited |
| `q2_k` | 2.5 | Basic | Fastest | Extreme compression |

== Related Pages ==
* [[step::Principle:unslothai_unsloth_Model_Preparation]]
* [[step::Principle:unslothai_unsloth_Weight_Merging]]
* [[step::Principle:unslothai_unsloth_GGUF_Conversion]]
* [[step::Principle:unslothai_unsloth_GGUF_Validation]]
* [[step::Principle:unslothai_unsloth_Hub_Upload]]
* [[step::Principle:unslothai_unsloth_Ollama_Integration]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Quantization_Method_Selection]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Optimization]]
