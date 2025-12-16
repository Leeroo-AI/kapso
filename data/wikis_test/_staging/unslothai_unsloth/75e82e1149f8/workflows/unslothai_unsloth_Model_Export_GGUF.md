# Workflow: Model Export to GGUF

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth Save Module|https://github.com/unslothai/unsloth/blob/main/unsloth/save.py]]
* [[source::Repo|Ollama Templates|https://github.com/unslothai/unsloth/blob/main/unsloth/ollama_template_mappers.py]]
* [[source::Repo|Save Tests|https://github.com/unslothai/unsloth/blob/main/tests/saving/test_unsloth_save.py]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Export]], [[domain::GGUF]], [[domain::Ollama]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
End-to-end process for exporting fine-tuned Unsloth models to GGUF format for deployment with llama.cpp, Ollama, or other inference frameworks.

=== Description ===
This workflow covers the complete export pipeline for taking a fine-tuned Unsloth model and converting it to GGUF (GPT-Generated Unified Format) for efficient CPU/GPU inference. The process handles:

1. **LoRA Merging** - Merging trained LoRA adapters back into base model weights
2. **Weight Dequantization** - Converting 4-bit quantized weights to full precision
3. **GGUF Conversion** - Using llama.cpp's convert script to create GGUF files
4. **Quantization** - Applying various GGUF quantization methods (q4_k_m, q5_k_m, q8_0, etc.)
5. **Ollama Integration** - Creating Modelfiles with proper chat templates for Ollama deployment

Unsloth supports 20+ GGUF quantization methods including:
- `q4_k_m`, `q5_k_m` - Recommended for quality/size balance
- `q8_0` - High quality, fast conversion
- `f16`, `bf16` - Full precision for maximum accuracy
- `q2_k`, `q3_k_m` - Aggressive compression for resource-constrained environments

=== Usage ===
Execute this workflow when:
- You have a fine-tuned Unsloth model (LoRA or merged)
- You need to deploy the model for local inference (llama.cpp, Ollama, LM Studio)
- You want to reduce model size while maintaining quality
- You need a Modelfile for Ollama deployment with proper chat formatting

**Input:** Fine-tuned model (with or without LoRA adapters)
**Output:** GGUF model file(s) + optional Ollama Modelfile

== Execution Steps ==

=== Step 1: Prepare Model for Export ===
[[step::Principle:unslothai_unsloth_LoRA_Weight_Merging]]

Ensure the model is loaded and ready for export. If continuing from training, the model is already in memory.

```python
from unsloth import FastLanguageModel

# Load existing fine-tuned model (if not already loaded)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "path/to/finetuned_model",  # or HF repo
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

=== Step 2: Save Merged Model to HuggingFace Format ===
[[step::Principle:unslothai_unsloth_LoRA_Weight_Merging]]

Before GGUF conversion, merge LoRA weights and save in HuggingFace format.

```python
# Merge LoRA adapters into base model and save as 16-bit
model.save_pretrained_merged(
    "merged_model_16bit",
    tokenizer,
    save_method = "merged_16bit",  # Full precision merge
    # Alternative: "merged_4bit" for smaller intermediate files
)
```

=== Step 3: Convert to GGUF Format ===
[[step::Principle:unslothai_unsloth_GGUF_Model_Quantization]]

Convert the merged model directly to GGUF using Unsloth's integrated llama.cpp support.

```python
# Direct conversion to GGUF with quantization
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method = "q4_k_m",  # Recommended for quality/size
)

# Or convert to multiple quantization methods
model.save_pretrained_gguf(
    "model_gguf_multi",
    tokenizer,
    quantization_method = ["q4_k_m", "q5_k_m", "q8_0"],
)
```

**Available Quantization Methods:**
| Method | Description | Use Case |
|--------|-------------|----------|
| `f16` | Float16, 100% accuracy | Maximum quality |
| `bf16` | BFloat16, 100% accuracy | Fast conversion |
| `q8_0` | 8-bit quantization | High quality, reasonable size |
| `q5_k_m` | 5-bit with Q6_K for some tensors | Balanced quality/size |
| `q4_k_m` | 4-bit with Q6_K for some tensors | **Recommended default** |
| `q3_k_m` | 3-bit quantization | Aggressive compression |
| `q2_k` | 2-bit quantization | Extreme compression |

=== Step 4: Create Ollama Modelfile (Optional) ===
[[step::Principle:unslothai_unsloth_Chat_Template_Formatting]]

Generate a Modelfile with proper chat template for Ollama deployment.

```python
# Automatic Modelfile generation with correct chat template
model.save_pretrained_gguf(
    "ollama_model",
    tokenizer,
    quantization_method = "q4_k_m",
    create_ollama_modelfile = True,  # Generate Modelfile
)
```

Unsloth automatically detects the correct chat template from 40+ supported formats including:
- Llama 3.x Instruct
- ChatML (Qwen, Mistral variants)
- Alpaca
- Gemma
- Phi-3
- And many more...

=== Step 5: Push to HuggingFace Hub (Optional) ===
[[step::Principle:unslothai_unsloth_GGUF_Model_Quantization]]

Upload GGUF files directly to HuggingFace Hub.

```python
# Push GGUF to HuggingFace Hub
model.push_to_hub_gguf(
    "username/model-name-GGUF",
    tokenizer,
    quantization_method = ["q4_k_m", "q5_k_m"],
    token = "hf_token",
)
```

=== Step 6: Deploy with Ollama ===
[[step::Principle:unslothai_unsloth_Chat_Template_Formatting]]

Use the generated files with Ollama for local inference.

```bash
# Create model in Ollama
ollama create my-model -f Modelfile

# Run inference
ollama run my-model "Hello, how are you?"
```

== Execution Diagram ==

{{#mermaid:graph TD
    A[Fine-tuned Model] --> B{Has LoRA?}
    B -->|Yes| C[Merge LoRA Weights]
    B -->|No| D[Model Ready]
    C --> D
    D --> E[Convert to GGUF]
    E --> F{Quantization Method}
    F -->|q4_k_m| G[4-bit GGUF]
    F -->|q8_0| H[8-bit GGUF]
    F -->|f16| I[16-bit GGUF]
    G --> J{Create Modelfile?}
    H --> J
    I --> J
    J -->|Yes| K[Generate Ollama Modelfile]
    J -->|No| L[GGUF Ready]
    K --> L
    L --> M{Push to Hub?}
    M -->|Yes| N[push_to_hub_gguf]
    M -->|No| O[Local Deployment]
}}

== Related Pages ==

=== Execution Steps ===
* [[step::Principle:unslothai_unsloth_LoRA_Weight_Merging]] - Steps 1-2: Model Preparation and Merging
* [[step::Principle:unslothai_unsloth_GGUF_Model_Quantization]] - Steps 3, 5: GGUF Conversion and Hub Upload
* [[step::Principle:unslothai_unsloth_Chat_Template_Formatting]] - Steps 4, 6: Ollama Integration

=== Key Implementations ===
* [[implemented_by::Implementation:unslothai_unsloth_unsloth_save_model]] - Core save functionality
* [[implemented_by::Implementation:unslothai_unsloth_save_to_gguf]] - GGUF conversion
* [[implemented_by::Implementation:unslothai_unsloth_OLLAMA_TEMPLATES]] - Chat template mappings
