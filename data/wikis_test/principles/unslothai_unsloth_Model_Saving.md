{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Save Guide|https://docs.unsloth.ai/basics/running-and-saving-models]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Deployment]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of persisting trained models to disk or HuggingFace Hub in various formats for deployment or continued training.

=== Description ===

Model saving in Unsloth supports multiple output formats:

**Save Methods:**
1. **LoRA only**: Save adapter weights (~50-200MB), requires base model at inference
2. **Merged 16-bit**: Combine adapters with base, save full model (multi-GB)
3. **Merged 4-bit**: Merge and requantize for inference efficiency
4. **GGUF**: Convert to llama.cpp format for local deployment

**Output Destinations:**
- Local directory
- HuggingFace Hub (public or private repos)

**Considerations:**
- LoRA saves are fastest and smallest
- Merged saves create standalone models
- GGUF enables deployment without Python

=== Usage ===

Save models when:
- Training is complete and you want to preserve checkpoints
- Preparing models for deployment
- Sharing models on HuggingFace Hub
- Creating backups during long training runs

== Practical Guide ==

=== Save LoRA Adapters Only ===
<syntaxhighlight lang="python">
# Fastest, smallest files
# Requires base model + PEFT to load
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Or using the unified API
model.save_pretrained_merged(
    "lora_model",
    tokenizer,
    save_method="lora",
)
</syntaxhighlight>

=== Save Merged Model ===
<syntaxhighlight lang="python">
# Standalone model, no PEFT needed
model.save_pretrained_merged(
    "merged_model",
    tokenizer,
    save_method="merged_16bit",
)
</syntaxhighlight>

=== Push to HuggingFace Hub ===
<syntaxhighlight lang="python">
# Push LoRA adapters
model.push_to_hub_merged(
    "your-username/my-lora-adapter",
    tokenizer,
    save_method="lora",
    token="hf_...",
    private=True,  # Optional
)

# Push merged model
model.push_to_hub_merged(
    "your-username/my-merged-model",
    tokenizer,
    save_method="merged_16bit",
    token="hf_...",
)
</syntaxhighlight>

=== Save to GGUF ===
<syntaxhighlight lang="python">
# For llama.cpp / Ollama deployment
model.save_pretrained_gguf(
    "gguf_model",
    tokenizer,
    quantization_method="q4_k_m",
)

# Push GGUF to Hub
model.push_to_hub_gguf(
    "your-username/my-model-GGUF",
    tokenizer,
    quantization_method=["q4_k_m", "q5_k_m"],
    token="hf_...",
)
</syntaxhighlight>

=== Checkpoint During Training ===
<syntaxhighlight lang="python">
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,  # Keep only last 3 checkpoints
    ),
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]

=== See Also ===
* [[Principle:unslothai_unsloth_Weight_Merging]] - Detailed LoRA merge process
* [[Principle:unslothai_unsloth_GGUF_Conversion]] - GGUF export details

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
