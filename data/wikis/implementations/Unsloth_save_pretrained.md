{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
* [[source::Doc|HuggingFace Model Sharing|https://huggingface.co/docs/hub/models-uploading]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Deployment]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Concrete tool for saving fine-tuned LoRA adapters or merged models in HuggingFace format.

=== Description ===
The `save_pretrained` method saves the fine-tuned model for later use or sharing. It can save just the LoRA adapters (small, ~100MB) or merge adapters into the base model and save the full weights. Also supports pushing directly to HuggingFace Hub.

=== Usage ===
Call this method after training to persist the model. Use adapter-only saving for efficient storage and sharing; use merged saving when deploying to inference systems that don't support PEFT.

== Code Signature ==
<syntaxhighlight lang="python">
# Save LoRA adapters only
model.save_pretrained(
    save_directory: str,
    save_method: str = "lora",  # "lora" or "merged_16bit" or "merged_4bit"
)

# Push to HuggingFace Hub
model.push_to_hub(
    repo_id: str,
    token: str,
    save_method: str = "lora",
)

# Save merged model (full weights)
model.save_pretrained_merged(
    save_directory: str,
    tokenizer: PreTrainedTokenizer,
    save_method: str = "merged_16bit",  # or "merged_4bit"
)
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * Trained model with LoRA adapters
    * Save directory path or HuggingFace repo ID
    * Save method selection
* **Produces:**
    * **lora**: LoRA adapter weights (~50-200MB)
    * **merged_16bit**: Full model weights in FP16 (~14GB for 7B)
    * **merged_4bit**: Full model weights in 4-bit (~4GB for 7B)

== Example Usage ==
<syntaxhighlight lang="python">
# After training...

# Option 1: Save LoRA adapters only (recommended for storage)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Option 2: Push adapters to HuggingFace Hub
model.push_to_hub(
    "your-username/llama-3-lora-finetuned",
    token = "hf_...",
)

# Option 3: Save merged 16-bit model
model.save_pretrained_merged(
    "merged_model",
    tokenizer,
    save_method = "merged_16bit",
)

# Option 4: Push merged model to Hub
model.push_to_hub_merged(
    "your-username/llama-3-merged-finetuned",
    tokenizer,
    save_method = "merged_16bit",
    token = "hf_...",
)
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]

=== Tips and Tricks ===
(No specific heuristics - straightforward operation)

