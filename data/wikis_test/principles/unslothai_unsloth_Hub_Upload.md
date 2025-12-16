{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Hub|https://huggingface.co/docs/hub]]
|-
! Domains
| [[domain::Deployment]], [[domain::Model_Distribution]], [[domain::HuggingFace]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of uploading trained models to HuggingFace Hub for distribution and deployment.

=== Description ===

Hub upload enables:
- Public or private model sharing
- Version control with Git LFS
- Model cards and documentation
- Automatic format detection by consumers

== Practical Guide ==

=== Push Merged Model ===
<syntaxhighlight lang="python">
model.push_to_hub_merged(
    "your-username/my-model",
    tokenizer,
    save_method="merged_16bit",
    token="hf_...",
    private=False,
    commit_message="Fine-tuned with Unsloth",
)
</syntaxhighlight>

=== Push LoRA Adapters ===
<syntaxhighlight lang="python">
model.push_to_hub_merged(
    "your-username/my-model-lora",
    tokenizer,
    save_method="lora",
    token="hf_...",
)
</syntaxhighlight>

=== Push GGUF Files ===
<syntaxhighlight lang="python">
model.push_to_hub_gguf(
    "your-username/my-model-GGUF",
    tokenizer,
    quantization_method=["q4_k_m", "q5_k_m"],
    token="hf_...",
)
</syntaxhighlight>

=== Create Repository First ===
<syntaxhighlight lang="python">
from huggingface_hub import create_repo

# Create repo before upload
create_repo(
    "your-username/my-model",
    private=True,
    token="hf_...",
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GGUF_Export]]
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
