# Implementation: unslothai_unsloth_push_to_hub

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Hub|https://huggingface.co/docs/hub]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Export]], [[domain::Distribution]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for uploading trained models to HuggingFace Hub for sharing and deployment.

=== Description ===

`model.push_to_hub` uploads trained models directly to HuggingFace Hub. With Unsloth, it can push:
- LoRA adapters only
- Merged 16-bit models
- GGUF quantized models

This enables easy sharing, collaboration, and deployment through HuggingFace's infrastructure.

=== Usage ===

Use this when:
- Sharing trained models publicly
- Storing models in HuggingFace for team access
- Setting up deployment pipelines via HF Inference Endpoints

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L1500-2000

=== Signature ===
<syntaxhighlight lang="python">
def push_to_hub(
    self,
    repo_id: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    save_method: str = "lora",
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Trained with Unsloth",
) -> str:
    """
    Push model to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        tokenizer: Tokenizer to push alongside model
        save_method: "lora", "merged_16bit", or "merged_4bit"
        token: HuggingFace API token
        private: Make repository private

    Returns:
        URL to the HuggingFace repository
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
# push_to_hub is added to model by Unsloth
</syntaxhighlight>

== Usage Examples ==

=== Push LoRA Adapters ===
<syntaxhighlight lang="python">
# Push only LoRA adapters (smallest, requires base model)
model.push_to_hub(
    repo_id = "username/my-model-lora",
    tokenizer = tokenizer,
    save_method = "lora",
    token = "hf_your_token",
)
</syntaxhighlight>

=== Push Merged Model ===
<syntaxhighlight lang="python">
# Push complete merged model (standalone)
model.push_to_hub(
    repo_id = "username/my-model-merged",
    tokenizer = tokenizer,
    save_method = "merged_16bit",
    token = "hf_your_token",
    private = True,  # Private repository
)
</syntaxhighlight>

=== Push GGUF ===
<syntaxhighlight lang="python">
# First convert to GGUF, then push
model.save_pretrained_gguf(
    "username/my-model-gguf",
    tokenizer = tokenizer,
    quantization_method = "q4_k_m",
    push_to_hub = True,
    token = "hf_your_token",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Hub_Upload]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
