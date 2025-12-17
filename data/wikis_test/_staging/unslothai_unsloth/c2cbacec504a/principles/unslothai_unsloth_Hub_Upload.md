# Principle: unslothai_unsloth_Hub_Upload

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Hub|https://huggingface.co/docs/hub]]
* [[source::Doc|Model Cards|https://huggingface.co/docs/hub/model-cards]]
|-
! Domains
| [[domain::NLP]], [[domain::Distribution]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for distributing trained models through HuggingFace Hub for collaboration, deployment, and sharing.

=== Description ===

Hub Upload enables sharing trained models with:
1. **Version control**: Git-based model versioning
2. **Model cards**: Documentation and metadata
3. **Access control**: Public or private repositories
4. **Deployment**: Inference Endpoints integration
5. **Community**: Likes, downloads, discussions

This is the standard way to distribute fine-tuned models in the ML community.

=== Usage ===

Use Hub Upload when:
- Sharing models with the community
- Collaborating with team members
- Setting up production deployment
- Creating model collections

== Theoretical Basis ==

=== Repository Structure ===

<syntaxhighlight lang="python">
# HuggingFace model repository structure
repo_structure = {
    "model files": {
        "model.safetensors": "Model weights",
        "config.json": "Model architecture config",
    },
    "tokenizer files": {
        "tokenizer.json": "Tokenizer data",
        "tokenizer_config.json": "Tokenizer settings",
        "special_tokens_map.json": "Special token mappings",
    },
    "documentation": {
        "README.md": "Model card (auto-generated)",
    },
    "for GGUF": {
        "*.gguf": "Quantized models",
        "Modelfile": "Ollama configuration",
    },
}
</syntaxhighlight>

=== Access Control ===

{| class="wikitable"
|-
! Visibility !! Who Can Access !! Use Case
|-
| Public || Anyone || Open source models
|-
| Private || Repository members || Team/company models
|-
| Gated || Approved users || Requires license agreement
|}

== Practical Guide ==

=== Authentication ===

<syntaxhighlight lang="python">
# Login to HuggingFace Hub
from huggingface_hub import login
login(token="hf_your_token")

# Or use environment variable
import os
os.environ["HF_TOKEN"] = "hf_your_token"
</syntaxhighlight>

=== Model Card Best Practices ===

<syntaxhighlight lang="markdown">
---
license: apache-2.0
base_model: unsloth/Llama-3.2-1B-Instruct
tags:
- unsloth
- fine-tuned
---

# Model Name

## Description
Brief description of what the model does.

## Training Details
- Base model: ...
- Training data: ...
- Training procedure: QLoRA with Unsloth

## Usage
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("username/model")
```
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_push_to_hub]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Model_Export]]
