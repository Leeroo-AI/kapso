# Implementation: huggingface_transformers_create_dummy_models

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Testing]], [[domain::CI_CD]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Tiny model generation utility creating minimal models for fast CI testing without requiring full model downloads.

=== Description ===

The `utils/create_dummy_models.py` module (1479 lines) generates "tiny" versions of all transformers models. These tiny models:
- Have minimal hidden dimensions (e.g., hidden_size=16)
- Have reduced layers (e.g., num_layers=2)
- Are small enough to run on CPU quickly
- Cover the full API surface for testing

This enables fast CI without downloading multi-GB checkpoints.

=== Usage ===

Run to regenerate tiny models when adding new architectures or updating existing ones. Models are uploaded to the HuggingFace Hub.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/create_dummy_models.py utils/create_dummy_models.py]
* '''Lines:''' 1-1479

=== Signature ===
<syntaxhighlight lang="python">
def get_tiny_config(model_type: str) -> PretrainedConfig:
    """Create minimal config for model type."""

def create_tiny_model(model_class: type, config: PretrainedConfig) -> PreTrainedModel:
    """Instantiate tiny model with random weights."""

def push_tiny_model(model: PreTrainedModel, repo_name: str) -> None:
    """Upload tiny model to Hub."""

def main():
    """
    Generate all tiny models.

    Args:
        --model_type: Specific model to generate
        --push: Upload to Hub
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# Generate all tiny models
python utils/create_dummy_models.py --all --push

# Generate specific model
python utils/create_dummy_models.py --model_type bert
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --model_type || str || No || Specific model to generate
|-
| --push || Flag || No || Upload to HuggingFace Hub
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Tiny models || Files/Hub || Minimal model checkpoints
|}

== Usage Examples ==

=== Generate Tiny Model ===
<syntaxhighlight lang="bash">
# Generate tiny BERT for testing
python utils/create_dummy_models.py --model_type bert

# Generated model available at:
# hf-internal-testing/tiny-bert
</syntaxhighlight>

== Related Pages ==
