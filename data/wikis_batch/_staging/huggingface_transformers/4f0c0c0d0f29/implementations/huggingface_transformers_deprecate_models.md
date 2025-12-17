# Implementation: huggingface_transformers_deprecate_models

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Maintenance]], [[domain::Code_Lifecycle]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Model deprecation automation tool that adds deprecation warnings, updates documentation, and prepares models for removal.

=== Description ===

The `utils/deprecate_models.py` module (377 lines) automates the model deprecation process. It:
- Adds deprecation warnings to model classes
- Updates model documentation with deprecation notices
- Modifies __init__.py to mark deprecated imports
- Creates tracking issues for removal timeline

=== Usage ===

Run when deprecating a model to ensure consistent deprecation messaging and documentation updates.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/deprecate_models.py utils/deprecate_models.py]
* '''Lines:''' 1-377

=== Signature ===
<syntaxhighlight lang="python">
def add_deprecation_warning(model_file: str, message: str) -> None:
    """Add deprecation warning to model class."""

def update_model_docs(model_type: str, deprecation_version: str) -> None:
    """Update documentation with deprecation notice."""

def main():
    """
    Deprecate specified models.

    Args:
        --model_type: Model to deprecate
        --version: Version when deprecated
        --removal_version: Planned removal version
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python utils/deprecate_models.py --model_type transfo_xl --version 4.40 --removal_version 5.0
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --model_type || str || Yes || Model to deprecate
|-
| --version || str || Yes || Current version
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Modified files || Files || Updated model files with warnings
|}

== Usage Examples ==

=== Deprecate Model ===
<syntaxhighlight lang="bash">
python utils/deprecate_models.py --model_type transfo_xl --version 4.40

# Adds to modeling_transfo_xl.py:
# warnings.warn("TransfoXLModel is deprecated...", FutureWarning)
</syntaxhighlight>

== Related Pages ==
