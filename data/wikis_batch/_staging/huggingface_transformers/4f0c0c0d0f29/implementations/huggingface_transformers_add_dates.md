# Implementation: huggingface_transformers_add_dates

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Documentation]], [[domain::Maintenance]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Documentation date manager that adds and updates "Added in version X.Y" annotations to model documentation.

=== Description ===

The `utils/add_dates.py` module (427 lines) manages version annotations in documentation. It:
- Tracks when models were added to transformers
- Adds "Added in version X.Y" badges to doc pages
- Updates existing annotations when moving between versions
- Generates changelog entries for new additions

=== Usage ===

Run during release preparation to ensure documentation includes accurate version information.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/add_dates.py utils/add_dates.py]
* '''Lines:''' 1-427

=== Signature ===
<syntaxhighlight lang="python">
def get_model_version(model_type: str) -> str:
    """Get version when model was added."""

def add_version_badge(doc_file: str, version: str) -> None:
    """Add version badge to documentation."""

def update_all_docs(version: str) -> None:
    """Update all model docs with versions."""

def main():
    """
    Manage documentation version annotations.

    Args:
        --add: Add new model version
        --update: Update existing versions
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python utils/add_dates.py --add llama --version 4.28
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --add || str || No || Model to add version for
|-
| --version || str || No || Version number
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Updated docs || Files || Documentation with version badges
|}

== Usage Examples ==

=== Add Version Badge ===
<syntaxhighlight lang="bash">
python utils/add_dates.py --add llama --version 4.28

# Adds to docs/source/en/model_doc/llama.md:
# <Tip>This model was added in version 4.28</Tip>
</syntaxhighlight>

== Related Pages ==
