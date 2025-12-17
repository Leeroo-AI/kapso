# Implementation: huggingface_transformers_modular_model_detector

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Code_Analysis]], [[domain::Refactoring]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Code similarity detector that identifies opportunities for modular refactoring by analyzing duplicate code patterns across model implementations.

=== Description ===

The `utils/modular_model_detector.py` module (913 lines) analyzes the codebase to find code duplication suitable for modular conversion. It:
- Compares model implementations using AST similarity metrics
- Identifies common patterns across model families
- Suggests base classes for inheritance hierarchies
- Reports duplication percentages for prioritization

=== Usage ===

Run when planning refactoring work to identify which models would benefit most from modular conversion.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/modular_model_detector.py utils/modular_model_detector.py]
* '''Lines:''' 1-913

=== Signature ===
<syntaxhighlight lang="python">
def compute_similarity(file1: str, file2: str) -> float:
    """Compute AST-based similarity between files."""

def find_duplication_opportunities() -> list[dict]:
    """Find models with high code similarity."""

def suggest_base_class(models: list[str]) -> str:
    """Suggest best base for inheritance."""

def main():
    """Analyze codebase for modular conversion opportunities."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python utils/modular_model_detector.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Model files || Files || Yes || modeling_*.py files to analyze
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Similarity report || stdout || Models ranked by duplication potential
|}

== Usage Examples ==

=== Find Duplication Opportunities ===
<syntaxhighlight lang="bash">
python utils/modular_model_detector.py

# Output:
# High similarity pairs:
#   llama <-> mistral: 87% similar
#   bert <-> roberta: 82% similar
# Suggested modular conversions:
#   mistral -> modular (base: llama)
</syntaxhighlight>

== Related Pages ==
