# Implementation: huggingface_transformers_models_to_deprecate

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Analytics]], [[domain::Maintenance]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Model usage analyzer that identifies low-usage models as candidates for deprecation based on download statistics.

=== Description ===

The `utils/models_to_deprecate.py` module (335 lines) analyzes HuggingFace Hub statistics to identify models that may be candidates for deprecation. It:
- Fetches download counts from Hub API
- Analyzes usage trends over time
- Ranks models by activity level
- Generates deprecation candidate reports

=== Usage ===

Run periodically to identify models that may need deprecation discussion. Informs maintenance planning.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/models_to_deprecate.py utils/models_to_deprecate.py]
* '''Lines:''' 1-335

=== Signature ===
<syntaxhighlight lang="python">
def get_model_downloads(model_type: str) -> int:
    """Fetch download count from Hub."""

def analyze_usage_trends(model_type: str, period_days: int) -> dict:
    """Analyze download trends over time."""

def get_deprecation_candidates(threshold: int) -> list[str]:
    """Get models below activity threshold."""

def main():
    """Generate deprecation candidate report."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python utils/models_to_deprecate.py --threshold 1000
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --threshold || int || No || Minimum downloads to keep
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Candidate report || stdout || Low-usage models ranked
|}

== Usage Examples ==

=== Find Low-Usage Models ===
<syntaxhighlight lang="bash">
python utils/models_to_deprecate.py

# Output:
# Deprecation candidates (< 1000 monthly downloads):
#   1. transfo_xl: 234 downloads
#   2. retribert: 156 downloads
</syntaxhighlight>

== Related Pages ==
