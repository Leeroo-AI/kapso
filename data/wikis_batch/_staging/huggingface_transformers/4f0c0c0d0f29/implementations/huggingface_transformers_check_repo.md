# Implementation: huggingface_transformers_check_repo

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Code_Quality]], [[domain::CI_CD]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Comprehensive repository consistency validation tool performing multiple checks on model structure, auto mappings, and naming conventions.

=== Description ===

The `utils/check_repo.py` module (1309 lines) is the central quality gate for transformers. It performs:
- Auto mapping consistency (model names in AUTO_*)
- Import structure validation
- Model file naming conventions
- Configuration consistency
- Test file coverage verification

Running all sub-checks ensures the repository maintains structural integrity.

=== Usage ===

Run as the main CI quality check. Aggregates multiple validation passes into a single command.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/check_repo.py utils/check_repo.py]
* '''Lines:''' 1-1309

=== Signature ===
<syntaxhighlight lang="python">
def check_model_list() -> list[str]:
    """Verify MODEL_MAPPING consistency."""

def check_auto_mapping() -> list[str]:
    """Verify AUTO_* class mappings."""

def check_model_structure(model_name: str) -> list[str]:
    """Verify model directory structure."""

def main():
    """
    Run all repository checks.

    Returns non-zero exit code if any check fails.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# Run all checks
python utils/check_repo.py

# Run specific check
python utils/check_repo.py --check model_list
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Repository || Directory || Yes || Transformers repository
|-
| --check || str || No || Specific check to run
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Check results || stdout || Pass/fail for each check
|-
| Exit code || int || 0 if all pass, 1 otherwise
|}

== Usage Examples ==

=== Full Repository Check ===
<syntaxhighlight lang="bash">
python utils/check_repo.py

# Output:
# Checking model list... OK
# Checking auto mappings... OK
# Checking model structure... 2 issues found
#   - llama: missing configuration_llama.py
#   - mistral: test file not found
</syntaxhighlight>

== Related Pages ==
