# Implementation: huggingface_transformers_tests_fetcher

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Testing]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Intelligent test selection system that identifies which tests to run based on modified files in a PR, using import dependency analysis.

=== Description ===

The `tests_fetcher` module (1187 lines) implements a two-stage test selection strategy:
1. **Stage 1**: Identify modified files by diffing from branch point, excluding doc-only changes
2. **Stage 2**: Build dependency graph via import analysis, then trace which tests are impacted

This dramatically reduces CI time by only running tests affected by changes. If too many models are impacted (>30), it falls back to running core model tests only.

=== Usage ===

Run as CLI script in CI pipelines to generate the list of tests to execute. Essential for efficient PR validation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/tests_fetcher.py utils/tests_fetcher.py]
* '''Lines:''' 1-1187

=== Signature ===
<syntaxhighlight lang="python">
def checkout_commit(repo: Repo, commit_id: str):
    """Context manager for temporary checkout."""

def clean_code(content: str) -> str:
    """Remove comments/docstrings to detect real changes."""

def get_modified_files() -> list[str]:
    """Get files modified in current PR."""

def get_module_dependencies(module: str) -> set[str]:
    """Extract import dependencies from module."""

def get_tests_for_modified_files(modified_files: list[str]) -> list[str]:
    """Map modified files to impacted tests."""

def main():
    """
    CLI entry point.

    Args:
        --diff_with_last_commit: Compare with previous commit (for main branch)
        --output_file: Write test list to file
        --filter_models: Only include core model tests
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# Fetch tests for PR
python utils/tests_fetcher.py

# Fetch tests for main branch commit
python utils/tests_fetcher.py --diff_with_last_commit
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| git repo || Directory || Yes || Transformers git repository
|-
| --diff_with_last_commit || Flag || No || Use last commit diff
|-
| --filter_models || Flag || No || Filter to core models only
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| test_list.txt || File || List of test files to run
|-
| stdout || Text || JSON with test selection info
|}

== Usage Examples ==

=== CI Pipeline Usage ===
<syntaxhighlight lang="bash">
# In GitHub Actions workflow
python utils/tests_fetcher.py --output_file test_list.txt
pytest $(cat test_list.txt)
</syntaxhighlight>

=== Development Usage ===
<syntaxhighlight lang="bash">
# See what tests would run for your changes
python utils/tests_fetcher.py

# Output example:
# Modified files: ['src/transformers/models/bert/modeling_bert.py']
# Impacted tests: ['tests/models/bert/test_modeling_bert.py']
</syntaxhighlight>

== Related Pages ==
