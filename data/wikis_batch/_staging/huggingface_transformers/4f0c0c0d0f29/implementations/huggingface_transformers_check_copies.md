# Implementation: huggingface_transformers_check_copies

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Code_Quality]], [[domain::Development_Tools]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Code duplication consistency enforcer that validates `# Copied from` annotations remain synchronized with their source implementations.

=== Description ===

The `utils/check_copies.py` module (1044 lines) ensures code consistency across the transformers codebase. Many implementations share code marked with `# Copied from module.Class` comments. This tool:
- Parses all `# Copied from` annotations
- Compares copied code against original source
- Reports drift/inconsistencies
- Can auto-fix by re-copying from source

This maintains consistency while allowing controlled code sharing across models.

=== Usage ===

Run in CI to detect stale copies, or manually with `--fix_and_overwrite` to update copied code. Part of the quality checks before merging.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/check_copies.py utils/check_copies.py]
* '''Lines:''' 1-1044

=== Signature ===
<syntaxhighlight lang="python">
def find_code_copies() -> list[tuple[str, str, str]]:
    """Find all # Copied from annotations."""

def compare_copy_with_source(copy_file: str, source: str) -> bool:
    """Check if copied code matches source."""

def fix_copy(copy_file: str, source: str) -> None:
    """Update copied code from source."""

def main():
    """
    CLI entry point.

    Args:
        --fix_and_overwrite: Auto-fix stale copies
        --check_all: Check all files (not just changed)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# Check for stale copies
python utils/check_copies.py

# Auto-fix stale copies
python utils/check_copies.py --fix_and_overwrite
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Source files || Files || Yes || Files with # Copied from annotations
|-
| --fix_and_overwrite || Flag || No || Update stale copies
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Validation report || stdout || List of stale/valid copies
|-
| Exit code || int || 0 if all copies valid, 1 otherwise
|}

== Usage Examples ==

=== Check Code Consistency ===
<syntaxhighlight lang="bash">
# Verify all copies are up-to-date
python utils/check_copies.py

# Example output:
# Found 150 code copies
# modeling_llama.py:LlamaAttention: STALE (differs from MistralAttention)
# modeling_bert.py:BertLayer: OK
</syntaxhighlight>

=== Auto-Fix Copies ===
<syntaxhighlight lang="bash">
# Update all stale copies from their sources
python utils/check_copies.py --fix_and_overwrite
</syntaxhighlight>

== Related Pages ==
