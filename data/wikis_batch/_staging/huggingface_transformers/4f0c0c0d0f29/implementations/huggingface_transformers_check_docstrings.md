# Implementation: huggingface_transformers_check_docstrings

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Documentation]], [[domain::Code_Quality]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Docstring validation tool ensuring function signatures match their documentation Args sections across the transformers codebase.

=== Description ===

The `utils/check_docstrings.py` module (1559 lines) validates that docstrings accurately reflect function signatures. It:
- Parses function signatures using AST
- Extracts Args sections from docstrings
- Compares parameter names and types
- Reports mismatches (missing args, extra args, type inconsistencies)

This ensures documentation stays synchronized with code changes.

=== Usage ===

Run in CI as part of quality checks. Also useful during development to catch documentation drift.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/check_docstrings.py utils/check_docstrings.py]
* '''Lines:''' 1-1559

=== Signature ===
<syntaxhighlight lang="python">
def get_function_signature(func: ast.FunctionDef) -> dict:
    """Extract signature from AST node."""

def parse_docstring_args(docstring: str) -> dict:
    """Parse Args section from docstring."""

def compare_signature_and_docstring(func: ast.FunctionDef) -> list[str]:
    """Return list of mismatches."""

def main():
    """
    CLI entry point.

    Args:
        --fix: Attempt to auto-fix docstrings
        --path: Specific file/directory to check
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# Check all docstrings
python utils/check_docstrings.py

# Check specific file
python utils/check_docstrings.py --path src/transformers/trainer.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Python files || Files || Yes || Source files to validate
|-
| --path || str || No || Specific path to check
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Validation report || stdout || List of docstring issues
|-
| Exit code || int || 0 if all valid, 1 otherwise
|}

== Usage Examples ==

=== Validate Docstrings ===
<syntaxhighlight lang="bash">
python utils/check_docstrings.py

# Example output:
# trainer.py:Trainer.__init__: Missing arg 'compute_loss_func' in docstring
# modeling_bert.py:BertModel.forward: Type mismatch for 'attention_mask'
</syntaxhighlight>

== Related Pages ==
