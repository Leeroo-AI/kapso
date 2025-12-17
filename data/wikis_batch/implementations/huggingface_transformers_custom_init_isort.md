# Implementation: huggingface_transformers_custom_init_isort

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Code_Quality]], [[domain::Formatting]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Custom import sorting tool that enforces transformers-specific ordering in __init__.py files while maintaining the lazy loading structure.

=== Description ===

The `utils/custom_init_isort.py` module (331 lines) sorts imports in init files while respecting:
- Lazy loading structure (_import_structure)
- Conditional backend imports
- Alphabetical ordering within groups
- Comment preservation

Standard isort can break the lazy loading system, so this custom tool is required.

=== Usage ===

Run after modifying __init__.py files to ensure consistent import ordering.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/custom_init_isort.py utils/custom_init_isort.py]
* '''Lines:''' 1-331

=== Signature ===
<syntaxhighlight lang="python">
def parse_init_imports(init_file: str) -> dict:
    """Parse import groups from init file."""

def sort_import_structure(structure: dict) -> dict:
    """Sort _import_structure alphabetically."""

def format_init_file(init_file: str) -> str:
    """Return formatted init file content."""

def main():
    """
    Sort imports in __init__.py files.

    Args:
        --check: Verify without modifying
        --fix: Apply sorting
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# Check sorting
python utils/custom_init_isort.py --check

# Fix sorting
python utils/custom_init_isort.py --fix
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| __init__.py files || Files || Yes || Init files to sort
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Formatted files || Files || Sorted init files (with --fix)
|}

== Usage Examples ==

=== Sort Init Files ===
<syntaxhighlight lang="bash">
# Check if sorting needed
python utils/custom_init_isort.py --check

# Apply sorting
python utils/custom_init_isort.py --fix
</syntaxhighlight>

== Related Pages ==
