# Implementation: huggingface_transformers_dependency_versions_check

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Installation|https://huggingface.co/docs/transformers/installation]]
|-
! Domains
| [[domain::Package_Management]], [[domain::Dependency_Validation]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Runtime dependency version validation ensuring compatible package versions are installed when importing transformers.

=== Description ===

The `dependency_versions_check` module validates that core runtime dependencies meet minimum version requirements when transformers is imported. It checks packages defined in `install_requires` (like numpy, tokenizers, huggingface-hub) against the version specifications in `dependency_versions_table.py`. The module also exports `dep_version_check()` for lazy version checking of optional dependencies.

=== Usage ===

This module is automatically executed on import of transformers. Use `dep_version_check()` explicitly when you need to validate optional dependency versions before using features that require them.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/src/transformers/dependency_versions_check.py src/transformers/dependency_versions_check.py]
* '''Lines:''' 1-64

=== Signature ===
<syntaxhighlight lang="python">
# Runtime check list
pkgs_to_check_at_runtime = [
    "python",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
    "pyyaml",
]

def dep_version_check(pkg: str, hint: str = None) -> None:
    """
    Check that a dependency meets version requirements.

    Args:
        pkg: Package name to check (must exist in deps table)
        hint: Optional hint message for version error

    Raises:
        ImportError: If version requirement not met
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.dependency_versions_check import dep_version_check
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pkg || str || Yes || Package name to validate
|-
| hint || str || No || Custom error message hint
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Returns None if version check passes
|-
| ImportError || Exception || Raised if version requirement not satisfied
|}

== Usage Examples ==

=== Checking Optional Dependency ===
<syntaxhighlight lang="python">
from transformers.dependency_versions_check import dep_version_check

# Check scipy before using feature that requires it
try:
    dep_version_check("scipy", hint="Required for special functions")
    import scipy
    # Use scipy features...
except ImportError as e:
    print(f"scipy not available: {e}")
</syntaxhighlight>

=== Automatic Runtime Validation ===
<syntaxhighlight lang="python">
# Version checks happen automatically on import
import transformers  # Validates core deps (numpy, tokenizers, etc.)

# If a dependency is outdated, you'll see:
# ImportError: transformers requires numpy>=1.17 but found numpy==1.16.0
</syntaxhighlight>

== Related Pages ==
