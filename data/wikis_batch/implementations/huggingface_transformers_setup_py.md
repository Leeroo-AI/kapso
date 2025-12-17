# Implementation: huggingface_transformers_setup_py

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Installation|https://huggingface.co/docs/transformers/installation]]
|-
! Domains
| [[domain::Package_Management]], [[domain::Installation]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Standard Python package installation script defining dependencies, extras, entry points, and package metadata for the transformers library.

=== Description ===

The `setup.py` script handles pip installation of transformers. It:
- Defines all dependencies with version constraints in `_deps` list
- Creates optional dependency groups (dev, testing, quality, etc.)
- Registers CLI entry points (`transformers-cli`)
- Handles stale egg-info cleanup for editable installs
- Documents the PyPI release process in extensive comments

=== Usage ===

Used by pip during `pip install transformers` or `pip install -e .` for development. Modify `_deps` to add/update dependency versions, then run `make deps_table_update`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/setup.py setup.py]
* '''Lines:''' 1-428

=== Signature ===
<syntaxhighlight lang="python">
# Core dependency list (synced to dependency_versions_table.py)
_deps = [
    "accelerate>=1.1.0",
    "datasets>=2.15.0",
    "huggingface-hub>=1.2.1,<2.0",
    "numpy>=1.17",
    "tokenizers>=0.22.0,<=0.23.0",
    "torch>=2.2",
    # ... 90+ dependencies
]

# Optional dependency groups
extras = {
    "dev": deps_list("torch", "pytest", "ruff", ...),
    "testing": deps_list("pytest", "parameterized", ...),
    "quality": deps_list("ruff", "libcst", ...),
    "torch": deps_list("torch", "accelerate"),
}

# Package setup
setup(
    name="transformers",
    version="4.XX.X",
    packages=find_packages("src"),
    install_requires=install_requires,
    extras_require=extras,
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# Install from PyPI
pip install transformers

# Install with extras
pip install transformers[torch]
pip install transformers[dev]

# Editable install for development
pip install -e ".[dev]"
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pip command || CLI || Yes || Installation command
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| installed package || Package || transformers package in site-packages
|-
| CLI entry points || Commands || transformers-cli command
|}

== Usage Examples ==

=== Standard Installation ===
<syntaxhighlight lang="bash">
# Basic install (core dependencies only)
pip install transformers

# With PyTorch support
pip install transformers[torch]

# With all development dependencies
pip install "transformers[dev]"

# Full extras (all backends)
pip install "transformers[torch,tokenizers,vision,audio]"
</syntaxhighlight>

=== Development Setup ===
<syntaxhighlight lang="bash">
# Clone and install in editable mode
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e ".[dev,testing,quality]"

# After modifying _deps, update version table
make deps_table_update
</syntaxhighlight>

== Related Pages ==
