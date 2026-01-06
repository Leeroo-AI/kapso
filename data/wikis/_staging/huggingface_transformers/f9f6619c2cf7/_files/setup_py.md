# File: `setup.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 428 |
| Classes | `DepsTableUpdateCommand` |
| Functions | `deps_list` |
| Imports | pathlib, re, setuptools, shutil |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the transformers package installation configuration for pip/PyPI, including dependencies, optional extras, metadata, and distribution settings. This is the standard Python packaging file that enables `pip install transformers`.

**Mechanism:** The script defines a comprehensive dependency list in `_deps` with version constraints, then creates a lookup table `deps` mapping package names to their version specifications. It defines numerous extras_require groups for optional features (torch, tokenizers, audio, vision, testing, quality, etc.) allowing users to install subsets like `pip install transformers[torch]` or `pip install transformers[dev]`. The `DepsTableUpdateCommand` custom command generates `src/transformers/dependency_versions_table.py` for programmatic access to dependency versions. Core install_requires includes essential packages (filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, tokenizers, safetensors, tqdm). The setup() call configures package metadata (name, version, author, description, license), package discovery (finds packages in src/), entry points (CLI command), Python version requirement (>=3.10), and PyPI classifiers. It also handles cleanup of stale egg-info directories.

**Significance:** This is the foundational file that makes transformers installable and distributable as a Python package. The extensive extras system allows users to install only what they need (e.g., torch-only, audio-only, dev with all tools), keeping base installations lightweight while supporting the library's broad capabilities. The dependency version table generation ensures consistency between setup.py and runtime dependency checking. The comprehensive metadata and release instructions in comments make this critical for the package's distribution on PyPI, where millions of users install it.
