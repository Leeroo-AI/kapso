# File: `setup.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 110 |
| Imports | setuptools |

## Understanding

**Status:** ✅ Explored

**Purpose:** Python package configuration file that defines PEFT's metadata, dependencies, and installation settings for distribution via PyPI and local installation.

**Mechanism:** Uses setuptools to configure the package with version "0.18.1.dev0", finds packages in the src/ directory, specifies core dependencies (torch>=1.13.0, transformers, accelerate>=0.21.0, huggingface_hub>=0.25.0, etc.), and defines optional dependency groups for quality/dev/test environments. Includes package data for type hints (py.typed) and CUDA extensions for BOFT. Contains detailed release checklist in comments covering version bumping, tagging, building wheels, PyPI upload, and documentation.

**Significance:** Essential packaging infrastructure that enables PEFT distribution through pip install peft. Defines the dependency contract between PEFT and the broader Hugging Face ecosystem, ensures minimum version requirements are enforced, and provides optional dependencies for contributors. The release checklist serves as critical documentation for maintainers performing releases, reducing errors in the multi-step release process.
