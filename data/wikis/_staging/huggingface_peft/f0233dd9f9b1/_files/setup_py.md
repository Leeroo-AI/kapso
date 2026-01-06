# File: `setup.py`

**Category:** package configuration

| Property | Value |
|----------|-------|
| Lines | 111 |
| Imports | setuptools.find_packages, setuptools.setup |

## Understanding

**Status:** Explored

**Purpose:** Python package configuration file that defines metadata, dependencies, and installation settings for the PEFT (Parameter-Efficient Fine-Tuning) library. Used by setuptools to build and distribute the package to PyPI.

**Mechanism:**
- Version: 0.18.1.dev0 (development version)
- Package structure:
  - Source code located in `src/` directory
  - Auto-discovers all packages under `src/`
  - Includes type hints file (py.typed) and CUDA kernels for BOFT tuner

- Dependencies:
  - Core requirements: numpy, torch>=1.13.0, transformers, accelerate>=0.21.0, safetensors, huggingface_hub>=0.25.0, psutil, pyyaml, packaging, tqdm
  - Python: >=3.10.0 (supports 3.10, 3.11, 3.12, 3.13)

- Extras (optional dependency groups):
  - `quality`: Code quality tools (black, ruff, hf-doc-builder)
  - `docs_specific`: Documentation building tools
  - `dev`: Combined quality + docs
  - `test`: Full test suite dependencies (pytest, datasets, diffusers, scipy, protobuf, sentencepiece, etc.)

- Package metadata:
  - License: Apache 2.0
  - Maintained by: HuggingFace team
  - Status: Production/Stable
  - Keywords: deep learning
  - Target audience: Developers, Education, Science/Research

- Includes detailed release checklist as comments (lines 89-111):
  - Version bumping in __init__.py and setup.py
  - Deprecation checks
  - Git tagging
  - Building distributions (bdist_wheel, sdist)
  - Testing on PyPI test server
  - Final PyPI upload
  - Release notes
  - Post-release version bump to .dev0

**Significance:** Critical infrastructure file for package distribution and dependency management. Defines the public API surface, installation requirements, and metadata for the PEFT library. The detailed release checklist ensures consistent release processes. The extras_require structure allows users to install minimal dependencies or full development environments as needed.
