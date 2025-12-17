# File: `use_existing_torch.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 18 |
| Imports | glob |

## Understanding

**Status:** ✅ Explored

**Purpose:** Removes PyTorch dependencies from vLLM requirements files to allow using pre-installed PyTorch installations.

**Mechanism:** Scans requirements/*.txt and pyproject.toml files, filters out lines containing "torch" (case-insensitive), and rewrites the files without those dependencies. Prints removed lines for verification.

**Significance:** Utility script for deployment scenarios where PyTorch is already installed (e.g., Docker images with CUDA-specific PyTorch versions). Prevents pip from reinstalling or upgrading PyTorch, which could break CUDA compatibility or waste installation time.
