# File: `use_existing_torch.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 18 |
| Imports | glob |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility script to remove PyTorch dependencies from requirements files, allowing use of a pre-installed PyTorch version.

**Mechanism:** Iterates through all requirements/*.txt files and pyproject.toml, finds and removes any lines containing "torch" (case-insensitive). Prints removed lines for visibility.

**Significance:** Development convenience tool for environments with custom PyTorch builds (e.g., ROCm, specific CUDA versions, or development versions). Prevents dependency conflicts when the user's PyTorch installation should not be overwritten by pip.
