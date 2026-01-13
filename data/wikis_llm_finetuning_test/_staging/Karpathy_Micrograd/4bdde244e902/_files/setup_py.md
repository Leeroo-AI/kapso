# File: `setup.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 22 |
| Imports | setuptools |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package configuration for PyPI distribution, enabling `pip install micrograd`.

**Mechanism:** Uses setuptools to define package metadata:
- Name: `micrograd`, version `0.1.0`
- Author: Andrej Karpathy
- Description: "A tiny scalar-valued autograd engine with a small PyTorch-like neural network library on top"
- Long description pulled from README.md
- Auto-discovers packages via `find_packages()`
- Requires Python >= 3.6
- MIT License

**Significance:** Allows the library to be installed as a proper Python package from PyPI or locally via `pip install -e .`. Makes importing micrograd convenient for users.
