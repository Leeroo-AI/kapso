# File: `unsloth/_auto_install.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 41 |
| Imports | packaging, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generates automatic installation commands for Unsloth based on detected CUDA and PyTorch versions.

**Mechanism:** Detects torch version, CUDA version, and device capability (Ampere GPU). Maps version combinations to specific unsloth-zoo build variants with correct CUDA toolkit compatibility strings. Validates supported versions and outputs pip install command with correct build identifier.

**Significance:** Enables users to quickly get the right Unsloth build for their environment without manual configuration of complex version dependencies. Prevents installation of incompatible builds that would fail at runtime.
