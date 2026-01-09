# File: `unsloth/_auto_install.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 41 |
| Imports | packaging, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generate platform-specific pip install command for Unsloth

**Mechanism:** Detects PyTorch version (2.1.1 through 2.9.1) and CUDA version (11.8, 12.1, 12.4, 12.6, 12.8, 13.0), validates compatibility constraints, maps to appropriate unsloth-zoo wheel identifier, and prints pip install command with correct version tag for user's environment

**Significance:** Critical installation utility that ensures users install the correct pre-built binary wheels matching their exact PyTorch/CUDA configuration, preventing ABI incompatibility issues and build errors
