# File: `unsloth/_auto_install.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 41 |
| Imports | packaging, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generates pip install command for correct Unsloth version based on CUDA and PyTorch versions.

**Mechanism:**
- Detects PyTorch version and CUDA version from `torch.version.cuda`
- Maps PyTorch version to compatible Unsloth wheel variant (e.g., `cu118-torch240`)
- Validates CUDA version is supported (11.8, 12.1, 12.4, 12.6, 12.8, 13.0)
- Validates PyTorch version is in supported range (2.1.1 to 2.9.x)
- Checks for Ampere+ GPU capability (compute capability >= 8)
- Prints the correct pip install command with version-specific extras

**Significance:** Helper utility for users to install the correct Unsloth version matching their environment. Ensures binary compatibility between Unsloth, PyTorch, and CUDA.
