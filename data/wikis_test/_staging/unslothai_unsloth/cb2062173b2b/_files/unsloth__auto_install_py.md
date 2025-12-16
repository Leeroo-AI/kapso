# File: `unsloth/_auto_install.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 41 |
| Imports | packaging, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generates the correct pip install command for Unsloth based on the user's PyTorch version, CUDA version, and GPU architecture. Provides installation guidance specific to the detected environment.

**Mechanism:**
- Validates PyTorch is installed and imports `torch`
- Extracts PyTorch version using regex and validates it's within supported range (2.1.0 to 2.9.2)
- Detects CUDA version and validates it's supported (11.8, 12.1, 12.4, 12.6, 12.8, 13.0)
- Checks GPU compute capability for Ampere architecture (compute capability >= 8.0)
- Maps PyTorch version to corresponding build suffix (e.g., `torch211`, `torch240`, `torch290`)
- Constructs pip install command with appropriate CUDA and PyTorch-specific wheel
- Prints installation command that includes both `unsloth-zoo` and versioned `unsloth` package

**Significance:** This utility helps users install the correct Unsloth build for their specific environment, avoiding compatibility issues. The version-specific builds are necessary because of PyTorch's ABI changes between versions and CUDA version requirements. This prevents installation failures and runtime errors due to version mismatches, making the installation process more reliable across different environments.
