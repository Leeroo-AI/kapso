# Dependency Versions Check

| Metadata | Value |
|----------|-------|
| **Sources** | `src/transformers/dependency_versions_check.py` |
| **Domains** | Package Management, Dependency Validation, Runtime Checks |
| **Last Updated** | 2025-12-18 |

## Overview

The Dependency Versions Check module validates that critical runtime dependencies meet the required version constraints defined in the transformers library. It performs automatic version checking at import time for core packages and provides on-demand validation for optional dependencies.

## Description

This module implements a two-tier dependency validation strategy:

1. **Automatic Runtime Checks**: A predefined list of critical packages (`pkgs_to_check_at_runtime`) is validated immediately when the module is imported, ensuring that essential dependencies like `numpy`, `tokenizers`, `huggingface-hub`, and `safetensors` meet version requirements.

2. **On-Demand Validation**: The `dep_version_check()` function allows selective validation of optional dependencies when specific features are used.

The module integrates with `dependency_versions_table.py` for centralized version requirements and uses the `versions` utility module for actual version comparison logic. Special handling is implemented for conditional dependencies like `tokenizers` and `accelerate`, which are only checked if installed.

### Key Features

- Automatic validation of 12 core runtime dependencies at import time
- Order-specific checking (e.g., `tqdm` before `tokenizers`) to avoid check failures
- Conditional validation for optional packages based on availability
- Integration with centralized dependency version table
- Custom error messaging through optional hints
- Fail-fast behavior for missing dependency definitions

## Usage

### Automatic Runtime Validation

```python
# Simply importing transformers triggers validation
import transformers  # Validates all pkgs_to_check_at_runtime
```

### Manual Dependency Checking

```python
from transformers.dependency_versions_check import dep_version_check

# Check a specific optional dependency with custom hint
dep_version_check("torch", hint="PyTorch is required for model training")
```

### Extending Runtime Checks

```python
# The runtime check list is defined at module level
pkgs_to_check_at_runtime = [
    "python",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
    "pyyaml",
]
```

## Code Reference

### Primary Functions

```python
def dep_version_check(pkg: str, hint: Optional[str] = None) -> None:
    """
    Validate a package version against requirements.

    Args:
        pkg: Package name (must exist in deps table)
        hint: Optional error message to display if check fails

    Raises:
        ValueError: If pkg not found in dependency table
        VersionMismatchError: If installed version doesn't meet requirements
    """
```

### Module-Level Execution

```python
# Runtime validation loop (executes at import)
for pkg in pkgs_to_check_at_runtime:
    if pkg in deps:
        # Special handling for conditional dependencies
        if pkg == "tokenizers":
            if not is_tokenizers_available():
                continue
        elif pkg == "accelerate":
            if not is_accelerate_available():
                continue
        require_version_core(deps[pkg])
    else:
        raise ValueError(f"can't find {pkg} in {deps.keys()}")
```

### Imports

```python
from .dependency_versions_table import deps
from .utils.versions import require_version, require_version_core
```

## I/O Contract

### Input Specifications

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pkg` | str | Yes | Package name to validate (must exist in `deps` table) |
| `hint` | str | No | Custom error message for validation failures |

### Output Specifications

| Return Type | Description |
|-------------|-------------|
| None | Function returns nothing on success |

### Side Effects

- Raises `ValueError` if package not found in dependency table
- Raises version-related exceptions if installed version doesn't meet requirements
- Imports conditional modules (`is_tokenizers_available`, `is_accelerate_available`) during runtime checks

### Error Conditions

| Error Type | Condition | Example |
|------------|-----------|---------|
| ValueError | Package not in deps table | `can't find custom_pkg in dict_keys([...])` |
| VersionMismatchError | Version constraint not met | Package version too old/new |

## Usage Examples

### Example 1: Basic Dependency Check

```python
from transformers.dependency_versions_check import dep_version_check

# Check PyTorch version before using torch features
try:
    dep_version_check("torch")
    # Safe to use torch features
    import torch
    model = model.to(torch.device("cuda"))
except Exception as e:
    print(f"PyTorch version check failed: {e}")
```

### Example 2: Dependency Check with Custom Hint

```python
from transformers.dependency_versions_check import dep_version_check

def use_accelerate_features():
    dep_version_check(
        "accelerate",
        hint="Accelerate 1.1.0+ required for distributed training features"
    )
    from accelerate import Accelerator
    accelerator = Accelerator()
    return accelerator
```

### Example 3: Conditional Feature Validation

```python
from transformers.dependency_versions_check import dep_version_check

def save_model_safetensors(model, path):
    """Save model using safetensors format."""
    # Validate safetensors is available and version is correct
    dep_version_check("safetensors")

    from safetensors.torch import save_file
    save_file(model.state_dict(), path)
```

### Example 4: Validation in Custom Extensions

```python
from transformers.dependency_versions_check import dep_version_check

class CustomTrainer:
    def __init__(self, use_deepspeed=False):
        if use_deepspeed:
            # Validate deepspeed version before initialization
            dep_version_check(
                "deepspeed",
                hint="DeepSpeed required for distributed training"
            )
            self.setup_deepspeed()
```

## Related Pages

<!-- Links to related implementation documentation -->
