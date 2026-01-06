# Dependency Versions Table

| Metadata | Value |
|----------|-------|
| **Sources** | `src/transformers/dependency_versions_table.py` |
| **Domains** | Package Management, Dependency Specifications, Build Configuration |
| **Last Updated** | 2025-12-18 |

## Overview

The Dependency Versions Table is an auto-generated centralized registry that defines version constraints for all package dependencies used throughout the transformers library. It serves as the single source of truth for dependency requirements across development, testing, and production environments.

## Description

This module contains a single dictionary (`deps`) that maps package names to their version specification strings. The table is automatically generated from the `_deps` dictionary in `setup.py` using the `make deps_table_update` command, ensuring consistency between the installed package requirements and runtime version checks.

### Key Characteristics

- **Auto-Generated**: File is programmatically created from `setup.py` to prevent manual editing errors
- **Comprehensive Coverage**: Includes 74 packages spanning ML frameworks, utilities, testing tools, and optional features
- **Flexible Constraints**: Uses pip-style version specifiers (>=, <=, !=, <) for precise control
- **Categorical Organization**: Covers core dependencies, ML frameworks, tokenization, testing, development tools, and domain-specific packages

### Dependency Categories

1. **Core Runtime**: `filelock`, `huggingface-hub`, `numpy`, `packaging`, `pyyaml`, `regex`, `requests`, `safetensors`, `tokenizers`, `tqdm`
2. **ML Frameworks**: `torch`, `accelerate`, `deepspeed`, `peft`, `timm`, `diffusers`
3. **Tokenization**: `sentencepiece`, `tiktoken`, `fugashi`, `ipadic`, `sudachipy`, `rjieba`
4. **Computer Vision**: `Pillow`, `opencv-python`, `torchvision`, `av`
5. **Audio Processing**: `librosa`, `torchaudio`, `phonemizer`, `pyctcdecode`
6. **Testing**: `pytest`, `pytest-asyncio`, `pytest-timeout`, `pytest-xdist`, `parameterized`
7. **Development**: `ruff`, `hf-doc-builder`, `cookiecutter`, `GitPython`
8. **Cloud/APIs**: `sagemaker`, `openai`, `fastapi`, `starlette`, `uvicorn`

## Usage

### Accessing Version Constraints

```python
from transformers.dependency_versions_table import deps

# Get version constraint for a specific package
torch_version = deps["torch"]  # "torch>=2.2"
pillow_version = deps["Pillow"]  # "Pillow>=10.0.1,<=15.0"
```

### Using in Version Checks

```python
from transformers.dependency_versions_table import deps
from transformers.utils.versions import require_version

# Validate a dependency version
require_version(deps["accelerate"])
```

### Updating the Table

```bash
# Modify _deps in setup.py, then regenerate the table
make deps_table_update
```

## Code Reference

### Data Structure

```python
deps = {
    "Pillow": "Pillow>=10.0.1,<=15.0",
    "accelerate": "accelerate>=1.1.0",
    "av": "av",
    "beautifulsoup4": "beautifulsoup4",
    # ... 70 more entries
    "mistral-common[opencv]": "mistral-common[opencv]>=1.6.3",
}
```

### Version Constraint Formats

| Format | Example | Description |
|--------|---------|-------------|
| Exact match | `"cookiecutter==1.7.3"` | Requires specific version |
| Minimum version | `"torch>=2.2"` | Version 2.2 or higher |
| Range constraint | `"Pillow>=10.0.1,<=15.0"` | Between 10.0.1 and 15.0 |
| Exclusion | `"regex!=2019.12.17"` | Any version except specified |
| No constraint | `"av"` | Any version accepted |
| Complex constraint | `"rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1"` | Multiple exclusions |

### Generation Process

```bash
# Step 1: Modify _deps in setup.py
_deps = {
    "new-package": "new-package>=1.0.0",
    # ...
}

# Step 2: Run make command
make deps_table_update

# Result: dependency_versions_table.py is regenerated
```

## I/O Contract

### Input Specifications

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| N/A | - | - | This is a data module with no function inputs |

### Output Specifications

| Export | Type | Description |
|--------|------|-------------|
| `deps` | dict[str, str] | Dictionary mapping package names to version specifications |

### Data Characteristics

- **Keys**: Package names as strings (case-sensitive, may include extras like `[opencv]`)
- **Values**: Pip-compatible version specifier strings
- **Size**: 74 entries (as of current snapshot)
- **Format**: Python dictionary literal

## Usage Examples

### Example 1: Runtime Dependency Validation

```python
from transformers.dependency_versions_table import deps
from transformers.utils.versions import require_version

def validate_torch_installation():
    """Ensure PyTorch meets minimum version requirements."""
    try:
        require_version(deps["torch"])
        return True
    except Exception as e:
        print(f"PyTorch version error: {e}")
        return False
```

### Example 2: Conditional Feature Loading

```python
from transformers.dependency_versions_table import deps
from transformers.utils import is_torch_available
import importlib.metadata

def can_use_timm_features():
    """Check if timm is available with correct version."""
    if not is_torch_available():
        return False

    try:
        installed_version = importlib.metadata.version("timm")
        required = deps["timm"]  # "timm>=1.0.20"
        # Version comparison logic here
        return True
    except Exception:
        return False
```

### Example 3: Installation Script Generation

```python
from transformers.dependency_versions_table import deps

def generate_requirements_file(packages, output_file="requirements.txt"):
    """Generate requirements.txt from selected packages."""
    with open(output_file, "w") as f:
        for pkg in packages:
            if pkg in deps:
                f.write(f"{deps[pkg]}\n")
            else:
                print(f"Warning: {pkg} not found in deps table")

# Generate requirements for audio processing
generate_requirements_file(
    ["torch", "torchaudio", "librosa", "phonemizer"],
    "requirements-audio.txt"
)
```

### Example 4: Dependency Audit Tool

```python
from transformers.dependency_versions_table import deps
import importlib.metadata

def audit_installed_packages():
    """Check which dependencies are installed and their versions."""
    report = []
    for pkg_spec in deps.values():
        # Extract package name (before version constraints)
        pkg_name = pkg_spec.split(">=")[0].split("==")[0].split("!=")[0].split("<")[0]

        try:
            version = importlib.metadata.version(pkg_name)
            report.append(f"{pkg_name}: {version} (required: {pkg_spec})")
        except importlib.metadata.PackageNotFoundError:
            report.append(f"{pkg_name}: NOT INSTALLED (required: {pkg_spec})")

    return "\n".join(report)
```

### Example 5: Testing Environment Setup

```python
from transformers.dependency_versions_table import deps

def get_test_dependencies():
    """Extract all testing-related dependencies."""
    test_packages = [
        "pytest", "pytest-asyncio", "pytest-timeout",
        "pytest-xdist", "pytest-order", "pytest-rerunfailures",
        "pytest-rich", "parameterized"
    ]

    return {pkg: deps[pkg] for pkg in test_packages if pkg in deps}

# Install test dependencies
# pip install {" ".join(get_test_dependencies().values())}
```

### Example 6: Version Constraint Parsing

```python
from transformers.dependency_versions_table import deps
import re

def parse_version_constraint(pkg_name):
    """Parse version constraint into components."""
    constraint = deps.get(pkg_name)
    if not constraint:
        return None

    # Extract operators and versions
    pattern = r'(>=|<=|==|!=|<|>)(\d+\.\d+(?:\.\d+)?)'
    matches = re.findall(pattern, constraint)

    return {
        "package": pkg_name,
        "raw": constraint,
        "constraints": [(op, ver) for op, ver in matches]
    }

# Example usage
torch_info = parse_version_constraint("torch")
# Result: {'package': 'torch', 'raw': 'torch>=2.2',
#          'constraints': [('>=', '2.2')]}
```

## Related Pages

<!-- Links to related implementation documentation -->
