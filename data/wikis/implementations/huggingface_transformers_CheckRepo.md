# check_repo.py - Repository Consistency Checker

| Metadata | Value |
|----------|-------|
| **Type** | Implementation |
| **Source File** | `utils/check_repo.py` |
| **Repository** | huggingface/transformers |
| **Domains** | Quality Assurance, Repository Management, Model Validation |
| **Last Updated** | 2025-12-18 |
| **Lines of Code** | 1,309 |

---

## Overview

The `check_repo.py` script is a comprehensive automated repository health and consistency checker for the Hugging Face Transformers library. It performs multiple validation checks to ensure that the repository maintains consistency across models, tests, documentation, and auto-mappings. This utility is integrated into the CI/CD pipeline via the `make repo-consistency` command.

---

## Description

### Purpose

This script serves as a critical quality assurance tool that validates:
- All models are properly defined in module `__init__` files
- All models are included in the main library initialization
- All models have corresponding test files with proper test coverage
- All public objects are documented
- All models are registered in at least one auto class
- All auto mappings are correctly defined and importable
- Deprecated models list is up-to-date
- Model forward methods accept `**kwargs`

### Key Features

1. **Model Discovery**: Automatically scans the models directory and verifies proper module structure
2. **Test Coverage Validation**: Ensures every model class has corresponding tests in the test suite
3. **Documentation Verification**: Checks that all public API objects have documentation pages
4. **Auto-Class Registration**: Validates that models are properly registered in AUTO_MODEL mappings
5. **Consistency Checks**: Verifies naming conventions, import structures, and configuration mappings
6. **Deprecation Management**: Ensures deprecated models are properly tracked and documented

### Architecture

The script is organized into several validation categories:
- **Model List Checks**: Verify models match directory structure
- **Init Checks**: Ensure proper initialization in `__init__.py` files
- **Testing Checks**: Validate test coverage and decorator ordering
- **Documentation Checks**: Confirm all public objects are documented
- **Auto-Configuration Checks**: Verify auto-class registrations
- **Special Checks**: Model-specific validations (e.g., `**kwargs` in forward methods)

---

## Code Reference

### Main Entry Point

```python
def check_repo_quality():
    """Check all models are tested and documented."""
    print("Repository-wide checks:")
    print("    - checking all models are included.")
    check_model_list()
    print("    - checking all models are public.")
    check_models_are_in_init()
    print("    - checking all models have tests.")
    check_all_decorator_order()
    check_all_models_are_tested()
    print("    - checking all objects have documentation.")
    check_all_objects_are_documented()
    print("    - checking all models are in at least one auto class.")
    check_all_models_are_auto_configured()
    print("    - checking all names in auto name mappings are defined.")
    check_all_auto_object_names_being_defined()
    print("    - checking all keys in auto name mappings are defined in `CONFIG_MAPPING_NAMES`.")
    check_all_auto_mapping_names_in_config_mapping_names()
    print("    - checking all auto mappings could be imported.")
    check_all_auto_mappings_importable()
    print("    - checking the DEPRECATED_MODELS constant is up to date.")
    check_deprecated_constant_is_up_to_date()
    print("    - checking all models accept **kwargs in their call.")
    check_models_have_kwargs()
```

### Key Functions

#### Model Validation

```python
def get_model_modules() -> list[str]:
    """Get all the model modules inside the transformers library (except deprecated models)."""

def get_models(module: types.ModuleType, include_pretrained: bool = False) -> list[tuple[str, type]]:
    """
    Get the objects in a module that are models.

    Args:
        module: The module from which we are extracting models.
        include_pretrained: Whether or not to include the PreTrainedModel subclass.

    Returns:
        List of models as tuples (class name, actual class).
    """

def check_model_list():
    """Checks the model listed as subfolders of `models` match the models available in `transformers.models`."""

def check_models_are_in_init():
    """Checks all models defined in the library are in the main init."""
```

#### Test Coverage Validation

```python
def find_tested_models(test_file: str) -> set[str]:
    """
    Parse the content of test_file to detect what's in `all_model_classes`.

    Args:
        test_file: The path to the test file to check

    Returns:
        The set of models tested in that file.
    """

def check_models_are_tested(module: types.ModuleType, test_file: str) -> list[str]:
    """Check models defined in a module are all tested in a given file.

    Args:
        module: The module in which we get the models.
        test_file: The path to the file where the module is tested.

    Returns:
        The list of error messages corresponding to models not tested.
    """

def check_all_models_are_tested():
    """Check all models are properly tested."""
```

#### Documentation Validation

```python
def find_all_documented_objects() -> list[str]:
    """
    Parse the content of all doc files to detect which classes and functions it documents.

    Returns:
        The list of all object names being documented.
        A dictionary mapping the object name to its documented methods
    """

def check_all_objects_are_documented():
    """Check all models are properly documented."""

def check_model_type_doc_match():
    """Check all doc pages have a corresponding model type."""
```

#### Auto-Configuration Validation

```python
def get_all_auto_configured_models() -> list[str]:
    """Return the list of all models in at least one auto class."""

def check_models_are_auto_configured(module: types.ModuleType, all_auto_models: list[str]) -> list[str]:
    """
    Check models defined in module are each in an auto class.

    Args:
        module: The module in which we get the models.
        all_auto_models: The list of all models in an auto class.

    Returns:
        The list of error messages corresponding to models not tested.
    """

def check_all_auto_object_names_being_defined():
    """Check all names defined in auto (name) mappings exist in the library."""

def check_all_auto_mappings_importable():
    """Check all auto mappings can be imported."""
```

### Important Constants

```python
# Paths (relative to repository root)
PATH_TO_TRANSFORMERS = "src/transformers"
PATH_TO_TESTS = "tests"
PATH_TO_DOC = "docs/source/en"

# Private models that should not be in the main init
PRIVATE_MODELS = [
    "AltRobertaModel",
    "DPRSpanPredictor",
    # ... (110+ entries)
]

# Models excluded from testing requirements
IGNORE_NON_TESTED = [
    # Building blocks of larger models
    "RecurrentGemmaModel",
    "FuyuForCausalLM",
    # ... (200+ entries)
]

# Models excluded from auto-configuration requirements
IGNORE_NON_AUTO_CONFIGURED = [
    "AlignTextModel",
    "AlignVisionModel",
    # ... (400+ entries)
]
```

---

## I/O Contract

### Input Specifications

| Input Type | Description | Format | Required |
|------------|-------------|--------|----------|
| Repository Structure | Transformers repository file system | Directory tree | Yes |
| Model Modules | Python modules in `src/transformers/models/` | Python modules | Yes |
| Test Files | Test files in `tests/models/` | Python test files | Yes |
| Documentation | Markdown files in `docs/source/en/` | Markdown files | Yes |

### Output Specifications

| Output Type | Description | Format | Conditions |
|-------------|-------------|--------|------------|
| Console Output | Progress messages for each check | Text | Always |
| Success | Silent exit with code 0 | Exit code | All checks pass |
| Failure | Exception with detailed error messages | Exception | Any check fails |
| Error Reports | List of failures with context | Text list | Validation errors found |

### Exit Codes

- **0**: All checks passed successfully
- **1**: One or more checks failed (raises Exception)

---

## Usage Examples

### Basic Usage

Run all consistency checks from the repository root:

```bash
python utils/check_repo.py
```

### Integration with Makefile

The script is integrated into the repository's quality checks:

```bash
make repo-consistency
```

### Example Output (Success)

```
Repository-wide checks:
    - checking all models are included.
    - checking all models are public.
    - checking all models have tests.
    - checking all objects have documentation.
    - checking all models are in at least one auto class.
    - checking all names in auto name mappings are defined.
    - checking all keys in auto name mappings are defined in `CONFIG_MAPPING_NAMES`.
    - checking all auto mappings could be imported.
    - checking the DEPRECATED_MODELS constant is up to date.
    - checking all models accept **kwargs in their call.
```

### Example Output (Failure)

```
Repository-wide checks:
    - checking all models are included.
    - checking all models are public.
    - checking all models have tests.
Exception: There were 2 failures:
BertNewModel is defined in transformers.models.bert.modeling_bert but is not tested in tests/models/bert/test_modeling_bert.py. Add it to the `all_model_classes` in that file or, if it inherits from `CausalLMModelTester`, fill in the model class attributes. If common tests should not applied to that model, add its name to `IGNORE_NON_TESTED`in the file `utils/check_repo.py`.
GPT2NewModel is defined in transformers.models.gpt2.modeling_gpt2 but is not present in any of the auto mapping. If that is intended behavior, add its name to `IGNORE_NON_AUTO_CONFIGURED` in the file `utils/check_repo.py`.
```

### Adding Exceptions

If a model should be excluded from certain checks, add it to the appropriate constant:

```python
# Example: Add a private model
PRIVATE_MODELS = [
    "AltRobertaModel",
    "MyNewPrivateModel",  # Add your model here
]

# Example: Add a model that doesn't need testing
IGNORE_NON_TESTED = [
    "RecurrentGemmaModel",
    "MyBuildingBlockModel",  # Building part of bigger model
]
```

### Custom Checks for New Models

When adding a new model architecture, ensure:

1. **Model Module**: Create directory in `src/transformers/models/your_model/`
2. **Init File**: Add model classes to `src/transformers/models/your_model/__init__.py`
3. **Main Init**: Export in `src/transformers/__init__.py`
4. **Test File**: Create `tests/models/your_model/test_modeling_your_model.py`
5. **Auto Mapping**: Register in appropriate AUTO_MODEL mappings
6. **Documentation**: Add markdown file in `docs/source/en/model_doc/your_model.md`

---

## Related Pages

<!-- Links to related documentation pages -->

---

## Implementation Notes

### AST Parsing

The script uses Python's `ast` module to parse model files and verify that forward methods accept `**kwargs`:

```python
def check_models_have_kwargs():
    """Checks that all model classes accept **kwargs in their forward pass."""
    for model_dir in models_dir.iterdir():
        with open(modeling_file, "r") as f:
            tree = ast.parse(f.read())
        # Check for **kwargs in forward method
        if forward_method.args.kwarg is None:
            failing_classes.append(class_name)
```

### Regular Expressions

The script uses regex patterns to parse test files and documentation:

```python
# Find all_model_classes in test files
all_models = re.findall(r"all_model_classes\s+=\s+\(\s*\(([^\)]*)\)", content)

# Find documented objects
raw_doc_objs = re.findall(r"\[\[autodoc\]\]\s+(\S+)\s+", content)
```

### Backend Validation

The script validates that required backends (PyTorch, TensorFlow, Flax) are installed for complete checks:

```python
def check_missing_backends():
    """Checks if all backends are installed."""
    missing_backends = []
    if not is_torch_available():
        missing_backends.append("PyTorch")
    # Warns if backends are missing
```

### Error Reporting

Failures are collected and reported at the end of each check:

```python
failures = []
for module in modules:
    new_failures = check_models_are_tested(module, test_file)
    if new_failures is not None:
        failures += new_failures
if len(failures) > 0:
    raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))
```
