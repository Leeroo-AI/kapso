# huggingface_transformers_DeprecateModels

## Metadata

| Attribute | Value |
|-----------|-------|
| Source | `/utils/deprecate_models.py` |
| Repository | huggingface/transformers |
| Commit | f9f6619c2cf7 |
| Domains | model-lifecycle, automation, repository-management |
| Last Updated | 2025-12-18 |

## Overview

The `deprecate_models.py` script automates the complete workflow for deprecating machine learning models in the Hugging Face Transformers library. It handles all necessary file system operations, documentation updates, configuration changes, and repository restructuring required to move models from active development to maintenance-only status.

## Description

This implementation provides a comprehensive automation tool for the model deprecation process in the Transformers library. The script performs multiple coordinated operations:

1. **Documentation Updates**: Adds maintenance mode warnings to model documentation pages with version-specific reinstall instructions
2. **Code Relocation**: Moves model implementations from active directory to deprecated subdirectory with automatic import path updates
3. **Test Cleanup**: Removes model-specific test suites to reduce CI overhead
4. **Configuration Management**: Updates auto-configuration mappings and deprecation lists
5. **Import Restructuring**: Updates all import statements and __init__.py files to reflect new file locations
6. **Metadata Cleanup**: Removes copied-from statements and updates special case configurations

The script uses Git operations for safe file movements, validates model existence before operations, and provides detailed logging throughout the process. It integrates with PyPI to determine the last stable release version for documentation warnings.

**Key Design Features**:
- Batch processing support for multiple models simultaneously
- Automatic validation of model paths and documentation
- Safe Git operations with proper tracking
- Integration with existing import sorting utilities
- Comprehensive error handling with skip mechanisms

## Usage

### Command Line Interface

```bash
python utils/deprecate_models.py --models <model_name1> <model_name2> ...
```

### Arguments

- `--models`: Space-separated list of model names to deprecate (required)

### Example

```bash
# Deprecate a single model
python utils/deprecate_models.py --models bert

# Deprecate multiple models
python utils/deprecate_models.py --models bert distilbert roberta
```

## Code Reference

### Main Functions

#### `get_last_stable_minor_release() -> str`

Retrieves the last stable minor release version of transformers from PyPI.

**Returns**: Version string (e.g., "4.35.2")

**Implementation Details**:
- Queries PyPI JSON API for transformers package metadata
- Parses current version to determine previous minor version
- Filters releases by major.minor version pattern
- Returns highest patch version for that minor release

---

#### `build_tip_message(last_stable_release: str) -> str`

Constructs a markdown warning message for deprecated model documentation.

**Parameters**:
- `last_stable_release`: Version string from PyPI

**Returns**: Formatted markdown tip block with reinstall instructions

---

#### `insert_tip_to_model_doc(model_doc_path: str, tip_message: str) -> None`

Inserts deprecation warning into model documentation file.

**Parameters**:
- `model_doc_path`: Path to markdown documentation file
- `tip_message`: Formatted warning message to insert

**Implementation Details**:
- Locates first heading line (starting with "# ")
- Inserts tip message immediately after title
- Preserves all other content unchanged

---

#### `get_model_doc_path(model: str) -> tuple[str | None, str | None]`

Locates documentation file for a given model, trying multiple naming conventions.

**Parameters**:
- `model`: Model identifier

**Returns**: Tuple of (doc_path, model_name) or (None, None) if not found

**Implementation Details**:
- Tries original name, hyphenated version, and no-separator version
- Returns first matching path found

---

#### `extract_model_info(model: str) -> dict | None`

Collects all necessary paths and metadata for model deprecation.

**Parameters**:
- `model`: Model identifier

**Returns**: Dictionary containing model_doc_path, model_doc_name, and model_path, or None if validation fails

---

#### `update_relative_imports(filename: str, model: str) -> None`

Updates import statements for moved files to account for new directory depth.

**Parameters**:
- `filename`: File path to update
- `model`: Model name (used for context)

**Implementation Details**:
- Replaces `from ..` with `from ...` throughout file
- Accounts for deprecated/ subdirectory adding extra level

---

#### `remove_copied_from_statements(model: str) -> None`

Strips "# Copied from" comments from all model files.

**Parameters**:
- `model`: Model identifier

**Implementation Details**:
- Processes all non-pycache files in model directory
- Removes lines containing "# Copied from"
- Preserves all other content

---

#### `move_model_files_to_deprecated(model: str) -> None`

Relocates model implementation files to deprecated subdirectory.

**Parameters**:
- `model`: Model identifier

**Implementation Details**:
- Creates deprecated/{model} directory if needed
- Uses git mv for proper version control tracking
- Updates relative imports in moved files
- Skips __pycache__ directories

---

#### `delete_model_tests(model: str) -> None`

Removes test suite for deprecated model.

**Parameters**:
- `model`: Model identifier

**Implementation Details**:
- Checks for existence of tests/models/{model}
- Uses git rm for recursive deletion with tracking

---

#### `update_main_init_file(models: list[str]) -> None`

Updates main __init__.py to point to new deprecated model locations.

**Parameters**:
- `models`: List of model identifiers

**Implementation Details**:
- Replaces `models.{model}` with `models.deprecated.{model}`
- Updates both import statements and string references
- Runs custom import sorting utility to maintain consistency

---

#### `remove_model_references_from_file(filename: str, models: list[str], condition: Callable) -> None`

Generic utility for removing model references from configuration files.

**Parameters**:
- `filename`: Relative path from repository root
- `models`: List of model identifiers
- `condition`: Function(line, model) -> bool determining if line should be removed

---

#### `remove_model_config_classes_from_config_check(model_config_classes: list[str]) -> None`

Removes deprecated model configs from attribute checking utility.

**Parameters**:
- `model_config_classes`: List of config class names (e.g., ["BertConfig"])

**Implementation Details**:
- Parses SPECIAL_CASES_TO_ALLOW dictionary structure
- Removes matching entries with their preceding comments
- Handles both single-line and multi-line entries

---

#### `add_models_to_deprecated_models_in_config_auto(models: list[str]) -> None`

Adds models to DEPRECATED_MODELS list in auto-configuration.

**Parameters**:
- `models`: List of model identifiers

**Implementation Details**:
- Locates DEPRECATED_MODELS list in configuration_auto.py
- Adds new model names to list
- Sorts alphabetically for consistency

---

#### `deprecate_models(models: list[str]) -> None`

Main orchestration function that executes the complete deprecation workflow.

**Parameters**:
- `models`: List of model identifiers to deprecate

**Workflow**:
1. Extract and validate model information
2. Collect model config class names from CONFIG_MAPPING
3. Filter out models that fail validation
4. Remove config classes from check_config_attributes.py
5. Build deprecation tip message
6. For each model:
   - Add warning to documentation
   - Remove copied-from statements
   - Move files to deprecated directory
   - Delete test suite
7. Update main __init__.py imports
8. Remove references from models/__init__.py, slow_documentation_tests.txt, not_doctested.txt
9. Add to DEPRECATED_MODELS list in configuration_auto.py

## I/O Contract

### Inputs

| Input Type | Description | Format | Example |
|------------|-------------|--------|---------|
| Command Line Arguments | Model names to deprecate | Space-separated strings | `--models bert distilbert` |
| File System | Model source files | Python modules in src/transformers/models/{model}/ | modeling_bert.py, configuration_bert.py |
| File System | Model documentation | Markdown files in docs/source/en/model_doc/ | bert.md |
| File System | Model tests | Python test modules in tests/models/{model}/ | test_modeling_bert.py |
| Git Repository | Repository state | Git working directory | Current branch state |
| PyPI API | Release information | JSON API response | https://pypi.org/pypi/transformers/json |

### Outputs

| Output Type | Description | Format | Location |
|-------------|-------------|--------|----------|
| File Modifications | Updated documentation with deprecation warnings | Markdown with tip blocks | docs/source/en/model_doc/{model}.md |
| File Relocations | Model files moved to deprecated directory | Python modules | src/transformers/models/deprecated/{model}/ |
| File Deletions | Removed test suites | Git rm operations | tests/models/{model}/ removed |
| Configuration Updates | Modified import mappings | Python dictionaries/lists | src/transformers/__init__.py |
| Configuration Updates | Updated deprecation list | Python list | src/transformers/models/auto/configuration_auto.py |
| Console Output | Progress and validation messages | Text logs | stdout |
| Git Changes | Staged changes ready for commit | Git index modifications | Repository working directory |

### Side Effects

- Modifies multiple files across the repository
- Creates new directories (deprecated model subdirectories)
- Deletes test directories
- Changes Git index (staged changes)
- Updates import paths throughout codebase
- Modifies configuration files used by auto-classes

## Usage Examples

### Basic Deprecation

```python
# From command line
python utils/deprecate_models.py --models transfo-xl

# The script will:
# 1. Validate that transfo-xl exists in models/ and has documentation
# 2. Get last stable release from PyPI (e.g., "4.35.2")
# 3. Add warning to docs/source/en/model_doc/transfo-xl.md
# 4. Move src/transformers/models/transfo_xl/ to src/transformers/models/deprecated/transfo_xl/
# 5. Delete tests/models/transfo_xl/
# 6. Update all import paths and configuration files
```

### Batch Deprecation

```python
# Deprecate multiple models at once
python utils/deprecate_models.py --models transfo-xl xlnet ctrl

# Processes all three models in sequence
# Performs file operations per-model but updates shared configs once
```

### Using Functions Programmatically

```python
# Get last stable release version
from deprecate_models import get_last_stable_minor_release

last_version = get_last_stable_minor_release()
# Returns: "4.35.2"

# Build warning message
from deprecate_models import build_tip_message

warning = build_tip_message("4.35.2")
# Returns: Markdown tip block with install instructions

# Check model documentation path
from deprecate_models import get_model_doc_path

doc_path, doc_name = get_model_doc_path("bert_generation")
# Tries: bert_generation.md, bert-generation.md, bertgeneration.md
# Returns: (Path(...), "bert-generation") if found

# Validate model before deprecation
from deprecate_models import extract_model_info

info = extract_model_info("bert")
if info:
    print(f"Model path: {info['model_path']}")
    print(f"Doc path: {info['model_doc_path']}")
    print(f"Doc name: {info['model_doc_name']}")
```

### Custom Filtering

```python
# Remove specific model references with custom condition
from deprecate_models import remove_model_references_from_file

models = ["bert", "gpt2"]

# Remove lines containing model names in specific format
remove_model_references_from_file(
    "utils/some_config.txt",
    models,
    lambda line, model: f"/{model}/" in line
)
```

### Tip Message Insertion

```python
from deprecate_models import insert_tip_to_model_doc, build_tip_message

# Create tip message
tip = build_tip_message("4.35.2")

# Insert into documentation (finds first # heading and inserts after it)
insert_tip_to_model_doc(
    "/path/to/docs/source/en/model_doc/model.md",
    tip
)
```

## Related Pages

- [Related implementation pages will be listed here]
