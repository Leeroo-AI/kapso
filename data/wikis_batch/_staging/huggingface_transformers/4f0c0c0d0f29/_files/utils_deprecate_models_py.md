# File: `utils/deprecate_models.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 377 |
| Functions | `get_last_stable_minor_release`, `build_tip_message`, `insert_tip_to_model_doc`, `get_model_doc_path`, `extract_model_info`, `update_relative_imports`, `remove_copied_from_statements`, `move_model_files_to_deprecated`, `... +7 more` |
| Imports | argparse, collections, custom_init_isort, git, os, packaging, pathlib, requests, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automates the multi-step process of deprecating model architectures in the transformers library.

**Mechanism:** Fetches the last stable release version from PyPI, generates deprecation warning tips for documentation, moves model files from `models/{model}/` to `models/deprecated/{model}/` using git commands, updates relative imports (from `..` to `...`), removes #Copied from comments, deletes test directories, updates __init__.py to point to deprecated paths, removes references from various check files, and adds models to DEPRECATED_MODELS list in configuration_auto.py. Uses GitPython for version-controlled moves.

**Significance:** Maintenance automation tool that standardizes the complex deprecation process, ensuring deprecated models remain accessible for backward compatibility while clearly signaling maintenance-only status to users and removing them from active testing/checking cycles.
