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

**Purpose:** Automates the deprecation process for transformer model architectures. Moves models to maintenance mode by updating documentation, relocating code, and modifying imports to indicate the models are no longer actively developed.

**Mechanism:** Takes a list of model names, fetches the last stable PyPI release version, adds warning tips to model documentation pages, removes "Copied from" statements, moves model files from src/transformers/models/{model} to src/transformers/models/deprecated/{model}, updates all import paths, deletes test files, removes from config checks, and adds to DEPRECATED_MODELS list. Uses git operations to track file movements.

**Significance:** Critical maintenance tool that ensures smooth lifecycle management of models. As the library grows, older models need deprecation without breaking existing users. This automates a complex multi-step process involving documentation, code organization, imports, and CI configuration changes while preserving backward compatibility.
