# File: `utils/check_doc_toc.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 133 |
| Functions | `clean_model_doc_toc`, `check_model_doc` |
| Imports | argparse, collections, yaml |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Maintains documentation table of contents by removing duplicate entries and sorting model entries alphabetically within each modality section.

**Mechanism:** Parses docs/source/en/_toctree.yml YAML file, navigates to API > Models sections, identifies duplicate entries by comparing "local" paths, validates that multiple occurrences have consistent titles, removes duplicates, and sorts remaining entries alphabetically by title (case-insensitive).

**Significance:** Documentation infrastructure tool that keeps the docs navigation organized and free of duplicates, ensuring users can easily find models and that the documentation structure remains maintainable as new models are added.
