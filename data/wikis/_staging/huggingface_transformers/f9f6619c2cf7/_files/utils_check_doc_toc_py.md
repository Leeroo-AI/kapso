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

**Purpose:** Ensures the documentation table of contents (_toctree.yml) is properly organized with sorted model entries and no duplicates.

**Mechanism:** The script loads the YAML table of contents file, navigates to the Models section under API documentation, and processes each modality subsection (text, vision, audio, etc.). For each modality, it detects duplicates by tracking local paths, validates that duplicate entries have consistent titles, removes redundant entries, and sorts all model entries alphabetically by title. The cleaned structure is written back to the YAML file if the `--fix_and_overwrite` flag is provided, otherwise the script raises an error if inconsistencies are found.

**Significance:** A clean, sorted table of contents is essential for documentation usability - users can quickly find models alphabetically, and duplicates cause confusion. This automated check prevents documentation merge conflicts and ensures consistency as new models are added. It's part of the `make style` (auto-fix) and `make quality` (validation) workflows, maintaining documentation quality standards across the repository.
