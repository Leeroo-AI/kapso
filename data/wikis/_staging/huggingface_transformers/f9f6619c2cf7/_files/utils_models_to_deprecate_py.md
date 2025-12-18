# File: `utils/models_to_deprecate.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 335 |
| Classes | `HubModelLister` |
| Functions | `get_list_of_repo_model_paths`, `get_list_of_models_to_deprecate` |
| Imports | argparse, collections, datetime, git, glob, huggingface_hub, json, os, pathlib, tqdm, ... +1 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Identifies model architectures that are candidates for deprecation based on low download counts and age criteria.

**Mechanism:** Scans the repository for modeling files, extracts their first commit dates using git history, then queries the Hugging Face Hub API to aggregate download counts across all models using specific tags. Filters models that are at least one year old with fewer than 5,000 downloads (configurable threshold). Handles model tag variations and extra tags through mapping dictionaries to ensure accurate download counting.

**Significance:** Maintenance tool that helps the library stay lean by identifying underutilized models that burden testing and maintenance resources. The data-driven approach ensures deprecation decisions are based on actual usage metrics rather than subjective assessment, with built-in safeguards to verify models aren't used as modules in other architectures.
