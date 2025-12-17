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

**Purpose:** Identifies model candidates for deprecation based on low download counts and commit age criteria.

**Mechanism:** Uses Git history to find models added over a year ago, queries HuggingFace Hub API via `HubModelLister` to aggregate download counts across model tags (handling name variations via `MODEL_FOLDER_NAME_TO_TAG_MAPPING` and `EXTRA_TAGS_MAPPING`), filters models below 5,000 downloads threshold, and validates against `MODEL_NAMES_MAPPING`. Supports caching via `models_info.json` for iterative analysis.

**Significance:** Maintenance tool for library sustainability, helping maintainers make data-driven decisions about model deprecation to reduce maintenance burden while ensuring widely-used models are preserved. Crucial for managing the ever-growing model catalog.
