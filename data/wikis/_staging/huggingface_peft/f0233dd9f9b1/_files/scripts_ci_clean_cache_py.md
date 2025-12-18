# File: `scripts/ci_clean_cache.py`

**Category:** CI utility script

| Property | Value |
|----------|-------|
| Lines | 68 |
| Functions | `find_old_revisions`, `delete_old_revisions` |
| Imports | datetime, huggingface_hub.scan_cache_dir, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** CI/CD utility script for cleaning old cached models from the Hugging Face Hub cache directory to free up disk space on CI runners.

**Mechanism:**
- Uses `huggingface_hub.scan_cache_dir()` to scan the local HF cache
- `find_old_revisions()`: Identifies cached model revisions older than a specified age (default 30 days)
  - Compares current time with last access time recorded in cache metadata
  - Returns list of commit hashes for old revisions
- `delete_old_revisions()`: Handles deletion of old cache entries
  - Calculates expected freed space
  - Prints candidates and expected space savings
  - Only deletes if `-d/--delete` flag is passed
- Exit codes:
  - 0: Candidates found (successfully cleaned or would clean)
  - 1: No candidates found

Command-line arguments:
- `-a/--max-age`: Maximum age in days (default: 30)
- `-d/--delete`: Actually delete files (dry-run by default)

**Significance:** Essential maintenance utility for CI/CD pipelines. Prevents disk space issues by automatically cleaning up old cached models that haven't been accessed recently. The dry-run default ensures safe testing before actual deletion. Used in GitHub Actions to maintain clean CI environments.
