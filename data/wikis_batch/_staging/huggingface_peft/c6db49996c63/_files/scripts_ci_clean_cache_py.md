# File: `scripts/ci_clean_cache.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 67 |
| Functions | `find_old_revisions`, `delete_old_revisions` |
| Imports | datetime, huggingface_hub, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** CI maintenance utility that cleans old cached model files from HuggingFace Hub cache to prevent disk space exhaustion during continuous integration testing.

**Mechanism:** Uses huggingface_hub.scan_cache_dir() to inventory cached repositories. The find_old_revisions() function compares each revision's last access timestamp against the current time to identify candidates exceeding max_age_days (default 30). The delete_old_revisions() function calculates expected space savings and optionally executes deletion when the -d flag is provided. Returns exit code 0 when candidates are found, 1 otherwise.

**Significance:** Critical CI infrastructure utility that maintains build environment health by preventing cache bloat. In CI environments that frequently download large language models, the cache can quickly consume hundreds of gigabytes. This script enables automated cleanup policies while providing dry-run mode for safety.
