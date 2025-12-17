# File: `utils/release.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 227 |
| Functions | `update_version_in_file`, `update_version_in_examples`, `global_version_update`, `remove_conversion_scripts`, `remove_internal_utils`, `get_version`, `pre_release_work`, `post_release_work` |
| Imports | argparse, os, packaging, pathlib, re |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automates version management for Transformers releases including version updates, security cleanup, and post-release transitions.

**Mechanism:** Uses regex patterns to update version strings in __init__.py, setup.py, example files, and UV script dependencies. For pre-release: increments version numbers (minor or patch), updates all files, and removes conversion scripts and internal utils for security. For post-release: transitions to next dev version. Supports both regular and patch releases with interactive version confirmation.

**Significance:** Core release infrastructure that ensures consistent versioning across the entire codebase and enforces security by removing potentially unsafe files (pickle, .bin converters) from release artifacts. Critical for maintaining release integrity and preventing security scanner issues in production deployments.
