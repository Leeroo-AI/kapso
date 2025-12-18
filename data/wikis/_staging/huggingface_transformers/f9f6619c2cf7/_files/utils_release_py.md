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

**Purpose:** Orchestrates the release process by updating version strings throughout the codebase, managing pre-release preparations including security cleanup, and handling post-release version bumps for development.

**Mechanism:** The script uses regex patterns defined in REPLACE_PATTERNS to locate and update version strings across __init__.py, setup.py, example files, and UV script dependencies, derives appropriate version numbers from the current version with support for minor releases, patch releases, and dev versions, removes conversion scripts and internal utilities for security reasons before releases, and provides interactive prompts for version confirmation while automating the mechanical work of updating dozens of files consistently.

**Significance:** This is a mission-critical tool for release management that ensures version consistency across the entire repository while preventing security vulnerabilities from insecure file handling code in release artifacts. It standardizes the release workflow, reduces human error in version updates, and maintains the boundary between development and production code, supporting both regular minor releases and emergency patch releases.
