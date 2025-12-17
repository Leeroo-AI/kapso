# File: `utils/check_doctest_list.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 86 |
| Functions | `clean_doctest_list` |
| Imports | argparse, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Maintains two doctest configuration files (not_doctested.txt and slow_documentation_tests.txt) by validating entries exist and sorting them alphabetically.

**Mechanism:** Reads each doctest list file line by line, checks that referenced paths exist as files or directories in the repo, identifies and reports non-existent paths, sorts entries alphabetically, and optionally overwrites the files with the sorted list.

**Significance:** CI infrastructure tool that prevents broken references in doctest exclusion lists, ensuring the documentation testing system remains functional and organized as files are added/removed/renamed.
