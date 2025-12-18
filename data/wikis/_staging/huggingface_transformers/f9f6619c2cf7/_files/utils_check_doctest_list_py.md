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

**Purpose:** Maintains clean and sorted lists of files that are excluded from doctest execution or run as slow documentation tests.

**Mechanism:** The script processes two files (`not_doctested.txt` and `slow_documentation_tests.txt`) that contain paths to documentation or code files. For each file, it validates that all listed paths actually exist in the repository (raising an error for non-existent paths) and checks that entries are in alphabetical order. If the `--fix_and_overwrite` flag is provided, it automatically sorts the lists; otherwise, it fails if the order is incorrect, prompting users to run the fix command.

**Significance:** These lists control doctest behavior - `not_doctested.txt` excludes files from doctest runs (useful for examples that require special setup or external resources), while `slow_documentation_tests.txt` marks tests that take significant time to run. Keeping these lists sorted makes them easier to review in PRs, reduces merge conflicts, and prevents accidental duplicates. The validation ensures the lists don't reference deleted files, maintaining test infrastructure integrity.
