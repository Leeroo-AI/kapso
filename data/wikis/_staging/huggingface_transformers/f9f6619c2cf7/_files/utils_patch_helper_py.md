# File: `utils/patch_helper.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 156 |
| Functions | `get_release_branch_name`, `checkout_branch`, `get_prs_by_label`, `get_commit_timestamp`, `cherry_pick_commit`, `commit_in_history`, `main` |
| Imports | json, subprocess, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automates the preparation of patch releases by identifying, sorting, and cherry-picking commits from pull requests labeled "for patch" in chronological order onto release branches.

**Mechanism:** The script derives the current release branch name from the transformers version, uses the GitHub CLI to fetch PRs with the "for patch" label, retrieves commit timestamps via git commands, sorts commits chronologically to avoid merge conflicts, and automatically cherry-picks each commit onto the release branch while checking if commits are already in history. It validates that commits exist in main before attempting cherry-picks and provides clear warnings for PRs that haven't been merged yet.

**Significance:** This tool streamlines the patch release workflow by eliminating manual commit identification and cherry-picking, reducing the risk of missing commits or encountering merge conflicts. It's essential for maintaining stable release branches and ensuring critical fixes are properly backported from main to release versions.
