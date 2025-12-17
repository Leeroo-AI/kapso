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

**Purpose:** Automates cherry-picking commits for release branch patch management using GitHub PR labels.

**Mechanism:** Derives release branch name from transformers version (e.g., `v4.41-release`), uses `gh` CLI to fetch PRs labeled "for patch", sorts commits chronologically by timestamp (`get_commit_timestamp`), verifies commit ancestry via `git merge-base`, and automatically cherry-picks commits not already in the release branch history. Handles failures gracefully with manual intervention prompts.

**Significance:** Streamlines the release patching workflow by automating the tedious and error-prone process of identifying and applying fixes to release branches. Ensures patches are applied in chronological order to minimize merge conflicts. Essential for maintaining stable releases while the main branch continues rapid development.
