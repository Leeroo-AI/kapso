# File: `utils/extract_pr_number_from_circleci.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 31 |
| Imports | os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extracts pull request numbers from CircleCI environment variables for GitHub Actions integration.

**Mechanism:** Checks two CircleCI environment variables: first tries `CIRCLE_PULL_REQUEST` (extracts number from URL like `https://github.com/org/repo/pull/123`), then falls back to `CIRCLE_BRANCH` (extracts from branch names like `pull/123/head`). Prints the extracted PR number to stdout for capture by GitHub Actions workflows.

**Significance:** Bridge utility used by `.github/workflows/trigger_circleci.yml` to pass PR context from GitHub Actions to CircleCI jobs, enabling proper PR-aware CI execution and status reporting across both CI systems.
