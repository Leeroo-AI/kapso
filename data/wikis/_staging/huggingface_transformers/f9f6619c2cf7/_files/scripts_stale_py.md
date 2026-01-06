# File: `scripts/stale.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 76 |
| Functions | `main` |
| Imports | datetime, github, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automates the management of stale issues in the huggingface/transformers GitHub repository by marking inactive issues and closing those that remain inactive after warnings.

**Mechanism:** Uses the PyGithub library to authenticate with the GitHub API (via GITHUB_TOKEN environment variable), retrieves all open issues from the transformers repository, and applies two-stage staleness logic: (1) If an issue hasn't been updated for 23 days, is at least 30 days old, and doesn't have exempt labels (like "good first issue", "feature request", "new model", "wip"), it adds a stale warning comment; (2) If an issue's last comment is from github-actions[bot] and hasn't been updated for 7 more days (30+ days total age), it closes the issue. The script includes exception handling for GitHub API errors and respects a curated list of label exemptions.

**Significance:** Maintains repository hygiene for one of the largest machine learning libraries on GitHub, preventing issue tracker bloat from abandoned or resolved-but-not-closed issues. This automation is crucial for a project with thousands of contributors and issues, helping maintainers focus on active issues while providing contributors a clear timeline and opportunity to re-engage before closure. The approach balances community engagement (warning period) with maintainer efficiency (automated closure).
