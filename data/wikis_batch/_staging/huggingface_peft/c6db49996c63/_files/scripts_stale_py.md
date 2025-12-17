# File: `scripts/stale.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 65 |
| Functions | `main` |
| Imports | datetime, github, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automated GitHub issue management bot that identifies and closes stale issues to keep the repository issue tracker clean and focused on active problems.

**Mechanism:** Uses PyGithub to fetch all open issues from huggingface/peft. For each issue, checks if it's been inactive for 23+ days (since last update) and at least 30 days old (since creation). Issues matching these criteria and not marked with exempt labels (like "good first issue", "feature request", "wip", etc.) receive a warning comment from the github-actions bot. If an issue remains inactive for 7 more days after the warning (30+ days total inactivity), it's automatically closed.

**Significance:** Repository maintenance automation that prevents issue tracker bloat while being respectful of important issues. By exempting feature requests and beginner-friendly issues, it focuses on closing abandoned bug reports and questions that have been resolved through inactivity. Helps maintainers focus attention on active issues rather than archaeological digs through old, potentially irrelevant reports.
