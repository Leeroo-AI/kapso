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

**Purpose:** Automates GitHub issue management by marking stale issues and closing inactive ones based on time-based heuristics.

**Mechanism:** Connects to the huggingface/transformers GitHub repository via GitHub API using GITHUB_TOKEN environment variable. Iterates through open issues, checking last update time and comment history. Issues inactive for 23+ days (and 30+ days old) receive a stale warning comment from github-actions bot. Issues with bot comment and 7+ days additional inactivity are automatically closed. Exempts issues with specific labels like "good first issue", "feature request", "new model", or "wip".

**Significance:** Essential repository maintenance automation reducing manual triage burden by managing issue lifecycle. Helps keep issue tracker focused on actionable items while providing fair warning before closure. Adapted from AllenNLP's proven approach, demonstrates best practices for open-source project hygiene and community engagement while respecting ongoing work through label-based exemptions.
