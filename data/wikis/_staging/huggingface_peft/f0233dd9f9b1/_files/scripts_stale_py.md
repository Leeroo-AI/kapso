# File: `scripts/stale.py`

**Category:** GitHub automation script

| Property | Value |
|----------|-------|
| Lines | 66 |
| Functions | `main` |
| Imports | datetime (datetime, timezone), github.Github, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Automated GitHub issue management script that marks inactive issues as stale and closes issues that remain stale without response. Adapted from the AllenNLP repository automation.

**Mechanism:**
- Connects to GitHub API using token from `GITHUB_TOKEN` environment variable
- Fetches all open issues from huggingface/peft repository
- For each open issue, applies two-stage staling process:

Stage 1 - Mark as stale (if all conditions met):
- Issue is at least 30 days old (created_at >= 30 days)
- No activity for 23+ days (updated_at > 23 days)
- Does NOT have any exempt labels (see LABELS_TO_EXEMPT)
- Action: Posts automated stale warning comment

Stage 2 - Close stale issues (if all conditions met):
- Last comment is from github-actions bot
- No activity for 7+ days after bot comment (updated_at > 7 days)
- Issue is at least 30 days old
- Does NOT have any exempt labels
- Action: Closes the issue

Exempt label categories:
- Good first/second/difficult issues (preserve for new contributors)
- Feature requests and new model requests (preserve product discussions)
- WIP (work in progress)
- "PRs welcome to address this" (documented known issues)

**Significance:** Essential repository hygiene automation that prevents issue tracker bloat while preserving important issues. The two-stage process (warn, then close) gives issue authors/interested parties time to respond before closure. Label exemptions ensure high-value issues are never auto-closed. Used in scheduled GitHub Actions workflows to maintain a manageable issue backlog.
