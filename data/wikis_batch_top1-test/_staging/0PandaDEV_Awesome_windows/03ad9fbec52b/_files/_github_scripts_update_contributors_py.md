# File: `.github/scripts/update_contributors.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 50 |
| Functions | `get_contributors`, `has_contributors_changed`, `update_readme` |
| Imports | os, re, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically updates the "Backers" section of README.md with contributor avatars from GitHub.

**Mechanism:**
1. `get_contributors()` - Fetches contributor list from GitHub API using `GITHUB_PAT` token and `GITHUB_REPOSITORY` environment variables, filtering out the `actions-user` bot
2. `has_contributors_changed()` - Checks if any new contributors need to be added by scanning README.md for missing GitHub profile URLs
3. `update_readme()` - Regenerates the "Backers" section with contributor avatars (using weserv.nl image proxy for circular cropping) and replaces the existing section using regex pattern matching

**Significance:** CI/CD automation script - runs as part of a GitHub Actions workflow to keep the contributor credits in the README up-to-date without manual intervention. Essential for maintaining the community acknowledgment section of this awesome-list repository.
