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

**Purpose:** GitHub Actions script that automatically updates the README's "Backers" section with contributor avatars and profile links.

**Mechanism:**
1. `get_contributors()` - Fetches repository contributors from GitHub API using `GITHUB_PAT` token, filtering out 'actions-user'
2. `has_contributors_changed()` - Compares current README content against contributor list to detect if any new contributors need to be added
3. `update_readme()` - Regenerates the "Backers" section with HTML `<a>` and `<img>` tags for each contributor's avatar (using weserv.nl image proxy for circular avatars), then uses regex to replace the section between "## Backers" and the reference links

**Significance:** CI/CD automation utility - runs as part of GitHub Actions workflow to keep the awesome-list's contributor acknowledgment section automatically updated. This is a maintenance convenience script, not core to the repository's purpose of curating Windows software recommendations.
