{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|0PandaDEV_awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Actions|https://docs.github.com/en/actions]]
* [[source::Doc|GitHub REST API|https://docs.github.com/en/rest]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Automation]], [[domain::GitHub_Actions]]
|-
! Last Updated
| [[last_updated::2026-01-08 14:00 GMT]]
|}

== Overview ==

Automated CI/CD workflow that fetches repository contributors from the GitHub API and updates the README "Backers" section with contributor avatars.

=== Description ===

This workflow runs daily via GitHub Actions cron schedule (or manual dispatch) to maintain an up-to-date list of repository contributors in the README. It fetches contributor data from the GitHub Contributors API, transforms avatar URLs through the weserv.nl image proxy for consistent circular styling, and regenerates the Backers section HTML. The workflow only commits changes when new contributors are detected, preventing unnecessary commits.

=== Usage ===

This workflow runs automatically on a daily schedule (midnight UTC). It can also be manually triggered via workflow_dispatch for immediate updates. The workflow requires a GitHub Personal Access Token (PAT) with repository access stored as a secret.

== Execution Steps ==

=== Step 1: Trigger Workflow ===
[[step::Principle:0PandaDEV_awesome-windows_Workflow_Trigger_Scheduling]]

The GitHub Actions workflow is triggered either by the daily cron schedule (at midnight UTC) or by manual workflow_dispatch. The workflow runs on ubuntu-latest with Python 3.x.

'''Trigger methods:'''
* Automatic: Cron schedule `0 0 * * *` (daily at midnight UTC)
* Manual: Repository Actions tab → "Update Contributors" → Run workflow

=== Step 2: Fetch Contributors from GitHub API ===
[[step::Principle:0PandaDEV_awesome-windows_GitHub_API_Integration]]

The Python script authenticates with the GitHub API using the PAT token and fetches the list of contributors for the repository. The actions-user bot is filtered out to only include human contributors.

'''API endpoint:'''
* GET `/repos/{owner}/{repo}/contributors`
* Returns: Array of contributor objects with login, avatar_url, contributions count

=== Step 3: Check for Changes ===
[[step::Principle:0PandaDEV_awesome-windows_Change_Detection]]

Before regenerating the Backers section, the script checks if any new contributors exist by scanning the current README content for GitHub profile URLs. This prevents unnecessary commits when the contributor list hasn't changed.

'''Detection logic:'''
* For each contributor, check if `https://github.com/{username}` exists in README
* If any contributor URL is missing, changes are required
* If all URLs exist, skip the update

=== Step 4: Generate Contributor HTML ===
[[step::Principle:0PandaDEV_awesome-windows_Avatar_HTML_Generation]]

For each contributor, generate an HTML anchor tag containing a circular avatar image. Avatar URLs are proxied through weserv.nl image service for consistent styling with circular mask and 7-day cache.

'''Generated HTML format:'''
* Avatar URL: `https://images.weserv.nl/?url={avatar_url}&fit=cover&mask=circle&maxage=7d`
* Image dimensions: 60x60 pixels
* Link: `https://github.com/{username}`

=== Step 5: Update README Content ===
[[step::Principle:0PandaDEV_awesome-windows_README_Content_Update]]

The script uses regex to locate and replace the existing Backers section in README.md while preserving the surrounding content. The section spans from `## Backers` to just before the `[oss]:` definition references.

'''Regex pattern:'''
* Match: `## Backers` section until next markdown references
* Replace with: New Backers header, contributor avatars, and support links

=== Step 6: Commit and Push Changes ===
[[step::Principle:0PandaDEV_awesome-windows_Git_Commit_Automation]]

If updates were made, the workflow configures git with the maintainer's identity, stages the README.md changes, commits with message "Update contributors", and pushes to the main branch.

'''Commit conditions:'''
* Only commits if the Python script output contains "Contributors updated"
* Skips commit if output is "No changes in contributors"

== Execution Diagram ==

{{#mermaid:graph TD
    A[Trigger Workflow] --> B[Fetch Contributors from GitHub API]
    B --> C{New Contributors?}
    C -->|No| D[Skip Update]
    C -->|Yes| E[Generate Contributor HTML]
    E --> F[Update README Content]
    F --> G[Commit and Push Changes]
}}

== Related Pages ==

=== Steps ===
* [[step::Principle:0PandaDEV_awesome-windows_Workflow_Trigger_Scheduling]]
* [[step::Principle:0PandaDEV_awesome-windows_GitHub_API_Integration]]
* [[step::Principle:0PandaDEV_awesome-windows_Change_Detection]]
* [[step::Principle:0PandaDEV_awesome-windows_Avatar_HTML_Generation]]
* [[step::Principle:0PandaDEV_awesome-windows_README_Content_Update]]
* [[step::Principle:0PandaDEV_awesome-windows_Git_Commit_Automation]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:0PandaDEV_awesome-windows_Idempotent_CI_CD_Design]]
* [[uses_heuristic::Heuristic:0PandaDEV_awesome-windows_Weserv_Image_Proxy_Pattern]]
* [[uses_heuristic::Heuristic:0PandaDEV_awesome-windows_Git_Identity_In_CI]]
