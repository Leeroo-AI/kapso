# Implementation: Get_contributors

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub REST API - Contributors|https://docs.github.com/en/rest/repos/repos#list-repository-contributors]]
* [[source::Doc|Requests Library|https://requests.readthedocs.io]]
|-
! Domains
| [[domain::GitHub_API]], [[domain::CI_CD]], [[domain::Automation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Concrete tool for fetching repository contributor data from the GitHub REST API with authentication.

=== Description ===

The `get_contributors()` function retrieves a list of all contributors to the repository using the GitHub REST API. It authenticates via a Personal Access Token (PAT) stored in environment variables, fetches contributor data including login names and avatar URLs, and filters out the `actions-user` bot account to return only human contributors.

=== Usage ===

Use this function when you need to programmatically retrieve the list of repository contributors for display, acknowledgment, or automation purposes. This is the first step in the contributor update automation workflow.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/scripts/update_contributors.py
* '''Lines:''' L6-11

=== Signature ===
<syntaxhighlight lang="python">
def get_contributors() -> list[dict]:
    """
    Fetch repository contributors from GitHub API.

    Returns:
        list[dict]: List of contributor dictionaries, each containing:
            - login (str): GitHub username
            - avatar_url (str): URL to contributor's avatar image
            - contributions (int): Number of contributions

        Note: Excludes 'actions-user' bot from results.

    Environment Variables Required:
        GITHUB_PAT: Personal Access Token for authentication
        GITHUB_REPOSITORY: Repository in 'owner/repo' format
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from update_contributors import get_contributors
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| GITHUB_PAT || str (env) || Yes || GitHub Personal Access Token for API authentication
|-
| GITHUB_REPOSITORY || str (env) || Yes || Repository identifier in 'owner/repo' format
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return value || list[dict] || List of contributor dicts with `login`, `avatar_url`, `contributions` keys
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
import os

# Set environment variables (typically done by GitHub Actions)
os.environ['GITHUB_PAT'] = 'ghp_xxxxxxxxxxxx'
os.environ['GITHUB_REPOSITORY'] = '0PandaDEV/awesome-windows'

# Fetch contributors
contributors = get_contributors()

# Process results
for contributor in contributors:
    print(f"{contributor['login']}: {contributor['avatar_url']}")
</syntaxhighlight>

=== Integration with GitHub Actions ===
<syntaxhighlight lang="yaml">
- name: Update contributors
  env:
    GITHUB_PAT: ${{ secrets.PAT }}
  run: |
    output=$(python .github/scripts/update_contributors.py)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_GitHub_API_Integration]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Python]]
