# Implementation: get_contributors Function

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub REST API|https://docs.github.com/en/rest/repos/repos#list-repository-contributors]]
* [[source::Doc|Requests Library|https://docs.python-requests.org/]]
|-
! Domains
| [[domain::API]], [[domain::Python]], [[domain::GitHub]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Python function that fetches repository contributor data from the GitHub REST API with token authentication and bot filtering.

=== Description ===

This is an '''API Doc''' for the `get_contributors()` function in the contributor update script. It uses the `requests` library to call GitHub's Contributors API, authenticates with a Personal Access Token, and filters out automated accounts.

=== Usage ===

Called at the start of the contributor update workflow to retrieve the current list of repository contributors. Returns a list of dictionaries containing `login` and `avatar_url` for each contributor.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/scripts/update_contributors.py:L6-11

=== Signature ===

<syntaxhighlight lang="python">
def get_contributors() -> list[dict]:
    """
    Fetch repository contributors from GitHub API.

    Returns:
        list[dict]: List of contributor objects with 'login' and 'avatar_url' keys.
                    Excludes 'actions-user' (GitHub Actions bot).

    Environment Variables:
        GITHUB_PAT: Personal Access Token for authentication
        GITHUB_REPOSITORY: Repository in 'owner/repo' format

    Raises:
        requests.exceptions.RequestException: On API errors
    """
</syntaxhighlight>

=== Full Implementation ===

<syntaxhighlight lang="python">
import os
import requests

def get_contributors():
    headers = {'Authorization': f"token {os.environ.get('GITHUB_PAT')}"}
    repo = os.environ.get('GITHUB_REPOSITORY')
    response = requests.get(
        f'https://api.github.com/repos/{repo}/contributors', headers=headers)
    return [contributor for contributor in response.json()
            if contributor['login'] != 'actions-user']
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
# From the script itself (not importable as a module)
# This function is defined in .github/scripts/update_contributors.py

import os
import requests

def get_contributors():
    ...
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| GITHUB_PAT || env var || Yes || Personal Access Token for GitHub API authentication
|-
| GITHUB_REPOSITORY || env var || Yes || Repository in 'owner/repo' format (auto-set in Actions)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| contributors || list[dict] || List of contributor objects
|-
| contributor.login || str || GitHub username
|-
| contributor.avatar_url || str || URL to user's avatar image
|}

=== Output Format ===

<syntaxhighlight lang="python">
[
    {
        "login": "0PandaDEV",
        "id": 70103896,
        "avatar_url": "https://avatars.githubusercontent.com/u/70103896?v=4",
        "contributions": 142
    },
    {
        "login": "contributor1",
        "id": 12345678,
        "avatar_url": "https://avatars.githubusercontent.com/u/12345678?v=4",
        "contributions": 5
    }
    # ... more contributors
]
</syntaxhighlight>

== Usage Examples ==

=== Example 1: Basic Usage ===
<syntaxhighlight lang="python">
import os

# Set environment (normally done by GitHub Actions)
os.environ['GITHUB_PAT'] = 'ghp_xxxx...'
os.environ['GITHUB_REPOSITORY'] = '0PandaDEV/awesome-windows'

# Fetch contributors
contributors = get_contributors()

# Output
print(f"Found {len(contributors)} contributors")
for c in contributors:
    print(f"  - {c['login']}: {c['avatar_url']}")
</syntaxhighlight>

=== Example 2: Within Workflow Context ===
<syntaxhighlight lang="python">
# As called in the main script
if __name__ == "__main__":
    contributors = get_contributors()

    # Output: List of dicts without 'actions-user'
    # [
    #   {"login": "0PandaDEV", "avatar_url": "https://..."},
    #   {"login": "user2", "avatar_url": "https://..."},
    # ]

    if has_contributors_changed(contributors):
        update_readme(contributors)
        print("Contributors updated")
    else:
        print("No changes in contributors")
</syntaxhighlight>

=== Example 3: API Response Structure ===
<syntaxhighlight lang="json">
// Raw GitHub API response (before filtering)
[
  {
    "login": "0PandaDEV",
    "id": 70103896,
    "node_id": "MDQ6VXNlcjcwMTAzODk2",
    "avatar_url": "https://avatars.githubusercontent.com/u/70103896?v=4",
    "gravatar_id": "",
    "url": "https://api.github.com/users/0PandaDEV",
    "html_url": "https://github.com/0PandaDEV",
    "type": "User",
    "contributions": 142
  },
  {
    "login": "actions-user",
    "id": 41898282,
    "avatar_url": "https://avatars.githubusercontent.com/u/41898282?v=4",
    "type": "Bot",
    "contributions": 10
  }  // This one gets filtered out
]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_GitHub_API_Integration]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_Python_Runtime_Environment]]
