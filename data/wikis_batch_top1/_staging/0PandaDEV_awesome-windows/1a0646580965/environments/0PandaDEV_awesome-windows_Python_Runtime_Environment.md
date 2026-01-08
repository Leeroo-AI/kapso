# Python_Runtime_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Python Documentation|https://docs.python.org/3/]]
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Scripting]]
|-
! Last Updated
| [[last_updated::2026-01-08 00:00 GMT]]
|}

== Overview ==

Python 3.x runtime environment with `requests` library for GitHub API integration and README automation scripts.

=== Description ===

This environment provides a Python 3.x runtime with the `requests` HTTP library for executing the contributor update automation script. The script fetches contributor data from the GitHub API, detects changes, generates HTML avatar blocks, and updates the README.md file using regex pattern matching.

=== Usage ===

Use this environment for any **Python-based automation** workflow that interacts with the GitHub API or performs file manipulation. This is the mandatory prerequisite for running the `get_contributors_Function`, `has_contributors_changed_Function`, `update_readme_HTML_Block`, and `update_readme_Regex_Replace` implementations.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| Runtime || Python 3.x || Any Python 3 version (3.7+ recommended)
|-
| Memory || Minimal || Script uses minimal resources
|-
| Disk || Minimal || Only README.md file I/O
|-
| Network || Internet connection || For GitHub API requests
|}

== Dependencies ==

=== System Packages ===
* `python3` >= 3.7

=== Python Packages ===
* `requests` (any version)

=== Standard Library (included) ===
* `os` - Environment variable access
* `re` - Regular expression operations

== Credentials ==

The following environment variables must be set:
* `GITHUB_PAT`: GitHub Personal Access Token with `read:user` scope for API authentication
* `GITHUB_REPOSITORY`: Repository identifier in `owner/repo` format (auto-set by GitHub Actions)

== Quick Install ==

<syntaxhighlight lang="bash">
# Upgrade pip and install requests
python -m pip install --upgrade pip
pip install requests
</syntaxhighlight>

== Code Evidence ==

Python imports from `.github/scripts/update_contributors.py:1-3`:
<syntaxhighlight lang="python">
import os
import re
import requests
</syntaxhighlight>

Environment variable usage from `.github/scripts/update_contributors.py:6-11`:
<syntaxhighlight lang="python">
def get_contributors():
    headers = {'Authorization': f"token {os.environ.get('GITHUB_PAT')}"}
    repo = os.environ.get('GITHUB_REPOSITORY')
    response = requests.get(
        f'https://api.github.com/repos/{repo}/contributors', headers=headers)
    return [contributor for contributor in response.json() if contributor['login'] != 'actions-user']
</syntaxhighlight>

Regex pattern for README update from `.github/scripts/update_contributors.py:38-39`:
<syntaxhighlight lang="python">
    pattern = r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"
    content = re.sub(pattern, new_block, content)
</syntaxhighlight>

Pip installation in workflow from `.github/workflows/update_contributors.yml:21-24`:
<syntaxhighlight lang="yaml">
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install requests
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ModuleNotFoundError: No module named 'requests'` || requests not installed || `pip install requests`
|-
|| `KeyError: 'GITHUB_PAT'` || Environment variable not set || Set `GITHUB_PAT` in workflow env
|-
|| `json.decoder.JSONDecodeError` || Invalid API response (rate limit) || Check API rate limits; verify PAT is valid
|-
|| `TypeError: 'NoneType' object is not iterable` || API returned null/None || Verify GITHUB_REPOSITORY is set correctly
|}

== Compatibility Notes ==

* '''Python 2 vs 3:''' Script requires Python 3 (f-strings, modern syntax)
* '''Virtual environments:''' Recommended but not required; GitHub Actions uses isolated environments
* '''Rate limits:''' Authenticated requests have 5000/hour limit; unauthenticated is 60/hour
* '''weserv.nl dependency:''' Avatar HTML generation relies on external image proxy service

== Related Pages ==

=== Required By ===
This environment is required by:
* Implementation: get_contributors_Function
* Implementation: has_contributors_changed_Function
* Implementation: update_readme_HTML_Block
* Implementation: update_readme_Regex_Replace
