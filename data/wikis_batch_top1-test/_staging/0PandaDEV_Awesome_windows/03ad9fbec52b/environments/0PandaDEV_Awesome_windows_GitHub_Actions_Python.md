# Environment: GitHub_Actions_Python

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|0PandaDEV/awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Actions Python Setup|https://github.com/actions/setup-python]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::CI_CD]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==
GitHub Actions runner with Python 3.x and requests library for API interactions.

=== Description ===
This environment provides a Python runtime on GitHub Actions runners (ubuntu-latest) for scripts that interact with the GitHub API. It uses the `actions/setup-python@v5` action to configure Python 3.x and installs the `requests` package via pip for making HTTP requests to the GitHub REST API.

=== Usage ===
Use this environment for any **automated README updates** or **API-based contributor management** workflows. It is the mandatory prerequisite for running Python scripts like `update_contributors.py` that fetch data from GitHub's REST API.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Ubuntu (latest) || GitHub Actions hosted runner
|-
| Runtime || Python 3.x || Configured via actions/setup-python@v5
|-
| Network || Outbound HTTPS || Required for GitHub API calls
|}

== Dependencies ==

=== System Packages ===
* `python3` - Python interpreter (via actions/setup-python)
* `pip` - Python package manager

=== Python Packages ===
* `requests` - HTTP library for API calls

== Credentials ==

The following environment variables must be set:
* `GITHUB_PAT`: GitHub Personal Access Token with repository read access for fetching contributor data
* `GITHUB_REPOSITORY`: Repository name in `owner/repo` format (auto-set by GitHub Actions)

== Quick Install ==

<syntaxhighlight lang="bash">
# Install dependencies (as done in workflow)
python -m pip install --upgrade pip
pip install requests
</syntaxhighlight>

== Code Evidence ==

Environment variable usage from `update_contributors.py:6-11`:
<syntaxhighlight lang="python">
def get_contributors():
    headers = {'Authorization': f"token {os.environ.get('GITHUB_PAT')}"}
    repo = os.environ.get('GITHUB_REPOSITORY')
    response = requests.get(
        f'https://api.github.com/repos/{repo}/contributors', headers=headers)
    return [contributor for contributor in response.json() if contributor['login'] != 'actions-user']
</syntaxhighlight>

Workflow Python setup from `update_contributors.yml:16-24`:
<syntaxhighlight lang="yaml">
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.x"

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
|| `KeyError: 'GITHUB_PAT'` || PAT not configured || Add PAT secret to repository settings and reference as `${{ secrets.PAT }}`
|-
|| `requests.exceptions.ConnectionError` || Network blocked || Ensure runner has outbound HTTPS access
|-
|| `401 Unauthorized` from API || Invalid or expired PAT || Regenerate Personal Access Token with correct scopes
|}

== Compatibility Notes ==

* '''GitHub Actions:''' Designed specifically for GitHub-hosted runners; self-hosted runners require manual Python installation
* '''Python Version:''' Uses `python-version: "3.x"` for latest stable Python 3
* '''Rate Limiting:''' GitHub API has rate limits; authenticated requests allow 5000 requests/hour

== Related Pages ==

* [[required_by::Implementation:0PandaDEV_Awesome_windows_get_contributors]]
* [[required_by::Implementation:0PandaDEV_Awesome_windows_has_contributors_changed]]
* [[required_by::Implementation:0PandaDEV_Awesome_windows_update_readme_generation]]
* [[required_by::Implementation:0PandaDEV_Awesome_windows_update_readme_replacement]]
