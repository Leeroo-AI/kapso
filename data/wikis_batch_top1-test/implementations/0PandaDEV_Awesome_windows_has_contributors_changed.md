# Implementation: Has_contributors_changed

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::Content_Detection]], [[domain::File_IO]], [[domain::Automation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Concrete tool for detecting changes between fetched contributor data and current README content.

=== Description ===

The `has_contributors_changed()` function compares a list of contributor data against the current README.md file content. It checks whether each contributor's GitHub profile URL exists in the README. If any contributor URL is missing, it indicates that the README needs to be updated with new contributor information.

=== Usage ===

Use this function after fetching contributor data from the GitHub API to determine whether the README needs updating. This check prevents unnecessary file writes and git commits when no new contributors have been added.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/scripts/update_contributors.py
* '''Lines:''' L14-22

=== Signature ===
<syntaxhighlight lang="python">
def has_contributors_changed(contributors: list[dict]) -> bool:
    """
    Check if any contributors are missing from the README.

    Args:
        contributors: List of contributor dictionaries with 'login' key.

    Returns:
        bool: True if any contributor GitHub URL is not in README, False otherwise.

    Algorithm:
        1. Read current README.md content
        2. For each contributor, check if 'https://github.com/{login}' exists
        3. Return True on first missing contributor, False if all present
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from update_contributors import has_contributors_changed
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| contributors || list[dict] || Yes || List of contributor dicts, each must have `login` key
|-
| README.md || file || Yes || Existing README file in working directory (read implicitly)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return value || bool || True if README needs updating, False if all contributors present
|}

== Usage Examples ==

=== Basic Change Detection ===
<syntaxhighlight lang="python">
# Assume contributors fetched from API
contributors = [
    {'login': 'user1', 'avatar_url': 'https://...'},
    {'login': 'user2', 'avatar_url': 'https://...'},
    {'login': 'new_contributor', 'avatar_url': 'https://...'},
]

# Check if any new contributors need to be added
if has_contributors_changed(contributors):
    print("New contributors detected - update needed")
    update_readme(contributors)
else:
    print("No changes in contributors")
</syntaxhighlight>

=== Conditional Workflow Execution ===
<syntaxhighlight lang="python">
if __name__ == "__main__":
    contributors = get_contributors()
    if has_contributors_changed(contributors):
        update_readme(contributors)
        print("Contributors updated")  # Triggers git commit step
    else:
        print("No changes in contributors")  # Skips git commit step
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_Content_Change_Detection]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Python]]
