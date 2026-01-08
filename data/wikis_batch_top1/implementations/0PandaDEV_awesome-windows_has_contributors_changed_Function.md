# Implementation: has_contributors_changed Function

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Python File I/O|https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files]]
|-
! Domains
| [[domain::Python]], [[domain::Optimization]], [[domain::File_I/O]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Python function that checks whether any contributors are missing from the README to determine if an update is needed.

=== Description ===

This is an '''API Doc''' for the `has_contributors_changed()` function in the contributor update script. It reads README.md and checks if each contributor's GitHub profile URL is present, returning early on the first missing contributor.

=== Usage ===

Called after fetching contributors to determine whether the README needs updating. If it returns `False`, the workflow skips the update and avoids an empty commit.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/scripts/update_contributors.py:L14-22

=== Signature ===

<syntaxhighlight lang="python">
def has_contributors_changed(contributors: list[dict]) -> bool:
    """
    Check if any contributors are missing from README.md.

    Args:
        contributors: List of contributor dicts with 'login' key

    Returns:
        bool: True if any contributor's profile URL is missing from README,
              False if all contributors are already listed.

    Side Effects:
        Reads README.md from current working directory
    """
</syntaxhighlight>

=== Full Implementation ===

<syntaxhighlight lang="python">
def has_contributors_changed(contributors):
    with open('README.md', 'r') as file:
        content = file.read()

    for contributor in contributors:
        username = contributor['login']
        if f"https://github.com/{username}" not in content:
            return True
    return False
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
# From the script itself (not importable as a module)
# This function is defined in .github/scripts/update_contributors.py

def has_contributors_changed(contributors):
    ...
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| contributors || list[dict] || Yes || List of contributor objects from `get_contributors()`
|-
| contributors[].login || str || Yes || GitHub username of contributor
|-
| README.md || file || Yes || Current README file (read from disk)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| changed || bool || True if update needed, False otherwise
|}

== Usage Examples ==

=== Example 1: No Changes Needed ===
<syntaxhighlight lang="python">
# All contributors already in README
contributors = [
    {"login": "0PandaDEV", "avatar_url": "..."},
    {"login": "contributor1", "avatar_url": "..."},
]

# README.md contains:
# <a href='https://github.com/0PandaDEV'>...
# <a href='https://github.com/contributor1'>...

result = has_contributors_changed(contributors)
# result = False (no update needed)
</syntaxhighlight>

=== Example 2: New Contributor ===
<syntaxhighlight lang="python">
# New contributor not in README
contributors = [
    {"login": "0PandaDEV", "avatar_url": "..."},
    {"login": "new_contributor", "avatar_url": "..."},  # NEW
]

# README.md only contains:
# <a href='https://github.com/0PandaDEV'>...

result = has_contributors_changed(contributors)
# result = True (update needed)
</syntaxhighlight>

=== Example 3: Within Workflow ===
<syntaxhighlight lang="python">
if __name__ == "__main__":
    contributors = get_contributors()

    if has_contributors_changed(contributors):
        # Proceed with update
        update_readme(contributors)
        print("Contributors updated")
    else:
        # Skip update, no empty commit
        print("No changes in contributors")
</syntaxhighlight>

=== Example 4: String Check Logic ===
<syntaxhighlight lang="python">
# The function checks for this exact pattern:
# f"https://github.com/{username}"

# For contributor {"login": "0PandaDEV"}:
# Searches for: "https://github.com/0PandaDEV"

# README Backers section contains:
# <a href='https://github.com/0PandaDEV'><img...

# The substring "https://github.com/0PandaDEV" IS found
# So this contributor is not flagged as missing
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_Change_Detection]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_Python_Runtime_Environment]]
