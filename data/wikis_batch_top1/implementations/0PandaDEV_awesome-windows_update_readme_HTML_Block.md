# Implementation: update_readme HTML Block

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|weserv.nl Documentation|https://images.weserv.nl/docs/]]
* [[source::Doc|Python f-strings|https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals]]
|-
! Domains
| [[domain::Python]], [[domain::HTML_Generation]], [[domain::String_Formatting]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Python code block within `update_readme()` that generates the HTML for contributor avatars using weserv.nl image proxy.

=== Description ===

This is an '''API Doc''' for the HTML generation portion of the `update_readme()` function. It constructs the Backers section content by iterating over contributors and building HTML elements with f-strings.

=== Usage ===

This code block executes after change detection confirms an update is needed. It generates the complete `## Backers` section including:
- Section header and description
- All contributor avatar links
- Support call-to-action with Buy Me A Coffee badge

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/scripts/update_contributors.py:L25-36

=== Implementation ===

<syntaxhighlight lang="python">
def update_readme(contributors):
    with open('README.md', 'r') as file:
        content = file.read()

    # Build the Backers section header
    new_block = "## Backers\n\nThanks to all contributors without you this project would not exist.\n\n"

    # Generate HTML for each contributor avatar
    for contributor in contributors:
        avatar_url = contributor['avatar_url']
        new_block += f"<a href='https://github.com/{contributor['login']}'>"
        new_block += f"<img src='https://images.weserv.nl/?url={avatar_url}&fit=cover&mask=circle&maxage=7d' "
        new_block += f"width='60' height='60' alt='{contributor['login']}'/></a> "

    # Add support call-to-action
    new_block += "\n\nPlease, consider supporting me as it is a lot of work to maintain this list! Thanks a lot.\n\n"
    new_block += "<a href=\"https://buymeacoffee.com/pandadev_\">"
    new_block += "<img src=\"https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black\"/></a>\n\n"

    # ... regex replacement follows in next step
</syntaxhighlight>

=== Avatar HTML Template ===

<syntaxhighlight lang="html">
<!-- Single avatar element -->
<a href='https://github.com/{login}'>
  <img src='https://images.weserv.nl/?url={avatar_url}&fit=cover&mask=circle&maxage=7d'
       width='60'
       height='60'
       alt='{login}'/>
</a>
</syntaxhighlight>

=== weserv.nl URL Construction ===

<syntaxhighlight lang="text">
Base URL: https://images.weserv.nl/

Parameters:
  url={avatar_url}    → Source image (GitHub avatar)
  fit=cover           → Fill 60x60 dimensions
  mask=circle         → Circular crop
  maxage=7d           → Cache for 7 days

Full URL Example:
https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/70103896?v=4&fit=cover&mask=circle&maxage=7d
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| contributors || list[dict] || Yes || Contributor list from `get_contributors()`
|-
| contributors[].login || str || Yes || GitHub username for link and alt text
|-
| contributors[].avatar_url || str || Yes || GitHub avatar URL for image source
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| new_block || str || Complete "## Backers" section HTML/markdown content
|}

== Usage Examples ==

=== Example 1: Single Contributor ===
<syntaxhighlight lang="python">
contributors = [
    {"login": "0PandaDEV", "avatar_url": "https://avatars.githubusercontent.com/u/70103896?v=4"}
]

# Generated HTML:
"""
## Backers

Thanks to all contributors without you this project would not exist.

<a href='https://github.com/0PandaDEV'><img src='https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/70103896?v=4&fit=cover&mask=circle&maxage=7d' width='60' height='60' alt='0PandaDEV'/></a>

Please, consider supporting me as it is a lot of work to maintain this list! Thanks a lot.

<a href="https://buymeacoffee.com/pandadev_"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black"/></a>

"""
</syntaxhighlight>

=== Example 2: Multiple Contributors ===
<syntaxhighlight lang="python">
contributors = [
    {"login": "0PandaDEV", "avatar_url": "https://avatars.githubusercontent.com/u/70103896?v=4"},
    {"login": "user1", "avatar_url": "https://avatars.githubusercontent.com/u/11111?v=4"},
    {"login": "user2", "avatar_url": "https://avatars.githubusercontent.com/u/22222?v=4"},
]

# Generated HTML block (avatars on same line, space-separated):
"""
<a href='...'><img .../></a> <a href='...'><img .../></a> <a href='...'><img .../></a>
"""
</syntaxhighlight>

=== Example 3: Rendered Output ===
<syntaxhighlight lang="text">
Rendered in GitHub README:

## Backers

Thanks to all contributors without you this project would not exist.

[○] [○] [○] [○]  ← Circular avatar images, clickable
 ↓   ↓   ↓   ↓
Links to GitHub profiles

Please, consider supporting me...

[Buy Me A Coffee] ← Badge button
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_Avatar_HTML_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_Python_Runtime_Environment]]
