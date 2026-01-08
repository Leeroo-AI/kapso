# Implementation: Update_readme_generation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|wsrv.nl Image Proxy|https://images.weserv.nl/docs/]]
|-
! Domains
| [[domain::Markdown_Generation]], [[domain::README_Automation]], [[domain::HTML_Generation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Concrete tool for generating the Backers markdown section with contributor avatars and links.

=== Description ===

This portion of the `update_readme()` function generates a formatted markdown/HTML block for the Backers section of the README. It iterates through contributor data to create avatar images with circular masks (via weserv.nl image proxy), wrapped in links to each contributor's GitHub profile. The block also includes a thank-you message and a Buy Me a Coffee support badge.

=== Usage ===

Use this code pattern when you need to generate a formatted contributor section with avatars. This step creates the new content that will replace the existing Backers section in the README.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/scripts/update_contributors.py
* '''Lines:''' L25-36

=== Signature ===
<syntaxhighlight lang="python">
def update_readme(contributors: list[dict]) -> None:
    """
    Generate and insert Backers section into README.

    Args:
        contributors: List of contributor dicts with 'login' and 'avatar_url' keys.

    Generation Logic (lines 29-36):
        1. Create header: "## Backers\\n\\nThanks to all contributors..."
        2. For each contributor:
           - Create <a> tag linking to GitHub profile
           - Create <img> tag with circular avatar via weserv.nl proxy
        3. Append support message with Buy Me a Coffee badge

    Image URL Template:
        https://images.weserv.nl/?url={avatar_url}&fit=cover&mask=circle&maxage=7d
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from update_contributors import update_readme
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| contributors || list[dict] || Yes || List with `login` and `avatar_url` keys per contributor
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| new_block || str (internal) || Generated markdown/HTML string for Backers section
|}

== Usage Examples ==

=== Generated Output Format ===
<syntaxhighlight lang="html">
## Backers

Thanks to all contributors without you this project would not exist.

<a href='https://github.com/0PandaDEV'><img src='https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/123&fit=cover&mask=circle&maxage=7d' width='60' height='60' alt='0PandaDEV'/></a> <a href='https://github.com/contributor2'><img src='https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/456&fit=cover&mask=circle&maxage=7d' width='60' height='60' alt='contributor2'/></a>

Please, consider supporting me as it is a lot of work to maintain this list! Thanks a lot.

<a href="https://buymeacoffee.com/pandadev_"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black"/></a>

</syntaxhighlight>

=== Image Proxy Parameters ===
<syntaxhighlight lang="text">
weserv.nl parameters used:
- url={avatar_url}  : Source image URL
- fit=cover         : Cover the entire area
- mask=circle       : Apply circular mask
- maxage=7d         : Cache for 7 days
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_README_Section_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Python]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:0PandaDEV_Awesome_windows_Image_Proxy_Caching]]
