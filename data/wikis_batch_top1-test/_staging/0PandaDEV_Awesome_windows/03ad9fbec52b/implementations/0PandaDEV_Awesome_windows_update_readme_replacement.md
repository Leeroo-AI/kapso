# Implementation: Update_readme_replacement

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Python re module|https://docs.python.org/3/library/re.html]]
|-
! Domains
| [[domain::Regex]], [[domain::Text_Processing]], [[domain::File_IO]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Concrete tool for replacing the Backers section in README using regex pattern matching.

=== Description ===

This portion of the `update_readme()` function uses Python's `re.sub()` to replace the existing Backers section with newly generated content. The regex pattern uses multiline mode to match from "## Backers" through all content until the reference link definitions (e.g., `[oss]:`). This ensures the entire section is replaced while preserving the rest of the README.

=== Usage ===

Use this pattern when you need to replace a specific section of a markdown file using regex. The pattern is designed to match multi-line content between section headers and reference link markers.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/scripts/update_contributors.py
* '''Lines:''' L38-41

=== Signature ===
<syntaxhighlight lang="python">
# Within update_readme() function:
import re

pattern = r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"
content = re.sub(pattern, new_block, content)

# Pattern explanation:
# (?ms)         - Enable MULTILINE and DOTALL modes
# ^## Backers   - Match "## Backers" at line start
# \s*\n         - Match optional whitespace and newline
# .*?           - Non-greedy match of any content (including newlines due to DOTALL)
# (?=^\[oss\]:) - Positive lookahead for "[oss]:" at line start (reference links section)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
import re
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pattern || str || Yes || Regex pattern: `r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:\)"`
|-
| new_block || str || Yes || Generated Backers section content from generation step
|-
| content || str || Yes || Current README.md file content
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| content || str || Updated README content with replaced Backers section
|-
| README.md || file || File written with updated content
|}

== Usage Examples ==

=== Regex Pattern Explanation ===
<syntaxhighlight lang="python">
import re

# The pattern matches the Backers section from header to reference links
pattern = r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"

# Example README structure:
readme_content = """
## Some Section
Content here...

## Backers
Old contributor content here...
Old avatars...

[oss]: /path/to/badge
[paid]: /path/to/badge
"""

# The pattern matches:
# - "## Backers\nOld contributor content...\n\n"
# - Stops BEFORE "[oss]:" (lookahead doesn't consume)

new_backers = "## Backers\n\nNew content here...\n\n"
updated = re.sub(pattern, new_backers, readme_content)
</syntaxhighlight>

=== Full File Update Flow ===
<syntaxhighlight lang="python">
def update_readme(contributors):
    # Read current file
    with open('README.md', 'r') as file:
        content = file.read()

    # Generate new block (see README_Section_Generation)
    new_block = generate_backers_block(contributors)

    # Replace using regex
    pattern = r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"
    content = re.sub(pattern, new_block, content)

    # Write updated file
    with open('README.md', 'w') as file:
        file.write(content)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_Regex_Content_Replacement]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Python]]
