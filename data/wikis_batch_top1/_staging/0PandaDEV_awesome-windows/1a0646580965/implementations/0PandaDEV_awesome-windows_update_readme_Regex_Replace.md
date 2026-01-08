# Implementation: update_readme Regex Replace

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Python re.sub|https://docs.python.org/3/library/re.html#re.sub]]
|-
! Domains
| [[domain::Python]], [[domain::Regex]], [[domain::File_I/O]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Python code that uses `re.sub()` to replace the Backers section in README.md with newly generated content.

=== Description ===

This is an '''API Doc''' for the regex replacement portion of the `update_readme()` function. It uses Python's `re.sub()` to atomically replace the `## Backers` section while preserving all other README content.

=== Usage ===

Executes after the HTML block is generated. Performs the actual file modification by:
1. Matching the existing Backers section using regex
2. Replacing it with the newly generated content
3. Writing the updated file to disk

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/scripts/update_contributors.py:L38-41

=== Implementation ===

<syntaxhighlight lang="python">
def update_readme(contributors):
    with open('README.md', 'r') as file:
        content = file.read()

    # ... HTML block generation (lines 29-36) ...

    # Regex pattern to match Backers section
    pattern = r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"

    # Replace matched section with new content
    content = re.sub(pattern, new_block, content)

    # Write updated content back to file
    with open('README.md', 'w') as file:
        file.write(content)
</syntaxhighlight>

=== Regex Pattern Breakdown ===

<syntaxhighlight lang="text">
Pattern: r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"

(?ms)           → Flags: m=multiline, s=dotall
^## Backers     → Match "## Backers" at start of line
\s*\n           → Match optional whitespace and newline
.*?             → Match any characters (non-greedy) including newlines
(?=^\[oss\]:)   → Positive lookahead: stop before "[oss]:" at line start
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
import re

# re.sub(pattern, replacement, string) → str
# Replaces all matches of pattern in string with replacement
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pattern || str || Yes || Regex pattern matching section to replace
|-
| new_block || str || Yes || Generated HTML content from previous step
|-
| content || str || Yes || Current README.md content
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| README.md || file || Updated file written to disk
|-
| content || str || String with replacement applied (intermediate)
|}

== Usage Examples ==

=== Example 1: Pattern Matching ===
<syntaxhighlight lang="python">
import re

# README content
readme = """
# Awesome Windows

Some intro text...

## Backers

Old contributor content here...

[oss]: ./assets/oss.png
[paid]: ./assets/paid.png
"""

# Pattern matches from "## Backers" to just before "[oss]:"
pattern = r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"

# Find the match
match = re.search(pattern, readme)
print(match.group())
# Output:
# ## Backers
#
# Old contributor content here...
#
</syntaxhighlight>

=== Example 2: Substitution ===
<syntaxhighlight lang="python">
import re

readme = "... ## Backers\n\nOld content\n\n[oss]: ..."

new_block = "## Backers\n\nNew content here!\n\n"
pattern = r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"

updated = re.sub(pattern, new_block, readme)
# Result: "... ## Backers\n\nNew content here!\n\n[oss]: ..."
</syntaxhighlight>

=== Example 3: File Update Flow ===
<syntaxhighlight lang="python">
def update_readme(contributors):
    # Step 1: Read current content
    with open('README.md', 'r') as file:
        content = file.read()

    # Step 2: Generate new Backers section
    new_block = "## Backers\n\n..."  # Built from contributors

    # Step 3: Replace section using regex
    pattern = r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"
    content = re.sub(pattern, new_block, content)

    # Step 4: Write updated content
    with open('README.md', 'w') as file:
        file.write(content)

# Before: ## Backers\n\nOld avatars\n\n[oss]: ...
# After:  ## Backers\n\nNew avatars\n\n[oss]: ...
</syntaxhighlight>

=== Example 4: Lookahead Explanation ===
<syntaxhighlight lang="python">
# Why (?=^\[oss\]:) lookahead?

# Without lookahead: .*?(?:\[oss\]:)
# Would CONSUME "[oss]:" and require it in replacement

# With lookahead: .*?(?=^\[oss\]:)
# Matches UP TO but NOT INCLUDING "[oss]:"
# So "[oss]:" remains in place after substitution

# This preserves the reference link definitions:
# [oss]: ./assets/oss.png
# [paid]: ./assets/paid.png
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_README_Content_Update]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_Python_Runtime_Environment]]
