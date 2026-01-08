# Principle: README Content Update

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Python re Module|https://docs.python.org/3/library/re.html]]
|-
! Domains
| [[domain::Text_Processing]], [[domain::Regex]], [[domain::File_I/O]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for updating specific sections of a markdown file using regex pattern matching and substitution.

=== Description ===

README Content Update is a text processing technique that replaces a defined section of a file while preserving all other content. This is essential for automated documentation updates where only a portion of the file should change.

The approach uses:
- '''Regex pattern:''' Match the section to replace
- '''Anchors:''' Use surrounding markers to bound the replacement
- '''Atomic replacement:''' Entire section swapped in one operation

This ensures:
- Other sections remain untouched
- Section boundaries are respected
- Content format is preserved

=== Usage ===

Apply this principle when:
- Automating documentation updates
- Replacing specific sections in templates
- Maintaining generated content within larger files
- Updating badges, stats, or dynamic content

== Theoretical Basis ==

'''Section Replacement Pattern:'''

<syntaxhighlight lang="text">
README.md Structure:
┌─────────────────────────────┐
│ # Title                     │
│ ... other content ...       │
├─────────────────────────────┤  ← Start anchor: "## Backers"
│ ## Backers                  │
│ ... backers content ...     │  ← Content to replace
│                             │
├─────────────────────────────┤  ← End anchor: "[oss]:"
│ [oss]: ...                  │
│ ... rest of file ...        │
└─────────────────────────────┘
</syntaxhighlight>

'''Regex Strategy:'''

{| class="wikitable"
|-
! Component !! Pattern !! Purpose
|-
| Start anchor || `^## Backers\s*\n` || Match section header
|-
| Content match || `.*?` (non-greedy) || Match all content between anchors
|-
| End anchor || `(?=^\[oss\]:)` (lookahead) || Stop before reference links
|-
| Flags || `(?ms)` || Multiline and dotall mode
|}

'''Regex Flags Explained:'''
- '''m (multiline):''' `^` and `$` match line boundaries, not just string boundaries
- '''s (dotall):''' `.` matches newlines too

'''Non-Greedy Matching:'''

Using `.*?` instead of `.*` ensures the match stops at the first occurrence of the end anchor, not the last.

== Practical Guide ==

=== Pattern Construction ===

<syntaxhighlight lang="python">
# Pattern components:
# (?ms)           - Flags: multiline, dotall
# ^## Backers     - Start at "## Backers" at line start
# \s*\n           - Consume whitespace and newline
# .*?             - Match content (non-greedy)
# (?=^\[oss\]:)   - Stop before "[oss]:" at line start (lookahead)

pattern = r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"
</syntaxhighlight>

=== Safe Replacement ===

1. Read entire file into memory
2. Apply regex substitution
3. Write entire file back

This ensures atomic updates—no partial writes.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_update_readme_Regex_Replace]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Automated_Contributor_Update]]
