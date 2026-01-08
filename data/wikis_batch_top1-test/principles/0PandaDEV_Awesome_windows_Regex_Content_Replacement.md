# Principle: Regex_Content_Replacement

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Python re module|https://docs.python.org/3/library/re.html]]
* [[source::Doc|Regular Expression HOWTO|https://docs.python.org/3/howto/regex.html]]
|-
! Domains
| [[domain::Text_Processing]], [[domain::Regex]], [[domain::File_Manipulation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Technique for replacing specific sections of text files using regular expression pattern matching.

=== Description ===

Regex Content Replacement uses regular expressions to identify and replace specific portions of text within files. Unlike simple string replacement, regex enables pattern-based matching that can handle variable content between known boundaries (anchors). This is essential for updating dynamic sections in otherwise static files.

The technique requires understanding of regex syntax including multiline mode, non-greedy matching, lookahead assertions, and capture groups. Proper anchor selection ensures only the intended content is replaced.

=== Usage ===

Use this principle when:
- Updating specific sections in markdown files
- Replacing content between known markers
- Performing complex find-and-replace operations
- Modifying configuration files programmatically

== Theoretical Basis ==

=== Pattern Components ===
<syntaxhighlight lang="text">
Start Anchor: Pattern that marks beginning of section
  Example: ^## Backers (header at line start)

Content Match: Pattern for content to replace
  Example: .*? (non-greedy match of any content)

End Anchor: Pattern that marks end of section
  Example: (?=^\[oss\]:) (lookahead for reference link)
</syntaxhighlight>

=== Regex Modes ===
{| class="wikitable"
|-
! Mode !! Flag !! Effect
|-
| MULTILINE || m || ^ and $ match line boundaries, not just string start/end
|-
| DOTALL || s || . matches newlines (normally doesn't)
|-
| Combined || (?ms) || Both modes enabled inline
|}

=== Lookahead Assertions ===
<syntaxhighlight lang="text">
(?=pattern)  - Positive lookahead: match only if pattern follows
(?!pattern)  - Negative lookahead: match only if pattern doesn't follow

Key property: Lookaheads don't consume characters
- The matched text stops BEFORE the lookahead pattern
- The lookahead pattern remains in the output
</syntaxhighlight>

=== Example Pattern ===
<syntaxhighlight lang="python">
# Pattern: r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"

# (?ms)         - Enable MULTILINE and DOTALL
# ^## Backers   - Match "## Backers" at line start
# \s*\n         - Optional whitespace, then newline
# .*?           - Non-greedy match (as little as possible)
# (?=^\[oss\]:) - Stop before "[oss]:" at line start

# This matches the entire Backers section without consuming
# the reference link definitions that follow
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_Awesome_windows_update_readme_replacement]]
