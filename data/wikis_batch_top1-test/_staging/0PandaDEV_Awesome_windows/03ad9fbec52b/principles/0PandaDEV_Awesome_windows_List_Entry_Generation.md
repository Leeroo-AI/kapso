# Principle: List_Entry_Generation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Awesome List Guidelines|https://github.com/sindresorhus/awesome/blob/main/contributing.md]]
* [[source::Doc|GitHub Markdown|https://docs.github.com/en/get-started/writing-on-github]]
|-
! Domains
| [[domain::Markdown]], [[domain::Content_Generation]], [[domain::Awesome_Lists]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Technique for constructing formatted markdown list entries with conditional badges and links.

=== Description ===

List Entry Generation creates properly formatted list items for awesome-list style repositories. Each entry follows a consistent pattern: bullet point, linked name, description, and optional badges. The generation process involves conditional logic to include/exclude badges based on metadata (open source status, paid status) and proper markdown link construction.

This ensures all entries maintain consistent formatting regardless of whether they were added manually or via automation.

=== Usage ===

Use this principle when:
- Automating additions to curated lists
- Generating consistent markdown list entries
- Building entry templates with conditional elements
- Creating badge-decorated list items

== Theoretical Basis ==

=== Entry Format ===
<syntaxhighlight lang="text">
Standard awesome-list entry format:
* [Name](URL) - Description [badge1] [badge2]

Components:
- * : Unordered list marker
- [Name](URL) : Markdown link
- - : Separator
- Description : Brief text
- [badge] : Optional badge images
</syntaxhighlight>

=== Badge Types ===
{| class="wikitable"
|-
! Badge !! Meaning !! Syntax
|-
| Open Source || Source code available || `[![oss][oss]](repo-url)` or `![oss]`
|-
| Paid || Requires payment || `![paid]`
|-
| Free || No cost || `![free]`
|-
| Favorite || Editor's choice || `![fav]`
|}

=== Conditional Badge Logic ===
<syntaxhighlight lang="bash">
# Check checkbox state in issue body
if echo "$BODY" | grep -q "\[X\] Open Source"; then
  if [ -n "$REPO_URL" ]; then
    # Clickable badge linking to repo
    OSS_BADGE="[![Open-Source Software][oss]]($REPO_URL)"
  else
    # Static badge (no repo provided)
    OSS_BADGE="![oss]"
  fi
else
  OSS_BADGE=""  # Not open source
fi
</syntaxhighlight>

=== Assembly Pattern ===
<syntaxhighlight lang="bash">
# Build entry from components
# Spaces between elements are important for rendering
NEW_ENTRY="* [$APP_NAME]($APP_URL) - $APP_DESCRIPTION $OPEN_SOURCE_ICON $PAID_ICON"

# Example output:
# * [VS Code](https://code.visualstudio.com) - Code editor [![oss][oss]](https://github.com/microsoft/vscode)
</syntaxhighlight>

=== Badge Reference Definitions ===
<syntaxhighlight lang="markdown">
# At bottom of README, define badge images:
[oss]: /assets/oss.svg "Open-Source Software"
[paid]: /assets/paid.svg "Paid Software"
[fav]: /assets/fav.svg "Editor's Choice"

# These enable [oss], [paid], [fav] shorthand throughout document
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_Awesome_windows_entry_builder]]
