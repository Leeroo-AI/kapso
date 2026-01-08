# Implementation: Entry_builder

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::Shell_Scripting]], [[domain::Markdown_Generation]], [[domain::String_Manipulation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Pattern document for constructing markdown list entries with conditional badge icons.

=== Description ===

This shell script pattern constructs a markdown list entry string for the Awesome Windows README. It reads checkbox states from the issue body using grep, conditionally adds badge icons (open-source, paid), and formats the final entry with app name, URL, description, and icons in the standard awesome-list format.

=== Usage ===

Use this pattern when generating list entries with conditional formatting based on checkbox inputs. The pattern demonstrates shell variable interpolation, conditional string building, and markdown link construction.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/covert_to_pr.yml
* '''Lines:''' L53-72

=== Signature ===
<syntaxhighlight lang="bash">
# Check if the application is open source
if echo "$ISSUE_BODY" | grep -q "\[X\] Open Source"; then
  if [ -n "$REPO_URL" ]; then
    OPEN_SOURCE_ICON="[![Open-Source Software][oss]]($REPO_URL)"
  else
    OPEN_SOURCE_ICON="![oss]"
  fi
else
  OPEN_SOURCE_ICON=""
fi

# Check if the application is paid
if echo "$ISSUE_BODY" | grep -q "\[X\] Paid"; then
  PAID_ICON="![paid]"
else
  PAID_ICON=""
fi

# Create the new entry
NEW_ENTRY="* [$APP_NAME]($APP_URL) - $APP_DESCRIPTION $OPEN_SOURCE_ICON $PAID_ICON"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# No import needed - standard shell commands
# Requires: bash, grep
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| ISSUE_BODY || str || Yes || Full issue body text to search for checkboxes
|-
| APP_NAME || str || Yes || Application name from parsing step
|-
| APP_URL || str || Yes || Application URL from parsing step
|-
| APP_DESCRIPTION || str || Yes || Description from parsing step
|-
| REPO_URL || str || No || Repository URL (optional, for OSS badge link)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| NEW_ENTRY || str || Complete markdown list entry string
|-
| OPEN_SOURCE_ICON || str || Badge markdown or empty string
|-
| PAID_ICON || str || Badge markdown or empty string
|}

== Usage Examples ==

=== Checkbox Detection ===
<syntaxhighlight lang="bash">
# Issue body contains checkbox results like:
# - [X] Open Source
# - [ ] Paid
# - [X] Freemium

# grep -q returns exit code 0 if found, non-zero if not
if echo "$ISSUE_BODY" | grep -q "\[X\] Open Source"; then
  # Checkbox is checked
fi

# Note: [X] is the exact pattern from GitHub Issue Forms
# [ ] would indicate unchecked
</syntaxhighlight>

=== Badge Construction ===
<syntaxhighlight lang="bash">
# Open Source with repo link (clickable badge)
OPEN_SOURCE_ICON="[![Open-Source Software][oss]]($REPO_URL)"
# Renders as: clickable [oss] badge linking to repo

# Open Source without repo link (static badge)
OPEN_SOURCE_ICON="![oss]"
# Renders as: static [oss] badge

# Paid badge
PAID_ICON="![paid]"
# Renders as: [paid] badge

# Badge references defined at bottom of README:
# [oss]: /assets/oss.svg
# [paid]: /assets/paid.svg
</syntaxhighlight>

=== Final Entry Format ===
<syntaxhighlight lang="markdown">
* [Visual Studio Code](https://code.visualstudio.com) - Free source-code editor made by Microsoft [![Open-Source Software][oss]](https://github.com/microsoft/vscode)

* [Sublime Text](https://www.sublimetext.com) - Sophisticated text editor for code ![paid]

* [Notepad++](https://notepad-plus-plus.org) - Free source code editor [![Open-Source Software][oss]](https://github.com/notepad-plus-plus/notepad-plus-plus)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_List_Entry_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu]]
