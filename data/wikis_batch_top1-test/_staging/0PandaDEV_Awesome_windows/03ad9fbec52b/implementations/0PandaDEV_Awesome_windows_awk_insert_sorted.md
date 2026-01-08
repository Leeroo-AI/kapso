# Implementation: Awk_insert_sorted

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GNU awk|https://www.gnu.org/software/gawk/manual/gawk.html]]
|-
! Domains
| [[domain::Text_Processing]], [[domain::Shell_Scripting]], [[domain::Sorting]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

External tool document for inserting list entries alphabetically within category sections using awk.

=== Description ===

This awk script inserts a new list entry into the README.md file in alphabetical order within the correct category section. It tracks the current category using `## ` headers, and when in the target category, compares each list entry (`* `) against the new entry to find the correct insertion point. The comparison is case-insensitive and starts after the `* ` prefix.

=== Usage ===

Use this awk script pattern when you need to maintain sorted order while inserting items into a markdown list. The script handles section detection, alphabetical comparison, and ensures the entry is inserted even at the end of a section.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/covert_to_pr.yml
* '''Lines:''' L75-99

=== Signature ===
<syntaxhighlight lang="bash">
awk -v new_entry="$NEW_ENTRY" -v category="$CATEGORY" '
BEGIN {in_category=0; added=0}
/^## / {
  if (in_category && !added) {
    print new_entry
    added=1
  }
  in_category = ($0 ~ "^## " category)
  print
  if (in_category) print ""
  next
}
in_category && /^\* / {
  if (!added && tolower(substr(new_entry, 3)) < tolower(substr($0, 3))) {
    print new_entry
    added=1
  }
  print
  next
}
{print}
END {
  if (in_category && !added) print new_entry
}
' README.md > README.md.tmp && mv README.md.tmp README.md
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# awk is available by default on Ubuntu runners
# No explicit import needed
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| new_entry || str (awk var) || Yes || The formatted list entry to insert
|-
| category || str (awk var) || Yes || Target category name (e.g., "IDEs")
|-
| README.md || file || Yes || Source file to modify
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| README.md || file || Modified file with new entry inserted alphabetically
|}

== Usage Examples ==

=== Algorithm Walkthrough ===
<syntaxhighlight lang="text">
Input README:
## Audio
* Audacity - Audio editor
* VLC - Media player

## IDEs
* IntelliJ - Java IDE
* Visual Studio - Microsoft IDE

## Networking
...

Insert: "* Notepad++ - Text editor" into "IDEs"

Algorithm:
1. See "## Audio" -> not target category, print line
2. See "## IDEs" -> target category! Set in_category=1, print line + blank
3. See "* IntelliJ" -> compare "notepad++" < "intellij"? No -> print line
4. See "* Visual Studio" -> compare "notepad++" < "visual studio"? Yes!
   -> Print new_entry FIRST, set added=1, then print current line
5. See "## Networking" -> new section, print line
6. END -> added=1, so don't print again

Result:
## IDEs
* IntelliJ - Java IDE
* Notepad++ - Text editor      <-- Inserted here
* Visual Studio - Microsoft IDE
</syntaxhighlight>

=== Key awk Patterns ===
<syntaxhighlight lang="awk">
# Detect category headers
/^## / {
  # If leaving target category without adding, add at end
  if (in_category && !added) {
    print new_entry
    added=1
  }
  # Check if this is our target category
  in_category = ($0 ~ "^## " category)
  # ...
}

# Compare list entries (case-insensitive)
# substr(new_entry, 3) removes "* " prefix
# tolower() for case-insensitive comparison
if (!added && tolower(substr(new_entry, 3)) < tolower(substr($0, 3))) {
  print new_entry  # Insert before current line
  added=1
}
print  # Always print current line
</syntaxhighlight>

=== Edge Cases ===
<syntaxhighlight lang="bash">
# Entry should be first in category (alphabetically earliest)
# -> Comparison triggers on first list item

# Entry should be last in category (alphabetically latest)
# -> END block handles: if (in_category && !added) print new_entry

# Category is last in file
# -> END block still works correctly
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_Alphabetical_Insertion]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu]]
