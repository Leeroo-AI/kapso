# Principle: Alphabetical_Insertion

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GNU awk|https://www.gnu.org/software/gawk/manual/gawk.html]]
* [[source::Blog|Sorting Algorithms|https://en.wikipedia.org/wiki/Sorting_algorithm]]
|-
! Domains
| [[domain::Text_Processing]], [[domain::Sorting]], [[domain::Automation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Technique for inserting items into sorted lists while maintaining alphabetical order.

=== Description ===

Alphabetical Insertion maintains sorted order when adding new items to lists. Instead of appending and re-sorting, this technique finds the correct position during a single pass through the file. For markdown lists with section headers, this requires tracking the current section and performing case-insensitive string comparison against existing entries.

The approach is more efficient than full re-sorting and preserves file structure (sections, formatting) that might be disrupted by naive sorting approaches.

=== Usage ===

Use this principle when:
- Adding entries to curated lists with maintained order
- Inserting items into categorized/sectioned documents
- Automating list maintenance with sort preservation
- Building efficient single-pass insertion algorithms

== Theoretical Basis ==

=== Algorithm Overview ===
<syntaxhighlight lang="text">
Input: File with sections, new entry, target section
Output: File with entry inserted alphabetically in target section

Algorithm:
1. Process file line by line
2. Track current section (from ## headers)
3. When in target section and at list item:
   a. Compare new entry with current entry
   b. If new < current, insert new entry first
4. Handle edge cases:
   - Entry at end of section
   - Section at end of file
</syntaxhighlight>

=== State Machine ===
<syntaxhighlight lang="text">
States:
- OUTSIDE: Not in target section
- INSIDE: In target section, entry not yet added
- DONE: Entry added, pass through remaining lines

Transitions:
- OUTSIDE -> INSIDE: Header matches target
- INSIDE -> DONE: Entry inserted (or section ends)
- Any -> OUTSIDE: Non-target header found
</syntaxhighlight>

=== Case-Insensitive Comparison ===
<syntaxhighlight lang="awk">
# Remove bullet point prefix (* ) before comparing
# Use tolower() for case-insensitive sorting

# substr($0, 3) removes "* " prefix
# tolower() normalizes case

if (tolower(substr(new_entry, 3)) < tolower(substr($0, 3))) {
  # new_entry comes before current line alphabetically
  print new_entry
  added = 1
}
</syntaxhighlight>

=== awk Implementation Pattern ===
<syntaxhighlight lang="awk">
BEGIN { in_section = 0; added = 0 }

# Section header detection
/^## / {
  # Flush pending insert if leaving section
  if (in_section && !added) {
    print new_entry
    added = 1
  }
  # Check if entering target section
  in_section = ($0 ~ "^## " category)
  print
  next
}

# List item processing (only in target section)
in_section && /^\* / {
  if (!added && tolower(substr(new_entry, 3)) < tolower(substr($0, 3))) {
    print new_entry
    added = 1
  }
  print
  next
}

# Default: pass through
{ print }

# Handle entry at end of file/section
END {
  if (in_section && !added) print new_entry
}
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_Awesome_windows_awk_insert_sorted]]
