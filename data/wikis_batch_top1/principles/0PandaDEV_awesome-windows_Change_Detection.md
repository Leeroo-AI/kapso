# Principle: Change Detection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Python String Methods|https://docs.python.org/3/library/stdtypes.html#string-methods]]
|-
! Domains
| [[domain::Automation]], [[domain::Optimization]], [[domain::Idempotency]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for detecting whether source data has changed before triggering expensive update operations.

=== Description ===

Change Detection is a fundamental optimization pattern that avoids unnecessary work by comparing current state to desired state. In the contributor update workflow, this means checking whether any contributors are missing from the README before performing updates.

Benefits:
- '''Efficiency:''' Skip no-op updates
- '''Idempotency:''' Safe to run repeatedly
- '''Reduced Noise:''' No empty commits
- '''Resource Conservation:''' Minimize CI/CD usage

=== Usage ===

Apply change detection when:
- Operations are expensive (file I/O, API calls, network)
- Updates should be idempotent
- You want to avoid empty commits
- The workflow runs frequently (daily, hourly)

== Theoretical Basis ==

'''Change Detection Strategies:'''

{| class="wikitable"
|-
! Strategy !! Description !! Use Case
|-
| Content Comparison || Compare old vs new content || Small files, text
|-
| Hash Comparison || Compare file hashes || Large files, binary
|-
| Timestamp Check || Compare modification times || File systems
|-
| Membership Check || Check if items exist || Lists, sets
|}

'''Membership Check Pattern:'''

For the contributor update workflow, we use membership checking:

<syntaxhighlight lang="text">
For each contributor in API response:
    If contributor's profile URL not in README:
        Return True (change detected)
Return False (no changes)
</syntaxhighlight>

This is efficient because:
- String `in` check is O(n) for README length
- Total complexity: O(contributors × README_length)
- README is small (~500 lines), so this is fast

'''Why Not Hash Comparison?'''

We can't use hash comparison because:
- We don't know the exact format of the new content yet
- We need to detect missing contributors, not content differences
- The README contains other content besides contributors

== Practical Guide ==

=== Implementation Pattern ===

<syntaxhighlight lang="python">
def has_changes(current_items, target_content):
    """
    Check if any current items are missing from target content.

    Args:
        current_items: List of items that should be present
        target_content: String content to search in

    Returns:
        bool: True if any item is missing, False if all present
    """
    for item in current_items:
        if item not in target_content:
            return True
    return False
</syntaxhighlight>

=== Short-Circuit Optimization ===

The function returns `True` immediately upon finding the first missing item, avoiding unnecessary checks:

<syntaxhighlight lang="python">
# Efficient: stops at first missing item
for contributor in contributors:
    if contributor not in readme:
        return True  # Early exit
return False

# Inefficient: checks all items even if first is missing
missing = [c for c in contributors if c not in readme]
return len(missing) > 0
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_has_contributors_changed_Function]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Automated_Contributor_Update]]
