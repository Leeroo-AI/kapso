# Principle: Edit Requirement Identification

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Awesome List Guidelines|https://github.com/sindresorhus/awesome/blob/main/contributing.md]]
|-
! Domains
| [[domain::Documentation]], [[domain::Quality_Assurance]]
|-
! Last Updated
| [[last_updated::2026-01-08 14:00 GMT]]
|}

== Overview ==

Principle for identifying and validating what changes are needed for an existing entry in a curated list.

=== Description ===

Edit Requirement Identification is the first step in modifying existing content in curated lists. It involves locating the current entry, understanding its current state, and determining exactly what needs to change.

This step is critical because it ensures:
- The target entry actually exists in the list
- The proposed change is necessary and valid
- The editor understands the full context before submitting changes

=== Usage ===

Apply this principle before submitting any edit request. Key tasks:

1. '''Locate the entry''' in README.md
2. '''Document current state''' (URL, description, category, attributes)
3. '''Identify specific changes needed''' (what fields to update)
4. '''Validate the change is appropriate''' (not a duplicate, meets guidelines)

== Theoretical Basis ==

'''Types of Edits:'''

{| class="wikitable"
|-
! Edit Type !! When to Use !! Examples
|-
| URL Update || Link is broken or changed || Site moved domains
|-
| Description Update || Info is stale or inaccurate || New features added
|-
| Category Change || Better category exists || Reclassify terminal emulator
|-
| Attribute Change || Licensing changed || App became open source
|-
| Removal || App no longer qualifies || Discontinued, malware, etc.
|}

'''Validation Checklist:'''

<syntaxhighlight lang="text">
1. Does the entry exist in the current list?
2. Is the proposed change factually accurate?
3. Does the changed entry still meet quality standards?
4. Would this change duplicate another entry?
5. Is there documentation/evidence for the change?
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_Manual_Edit_Identification]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Editing_Software_Entry]]
