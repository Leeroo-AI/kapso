# Principle: Edit Request Submission

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Issue Forms|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository]]
|-
! Domains
| [[domain::Documentation]], [[domain::GitHub]], [[domain::Community_Contribution]]
|-
! Last Updated
| [[last_updated::2026-01-08 14:00 GMT]]
|}

== Overview ==

Principle for submitting structured edit requests using GitHub Issue Forms to propose changes to existing list entries.

=== Description ===

Edit Request Submission uses GitHub Issue Forms to collect structured information about proposed changes. Unlike free-form issues, the form enforces required fields and provides dropdown selections for common options.

Benefits of form-based submission:
- '''Completeness:''' Required fields ensure all necessary info is provided
- '''Consistency:''' Standard format makes review easier
- '''Categorization:''' Automatic labeling with "Edit" tag
- '''Assignment:''' Auto-assigns to maintainer

=== Usage ===

Use the "Edit Application" issue template when:
- You need to propose changes to an existing entry
- You have identified the specific changes required
- You have evidence/sources for the proposed changes

Do NOT use when:
- Adding a new application (use "Add Application" instead)
- Reporting a bug in the repository itself
- General questions or discussions

== Theoretical Basis ==

'''Edit Form Structure:'''

The form collects information in logical groups:

{| class="wikitable"
|-
! Section !! Fields !! Purpose
|-
| Identification || App name, Current URL || Locate the entry
|-
| Edit Type || Dropdown selection || Categorize the change
|-
| Proposed Changes || Description, New values || Specify what to change
|-
| Evidence || Additional info || Support the change
|-
| Agreement || Code of conduct checkbox || Ensure compliance
|}

'''Edit Type Options:'''
<syntaxhighlight lang="text">
- Update URL
- Update Description
- Update Category (with dropdown for 31+ categories)
- Update Attributes (checkboxes: Open Source, Paid, Freemium)
- Remove Application
- Other (specify in description)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_GitHub_Edit_Form_Schema]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Editing_Software_Entry]]
