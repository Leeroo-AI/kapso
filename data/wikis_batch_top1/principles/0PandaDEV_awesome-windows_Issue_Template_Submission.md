# Principle: Issue Template Submission

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Issue Forms|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms]]
|-
! Domains
| [[domain::Automation]], [[domain::Community_Contribution]], [[domain::GitHub_Features]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for submitting structured application entries via GitHub Issue Forms with automated validation and PR conversion.

=== Description ===

Issue Template Submission leverages GitHub's Issue Forms feature to provide a structured, validated submission experience. Contributors fill out a YAML-defined form instead of writing freeform text, ensuring all required information is captured in a consistent format.

The form schema enforces:
- Required fields (name, URL, category, description)
- Dropdown selection for categories (preventing typos)
- Checkbox inputs for attributes
- Code of Conduct acknowledgment

Upon submission, the issue is labeled "[ADD]" and assigned to the maintainer, who can trigger automated PR creation via the /convert command.

=== Usage ===

Use Issue Template submission when:
- You want the simplest, guided submission process
- You prefer form-based input over markdown editing
- Your submission fits standard categories and formatting
- You want automated PR creation

This path is recommended for first-time contributors and those unfamiliar with Git workflows.

== Theoretical Basis ==

'''GitHub Issue Forms Architecture:'''

Issue Forms use YAML schema to define:
- '''input''' - Single-line text fields
- '''dropdown''' - Enumerated selections
- '''textarea''' - Multi-line text
- '''checkboxes''' - Boolean flags
- '''markdown''' - Static instructional text

'''Schema Validation:'''
- `required: true` fields must be completed
- Dropdown options constrain input to valid values
- Checkboxes for Code of Conduct require explicit agreement

'''Automation Chain:'''
<syntaxhighlight lang="text">
Issue Submitted → [ADD] Label Applied → Maintainer Reviews
                                              ↓
                                    /convert Command
                                              ↓
                                   covert_to_pr.yml Triggered
                                              ↓
                                    Automated PR Created
</syntaxhighlight>

== Practical Guide ==

=== Step-by-Step Process ===

1. Navigate to repository Issues tab
2. Click "New Issue"
3. Select "Add Application" template
4. Complete all required fields:
   - Application Name
   - Application URL
   - Category (dropdown)
   - Description
   - Attributes (checkboxes)
   - Code of Conduct agreement
5. Submit issue
6. Wait for maintainer to trigger PR creation

=== Field Requirements ===

{| class="wikitable"
|-
! Field !! Type !! Required !! Validation
|-
| Application Name || input || Yes || Non-empty string
|-
| Application URL || input || Yes || Valid URL format
|-
| Category || dropdown || Yes || One of 31 predefined options
|-
| Description || textarea || Yes || Non-empty text
|-
| Attributes || checkboxes || No || Multiple selection allowed
|-
| Repository URL || input || No || Valid URL if provided
|-
| Code of Conduct || checkboxes || Yes || Must check agreement
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_GitHub_Issue_Forms_Schema]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Adding_Software_Entry]]
