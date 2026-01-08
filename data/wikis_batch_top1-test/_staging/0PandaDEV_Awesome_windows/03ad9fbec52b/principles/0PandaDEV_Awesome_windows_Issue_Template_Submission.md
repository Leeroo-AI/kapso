# Principle: Issue_Template_Submission

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GitHub Issue Forms|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms]]
* [[source::Doc|Issue Templates|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository]]
|-
! Domains
| [[domain::GitHub]], [[domain::Forms]], [[domain::Data_Collection]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Technique for collecting structured data from users via GitHub Issue Forms with validation and formatting.

=== Description ===

Issue Template Submission uses GitHub Issue Forms to create structured data collection interfaces within the GitHub Issues system. Unlike traditional markdown issue templates, Issue Forms provide form-based inputs with validation, dropdowns, checkboxes, and required fields. The submitted data is formatted into a parseable markdown structure with consistent headers.

This enables automation workflows to reliably extract submitted data, as the output format is predictable regardless of user input formatting.

=== Usage ===

Use this principle when:
- Collecting structured data from community contributors
- Building issue-based submission systems
- Creating request forms with validation
- Enabling automation from user input

== Theoretical Basis ==

=== Form Element Types ===
{| class="wikitable"
|-
! Type !! Purpose !! Validation
|-
| input || Single-line text || required, placeholder
|-
| textarea || Multi-line text || required, placeholder
|-
| dropdown || Single selection from options || required
|-
| checkboxes || Multiple selections || required (for specific options)
|-
| markdown || Display-only text || None
|}

=== YAML Schema Structure ===
<syntaxhighlight lang="yaml">
name: Template Name
description: Short description
labels: ["auto-label"]
title: "[PREFIX] "
assignees:
  - username
body:
  - type: input|textarea|dropdown|checkboxes|markdown
    id: field-id
    attributes:
      label: Field Label
      description: Help text
      placeholder: Example input
      options: [...]  # For dropdown/checkboxes
    validations:
      required: true|false
</syntaxhighlight>

=== Output Format ===
<syntaxhighlight lang="markdown">
### Field Label

User's input value

### Another Field

Another value

### Checkbox Field

- [X] Selected option
- [ ] Unselected option
</syntaxhighlight>

=== Parsing Pattern ===
<syntaxhighlight lang="text">
The ### headers provide anchors for parsing:
1. Find "### Field Name" line
2. Extract content until next "###" or end
3. Trim whitespace

This works reliably because Issue Forms
enforce consistent output formatting.
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_Awesome_windows_add_app_form]]
