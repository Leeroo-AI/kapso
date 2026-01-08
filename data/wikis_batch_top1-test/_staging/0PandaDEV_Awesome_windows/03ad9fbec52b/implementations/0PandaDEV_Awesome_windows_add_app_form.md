# Implementation: Add_app_form

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Issue Forms|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms]]
|-
! Domains
| [[domain::GitHub_Issues]], [[domain::Forms]], [[domain::Data_Collection]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Pattern document defining the structured issue form for application submissions to the Awesome Windows list.

=== Description ===

The `add_app.yml` issue template defines a GitHub Issue Form that collects structured data for new application submissions. It uses GitHub's YAML-based form syntax to create input fields, dropdowns, textareas, and checkboxes. The form automatically assigns the "Add" label to submissions, enabling workflow automation triggers.

=== Usage ===

Use this form schema when creating structured issue templates that need to collect validated, parseable data. This pattern is essential for issue-to-PR automation workflows where field values must be extracted programmatically.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/ISSUE_TEMPLATE/add_app.yml
* '''Lines:''' L1-117

=== Signature ===
<syntaxhighlight lang="yaml">
name: Add Application
description: Suggest an application to be added to the Awesome Windows list
labels: ["Add"]
title: "[ADD] "
assignees:
  - 0pandadev
body:
  - type: input          # Application Name (required)
  - type: input          # Application URL (required)
  - type: dropdown       # Category (required, 32 options)
  - type: textarea       # Description (required)
  - type: checkboxes     # Attributes: Open Source, Paid, Freemium
  - type: input          # Repository URL (optional)
  - type: textarea       # Additional Information (optional)
  - type: checkboxes     # Code of Conduct agreement (required)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="text">
# GitHub automatically discovers templates in .github/ISSUE_TEMPLATE/
# No explicit import needed - place YAML file in correct directory
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| app-name || input || Yes || Application name text field
|-
| app-url || input || Yes || Official website or download URL
|-
| app-category || dropdown || Yes || Category selection from 32 predefined options
|-
| app-description || textarea || Yes || Brief description of application features
|-
| app-attributes || checkboxes || No || Open Source, Paid, Freemium flags
|-
| repo-url || input || No || Source repository URL if open source
|-
| additional-info || textarea || No || Extra information (requirements, version, etc.)
|-
| terms || checkboxes || Yes || Code of Conduct agreement
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| GitHub Issue || Issue || Created with "Add" label and structured body
|-
| Issue Body || markdown || Formatted with `### Field Name` headers for parsing
|}

== Usage Examples ==

=== Form Field Definition ===
<syntaxhighlight lang="yaml">
- type: input
  id: app-name
  attributes:
    label: Application Name
    description: What is the name of the application?
    placeholder: e.g., Awesome App
  validations:
    required: true

- type: dropdown
  id: app-category
  attributes:
    label: Category
    description: Which category does this application belong to?
    options:
      - API Development
      - Application Launchers
      - Audio
      # ... 29 more categories
      - Other (please specify in description)
  validations:
    required: true
</syntaxhighlight>

=== Generated Issue Body Format ===
<syntaxhighlight lang="markdown">
### Application Name

Visual Studio Code

### Application URL

https://code.visualstudio.com

### Category

IDEs

### Description

Free source-code editor made by Microsoft with support for debugging, syntax highlighting, and Git integration.

### Application Attributes

- [X] Open Source
- [ ] Paid
- [ ] Freemium (free with paid features)

### Repository URL

https://github.com/microsoft/vscode

### Additional Information

_No response_

### Code of Conduct

- [X] I agree to follow this project's Code of Conduct
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_Issue_Template_Submission]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Issues]]
