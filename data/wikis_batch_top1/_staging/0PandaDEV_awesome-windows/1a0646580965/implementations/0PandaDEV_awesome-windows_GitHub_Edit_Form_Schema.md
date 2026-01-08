# Implementation: GitHub Edit Form Schema

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Issue Forms|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository]]
|-
! Domains
| [[domain::GitHub]], [[domain::YAML]], [[domain::Forms]]
|-
! Last Updated
| [[last_updated::2026-01-08 14:00 GMT]]
|}

== Overview ==

YAML schema defining the "Edit Application" GitHub Issue Form template for structured edit requests.

=== Description ===

This External Tool Doc describes the GitHub Issue Forms YAML configuration that renders a structured form for edit requests. The form collects all information needed to process edits: application identification, edit type, proposed changes, and supporting evidence.

The form uses GitHub's Issue Forms syntax with input fields, dropdowns, textareas, and checkboxes to guide users through the edit submission process.

=== Usage ===

Users access this form by:
1. Navigating to the repository's Issues tab
2. Clicking "New Issue"
3. Selecting "Edit Application" template
4. Filling out the required fields
5. Submitting the issue

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/ISSUE_TEMPLATE/edit_app.yml
* '''Lines:''' L1-137

=== Schema ===

<syntaxhighlight lang="yaml">
name: Edit Application
description: Suggest an edit to an existing application in the Awesome Windows list
labels: ["Edit"]
title: "[EDIT] "
assignees:
  - 0pandadev
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to suggest an edit...

  - type: input
    id: app-name
    attributes:
      label: Application Name
      description: What is the name of the application you want to edit?
      placeholder: e.g., Awesome App
    validations:
      required: true

  - type: input
    id: app-url
    attributes:
      label: Current Application URL
      description: Provide the current URL listed for the application
    validations:
      required: true

  - type: dropdown
    id: edit-type
    attributes:
      label: Type of Edit
      options:
        - Update URL
        - Update Description
        - Update Category
        - Update Attributes (Free/Open Source/Paid/Freemium)
        - Remove Application
        - Other (please specify in description)
    validations:
      required: true

  # ... additional fields for new values, categories, checkboxes
</syntaxhighlight>

=== Import ===

This is not a code import; it's a YAML file parsed by GitHub's Issue Forms engine.

== I/O Contract ==

=== Inputs (Form Fields) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| app-name || input || Yes || Name of application to edit
|-
| app-url || input || Yes || Current URL listed
|-
| edit-type || dropdown || Yes || Type of edit (6 options)
|-
| edit-description || textarea || Yes || Detailed description of changes
|-
| new-url || input || No || New URL if updating URL
|-
| new-category || dropdown || No || New category if recategorizing
|-
| new-attributes || checkboxes || No || Updated attributes
|-
| new-repo-url || input || No || Repository URL for open source
|-
| additional-info || textarea || No || Supporting evidence
|-
| terms || checkbox || Yes || Code of conduct agreement
|}

=== Outputs (Issue Created) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| issue || GitHub Issue || Created issue with "Edit" label
|-
| title || String || "[EDIT] " + app name
|-
| assignee || User || Auto-assigned to @0pandadev
|-
| body || Markdown || Rendered form responses
|}

== Usage Examples ==

=== Example: URL Update Request ===

When a user fills out the form to update a broken URL:

<syntaxhighlight lang="yaml">
# User input
app-name: "PowerToys"
app-url: "https://oldurl.com/powertoys"
edit-type: "Update URL"
edit-description: "The official URL has moved to Microsoft's GitHub"
new-url: "https://github.com/microsoft/PowerToys"
terms: true

# Resulting issue
Title: "[EDIT] PowerToys"
Labels: ["Edit"]
Assignee: @0pandadev
Body: |
  ### Application Name
  PowerToys

  ### Current Application URL
  https://oldurl.com/powertoys

  ### Type of Edit
  Update URL

  ### Edit Description
  The official URL has moved to Microsoft's GitHub

  ### New URL (if applicable)
  https://github.com/microsoft/PowerToys
</syntaxhighlight>

== Related Pages ==

=== Principle ===
* [[principle::Principle:0PandaDEV_awesome-windows_Edit_Request_Submission]]

=== Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Web_Environment]]
