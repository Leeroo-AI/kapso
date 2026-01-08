# Environment: GitHub_Issues

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|0PandaDEV/awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Issue Forms|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::User_Interface]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==
GitHub Issue Forms feature for structured user input via YAML-defined templates.

=== Description ===
This environment provides the GitHub Issue Forms feature, which enables repositories to define structured submission forms using YAML templates. Unlike Markdown-based issue templates, Issue Forms create interactive web forms with input validation, dropdowns, checkboxes, and required fields. The submitted data is formatted into a structured issue body that can be parsed programmatically.

=== Usage ===
Use this environment for any **user-facing submission workflow** that requires structured data input. It is the mandatory prerequisite for the App Submission workflow, allowing users to submit new applications via a guided form interface rather than freeform text.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| Platform || GitHub.com || Issue Forms not available on GitHub Enterprise Server (GHES) < 3.6
|-
| Repository || Public or Private || Issue Forms work on both, but public repos get more visibility
|-
| Browser || Modern browser || Required for form rendering (Chrome, Firefox, Safari, Edge)
|}

== Dependencies ==

=== GitHub Features ===
* Issue Forms (YAML-based templates in `.github/ISSUE_TEMPLATE/`)
* Labels (auto-applied via template)
* Assignees (auto-assigned via template)

=== Template Components ===
* `input` - Single-line text input
* `textarea` - Multi-line text input
* `dropdown` - Selection from predefined options
* `checkboxes` - Multiple choice options
* `markdown` - Informational text blocks

== Credentials ==

No credentials required for form submission. Users must be:
* Logged in to GitHub
* Have permission to create issues in the repository

== Quick Install ==

<syntaxhighlight lang="bash">
# No installation required
# Issue Forms are automatically rendered when:
# 1. Template file exists at .github/ISSUE_TEMPLATE/*.yml
# 2. User clicks "New Issue" button
# 3. GitHub serves the form interface
</syntaxhighlight>

== Code Evidence ==

Form definition from `add_app.yml:1-7`:
<syntaxhighlight lang="yaml">
name: Add Application
description: Suggest an application to be added to the Awesome Windows list
labels: ["Add"]
title: "[ADD] "
assignees:
  - 0pandadev
body:
</syntaxhighlight>

Input field definition from `add_app.yml:14-21`:
<syntaxhighlight lang="yaml">
- type: input
  id: app-name
  attributes:
    label: Application Name
    description: What is the name of the application?
    placeholder: e.g., Awesome App
  validations:
    required: true
</syntaxhighlight>

Dropdown field definition from `add_app.yml:32-73`:
<syntaxhighlight lang="yaml">
- type: dropdown
  id: app-category
  attributes:
    label: Category
    description: Which category does this application belong to?
    options:
      - API Development
      - Application Launchers
      - Audio
      # ... 30+ categories
  validations:
    required: true
</syntaxhighlight>

Checkbox field definition from `add_app.yml:84-92`:
<syntaxhighlight lang="yaml">
- type: checkboxes
  id: app-attributes
  attributes:
    label: Application Attributes
    description: Check all that apply
    options:
      - label: Open Source
      - label: Paid
      - label: Freemium (free with paid features)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| Form not rendering || Invalid YAML syntax || Validate YAML with a linter before committing
|-
|| "Required field" blocking submission || User skipped mandatory field || Ensure all required fields are completed
|-
|| Template not appearing in "New Issue" || File not in `.github/ISSUE_TEMPLATE/` || Move file to correct directory
|}

== Compatibility Notes ==

* '''GitHub Enterprise Server:''' Issue Forms require GHES 3.6 or later
* '''Forks:''' Issue Forms are inherited from upstream repositories
* '''API Parsing:''' Issue body uses `### Field Label` headers for field identification
* '''Checkbox Parsing:''' Checked items appear as `[X]`, unchecked as `[ ]`

== Related Pages ==

* [[required_by::Implementation:0PandaDEV_Awesome_windows_add_app_form]]
