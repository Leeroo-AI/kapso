# Implementation: GitHub Issue Forms Schema

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Issue Forms Syntax|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms]]
|-
! Domains
| [[domain::Automation]], [[domain::GitHub_Features]], [[domain::YAML_Configuration]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

External tool documentation for GitHub Issue Forms YAML schema used to structure application submissions.

=== Description ===

This is an '''External Tool Doc''' describing how awesome-windows uses GitHub's Issue Forms feature. The schema at `.github/ISSUE_TEMPLATE/add_app.yml` defines a structured form that appears when users create a new issue, replacing freeform text input with validated fields.

=== Usage ===

This schema is automatically rendered by GitHub when a user clicks "New Issue" and selects the "Add Application" template. Contributors interact with the rendered HTML form, not the YAML directly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/ISSUE_TEMPLATE/add_app.yml:L1-117

=== External Reference ===
* [https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms GitHub Issue Forms Documentation]

=== Schema Definition ===

<syntaxhighlight lang="yaml">
name: Add Application
description: Suggest an application to be added to the Awesome Windows list
labels: ["Add"]
title: "[ADD] "
assignees:
  - 0pandadev
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to suggest an application for the Awesome Windows list!
        Please fill out the form below to provide details about the application.

  - type: input
    id: app-name
    attributes:
      label: Application Name
      description: What is the name of the application?
      placeholder: e.g., Awesome App
    validations:
      required: true

  - type: input
    id: app-url
    attributes:
      label: Application URL
      description: Provide the official website or download link for the application
      placeholder: https://example.com/awesome-app
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
        # ... 31 total options
        - Other (please specify in description)
    validations:
      required: true

  - type: textarea
    id: app-description
    attributes:
      label: Description
      description: Provide a brief description of the application and its key features
      placeholder: This application is awesome because...
    validations:
      required: true

  - type: checkboxes
    id: app-attributes
    attributes:
      label: Application Attributes
      description: Check all that apply
      options:
        - label: Open Source
        - label: Paid
        - label: Freemium (free with paid features)

  - type: input
    id: repo-url
    attributes:
      label: Repository URL
      description: If the application is open source, provide the URL to its repository
      placeholder: https://github.com/example/awesome-app
    validations:
      required: false

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Form Fields) ===
{| class="wikitable"
|-
! Field ID !! Type !! Required !! Description
|-
| app-name || input || Yes || Application name (string)
|-
| app-url || input || Yes || Official URL (URL string)
|-
| app-category || dropdown || Yes || Category selection (1 of 31 options)
|-
| app-description || textarea || Yes || Feature description (multi-line text)
|-
| app-attributes || checkboxes || No || Open Source, Paid, Freemium flags
|-
| repo-url || input || No || Repository URL for OSS apps
|-
| terms || checkboxes || Yes || Code of Conduct agreement (must check)
|}

=== Outputs (Issue Created) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Issue Title || String || "[ADD] " + user-provided suffix
|-
| Issue Body || Markdown || Structured form responses in markdown format
|-
| Labels || Array || ["Add"] automatically applied
|-
| Assignees || Array || ["0pandadev"] automatically assigned
|}

== Usage Examples ==

=== Example 1: Accessing the Form ===
<syntaxhighlight lang="text">
1. Navigate to: https://github.com/0PandaDEV/awesome-windows/issues/new/choose
2. Click "Get started" next to "Add Application"
3. Form renders with all fields
4. Fill in required fields and submit
</syntaxhighlight>

=== Example 2: Resulting Issue Body ===

After submission, the issue body contains:

<syntaxhighlight lang="markdown">
### Application Name

Visual Studio Code

### Application URL

https://code.visualstudio.com

### Category

IDEs

### Description

Lightweight but powerful source code editor with built-in Git support

### Application Attributes

- [X] Open Source
- [ ] Paid
- [ ] Freemium (free with paid features)

### Repository URL

https://github.com/microsoft/vscode

### Code of Conduct

- [X] I agree to follow this project's Code of Conduct
</syntaxhighlight>

=== Example 3: Category Dropdown Options ===
<syntaxhighlight lang="yaml">
options:
  - API Development
  - Application Launchers
  - Audio
  - Backup
  - Browsers
  - Cloud Storage
  - Command Line Tools
  - Communication
  - Compression
  - Customization
  - Data Recovery
  - Databases
  - Developer Utilities
  - Email
  - File Management
  - Games
  - Graphics
  - IDEs
  - Networking
  - Office Suites
  - Productivity
  - Proxy and VPN Tools
  - Remote Access
  - Screen Capture
  - Screenshot
  - Security
  - System Utilities
  - Terminal
  - Text Editors
  - Version Control
  - Video Utilities
  - Virtualization
  - Window Management
  - Other (please specify in description)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_Issue_Template_Submission]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Web_Environment]]
