# Implementation: Manual Information Preparation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Issue Forms|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms]]
|-
! Domains
| [[domain::Documentation]], [[domain::Community_Contribution]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Pattern documentation for preparing application metadata to contribute to the awesome-windows curated list.

=== Description ===

This is a user-driven preparation process (not an API) where contributors gather required information before submission. The structure is defined by the issue template schema and CONTRIBUTING.md guidelines.

=== Usage ===

Use this pattern when preparing to submit a new Windows application to awesome-windows. Complete all required fields before proceeding to either the Issue Template or Manual PR submission path.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''Guidelines File:''' CONTRIBUTING.md:L28-37
* '''Form Schema:''' .github/ISSUE_TEMPLATE/add_app.yml:L14-101

=== Interface Specification ===

This is a '''Pattern Doc''' - a user-defined process with the following structured fields:

<syntaxhighlight lang="yaml">
# Required Fields
app-name: string          # Application name (e.g., "Visual Studio Code")
app-url: string           # Official URL (e.g., "https://code.visualstudio.com")
app-category: enum        # One of 31 predefined categories
app-description: string   # Brief description of features

# Optional Fields
app-attributes:           # Checkboxes
  - Open Source           # Check if source code is public
  - Paid                  # Check if purchase required
  - Freemium              # Check if free with paid features
repo-url: string          # Repository URL (if open source)
</syntaxhighlight>

=== Category Options ===

<syntaxhighlight lang="text">
API Development, Application Launchers, Audio, Backup, Browsers,
Cloud Storage, Command Line Tools, Communication, Compression,
Customization, Data Recovery, Databases, Developer Utilities,
Email, File Management, Games, Graphics, IDEs, Networking,
Office Suites, Productivity, Proxy and VPN Tools, Remote Access,
Screen Capture, Screenshot, Security, System Utilities, Terminal,
Text Editors, Version Control, Video Utilities, Virtualization,
Window Management, Other
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| User knowledge || Mental model || Yes || Contributor's familiarity with the application
|-
| Application website || URL || Yes || Source for official name, URL, description
|-
| Repository (if OSS) || URL || No || GitHub/GitLab repository for open source apps
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Structured metadata || Dict || Complete application information ready for submission
|-
| Category selection || Enum || Single category from predefined list
|-
| Attribute flags || List[bool] || Open Source, Paid, Freemium checkboxes
|}

== Usage Examples ==

=== Example 1: Open Source Application ===
<syntaxhighlight lang="yaml">
# Preparing to submit VS Code
app-name: "Visual Studio Code"
app-url: "https://code.visualstudio.com"
app-category: "IDEs"
app-description: "Lightweight but powerful source code editor with built-in Git support and extensions"
app-attributes:
  - Open Source: true
  - Paid: false
  - Freemium: false
repo-url: "https://github.com/microsoft/vscode"
</syntaxhighlight>

=== Example 2: Paid Application ===
<syntaxhighlight lang="yaml">
# Preparing to submit Sublime Text
app-name: "Sublime Text"
app-url: "https://www.sublimetext.com"
app-category: "Text Editors"
app-description: "Sophisticated text editor for code, markup and prose with excellent performance"
app-attributes:
  - Open Source: false
  - Paid: true
  - Freemium: false
repo-url: ""  # Not applicable
</syntaxhighlight>

=== Example 3: Freemium Application ===
<syntaxhighlight lang="yaml">
# Preparing to submit 1Password
app-name: "1Password"
app-url: "https://1password.com"
app-category: "Security"
app-description: "Password manager and secure digital vault with team sharing features"
app-attributes:
  - Open Source: false
  - Paid: false
  - Freemium: true
repo-url: ""
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_Application_Information_Gathering]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Web_Environment]]
