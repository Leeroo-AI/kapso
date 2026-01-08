# Implementation: Manual Edit Review

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Awesome List Guidelines|https://github.com/sindresorhus/awesome/blob/main/contributing.md]]
|-
! Domains
| [[domain::Quality_Assurance]], [[domain::Documentation]]
|-
! Last Updated
| [[last_updated::2026-01-08 14:00 GMT]]
|}

== Overview ==

Manual maintainer process for reviewing and implementing edit requests to list entries.

=== Description ===

This is a Pattern Doc describing the maintainer-driven review process for edit requests. Unlike the "Adding Software Entry" workflow which has automated issue-to-PR conversion, edits require manual review because:
- Changes to existing content need validation
- Removals require judgment calls
- Category changes may affect ordering
- Multiple approaches may be valid

=== Usage ===

Maintainers execute this process when:
- A new issue with "Edit" label appears
- The issue was created via the edit_app.yml template
- Review is needed to validate and implement changes

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' Manual process (no specific file)
* '''Reference:''' CONTRIBUTING.md (guidelines), README.md (target file)

=== Process Definition ===

<syntaxhighlight lang="text">
PROCESS: Edit Review
INPUT:
  - GitHub Issue with "Edit" label
  - Form responses (app name, edit type, new values)
OUTPUT:
  - Issue closed with resolution comment
  - (If approved) README.md updated via commit
  - (If rejected) Explanation provided
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| edit_issue || GitHub Issue || Yes || Issue containing edit request
|-
| readme_content || Text || Yes || Current README.md
|-
| guidelines || Text || Yes || CONTRIBUTING.md and awesome-list rules
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| decision || Enum || APPROVE, MODIFY, REQUEST_INFO, REJECT
|-
| updated_readme || Text || Modified README (if approved)
|-
| commit_message || String || Description of changes made
|-
| issue_comment || Text || Resolution message to requester
|}

== Usage Examples ==

=== Example 1: Approved URL Update ===

<syntaxhighlight lang="text">
INPUT:
  Issue #42: "[EDIT] PowerToys"
  Edit Type: Update URL
  New URL: https://github.com/microsoft/PowerToys

REVIEW STEPS:
1. Open the old URL - verify it's broken/redirects
2. Open the new URL - verify it's the official source
3. Check that Microsoft owns the repository
4. Find "PowerToys" entry in README.md

DECISION: APPROVE

IMPLEMENTATION:
1. Edit README.md
2. Replace old URL with new URL
3. Maintain alphabetical order (no change needed)
4. Commit: "Update PowerToys URL (#42)"
5. Close issue with thank you comment
</syntaxhighlight>

=== Example 2: Rejected Removal Request ===

<syntaxhighlight lang="text">
INPUT:
  Issue #43: "[EDIT] SomeApp"
  Edit Type: Remove Application
  Reason: "I don't like this app"

REVIEW STEPS:
1. Check if SomeApp meets removal criteria
2. Verify app is still functional and maintained
3. Check if it still meets awesome-list quality bar

DECISION: REJECT

RESPONSE:
"Thank you for the suggestion, but personal preference isn't
a sufficient reason for removal. SomeApp is still actively
maintained and meets our quality guidelines.

Removals are typically for:
- Discontinued/abandoned apps
- Consistently broken/unavailable
- Security/safety concerns
- No longer meeting quality standards

If you have specific concerns about the app, please
provide more details."
</syntaxhighlight>

== Related Pages ==

=== Principle ===
* [[principle::Principle:0PandaDEV_awesome-windows_Edit_Review_Process]]

=== Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Web_Environment]]
