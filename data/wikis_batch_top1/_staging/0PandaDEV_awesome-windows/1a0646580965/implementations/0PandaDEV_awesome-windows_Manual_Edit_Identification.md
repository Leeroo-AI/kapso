# Implementation: Manual Edit Identification

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::Documentation]], [[domain::Quality_Assurance]]
|-
! Last Updated
| [[last_updated::2026-01-08 14:00 GMT]]
|}

== Overview ==

Manual process for identifying and documenting required edits to existing awesome-list entries.

=== Description ===

This is a Pattern Doc describing the user-driven process of identifying what changes are needed for an existing entry. Unlike automated tools, this relies on human judgment to:
- Navigate the README and locate the target entry
- Compare current information against authoritative sources
- Document the specific changes required

=== Usage ===

Use this process when you notice an entry needs updating. Typical triggers:
- Broken link encountered
- Outdated information noticed
- Category seems incorrect
- Application status changed (e.g., went open source)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' README.md (entries to edit)
* '''Reference:''' .github/ISSUE_TEMPLATE/edit_app.yml (for required fields)

=== Process Definition ===

This is a manual process, not an API. The "signature" is the mental checklist:

<syntaxhighlight lang="text">
PROCESS: Edit Identification
INPUT:
  - Trigger (broken link, user report, routine check)
OUTPUT:
  - Application name
  - Current URL
  - Type of edit needed
  - New values for changed fields
  - Justification/evidence
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| trigger || Event || Yes || What prompted the edit (broken link, stale info, etc.)
|-
| readme_content || Text || Yes || Current README.md content
|-
| external_sources || URLs || No || Authoritative sources for verification
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| app_name || String || Name of application to edit
|-
| current_url || URL || Currently listed URL for the app
|-
| edit_type || Enum || Category of change (URL/Description/Category/Attributes/Remove)
|-
| new_values || Dict || New values for changed fields
|-
| evidence || Text || Supporting information for the change
|}

== Usage Examples ==

=== Example 1: Broken Link Detection ===

<syntaxhighlight lang="text">
1. User clicks on "Awesome App" link in README
2. Browser returns 404 error
3. User searches for "Awesome App" official site
4. Finds new URL at https://newdomain.com/awesome-app
5. Documents:
   - App Name: Awesome App
   - Current URL: https://olddomain.com/awesome-app (broken)
   - Edit Type: Update URL
   - New URL: https://newdomain.com/awesome-app
   - Evidence: Old domain expired, verified new domain is official
</syntaxhighlight>

=== Example 2: Open Source Status Change ===

<syntaxhighlight lang="text">
1. User notices app recently released source code
2. Verifies by finding GitHub repository
3. Documents:
   - App Name: Closed Source App
   - Current URL: https://example.com/app
   - Edit Type: Update Attributes
   - New Attributes: [x] Open Source
   - New Repo URL: https://github.com/example/app
   - Evidence: Official announcement tweet/blog post
</syntaxhighlight>

== Related Pages ==

=== Principle ===
* [[principle::Principle:0PandaDEV_awesome-windows_Edit_Requirement_Identification]]

=== Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Web_Environment]]
