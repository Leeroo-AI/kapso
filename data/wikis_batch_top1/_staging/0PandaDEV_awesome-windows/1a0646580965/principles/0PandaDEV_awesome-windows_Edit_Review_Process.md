# Principle: Edit Review Process

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Awesome List Guidelines|https://github.com/sindresorhus/awesome/blob/main/contributing.md]]
|-
! Domains
| [[domain::Quality_Assurance]], [[domain::Community_Contribution]]
|-
! Last Updated
| [[last_updated::2026-01-08 14:00 GMT]]
|}

== Overview ==

Principle for maintainer review and implementation of proposed edits to list entries.

=== Description ===

The Edit Review Process ensures quality control over changes to the curated list. Maintainers evaluate edit requests against quality guidelines, verify the accuracy of proposed changes, and implement approved edits.

Unlike the "Adding Software Entry" workflow which uses automated issue-to-PR conversion, edit requests are typically reviewed and implemented manually by maintainers due to the complexity of validating changes to existing entries.

=== Usage ===

This principle guides maintainer actions when processing edit requests:

1. '''Validate accuracy:''' Verify proposed changes are factually correct
2. '''Check compliance:''' Ensure changes maintain list quality standards
3. '''Implement or reject:''' Apply changes or close with explanation
4. '''Acknowledge contributor:''' Credit the contributor in commit message

== Theoretical Basis ==

'''Review Decision Framework:'''

{| class="wikitable"
|-
! Factor !! Accept !! Reject
|-
| Accuracy || Changes are verifiable || Unsubstantiated claims
|-
| Quality || Improves or maintains quality || Degrades entry quality
|-
| Relevance || Still fits list scope || No longer appropriate
|-
| Completeness || All required info provided || Missing key details
|}

'''Common Edit Outcomes:'''

<syntaxhighlight lang="text">
APPROVE - Changes implemented as-is
MODIFY - Changes implemented with adjustments
REQUEST_INFO - Ask for clarification/sources
REJECT - Changes not appropriate (with explanation)
</syntaxhighlight>

'''Removal Criteria:'''
- Application discontinued or abandoned
- URL consistently broken
- Application no longer meets quality standards
- Security/safety concerns
- Duplicate of better entry

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_Manual_Edit_Review]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Editing_Software_Entry]]
