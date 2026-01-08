# Principle: Awesome Lint Validation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Repo|awesome-lint|https://github.com/sindresorhus/awesome-lint]]
* [[source::Doc|Awesome List Guidelines|https://github.com/sindresorhus/awesome/blob/main/awesome.md]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Linting]], [[domain::Quality_Assurance]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for automated validation of awesome-list format compliance via GitHub Actions linting.

=== Description ===

Awesome Lint Validation is an automated quality gate that runs on every pull request to ensure contributions follow the awesome-list specification. It validates:

- Markdown formatting
- List structure
- Link validity
- Alphabetical ordering
- Content guidelines

The linting step protects list quality by catching formatting errors before human review, reducing maintainer burden and ensuring consistency across all entries.

=== Usage ===

This validation runs automatically on all pull requests (both manual PRs and auto-generated PRs from the Issue Template path). Contributors should ensure their entries pass linting before requesting review.

Common lint failures:
- Entries not in alphabetical order
- Missing or malformed links
- Incorrect markdown syntax
- Missing description text

== Theoretical Basis ==

'''Awesome List Specification:'''

The awesome-lint tool validates against the [sindresorhus/awesome](https://github.com/sindresorhus/awesome) specification:

<syntaxhighlight lang="text">
Requirements checked:
├── README.md exists
├── Has Awesome badge
├── Has TOC (Table of Contents)
├── All list items have descriptions
├── No trailing whitespace
├── Alphabetically sorted (within categories)
├── All links are valid (HTTP 200)
├── Proper markdown formatting
└── No duplicate entries
</syntaxhighlight>

'''CI/CD Integration:'''

<syntaxhighlight lang="text">
PR Opened/Updated
       │
       ▼
GitHub Actions Triggered
       │
       ▼
awesome-lint-action Runs
       │
       ├─ Pass → ✅ Check succeeds
       │
       └─ Fail → ❌ Check fails with details
</syntaxhighlight>

'''PR Status Checks:'''

Pull requests cannot be merged until linting passes. This ensures all merged content adheres to awesome-list standards.

== Practical Guide ==

=== Pre-Submission Checklist ===

Before submitting, verify your entry:

- [ ] Entry is in correct category section
- [ ] Entry follows format: `* [Name](url) - Description.`
- [ ] Entry is alphabetically sorted within its category
- [ ] All URLs are valid and accessible
- [ ] Description is present and ends with period
- [ ] No trailing whitespace

=== Common Lint Errors ===

{| class="wikitable"
|-
! Error !! Cause !! Fix
|-
| "Not alphabetically sorted" || Entry in wrong position || Move entry to correct alphabetical position
|-
| "Missing description" || No text after URL || Add "- Description text."
|-
| "Link is dead" || URL returns error || Update to working URL
|-
| "Trailing whitespace" || Spaces at end of line || Remove trailing spaces
|-
| "Missing period" || Description doesn't end with . || Add period at end
|}

=== Running Lint Locally ===

<syntaxhighlight lang="bash">
# Install awesome-lint
npm install -g awesome-lint

# Run against README
awesome-lint README.md
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_Awesome_Lint_Action_Execution]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Adding_Software_Entry]]
