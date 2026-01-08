# Principle: PR Review Process

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Repo|create-pull-request|https://github.com/peter-evans/create-pull-request]]
* [[source::Doc|GitHub Actions|https://docs.github.com/en/actions]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Automation]], [[domain::Code_Review]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for maintainer-driven review and automated conversion of issue submissions to pull requests.

=== Description ===

PR Review Process is the final step in the contribution workflow where the maintainer reviews submissions and merges approved content. For Issue Template submissions, this includes a unique automation: the maintainer can trigger automatic PR creation by commenting `/convert` on an issue.

The automation:
1. Parses issue body to extract application metadata
2. Generates the correct README.md entry
3. Creates a pull request with proper formatting
4. Closes the original issue with a reference to the PR

This allows maintainers to accept community suggestions with minimal manual editing.

=== Usage ===

This step applies differently based on submission path:

'''For Manual PRs (Step 4):'''
- Standard code review process
- Maintainer reviews diff and formatting
- Requests changes if needed
- Merges when lint passes and content is appropriate

'''For Issue Template Submissions (Step 3):'''
- Maintainer reviews issue content
- If approved, comments `/convert` to trigger automation
- Automation creates PR from issue metadata
- PR goes through standard lint validation
- Maintainer merges the auto-generated PR

== Theoretical Basis ==

'''Issue-to-PR Conversion Flow:'''

<syntaxhighlight lang="text">
Issue Created                    Maintainer Reviews
[ADD] New App                         │
   │                                  │
   ▼                                  ▼
Labels: ["Add"]              /convert Comment
Assigned: 0pandadev                   │
                                      ▼
                          covert_to_pr.yml Triggered
                                      │
                            ┌─────────┴─────────┐
                            ▼                   ▼
                       Parse Issue          Update README
                       (github-script)      (AWK script)
                            │                   │
                            └────────┬──────────┘
                                     ▼
                            Create Pull Request
                            (peter-evans/create-pull-request)
                                     │
                                     ▼
                              Close Issue
                              (github-script)
</syntaxhighlight>

'''Trigger Conditions:'''
- Event: `issue_comment` with body `/convert`
- Actor: Must be user `0pandadev`
- Label: Issue must have "Add" label
- Alternative: `workflow_dispatch` with issue number input

'''Entry Generation Logic:'''

The AWK script in covert_to_pr.yml:
1. Extracts fields from issue body using pattern matching
2. Determines OSS/paid icons from checkboxes
3. Finds correct category section in README.md
4. Inserts entry in alphabetical order

== Practical Guide ==

=== For Contributors (Waiting for Review) ===

After submitting via Issue Template:
1. Wait for maintainer to review your issue
2. Respond to any questions or feedback
3. Maintainer will comment `/convert` if approved
4. PR is automatically created
5. Lint must pass before merge

=== For Manual PR Contributors ===

After submitting your PR:
1. Ensure lint checks pass
2. Address any review feedback
3. Update PR if changes requested
4. Wait for maintainer approval and merge

=== For Maintainers ===

Reviewing Issue Template submissions:
1. Verify application is appropriate for list
2. Check URL validity
3. Verify category selection
4. Comment `/convert` to trigger automation
5. Review generated PR
6. Merge if lint passes

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_Issue_To_PR_Conversion]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Adding_Software_Entry]]
