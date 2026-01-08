# Implementation: Contribution Method Decision

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Contribution Guide|https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions]]
|-
! Domains
| [[domain::Documentation]], [[domain::Decision_Making]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Pattern documentation for deciding between automated Issue Template or manual Pull Request contribution methods.

=== Description ===

This is a '''Pattern Doc''' describing the decision process contributors use when choosing how to submit their application entry. The repository documentation at CONTRIBUTING.md line 30 explicitly offers both paths.

=== Usage ===

Use this decision pattern after completing application information gathering. The choice determines which subsequent workflow step to follow (Step 3 for Issue Template, Step 4 for Manual PR).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' CONTRIBUTING.md:L28-37

=== Documented Decision Point ===

From CONTRIBUTING.md:

<syntaxhighlight lang="markdown">
## How to add to this list

Either make an issue with the template Add Application, which will then
automatically create a pull request, or make your manual changes as follows:

If you don't have a [GitHub account](https://github.com/join), make one!

1. Fork this repo.
2. Make changes under correct section in `README.md`
3. Update `Contents` (if applicable)
4. Commit and open a Pull Request
</syntaxhighlight>

=== Interface Specification ===

<syntaxhighlight lang="python">
# Decision interface (conceptual)
def select_submission_method(
    git_experience: str,        # "beginner", "intermediate", "advanced"
    needs_customization: bool,  # True if special formatting needed
    time_priority: str          # "fast", "thorough"
) -> str:
    """
    Returns: "issue_template" or "manual_pr"
    """
    if git_experience == "beginner":
        return "issue_template"
    if needs_customization:
        return "manual_pr"
    if time_priority == "fast":
        return "issue_template"
    return "manual_pr"  # Default for experienced users
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Application metadata || Dict || Yes || Completed information from Step 1
|-
| User preference || Enum || Yes || Preference for automation vs control
|-
| Git familiarity || Enum || Yes || User's comfort level with Git workflows
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| submission_method || Enum || Either "issue_template" or "manual_pr"
|-
| next_step || Int || 3 (Issue Template) or 4 (Manual PR)
|}

== Usage Examples ==

=== Example 1: Beginner Contributor ===
<syntaxhighlight lang="python">
# Scenario: First-time contributor unfamiliar with Git
decision = select_submission_method(
    git_experience="beginner",
    needs_customization=False,
    time_priority="fast"
)
# Result: "issue_template"
# Action: Proceed to Step 3 - Issue Template Submission
</syntaxhighlight>

=== Example 2: Experienced Developer ===
<syntaxhighlight lang="python">
# Scenario: Developer who wants to submit multiple apps at once
decision = select_submission_method(
    git_experience="advanced",
    needs_customization=True,  # Multiple entries in one PR
    time_priority="thorough"
)
# Result: "manual_pr"
# Action: Proceed to Step 4 - Manual PR Submission
</syntaxhighlight>

=== Example 3: Quick Submission ===
<syntaxhighlight lang="python">
# Scenario: Experienced user who just wants fastest path
decision = select_submission_method(
    git_experience="intermediate",
    needs_customization=False,
    time_priority="fast"
)
# Result: "issue_template"
# Action: Proceed to Step 3 - Issue Template Submission
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_Submission_Path_Selection]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Web_Environment]]
