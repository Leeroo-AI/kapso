# Alphabetical_Ordering_Convention

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|CONTRIBUTING.md|https://github.com/0PandaDEV/awesome-windows/blob/main/CONTRIBUTING.md]]
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::Documentation]], [[domain::Style_Guide]]
|-
! Last Updated
| [[last_updated::2026-01-08 00:00 GMT]]
|}

== Overview ==

Entries in awesome lists must be inserted alphabetically within their category to maintain consistent organization and searchability.

=== Description ===

When adding a new application to an awesome list, the entry must be placed in alphabetical order within the relevant category section. This convention ensures consistent organization across all categories and makes it easier for users to locate specific applications. The alphabetical sorting is case-insensitive and based on the display name of the application (not the URL).

=== Usage ===

Use this heuristic when **adding a new entry** to any category in the README.md file. Both manual PR submissions and automated issue-to-PR conversions must respect this ordering convention.

== The Insight (Rule of Thumb) ==

* **Action:** Insert new entries in alphabetical order within the target category.
* **Value:** Case-insensitive sorting by application name.
* **Trade-off:** None - this is a mandatory style guideline.
* **Implementation:** The automated PR conversion uses `awk` with `tolower()` for case-insensitive comparison.

== Reasoning ==

Alphabetical ordering provides:
1. **Consistency:** All categories follow the same organizational pattern
2. **Discoverability:** Users can quickly scan for applications
3. **Maintainability:** PRs are easier to review when placement is deterministic
4. **Conflict reduction:** Alphabetical insertion minimizes merge conflicts compared to appending

== Code Evidence ==

Contributing guidelines from `CONTRIBUTING.md:19`:
<syntaxhighlight lang="text">
- Link additions should be added in alphabetical order in the relevant category.
</syntaxhighlight>

Automated alphabetical insertion from `.github/workflows/covert_to_pr.yml:87-91`:
<syntaxhighlight lang="awk">
          in_category && /^\* / {
            if (!added && tolower(substr(new_entry, 3)) < tolower(substr($0, 3))) {
              print new_entry
              added=1
            }
</syntaxhighlight>

== Related Pages ==

=== Used By ===
This heuristic is referenced by:
* Implementation: Git_Fork_Edit_Workflow
* Implementation: Issue_To_PR_Conversion
* Workflow: Adding_Software_Entry
