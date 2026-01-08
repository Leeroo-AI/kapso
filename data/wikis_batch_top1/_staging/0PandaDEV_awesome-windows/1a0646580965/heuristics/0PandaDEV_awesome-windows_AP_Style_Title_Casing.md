# AP_Style_Title_Casing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|AP Stylebook|https://www.apstylebook.com/]]
* [[source::Doc|CONTRIBUTING.md|https://github.com/0PandaDEV/awesome-windows/blob/main/CONTRIBUTING.md]]
|-
! Domains
| [[domain::Documentation]], [[domain::Style_Guide]]
|-
! Last Updated
| [[last_updated::2026-01-08 00:00 GMT]]
|}

== Overview ==

Application names and list entries must follow AP (Associated Press) style title capitalization for consistent formatting.

=== Description ===

AP style title casing is a specific capitalization convention used in journalism and professional writing. In AP style:
- Capitalize the first and last word
- Capitalize all major words (nouns, verbs, adjectives, adverbs)
- Lowercase minor words (articles, conjunctions, prepositions under 4 letters)

This ensures professional, consistent presentation across the awesome list.

=== Usage ===

Use this heuristic when **writing application names or descriptions** in list entries. The format should be: `* [App Name](link) - Description.`

== The Insight (Rule of Thumb) ==

* **Action:** Apply AP style title capitalization to application names.
* **Value:** Use the helper tool at http://titlecapitalization.com for verification.
* **Trade-off:** Requires additional attention during review; automated linting may catch issues.
* **Format:** `* [Title-Cased Name](url)`

== Reasoning ==

Consistent title casing:
1. **Professional appearance:** Matches industry standards for curated lists
2. **Readability:** Properly capitalized titles are easier to scan
3. **Brand accuracy:** Some applications have specific capitalization (e.g., "GitHub" not "Github")
4. **Linting compliance:** awesome-lint validates against common capitalization errors

== Code Evidence ==

Contributing guidelines from `CONTRIBUTING.md:18`:
<syntaxhighlight lang="text">
- Use [title-casing](http://titlecapitalization.com) (AP style) in the following format: `* [List Name](link)`
</syntaxhighlight>

== Related Pages ==

=== Used By ===
This heuristic is referenced by:
* Implementation: Manual_Information_Preparation
* Implementation: Git_Fork_Edit_Workflow
* Workflow: Adding_Software_Entry
