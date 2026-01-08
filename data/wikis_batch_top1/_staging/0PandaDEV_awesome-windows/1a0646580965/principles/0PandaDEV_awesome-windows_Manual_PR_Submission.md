# Principle: Manual PR Submission

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Fork Workflow|https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork]]
|-
! Domains
| [[domain::Git]], [[domain::Community_Contribution]], [[domain::Version_Control]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for contributing application entries through the traditional fork-edit-commit-pull-request Git workflow.

=== Description ===

Manual PR Submission is the traditional open-source contribution method where contributors fork the repository, make direct edits to README.md, and submit a pull request. This provides full control over formatting and allows for complex or batch contributions.

Unlike the Issue Template path, Manual PR requires:
- Git/GitHub proficiency
- Understanding of markdown formatting
- Knowledge of alphabetical ordering within categories
- Familiarity with AP-style title casing

=== Usage ===

Use Manual PR submission when:
- You need custom formatting not supported by the issue template
- You want to submit multiple applications in one PR
- You prefer direct control over your contribution
- You're comfortable with Git fork workflows

This path is recommended for experienced contributors and developers.

== Theoretical Basis ==

'''Fork-Based Workflow:'''

<syntaxhighlight lang="text">
Upstream Repo                    Your Fork
     │                               │
     │    ┌─── Fork ────────────────►│
     │    │                          │
     │    │                          ▼
     │    │                     Edit README.md
     │    │                          │
     │    │                          ▼
     │    │                     Commit Changes
     │    │                          │
     │◄───┼─── Pull Request ◄────────┘
     │    │
     ▼    │
  Merge ──┘
</syntaxhighlight>

'''Entry Format Requirements:'''

From CONTRIBUTING.md:
- Use title-casing (AP style)
- Format: `* [List Name](link) - Description`
- Add in alphabetical order within category
- Use correct icons for OSS/paid status

'''Markdown Entry Structure:'''
<syntaxhighlight lang="markdown">
* [App Name](https://url.com) - Brief description. [![Open-Source Software][oss]](repo-url) ![paid]
</syntaxhighlight>

== Practical Guide ==

=== Step 1: Fork Repository ===
Click "Fork" on the awesome-windows repository page.

=== Step 2: Edit README.md ===
Locate the appropriate category section and add your entry:

<syntaxhighlight lang="markdown">
## IDEs

* [Android Studio](https://developer.android.com/studio) - IDE for Android development. [![Open-Source Software][oss]](https://android.googlesource.com/platform/tools/adt/idea/)
* [Visual Studio Code](https://code.visualstudio.com/) - Lightweight but powerful editor. [![Open-Source Software][oss]](https://github.com/microsoft/vscode)
* [YOUR APP HERE]  ← Insert alphabetically
</syntaxhighlight>

=== Step 3: Verify Formatting ===
- [ ] Entry uses AP-style title casing
- [ ] URL is official website or download link
- [ ] Description is concise (one sentence)
- [ ] Entry is in alphabetical order
- [ ] Icons are correct (oss for open source, paid for paid)

=== Step 4: Commit and PR ===
1. Commit with descriptive message: "Add [App Name] to [Category]"
2. Open Pull Request against main branch
3. Fill in PR description with application details

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_Git_Fork_Edit_Workflow]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Adding_Software_Entry]]
