# Principle: Application Information Gathering

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Awesome List Guidelines|https://github.com/sindresorhus/awesome/blob/main/contributing.md]]
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::Documentation]], [[domain::Community_Contribution]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for collecting and structuring application metadata before submitting to a curated awesome-list repository.

=== Description ===

Application Information Gathering is the first step in contributing to any curated list. It involves identifying all required metadata about a software application including its name, official URL, category classification, feature description, licensing model (open source/paid/freemium), and repository location if applicable.

This step ensures that contributions are complete and well-formatted before any submission attempt, reducing rejection rates and maintainer overhead.

=== Usage ===

Apply this principle when preparing to contribute a new application to the awesome-windows list. This is a prerequisite step that must be completed before choosing between the Issue Template or Manual PR submission paths.

Key questions to answer:
- What is the application's official name and URL?
- Which category does it best fit into?
- Is it open source, paid, or freemium?
- What is a concise, useful description of its features?

== Theoretical Basis ==

Curated lists follow specific conventions to maintain quality and consistency:

'''Required Fields:'''
* Application Name - Using AP-style title casing
* Application URL - Official website or download page
* Category - Must match one of the 31+ defined categories
* Description - Brief, factual summary of key features

'''Optional Fields:'''
* Repository URL - For open source applications
* Additional attributes - Open Source, Paid, Freemium flags

'''Quality Criteria:'''
* No duplicate entries
* Alphabetical ordering within categories
* Proper markdown formatting

== Practical Guide ==

=== Step 1: Identify Application Details ===
Gather the official name and URL from the application's website or store listing.

=== Step 2: Choose Category ===
Review existing categories in README.md and select the best match. Categories include:
- API Development, Application Launchers, Audio, Backup, Browsers
- Cloud Storage, Command Line Tools, Communication, Compression
- And 22+ more...

=== Step 3: Write Description ===
Write a brief, objective description focusing on:
- Primary function
- Key differentiating features
- Target use case

=== Step 4: Check Licensing ===
Determine if the application is:
- Open Source (has public repository)
- Paid (requires purchase)
- Freemium (free with paid features)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_Manual_Information_Preparation]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Adding_Software_Entry]]
