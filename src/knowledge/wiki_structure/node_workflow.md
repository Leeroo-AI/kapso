# Node Type: Workflow

## Definition

**Role:** Recipe (High-value "Job to be Done")

A **Workflow** is an ordered sequence of Principles that delivers a high-value business outcome. It is temporal (Start → End) and represents what users actually want to accomplish.

## Purpose

- Documents end-to-end processes
- Maps the "jobs to be done" in the repository
- Shows how Principles combine to solve real problems
- Provides execution diagrams for visual understanding

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| Type | Recipe |
| Function | High-value "Job to be Done" |
| Scope | Ordered sequence of Principles |
| Nature | Temporal (Start → End) |

## The Textbook Test

Ask: "Could I write a Wikipedia article about this logic without mentioning this repository?"

- **YES** → It's a Principle (theory)
- **NO** → It's a Workflow (business process)

**Example Distinctions:**
- Principle: "AES Encryption" (static, theoretical, mechanism)
- Workflow: "Secure User Login" (temporal, operational, business process)

## Required Metadata

| Field | Description |
|-------|-------------|
| Identifier | Meaningful unique string (e.g., `Model_Inference`) |
| Repo URL | GitHub repository URL from `metadata.json` |
| Domain(s) | Up to 3 domain tags, comma-separated |
| Last Updated | Datetime in `YYYY-MM-DD HH:MM GMT` format |

## Template Structure

```mediawiki
= Workflow: {Name} =
[[Category:Workflows]]

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Identifier
|| {unique_identifier}
|-
! Repo URL
|| [{repoUrl} {repo_name}]
|-
! Domain(s)
|| {Domain_Tag_1}, {Domain_Tag_2}
|-
! Last Updated
|| {YYYY-MM-DD HH:MM GMT}
|}

== Overview ==
{Description of the end-to-end process derived from Intent.}

== Execution Steps ==
=== Step 1: {Step_Name} ===
[[step::Principle:{Step_1_Abstract}]]

{Detailed description of what happens in this step.}

=== Step 2: {Step_Name} ===
[[step::Principle:{Step_2_Abstract}]]

{Detailed description of this step.}

=== Step 3: {Step_Name} ===
[[step::Principle:{Step_3_Abstract}]]

{Detailed description of this step.}

== Execution Diagram ==
{{#mermaid:graph TD
    A[{Step_1}] --> B[{Step_2}]
    B --> C[{Step_3}]
}}

== Code References ==
'''GitHub Repository:''' [{repoUrl} {repo_name}]

'''Key Files:'''
* [{repoUrl}/blob/{branch}/path/to/file.py file.py] - Description
```

## Coverage Target

**Minimum:** 3-5 pages per repository

## Discovery Protocol

### Top-Down (Intent)
- Check "Quick Start" in README
- Look at `examples/` directory
- If it says "Finetune Llama," create `Workflow:Finetune_Llama`

### Bottom-Up (Synthesis)
- Trace the call graph
- If `main()` calls `Loader` → `Cleaner` → `Trainer`, this chain is a Workflow

## Critical Rules

1. **Abstract Steps:** Use `[[step::Principle:Data_Loading]]`, NOT "call pandas.read_csv"
2. **Executability:** Every step Principle MUST have a corresponding Implementation
3. **No Dangling Links:** Every linked Principle must have its own page

## Semantic Links

| Link Type | Target | Description |
|-----------|--------|-------------|
| `step::Principle` | Principle | A step in the workflow |

