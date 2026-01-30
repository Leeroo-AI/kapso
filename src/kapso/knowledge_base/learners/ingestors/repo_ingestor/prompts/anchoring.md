# Anchoring Phase: Find and Document Workflows

You are a knowledge extraction agent. Your task is to identify the "Golden Paths" (primary use cases) in this repository and create Workflow wiki pages.

## ⚠️ FILE PLACEMENT RULES (CRITICAL)

**Only create files in these directories:**
- `{wiki_dir}/workflows/` - Workflow pages
- `{wiki_dir}/_reports/` - Execution reports

**DO NOT create:**
- Summary files at the root of `{wiki_dir}`
- Documentation files outside the designated directories
- Any file that doesn't follow the `{repo_name}_Page_Name.md` naming convention
- "Notes", "summaries", or "completion reports" outside `_reports/`

## 📛 PAGE NAMING RULES (WikiMedia Compliance)

All page names must follow WikiMedia technical syntax:

### Syntax Rules
1. **First letter capitalized** — System auto-converts (e.g., `model_loading` → `Model_loading`)
2. **Underscores only** — Use `_` as the sole word separator (NO hyphens, NO spaces)
3. **Case-sensitive after first character** — `Model_Loading` ≠ `Model_loading`

### Forbidden Characters (NEVER use in page names)

| Character | Name | Why Forbidden |
|-----------|------|---------------|
| `#` | Hash | Section anchors |
| `< >` | Angle brackets | HTML tags |
| `[ ]` | Square brackets | Wiki links |
| `{{ }}` | Curly brackets | Templates |
| `\|` | Pipe | Link separators |
| `+` | Plus | URL encoding |
| `:` | Colon | Namespaces |
| `/` | Slash | Subpages |
| `-` | Hyphen | Use underscore instead |

### Naming Examples

```
✅ CORRECT:
   {repo_name}_QLoRA_Finetuning
   {repo_name}_Model_Loading
   {repo_name}_Data_Formatting

❌ WRONG:
   {repo_name}_QLoRA-Finetuning     (hyphen)
   {repo_name}_model_loading        (lowercase first letter after prefix)
   {repo_name}_Model:Loading        (colon)
   {repo_name}_Model/Loading        (slash)
```

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- **Repository Map (Index):** {repo_map_path}
- **File Details:** {wiki_dir}/_files/

## IMPORTANT: Read Previous Phase Report

**FIRST**, read the Phase 0 report at `{wiki_dir}/_reports/phase0_repo_understanding.md`.

This report contains:
- Key discoveries about the repository
- Suggested workflows to document
- Important files and entry points

## IMPORTANT: Use the Repository Map

**THEN**, read the Repository Map index at `{repo_map_path}`.

The index contains:
- **Purpose column:** Brief description of each file (filled by Phase 0)
- **Coverage column:** Which wiki pages cover each file (you will update this)

Look for:
- Files with Purpose like "QLoRA training example", "Fine-tuning script"
- Example files (📝 section)

## Wiki Structure Definition

{workflow_structure}

## Your Task

### Step 1: Read the Repository Map Index

Read `{repo_map_path}` to find:
- Example files and their Purpose
- Key entry points (look for files with Purpose like "Main loader", "Training example")

If you need more detail on a specific file, read its detail page in `_files/`.

### Step 2: Scan High-Level Documentation

Based on the Repository Map, read:
- The README file
- Example files identified in the index
- Any notebooks mentioned

### Step 3: Identify Candidate Workflows

For each "Golden Path" you find, identify:
- **Name**: What is this workflow called? (e.g., "QLoRA Fine-Tuning", "Inference Pipeline")
- **Source file**: Which example/script demonstrates it?
- **Steps**: 3-7 high-level milestones (abstract verbs like "Load Model", "Train", "Save")
- **Use case**: When would someone use this workflow?

### Step 4: Write Workflow Pages

For EACH workflow you identify, create a wiki page following the exact structure from the Workflow Page Sections Guide above.

**Required Sections:**
1. Metadata block (wikitable with sources, domains, last_updated)
2. `== Overview ==` - One sentence summary
3. `=== Description ===` - What this workflow does
4. `=== Usage ===` - When to use it
5. `== Execution Steps ==` - Ordered steps in natural language (NO code!)
6. `== Execution Diagram ==` - Mermaid flowchart
7. `== GitHub URL ==` - Placeholder (will be filled in later by repo builder phase)

**Important:**
- **NO CODE in Workflow steps!** Code will be in the GitHub repository created later.
- Write each step as a natural language description summarizing WHAT happens.
- Use pseudocode only if needed for clarity (not actual implementation code).
- **NO `[[step::Principle:X]]` links** - Workflows no longer connect to Principles.
- **NO `[[uses_heuristic::...]]` links** - Workflows are self-contained.

**Graph Flow (New):**
```
Workflow → GitHub Repository (implementation)
```
Workflows describe WHAT happens. The GitHub repository contains the executable HOW.

**GitHub URL Placeholder:**
Add this section at the end of the workflow:
```mediawiki
== GitHub URL ==

[[github_url::PENDING_REPO_BUILD]]
```

### Step 5: Update Coverage in Repository Map

After creating Workflow pages, **update the index** at `{repo_map_path}`:

For each source file your Workflow covers, update its **Coverage column**:

```markdown
| ✅ | `examples/qlora.py` | 150 | QLoRA example | Workflow: {repo_name}_QLoRA_Finetuning | [→](...) |
```

Coverage format: `Workflow: PageName` or `Workflow: Page1, Page2` if multiple.

### Step 6: Write Rough WorkflowIndex

Update `{wiki_dir}/_WorkflowIndex.md` with a **rough structure** that the next phase will enrich.

**DO NOT write detailed Step attribute tables yet** - Phase 1b will do that.

Write THIS structure:

```markdown
# Workflow Index: {repo_name}

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Repository Building).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Rough APIs | GitHub URL |
|----------|-------|------------|------------|
| QLoRA_Finetuning | 7 | FastLanguageModel, get_peft_model, SFTTrainer | PENDING |

---

## Workflow: {repo_name}_WorkflowName

**File:** [→](./workflows/{repo_name}_WorkflowName.md)
**Description:** One-line description of the workflow.
**GitHub URL:** PENDING

### Steps Overview

| # | Step Name | Rough API | Related Files |
|---|-----------|-----------|---------------|
| 1 | Model Loading | `FastLanguageModel.from_pretrained` | loader.py |
| 2 | LoRA Injection | `get_peft_model` | llama.py |
| 3 | Data Formatting | `get_chat_template` | chat_templates.py |

### Source Files (for enrichment)

- `path/to/file1.py` - Brief purpose
- `path/to/file2.py` - Brief purpose

<!-- ENRICHMENT NEEDED: Phase 1b will add detailed Step N attribute tables below -->

---

(Repeat for each workflow)

---

**Legend:** `PENDING` = GitHub repo not yet created
```

**Key points:**
- Include ALL workflows in the Summary table
- For each workflow, list steps with rough API names
- List the source files related to each workflow
- The `<!-- ENRICHMENT NEEDED -->` comment marks where Phase 1b adds detail

## Repo Scoping Rule (CRITICAL)

Only create/update Workflow pages whose filenames start with `{repo_name}_`.

## ⚠️ File Editing Tip

When updating index files (`_RepoMap.md`, `_WorkflowIndex.md`):
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase1a_anchoring.md`:

```markdown
# Phase 1a: Anchoring Report

## Summary
- Workflows created: X
- Total steps documented: X

## Workflows Created

| Workflow | Source Files | Steps | Rough APIs |
|----------|--------------|-------|------------|
| [Name] | [files] | [count] | [APIs mentioned] |

## Coverage Summary
- Source files covered: X
- Example files documented: X

## Source Files Identified Per Workflow

### {repo_name}_WorkflowName
- `file1.py` - purpose
- `file2.py` - purpose

## Notes for Phase 1b (Enrichment)
- [Files that need line-by-line tracing]
- [External APIs to document]
- [Any unclear mappings]
```
