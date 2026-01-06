# Anchoring Phase: Find and Document Workflows

You are a knowledge extraction agent. Your task is to identify the "Golden Paths" (primary use cases) in this repository and create Workflow wiki pages.

## ⚠️ FILE PLACEMENT RULES (CRITICAL)

**Only create files in these directories:**
- `{wiki_dir}/workflows/` - Workflow pages
- `{wiki_dir}/_reports/` - Execution reports

**DO NOT create:**
- Summary files at the root of `{wiki_dir}`
- Documentation files outside the designated directories
- Any file that doesn't follow the `{repo_name}_PageName.md` naming convention
- "Notes", "summaries", or "completion reports" outside `_reports/`

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

## About Status Tags in Connections

When you reference pages in the Connections column:
- Use `⬜Impl:Name` for Implementations that DON'T exist yet (later phases will create them)
- Use `⬜Principle:Name` for Principles that DON'T exist yet
- Use `✅Type:Name` only if that page already exists

Since you're the first phase creating wiki pages, most references will be `⬜` (pending creation by later phases).

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
5. `== Execution Steps ==` - Ordered steps with `[[step::Principle:{repo_name}_X]]` links
6. `== Execution Diagram ==` - Mermaid flowchart
7. `== Related Pages ==` - Step links and heuristic links

**Important:**
- **NO CODE in Workflow steps!** Code belongs in Implementation pages (linked through Principles).
- Write each step as a natural language description summarizing WHAT happens.
- Use pseudocode only if needed for clarity (not actual implementation code).
- For the `[[step::Principle:{repo_name}_X]]` links, use descriptive names matching the concept.
- Do NOT add `[[uses_heuristic::...]]` links yet - those come later.

**Graph Flow:**
```
Workflow Step → Principle (theory) → Implementation (actual code)
```
Workflows describe WHAT. Principles explain WHY. Implementations show HOW with real code.

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
> This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Principles | Rough APIs |
|----------|-------|------------|------------|
| QLoRA_Finetuning | 7 | 7 | FastLanguageModel, get_peft_model, SFTTrainer |

---

## Workflow: {repo_name}_WorkflowName

**File:** [→](./workflows/{repo_name}_WorkflowName.md)
**Description:** One-line description of the workflow.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Model Loading | Model_Loading | `FastLanguageModel.from_pretrained` | loader.py |
| 2 | LoRA Injection | LoRA_Configuration | `get_peft_model` | llama.py |
| 3 | Data Formatting | Data_Formatting | `get_chat_template` | chat_templates.py |

### Source Files (for enrichment)

- `path/to/file1.py` - Brief purpose
- `path/to/file2.py` - Brief purpose

<!-- ENRICHMENT NEEDED: Phase 1b will add detailed Step N attribute tables below -->

---

(Repeat for each workflow)

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
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
