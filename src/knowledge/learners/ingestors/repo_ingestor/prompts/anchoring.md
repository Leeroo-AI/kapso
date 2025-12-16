# Anchoring Phase: Find and Document Workflows

You are a knowledge extraction agent. Your task is to identify the "Golden Paths" (primary use cases) in this repository and create Workflow wiki pages.

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
- Use REAL code examples from the repository
- Include actual import statements and function calls from the repo
- For the `[[step::Principle:{repo_name}_X]]` links, use descriptive placeholder names
- Do NOT add `[[uses_heuristic::...]]` links yet - those come later

### Step 5: Update Coverage in Repository Map

After creating Workflow pages, **update the index** at `{repo_map_path}`:

For each source file your Workflow covers, update its **Coverage column**:

```markdown
| ✅ | `examples/qlora.py` | 150 | QLoRA example | Workflow: {repo_name}_QLoRA_Finetuning | [→](...) |
```

Coverage format: `Workflow: PageName` or `Workflow: Page1, Page2` if multiple.

### Step 6: Update the Workflow Index (IMMEDIATELY)

**⚠️ CRITICAL:** Update `{wiki_dir}/_WorkflowIndex.md` **IMMEDIATELY after creating each Workflow page**.

| Column | Content |
|--------|---------|
| Page | Workflow page name (without .md) |
| File | Link: `[→](./workflows/{repo_name}_X.md)` |
| Connections | All links with **per-reference status** (see format below) |
| Notes | Brief description |

**Connections Format (use FULL page names with `{repo_name}_` prefix):**
- `⬜Impl:{repo_name}_FastLanguageModel` = Implementation not created yet
- `⬜Impl:{repo_name}_SFTTrainer` = Implementation not created yet
- `⬜Principle:{repo_name}_LoRA` = Principle not created yet

**Example row:**
```
| {repo_name}_QLoRA_Finetuning | [→](./workflows/...) | ⬜Impl:{repo_name}_FastLanguageModel, ⬜Impl:{repo_name}_SFTTrainer, ⬜Principle:{repo_name}_LoRA | Main fine-tuning workflow |
```

Since you're the first phase, all connections will be `⬜` (later phases create these pages and update to `✅`).

**Note:** Use the SAME names you use in Related Pages section (e.g., `[[step::Principle:{repo_name}_LoRA]]`).

## Output Instructions

Write .md files to: `{wiki_dir}/workflows/`

**Filename format:** `{repo_name}_WorkflowName.md`

**Examples:**
- `{repo_name}_QLoRA_Finetuning.md`
- `{repo_name}_DPO_Alignment.md`
- `{repo_name}_Model_Export.md`

Write at least 3-5 workflow pages for the most important use cases.

## Repo Scoping Rule (CRITICAL)

Only create/update Workflow pages whose filenames start with `{repo_name}_`.

## ⚠️ File Editing Tip

When updating index files (`_RepoMap.md`, `_WorkflowIndex.md`):
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase1_anchoring.md`:

```markdown
# Phase 1: Anchoring Report

## Workflows Created
| Workflow | Source Files | Steps |
|----------|--------------|-------|
| [Name] | [files] | [step count] |

## Coverage Summary
- Source files covered: X
- Example files documented: X

## Notes for Excavation Phase
- [Key APIs to trace from workflows]
- [Important classes/functions used]
```
