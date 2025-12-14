# Anchoring Phase: Find and Document Workflows

You are a knowledge extraction agent. Your task is to identify the "Golden Paths" (primary use cases) in this repository and create Workflow wiki pages.

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}

## Wiki Structure Definition

{workflow_structure}

## Your Task

### Step 1: Scan High-Level Documentation

Read these files to find complete execution paths:
- `README.md` (especially Quick Start, Getting Started, Examples sections)
- `examples/*.py` or `examples/**/*.py` (standalone scripts)
- `notebooks/*.ipynb` (Jupyter notebooks)
- Any `docs/` or `documentation/` folders

### Step 2: Identify Candidate Workflows

For each "Golden Path" you find, identify:
- **Name**: What is this workflow called? (e.g., "QLoRA Fine-Tuning", "Inference Pipeline")
- **Source file**: Which example/script demonstrates it?
- **Steps**: 3-7 high-level milestones (abstract verbs like "Load Model", "Train", "Save")
- **Use case**: When would someone use this workflow?

### Step 3: Write Workflow Pages

For EACH workflow you identify, create a wiki page following the exact structure from the Workflow Page Sections Guide above.

**Required Sections:**
1. Metadata block (wikitable with sources, domains, last_updated)
2. `== Overview ==` - One sentence summary
3. `=== Description ===` - What this workflow does
4. `=== Usage ===` - When to use it
5. `== Execution Steps ==` - Ordered steps with `[[step::Principle:Step_Name]]` links
6. `== Execution Diagram ==` - Mermaid flowchart
7. `== Related Pages ==` - Step links and heuristic links

**Important:**
- Use REAL code examples from the repository
- The `[[step::Principle:X]]` links should use placeholder names for now (will be updated in synthesis phase)
- Include actual import statements and function calls from the repo

## Output Instructions

Write .md files to: `{wiki_dir}/workflows/`

**Filename format:** `{repo_name}_WorkflowName.md`
- Use underscores instead of spaces
- Use PascalCase for workflow names

**Examples:**
- `unsloth_QLoRA_Finetuning.md`
- `unsloth_DPO_Alignment.md`
- `unsloth_Model_Export.md`

Write at least 1-3 workflow pages for the most important use cases in the repository.

