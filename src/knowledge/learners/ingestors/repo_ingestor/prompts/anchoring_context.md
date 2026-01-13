# Phase 1b: WorkflowIndex Enrichment

You are a Code Analyst. Your task is to ENRICH the rough WorkflowIndex with detailed implementation context that the Repository Builder phase needs.

## ⚠️ FILE PLACEMENT RULES (CRITICAL)

**Only modify/create files in these locations:**
- `{wiki_dir}/_WorkflowIndex.md` - The main file you're enriching
- `{wiki_dir}/_reports/` - Execution reports

**DO NOT create:**
- Summary files at the root of `{wiki_dir}`
- Documentation files outside the designated directories
- "Notes", "summaries", or "completion reports" outside `_reports/`

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- **Rough WorkflowIndex:** {wiki_dir}/_WorkflowIndex.md
- **Repository Map:** {repo_map_path}
- **File Details:** {wiki_dir}/_files/

## What You Have (Input)

Phase 1a created a **rough WorkflowIndex** with:
- Summary table (Workflow | Steps | Rough APIs | GitHub URL)
- Per-workflow sections with Steps Overview tables
- Source file hints for each workflow

## What You Must Add (Output)

For EACH step in EACH workflow, add a **detailed attribute table** with:
- Exact API signature
- Source file path and line numbers
- External dependencies
- Key parameters with types
- Input/Output specifications

This information will be used by the Repository Builder to create GitHub repositories.

## Your Task

### Step 1: Read the Rough WorkflowIndex

Read `{wiki_dir}/_WorkflowIndex.md` to understand:
- Which workflows exist
- What steps each workflow has
- Which files are mentioned as related

### Step 2: Cross-Reference with Repository Map

Read `{repo_map_path}` and use the **Purpose** column to:
- Verify the rough API mappings are correct
- Find additional context about each file
- Identify exact file paths

### Step 3: Read File Details

For each source file mentioned in the WorkflowIndex:
1. Read its detail page in `{wiki_dir}/_files/`
2. Note the classes, functions, and imports listed
3. Use the Understanding section for context

### Step 4: Trace to Source Code

For each step's API:
1. Read the actual source file in `{repo_path}`
2. Find the function/class definition
3. Extract:
   - **Full signature** with all parameters
   - **Line numbers** (start-end)
   - **Imports** (external dependencies)
   - **Return type** and docstring

### Step 5: Write Enriched WorkflowIndex

**REPLACE the placeholder** `<!-- ENRICHMENT NEEDED -->` with detailed Step attribute tables.

For EACH step, add this structure:

```markdown
### Step N: Step_Name

| Attribute | Value |
|-----------|-------|
| **API Call** | `ClassName.method(param1: Type, param2: Type) -> ReturnType` |
| **Source Location** | `path/to/file.py:L100-200` |
| **External Dependencies** | `library1`, `library2` |
| **Key Parameters** | `param1: Type` - description, `param2: Type` - description |
| **Inputs** | What this step consumes (from previous step or user) |
| **Outputs** | What this step produces (for next step or final result) |
```

**Also add** the Implementation Extraction Guide table at the end of each workflow section:

```markdown
### Implementation Extraction Guide

| Step | API | Source | Dependencies | Type |
|------|-----|--------|--------------|------|
| Data_Preparation | `get_chat_template` | `chat_templates.py:L50-100` | transformers | API Doc |
| Model_Loading | `FastLanguageModel.from_pretrained` | `loader.py:L120-620` | bitsandbytes | API Doc |
| Training | `SFTTrainer` | TRL (external) | trl | Wrapper Doc |
```

## Implementation Types

Mark each implementation with its type:

| Type | When to Use | Example |
|------|-------------|---------|
| **API Doc** | Function/class defined in this repo | `FastLanguageModel.from_pretrained` |
| **Wrapper Doc** | External library with repo-specific usage patterns | `SFTTrainer` (TRL library) |
| **Pattern Doc** | User-defined interface or pattern | `reward_function(prompts, completions)` |
| **External Tool Doc** | CLI tool or external system | `llama-cli` for GGUF validation |

## Verification Checklist

Before finishing, verify:

- [ ] Every workflow section has detailed Step N tables
- [ ] Every step table has ALL 6 attributes filled in
- [ ] Source locations include file path AND line numbers (e.g., `file.py:L100-200`)
- [ ] Implementation Extraction Guide exists for each workflow
- [ ] No `<!-- ENRICHMENT NEEDED -->` comments remain

## Repo Scoping Rule (CRITICAL)

Only enrich WorkflowIndex content for workflows whose names start with `{repo_name}_`.

## ⚠️ File Editing Tip

When updating `_WorkflowIndex.md`:
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase1b_anchoring_context.md`:

```markdown
# Phase 1b: WorkflowIndex Enrichment Report

## Summary
- Workflows enriched: X
- Steps with detailed tables: X
- Source files traced: X

## Enrichment Details

| Workflow | Steps Enriched | APIs Traced | Line Numbers Found |
|----------|----------------|-------------|-------------------|
| [Name] | X | X | Yes/Partial/No |

## Implementation Types Found

| Type | Count | Examples |
|------|-------|----------|
| API Doc | X | [list] |
| Wrapper Doc | X | [list] |
| Pattern Doc | X | [list] |
| External Tool Doc | X | [list] |

## Source Files Traced

| File | Lines | APIs Extracted |
|------|-------|----------------|
| loader.py | L120-620 | from_pretrained |
| llama.py | L2578-3100 | get_peft_model |

## Issues Found
- [Any APIs that couldn't be traced]
- [Files that don't exist]
- [Unclear mappings]

## Ready for Repository Builder
- [ ] All Step tables complete
- [ ] All source locations verified
- [ ] Implementation Extraction Guides complete
```
