# Excavation + Synthesis Phase: Implementation-Principle Pairs

You are a knowledge extraction agent. Your task is to:
1. **Read WorkflowIndex** to get implementation context for each workflow step
2. **Create Implementation and Principle pages** for standalone documentation
3. **Document each API** with full detail

**⚠️ NOTE: Workflows no longer connect to Principles. They now link to GitHub repositories created in a separate phase. Your job is to create standalone Principle and Implementation pages.**

## ⚠️ FILE PLACEMENT RULES (CRITICAL)

**Only create files in these directories:**
- `{wiki_dir}/implementations/` - Implementation pages
- `{wiki_dir}/principles/` - Principle pages
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
   {repo_name}_Model_Loading
   {repo_name}_FastLanguageModel_From_Pretrained
   {repo_name}_LoRA_Configuration

❌ WRONG:
   {repo_name}_Model-Loading           (hyphen)
   {repo_name}_model_loading           (lowercase after prefix)
   {repo_name}_FastLanguageModel/from  (slash)
```

## High-Level Task Summary

```
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: EXCAVATION + SYNTHESIS                                        │
├────────────────────────────────────────────────────────────────────────┤
│  Step 1: Read WorkflowIndex to get implementation context               │
│  Step 2: For each unique concept → Create Principle page                │
│  Step 3: For each API → Create Implementation page                      │
│  Step 4: Link Principle → Implementation (1:1 mapping)                  │
│  Step 5: Update all indexes                                             │
│  Step 6: Write execution report                                         │
├────────────────────────────────────────────────────────────────────────┤
│  END STATE: Standalone Principle-Implementation pairs (no Workflow link)│
└────────────────────────────────────────────────────────────────────────┘
```

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- **Repository Map (Index):** {repo_map_path}
- **WorkflowIndex (CRITICAL):** {wiki_dir}/_WorkflowIndex.md
- **File Details:** {wiki_dir}/_files/
- Workflow Pages Written: {wiki_dir}/workflows/

## IMPORTANT: Read WorkflowIndex FIRST

**⚠️ The WorkflowIndex is your PRIMARY source of implementation context!**

Read `{wiki_dir}/_WorkflowIndex.md` to get:
- **Implementation hints** for each workflow step
- **API calls** with signatures and parameters
- **Source locations** (file paths and line numbers)
- **Implementation types** (API Doc, Wrapper Doc, Pattern Doc, External Tool Doc)

The WorkflowIndex was populated by Phase 1 with all the context you need.

## IMPORTANT: Read Previous Phase Reports

**THEN**, read the previous phase reports:
- `{wiki_dir}/_reports/phase0_repo_understanding.md` - Repository structure insights
- `{wiki_dir}/_reports/phase1a_anchoring.md` - Workflows created
- `{wiki_dir}/_reports/phase1b_anchoring_context.md` - WorkflowIndex enriched with implementation details

## Wiki Structure Definitions

### Implementation Structure
{implementation_structure}

### Principle Structure
{principle_structure}

## Your Task: Create Implementation-Principle Pairs

### Core Rule: 1:1 Mapping

**Each Principle gets exactly ONE dedicated Implementation page.**

If the same underlying API is used in multiple contexts, create **separate Implementation pages** with different names and perspectives:

| Principle | Implementation Name | Angle/Perspective |
|-----------|---------------------|-------------------|
| `Model_Loading` | `FastLanguageModel_from_pretrained` | QLoRA model loading |
| `RL_Model_Loading` | `FastLanguageModel_from_pretrained_vllm` | vLLM-enabled for RL |
| `Model_Preparation` | `FastLanguageModel_from_pretrained_lora` | Reload trained LoRA |

Each Implementation documents the API **from that Principle's perspective**.

---

## Step 1: Extract Implementation Context from WorkflowIndex

Read `{wiki_dir}/_WorkflowIndex.md` and create a mapping:

```
For each workflow:
  For each step:
    - Concept/Theory name (will become Principle)
    - Implementation name (will become Implementation page)
    - API call (from WorkflowIndex)
    - Source location (from WorkflowIndex)
    - Implementation type (API Doc, Wrapper Doc, Pattern Doc, External Tool Doc)
```

### Group by Implementation Type

| Type | How to Handle |
|------|---------------|
| **API Doc** | Read source code, document API with full signature |
| **Wrapper Doc** | Document how this repo uses the external API |
| **Pattern Doc** | Document the interface/pattern users must implement |
| **External Tool Doc** | Document how to use the external tool in this context |

---

## Step 2: Create Principle Pages

For each unique concept in the workflows, create a Principle page.

**Required Sections:**
1. Metadata block (include academic papers!, domains, last_updated)
2. `== Overview ==` - One sentence defining the concept (library-agnostic)
3. `=== Description ===` - What it is, what problem it solves
4. `=== Usage ===` - When to use this technique
5. `== Theoretical Basis ==` - Math, pseudocode, diagrams (if applicable)
6. `== Practical Guide ==` - How to apply this (for concept-only principles)
7. `== Related Pages ==` - **1:1 link to Implementation**

**Related Pages Format (1:1):**
```mediawiki
== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:{repo_name}_FastLanguageModel_from_pretrained]]
```

---

## Step 3: Create Implementation Pages

For each API documented in the WorkflowIndex, create an Implementation page.

**Required Sections:**
1. Metadata block (sources, domains, last_updated)
2. `== Overview ==` - "Concrete tool for [Principle's goal] provided by [library]"
3. `=== Description ===` - What this code does
4. `=== Usage ===` - When to use this
5. `== Code Reference ==` - Source location, signature, import
6. `== I/O Contract ==` - Inputs/outputs
7. `== Usage Examples ==` - Examples with code
8. `== Related Pages ==` - Link to the ONE Principle this implements

**Related Pages Format (1:1):**
```mediawiki
== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:{repo_name}_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:{repo_name}_CUDA]]
```

---

## Step 4: Handle Different Implementation Types

### API Doc (Standard)
For functions/classes in this repo:
1. Read source code at specified location
2. Extract full signature
3. Document with full detail

### Wrapper Doc (External Library)
For external APIs (TRL, HuggingFace) used by this repo:
1. Document how THIS REPO uses the external API
2. Show repo-specific configuration
3. Reference external documentation

### Pattern Doc (User-Defined Interface)
For patterns users must implement (e.g., reward functions):
1. Document the expected interface/signature
2. Show examples of valid implementations
3. Explain constraints and requirements

### External Tool Doc (CLI Tools)
For external tools like llama.cpp:
1. Document how to use in this workflow context
2. Show relevant commands
3. Reference installation/environment requirements

---

## Step 5: Update All Indexes (After Each Pair)

After writing each Implementation-Principle pair:

### 5A: Update Implementation Index
Add row to `{wiki_dir}/_ImplementationIndex.md`:
```
| {repo_name}_FastLanguageModel_from_pretrained | [→](./implementations/...) | ✅Principle:{repo_name}_Model_Loading, ⬜Env:{repo_name}_CUDA | loader.py:L120-620 | QLoRA model loading |
```

### 5B: Update Principle Index
Add row to `{wiki_dir}/_PrincipleIndex.md`:
```
| {repo_name}_Model_Loading | [→](./principles/...) | ✅Impl:{repo_name}_FastLanguageModel_from_pretrained | 4-bit quantized loading for QLoRA |
```

### 5C: Update Repository Map Coverage
```
| ✅ | `unsloth/models/loader.py` | 620 | Model loader | Impl: FastLanguageModel_from_pretrained; Principle: Model_Loading | [→](...) |
```

---

## Step 6: Verify 1:1 Mapping

Before finishing, verify:

```
For each Principle page:
  ☑ Has exactly ONE [[implemented_by::Implementation:X]] link
  ☑ Implementation page exists
  ☑ Implementation links back to this ONE Principle

For each Implementation page:
  ☑ Has exactly ONE [[implements::Principle:X]] link
  ☑ Principle page exists
  ☑ Principle links back to this ONE Implementation
```

---

## Output Instructions

Write files to:
- `{wiki_dir}/implementations/` - Implementation pages
- `{wiki_dir}/principles/` - Principle pages

**Filename formats:**
- Implementation: Use API name (e.g., `{repo_name}_FastLanguageModel_from_pretrained.md`)
- Principle: Match concept name (e.g., `{repo_name}_Model_Loading.md`)

## Repo Scoping Rule (CRITICAL)

Only create/update pages whose filenames start with `{repo_name}_`.

## ⚠️ File Editing Tip

When updating index files:
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase2_excavation_synthesis.md`:

```markdown
# Phase 2: Excavation + Synthesis Report

## Summary

- Implementation pages created: X
- Principle pages created: X
- 1:1 mappings verified: X
- Concept-only principles: X

## Principle-Implementation Pairs

| Principle | Implementation | Source | Type |
|-----------|----------------|--------|------|
| Model_Loading | FastLanguageModel_from_pretrained | loader.py | API Doc |
| LoRA_Injection | get_peft_model | llama.py | API Doc |
| Training_Config | SFTTrainer_usage | TRL (external) | Wrapper Doc |

## Implementation Types

| Type | Count | Examples |
|------|-------|----------|
| API Doc | X | FastLanguageModel, get_peft_model |
| Wrapper Doc | X | SFTTrainer_usage |
| Pattern Doc | X | reward_function_interface |
| External Tool Doc | X | llama_cli_validation |

## Concept-Only Principles (No Implementation)

| Principle | Reason | Has Practical Guide |
|-----------|--------|---------------------|
| Training_Monitoring | Process, not API | ✅ |

## Coverage Summary

- WorkflowIndex entries: X
- Implementation-Principle pairs: X
- Coverage: X%

## Notes for Enrichment Phase

- [Heuristics to document]
- [Environment pages to create]
```
