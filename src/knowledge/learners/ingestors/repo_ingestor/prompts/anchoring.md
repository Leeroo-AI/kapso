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

### Step 6: Update the Enhanced WorkflowIndex (CRITICAL)

**⚠️ CRITICAL:** Update `{wiki_dir}/_WorkflowIndex.md` **IMMEDIATELY after creating each Workflow page**.

The WorkflowIndex is the **bridge document** that preserves implementation context for Phase 2. Without proper implementation hints, Phase 2 cannot create correct Principle→Implementation mappings.

#### WorkflowIndex Structure

```markdown
# Workflow Index: {repo_name}

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).

---

## Summary

| Workflow | Steps | Principles | Implementation APIs |
|----------|-------|------------|---------------------|
| QLoRA_Finetuning | 6 | 6 | FastLanguageModel, get_peft_model, SFTTrainer |

---

## Workflow: {repo_name}_WorkflowName

**File:** [→](./workflows/{repo_name}_WorkflowName.md)
**Description:** One-line description of the workflow.

### Steps Overview

| # | Step Name | Principle | Implementation | Status |
|---|-----------|-----------|----------------|--------|
| 1 | Load Model | Model_Loading | `FastLanguageModel.from_pretrained` | ⬜ |
| 2 | Apply LoRA | LoRA_Injection | `FastLanguageModel.get_peft_model` | ⬜ |

### Step 1: Model_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `{repo_name}_Model_Loading` |
| **Implementation** | `{repo_name}_FastLanguageModel_from_pretrained` |
| **API Call** | `FastLanguageModel.from_pretrained(model_name, max_seq_length, load_in_4bit, dtype)` |
| **Source Location** | `unsloth/models/loader.py:L120-620` |
| **External Dependencies** | `transformers`, `bitsandbytes` |
| **Environment** | `{repo_name}_CUDA` |
| **Key Parameters** | `model_name: str`, `max_seq_length: int`, `load_in_4bit: bool`, `dtype: Optional[torch.dtype]` |
| **Inputs** | Model name/path (HuggingFace ID or local path) |
| **Outputs** | `Tuple[PeftModel, PreTrainedTokenizer]` |

### Step 2: LoRA_Injection

| Attribute | Value |
|-----------|-------|
| **Principle** | `{repo_name}_LoRA_Injection` |
| **Implementation** | `{repo_name}_get_peft_model` |
| **API Call** | `FastLanguageModel.get_peft_model(model, r, lora_alpha, target_modules, ...)` |
| **Source Location** | `unsloth/models/llama.py:L2578-2800` |
| **External Dependencies** | `peft` |
| **Environment** | `{repo_name}_CUDA` |
| **Key Parameters** | `model`, `r: int`, `lora_alpha: int`, `target_modules: List[str]` |
| **Inputs** | Model from Step 1 |
| **Outputs** | `PeftModel` with LoRA adapters |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Model_Loading | `FastLanguageModel_from_pretrained` | `from_pretrained` | `loader.py` | API Doc |
| LoRA_Injection | `get_peft_model` | `get_peft_model` | `llama.py` | API Doc |
| SFT_Training | `SFTTrainer_usage` | `SFTTrainer` | TRL (external) | Wrapper Doc |
| Reward_Definition | `reward_function_interface` | N/A | User code | Pattern Doc |
```

#### Required Information Per Step

For EACH workflow step, you MUST capture:

| Field | Description | How to Find |
|-------|-------------|-------------|
| **Principle** | Principle page name (must match `[[step::...]]` link) | From workflow step |
| **Implementation** | Suggested Implementation page name | Derive from API name |
| **API Call** | Exact function/method with key parameters | From example code/docs |
| **Source Location** | File path and line numbers in repo | Read source file |
| **External Dependencies** | Non-repo libraries used | From imports |
| **Environment** | Environment page name | From system requirements |
| **Key Parameters** | Important params with types | From function signature |
| **Inputs** | What this step consumes | From data flow |
| **Outputs** | What this step produces | From return type |

#### Implementation Types

Mark each implementation with its type:

| Type | When to Use | Example |
|------|-------------|---------|
| **API Doc** | Function/class in this repo | `FastLanguageModel.from_pretrained` |
| **Wrapper Doc** | External library with repo-specific usage | `SFTTrainer` (TRL wrapper) |
| **Pattern Doc** | User-defined interface/pattern | `reward_function(completions, prompts)` |
| **External Tool Doc** | CLI or external tool | `llama-cli` for GGUF validation |

#### 1:1 Principle-Implementation Mapping Rule

**CRITICAL:** Each Principle should map to ONE dedicated Implementation page.

If the same API is used by multiple Principles (from different angles), create **separate Implementation pages** with different names:

| Principle | Implementation Name | Angle |
|-----------|---------------------|-------|
| `Model_Loading` | `FastLanguageModel_from_pretrained` | QLoRA loading |
| `RL_Model_Loading` | `FastLanguageModel_from_pretrained_vllm` | vLLM-enabled loading |
| `Model_Preparation` | `FastLanguageModel_from_pretrained_lora` | Reload trained LoRA |

This ensures Phase 2 creates Principle-specific documentation for each use case.

---

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

## Summary
- Workflows created: X
- Total steps documented: X
- Implementation hints captured: X

## Workflows Created

| Workflow | Source Files | Steps | Implementation APIs |
|----------|--------------|-------|---------------------|
| [Name] | [files] | [count] | [APIs used] |

## Coverage Summary
- Source files covered: X
- Example files documented: X

## Implementation Context Captured

| Workflow | Principles | API Docs | Wrapper Docs | Pattern Docs |
|----------|------------|----------|--------------|--------------|
| QLoRA_Finetuning | 6 | 4 | 1 | 0 |

## Notes for Excavation Phase

### APIs to Extract (with Source Locations)
| API | Source | Used By Principles |
|-----|--------|-------------------|
| FastLanguageModel.from_pretrained | loader.py:L120-620 | Model_Loading |
| get_peft_model | llama.py:L2578-2800 | LoRA_Injection |

### External Dependencies to Document
- TRL: SFTTrainer, GRPOTrainer
- HuggingFace: tokenizer.apply_chat_template

### User-Defined Patterns to Document
- reward_function interface for GRPO
```
