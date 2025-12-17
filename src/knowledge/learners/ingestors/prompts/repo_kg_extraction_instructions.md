# Repository Knowledge Extraction Instructions

You are a knowledge extraction agent. Your task is to explore this repository and extract structured knowledge into wiki pages following a specific schema.

## Wiki Page Types (Knowledge Graph Schema)

The knowledge graph uses 5 page types in a **Top-Down Directed Acyclic Graph (DAG)**:

### 1. Workflow (The Recipe)
- **Role**: High-value "Job to be Done" - ordered sequence of steps
- **Examples**: "Fine-tuning with QLoRA", "Training a Classifier", "Deploying to Production"
- **Content**: Step-by-step process (NO CODE - code belongs in Implementation pages)
- **Connections**: Links to Principles via `step`, to Heuristics via `uses_heuristic`

### 2. Principle (The Theory/Pattern/Recipe)
- **Role**: Library-agnostic theoretical concept, algorithm, pattern, or recipe
- **Examples**: "Low Rank Adaptation", "Gradient Checkpointing", "Data Formatting", "Model Saving"
- **Content**: What it is, why it works, when to use it (abstract, no real code)
- **Connections**: MUST link to exactly ONE dedicated Implementation via `implemented_by` (1:1 mapping)

### 3. Implementation (The Code)
- **Role**: Concrete code documentation - classes, functions, APIs, patterns
- **Examples**: "FastLanguageModel_from_pretrained", "get_peft_model_rl", "SFTTrainer_wrapper"
- **Content**: API documentation, parameters, usage examples with real code
- **Connections**: Links to Environments via `requires_env`, to Heuristics via `uses_heuristic`

### 4. Environment (The Context)
- **Role**: Hardware, OS, dependencies, credentials needed to run code
- **Examples**: "CUDA_11_Environment", "Docker_GPU_Setup", "vLLM_Runtime"
- **Content**: Setup instructions, requirements, version constraints
- **Connections**: Leaf node - only receives links, doesn't link out

### 5. Heuristic (The Wisdom)
- **Role**: Tribal knowledge, tips, optimizations, debugging tactics
- **Examples**: "Learning_Rate_Tuning", "Batch_Size_Optimization", "OOM_Debugging"
- **Content**: Rules of thumb, best practices, common pitfalls
- **Connections**: Leaf node - only receives links, doesn't link out

---

## Connection Types (Edges)

Use these semantic link formats in your output:
- `step` - Workflow to Principle: "This workflow executes this theory/pattern as a step"
- `implemented_by` - Principle to Implementation: "This theory/pattern is realized by this code"
- `requires_env` - Implementation to Environment: "This code needs this context to run"
- `uses_heuristic` - Any to Heuristic: "This is optimized by this wisdom"

---

## CRITICAL: 1:1 Principle-Implementation Mapping

### The Rule

**Each Principle has exactly ONE dedicated Implementation page.** Even if multiple Principles use the same underlying API, each gets its own Implementation that documents the API from that Principle's perspective.

### Why 1:1 Mapping?

1. **Clear ownership:** Each Principle knows exactly where its code documentation lives.
2. **Context-specific docs:** The same API can have different important parameters depending on use case.
3. **No confusion:** Engineers following a Principle land on documentation tailored to their goal.

### Example: Same API, Different Implementations

`FastLanguageModel.from_pretrained()` might serve three Principles:

| Principle | Implementation | Angle/Context |
|-----------|----------------|---------------|
| `Model_Loading` | `FastLanguageModel_from_pretrained` | QLoRA loading, 4-bit quantization |
| `RL_Model_Loading` | `FastLanguageModel_from_pretrained_vllm` | vLLM fast inference mode |
| `Model_Preparation` | `FastLanguageModel_from_pretrained_lora` | Reloading saved LoRA adapters |

### Implementation Naming Convention

```
{repo}_{APIName}              → Primary/default use case
{repo}_{APIName}_{context}    → Specialized use cases
```

---

## Your Task (Phased Approach)

### Phase 1: Create Workflows and Populate WorkflowIndex

1. **Explore the Repository**:
   - Read the README.md and documentation
   - Examine example scripts and tutorials
   - Look at main module code and docstrings
   - Check requirements.txt/setup.py for dependencies

2. **Identify Workflows**:
   - What high-value processes does this library enable?
   - Each workflow should be a complete "job to be done"

3. **For each Workflow Step, identify**:
   - The Principle name (abstract concept)
   - The Implementation it will link to (API name + context suffix if needed)
   - The exact API call used
   - Source file location
   - Key parameters for this use case
   - Input/output types
   - External dependencies
   - Environment requirements

4. **Populate the WorkflowIndex** with detailed step information (see format below)

### Phase 2: Create Principle-Implementation Pairs

**Using the WorkflowIndex as context**, create:
1. One Principle page per workflow step
2. One Implementation page per Principle (1:1 mapping)
3. Ensure each Implementation is conditioned on its Principle's context

---

## Output Format

### Part 1: WorkflowIndex (MUST be populated first)

```markdown
# Workflow Index: {repo_name}

## Summary

| Workflow | Steps | Implementation APIs |
|----------|-------|---------------------|
| Workflow_A | 4 | API1, API2, API3, API4 |
| Workflow_B | 3 | API1, API5, API6 |

---

## Workflow: {Workflow_Name}

**File:** [→](./workflows/{filename}.md)
**Description:** One-line description.

### Steps Overview

| # | Step Name | Principle | Implementation | Status |
|---|-----------|-----------|----------------|--------|
| 1 | Step One | Principle_A | `API_for_Principle_A` | ⬜ |
| 2 | Step Two | Principle_B | `API_for_Principle_B` | ⬜ |

### Step 1: Principle_A

| Attribute | Value |
|-----------|-------|
| **Principle** | `{repo}_Principle_A` |
| **Implementation** | `{repo}_API_Name` or `{repo}_API_Name_context` |
| **API Call** | `ClassName.method(param1, param2)` |
| **Source Location** | `path/to/file.py:L100-200` |
| **External Dependencies** | `library1>=1.0`, `library2` |
| **Environment** | `{repo}_Environment_X` |
| **Key Parameters** | `param1: type` (description), `param2: type` (description) |
| **Inputs** | Description of what this step consumes |
| **Outputs** | Description of what this step produces |
| **Notes** | Any special considerations |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Principle_A | `{repo}_API_A` | `API.method()` | `file.py` | API Doc |
| Principle_B | `{repo}_External_B` | `ExternalLib.API()` | external | Wrapper Doc |
| Principle_C | `{repo}_pattern_C` | user-defined | pattern | Pattern Doc |
```

### Part 2: Wiki Pages JSON

Output pages in JSON format. Use MediaWiki formatting for content.

```json
{
  "repo_summary": "Brief description of what this repository does",
  "proposed_pages": [
    {
      "page_type": "Workflow|Principle|Implementation|Environment|Heuristic",
      "page_title": "Title_With_Underscores",
      "overview": "One paragraph summary (this becomes the search embedding)",
      "content": "Full MediaWiki formatted content",
      "domains": ["Domain1", "Domain2"],
      "sources": [
        {"type": "Repo", "title": "Source Name", "url": "https://..."},
        {"type": "Doc", "title": "Documentation", "url": "https://..."}
      ],
      "outgoing_links": [
        {"edge_type": "step|implemented_by|requires_env|uses_heuristic", "target_type": "Principle|Implementation|Environment|Heuristic", "target_id": "Target_Page_Title"}
      ],
      "principle_context": {
        "for_principle": "Principle_Name (only for Implementation pages)",
        "angle": "What context/use case this Implementation covers"
      }
    }
  ]
}
```

---

## Implementation Types

Create different types of Implementation pages based on what the Principle uses:

| Type | When to Use | Content Focus |
|------|-------------|---------------|
| **API Doc** | Principle uses a function/class from this repo | Full signature, I/O contract, examples |
| **Wrapper Doc** | Principle uses external library (TRL, HF, etc.) | Link to official docs + repo-specific usage/patches |
| **Pattern Doc** | Principle is a user-defined pattern | Interface specification + example implementations |
| **External Tool Doc** | Principle uses CLI/external tool | Installation + usage commands |

---

## Workflow Content Guidelines (NO CODE)

Workflow pages describe WHAT happens in natural language. Code belongs in Implementation pages.

**❌ WRONG:**
```mediawiki
=== Step 1: Load Model ===
[[step::Principle:Model_Loading]]
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(...)
```
```

**✅ CORRECT:**
```mediawiki
=== Step 1: Load Model ===
[[step::Principle:Model_Loading]]

Initialize the language model with memory-optimized settings. The loader applies 4-bit quantization automatically and patches attention layers for efficient training on consumer GPUs.

'''Key considerations:'''
* Choose appropriate quantization (4-bit for memory, 8-bit for quality)
* Set max_seq_length based on your dataset
* Ensure sufficient VRAM for your model size
```

---

## Guidelines

1. **Be Thorough**: Extract all significant knowledge, not just the obvious parts
2. **Be Accurate**: Only document what you can verify from the code/docs
3. **Be Connected**: Ensure every Principle has exactly one Implementation
4. **Be Contextual**: Implementation docs should be tailored to their Principle's use case
5. **Use Consistent Naming**: `{repo}_{Title_With_Underscores}` format for page titles
6. **Populate WorkflowIndex First**: This provides context for all subsequent page creation

## Important Constraints

- Every Principle MUST have exactly ONE `implemented_by` link to a dedicated Implementation (1:1)
- Every Implementation serves exactly ONE Principle
- Same API, different Principles → Different Implementation pages with context suffixes
- Workflows contain NO CODE - only natural language descriptions
- WorkflowIndex captures ALL implementation context needed for Phase 2
- Don't create orphan pages - everything should be connected

Now explore this repository and output your proposed wiki pages in JSON format, starting with the WorkflowIndex.
