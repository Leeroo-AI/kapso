# Excavation + Synthesis Phase: Implementation-Principle Pairs

You are a knowledge extraction agent. Your task is to:
1. **Read WorkflowIndex** to get implementation context for each Principle
2. **Create 1:1 Principle-Implementation pairs** based on the context
3. **Document each API from its Principle's perspective** (angle-based documentation)

**⚠️ CRITICAL GOAL: Every Principle gets exactly ONE dedicated Implementation page. Same API can have multiple Implementation pages if used by different Principles (from different angles).**

## High-Level Task Summary

```
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: EXCAVATION + SYNTHESIS                                        │
├────────────────────────────────────────────────────────────────────────┤
│  Step 1: Read WorkflowIndex to get implementation context per step      │
│  Step 2: For each Principle → Create DEDICATED Implementation page      │
│  Step 3: Document API from that Principle's perspective (angle)         │
│  Step 4: Link them bidirectionally (1:1 mapping)                        │
│  Step 5: Update all indexes                                             │
│  Step 6: Write execution report                                         │
├────────────────────────────────────────────────────────────────────────┤
│  END STATE: 1:1 Principle-Implementation mapping for all workflow steps │
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
- **1:1 mappings** between Principles and suggested Implementation names

The WorkflowIndex was populated by Phase 1 with all the context you need.

### Example WorkflowIndex Entry

```markdown
### Step 2: Model_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `{repo_name}_Model_Loading` |
| **Implementation** | `{repo_name}_FastLanguageModel_from_pretrained` |
| **API Call** | `FastLanguageModel.from_pretrained(model_name, max_seq_length, load_in_4bit, dtype)` |
| **Source Location** | `unsloth/models/loader.py:L120-620` |
| **External Dependencies** | `transformers`, `bitsandbytes` |
| **Environment** | `{repo_name}_CUDA` |
| **Key Parameters** | `model_name: str`, `max_seq_length: int`, `load_in_4bit: bool` |
| **Inputs** | Model name/path (HuggingFace ID or local) |
| **Outputs** | `Tuple[PeftModel, PreTrainedTokenizer]` |
```

Use this context to:
1. Know exactly which API to document
2. Know where to find the source code
3. Know what parameters to focus on
4. Know what Principle this Implementation serves

## IMPORTANT: Read Previous Phase Reports

**THEN**, read the previous phase reports:
- `{wiki_dir}/_reports/phase0_repo_understanding.md` - Repository structure insights
- `{wiki_dir}/_reports/phase1_anchoring.md` - Workflows created, APIs to trace

## Wiki Structure Definitions

### Implementation Structure
{implementation_structure}

### Principle Structure
{principle_structure}

## Your Task: Create 1:1 Principle-Implementation Pairs

### Core Rule: 1:1 Mapping

**Each Principle gets exactly ONE dedicated Implementation page.**

If the same underlying API (e.g., `FastLanguageModel.from_pretrained`) is used by multiple Principles, create **separate Implementation pages** with different names and perspectives:

| Principle | Implementation Name | Angle/Perspective |
|-----------|---------------------|-------------------|
| `Model_Loading` | `FastLanguageModel_from_pretrained` | QLoRA model loading |
| `RL_Model_Loading` | `FastLanguageModel_from_pretrained_vllm` | vLLM-enabled for RL |
| `Model_Preparation` | `FastLanguageModel_from_pretrained_lora` | Reload trained LoRA |

Each Implementation documents the API **from that Principle's perspective**:
- Focus on parameters relevant to that use case
- Show examples tailored to that workflow
- Document I/O specific to that context

---

## Step 1: Extract Implementation Context from WorkflowIndex

Read `{wiki_dir}/_WorkflowIndex.md` and create a mapping:

```
For each workflow:
  For each step:
    - Principle name (from WorkflowIndex)
    - Implementation name (from WorkflowIndex)
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

## Step 2: For Each Principle, Create Its Dedicated Implementation

Process each Principle from the WorkflowIndex:

```
┌─────────────────────────────────────────────────────────────────┐
│  FOR EACH PRINCIPLE (from WorkflowIndex)                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Get implementation context from WorkflowIndex                │
│  2. Read source code at specified location                       │
│  3. Write Implementation page (from Principle's angle)           │
│  4. Write Principle page                                         │
│  5. Link them 1:1                                                │
│  6. Update indexes                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2A: Write the Implementation Page (Angle-Based)

Create `{wiki_dir}/implementations/{{Implementation_Name}}.md`

**File naming:** Use the Implementation name from WorkflowIndex (e.g., `{repo_name}_FastLanguageModel_from_pretrained.md`)

**Key Principle:** Document the API **from the Principle's perspective**:
- Focus on parameters relevant to THIS use case
- Show examples tailored to THIS workflow
- Document I/O specific to THIS context

**Required Sections:**
1. Metadata block (sources, domains, last_updated)
2. `== Overview ==` - "Concrete tool for [Principle's goal] provided by [library]"
3. `=== Description ===` - What this code does **for this Principle's use case**
4. `=== Usage ===` - When to use this **in this workflow context**
5. `== Code Reference ==` - Source location, signature, import (from WorkflowIndex)
6. `== I/O Contract ==` - Inputs/outputs **relevant to this Principle**
7. `== Usage Examples ==` - Examples **tailored to this Principle's workflow**
8. `== Related Pages ==` - Link to the ONE Principle this implements

**Implementation Angle Example:**

For `Model_Loading` Principle:
```mediawiki
== Overview ==

Concrete tool for loading Large Language Models with 4-bit quantization for memory-efficient fine-tuning.

=== Description ===

`FastLanguageModel.from_pretrained` loads a pre-trained model with automatic 4-bit NF4 quantization for QLoRA fine-tuning. It handles device mapping, quantization configuration, and attention backend selection.

<!-- Focus on QLoRA loading parameters -->

== Usage Examples ==

=== QLoRA Fine-tuning Setup ===
<syntaxhighlight lang="python">
# Load model for QLoRA fine-tuning
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,   # <-- QLoRA angle: 4-bit quantization
    dtype = None,
)
</syntaxhighlight>
```

For `RL_Model_Loading` Principle (same API, different angle):
```mediawiki
== Overview ==

Concrete tool for loading models with vLLM backend for fast inference during reinforcement learning.

=== Description ===

`FastLanguageModel.from_pretrained` with `fast_inference=True` loads a model with vLLM backend, enabling fast generation sampling during GRPO training.

<!-- Focus on vLLM/RL parameters -->

== Usage Examples ==

=== vLLM-Enabled RL Loading ===
<syntaxhighlight lang="python">
# Load model with vLLM for RL training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    fast_inference = True,     # <-- RL angle: vLLM backend
    max_lora_rank = 64,        # <-- RL angle: higher rank
    gpu_memory_utilization = 0.6,
)
</syntaxhighlight>
```

**Related Pages Format (1:1):**
```mediawiki
== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:{repo_name}_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:{repo_name}_CUDA]]
```

### 2B: Write the Principle Page

Create `{wiki_dir}/principles/{{Principle_Name}}.md`

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

## Step 3: Handle Different Implementation Types

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

```mediawiki
== Overview ==

Usage pattern for TRL's SFTTrainer within Unsloth's optimized training pipeline.

=== Description ===

SFTTrainer is TRL's supervised fine-tuning trainer. Unsloth patches it at import time to use optimized kernels for cross-entropy loss and RMS normalization.

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/huggingface/trl TRL]
* '''Unsloth Patches:''' unsloth/trainer.py

=== Import ===
<syntaxhighlight lang="python">
# Import Unsloth FIRST to apply patches
from unsloth import FastLanguageModel

# Then import SFTTrainer (now patched)
from trl import SFTTrainer, SFTConfig
</syntaxhighlight>
```

### Pattern Doc (User-Defined Interface)
For patterns users must implement (e.g., reward functions):
1. Document the expected interface/signature
2. Show examples of valid implementations
3. Explain constraints and requirements

```mediawiki
== Overview ==

Interface specification for reward functions used in GRPO reinforcement learning.

=== Description ===

Reward functions are user-defined callables that score model completions. They must follow a specific signature to work with GRPOTrainer.

== Interface Specification ==

=== Required Signature ===
<syntaxhighlight lang="python">
def reward_function(
    completions: List[str],  # Model-generated completions
    prompts: List[str],      # Original prompts
    **kwargs                 # Optional additional context
) -> List[float]:            # Reward scores (one per completion)
    ...
</syntaxhighlight>

=== Constraints ===
* Return list must have same length as completions
* Scores should be normalized (e.g., -1 to 1, or 0 to 1)
* Higher scores = better completions

== Usage Examples ==

=== Format Checking Reward ===
<syntaxhighlight lang="python">
def format_reward(completions, prompts):
    rewards = []
    for completion in completions:
        score = 0.0
        if "<think>" in completion:
            score += 0.5
        if "\\boxed{{" in completion:
            score += 0.5
        rewards.append(score)
    return rewards
</syntaxhighlight>
```

### External Tool Doc (CLI Tools)
For external tools like llama.cpp:
1. Document how to use in this workflow context
2. Show relevant commands
3. Reference installation/environment requirements

```mediawiki
== Overview ==

Validation of GGUF model files using llama.cpp's CLI tools.

=== Description ===

After exporting a model to GGUF format, validate it loads correctly using llama.cpp's `llama-cli` tool.

== Code Reference ==

=== Tool Location ===
* '''External Tool:''' llama.cpp
* '''Build From:''' [https://github.com/ggerganov/llama.cpp llama.cpp GitHub]

=== Environment ===
* [[requires_env::Environment:{repo_name}_llama_cpp]]

== Usage Examples ==

=== Basic Validation ===
<syntaxhighlight lang="bash">
# Test model loads and generates
./llama-cli -m ./model-q4_k_m.gguf -p "Hello" -n 50
</syntaxhighlight>
```

---

## Step 4: Update All Indexes (After Each Pair)

After writing each Implementation-Principle pair:

### 4A: Update Implementation Index
Add row to `{wiki_dir}/_ImplementationIndex.md`:
```
| {repo_name}_FastLanguageModel_from_pretrained | [→](./implementations/...) | ✅Principle:{repo_name}_Model_Loading, ⬜Env:{repo_name}_CUDA | loader.py:L120-620 | QLoRA model loading |
```

### 4B: Update Principle Index
Add row to `{wiki_dir}/_PrincipleIndex.md`:
```
| {repo_name}_Model_Loading | [→](./principles/...) | ✅Impl:{repo_name}_FastLanguageModel_from_pretrained | 4-bit quantized loading for QLoRA |
```

### 4C: Update Workflow Index
Change `⬜Impl:X` to `✅Impl:X` and `⬜Principle:X` to `✅Principle:X`

### 4D: Update Repository Map Coverage
```
| ✅ | `unsloth/models/loader.py` | 620 | Model loader | Impl: FastLanguageModel_from_pretrained; Principle: Model_Loading | [→](...) |
```

---

## Step 5: Verify 1:1 Mapping

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

If a Principle has no API (concept-only), it still needs a Practical Guide section instead of an Implementation link.

---

## Step 6: Final Verification

**⚠️ DO NOT SKIP THIS STEP!**

Before writing the execution report:

### 6A: Count Mappings
```
Total Principles from WorkflowIndex: ___
Principles with 1:1 Implementation: ___
Concept-only Principles (no API): ___
Coverage: ___% (MUST be 100%)
```

### 6B: Verify Angles
For Principles sharing the same underlying API:
- Each has a dedicated Implementation page
- Each Implementation focuses on that Principle's angle
- Names differentiate the angles (e.g., `_vllm`, `_lora`, etc.)

---

## Output Instructions

Write files to:
- `{wiki_dir}/implementations/` - Implementation pages
- `{wiki_dir}/principles/` - Principle pages

**Filename formats:**
- Implementation: Use name from WorkflowIndex (e.g., `{repo_name}_FastLanguageModel_from_pretrained.md`)
- Principle: Match workflow references exactly (e.g., `{repo_name}_Model_Loading.md`)

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

## 1:1 Principle-Implementation Pairs

| Principle | Implementation | Source | Angle |
|-----------|----------------|--------|-------|
| Model_Loading | FastLanguageModel_from_pretrained | loader.py | QLoRA loading |
| RL_Model_Loading | FastLanguageModel_from_pretrained_vllm | loader.py | vLLM for RL |
| LoRA_Injection | get_peft_model | llama.py | Standard LoRA |

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
- 1:1 Implementation-Principle pairs: X
- Coverage: X% (should be 100%)

## Notes for Enrichment Phase

- [Heuristics to document]
- [Environment pages to create]
```
