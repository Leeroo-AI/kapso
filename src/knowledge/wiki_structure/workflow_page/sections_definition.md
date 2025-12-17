# Workflow Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for a **Workflow** page. Every section is mandatory to ensure the recipe is reproducible and understandable.

---

## 1. Metadata Block (Top of Page)
**Goal:** Provide structured context for the graph parser.
**Format:** Semantic MediaWiki Table (Right-aligned).

### Fields Explanation
1.  **Knowledge Sources:** HIGH-LEVEL references only (not individual file paths!).
    *   *Syntax:* `[[source::{Type}|{Title}|{URL}]]`
    *   *Types:*
        - `Repo` → Link to the **repository root URL**, not individual files
        - `Doc` → Official documentation websites
        - `Blog` → Tutorials, blog posts
        - `Paper` → Academic papers (arXiv, etc.)
    *   ⚠️ **DO NOT put individual file paths here** (e.g., `unsloth/save.py`)
    *   Specific file references belong in the linked Implementation pages
2.  **Domains:** Categorization tags.
    *   *Syntax:* `[[domain::{Tag}]]`
    *   *Examples:* `LLM_Ops`, `Data_Engineering`, `Training`.
3.  **Last Updated:** Freshness marker.
    *   *Syntax:* `[[last_updated::{YYYY-MM-DD HH:MM GMT}]]`

**Sample:**
```mediawiki
{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Blog|Fine-tuning Llama 2|https://www.philschmid.de/sagemaker-llama2-qlora]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2023-11-20 14:00 GMT]]
|}
```

**❌ WRONG (individual file paths in Knowledge Sources):**
```mediawiki
* [[source::Repo|Unsloth Save Module|https://github.com/unslothai/unsloth/blob/main/unsloth/save.py]]  ← WRONG!
```

**✅ CORRECT (high-level repo URL):**
```mediawiki
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]  ← RIGHT!
```

---

## 2. Overview Block (The "Card")

### `== Overview ==`
**Instruction:** Define the goal in **one sentence**.
*   *Purpose:* Search snippet.
*   *Content:* "End-to-end process for {Goal} using {Technique}."

**Sample:**
```mediawiki
== Overview ==
End-to-end process for parameter-efficient fine-tuning (PEFT) of Llama-2 models on custom datasets.
```

### `=== Description ===` (The "What")
**Instruction:** Explain the **Process**.
*   *Content:*
    1.  **Goal:** What is the output? (e.g., "A LoRA adapter").
    2.  **Scope:** What does it cover? (e.g., "From raw text to saved weights").
    3.  **Strategy:** High-level approach (e.g., "Uses QLoRA to minimize memory").

**Sample:**
```mediawiki
=== Description ===
This workflow outlines the standard procedure for fine-tuning Large Language Models (LLMs) on consumer hardware. It leverages Quantization and Low-Rank Adapters (LoRA) to reduce memory requirements, allowing training of 7B+ models on single GPUs. The process covers data formatting, model quantization, adapter training, and merging.
```

### `=== Usage ===` (The "When")
**Instruction:** Define the **Business Trigger**.
*   *Purpose:* Tells the user when to run this.
*   *Content:*
    *   *Input State:* "You have a JSON dataset..."
    *   *Desired Output:* "You need a specialized model for X..."
*   *Goal:* Answer "Is this the right recipe for my task?"

**Sample:**
```mediawiki
=== Usage ===
Execute this workflow when you have a domain-specific dataset (instruction-tuning style) and need to adapt a base Llama-2 model to follow instructions, but have limited GPU resources (e.g., <24GB VRAM).
```

---

## 3. The Recipe

### `== Execution Steps ==`
**Instruction:** The ordered list of steps in natural language.

**⚠️ NO CODE IN WORKFLOW STEPS!** Actual code belongs in Implementation pages, which are linked through Principles.

*   *Structure:* Use Level 3 Headers (`===`) for each step.
*   *Content per Step:*
    1.  **Link:** `[[step::Principle:{Principle_Name}]]` — links to the theoretical concept.
    2.  **Description:** Natural language summary of what this step accomplishes. This should echo/summarize the Overview of the linked Principle.
    3.  **Pseudocode (optional):** If needed for clarity, use high-level pseudocode (not actual implementation code).

**Graph Flow Reminder:**
```
Workflow Step → Principle (theory) → Implementation (actual code)
```
The Workflow describes WHAT happens. The Principle explains WHY. The Implementation shows HOW with real code.

**Sample:**
```mediawiki
== Execution Steps ==

=== Step 1: Data Preparation ===
[[step::Principle:Data_Formatting]]

Transform raw training data into the structured prompt format expected by the model. This involves mapping input fields to a consistent template (e.g., instruction/input/output structure) and applying the model's chat template for proper tokenization boundaries.

'''Key considerations:'''
* Ensure all examples follow the same schema
* Apply the correct chat template for your model family
* Validate that special tokens are properly inserted

=== Step 2: Model Quantization ===
[[step::Principle:Quantization]]

Load the base model in reduced precision to minimize memory footprint. The quantization process maps 16-bit weights to a lower bit representation (e.g., 4-bit NormalFloat) while preserving model quality through careful calibration.

'''Pseudocode:'''
  1. Load model configuration
  2. Apply quantization config (4-bit NF4 with double quantization)
  3. Load weights with on-the-fly dequantization for compute

=== Step 3: Adapter Training ===
[[step::Principle:Low_Rank_Adaptation]]

Inject low-rank adapter matrices into the frozen base model's attention and feedforward layers. Only these small adapter weights are trained, dramatically reducing memory requirements and training time while preserving the base model's capabilities.

'''What happens:'''
* Original weight matrix W remains frozen
* Two small matrices A and B are added: W' = W + BA
* Only A and B are updated during training (typically <1% of total parameters)
```

**❌ WRONG (actual code in workflow step):**
```mediawiki
=== Step 1: Load Model ===
[[step::Principle:Model_Loading]]
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(...)  # ← WRONG! Code belongs in Implementation
```
```

**✅ CORRECT (natural language description):**
```mediawiki
=== Step 1: Load Model ===
[[step::Principle:Model_Loading]]

Initialize the language model with memory-optimized settings. The loader applies 4-bit quantization automatically and patches attention layers for efficient training on consumer GPUs.
```

---

## 4. Visualization

### `== Execution Diagram ==`
**Instruction:** Create a Mermaid flowchart of the steps.
*   **Purpose:** Visual overview of the process flow.
*   **Format:** `{{#mermaid:graph TD ... }}`
*   **Nodes:** Use the Step Names.
*   **Edges:** Show the logical flow (usually linear or branching).

**Sample:**
```mediawiki
== Execution Diagram ==
{{#mermaid:graph TD
    A[Data Preparation] --> B[Model Quantization]
    B --> C[Adapter Training]
}}
```

---

## 5. Graph Connections

### `== Related Pages ==`
**Instruction:** Define outgoing connections using semantic wiki links.

Workflow pages have outgoing connections to:

*   **Principle:** `[[step::Principle:{Principle_Name}]]`
    *   *Meaning:* "This workflow executes this theory as a step."
*   **Heuristic:** `[[uses_heuristic::Heuristic:{Heuristic_Name}]]`
    *   *Meaning:* "This whole process is guided by this wisdom."

**Sample:**
```mediawiki
== Related Pages ==
* [[step::Principle:Data_Formatting]]
* [[step::Principle:Quantization]]
* [[step::Principle:Low_Rank_Adaptation]]
* [[uses_heuristic::Heuristic:Gradient_Checkpointing_Optimization]]
* [[uses_heuristic::Heuristic:Packing_Short_Sequences]]
```

**Connection Types for Workflow:**
| Edge Property | Target Node | Meaning |
|:--------------|:------------|:--------|
| `step` | Principle | "This workflow executes this theory as a step" |
| `uses_heuristic` | Heuristic | "This process is guided by this wisdom" |

---

## 6. WorkflowIndex Update (CRITICAL)

After creating a Workflow page, you **MUST** update the `_WorkflowIndex.md` file with detailed implementation context for each step. This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).

### Why This Matters

The WorkflowIndex preserves **implementation context** that Phase 2 needs to create correct Principle→Implementation mappings. Without this context, Phase 2 must guess which APIs implement which Principles, leading to incorrect connections.

### Required Information Per Step

For each workflow step, capture in the WorkflowIndex:

| Field | Description | Example |
|-------|-------------|---------|
| **Principle** | The linked Principle name | `Model_Loading` |
| **Implementation** | Suggested Implementation page name | `FastLanguageModel_from_pretrained` |
| **API Call** | Exact function/method signature | `FastLanguageModel.from_pretrained(model_name, load_in_4bit, ...)` |
| **Source Location** | File path and line numbers | `unsloth/models/loader.py:L120-620` |
| **External Dependencies** | Libraries outside the repo | `transformers`, `bitsandbytes` |
| **Environment** | Required environment page | `unslothai_unsloth_CUDA` |
| **Key Parameters** | Important params with types | `model_name: str`, `load_in_4bit: bool` |
| **Inputs** | What this step consumes | Model name/path |
| **Outputs** | What this step produces | `Tuple[PeftModel, Tokenizer]` |

### WorkflowIndex Structure

```markdown
## Workflow: {Workflow_Name}

**File:** [→](./workflows/{filename}.md)
**Description:** One-line description.

### Steps Overview

| # | Step Name | Principle | Implementation | Status |
|---|-----------|-----------|----------------|--------|
| 1 | Step One | Principle_A | `api_call_a` | ✅ |
| 2 | Step Two | Principle_B | `api_call_b` | ✅ |

### Step 1: Principle_A

| Attribute | Value |
|-----------|-------|
| **Principle** | `repo_Principle_A` |
| **Implementation** | `repo_Implementation_A` |
| **API Call** | `ClassName.method(param1, param2)` |
| **Source Location** | `path/to/file.py:L100-200` |
| **External Dependencies** | `library1`, `library2` |
| **Environment** | `repo_Environment_X` |
| **Key Parameters** | `param1: type`, `param2: type` |
| **Inputs** | Description of inputs |
| **Outputs** | Description of outputs |
```

### Implementation Types to Document

| Type | When to Use | Example |
|------|-------------|---------|
| **API Doc** | Unsloth function/class | `FastLanguageModel.from_pretrained` |
| **Wrapper Doc** | External API with Unsloth usage | `SFTTrainer` (TRL) |
| **Pattern Doc** | User-defined pattern | `reward_function_interface` |
| **External Tool Doc** | CLI/external tool | `llama_cli_validation` |

### Extraction Hints for Phase 2

Include a summary section at the end of each workflow listing all APIs to extract:

```markdown
### Implementation Extraction Guide

| Principle | Implementation | API | Source |
|-----------|----------------|-----|--------|
| Model_Loading | `FastLanguageModel_from_pretrained` | `from_pretrained` | `loader.py` |
| LoRA_Injection | `get_peft_model` | `get_peft_model` | `llama.py` |
```

### 1:1 Principle-Implementation Mapping

**Key Rule:** Each Principle gets its own Implementation page, even if the underlying API is the same.

Example: `FastLanguageModel.from_pretrained` is used by:
- `Model_Loading` → Create `FastLanguageModel_from_pretrained` (QLoRA angle)
- `RL_Model_Loading` → Create `FastLanguageModel_from_pretrained_vllm` (vLLM angle)
- `Model_Preparation` → Create `FastLanguageModel_from_pretrained_lora` (reload LoRA angle)

Each Implementation documents the API **from that Principle's perspective** with relevant parameters and use cases.

