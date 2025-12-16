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
**Instruction:** The ordered list of steps.
*   *Structure:* Use Level 3 Headers (`===`) for each step.
*   *Content per Step:*
    1.  **Link:** `[[step::Principle:{Principle_Name}]]` (The Theory).
    2.  **Context:** Specifics for *this* workflow (e.g., "Use 4-bit quantization here").

**Sample:**
```mediawiki
== Execution Steps ==
=== Step 1: Data Preparation ===
[[step::Principle:Data_Formatting]]
Convert the raw JSON/CSV data into the specific prompt template structure (e.g., Alpaca format) required by the model.

=== Step 2: Model Quantization ===
[[step::Principle:Quantization]]
Load the base model in 4-bit precision (NF4) to fit in memory.

=== Step 3: Adapter Training ===
[[step::Principle:Low_Rank_Adaptation]]
Inject LoRA adapters into Linear layers and train only these adapters using the formatted data.
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

