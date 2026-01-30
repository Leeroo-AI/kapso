# Principle Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for a **Principle** page. Every section is mandatory to ensure the graph remains theoretically sound and executable.

---

## Page Title Requirements (WikiMedia Compliance)

### Naming Format
```
{repo_namespace}_{Principle_Name}.md
```

### WikiMedia Syntax Rules
1. **First character capitalized** — Auto-converted by system
2. **Underscores only** — Use `_` as word separator (NO hyphens, NO spaces)
3. **Case-sensitive after first character**

### Forbidden Characters
Never use: `#`, `<`, `>`, `[`, `]`, `{`, `}`, `|`, `+`, `:`, `/`, `-` (hyphen)

### Examples
| Correct | Incorrect | Issue |
|---------|-----------|-------|
| `Owner_Repo_Model_Loading.md` | `Owner-Repo_Model-Loading.md` | Hyphens |
| `Owner_Repo_LoRA_Configuration.md` | `owner_repo_lora_configuration.md` | Lowercase |
| `Owner_Repo_Gradient_Checkpointing.md` | `Owner_Repo_Gradient/Checkpointing.md` | Slash |

---

## 0. Page Title (REQUIRED - First Line)

**Goal:** Provide a human-readable H1 title as the very first line of the page.

**Format:** `# Principle: {Page_Name}`

Where `{Page_Name}` is the page name WITHOUT the repo namespace prefix.

**Sample:**
```mediawiki
# Principle: Model_Loading
```

For a file named `Owner_Repo_Model_Loading.md`, the title is:
- ✅ `# Principle: Model_Loading` (correct - no repo prefix)
- ❌ `# Principle: Owner_Repo_Model_Loading` (wrong - includes repo prefix)

---

## 1. Metadata Block (Top of Page)
**Goal:** Provide structured context for the graph parser.
**Format:** Semantic MediaWiki Table (Right-aligned).

### Fields Explanation
1.  **Knowledge Sources:** The theoretical provenance.
    *   *Syntax:* `[[source::{Type}|{Title}|{URL}]]`
    *   *Types:* `Paper` (Arxiv), `Blog` (Explanation), `Textbook`.
2.  **Domains:** Categorization tags.
    *   *Syntax:* `[[domain::{Tag}]]`
    *   *Examples:* `Deep_Learning`, `Optimization`, `Data_Science`.
3.  **Last Updated:** Freshness marker.
    *   *Syntax:* `[[last_updated::{YYYY-MM-DD HH:MM GMT}]]`

**Sample:**
```mediawiki
{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Attention Is All You Need|https://arxiv.org/abs/1706.03762]]
* [[source::Blog|Illustrated Transformer|https://jalammar.github.io/illustrated-transformer/]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2023-11-20 14:00 GMT]]
|}
```

---

## 2. Overview Block (The "Card")

### `== Overview ==`
**Instruction:** Define the concept in **one clear sentence**.
*   *Purpose:* The "Headline" for search results.
*   *Content:* "A {Type of Algorithm/Mechanism} that {Primary Function}."
*   *Constraint:* Must be abstract (no library names).

**Sample:**
```mediawiki
== Overview ==
Mechanism that allows neural networks to weigh the importance of different input tokens dynamically based on their relevance to each other.
```

### `=== Description ===` (The "What")
**Instruction:** Detailed educational explanation.
*   *Content:*
    1.  **Definition:** What is it?
    2.  **Problem Solved:** What limitation of previous methods does it fix? (e.g., "Solves the vanishing gradient problem in RNNs").
    3.  **Context:** Where does it fit in the ML landscape?
*   *Goal:* A student reading this should understand *what* the concept is without seeing code.

**Sample:**
```mediawiki
=== Description ===
Self-Attention is a mechanism relating different positions of a single sequence in order to compute a representation of the sequence. It addresses the critical limitation of Recurrent Neural Networks (RNNs) in handling long-range dependencies by allowing the model to "attend" to any state in the past directly, regardless of distance. This parallelization capability is what enables the scalability of Transformer models.
```

### `=== Usage ===` (The "When")
**Instruction:** Define the **Design/Architecture Trigger**.
*   *Purpose:* Decision support for System Design.
*   *Content:* Under what conditions is this the *right choice*?
    *   *Task Type:* (e.g., "Sequence-to-Sequence tasks").
    *   *Constraint:* (e.g., "When parallel training is required").
*   *Goal:* Answer "Why should I add this block to my architecture?"

**Sample:**
```mediawiki
=== Usage ===
Use this principle when designing architectures for sequence modeling tasks (NLP, Time Series) where capturing long-term context is critical and parallel training is required. It is the fundamental building block of Modern Large Language Models (LLMs) and should be preferred over RNNs for large-scale data.
```

---

## 3. The Core Theory

### `== Theoretical Basis ==`
**Instruction:** The "Math" or "Logic".
*   *Purpose:* Defines the mechanism rigorously.
*   *Content:* Key equations (using `<math>` tags) or logical steps.
*   *Goal:* Distinguish this principle from others (e.g., how Attention differs from Convolution).

**⚠️ Code Policy:**
*   **Pseudo-code IS allowed** — to describe algorithms at an abstract level.
*   **Actual implementation code is NOT allowed** — Principle pages are the abstraction layer. Real code belongs in the linked Implementation pages.

**Sample:**
```mediawiki
== Theoretical Basis ==
The core operation is a scaled dot-product attention:
<math>
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
</math>
Where Q (Query), K (Key), and V (Value) are projections of the input sequence.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Abstract algorithm description (NOT real implementation)
scores = Q @ K.transpose() / sqrt(d_k)
weights = softmax(scores)
output = weights @ V
</syntaxhighlight>
```

---

## 4. Graph Connections

### `== Related Pages ==`
**Instruction:** Define outgoing connections using semantic wiki links.

Principle pages have outgoing connections to:

*   **Implementation:** `[[implemented_by::Implementation:{Implementation_Name}]]`
    *   *Meaning:* "This theory is realized by this code."
    *   *Constraint:* **MANDATORY** — Must have exactly ONE dedicated implementation.
*   **Heuristic:** `[[uses_heuristic::Heuristic:{Heuristic_Name}]]`
    *   *Meaning:* "This theory is optimized by this wisdom."

**Sample:**
```mediawiki
== Related Pages ==
* [[implemented_by::Implementation:PyTorch_MultiheadAttention]]
* [[uses_heuristic::Heuristic:FlashAttention_Optimization]]
```

**Connection Types for Principle:**
| Edge Property | Target Node | Meaning | Constraint |
|:--------------|:------------|:--------|:-----------|
| `implemented_by` | Implementation | "This theory runs via this code" | **MANDATORY (1:1)** |
| `uses_heuristic` | Heuristic | "Optimized by this wisdom" | Optional |

---

## 5. 1:1 Principle-Implementation Mapping (CRITICAL)

### The Rule

**Each Principle has exactly ONE dedicated Implementation page.** Even if multiple Principles use the same underlying API, each gets its own Implementation that documents the API from that Principle's perspective.

### Why 1:1 Mapping?

1. **Clear ownership:** Each Principle knows exactly where its code documentation lives.
2. **Context-specific docs:** The same API can have different important parameters depending on use case.
3. **No confusion:** Engineers following a Principle land on documentation tailored to their goal.
4. **Maintainability:** Updates to one use case don't affect others.

### Example: Same API, Different Implementations

`FastLanguageModel.from_pretrained()` is used by three Principles:

| Principle | Implementation | Angle/Context |
|-----------|----------------|---------------|
| `Model_Loading` | `FastLanguageModel_from_pretrained` | QLoRA loading, 4-bit quantization |
| `RL_Model_Loading` | `FastLanguageModel_from_pretrained_vllm` | vLLM fast inference mode |
| `Model_Preparation` | `FastLanguageModel_from_pretrained_lora` | Reloading saved LoRA adapters |

Each Implementation page:
- Documents the same underlying API
- Emphasizes parameters relevant to that use case
- Provides examples tailored to that workflow context
- Links to the appropriate Environment pages

### Implementation Naming Convention

When the same API serves multiple Principles, use suffixes to distinguish:

```
{repo}_{APIName}              → Primary/default use case
{repo}_{APIName}_{context}    → Specialized use cases
```

Examples:
- `unslothai_unsloth_FastLanguageModel_from_pretrained` (default QLoRA)
- `unslothai_unsloth_FastLanguageModel_from_pretrained_vllm` (RL with vLLM)
- `unslothai_unsloth_get_peft_model` (SFT LoRA)
- `unslothai_unsloth_get_peft_model_rl` (RL high-rank LoRA)

### What Goes in the WorkflowIndex

The `_WorkflowIndex.md` should specify which Implementation each Principle links to:

```markdown
| Principle | Implementation | API | Angle |
|-----------|----------------|-----|-------|
| Model_Loading | `FastLanguageModel_from_pretrained` | `from_pretrained` | QLoRA |
| RL_Model_Loading | `FastLanguageModel_from_pretrained_vllm` | `from_pretrained` | vLLM |
```

This ensures Phase 2 creates the correct Implementation pages with correct mappings.

