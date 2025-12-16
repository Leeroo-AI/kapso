# Principle Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for a **Principle** page. Every section is mandatory to ensure the graph remains theoretically sound and executable.

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
    *   *Constraint:* **MANDATORY** — Must list at least one implementation.
*   **Heuristic:** `[[uses_heuristic::Heuristic:{Heuristic_Name}]]`
    *   *Meaning:* "This theory is optimized by this wisdom."

**Sample:**
```mediawiki
== Related Pages ==
* [[implemented_by::Implementation:PyTorch_MultiheadAttention]]
* [[implemented_by::Implementation:TensorFlow_Attention_Layer]]
* [[uses_heuristic::Heuristic:FlashAttention_Optimization]]
```

**Connection Types for Principle:**
| Edge Property | Target Node | Meaning | Constraint |
|:--------------|:------------|:--------|:-----------|
| `implemented_by` | Implementation | "This theory runs via this code" | **MANDATORY (1+)** |
| `uses_heuristic` | Heuristic | "Optimized by this wisdom" | Optional |

