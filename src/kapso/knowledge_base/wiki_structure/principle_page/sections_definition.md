# Principle Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for a '''Principle''' page. Every section is mandatory to ensure the graph remains theoretically sound and executable.

'''IMPORTANT:''' All wiki pages use MediaWiki syntax, NOT Markdown. See the syntax reference at the end of this document.

---

## Page Title Requirements (WikiMedia Compliance)

### Naming Format
```
{repo_namespace}_{Principle_Name}.md
```

### WikiMedia Syntax Rules
# First letter capitalized — Auto-converted by system
# Underscores only — Use `_` as word separator (NO hyphens, NO spaces)
# Case-sensitive after first character

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

'''Goal:''' Provide a human-readable H1 title as the very first line of the page.

'''Format:''' `# Principle: {Page_Name}`

Where `{Page_Name}` is the page name WITHOUT the repo namespace prefix.

'''Sample:'''
```mediawiki
# Principle: Model_Loading
```

For a file named `Owner_Repo_Model_Loading.md`, the title is:
* ✅ `# Principle: Model_Loading` (correct - no repo prefix)
* ❌ `# Principle: Owner_Repo_Model_Loading` (wrong - includes repo prefix)

---

## 1. Metadata Block (Top of Page)

'''Goal:''' Provide structured context for the graph parser.

'''Format:''' Semantic MediaWiki Table (Right-aligned).

### Fields Explanation

'''Knowledge Sources:''' The theoretical provenance.
* ''Syntax:'' `[[source::{Type}|{Title}|{URL}]]`
* ''Types:'' `Paper` (Arxiv), `Blog` (Explanation), `Textbook`.

'''Domains:''' Categorization tags.
* ''Syntax:'' `[[domain::{Tag}]]`
* ''Examples:'' `Deep_Learning`, `Optimization`, `Data_Science`.

'''Last Updated:''' Freshness marker.
* ''Syntax:'' `[[last_updated::{YYYY-MM-DD HH:MM GMT}]]`

'''Sample:'''
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

'''Instruction:''' Define the concept in one clear sentence.
* ''Purpose:'' The "Headline" for search results.
* ''Content:'' "A {Type of Algorithm/Mechanism} that {Primary Function}."
* ''Constraint:'' Must be abstract (no library names).

'''Sample:'''
```mediawiki
== Overview ==

Mechanism that allows neural networks to weigh the importance of different input tokens dynamically based on their relevance to each other.
```

### `=== Description ===` (The "What")

'''Instruction:''' Detailed educational explanation.
* ''Content:''
*# '''Definition:''' What is it?
*# '''Problem Solved:''' What limitation of previous methods does it fix? (e.g., "Solves the vanishing gradient problem in RNNs").
*# '''Context:''' Where does it fit in the ML landscape?
* ''Goal:'' A student reading this should understand ''what'' the concept is without seeing code.

'''Sample:'''
```mediawiki
=== Description ===

Self-Attention is a mechanism relating different positions of a single sequence in order to compute a representation of the sequence. It addresses the critical limitation of Recurrent Neural Networks (RNNs) in handling long-range dependencies by allowing the model to "attend" to any state in the past directly, regardless of distance. This parallelization capability is what enables the scalability of Transformer models.
```

### `=== Usage ===` (The "When")

'''Instruction:''' Define the design/architecture trigger.
* ''Purpose:'' Decision support for System Design.
* ''Content:'' Under what conditions is this the ''right choice''?
** ''Task Type:'' (e.g., "Sequence-to-Sequence tasks").
** ''Constraint:'' (e.g., "When parallel training is required").
* ''Goal:'' Answer "Why should I add this block to my architecture?"

'''Sample:'''
```mediawiki
=== Usage ===

Use this principle when designing architectures for sequence modeling tasks (NLP, Time Series) where capturing long-term context is critical and parallel training is required. It is the fundamental building block of Modern Large Language Models (LLMs) and should be preferred over RNNs for large-scale data.
```

---

## 3. The Core Theory

### `== Theoretical Basis ==`

'''Instruction:''' The "Math" or "Logic".
* ''Purpose:'' Defines the mechanism rigorously.
* ''Content:'' Key equations (using `<math>` tags) or logical steps.
* ''Goal:'' Distinguish this principle from others (e.g., how Attention differs from Convolution).

'''⚠️ Code Policy:'''
* '''Pseudo-code IS allowed''' — to describe algorithms at an abstract level.
* '''Actual implementation code is NOT allowed''' — Principle pages are the abstraction layer. Real code belongs in the linked Implementation pages.

'''Sample:'''
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

'''Instruction:''' Define outgoing connections using semantic wiki links.

Principle pages have outgoing connections to:

* '''Implementation:''' `[[implemented_by::Implementation:{Implementation_Name}]]`
** ''Meaning:'' "This theory is realized by this code."
** ''Constraint:'' '''MANDATORY''' — At least one Implementation; link every page that realizes this theory.
* '''Heuristic:''' `[[uses_heuristic::Heuristic:{Heuristic_Name}]]`
** ''Meaning:'' "This theory is optimized by this wisdom."

'''Sample:'''
```mediawiki
== Related Pages ==

* [[implemented_by::Implementation:PyTorch_MultiheadAttention]]
* [[uses_heuristic::Heuristic:FlashAttention_Optimization]]
```

'''Connection Types for Principle:'''

| Edge Property | Target Node | Meaning | Constraint |
|:--------------|:------------|:--------|:-----------|
| `implemented_by` | Implementation | "This theory runs via this code" | '''MANDATORY (1+)''' |
| `uses_heuristic` | Heuristic | "Optimized by this wisdom" | Optional |

---

## 5. Principles and Implementations (CRITICAL)

### The Rule

'''One Implementation page per unit of code, linked by every Principle it realizes.''' A Principle links to at least one Implementation; when several Principles use the same API, they all link to its one page. Never create a second Implementation page for the same code from another Principle's perspective.

'''A Principle only for a concept with a theory behind it''' — a method, an algorithm, a design idea whose why holds outside this repository. A helper, a utility or glue code is documented inside the Implementation page of the code that uses it, not as a Principle.

### Example: Same API, One Implementation

`FastLanguageModel.from_pretrained()` is used by three Principles:

| Principle | Implementation |
|-----------|----------------|
| `Model_Loading` | `FastLanguageModel_from_pretrained` |
| `RL_Model_Loading` | `FastLanguageModel_from_pretrained` |
| `Model_Preparation` | `FastLanguageModel_from_pretrained` |

The one Implementation page:
* Documents the API once, with the parameters that matter in each context
* Carries an example per context (QLoRA loading, vLLM for RL, reloading adapters)
* Links to the Environment pages any of those contexts require

### What Goes in the WorkflowIndex

The `_WorkflowIndex.md` should specify which Implementation each Principle links to:

```markdown
| Principle | Implementation | API |
|-----------|----------------|-----|
| Model_Loading | `FastLanguageModel_from_pretrained` | `from_pretrained` |
| RL_Model_Loading | `FastLanguageModel_from_pretrained` | `from_pretrained` |
```

This ensures Phase 2 creates each Implementation page once, with all its links.

---

## MediaWiki Syntax Reference (CRITICAL)

'''Use MediaWiki syntax, NOT Markdown!''' This is critical for proper rendering.

### Text Formatting

| Format | MediaWiki (CORRECT) | Markdown (WRONG) |
|--------|---------------------|------------------|
| Bold | `'''bold text'''` | `**bold text**` |
| Italic | `''italic text''` | `*italic text*` |
| Bold+Italic | `'''''both'''''` | `***both***` |

### Headers (inside wiki pages)

```mediawiki
== Level 2 Header ==
=== Level 3 Header ===
==== Level 4 Header ====
```

NOT: `## Header` or `### Header`

### Lists

```mediawiki
* Bullet item 1
* Bullet item 2
** Nested bullet

# Numbered item 1
# Numbered item 2
## Nested numbered
```

### Whitespace Rules

* '''Blank line required''' after headers before content
* '''Blank line required''' between paragraphs
* '''No blank line''' between list items (unless separating groups)

