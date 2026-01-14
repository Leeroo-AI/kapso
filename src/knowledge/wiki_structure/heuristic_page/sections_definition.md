# Heuristic Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for a **Heuristic** page. Every section is mandatory to ensure actionable wisdom is captured effectively.

---

## Page Title Requirements (WikiMedia Compliance)

### Naming Format
```
{repo_namespace}_{Heuristic_Name}.md
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
| `Owner_Repo_Batch_Size_Multiplier_Tip.md` | `Owner-Repo_Batch-Size-Multiplier.md` | Hyphens |
| `Owner_Repo_Gradient_Checkpointing_Tip.md` | `owner_repo_gradient_checkpointing_tip.md` | Lowercase |
| `Owner_Repo_Memory_Tips.md` | `Owner_Repo_Memory/Tips.md` | Slash |

---

## 0. Page Title (REQUIRED - First Line)

**Goal:** Provide a human-readable H1 title as the very first line of the page.

**Format:** `# Heuristic: {Page_Name}`

Where `{Page_Name}` is the page name WITHOUT the repo namespace prefix.

**Sample:**
```mediawiki
# Heuristic: Gradient_Checkpointing_Tip
```

For a file named `Owner_Repo_Gradient_Checkpointing_Tip.md`, the title is:
- ✅ `# Heuristic: Gradient_Checkpointing_Tip` (correct - no repo prefix)
- ❌ `# Heuristic: Owner_Repo_Gradient_Checkpointing_Tip` (wrong - includes repo prefix)

---

## 1. Metadata Block (Top of Page)
**Goal:** Provide structured context for the graph parser.
**Format:** Semantic MediaWiki Table (Right-aligned).

### Fields Explanation
1.  **Knowledge Sources:** Where did this wisdom come from?
    *   *Why:* Heuristics are often anecdotal. Tracking the source (e.g., a specific GitHub Issue or Blog) is critical for validation.
    *   *Syntax:* `[[source::{Type}|{Title}|{URL}]]`
    *   *Types:* `Discussion` (GitHub Issues/Discord), `Blog`, `Paper` (Empirical findings), `Experience` (Internal).
2.  **Domains:** Categorization tags.
    *   *Syntax:* `[[domain::{Tag}]]`
    *   *Examples:* `Optimization`, `Debugging`, `LLMs`, `Distributed_Training`.
3.  **Last Updated:** Freshness marker.
    *   *Syntax:* `[[last_updated::{YYYY-MM-DD HH:MM GMT}]]`

**Sample:**
```mediawiki
{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Discussion|OOM on A100 issue|https://github.com/huggingface/transformers/issues/1234]]
* [[source::Blog|Training Llama Tips|https://medium.com/ai-tips/llama]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2023-11-15 10:00 GMT]]
|}
```

---

## 2. Overview Block (The "Card")
This section helps users deciding if this advice is relevant to their problem.

### `== Overview ==`
**Instruction:** Write a single, concise sentence summary of the tactic.
*   *Purpose:* The search snippet.
*   *Do:* Be specific about the benefit (e.g., "Reduces VRAM usage by 50%") or the fix (e.g., "Fixes NaN loss in fp16").
*   *Don't:* Be vague ("A tip for training").

**Sample:**
```mediawiki
== Overview ==
Memory optimization technique using Gradient Checkpointing to reduce VRAM usage by 50-60%.
```

### `=== Description ===` (The "What")
**Instruction:** Explain the **Insight**.
*   *Purpose:* Describes the mechanism of the heuristic.
*   *Content:* What is the trick? How does it work conceptually?
*   *Example:* "Gradient Checkpointing trades compute for memory by not storing intermediate activations during the forward pass, recomputing them during the backward pass."

**Sample:**
```mediawiki
=== Description ===
Gradient Checkpointing (activation checkpointing) drastically reduces memory usage during training. Instead of storing all intermediate activations for the backward pass, it stores only a subset and recomputes the rest on-the-fly. This effectively trades a small increase in computation time (20-30%) for a massive reduction in peak memory usage (up to 50-60%).
```

### `=== Usage ===` (The "When")
**Instruction:** Define the **Trigger Condition**.
*   *Purpose:* Tells the agent/user *when* to apply this.
*   *Content:* Specific symptoms (OOM), constraints (Low VRAM), goals (Max Batch Size), or debugging states (NaN loss).
*   *Goal:* Answer "Why should I use this trick right now?"

**Sample:**
```mediawiki
=== Usage ===
Use this heuristic when you are **VRAM constrained** (e.g., getting CUDA OOM errors) or need to fit a model that is too large for your GPU memory. It is standard practice when fine-tuning 7B+ parameter models on consumer hardware (e.g., RTX 3090/4090).
```

---

## 3. The Core Wisdom

### `== The Insight (Rule of Thumb) ==`
**Instruction:** The actionable advice. Can be a Configuration Value, a Workflow Step, or a Code Fix.
*   *Format:* Bullet points are preferred.
*   *Fields:*
    *   **Action:** What to change? (e.g., "Set param X").
    *   **Value:** Specific numbers/settings (e.g., "lr=2e-4").
    *   **Trade-off:** What do you lose? (e.g., "Slower training", "Lower precision").

**Sample:**
```mediawiki
== The Insight (Rule of Thumb) ==
* **Action:** Set `gradient_checkpointing=True` in `TrainingArguments` or call `model.gradient_checkpointing_enable()`.
* **Value:** N/A (Boolean flag).
* **Trade-off:** Reduces VRAM usage by ~50-60% at the cost of ~20% slower training speed.
* **Compatibility:** Works with almost all Transformer models; requires `use_cache=False` during training.
```

### `== Reasoning ==`
**Instruction:** The "Why" and the "Proof".
*   *Purpose:* Justifies the advice/trade-off.
*   *Content:* Theoretical explanation OR Empirical evidence (e.g., "Observed 30% speedup in experiment X").
*   *Evidence:* Include benchmark tables or log snippets if available.

**Sample:**
```mediawiki
== Reasoning ==
Deep Transformers have massive activation maps (Batch x SeqLen x Hidden). Storing these for backprop is the primary VRAM bottleneck. Recomputing them is compute-bound but allows fitting significantly larger batch sizes. Benchmarks on Llama-2-7b show VRAM dropping from 22GB to 11GB with this flag.
```

---

## 4. Graph Connections

### `== Related Pages ==`
**Instruction:** Document which pages reference this heuristic using semantic backlinks.

Heuristics are **Leaf Nodes** — they receive incoming connections from other pages. Use the `used_by` edge type to document these references.

**⚠️ IMPORTANT:** Only add backlinks for pages that ACTUALLY have a forward `[[uses_heuristic::Heuristic:X]]` link pointing to this heuristic. Do NOT add backlinks speculatively.

**Sample:**
```mediawiki
== Related Pages ==
* [[used_by::Implementation:Owner_Repo_Git_Fork_Edit_Workflow]]
* [[used_by::Implementation:Owner_Repo_Issue_To_PR_Conversion]]
* [[used_by::Principle:Owner_Repo_Memory_Optimization]]
```

**Edge Type for Heuristic Backlinks:**

| Edge Property | Meaning | Source Page Types |
|:--------------|:--------|:------------------|
| `used_by` | "This heuristic is used by X" | Implementation, Principle |

**Note:** Do NOT use `[[uses_heuristic::...]]` on Heuristic pages — that edge type belongs on the source pages pointing TO this heuristic.

**Why these source types?** Heuristics are practical tips that are referenced by Implementations (code-level optimizations) and Principles (theoretical considerations). Workflows no longer have wiki link connections — their implementations live in GitHub repositories.
