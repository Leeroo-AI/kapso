# Implementation Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for an **Implementation** page. Every section is mandatory to ensuring code is executable and correctly interfaced.

---

## 1. Metadata Block (Top of Page)
**Goal:** Provide structured context for the graph parser.
**Format:** Semantic MediaWiki Table (Right-aligned).

### Fields Explanation
1.  **Knowledge Sources:** The provenance of this code definition.
    *   *Syntax:* `[[source::{Type}|{Title}|{URL}]]`
    *   *Types:* `Repo` (Source Code), `Doc` (API Reference), `Paper` (Theoretical Basis).
2.  **Domains:** Categorization tags.
    *   *Syntax:* `[[domain::{Tag}]]`
    *   *Examples:* `Vision`, `NLP`, `Preprocessing`, `Model_Architecture`.
3.  **Last Updated:** Freshness marker.
    *   *Syntax:* `[[last_updated::{YYYY-MM-DD HH:MM GMT}]]`

**Sample:**
```mediawiki
{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Trainer API|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2023-11-20 14:00 GMT]]
|}
```

---

## 2. Overview Block (The "Card")

### `== Overview ==`
**Instruction:** Write a single sentence defining the tool.
*   *Purpose:* Search snippet.
*   *Content:* "Concrete tool for {Functionality} provided by {Library}."

**Sample:**
```mediawiki
== Overview ==
Concrete tool for training Transformer models provided by the HuggingFace library.
```

### `=== Description ===` (The "What")
**Instruction:** Explain the **Code Entity**.
*   *Purpose:* Contextualize the code.
*   *Content:* What is this class/function? What library does it belong to? What is its primary role in the stack?

**Sample:**
```mediawiki
=== Description ===
The `Trainer` class provides a complete training loop for PyTorch models. It abstracts away the boilerplate of training (gradient accumulation, distributed training, logging) and integrates seamlessly with `TrainingArguments`.
```

### `=== Usage ===` (The "When")
**Instruction:** Define the **Execution Trigger**.
*   *Purpose:* Tells the agent when to import this.
*   *Content:* Specific task scenarios (e.g., "Fine-tuning on custom datasets").
*   *Goal:* Answer "When should I write `import ThisClass`?"

**Sample:**
```mediawiki
=== Usage ===
Import this class when you need to fine-tune a standard Transformer model on a dataset and want managed logging/checkpointing without writing a custom PyTorch loop.
```

---

## 3. Technical Specifications

### `== Code Signature ==`
**Instruction:** Provide the **Function/Class Signature**.
*   *Format:* `syntaxhighlight` block with appropriate language tag (e.g., `python`, `cpp`, `bash`).
*   *Content:* The class/function definition line and constructor/entry point. Crucial for agents to know how to instantiate or call it.

**Sample:**
```mediawiki
== Code Signature ==
<syntaxhighlight lang="python">
class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        ...
    ):
</syntaxhighlight>
```

### `== I/O Contract ==`
**Instruction:** Define Inputs and Outputs rigorously.
*   *Purpose:* Defines the data interface for chaining this tool with others.
*   *Format:* Bullet points.
*   *Content:*
    *   **Consumes:** Data types/classes required (e.g., `torch.Tensor`, `Pandas DataFrame`).
    *   **Produces:** Artifacts generated (e.g., `Model Weights`, `Metrics Dict`).

**Sample:**
```mediawiki
== I/O Contract ==
* **Consumes:**
    * `model`: A `PreTrainedModel` instance.
    * `dataset`: A HuggingFace `Dataset` object.
* **Produces:**
    * `checkpoints`: Saved model weights in the output directory.
    * `metrics`: A dictionary of loss/accuracy values.
```

---

## 4. Graph Connections

### `== Related Pages ==`
**Instruction:** Define outgoing and incoming edges.

#### `=== Context & Requirements ===`
*   *Direction:* Outgoing (Dependency).
*   *Concept:* What environment allows this to run?
*   *Syntax:* `* [[requires_env::Environment:{Env_Name}]]`

#### `=== Tips and Tricks ===`
*   *Direction:* Outgoing (Attribute).
*   *Concept:* What heuristics apply to this code?
*   *Syntax:* `* [[uses_heuristic::Heuristic:{Heuristic_Name}]]`

**Sample:**
```mediawiki
== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:PyTorch_CUDA_11_8]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Gradient_Checkpointing_Optimization]]
```

