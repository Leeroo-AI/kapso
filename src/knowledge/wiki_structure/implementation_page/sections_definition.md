# Implementation Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for an **Implementation** page. Every section is mandatory to ensuring code is executable and correctly interfaced.

---

## 1. Metadata Block (Top of Page)
**Goal:** Provide structured context for the graph parser.
**Format:** Semantic MediaWiki Table (Right-aligned).

### Fields Explanation
1.  **Knowledge Sources:** HIGH-LEVEL references only (not file paths!).
    *   *Syntax:* `[[source::{Type}|{Title}|{URL}]]`
    *   *Types:*
        - `Repo` → Link to the **repository root URL**, not individual files
        - `Doc` → Official documentation websites
        - `Paper` → Academic papers (arXiv, etc.)
        - `Blog` → Blog posts, tutorials
    *   ⚠️ **DO NOT put file paths here** (e.g., `unsloth/models/loader.py`)
    *   File paths and line numbers belong in `== Code Reference ==` section below
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
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2023-11-20 14:00 GMT]]
|}
```

**❌ WRONG (file paths in Knowledge Sources):**
```mediawiki
* [[source::Repo|FastVisionModel Loader|unsloth/models/loader.py]]  ← WRONG!
```

**✅ CORRECT (high-level repo URL):**
```mediawiki
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]  ← RIGHT!
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

### `== Code Reference ==`
**Instruction:** Provide **exact, executable code reference** with source location.
*   *Purpose:* Enable agents to locate, understand, and call this code.
*   *Format:* Three parts: Source Location, Signature, and Import Statement.

#### Source Location (REQUIRED)
Link to the exact GitHub file with line numbers.
*   *Format:* `[{GitHub_URL}#L{start}-L{end} {file_path}]`
*   *Content:* Direct link to the source file with line range.

#### Code Signature (REQUIRED)
The complete function/class signature with all parameters and types.
*   *Format:* `syntaxhighlight` block with language tag.
*   *Content:* Full signature including default values and type hints.

#### Import Statement (REQUIRED)
Exact import needed to use this code.
*   *Format:* `syntaxhighlight` block.
*   *Content:* The `from X import Y` or `import X` statement.

**Sample:**
```mediawiki
== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L297-L450 src/transformers/trainer.py]
* '''Lines:''' 297-450

=== Signature ===
<syntaxhighlight lang="python">
class Trainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        """
        Args:
            model: The model to train, evaluate or use for predictions.
            args: TrainingArguments with hyperparameters.
            data_collator: Function to form a batch from dataset elements.
            train_dataset: Dataset for training.
            eval_dataset: Dataset(s) for evaluation.
            tokenizer: Tokenizer for processing text.
            compute_metrics: Function to compute metrics during evaluation.
            callbacks: List of callbacks to customize training loop.
            optimizers: Tuple of (optimizer, scheduler) to use.
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
</syntaxhighlight>
```

---

### `== I/O Contract ==`
**Instruction:** Define Inputs and Outputs rigorously with types and shapes.
*   *Purpose:* Defines the data interface for chaining this tool with others.
*   *Format:* Structured tables for clarity.

#### Inputs (Consumes)
Document each input parameter:
*   **Name:** Parameter name
*   **Type:** Python type or class
*   **Required:** Yes/No
*   **Description:** What this input represents

#### Outputs (Produces)
Document each output:
*   **Name:** Output name or return value
*   **Type:** Python type or class
*   **Description:** What is produced

**Sample:**
```mediawiki
== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || The model to train
|-
| args || TrainingArguments || Yes || Hyperparameters and config
|-
| train_dataset || Dataset || Yes || Training data
|-
| eval_dataset || Dataset || No || Evaluation data (optional)
|-
| tokenizer || PreTrainedTokenizer || No || For padding/truncation
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| train() returns || TrainOutput || Contains global_step, training_loss, metrics
|-
| checkpoints || Files || Saved to args.output_dir every args.save_steps
|-
| logs || Dict || Training metrics logged to args.logging_dir
|}
```

---

### `== Usage Examples ==`
**Instruction:** Provide **complete, runnable code examples**.
*   *Purpose:* Show exactly how to use this implementation in practice.
*   *Format:* One or more `syntaxhighlight` blocks with comments.
*   *Content:* Real code that can be copy-pasted and executed.

#### Requirements for Examples:
1. **Complete:** Include all imports and setup
2. **Runnable:** Code should work if copy-pasted
3. **Commented:** Explain what each step does
4. **Realistic:** Use plausible values and data

**Sample:**
```mediawiki
== Usage Examples ==

=== Basic Training ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset

# 1. Load model and dataset
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
dataset = load_dataset("glue", "mrpc")

# 2. Define training arguments
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# 3. Create trainer and train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# 4. Run training
trainer.train()

# 5. Save final model
trainer.save_model("./final_model")
</syntaxhighlight>

=== With Custom Metrics ===
<syntaxhighlight lang="python">
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    """Custom metrics function for Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,  # Pass custom metrics
)
</syntaxhighlight>
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

