# Implementation Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for an **Implementation** page. Every section is mandatory to ensuring code is executable and correctly interfaced.

---

## Page Title Requirements (WikiMedia Compliance)

### Naming Format
```
{repo_namespace}_{Implementation_Name}.md
```

For angle-based implementations (same API documented from different Principle perspectives):
```
{repo_namespace}_{ClassName}_{Method}_For_{PrincipleContext}.md
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
| `Owner_Repo_FastLanguageModel_From_Pretrained.md` | `Owner-Repo_FastLanguageModel.from_pretrained.md` | Hyphens, dot |
| `Owner_Repo_SFTTrainer_Train.md` | `owner_repo_sfttrainer_train.md` | Lowercase |
| `Owner_Repo_Model_Save_For_GGUF.md` | `Owner_Repo_Model/Save_For_GGUF.md` | Slash |

---

## 0. Page Title (REQUIRED - First Line)

**Goal:** Provide a human-readable H1 title as the very first line of the page.

**Format:** `# Implementation: {Page_Name}`

Where `{Page_Name}` is the page name WITHOUT the repo namespace prefix.

**Sample:**
```mediawiki
# Implementation: FastLanguageModel_From_Pretrained
```

For a file named `Owner_Repo_FastLanguageModel_From_Pretrained.md`, the title is:
- ✅ `# Implementation: FastLanguageModel_From_Pretrained` (correct - no repo prefix)
- ❌ `# Implementation: Owner_Repo_FastLanguageModel_From_Pretrained` (wrong - includes repo prefix)

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
Specify the repository and file path. Line numbers are optional but recommended.
*   *Format:* Simple bullet list with Repository and File.
*   *Content:* Repository name/URL and the file path within the repo.
*   *Lines (Optional):* If included, line numbers should cover the **entire implementation** (class body, function body, all related code), NOT just the signature.

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
* '''File:''' src/transformers/trainer.py

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
**Instruction:** Define outgoing connections from this page using semantic wiki links.

Implementation pages have outgoing connections to:

*   **Environment:** `[[requires_env::Environment:{Env_Name}]]`
    *   *Meaning:* "I need this hardware/software context to run."
*   **Heuristic:** `[[uses_heuristic::Heuristic:{Heuristic_Name}]]`
    *   *Meaning:* "I have these known optimizations or tips."

**Sample:**
```mediawiki
== Related Pages ==
* [[requires_env::Environment:PyTorch_CUDA_11_8]]
* [[uses_heuristic::Heuristic:Memory_Management]]
* [[uses_heuristic::Heuristic:Batch_Size_Tips]]
```

**Connection Types for Implementation:**
| Edge Property | Target Node | Meaning |
|:--------------|:------------|:--------|
| `requires_env` | Environment | "Needs this context to run" |
| `uses_heuristic` | Heuristic | "Uses this optimization/tip" |

---

## 5. Principle-Conditioned Documentation (CRITICAL)

### The 1:1 Rule

**Each Implementation page is dedicated to exactly ONE Principle.** The same underlying API may have multiple Implementation pages if it serves different Principles.

### Why Principle-Conditioned?

The same API can be used in different contexts with different:
- **Important parameters:** QLoRA cares about `load_in_4bit`, RL cares about `fast_inference`
- **Typical values:** SFT uses `r=16`, RL uses `r=64`
- **Examples:** Different code snippets for different use cases
- **Environment requirements:** RL needs vLLM, basic loading doesn't

### How to Write Principle-Conditioned Docs

When creating an Implementation page, **always know which Principle it serves** (from the WorkflowIndex). Then:

1. **Title reflects the angle:**
   - `FastLanguageModel_from_pretrained` (default QLoRA)
   - `FastLanguageModel_from_pretrained_vllm` (RL with vLLM)

2. **Overview mentions the context:**
   ```mediawiki
   == Overview ==
   Loads language models with vLLM fast inference backend for reinforcement learning workflows.
   ```

3. **Usage focuses on the specific trigger:**
   ```mediawiki
   === Usage ===
   Use this when setting up GRPO/PPO training with vLLM-accelerated generation.
   NOT for standard SFT training (use FastLanguageModel_from_pretrained instead).
   ```

4. **Key Parameters highlight what matters for this use case:**
   - For QLoRA: `load_in_4bit`, `dtype`, `max_seq_length`
   - For RL: `fast_inference`, `max_lora_rank`, `gpu_memory_utilization`

5. **Examples are tailored to the workflow:**
   - QLoRA example shows loading for fine-tuning
   - RL example shows loading with vLLM settings

### Implementation Types

| Type | Description | Example |
|------|-------------|---------|
| **API Doc** | Documents a function/class in the repo | `FastLanguageModel.from_pretrained` |
| **Wrapper Doc** | Documents external API with repo-specific usage | `SFTTrainer` (from TRL, with Unsloth context) |
| **Pattern Doc** | Documents a user-defined pattern/interface | `reward_function_interface` |
| **External Tool Doc** | Documents CLI or external tool | `llama_cli_validation` |

### Wrapper Docs for External APIs

When documenting external APIs (like TRL's `SFTTrainer`), create a **Wrapper Doc** that:

1. Links to official documentation
2. Explains repo-specific usage and patches
3. Shows examples in the context of this repo's workflow
4. Documents which parameters are affected by the repo's optimizations

**Sample for TRL SFTTrainer wrapper:**
```mediawiki
== Overview ==
HuggingFace TRL's SFTTrainer for supervised fine-tuning, automatically optimized by Unsloth patches.

=== Description ===
When Unsloth is imported, it patches SFTTrainer to use:
* Fused cross-entropy loss
* Optimized gradient checkpointing
* Padding-free training

=== External Reference ===
* [https://huggingface.co/docs/trl/sft_trainer TRL SFTTrainer Documentation]

=== Unsloth-Specific Usage ===
...
```

### Pattern Docs for User-Defined Code

Some Principles don't have a library API—they're patterns users implement themselves. Document the **interface**, not an implementation:

**Sample for reward_function_interface:**
```mediawiki
== Overview ==
Interface specification for user-defined reward functions in GRPO training.

=== Description ===
Reward functions score model completions to guide reinforcement learning. This is a user-defined pattern, not a library API.

== Interface Specification ==
<syntaxhighlight lang="python">
def reward_function(
    completions: List[str],  # Generated completions
    prompts: List[str]       # Original prompts
) -> List[float]:            # Reward scores
    """
    Returns a list of float rewards, one per completion.
    Higher values indicate better completions.
    """
    ...
</syntaxhighlight>

== Example Implementations ==
=== Rule-Based Reward ===
...
=== Model-Based Reward ===
...
```

### Reading from WorkflowIndex

When creating Implementation pages in Phase 2, **always check the WorkflowIndex** for:

| Field | Use For |
|-------|---------|
| **Principle** | Know which Principle this Implementation serves |
| **API Call** | The exact function signature to document |
| **Source Location** | Where to find the code |
| **Key Parameters** | Which parameters to emphasize |
| **Inputs/Outputs** | For the I/O Contract section |
| **Environment** | Which Environment pages to link |
| **External Dependencies** | Whether this needs a Wrapper Doc |

