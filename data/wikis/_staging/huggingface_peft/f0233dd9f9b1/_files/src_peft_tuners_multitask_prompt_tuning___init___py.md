# File: `src/peft/tuners/multitask_prompt_tuning/__init__.py`

**Category:** Prompt Tuning Implementation

| Property | Value |
|----------|-------|
| Lines | 25 |
| Imports | config, model, peft |

## Understanding

**Status:** Fully explored

**Purpose:** Package initialization file that exports Multitask Prompt Tuning components and registers the method with PEFT.

**Mechanism:**
1. Imports core components: `MultitaskPromptTuningConfig`, `MultitaskPromptTuningInit`, `MultitaskPromptEmbedding`
2. Exports all classes through `__all__`
3. Registers multitask_prompt_tuning as PEFT method:
   - Method name: "multitask_prompt_tuning"
   - Config class: `MultitaskPromptTuningConfig`
   - Model class: `MultitaskPromptEmbedding`

**Significance:** This enables Multitask Prompt Tuning, an extension of standard prompt tuning that shares knowledge across multiple tasks. Based on research from MIT-IBM Watson (https://huggingface.co/papers/2303.02861), it learns task-specific prompts that can leverage shared representations and transfer learning from source tasks to target tasks. This is particularly useful for few-shot learning scenarios and multi-task learning where tasks share common structure.

## Key Components

- **MultitaskPromptTuningInit**: Enum defining initialization strategies
- **MultitaskPromptTuningConfig**: Configuration for multitask prompt tuning
- **MultitaskPromptEmbedding**: Main implementation class
- **Registration**: Makes method available throughout PEFT ecosystem
