# Implementation: dataset_mapping_pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Datasets|https://huggingface.co/docs/datasets]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Data_Engineering]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Pattern specification for preparing datasets for GRPO reinforcement learning, transforming raw data into the prompt format required by GRPOTrainer.

=== Description ===

This is a **Pattern Doc** - it documents a user-defined interface that must be implemented for GRPO training. The pattern specifies how to transform raw datasets into the format GRPOTrainer expects: a dataset with a "prompt" column containing formatted prompts ready for generation.

Key requirements:
* Output must have "prompt" column
* Prompts should use the configured chat template
* Should not include assistant responses (model generates these)

=== Usage ===

Implement this pattern as a mapping function applied to your raw dataset. The function should format each example as a prompt using `tokenizer.apply_chat_template()`.

== Interface Specification ==

=== Required Output Format ===
<syntaxhighlight lang="python">
# Dataset must have "prompt" column
{
    "prompt": str  # Formatted prompt for generation
}
</syntaxhighlight>

=== Mapping Function Signature ===
<syntaxhighlight lang="python">
def format_for_grpo(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Transform a dataset example into GRPO format.

    Args:
        example: Raw dataset row with problem/question fields

    Returns:
        Dict with "prompt" key containing formatted string
    """
    # Format as chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]}
    ]

    # Apply template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # Add assistant turn start
    )

    return {"prompt": prompt}
</syntaxhighlight>

== Usage Examples ==

=== Math Reasoning Dataset ===
<syntaxhighlight lang="python">
from datasets import load_dataset

# Define system prompt for reasoning
SYSTEM_PROMPT = """You are a helpful assistant. For math problems:
1. Think step-by-step inside <think></think> tags
2. Provide final answer in \\boxed{} format"""

# Load raw dataset
dataset = load_dataset("openai/gsm8k", "main")["train"]

# Define mapping function
def format_prompt(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return {"prompt": prompt}

# Apply mapping
dataset = dataset.map(format_prompt)

# Now dataset has "prompt" column ready for GRPOTrainer
print(dataset[0]["prompt"])
</syntaxhighlight>

=== Code Generation Dataset ===
<syntaxhighlight lang="python">
from datasets import load_dataset

SYSTEM_PROMPT = "You are a coding assistant. Write clean, correct code."

dataset = load_dataset("my_code_problems")["train"]

def format_code_prompt(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {example['problem']}\n\nWrite a Python solution:"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return {"prompt": prompt}

dataset = dataset.map(format_code_prompt)
</syntaxhighlight>

=== Instruction Following Dataset ===
<syntaxhighlight lang="python">
from datasets import load_dataset

# No system prompt, simple instruction format
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")["train"]

def format_instruction(example):
    # Take first user message as prompt
    user_message = example["messages"][0]["content"]
    messages = [{"role": "user", "content": user_message}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return {"prompt": prompt}

dataset = dataset.map(format_instruction)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_RL_Dataset_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment]]
