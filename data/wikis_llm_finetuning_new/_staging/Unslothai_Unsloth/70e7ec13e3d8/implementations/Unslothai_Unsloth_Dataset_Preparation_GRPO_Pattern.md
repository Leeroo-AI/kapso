# Implementation: Dataset_Preparation_GRPO_Pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Datasets|https://huggingface.co/docs/datasets]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Data_Preprocessing]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Pattern documentation for preparing datasets for GRPO reinforcement learning training.

=== Description ===

This is a **Pattern Doc** describing the user-defined data preparation interface for GRPO training. There is no single library function to call; instead, users must format their data according to the expected schema.

GRPO requires different data formats for different phases:
* **SFT Warmup**: Standard conversational format with prompts and completions
* **GRPO Training**: Prompt-only format where completions are generated on-the-fly

=== Usage ===

Follow this pattern when preparing data for any GRPO or on-policy RL training. The exact implementation depends on your data source and format, but the output schema must match what GRPOTrainer expects.

== Interface Specification ==

=== SFT Warmup Dataset Schema ===
<syntaxhighlight lang="python">
# Required schema for SFT warmup phase
{
    "prompt": str,      # The user prompt/question
    "completion": str,  # The expected response (for SFT loss)
}

# Alternative: Full conversation format
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
}
</syntaxhighlight>

=== GRPO Training Dataset Schema ===
<syntaxhighlight lang="python">
# Minimal required schema for GRPO
{
    "prompt": str,  # The formatted prompt (with chat template applied)
}

# Optional: Include answer for reward verification
{
    "prompt": str,
    "answer": str,  # Ground truth for reward function verification
}
</syntaxhighlight>

== Example Implementations ==

=== Basic Math Dataset Preparation ===
<syntaxhighlight lang="python">
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
)

# Apply chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Load raw data
raw_data = [
    {"question": "What is 15% of 80?", "answer": "12"},
    {"question": "Solve: 3x + 5 = 20", "answer": "5"},
]

def format_for_grpo(example):
    """Format a single example for GRPO training."""
    messages = [
        {"role": "user", "content": example["question"]}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Add assistant prefix
    )
    return {
        "prompt": prompt,
        "answer": example["answer"],  # Keep for reward function
    }

# Create GRPO dataset
grpo_dataset = Dataset.from_list(raw_data)
grpo_dataset = grpo_dataset.map(format_for_grpo)

print(grpo_dataset[0]["prompt"])
# Output: <|begin_of_text|>...<|start_header_id|>user<|end_header_id|>
#         What is 15% of 80?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
</syntaxhighlight>

=== Two-Phase Dataset (SFT + GRPO) ===
<syntaxhighlight lang="python">
from datasets import Dataset

# Phase 1: SFT warmup data (with completions)
sft_data = [
    {
        "prompt": "What is 2+2?",
        "completion": "<think>\nI need to add 2 and 2.\n2 + 2 = 4\n</think>\nThe answer is 4."
    },
    {
        "prompt": "What is 3*4?",
        "completion": "<think>\nI need to multiply 3 by 4.\n3 * 4 = 12\n</think>\nThe answer is 12."
    },
]

def format_sft(example, tokenizer):
    """Format for SFT with full conversation."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(messages, tokenize=False)
    }

sft_dataset = Dataset.from_list(sft_data)
sft_dataset = sft_dataset.map(lambda x: format_sft(x, tokenizer))

# Phase 2: GRPO data (prompts only)
grpo_data = [
    {"question": "What is 25% of 200?", "answer": "50"},
    {"question": "Solve: 2x - 10 = 30", "answer": "20"},
]

def format_grpo(example, tokenizer):
    """Format for GRPO (prompt only)."""
    messages = [{"role": "user", "content": example["question"]}]
    return {
        "prompt": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        "answer": example["answer"],
    }

grpo_dataset = Dataset.from_list(grpo_data)
grpo_dataset = grpo_dataset.map(lambda x: format_grpo(x, tokenizer))
</syntaxhighlight>

=== Loading from HuggingFace Hub ===
<syntaxhighlight lang="python">
from datasets import load_dataset

# Load a math dataset from HuggingFace
raw_dataset = load_dataset("gsm8k", "main", split="train")

def format_gsm8k(example, tokenizer):
    """Format GSM8K for GRPO."""
    # GSM8K has 'question' and 'answer' fields
    messages = [{"role": "user", "content": example["question"]}]

    # Extract numeric answer from GSM8K format
    answer_text = example["answer"]
    # GSM8K answers end with "#### <number>"
    numeric_answer = answer_text.split("####")[-1].strip()

    return {
        "prompt": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        "answer": numeric_answer,
    }

grpo_dataset = raw_dataset.map(lambda x: format_gsm8k(x, tokenizer))
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| raw_data || List[Dict] or Dataset || Yes || Raw data with questions/prompts
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer with chat template applied
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| sft_dataset || Dataset || (Optional) Dataset with "text" column for SFT warmup
|-
| grpo_dataset || Dataset || Dataset with "prompt" column for GRPO training
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Dataset_Preparation_GRPO]]

