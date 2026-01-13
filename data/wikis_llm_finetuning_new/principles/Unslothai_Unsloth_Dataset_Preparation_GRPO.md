# Principle: Dataset_Preparation_GRPO

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO: Group Relative Policy Optimization|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|TRL Documentation|https://huggingface.co/docs/trl]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Data_Preprocessing]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for preparing datasets for GRPO reinforcement learning training, focusing on prompt-only format for on-policy generation.

=== Description ===

Dataset preparation for GRPO differs from supervised fine-tuning because:

1. **Prompts Only**: GRPO generates completions during training, so datasets only need prompts (not responses)
2. **Reward Computation**: Responses are scored by reward functions, not compared to ground truth
3. **SFT Warmup**: Optional supervised warmup requires prompt-response pairs before RL phase

The dataset structure depends on the training phase:
* **SFT Warmup Phase**: Standard prompt-response pairs (like QLoRA)
* **GRPO Phase**: Prompts only, completions generated on-the-fly

=== Usage ===

Use this principle when:
* Preparing data for GRPO or similar on-policy RL training
* The training involves generating completions and scoring them
* You need to separate warmup data from RL training data
* Building reward datasets for reasoning or math tasks

This step comes after model loading and before LoRA adapter setup.

== Theoretical Basis ==

'''GRPO Data Flow:'''
<syntaxhighlight lang="python">
# Pseudo-code for GRPO data requirements

# Phase 1: SFT Warmup (optional but recommended)
sft_dataset = {
    "prompt": "Solve: 2+2",
    "completion": "Let me solve this step by step...\n2+2 = 4\nFinal answer: 4"
}

# Phase 2: GRPO Training
grpo_dataset = {
    "prompt": "Solve: What is 15% of 80?"
    # No completion needed - model generates and reward function scores
}
</syntaxhighlight>

'''Dataset Schema for GRPO:'''
{| class="wikitable"
|-
! Phase !! Required Fields !! Optional Fields
|-
| SFT Warmup || prompt, completion || system_message
|-
| GRPO Training || prompt || answer (for reward verification)
|}

'''Why Prompts Only for GRPO:'''
In on-policy RL, the model must generate from its current policy to compute gradients:

<math>
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) A_t \right]
</math>

Using pre-computed responses would be off-policy and violate the gradient computation.

== Practical Guide ==

Since this is a user-defined pattern, here's how to prepare your data:

=== Step 1: Format for SFT Warmup ===
<syntaxhighlight lang="python">
from datasets import Dataset

# SFT warmup data needs prompt + completion
sft_data = [
    {
        "prompt": "What is 2+2?",
        "completion": "<think>\nI need to add 2 and 2.\n2 + 2 = 4\n</think>\n4"
    },
    # ... more examples
]
sft_dataset = Dataset.from_list(sft_data)
</syntaxhighlight>

=== Step 2: Format for GRPO Phase ===
<syntaxhighlight lang="python">
from datasets import Dataset

# GRPO data only needs prompts
grpo_data = [
    {"prompt": "What is 15% of 80?"},
    {"prompt": "Solve: 3x + 5 = 20"},
    # ... more examples
]
grpo_dataset = Dataset.from_list(grpo_data)
</syntaxhighlight>

=== Step 3: Apply Chat Template ===
<syntaxhighlight lang="python">
def format_prompt(example, tokenizer):
    messages = [{"role": "user", "content": example["prompt"]}]
    return {
        "prompt": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    }

grpo_dataset = grpo_dataset.map(
    lambda x: format_prompt(x, tokenizer)
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_Dataset_Preparation_GRPO_Pattern]]

