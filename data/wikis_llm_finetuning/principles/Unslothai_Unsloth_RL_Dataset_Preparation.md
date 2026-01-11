# Principle: RL_Dataset_Preparation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|HuggingFace Datasets|https://huggingface.co/docs/datasets]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Data_Engineering]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Technique for preparing datasets for reinforcement learning training, transforming raw data into prompt-only format for model generation and reward evaluation.

=== Description ===

RL Dataset Preparation differs fundamentally from SFT preparation:

| Aspect | SFT | RL (GRPO) |
|--------|-----|-----------|
| Dataset contains | Prompt + Response | Prompt only |
| Model role | Learn to reproduce response | Generate new responses |
| Responses | Fixed in dataset | Generated during training |
| Reward | Implicit (cross-entropy) | Explicit (reward function) |

For GRPO, the dataset provides prompts that the model completes during training. Multiple completions are generated per prompt and scored by reward functions.

=== Usage ===

Apply RL Dataset Preparation when:
* Training with GRPO, PPO, or similar on-policy RL
* You have problems/questions but want the model to learn solutions
* Reward functions will evaluate generated responses

Key requirements:
* "prompt" column with formatted prompts
* No ground truth responses (model generates these)
* Prompts should trigger the desired behavior (reasoning, coding, etc.)

== Theoretical Basis ==

=== On-Policy Data Flow ===

<math>
\text{Prompt} \xrightarrow{\text{Model}} \text{Completions} \xrightarrow{\text{Reward}} \text{Scores} \xrightarrow{\text{GRPO}} \nabla\theta
</math>

1. Sample prompt from dataset
2. Generate N completions with current policy
3. Score completions with reward function
4. Compute group-relative advantages
5. Update policy parameters

=== Prompt Design ===

Effective prompts should:
* Clearly state the task
* Include formatting instructions (if reward checks format)
* Set up for verifiable outputs

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# RL dataset preparation (abstract)
def prepare_rl_dataset(raw_dataset, tokenizer, system_prompt):
    def format_for_rl(example):
        # Build prompt with chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": example["question"]})

        # Format with template, add generation prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Start assistant turn
        )

        return {"prompt": prompt}

    return raw_dataset.map(format_for_rl)
</syntaxhighlight>

=== Dataset Size Considerations ===

RL training typically uses smaller datasets than SFT:
* Each prompt is used multiple times (different completions)
* Reward signal provides more information per example
* Focus on diverse, high-quality prompts over quantity

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_dataset_mapping_pattern]]
