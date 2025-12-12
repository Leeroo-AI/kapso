{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Paper|DeepSeekMath GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|TRL GRPOTrainer|https://huggingface.co/docs/trl]]
|-
! Domains
| [[domain::LLMs]], [[domain::RLHF]], [[domain::Reasoning]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
End-to-end process for improving model reasoning capabilities using Group Relative Policy Optimization (GRPO) reinforcement learning with Unsloth.

=== Description ===
This workflow uses GRPO to enhance a model's reasoning abilities, particularly for math, coding, and logic tasks. GRPO generates multiple responses per prompt, scores them with a reward function, and trains the model to prefer higher-scored responses using group-relative normalization. Unsloth's optimizations make GRPO training feasible on consumer GPUs.

=== Usage ===
Execute this workflow when you want to improve model reasoning beyond what SFT can achieve. Requires a base model (ideally already SFT'd) and a reward function or verifier. Best for tasks with verifiable correctness (math, code execution, factual questions).

== Execution Steps ==
=== Step 1: Load Pre-trained Model ===
[[step::Principle:Quantization]]
Load a base model, ideally one that has been SFT'd on relevant tasks.

<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
</syntaxhighlight>

=== Step 2: Add LoRA Adapters ===
[[step::Principle:Low_Rank_Adaptation]]
Configure LoRA for GRPO training.

<syntaxhighlight lang="python">
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    use_gradient_checkpointing = "unsloth",
)
</syntaxhighlight>

=== Step 3: Define Reward Function ===
[[step::Principle:Group_Relative_Policy_Optimization]]
Create a reward function that scores response quality.

<syntaxhighlight lang="python">
def reward_function(prompts, responses, ground_truths):
    """
    Score responses based on correctness.
    Returns list of rewards (floats).
    """
    rewards = []
    for prompt, response, gt in zip(prompts, responses, ground_truths):
        # Example: Check if answer is correct
        if extract_answer(response) == gt:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# Or use a reward model
# from transformers import AutoModelForSequenceClassification
# reward_model = AutoModelForSequenceClassification.from_pretrained("...")
</syntaxhighlight>

=== Step 4: Prepare Dataset ===
[[step::Principle:Group_Relative_Policy_Optimization]]
Format dataset with prompts and optional ground truth for reward computation.

<syntaxhighlight lang="python">
from datasets import load_dataset

# Math reasoning dataset example
dataset = load_dataset("openai/gsm8k", "main", split="train")

def format_grpo(example):
    return {
        "prompt": example["question"],
        "ground_truth": example["answer"].split("####")[-1].strip()
    }

dataset = dataset.map(format_grpo)
</syntaxhighlight>

=== Step 5: Configure GRPO Training ===
[[step::Principle:Group_Relative_Policy_Optimization]]
Set up GRPOTrainer with appropriate hyperparameters.

<syntaxhighlight lang="python">
from trl import GRPOTrainer, GRPOConfig

args = GRPOConfig(
    output_dir = "grpo_outputs",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 1e-5,
    max_steps = 500,
    num_generations = 4,
    max_new_tokens = 512,
    optim = "adamw_8bit",
    logging_steps = 1,
)

trainer = GRPOTrainer(
    model = model,
    args = args,
    train_dataset = dataset,
    tokenizer = tokenizer,
    reward_fn = reward_function,
)
</syntaxhighlight>

=== Step 6: Train and Save ===
Execute training and save the improved model.

<syntaxhighlight lang="python">
# Train
trainer.train()

# Save
model.save_pretrained("grpo_model")
tokenizer.save_pretrained("grpo_model")
</syntaxhighlight>

== Execution Diagram ==
{{#mermaid:graph TD
    A[Load Base Model] --> B[Add LoRA Adapters]
    B --> C[Define Reward Function]
    C --> D[Prepare Prompt Dataset]
    D --> E[Configure GRPOTrainer]
    E --> F[GRPO Training Loop]
    F --> G[Generate N Responses]
    G --> H[Score with Reward]
    H --> I[Normalize Within Group]
    I --> J[Policy Gradient Update]
    J --> F
    F --> K[Save Improved Model]
}}

== Related Pages ==
=== Execution Steps ===
* [[step::Principle:Quantization]] - Step 1
* [[step::Principle:Low_Rank_Adaptation]] - Step 2
* [[step::Principle:Group_Relative_Policy_Optimization]] - Steps 3-5

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Unsloth_Gradient_Checkpointing_Optimization]]
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]
* [[uses_heuristic::Heuristic:Batch_Size_Optimization]]

