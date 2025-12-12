{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Paper|DPO Paper|https://arxiv.org/abs/2305.18290]]
* [[source::Doc|TRL DPOTrainer|https://huggingface.co/docs/trl/dpo_trainer]]
|-
! Domains
| [[domain::LLMs]], [[domain::RLHF]], [[domain::Alignment]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
End-to-end process for aligning language models with human preferences using Direct Preference Optimization (DPO) with Unsloth.

=== Description ===
This workflow aligns a fine-tuned model to produce preferred responses over rejected alternatives. DPO simplifies the traditional RLHF pipeline by eliminating the need for a separate reward model, directly optimizing on preference pairs. Use after SFT to improve response quality, safety, and helpfulness.

=== Usage ===
Execute this workflow when you have preference data (chosen vs rejected response pairs) and want to align model behavior. Typically applied after SFT training. Effective for reducing harmful outputs, improving helpfulness, and matching specific writing styles.

== Execution Steps ==
=== Step 1: Load SFT Model ===
[[step::Principle:Quantization]]
Load a model that has been supervised fine-tuned (SFT).

<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
</syntaxhighlight>

=== Step 2: Add LoRA Adapters ===
[[step::Principle:Low_Rank_Adaptation]]
Configure LoRA for DPO training.

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

=== Step 3: Prepare Preference Dataset ===
[[step::Principle:Direct_Preference_Optimization]]
Format dataset with prompt, chosen, and rejected columns.

<syntaxhighlight lang="python">
from datasets import load_dataset

# Load preference dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# Dataset should have columns:
# - prompt: The input/question
# - chosen: The preferred response
# - rejected: The dispreferred response

# Example format:
# {
#     "prompt": "What is the capital of France?",
#     "chosen": "The capital of France is Paris.",
#     "rejected": "I don't know the answer."
# }
</syntaxhighlight>

=== Step 4: Configure DPO Training ===
[[step::Principle:Direct_Preference_Optimization]]
Set up DPOTrainer with appropriate hyperparameters.

<syntaxhighlight lang="python">
from trl import DPOTrainer, DPOConfig

args = DPOConfig(
    output_dir = "dpo_outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,
    max_steps = 200,
    beta = 0.1,
    optim = "adamw_8bit",
    logging_steps = 1,
    max_length = 1024,
    max_prompt_length = 512,
)
</syntaxhighlight>

=== Step 5: Train with DPO ===
Execute DPO training.

<syntaxhighlight lang="python">
trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = args,
    train_dataset = dataset,
    tokenizer = tokenizer,
)

trainer.train()
</syntaxhighlight>

=== Step 6: Save Aligned Model ===
Save the preference-aligned model.

<syntaxhighlight lang="python">
model.save_pretrained("dpo_model")
tokenizer.save_pretrained("dpo_model")

model.save_pretrained_gguf("dpo_gguf", tokenizer, quantization_method="q4_k_m")
</syntaxhighlight>

== Execution Diagram ==
{{#mermaid:graph TD
    A[Load SFT Model] --> B[Add LoRA Adapters]
    B --> C[Prepare Preference Data]
    C --> D[Configure DPOTrainer]
    D --> E[DPO Training Loop]
    E --> F[Compute Chosen Log Probs]
    E --> G[Compute Rejected Log Probs]
    F --> H[DPO Loss Calculation]
    G --> H
    H --> E
    E --> I[Save Aligned Model]
}}

== Related Pages ==
=== Execution Steps ===
* [[step::Principle:Quantization]] - Step 1
* [[step::Principle:Low_Rank_Adaptation]] - Step 2
* [[step::Principle:Direct_Preference_Optimization]] - Steps 3-5

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Unsloth_Gradient_Checkpointing_Optimization]]
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]

