{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|RL Guide|https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide]]
* [[source::Blog|GRPO Blog|https://unsloth.ai/blog/grpo]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::GRPO]], [[domain::Reasoning]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==
End-to-end process for training reasoning-enhanced Large Language Models using Group Relative Policy Optimization (GRPO) with Unsloth's memory-efficient RL implementation.

=== Description ===
This workflow covers reinforcement learning fine-tuning using GRPO and related algorithms (GSPO, DrGRPO, DAPO, PPO) to enhance model reasoning capabilities. Unsloth's RL implementation provides:

* **Memory Efficiency**: 80% less VRAM than standard implementations through optimized gradients
* **vLLM Integration**: Fast batch generation using vLLM for efficient sampling during RL
* **Automatic Patching**: TRL trainers are automatically optimized when importing Unsloth
* **Long Context RL**: Support for extended context lengths during reasoning

Supported RL algorithms:
* **GRPO**: Group Relative Policy Optimization (recommended)
* **GSPO**: Group Supervised Policy Optimization (vision models)
* **DrGRPO**: Dr. GRPO variant with reward scaling
* **DAPO**: Dynamic Advantage Policy Optimization
* **PPO**: Proximal Policy Optimization
* **DPO/ORPO/KTO**: Preference-based methods

=== Usage ===
Execute this workflow when:
* You want to enhance a model's reasoning, math, or coding capabilities
* You have a reward function or reward model to guide training
* You need to train models to follow complex instructions more accurately
* You have access to 24GB+ VRAM (vLLM sampling is GPU-intensive)

Input requirements:
* Base model (can be previously SFT-trained)
* Prompt dataset with diverse reasoning tasks
* Reward function or reward model
* GPU with 24GB+ VRAM (for vLLM colocate mode)

Output:
* Model with enhanced reasoning/instruction-following capabilities
* Trained LoRA adapters or merged model

== Execution Steps ==

=== Step 1: Import Unsloth and TRL Components ===
[[step::Principle:unslothai_unsloth_RL_Setup]]
Import Unsloth first to enable automatic TRL patching, then import the GRPO trainer and config.

```python
# IMPORTANT: Import unsloth FIRST to enable RL optimizations
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("grpo", FastLanguageModel)

import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
```

=== Step 2: Load Model with vLLM Fast Inference ===
[[step::Principle:unslothai_unsloth_RL_Model_Loading]]
Load the model with `fast_inference=True` to enable vLLM for efficient batch generation during RL.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length = 4096,
    load_in_4bit = True,
    dtype = None,
    # Enable vLLM for fast RL sampling
    fast_inference = True,
    max_lora_rank = 64,
    gpu_memory_utilization = 0.6,  # Reserve memory for gradients
)
```

=== Step 3: Apply LoRA Adapters for RL ===
[[step::Principle:unslothai_unsloth_RL_LoRA_Setup]]
Apply LoRA with higher rank for RL training to capture complex reasoning patterns.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,  # Higher rank for RL (32-128 typical)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

=== Step 4: Prepare RL Dataset with Prompts ===
[[step::Principle:unslothai_unsloth_RL_Data_Preparation]]
Prepare the dataset with prompts for generation. Unlike SFT, RL only needs prompts - responses are generated during training.

```python
# Load dataset with reasoning prompts
dataset = load_dataset("json", data_files="reasoning_prompts.json", split="train")

# Format prompts using chat template
def format_prompt(examples):
    prompts = []
    for query in examples["query"]:
        messages = [
            {"role": "system", "content": "You are a helpful reasoning assistant."},
            {"role": "user", "content": query}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return {"prompt": prompts}

dataset = dataset.map(format_prompt, batched=True)
```

=== Step 5: Define Reward Function ===
[[step::Principle:unslothai_unsloth_Reward_Definition]]
Create a reward function that scores model outputs. This guides the RL optimization.

```python
import re

def reward_function(completions, prompts):
    """
    Example reward function for math/reasoning tasks.
    Returns list of rewards (one per completion).
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        reward = 0.0

        # Reward for structured reasoning
        if "<think>" in completion and "</think>" in completion:
            reward += 1.0

        # Reward for final answer format
        if "\\boxed{" in completion or "Answer:" in completion:
            reward += 1.0

        # Penalize very short responses
        if len(completion.split()) < 10:
            reward -= 1.0

        # Add custom reward logic (e.g., check correctness)
        # ...

        rewards.append(reward)
    return rewards
```

=== Step 6: Configure GRPO Training ===
[[step::Principle:unslothai_unsloth_GRPO_Configuration]]
Set up the GRPO configuration with optimized hyperparameters.

```python
training_args = GRPOConfig(
    # Basic settings
    output_dir = "grpo_outputs",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,

    # GRPO-specific settings
    num_generations = 8,         # Samples per prompt
    max_new_tokens = 1024,       # Max generation length
    temperature = 0.7,           # Sampling temperature

    # RL hyperparameters
    beta = 0.001,                # KL penalty coefficient
    loss_type = "bnpo",          # GRPO loss variant

    # Training settings
    learning_rate = 5e-6,        # Lower LR for RL
    max_steps = 500,
    logging_steps = 1,
    save_steps = 100,

    # Optimization
    optim = "adamw_8bit",
    warmup_ratio = 0.1,

    # Precision
    bf16 = torch.cuda.is_bf16_supported(),
    fp16 = not torch.cuda.is_bf16_supported(),
)
```

=== Step 7: Initialize and Run GRPO Training ===
[[step::Principle:unslothai_unsloth_GRPO_Training]]
Create the trainer and run the RL optimization loop.

```python
trainer = GRPOTrainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    processing_class = tokenizer,
    reward_funcs = reward_function,
)

# Run training
trainer.train()
```

=== Step 8: Save RL-Trained Model ===
[[step::Principle:unslothai_unsloth_RL_Model_Saving]]
Save the trained model - adapters or merged weights.

```python
# Save LoRA adapters
model.save_pretrained("grpo_lora_model")
tokenizer.save_pretrained("grpo_lora_model")

# Or merge and save
model.save_pretrained_merged(
    "grpo_merged_model",
    tokenizer,
    save_method = "merged_16bit",
)

# Push to Hub
model.push_to_hub_merged(
    "your-username/model-grpo",
    tokenizer,
    save_method = "merged_16bit",
    token = "hf_...",
)
```

== Execution Diagram ==
{{#mermaid:graph TD
    A[Import Unsloth & PatchFastRL] --> B[Load Model with fast_inference]
    B --> C[Apply High-Rank LoRA]
    C --> D[Prepare Prompt Dataset]
    D --> E[Define Reward Function]
    E --> F[Configure GRPO Training]
    F --> G[Initialize GRPOTrainer]
    G --> H[Run RL Training Loop]
    H --> I{Training Complete}
    I --> J[Save LoRA Adapters]
    I --> K[Merge & Save Model]
    I --> L[Push to Hub]
}}

== GRPO Configuration Reference ==

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 8 | Number of completions per prompt |
| `beta` | 0.001 | KL divergence penalty |
| `loss_type` | "bnpo" | Loss variant (grpo/bnpo/dr_grpo/dapo) |
| `temperature` | 0.7 | Sampling temperature |
| `max_new_tokens` | 1024 | Maximum generation length |
| `vllm_mode` | "colocate" | vLLM mode (colocate/server) |

== Related Pages ==
* [[step::Principle:unslothai_unsloth_RL_Setup]]
* [[step::Principle:unslothai_unsloth_RL_Model_Loading]]
* [[step::Principle:unslothai_unsloth_RL_LoRA_Setup]]
* [[step::Principle:unslothai_unsloth_RL_Data_Preparation]]
* [[step::Principle:unslothai_unsloth_Reward_Definition]]
* [[step::Principle:unslothai_unsloth_GRPO_Configuration]]
* [[step::Principle:unslothai_unsloth_GRPO_Training]]
* [[step::Principle:unslothai_unsloth_RL_Model_Saving]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_RL_Hyperparameters]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_LoRA_Rank_Selection]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Optimization]]
