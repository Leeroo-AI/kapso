{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|RL Guide|https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide]]
* [[source::Doc|GRPO Guide|https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide#training-with-grpo]]
|-
! Domains
| [[domain::LLMs]], [[domain::Reinforcement_Learning]], [[domain::GRPO]], [[domain::Reasoning]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

== Overview ==
End-to-end process for training reasoning models using Group Relative Policy Optimization (GRPO) with 80% less VRAM and 2x faster training than standard implementations.

=== Description ===
This workflow covers Reinforcement Learning from Human Feedback (RLHF) using GRPO, a simplified alternative to PPO that eliminates the need for a separate critic model. GRPO is particularly effective for training reasoning capabilities in LLMs.

The process covers:
1. **Base Model Preparation**: Load and apply LoRA adapters to base model
2. **Reward Function Design**: Define multiple reward functions for format, correctness, and reasoning
3. **GRPO Training**: Execute online RL with group-relative advantage estimation
4. **Evaluation**: Benchmark reasoning capabilities (e.g., AIME math problems)
5. **Model Export**: Save trained reasoning model

Key features:
- **vLLM Integration**: Fast inference for generation during training
- **Multi-reward Functions**: Combine format, correctness, and custom rewards
- **Memory Efficient**: 80% less VRAM than standard PPO implementations
- **FP8 Support**: Optional FP8 training for even lower memory usage
- **Hybrid Training**: Combine SFT warm-up with GRPO fine-tuning

=== Usage ===
Execute this workflow when:
- You want to train reasoning capabilities (math, coding, logic)
- You have reward functions to guide model behavior
- You need more control than SFT provides
- You want to align model outputs with specific formats or criteria

Input requirements:
- A base model (already fine-tuned via SFT recommended)
- Dataset with prompts and expected answers
- Reward functions (format checking, answer verification)
- GPU with sufficient VRAM (16GB+ recommended for 7B models)

== Execution Steps ==

=== Step 1: Load Model with vLLM Inference ===
[[step::Principle:unslothai_unsloth_Model_Loading]]
Load the base model with `fast_inference=True` to enable vLLM-accelerated generation during GRPO training.

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False,  # False for LoRA 16-bit
    fast_inference = True,  # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8,
)
```

=== Step 2: Apply LoRA Configuration ===
[[step::Principle:unslothai_unsloth_LoRA_Configuration]]
Configure LoRA adapters with appropriate rank for RL training. Higher ranks (64, 128) improve reasoning capacity.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

=== Step 3: Configure Chat Template ===
[[step::Principle:unslothai_unsloth_Data_Formatting]]
Apply the appropriate chat template for structured reasoning outputs.

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
```

=== Step 4: Define Reward Functions ===
[[step::Principle:unslothai_unsloth_Reward_Functions]]
Create reward functions that guide the model toward desired behavior. Combine format compliance, answer correctness, and reasoning quality rewards.

```python
import re

# Reward for exact format matching
def match_format_exactly(completions, **kwargs):
    """Reward for following <reasoning>...</reasoning><answer>...</answer> format"""
    pattern = r"^[\s]*<reasoning>.+?</reasoning>.*?<answer>.+?</answer>[\s]*$"

    responses = [completion[0]["content"] for completion in completions]
    rewards = [
        3.0 if re.match(pattern, response, re.DOTALL) else 0.0
        for response in responses
    ]
    return rewards

# Reward for correct answers
def check_answer_correctness(prompts, completions, answer, **kwargs):
    """Reward for matching expected answer"""
    def extract_answer(text):
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return re.sub(r"[%$,]", "", match.group(1)).strip() if match else ""

    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_answer(r) for r in responses]

    scores = []
    for guess, true_answer in zip(extracted, answer):
        if not guess:
            scores.append(0)
        elif guess == true_answer:
            scores.append(3.0)
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    scores.append(1.0)
                else:
                    scores.append(-1.5)
            except:
                scores.append(-1.5)
    return scores
```

=== Step 5: Prepare Training Dataset ===
[[step::Principle:unslothai_unsloth_Data_Formatting]]
Format the dataset with prompts and expected answers. GRPO uses prompt-answer pairs for online RL.

```python
from datasets import load_dataset

# Load dataset (e.g., GSM8K for math reasoning)
dataset = load_dataset("openai/gsm8k", "main", split="train")

system_prompt = """You are given a problem. Think step by step.
Place your reasoning between <reasoning> and </reasoning>.
Provide your final answer between <answer> and </answer>."""

def format_dataset(example):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"].split("####")[1].strip(),
    }

dataset = dataset.map(format_dataset)
```

=== Step 6: Execute GRPO Training ===
[[step::Principle:unslothai_unsloth_GRPO_Training]]
Run GRPO training with TRL's GRPOTrainer. The trainer generates multiple completions per prompt and uses group-relative advantages.

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate = 5e-6,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch_fused",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 8,  # Completions per prompt
    max_prompt_length = 512,
    max_completion_length = 1536,
    max_steps = 1000,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = "grpo_outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        check_answer_correctness,
    ],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()
```

=== Step 7: Evaluate and Export ===
[[step::Principle:unslothai_unsloth_Model_Export]]
Evaluate the trained model on benchmarks and export for deployment.

```python
# Save checkpoint
model.save_pretrained("grpo_checkpoint")
tokenizer.save_pretrained("grpo_checkpoint")

# Save merged model for deployment
model.save_pretrained_merged(
    "grpo_merged_model",
    tokenizer,
    save_method = "merged_16bit",
)

# Or save as GGUF
model.save_pretrained_gguf(
    "grpo_model_gguf",
    tokenizer,
    quantization_method = "q4_k_m",
)
```

== Execution Diagram ==
{{#mermaid:graph TD
    A[Step 1: Load Model + vLLM] --> B[Step 2: Apply LoRA]
    B --> C[Step 3: Configure Chat Template]
    C --> D[Step 4: Define Reward Functions]
    D --> E[Step 5: Prepare Dataset]
    E --> F[Step 6: GRPO Training]
    F --> G[Step 7: Evaluate & Export]

    subgraph "GRPO Training Loop"
        F1[Generate Completions] --> F2[Compute Rewards]
        F2 --> F3[Calculate Group Advantages]
        F3 --> F4[Policy Gradient Update]
        F4 --> F1
    end
}}

== Related Pages ==
* [[step::Principle:unslothai_unsloth_Model_Loading]]
* [[step::Principle:unslothai_unsloth_LoRA_Configuration]]
* [[step::Principle:unslothai_unsloth_Data_Formatting]]
* [[step::Principle:unslothai_unsloth_Reward_Functions]]
* [[step::Principle:unslothai_unsloth_GRPO_Training]]
* [[step::Principle:unslothai_unsloth_Model_Export]]
