# Heuristic: unslothai_unsloth_RL_Hyperparameters

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|rl.py|unsloth/models/rl.py]]
* [[source::Doc|TRL|https://github.com/huggingface/trl]]
* [[source::Discussion|GRPO Paper|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::LLMs]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

## Overview

Hyperparameter recommendations for GRPO/DPO/PPO reinforcement learning training with Unsloth, including loss types, temperature, and batch size configuration.

### Description

Reinforcement learning for LLMs requires careful hyperparameter tuning to achieve stable training and good reward optimization. Unsloth patches TRL trainers with optimized defaults and adds validation to prevent common mistakes.

Key parameters:
- **Loss type**: BNPO (per-token loss) vs GRPO (sequence-level)
- **Beta (KL coefficient)**: Controls deviation from reference model
- **Temperature**: Sampling temperature during generation
- **Batch/generation size**: Must align properly for GRPO

### Usage

Apply these heuristics when:
- Setting up GRPO, DPO, PPO, or other RL trainers
- Experiencing training instability or divergence
- Optimizing reward vs KL trade-offs
- Debugging NaN losses or mode collapse

## The Insight (Rule of Thumb)

### GRPO-Specific Defaults

From `rl.py:760-777`:
```python
# GRPO paper defaults
if trainer_file == "grpo_trainer":
    replacements = {
        "loss_type": "bnpo",     # Default GRPO paper
        "beta": 0.001,           # Recommended as seen in verl
        "auto_find_batch_size": False,  # Cannot work on GRPO
        "vllm_importance_sampling_correction": False,
    }
```

* **Loss Type**: Use `"bnpo"` (per-token loss) not `"grpo"` for better gradients
* **Beta**: `0.001` - much lower than TRL's old default of `0.04`
* **num_generations**: Default `8` - number of completions per prompt

### Batch Size Rules

**CRITICAL**: `per_device_train_batch_size * gradient_accumulation_steps * world_size` must be divisible by `num_generations`:

From `rl.py:866-877`:
```python
if steps_per_generation is None and generation_batch_size is None:
    ga = gradient_accumulation_steps
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if (ga * world_size * per_device_train_batch_size) % num_generations != 0:
        print('Unsloth: We will change the batch size to `num_generations`')
        per_device_train_batch_size = num_generations
```

### Temperature Validation

From `rl.py:895-903`:
```python
if temperature <= 0:
    raise MathError('Please set a positive non-zero temperature')
elif temperature >= 10:
    raise MathError('Please set temperature less than 10')
```

* **Recommended**: `0.6` to `1.0` for diverse yet coherent generations
* **Avoid**: `<= 0` (deterministic/broken) or `>= 10` (random noise)

### Learning Rate Validation

From `rl.py:779-784`:
```python
if learning_rate < 1e-7:
    print(f'Your learning rate of `{learning_rate}` is too small!')
if learning_rate > 1:
    print(f'Your learning rate of `{learning_rate}` is way too large!')
```

* **Recommended range**: `1e-6` to `5e-5`
* **For RL**: Often lower than SFT (`1e-6` to `2e-5`)

### Loss Type Variants

| Loss Type | Description | When to Use |
|-----------|-------------|-------------|
| `bnpo` | Binary NPO (per-token) | Default, most stable |
| `grpo` | Group RPO (sequence-level) | Original GRPO paper |
| `dr_grpo` | Dr. GRPO with reward scaling | When rewards vary widely |
| `dapo` | DAPO with truncation masking | When using `epsilon_high=0.28` |

### DAPO Configuration

From `rl.py:836-858`:
```python
elif loss_type.lower() == 'dapo':
    if mask_truncated_completions != True:
        print('The DAPO paper recommends `mask_truncated_completions = True`')
    if epsilon_high != 0.28:
        print('The DAPO paper recommends `epsilon_high = 0.28`')
    if beta != 0.0:
        print(f'The DAPO paper recommends setting `beta = 0.0` to remove the KL term')
    mask_truncated_completions = True
    epsilon_high = 0.28
```

## Reasoning

**Why beta=0.001?**
The KL penalty prevents the policy from diverging too far from the reference model. High beta (0.04) overly constrains learning; low beta (0.001) allows more exploration while maintaining stability.

**Why BNPO over GRPO?**
BNPO computes loss per-token, providing denser gradient signals. Sequence-level GRPO can lead to credit assignment issues on long sequences.

**Why batch_size must align with num_generations?**
GRPO requires grouping completions from the same prompt. If batch_size isn't divisible by num_generations, completions get misaligned across prompts.

## Code Evidence

Default RL configuration from `rl.py:721-752`:
```python
replacements = {
    "output_dir": None,
    "logging_nan_inf_filter": False,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "seed": 3407,
    "optim": "adamw_8bit",
    "learning_rate": 5e-05,
    "per_device_eval_batch_size": 4,
    "eval_accumulation_steps": 2,
    "torch_empty_cache_steps": 250,
    "logging_steps": 1,
    "max_seq_length": None,
    "num_generations": 8,
    "top_k": None,
    "vllm_mode": "colocate",
    "generation_kwargs": {},
    "bf16": False,
    "fp16": False,
    "report_to": "none",
    "include_tokens_per_second": False,
    "include_num_input_tokens_seen": False,
    "auto_find_batch_size": False,
    "dataloader_pin_memory": True,
}
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_PatchFastRL]]
* [[uses_heuristic::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
