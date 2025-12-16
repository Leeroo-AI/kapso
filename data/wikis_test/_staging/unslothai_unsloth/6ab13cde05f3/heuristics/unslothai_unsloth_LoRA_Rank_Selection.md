# Heuristic: unslothai_unsloth_LoRA_Rank_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|llama.py|unsloth/models/llama.py]]
* [[source::Experience|Fine-tuning best practices]]
|-
! Domains
| [[domain::Optimization]], [[domain::LLMs]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

## Overview

LoRA rank selection heuristic balancing model capacity vs VRAM usage and training speed, with recommended values between 8-128 depending on task complexity.

### Description

LoRA (Low-Rank Adaptation) rank determines the dimensionality of the low-rank decomposition matrices. Higher rank provides more expressive power but increases memory usage and training time. The optimal rank depends on:

- **Task complexity**: Simple tasks (classification) need lower rank; complex tasks (reasoning) benefit from higher rank
- **Model size**: Larger base models may work with lower relative ranks
- **Available VRAM**: Higher ranks require more memory for adapter weights
- **RL training**: Reinforcement learning typically requires higher ranks (64-128) for stability

### Usage

Apply this heuristic when:
- Configuring `get_peft_model()` with LoRA parameters
- Encountering training instability with RL methods
- Optimizing for VRAM-constrained hardware
- Balancing training speed vs model quality

## The Insight (Rule of Thumb)

* **Action**: Set `r` (LoRA rank) in `get_peft_model()` based on task type
* **Values**:
  - **Simple SFT**: `r=8` to `r=16` - Fastest training, lowest memory
  - **General SFT**: `r=32` - Good balance for most fine-tuning
  - **Complex tasks**: `r=64` - Better quality for reasoning/coding
  - **RL training (GRPO/PPO)**: `r=64` to `r=128` - Required for stable RL updates
* **Trade-off**: Each doubling of rank increases adapter parameters ~2x and training time ~10-15%
* **lora_alpha**: Set `lora_alpha = r` or `lora_alpha = 2*r` for consistent scaling

### RL-Specific Guidance

From `rl.py` configuration defaults:
```python
replacements = {
    "max_lora_rank": 64,  # Default for vLLM LoRA
}
```

For GRPO training, higher ranks help maintain training stability during policy updates.

## Reasoning

The rank `r` determines how many parameters are added via LoRA. With rank `r`, each LoRA layer adds `r * (d_in + d_out)` parameters instead of `d_in * d_out` for the full weight matrix.

**Why higher ranks for RL:**
- RL requires larger gradient updates due to reward signal variance
- Low-rank constraints can bottleneck policy expressiveness
- GRPO/PPO benefit from more adapter capacity to capture reward-driven changes

**Memory formula:**
```
LoRA params per layer ≈ 2 * r * hidden_size
Total LoRA params ≈ n_layers * n_target_modules * 2 * r * hidden_size
```

For a 7B model with `r=32` targeting attention modules:
- ~16M LoRA parameters (0.2% of base model)

For `r=128`:
- ~64M LoRA parameters (0.9% of base model)

## Code Evidence

Default LoRA configuration pattern from `llama.py`:
```python
def get_peft_model(
    model,
    r = 16,  # Default rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    ...
):
```

RL trainer defaults from `rl.py:729-749`:
```python
replacements = {
    "max_seq_length": None,
    "num_generations": 8,
    "top_k": None,
    "vllm_mode": "colocate",
    "generation_kwargs": {},
}
```

vLLM LoRA rank limit from `loader.py:144-145`:
```python
max_lora_rank = 64,  # Maximum LoRA rank for vLLM
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_get_peft_model]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
