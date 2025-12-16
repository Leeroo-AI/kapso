# Heuristic: Learning Rate Guidelines

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

## Overview

Recommended learning rates for different fine-tuning scenarios with QLoRA and full fine-tuning.

### Description

Learning rate is one of the most critical hyperparameters. Unsloth provides guidance for different training modes and supports separate learning rates for embeddings vs. LoRA weights.

### Usage

Apply this heuristic when:
- Setting up TrainingArguments/SFTConfig
- Troubleshooting training instability
- Fine-tuning embeddings alongside LoRA adapters

## The Insight (Rule of Thumb)

### Standard QLoRA Learning Rates

| Scenario | Learning Rate | Notes |
|----------|---------------|-------|
| General Fine-tuning | 2e-4 | **Default recommendation** |
| Instruction Tuning | 1e-4 to 2e-4 | Standard range |
| Complex Tasks (Math/Code) | 1e-4 | Lower for stability |
| Short Training (<100 steps) | 5e-5 | Prevent overfitting |
| Long Training (>1000 steps) | 1e-4 with decay | Use cosine scheduler |

### Embedding Learning Rate

* **Action:** Use `embedding_learning_rate` for separate control
* **Default:** 5e-5 (10-20x lower than LoRA LR)
* **Reasoning:** Embeddings are more sensitive and can diverge with high LR

From `unsloth/trainer.py:142-143`:
```python
def _create_unsloth_optimizer(
    model, optimizer_cls, optimizer_kwargs,
    embedding_lr = 5e-5,  # Default embedding LR
):
```

From `unsloth/trainer.py:158-161`:
```python
if name.endswith("modules_to_save.default.weight"):
    print(f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
    param_groups["embeddings"][name] = param
```

### Model Size Scaling

| Model Size | Suggested LR | Batch Size Recommendation |
|------------|--------------|---------------------------|
| 1-3B | 2e-4 to 3e-4 | 8-16 |
| 7-8B | 2e-4 | 4-8 |
| 13B | 1e-4 to 2e-4 | 2-4 |
| 70B | 5e-5 to 1e-4 | 1-2 |

### Scheduler Recommendations

* **Short runs (<500 steps):** Linear or constant
* **Medium runs (500-2000 steps):** Cosine with warmup
* **Long runs (>2000 steps):** Cosine with warmup_ratio=0.03

### Warmup Guidelines

* **Warmup Steps:** 5-10% of total steps, or 10-100 steps minimum
* **Warmup Ratio:** 0.03-0.1 typical

## Reasoning

### Why 2e-4 as Default

The QLoRA paper (Dettmers et al., 2023) established 2e-4 as a strong baseline:
- Tested across multiple model sizes and tasks
- Works well with AdamW optimizer
- Higher than full fine-tuning (1e-5 to 5e-5) due to LoRA's lower parameter count

### Why Separate Embedding LR

Embeddings (and lm_head when unfrozen) are:
- Directly connected to token representations
- More prone to catastrophic forgetting
- Require more stable updates to preserve pre-trained knowledge

### Training Stability Indicators

Signs of LR being too high:
- Loss spikes or NaN values
- Outputs become repetitive or nonsensical
- Validation loss increases while training loss decreases

Signs of LR being too low:
- Loss decreases very slowly
- Model barely changes from base
- Many epochs needed for improvement

## Related Pages

### Used By

* [[uses_heuristic::Implementation:unslothai_unsloth_UnslothTrainer]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Principle:unslothai_unsloth_Supervised_Fine_Tuning]]
