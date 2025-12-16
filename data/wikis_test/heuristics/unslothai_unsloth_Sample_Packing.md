# Heuristic: unslothai_unsloth_Sample_Packing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL Documentation|https://huggingface.co/docs/trl/]]
|-
! Domains
| [[domain::Training]], [[domain::Data_Processing]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

Enable sample packing for >2x faster training by concatenating short sequences to fill the context window.

### Description

Sample packing (also called sequence packing) concatenates multiple training examples into single sequences up to `max_seq_length`. Benefits:

1. **Eliminates padding waste**: Short examples don't waste compute on pad tokens
2. **Better GPU utilization**: Full context windows maximize throughput
3. **Faster training**: 2x+ speedup on datasets with variable-length examples

Unsloth auto-enables packing when `packing=True` in SFTConfig, or uses padding-free batching as fallback.

### Usage

Use this heuristic when:
- Training on datasets with variable-length examples
- Seeking maximum training throughput
- GPU utilization seems low

## The Insight (Rule of Thumb)

* **Action:** Set `packing=True` in training config
* **Alternative:** Padding-free batching (auto-enabled if packing unavailable)
* **Trade-off:** ~2x+ speedup, but requires proper attention masking
* **Blocklist:** Some models don't support packing:
  - `gemma2`: Soft-capped attention issues
  - `gpt_oss`: Flex attention incompatibility

### Enabling Packing

```python
from trl import SFTConfig

config = SFTConfig(
    packing=True,  # Enable sample packing
    max_seq_length=2048,
    # ...
)
```

### Auto-Detection

Unsloth automatically:
1. Checks if packing is supported for the model
2. Falls back to padding-free batching if packing fails
3. Prints status message: "🦥 Unsloth: Packing enabled - training is >2x faster!"

## Reasoning

Without packing, a batch of examples with lengths [100, 500, 200, 800] gets padded to [800, 800, 800, 800], wasting 62.5% of compute.

With packing, examples are concatenated with attention masks to prevent cross-contamination:
```
[example1][example2][example3]...[exampleN][PAD]
```

Attention masks ensure each example only attends to itself, maintaining correct gradients.

## Code Evidence

From `unsloth/trainer.py:56-59`:
```python
PADDING_FREE_BLOCKLIST = {
    "gemma2",  # - gemma2:  Uses slow_attention_softcapping which has torch.compile issues
    "gpt_oss",  # - gpt_oss: Uses Flex Attention which doesn't handle padding_free correctly
}
```

From `unsloth/trainer.py:393-395`:
```python
if (not blocked and trainer_packing and ...):
    enable_sample_packing(self.model, self)
    print("🦥 Unsloth: Packing enabled - training is >2x faster and uses less VRAM!")
```

From `unsloth/trainer.py:396-403`:
```python
elif not blocked and trainer_padding_free:
    enable_padding_free_metadata(self.model, self)
    message = (
        "🦥 Unsloth: Padding-free auto-enabled, enabling faster training."
        if auto_padding_free_active
        else "🦥 Unsloth: Padding-free enabled, enabling faster training."
    )
    print(message)
```

## Related Pages

* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Principle:unslothai_unsloth_SFT_Training]]
