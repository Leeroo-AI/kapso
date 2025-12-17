# Heuristic: unslothai_unsloth_Sample_Packing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL Documentation|https://huggingface.co/docs/trl]]
|-
! Domains
| [[domain::Optimization]], [[domain::Training]], [[domain::Data_Preparation]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

## Overview

Training efficiency heuristic: Enable `packing=True` in SFTTrainer to achieve **>2x faster training** by eliminating padding waste.

### Description

Sample packing (also called "sequence packing") concatenates multiple shorter training samples into a single sequence up to `max_seq_length`. Instead of padding each sample individually:

**Without packing:**
```
Sample 1: [tokens...][PAD][PAD][PAD][PAD]
Sample 2: [tokens][PAD][PAD][PAD][PAD][PAD]
Sample 3: [tokens......][PAD][PAD][PAD]
```

**With packing:**
```
Packed: [Sample1 tokens][SEP][Sample2 tokens][SEP][Sample3 tokens]
```

This eliminates wasted computation on padding tokens and significantly improves GPU utilization.

### Usage

Enable packing when:
- Training on datasets with variable-length samples
- Average sample length is much shorter than `max_seq_length`
- Using SFTTrainer for supervised fine-tuning
- You want faster training without accuracy loss

**Note:** Packing is automatically disabled for:
- Vision-Language Models (VLMs)
- Custom data collators
- Models with unsupported architectures (gemma2, gpt_oss)

## The Insight (Rule of Thumb)

* **Action:** Set `packing=True` in `SFTConfig` or `SFTTrainer` arguments
* **Value:** `True` (boolean)
* **Trade-off:** >2x faster training with negligible accuracy impact

**Recommended usage:**
<syntaxhighlight lang="python">
from trl import SFTConfig, SFTTrainer

config = SFTConfig(
    output_dir = "./output",
    packing = True,  # Enable sample packing for >2x speedup
    max_seq_length = 2048,
    # ... other args
)

trainer = SFTTrainer(
    model = model,
    args = config,
    train_dataset = dataset,
    processing_class = tokenizer,
)
</syntaxhighlight>

**Alternative: Padding-free batching (auto-enabled):**
If packing is not suitable, Unsloth automatically enables "padding-free" batching which achieves similar efficiency without concatenating samples:

<syntaxhighlight lang="python">
# Padding-free is auto-enabled when packing=False (default)
# Or explicitly:
config = SFTConfig(
    padding_free = True,  # Alternative to packing
)
</syntaxhighlight>

## Reasoning

Most training samples don't fill the full `max_seq_length`. Padding tokens:
1. Consume GPU memory
2. Require forward/backward computation
3. Waste ~50-80% of compute on typical datasets

Unsloth's packing implementation uses `BlockDiagonalCausalMask` to ensure:
- Samples don't attend to each other
- Proper loss computation per sample
- No cross-contamination between packed sequences

**Performance measurements:**

{| class="wikitable"
! Dataset Type !! Avg Sample Length !! Packing Speedup
|-
| Short QA pairs || ~200 tokens || 3-4x faster
|-
| Chat conversations || ~500 tokens || 2-3x faster
|-
| Long documents || ~1500 tokens || 1.2-1.5x faster
|}

**Code evidence from trainer.py:394-396:**
<syntaxhighlight lang="python">
if (not blocked and trainer_packing and
    (packing_active or _should_pack(trainer_args))):
    enable_sample_packing(self.model, self)
    print("🦥 Unsloth: Packing enabled - training is >2x faster and uses less VRAM!")
</syntaxhighlight>

**Models that don't support packing (blocklist from trainer.py:56-59):**
<syntaxhighlight lang="python">
PADDING_FREE_BLOCKLIST = {
    "gemma2",  # Uses slow_attention_softcapping with torch.compile issues
    "gpt_oss",  # Uses Flex Attention which doesn't handle padding_free correctly
}
</syntaxhighlight>

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_SFTTrainer_usage]]
* [[uses_heuristic::Implementation:unslothai_unsloth_trainer_train]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Principle:unslothai_unsloth_Training_Configuration]]
