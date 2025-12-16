# Heuristic: unslothai_unsloth_Padding_Free_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|trainer.py|unsloth/trainer.py]]
|-
! Domains
| [[domain::Optimization]], [[domain::LLMs]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

## Overview

Padding-free (sample packing) training optimization that eliminates wasted computation on padding tokens, providing 2x+ speedup for variable-length sequences.

### Description

Traditional batching pads all sequences to the longest in the batch, wasting computation on padding tokens. Unsloth implements padding-free training that concatenates sequences with boundary markers, allowing efficient attention computation without padding waste.

Key benefits:
- **2x+ speedup**: No computation wasted on padding
- **Better GPU utilization**: Higher effective batch sizes
- **Memory efficiency**: More sequences per batch

Limitations:
- Not compatible with custom data collators
- Blocked for certain model types (Gemma2, GPT-OSS, VLMs)
- May interfere with some evaluation metrics

### Usage

Apply this heuristic when:
- Training on datasets with variable-length sequences
- Optimizing training throughput
- Working within VRAM constraints

## The Insight (Rule of Thumb)

### When to Enable

* **Auto-enabled by default**: Unsloth auto-enables padding-free for SFTTrainer
* **Explicit enable**: Set `packing=True` in SFTConfig
* **Explicit padding-free**: Set `padding_free=True` for lightweight packing

### Blocklist - Models That Don't Support It

From `trainer.py:56-59`:
```python
PADDING_FREE_BLOCKLIST = {
    "gemma2",   # Uses slow_attention_softcapping with torch.compile issues
    "gpt_oss",  # Uses Flex Attention which doesn't handle padding_free correctly
}
```

Additional blocks:
- **Vision-Language Models**: Custom collators required
- **ProcessorMixin models**: Have their own batching logic
- **Custom data collators**: May conflict with packing

### Disable Auto-Packing

Via environment variable:
```bash
export UNSLOTH_DISABLE_AUTO_PADDING_FREE=1
```

Or programmatically:
```python
config._unsloth_disable_auto_packing = True
```

### Checking If Active

After trainer initialization, check:
```python
if trainer.args.packing:
    print("Sample packing active")
elif trainer.args.padding_free:
    print("Padding-free batching active")
else:
    print("Standard batching")
```

### Output Message

When enabled, you'll see:
```
🦥 Unsloth: Packing enabled - training is >2x faster and uses less VRAM!
```
or:
```
🦥 Unsloth: Padding-free auto-enabled, enabling faster training.
```

## Reasoning

**Why padding wastes computation:**
With sequences of lengths [128, 256, 512, 1024], standard batching pads all to 1024:
- Total tokens computed: 4 * 1024 = 4096
- Actual content tokens: 128 + 256 + 512 + 1024 = 1920
- Waste ratio: 53%

**How packing works:**
Concatenate sequences with separator markers:
```
[SEQ1][SEP][SEQ2][SEP][SEQ3][SEP][SEQ4]
```
Use block-diagonal attention mask to prevent cross-sequence attention.

**Why certain models are blocked:**
- Gemma2: Uses softmax capping that breaks under torch.compile
- GPT-OSS: Flex Attention implementation incompatible
- VLMs: Image tokens require special handling

**Memory benefit:**
Without packing, batch_size=4 with max_len=1024 uses 4*1024=4096 positions.
With packing, same positions can fit ~8 average-sized sequences.

## Code Evidence

Auto-packing logic from `trainer.py:344-403`:
```python
packing_active = False
if _should_pack(config_arg) and not blocked:
    configure_sample_packing(config_arg)
    packing_active = True
    logger.info("Unsloth: Sample packing enabled for SFTTrainer instance.")

auto_padding_free_active = False
padding_free_requested = getattr(config_arg, "padding_free", None) is True
if not blocked:
    if padding_free_requested:
        configure_padding_free(config_arg)
    elif _should_auto_padding_free(config_arg):
        configure_padding_free(config_arg)
        auto_padding_free_active = True
        logger.info(
            "Unsloth: Padding-free batching auto-enabled for SFTTrainer instance."
        )
```

Blocking conditions from `trainer.py:316-326`:
```python
blocked = (
    (data_collator is not None)
    or isinstance(processing_class, ProcessorMixin)
    or is_vlm
    or is_unsupported_model
    or (os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1")
)
```

Skip message handling from `trainer.py:333-342`:
```python
if blocked and requested_pack:
    reason = "custom data collator"
    if data_collator is None and isinstance(processing_class, ProcessorMixin):
        reason = "processor-based model"
    elif is_vlm:
        reason = "vision-language model"
    elif is_unsupported_model:
        reason = f"unsupported model type(s): {', '.join(model_types)}"
    message = "Unsloth: Sample packing skipped " f"({reason} detected)."
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
