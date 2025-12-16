# Heuristic: unslothai_unsloth_LoRA_Dropout_Bias

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Discussion|Unsloth GitHub Issues|https://github.com/unslothai/unsloth/issues]]
|-
! Domains
| [[domain::LoRA]], [[domain::Fine_Tuning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

Performance guidance: Use `lora_dropout=0` and `bias="none"` to enable Unsloth's fast patching optimizations.

### Description

Unsloth's kernel optimizations provide significant speedups for LoRA layers, but require specific settings:

1. **`lora_dropout=0`**: Dropout in LoRA layers prevents kernel fusion
2. **`bias="none"`**: Bias terms in LoRA prevent certain optimizations

Using non-default values triggers warning messages and falls back to slower code paths.

### Usage

Use this heuristic when:
- Configuring LoRA parameters for maximum speed
- Understanding performance warnings during training
- Deciding between regularization and speed

## The Insight (Rule of Thumb)

* **Action:** Set `lora_dropout=0.0` and `bias="none"` in `get_peft_model()`
* **Default Values:** These are already the Unsloth defaults
* **Trade-off:**
  - `lora_dropout=0`: Lose dropout regularization, gain ~10-20% speed
  - `bias="none"`: Lose bias learning, gain optimized kernels

### When Dropout Might Help

- Very small datasets (<1000 examples): May need regularization
- Overfitting observed: Consider `lora_dropout=0.05-0.1`
- Accept the speed penalty for better generalization

### When Bias Might Help

- Task requires precise output calibration
- Base model has biases that need adjustment
- Accept slower training for potentially better results

## Reasoning

Unsloth's optimized forward passes fuse multiple operations (QKV projection, LoRA update, activation) into single kernels. Dropout and bias break these fusion patterns:

- **Dropout**: Requires random mask generation and element-wise multiplication between fused ops
- **Bias**: Adds extra memory accesses and prevents certain GEMM optimizations

In practice, most LoRA fine-tuning works well without dropout (the low-rank constraint provides implicit regularization).

## Code Evidence

From `unsloth/models/llama.py:2767-2777`:
```python
if lora_dropout != 0:
    logger.warning_once(
        f"Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = {lora_dropout}.\n"
        f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
    )

if bias != "none":
    logger.warning_once(
        f"Unsloth: bias = `none` is supported for fast patching. You are using bias = {bias}.\n"
        f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
    )
```

Default values in function signature:
```python
def get_peft_model(
    model,
    r = 16,
    # ...
    lora_dropout = 0.0,  # Default: no dropout
    bias = "none",       # Default: no bias
    # ...
):
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_get_peft_model]]
* [[uses_heuristic::Principle:unslothai_unsloth_LoRA_Configuration]]
