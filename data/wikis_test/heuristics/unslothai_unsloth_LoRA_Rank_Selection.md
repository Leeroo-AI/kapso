# Heuristic: LoRA Rank Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|LoRA: Low-Rank Adaptation|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]], [[domain::Parameter_Efficient_Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

## Overview

Guidelines for selecting LoRA rank (r) parameter to balance training quality, speed, and memory usage.

### Description

LoRA rank determines the dimensionality of the low-rank decomposition matrices (A and B). Higher ranks capture more complex adaptations but require more memory and compute. Unsloth provides optimized defaults and allows ranks from 1 to 256+.

### Usage

Apply this heuristic when:
- Configuring `get_peft_model()` for fine-tuning
- Balancing memory constraints against model quality
- Choosing between instruction tuning vs. complex task adaptation

## The Insight (Rule of Thumb)

* **Default Rank:** `r=16` is the Unsloth default, suitable for most instruction-following tasks
* **Simple Tasks:** `r=8` or lower for basic chat/instruction fine-tuning with limited data
* **Complex Tasks:** `r=32-64` for domain adaptation, code generation, or reasoning tasks
* **Very Large Models (70B+):** Consider `r=64-128` as larger models benefit from higher ranks
* **Trade-off:** Higher rank = more parameters = slower training + more VRAM

### Recommended Configurations

| Use Case | Recommended Rank | Target Modules |
|----------|------------------|----------------|
| Basic Instruction Tuning | r=16 | q_proj, k_proj, v_proj, o_proj |
| Domain Adaptation | r=32-64 | All attention + gate_proj, up_proj, down_proj |
| Reasoning/Math | r=64+ | All modules including embed_tokens |
| Memory Constrained | r=8 | q_proj, v_proj only |

### Alpha Scaling

* **Rule:** Set `lora_alpha = r` (or `lora_alpha = 2*r` for stronger adaptation)
* The effective learning rate scales as `alpha/r`, so:
  - `alpha=r`: 1x strength (default)
  - `alpha=2r`: 2x strength (use with lower LR)

## Reasoning

From the LoRA paper (Hu et al., 2021):
- Rank r=4-8 is often sufficient for many NLP tasks
- Higher ranks don't always improve quality but always increase cost
- The attention layers (Q, K, V, O projections) are the most impactful

Unsloth's experience:
- Modern LLMs often need r=16+ for best results
- Including MLP layers (gate, up, down) significantly improves quality
- `lora_dropout=0` is recommended by Unsloth as dropout provides minimal benefit for QLoRA

### Code Evidence

From `unsloth/models/loader.py:143-145`:
```python
max_lora_rank = 64,  # Default max rank for vLLM
```

Common pattern in examples:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,  # Unsloth optimized
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)
```

## Related Pages

### Used By

* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[uses_heuristic::Implementation:unslothai_unsloth_FastVisionModel]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Principle:unslothai_unsloth_Low_Rank_Adaptation]]
