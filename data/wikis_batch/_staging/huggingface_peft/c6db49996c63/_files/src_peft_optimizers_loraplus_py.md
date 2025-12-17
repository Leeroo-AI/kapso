# File: `src/peft/optimizers/loraplus.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 121 |
| Functions | `create_loraplus_optimizer` |
| Imports | __future__, operator, peft_model, torch, transformers, tuners |

## Understanding

**Status:** ✅ Explored

**Purpose:** Factory function for creating optimizers with LoRA+ learning rate scheduling.

**Mechanism:** Organizes model parameters into four groups with different learning rates: groupA (lora_A and base params) at lr, groupB (lora_B) at lr × ratio, groupB_no_decay (no weight decay), and embedding (special lr). The lr ratio ηB/ηA should be ≥1 and larger for difficult tasks. Supports 8-bit optimizers with special handling for embeddings.

**Significance:** Implements LoRA+ (a research paper showing differential learning rates improve LoRA training). The key insight is that B matrices should learn faster than A matrices. This simple technique can significantly improve final model quality, especially on challenging tasks. Widely adopted in the LoRA community.
