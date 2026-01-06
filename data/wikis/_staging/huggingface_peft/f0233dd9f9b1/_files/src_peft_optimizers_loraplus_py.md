# File: `src/peft/optimizers/loraplus.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 121 |
| Functions | `create_loraplus_optimizer` |
| Imports | __future__, operator, peft_model, torch, transformers, tuners |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements LoRA+ training strategy that uses different learning rates for LoRA A and B matrices, with B having a higher learning rate for faster convergence.

**Mechanism:** create_loraplus_optimizer() creates parameter groups with different learning rates: groupA (lora_A, lr), groupB (lora_B and biases, lr * loraplus_lr_ratio), groupB_no_decay (no weight decay), and embedding (custom lr). Uses standard optimizer (AdamW, etc.) but with custom parameter grouping. Handles 8-bit optimizers specially for embeddings.

**Significance:** Training improvement from LoRA+ paper showing that using higher learning rate for B matrix (typically 8-16x) enables faster convergence and better performance. Simple to implement as it only requires parameter grouping, works with any optimizer, and is especially effective for difficult tasks requiring feature updates.
