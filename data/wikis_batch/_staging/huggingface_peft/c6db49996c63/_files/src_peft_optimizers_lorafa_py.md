# File: `src/peft/optimizers/lorafa.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 256 |
| Classes | `LoraFAOptimizer` |
| Functions | `create_lorafa_optimizer` |
| Imports | __future__, accelerate, collections, math, peft_model, torch, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the LoRA-FA (Frozen-A) optimizer that improves LoRA training by optimizing only the B matrices.

**Mechanism:** Custom optimizer based on AdamW. Freezes lora_A parameters and applies gradient projection to lora_B updates using the formula g^B = (r/α)² (A^T A)^(-1) g. This projection minimizes ||g_LoRA-FA - g||_F² where g is the full gradient. Falls back to standard AdamW for non-LoRA parameters. create_lorafa_optimizer is a helper that sets up the optimizer with proper scaling factors.

**Significance:** Implements a research advancement (LoRA-FA paper) that can reduce memory usage and improve convergence for LoRA training. Particularly useful for large models where the gradient computation overhead matters. Demonstrates PEFT's commitment to incorporating cutting-edge research into practical tools.
