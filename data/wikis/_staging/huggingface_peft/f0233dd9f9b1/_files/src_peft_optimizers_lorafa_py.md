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

**Purpose:** Implements LoRA-FA (Frozen-A) optimizer that freezes LoRA A matrices and optimizes only B matrices with projected gradients, improving training efficiency.

**Mechanism:** LoraFAOptimizer extends AdamW but modifies gradient updates for LoRA B matrices: projects gradients through (A^T A)^-1 to minimize ||gradient_projected - gradient_full||. Freezes lora_A parameters (requires_grad=False). Falls back to standard AdamW for non-LoRA parameters. create_lorafa_optimizer() is a helper for easy instantiation with PeftModel.

**Significance:** Training optimization that reduces memory and computation by only updating LoRA B matrices while maintaining performance. Based on paper showing that freezing A and using projected gradients for B achieves similar results to full LoRA training but with ~2x speedup and less memory usage.
