# File: `src/peft/optimizers/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 19 |
| Imports | lorafa, loraplus |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization for specialized optimizers designed for LoRA training.

**Mechanism:** Exports create_lorafa_optimizer and create_loraplus_optimizer factory functions from their respective modules. These optimizers implement research papers that improve LoRA training efficiency.

**Significance:** Provides easy access to advanced LoRA optimizers that can improve training speed and final model quality. While not required for basic PEFT usage, these optimizers represent state-of-the-art techniques for parameter-efficient fine-tuning and are important for users seeking optimal results.
