# File: `unsloth-cli.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 441 |
| Functions | `run` |
| Imports | argparse, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Documented

**Purpose:** Comprehensive CLI for fine-tuning language models with LoRA/QLoRA

**Mechanism:** Provides command-line interface with arguments for model selection, quantization levels, PEFT configuration, training parameters, dataset preparation, and saving options; integrates FastLanguageModel, SFTTrainer, and various saving methods

**Significance:** Production-ready fine-tuning script that serves as the main entry point for users to fine-tune models with Unsloth, supporting full workflows from loading to training to deployment
