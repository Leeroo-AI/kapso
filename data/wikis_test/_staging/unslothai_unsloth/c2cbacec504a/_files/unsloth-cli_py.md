# File: `unsloth-cli.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 441 |
| Functions | `run` |
| Imports | argparse, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Command-line interface providing end-to-end fine-tuning workflow with extensive configuration options for model loading, PEFT/LoRA parameters, training, and model saving.

**Mechanism:** Implements argument parser with organized argument groups for model options, LoRA configuration, training parameters, reporting, and saving/pushing functionality. Main run() function orchestrates loading model/tokenizer, configuring LoRA, loading dataset, creating SFTTrainer, training, and saving model in requested format (merged or GGUF with optional quantization).

**Significance:** Provides accessible entry point for users to fine-tune models without writing complex code. Demonstrates best practices for configuring Unsloth. Supports common workflows like fine-tuning on Alpaca dataset and pushing to HuggingFace Hub.
