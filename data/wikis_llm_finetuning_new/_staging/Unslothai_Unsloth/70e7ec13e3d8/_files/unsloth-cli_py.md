# File: `unsloth-cli.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 473 |
| Functions | `run` |
| Imports | argparse, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** A command-line interface script for fine-tuning language models using Unsloth's FastLanguageModel with LoRA/PEFT, providing a complete workflow from model loading through training to saving and pushing to Hugging Face Hub.

**Mechanism:** The script uses argparse to define extensive CLI options organized into groups: Model Options (model name, sequence length, dtype, 4-bit quantization), LoRA Options (rank, alpha, dropout, gradient checkpointing, RSLoRA), Training Options (batch sizes, learning rate, optimizer, scheduler), Report Options (tensorboard/wandb integration), and Save/Push Options (GGUF export, quantization methods, Hub upload). The `run()` function lazily imports Unsloth components, loads the model with `FastLanguageModel.from_pretrained()`, applies PEFT configuration with `get_peft_model()`, loads datasets (supporting HuggingFace datasets, ModelScope, and raw text files via `RawTextDataLoader`), formats data using an Alpaca-style prompt template, configures `SFTTrainer` from TRL with automatic bf16/fp16 detection, trains the model, and optionally saves to GGUF format or pushes to Hugging Face Hub with various quantization methods.

**Significance:** This is a major user-facing entry point for Unsloth, providing a no-code solution for fine-tuning LLMs. It serves as both a practical tool for quick experimentation and a reference implementation demonstrating recommended Unsloth usage patterns. The script supports distributed training, multiple dataset sources, flexible LoRA configurations, and various model export formats (merged, LoRA-only, GGUF with multiple quantization levels).
