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

**Purpose:** Command-line interface entry point for fine-tuning language models with Unsloth, providing a configurable starter script with comprehensive options.

**Mechanism:** Uses argparse to define six option groups (Model, LoRA, Training, Report, Save, Push) with ~40 configurable parameters. The run() function orchestrates the complete training workflow: loads model/tokenizer via FastLanguageModel.from_pretrained() with device mapping, applies PEFT/LoRA configuration via get_peft_model(), loads datasets from Hugging Face/ModelScope/local files (including raw text with RawTextDataLoader), formats data using Alpaca prompt template, configures SFTTrainer with SFTConfig, executes training, and optionally saves/pushes models in merged or GGUF formats with specified quantization methods. Supports distributed training, gradient checkpointing, multiple optimizers, various quantization methods, and integration with tracking platforms (tensorboard, wandb, etc.).

**Significance:** Primary user-facing tool that democratizes LLM fine-tuning by packaging Unsloth's optimizations into an accessible CLI with sensible defaults, enabling users to fine-tune models via command-line arguments without writing Python code, while remaining customizable for advanced users who can modify the script for specific use cases.
