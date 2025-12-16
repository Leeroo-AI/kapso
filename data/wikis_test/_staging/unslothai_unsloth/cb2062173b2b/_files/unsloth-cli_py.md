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

**Purpose:** Command-line interface for end-to-end model fine-tuning with Unsloth. Provides a batteries-included script for training, saving, and pushing models to HuggingFace Hub without writing Python code.

**Mechanism:**
- **Argument groups**: Organized CLI into logical sections:
  - **Model Options** (`--model_name`, `--max_seq_length`, `--dtype`, `--load_in_4bit`, `--dataset`): Model selection and loading
  - **LoRA Options** (`--r`, `--lora_alpha`, `--lora_dropout`, `--bias`, `--use_rslora`): PEFT configuration with sensible defaults
  - **Training Options** (`--batch_size`, `--gradient_accumulation_steps`, `--learning_rate`, `--warmup_steps`, `--max_steps`, `--packing`): Training hyperparameters
  - **Report Options** (`--report_to`, `--logging_steps`): Integration with tensorboard/wandb/mlflow/etc.
  - **Save Options** (`--save_model`, `--save_method`, `--save_gguf`, `--quantization`): Export formats (merged_16bit, merged_4bit, lora, GGUF)
  - **Push Options** (`--push_model`, `--hub_path`, `--hub_token`): HuggingFace Hub upload
- **Workflow**:
  1. Loads model and tokenizer using `FastLanguageModel.from_pretrained()`
  2. Applies PEFT (LoRA) with `FastLanguageModel.get_peft_model()`
  3. Formats dataset with Alpaca prompt template (instruction/input/output)
  4. Configures SFTConfig with fp16/bf16 auto-detection
  5. Creates SFTTrainer and runs training
  6. Saves model in specified format(s) (can save multiple GGUF quantizations in one run)
  7. Optionally pushes to HuggingFace Hub
- **Convenience features**:
  - Auto-detects ModelScope vs HuggingFace datasets via `UNSLOTH_USE_MODELSCOPE` env var
  - Handles distributed training with `prepare_device_map()`
  - Supports saving multiple GGUF quantizations in single run (`--quantization q4_k_m q8_0`)
  - Provides extensive help text with common values for each parameter
- **Default values**: Reasonable defaults for quick start (r=16, lr=2e-4, batch_size=2, max_steps=400)

**Significance:** This CLI democratizes fine-tuning by removing the need to write boilerplate code. Users can go from zero to fine-tuned model with a single command. The argument organization makes it discoverable (--help shows grouped options). The multiple save formats in one run is particularly valuable (saves time vs re-training for different formats). This script embodies "convention over configuration" - works out-of-the-box for most use cases, but allows full customization. It's likely the entry point for many Unsloth users who want to fine-tune models without learning the API first.
