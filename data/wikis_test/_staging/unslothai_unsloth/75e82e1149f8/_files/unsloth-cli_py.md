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

**Purpose:** Command-line interface for fine-tuning models with Unsloth without writing Python code.

**Mechanism:**
- Parses command-line arguments for all training configuration options
- Model options: `--model_name`, `--max_seq_length`, `--dtype`, `--load_in_4bit`
- PEFT/LoRA options: `--r`, `--lora_alpha`, `--lora_dropout`, `--bias`, `--use_rslora`
- Training options: `--per_device_train_batch_size`, `--gradient_accumulation_steps`, `--learning_rate`, `--max_steps`, `--optim`
- Save options: `--save_model`, `--save_path`, `--save_gguf`, `--quantization`
- Push options: `--push_model`, `--hub_path`, `--hub_token`
- Uses Alpaca prompt format by default for instruction tuning
- Supports ModelScope datasets via `UNSLOTH_USE_MODELSCOPE` environment variable
- Handles distributed training via `prepare_device_map()`
- Can save in GGUF format with multiple quantization methods or merged 16-bit

**Significance:** Lowers barrier to entry for users unfamiliar with Python. Enables quick experimentation via command line without writing training scripts.
