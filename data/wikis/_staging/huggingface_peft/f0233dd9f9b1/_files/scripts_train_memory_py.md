# File: `scripts/train_memory.py`

**Category:** benchmarking script

| Property | Value |
|----------|-------|
| Lines | 277 |
| Functions | `init_accelerator`, `get_data`, `train` |
| Imports | argparse, collections.Counter, contextlib, datasets.load_dataset, functools.partial, gc, os, peft, sys, tempfile, time, torch, transformers, warnings |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Benchmarking script that measures memory consumption, training time, model size, and optionally analyzes activation memory during PEFT training on a small text dataset. Used for performance profiling and comparison.

**Mechanism:**
- `init_accelerator()`: Initializes device, resets peak memory stats, sets random seeds for reproducibility

- `get_data()`: Loads training data
  - Uses "ybelkada/english_quotes_copy" dataset
  - Tokenizes with manual truncation to max_seq_length
  - Removes non-tensor columns (quote, author, tags)

- `train()`: Main training loop with comprehensive metrics:
  - Model setup: Supports multiple dtypes (float32, float16, bfloat16, int8, int4)
  - Quantization: Uses BitsAndBytesConfig for int4/int8 with kbit training prep
  - PEFT application: Loads LoraConfig from path if rank > 0, else full fine-tuning
  - Memory tracking: Logs memory allocation at each step
  - Optional tensor monitoring: Captures all saved tensors during backprop using `saved_tensors_hooks`
  - Training: Simple AdamW optimizer, manual forward/backward passes
  - Metrics collected:
    - Average/max GPU memory consumption
    - Total training time
    - Saved model file size
    - If monitor_tensors enabled: dtype/shape distributions, activation vs parameter sizes

Command-line arguments:
- `model_id`: HuggingFace model to train
- `--rank`: LoRA rank (0 for full fine-tuning, default: 8)
- `--dtype`: Data type (float32/float16/bfloat16/int8/int4, default: float32)
- `--monitor_tensors`: Enable tensor analysis for single step
- `--max_seq_length`: Max sequence length (default: 128)
- `--batch_size`: Batch size (default: 1)
- `--max_steps`: Training steps (default: 50)
- `--path_config`: Path to LoRA config JSON

**Significance:** Critical benchmarking tool for PEFT development and method comparison. Provides standardized metrics for evaluating different PEFT methods' memory footprint, speed, and storage requirements. The tensor monitoring feature enables deep analysis of activation memory, crucial for optimizing memory-efficient training. Used to generate data for the method_comparison benchmarks and documentation.
