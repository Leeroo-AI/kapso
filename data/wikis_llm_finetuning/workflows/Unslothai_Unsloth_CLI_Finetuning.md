# Workflow: CLI_Finetuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::CLI]]
|-
! Last Updated
| [[last_updated::2026-01-09 17:00 GMT]]
|}

== Overview ==
Command-line interface workflow for fine-tuning language models using Unsloth's optimized pipeline with argparse-based configuration.

=== Description ===
This workflow provides a complete CLI-driven approach to fine-tuning large language models. It wraps the QLoRA fine-tuning pipeline into a standalone Python script (`unsloth-cli.py`) that accepts command-line arguments for all configuration options. The CLI integrates RawTextDataLoader for processing local text files and supports both HuggingFace datasets and ModelScope datasets.

Key features:
* Argparse-based configuration for all training parameters
* Smart dataset loading with automatic format detection
* Support for raw text files (.txt, .md, .json, .jsonl)
* Optional GGUF export and Hub push
* Distributed training support

=== Usage ===
Execute this workflow when you want to:
* Fine-tune models from the command line without writing Python code
* Process local text files directly without manual preprocessing
* Run training jobs in automated pipelines or CI/CD systems
* Quickly experiment with different configurations via CLI flags

== Execution Steps ==

=== Step 1: Model Loading ===
[[step::Principle:Unslothai_Unsloth_Model_Loading]]

Initialize the language model from HuggingFace Hub or a local path with 4-bit quantization. The CLI accepts `--model_name`, `--max_seq_length`, `--dtype`, and `--load_in_4bit` flags to configure loading.

'''Key CLI arguments:'''
* `--model_name "unsloth/llama-3-8b"` - Model identifier
* `--max_seq_length 2048` - Maximum sequence length
* `--load_in_4bit` - Enable 4-bit quantization

=== Step 2: LoRA Configuration ===
[[step::Principle:Unslothai_Unsloth_LoRA_Configuration]]

Configure LoRA adapters with customizable rank and target modules. All PEFT parameters are exposed as CLI arguments.

'''Key CLI arguments:'''
* `--r 16` - LoRA rank
* `--lora_alpha 16` - LoRA alpha scaling
* `--lora_dropout 0.0` - Dropout rate
* `--use_rslora` - Enable rank-stabilized LoRA
* `--use_gradient_checkpointing "unsloth"` - Memory optimization

=== Step 3: Dataset Loading ===
[[step::Principle:Unslothai_Unsloth_CLI_Data_Loading]]

Smart dataset loading that automatically detects data format:
* Local text files (.txt, .md, .json, .jsonl) use RawTextDataLoader
* HuggingFace datasets use standard `load_dataset`
* ModelScope datasets supported via environment variable

'''Key CLI arguments:'''
* `--dataset "yahma/alpaca-cleaned"` - Dataset path or HF identifier
* `--raw_text_file path/to/file.txt` - Direct path to raw text
* `--chunk_size` - Chunk size for raw text processing
* `--stride` - Stride for overlapping chunks

=== Step 4: Training Configuration ===
[[step::Principle:Unslothai_Unsloth_Training_Configuration]]

Configure SFTConfig with all standard training hyperparameters exposed via CLI.

'''Key CLI arguments:'''
* `--per_device_train_batch_size 4` - Batch size per GPU
* `--gradient_accumulation_steps 8` - Gradient accumulation
* `--learning_rate 2e-6` - Learning rate
* `--max_steps 400` - Maximum training steps
* `--warmup_steps 5` - Warmup steps
* `--optim "adamw_8bit"` - Optimizer
* `--packing` - Enable sample packing

=== Step 5: Training Execution ===
[[step::Principle:Unslothai_Unsloth_Supervised_Finetuning]]

Execute SFTTrainer with the configured parameters. Supports distributed training via device map auto-detection.

=== Step 6: Model Export ===
[[step::Principle:Unslothai_Unsloth_Model_Saving]]

Save the trained model in various formats with optional Hub push.

'''Key CLI arguments:'''
* `--save_model` - Enable saving
* `--save_path "model"` - Save directory
* `--save_gguf` - Export to GGUF format
* `--quantization "q4_k_m"` - GGUF quantization method
* `--push_model` - Push to HuggingFace Hub
* `--hub_path "username/model"` - Hub repository
* `--hub_token` - HuggingFace token

== Execution Diagram ==
{{#mermaid:graph TD
    A[Parse CLI Arguments] --> B[Load Model with 4-bit Quantization]
    B --> C[Configure LoRA Adapters]
    C --> D[Smart Dataset Loading]
    D --> E[Configure SFTTrainer]
    E --> F[Execute Training]
    F --> G{Save Options}
    G -->|GGUF| H[Export GGUF]
    G -->|Merged| I[Save Merged Weights]
    H --> J{Push to Hub?}
    I --> J
    J -->|Yes| K[Upload to HuggingFace]
    J -->|No| L[Done]
    K --> L
}}

== Related Pages ==
* [[step::Principle:Unslothai_Unsloth_Model_Loading]]
* [[step::Principle:Unslothai_Unsloth_LoRA_Configuration]]
* [[step::Principle:Unslothai_Unsloth_Training_Configuration]]
* [[step::Principle:Unslothai_Unsloth_Supervised_Finetuning]]
* [[step::Principle:Unslothai_Unsloth_Model_Saving]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Gradient_Checkpointing_Tip]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Sample_Packing_Tip]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_LoRA_Rank_Selection_Tip]]
