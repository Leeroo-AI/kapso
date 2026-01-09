# Implementation: CLI

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::CLI]], [[domain::Training]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Command-line interface for fine-tuning language models using Unsloth's optimized training pipeline.

=== Description ===
The `unsloth-cli.py` script provides a complete CLI entry point for the Unsloth fine-tuning workflow. It wraps `FastLanguageModel.from_pretrained()`, `get_peft_model()`, and `SFTTrainer` into a single command with configurable arguments for model loading, LoRA configuration, training parameters, and model saving/export.

=== Usage ===
Use this CLI when you want to fine-tune a model without writing Python code, or for scripted automation of training jobs. Supports HuggingFace datasets, local raw text files, and ModelScope datasets.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py unsloth-cli.py]
* '''Lines:''' 1-473

=== Signature ===
<syntaxhighlight lang="python">
def run(args):
    """
    Main entry point for CLI fine-tuning.

    Args:
        args: Parsed argparse namespace with:
            - model_name: HuggingFace model ID
            - max_seq_length: Maximum sequence length
            - dtype: Data type (auto-detected if None)
            - load_in_4bit: Enable 4-bit quantization
            - r: LoRA rank
            - lora_alpha: LoRA alpha parameter
            - lora_dropout: Dropout rate
            - per_device_train_batch_size: Batch size
            - learning_rate: Learning rate
            - max_steps: Maximum training steps
            - save_model: Whether to save model
            - save_gguf: Whether to export as GGUF
            - quantization: GGUF quantization method(s)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# Run directly from command line
python unsloth-cli.py --model_name "unsloth/llama-3-8b" --load_in_4bit
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| --model_name || str || No (default: unsloth/llama-3-8b) || HuggingFace model ID
|-
| --max_seq_length || int || No (default: 2048) || Maximum sequence length
|-
| --load_in_4bit || flag || No || Enable 4-bit QLoRA quantization
|-
| --dataset || str || No (default: yahma/alpaca-cleaned) || Dataset name or path
|-
| --r || int || No (default: 16) || LoRA rank
|-
| --lora_alpha || int || No (default: 16) || LoRA alpha scaling
|-
| --max_steps || int || No (default: 400) || Maximum training steps
|-
| --learning_rate || float || No (default: 2e-4) || Learning rate
|-
| --save_model || flag || No || Save model after training
|-
| --save_gguf || flag || No || Export to GGUF format
|-
| --quantization || str/list || No (default: q8_0) || GGUF quantization method(s)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Trained model || Directory || Saved to --save_path (default: model/)
|-
| GGUF files || Files || If --save_gguf enabled, GGUF model files
|-
| Logs || Directory || Training logs at --output_dir (default: outputs/)
|}

== Usage Examples ==

=== Basic Fine-Tuning ===
<syntaxhighlight lang="bash">
# Basic QLoRA fine-tuning with defaults
python unsloth-cli.py \
    --model_name "unsloth/llama-3-8b" \
    --load_in_4bit \
    --max_steps 100 \
    --save_model \
    --save_path "./my_model"
</syntaxhighlight>

=== Full Configuration Example ===
<syntaxhighlight lang="bash">
# Full configuration with GGUF export
python unsloth-cli.py \
    --model_name "unsloth/llama-3-8b" \
    --max_seq_length 8192 \
    --load_in_4bit \
    --r 64 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 5 \
    --max_steps 400 \
    --learning_rate 2e-6 \
    --optim "adamw_8bit" \
    --weight_decay 0.005 \
    --lr_scheduler_type "linear" \
    --output_dir "outputs" \
    --report_to "tensorboard" \
    --save_model \
    --save_gguf \
    --save_path "model" \
    --quantization "q4_k_m" "q8_0"
</syntaxhighlight>

=== Raw Text Training ===
<syntaxhighlight lang="bash">
# Train on raw text file
python unsloth-cli.py \
    --model_name "unsloth/llama-3-8b" \
    --load_in_4bit \
    --raw_text_file "./my_corpus.txt" \
    --chunk_size 2048 \
    --stride 512 \
    --max_steps 200 \
    --save_model
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
