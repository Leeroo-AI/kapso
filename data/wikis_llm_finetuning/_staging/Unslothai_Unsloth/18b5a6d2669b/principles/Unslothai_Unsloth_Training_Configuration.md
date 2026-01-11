# Principle: Training_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|AdamW Optimizer|https://arxiv.org/abs/1711.05101]]
* [[source::Blog|Learning Rate Schedules|https://huggingface.co/docs/transformers/main_classes/optimizer_schedules]]
* [[source::Paper|8-bit Optimizers|https://arxiv.org/abs/2110.02861]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Selection and configuration of training hyperparameters including learning rate, batch size, optimizer, and precision settings for effective LoRA fine-tuning.

=== Description ===

Training Configuration for QLoRA fine-tuning involves selecting hyperparameters that balance training stability, convergence speed, and memory efficiency. Key decisions include:

* **Learning rate**: Typically 1e-4 to 3e-4 for LoRA (higher than full fine-tuning)
* **Batch size × Gradient accumulation**: Effective batch size for stable gradients
* **Optimizer**: 8-bit AdamW for memory efficiency with minimal accuracy loss
* **Precision**: bf16 preferred (if supported), fp16 as fallback
* **Warmup**: Brief warmup (1-5% of steps) for stable initial training

These choices are interdependent—higher learning rates may require smaller batches or more warmup.

=== Usage ===

Configure training parameters after model and LoRA setup, before trainer initialization. Consider:
* GPU memory constraints (batch size, precision)
* Dataset size (epochs vs max_steps)
* Task complexity (learning rate, total steps)
* Checkpoint strategy (save frequency vs disk space)

== Theoretical Basis ==

=== Learning Rate Selection ===

LoRA typically uses higher learning rates than full fine-tuning because:
1. Fewer parameters → larger per-parameter gradients needed for same model change
2. Low-rank constraint limits expressivity → need stronger signal
3. Base weights frozen → only adapters must capture task knowledge

Recommended range: 1e-4 to 5e-4 (vs 1e-5 to 5e-5 for full fine-tuning)

=== Effective Batch Size ===

<math>
\text{Effective Batch} = \text{per\_device\_batch} \times \text{num\_gpus} \times \text{gradient\_accumulation}
</math>

Larger effective batches:
* More stable gradients
* Can support higher learning rates
* Better hardware utilization

Typical effective batch sizes: 8-64 for fine-tuning

=== 8-bit Optimizer States ===

AdamW maintains two momentum buffers per parameter:
* First moment (m): Running mean of gradients
* Second moment (v): Running mean of squared gradients

8-bit quantization reduces optimizer state memory by 75%:

<math>
\text{Memory}_{8bit} = \frac{1}{4} \times \text{Memory}_{fp32}
</math>

With dynamic scaling and block-wise quantization, accuracy loss is negligible.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Training configuration selection (abstract)
def configure_training(model, dataset_size, gpu_memory):
    # Start with defaults
    config = {
        "learning_rate": 2e-4,
        "batch_size": 2,
        "grad_accum": 4,
        "warmup_ratio": 0.03,
        "optimizer": "adamw_8bit",
    }

    # Adjust for GPU memory
    if gpu_memory < 16:
        config["batch_size"] = 1
        config["grad_accum"] = 8

    # Adjust for dataset size
    steps_per_epoch = dataset_size // (config["batch_size"] * config["grad_accum"])
    config["warmup_steps"] = int(steps_per_epoch * config["warmup_ratio"])

    return config
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_UnslothTrainingArguments]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Embedding_Learning_Rate_Tip]]
