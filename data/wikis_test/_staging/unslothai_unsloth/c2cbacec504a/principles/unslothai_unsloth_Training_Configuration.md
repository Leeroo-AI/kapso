# Principle: unslothai_unsloth_Training_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Trainer|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Paper|AdamW Optimizer|https://arxiv.org/abs/1711.05101]]
* [[source::Paper|Learning Rate Schedules|https://arxiv.org/abs/1608.03983]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for configuring training hyperparameters and optimization settings for supervised fine-tuning of language models.

=== Description ===

Training Configuration encompasses the selection of:

1. **Batch size and gradient accumulation**: Balancing memory vs training speed
2. **Learning rate and scheduler**: Controlling optimization dynamics
3. **Optimizer choice**: AdamW variants optimized for transformers
4. **Regularization**: Weight decay, dropout for generalization
5. **Precision settings**: Mixed precision for memory efficiency

Proper configuration is critical for stable training and good final model performance.

=== Usage ===

Use this principle when:
- Setting up any fine-tuning run
- Debugging training instability (loss spikes, NaN values)
- Optimizing training speed vs quality tradeoffs
- Adapting recipes for different hardware (GPU VRAM, batch size constraints)

== Theoretical Basis ==

=== Effective Batch Size ===

The effective batch size determines training dynamics:

<math>
\text{effective\_batch\_size} = \text{per\_device\_batch\_size} \times \text{gradient\_accumulation} \times \text{num\_gpus}
</math>

'''Pseudo-code for batch size selection:'''
<syntaxhighlight lang="python">
# Determine maximum per-device batch size
def find_max_batch_size(model, seq_length, gpu_vram):
    # Approximate memory per sample (very rough)
    # Model + gradients + optimizer states + activations
    mem_per_sample = estimate_memory(model, seq_length)

    # Leave headroom for fragmentation and peaks
    available_vram = gpu_vram * 0.8

    max_batch_size = int(available_vram / mem_per_sample)
    return max(1, max_batch_size)

# Common effective batch sizes for LLM fine-tuning
# Small models (1-3B): 32-64
# Medium models (7-13B): 16-32
# Large models (30B+): 8-16
</syntaxhighlight>

=== Learning Rate Selection ===

Learning rate scales with batch size (linear scaling rule):

<math>
\text{lr} = \text{base\_lr} \times \frac{\text{effective\_batch\_size}}{\text{reference\_batch\_size}}
</math>

For LoRA fine-tuning, typical base learning rates:

{| class="wikitable"
|-
! Training Type !! Base LR !! Reference Batch Size
|-
| QLoRA (4-bit) || 2e-4 to 5e-4 || 32
|-
| LoRA (16-bit) || 1e-4 to 2e-4 || 32
|-
| Full fine-tune || 1e-5 to 5e-5 || 32
|}

=== Warmup and Scheduling ===

Learning rate schedules prevent early training instability:

<syntaxhighlight lang="python">
# Linear warmup + linear/cosine decay
def learning_rate_schedule(step, total_steps, warmup_steps, max_lr):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step / warmup_steps)
    else:
        # Linear decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * (1 - progress)

# Recommended warmup: 3-10% of total steps
warmup_ratio = 0.03  # 3% warmup
</syntaxhighlight>

=== AdamW Optimizer ===

AdamW with decoupled weight decay:

<math>
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\theta_t &= \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
\end{align}
</math>

8-bit AdamW reduces memory:
<syntaxhighlight lang="python">
# Standard AdamW: 8 bytes per parameter (m + v in fp32)
# 8-bit AdamW: 2 bytes per parameter (m + v in int8)
# Memory savings: ~75% on optimizer states

# 7B model optimizer memory:
# fp32 AdamW: 7B * 8 = 56GB
# 8-bit AdamW: 7B * 2 = 14GB
</syntaxhighlight>

=== Gradient Accumulation ===

Accumulation allows large effective batches with limited VRAM:

<syntaxhighlight lang="python">
# Without accumulation
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# With accumulation (4 steps)
accumulated_loss = 0
for i, batch in enumerate(dataloader):
    loss = model(batch).loss / 4  # Scale loss
    loss.backward()
    accumulated_loss += loss.item()

    if (i + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
</syntaxhighlight>

== Practical Guide ==

=== Recommended Starting Configuration ===

<syntaxhighlight lang="python">
# Balanced configuration for most QLoRA runs
config = {
    # Batch settings
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,  # effective = 8

    # Optimizer
    "optim": "adamw_8bit",  # Memory efficient
    "learning_rate": 2e-4,
    "weight_decay": 0.01,

    # Schedule
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.03,

    # Training length
    "num_train_epochs": 1,  # or max_steps

    # Precision (auto-detect)
    "bf16": True,  # Use bf16 if available
}
</syntaxhighlight>

=== Hardware-Specific Adjustments ===

{| class="wikitable"
|-
! GPU VRAM !! Recommended Batch Size !! Gradient Accum !! Notes
|-
| 8GB (RTX 3060) || 1 || 8-16 || Use `load_in_4bit=True`
|-
| 16GB (RTX 4080) || 2-4 || 4-8 || Comfortable for 7B models
|-
| 24GB (RTX 4090) || 4-8 || 2-4 || Can handle 13B with 4-bit
|-
| 48GB (A6000) || 8-16 || 1-2 || 30B+ models possible
|}

=== Debugging Training Issues ===

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss spikes | LR too high | Reduce LR by 2-5x |
| NaN loss | Numerical instability | Use bf16, add gradient clipping |
| No improvement | LR too low | Increase LR, check data |
| OOM error | Batch too large | Reduce batch, increase accum |

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_SFTTrainer_usage]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
