# Heuristic: huggingface_transformers_Batch_Size_Optimization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Discussion|OOM handling patterns]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Automatic batch size finding mechanism to prevent OOM errors by dynamically reducing batch size until training succeeds.

=== Description ===

The Trainer implements an automatic batch size finder that catches CUDA out-of-memory errors and progressively reduces the batch size until a working configuration is found. This eliminates the trial-and-error process of manually finding the maximum batch size for a given model and GPU combination.

=== Usage ===

Use this heuristic when:
- Starting training on a **new model/GPU combination**
- Training on **variable-length sequences** where memory varies
- Running on **shared GPU resources** where available memory is uncertain
- Wanting to **maximize throughput** without manual tuning

Enable with `TrainingArguments(auto_find_batch_size=True)`.

== The Insight (Rule of Thumb) ==

* **Action**: Set `auto_find_batch_size=True` in `TrainingArguments`
* **Value**: Batch size starts at configured value and halves on each OOM
* **Trade-off**:
  - Pros: Eliminates manual batch size tuning, prevents training failures
  - Cons: Initial training attempts may fail (expected), slight startup overhead

* **Total Batch Size Formula**:
  ```
  total_batch_size = micro_batch * grad_accum * dp_world_size
  dp_world_size = world_size / (tp_size * cp_size * sp_size)
  ```

== Reasoning ==

**Why automatic batch size finding:**

1. **Memory Variability**: GPU memory availability varies based on other processes, CUDA context size, and model initialization overhead.

2. **Model-Specific Requirements**: Different models have vastly different memory footprints for the same batch size due to architecture differences (attention patterns, hidden sizes, sequence lengths).

3. **Gradient Accumulation Compensation**: When batch size is reduced, gradient accumulation can be increased to maintain the same effective batch size, preserving training dynamics.

4. **Distributed Training Complexity**: Total batch size depends on multiple parallelism dimensions (TP, CP, SP), making manual calculation error-prone.

== Code Evidence ==

Batch size finder wrapper from `trainer.py:2152-2154`:

<syntaxhighlight lang="python">
inner_training_loop = find_executable_batch_size(
    self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
)
</syntaxhighlight>

Total batch size calculation from `trainer.py:2205-2218`:

<syntaxhighlight lang="python">
def get_total_train_batch_size(self, args) -> int:
    """
    Calculates total batch size (micro_batch * grad_accum * dp_world_size).

    Accounts for all parallelism dimensions: TP, CP, and SP.

    Formula: dp_world_size = world_size // (tp_size * cp_size * sp_size)

    Where:
    - TP (Tensor Parallelism): Model layers split across GPUs
    - CP (Context Parallelism): Sequences split using Ring Attention (FSDP2)
    - SP (Sequence Parallelism): Sequences split using ALST/Ulysses (DeepSpeed)
    """
</syntaxhighlight>

Loss normalization for gradient accumulation from `trainer.py:3824-3827`:

<syntaxhighlight lang="python">
if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
    # Normalize loss by gradient accumulation steps
    loss = loss / self.current_gradient_accumulation_steps
</syntaxhighlight>

== Best Practices ==

1. **Start with a reasonable estimate**: Set initial batch size based on model size
   - 7B model: Start with batch_size=4 on 24GB GPU
   - 13B model: Start with batch_size=2 on 24GB GPU
   - 70B model: Requires model parallelism

2. **Use gradient accumulation**: Maintain effective batch size when micro-batch is reduced
   ```python
   TrainingArguments(
       per_device_train_batch_size=4,
       gradient_accumulation_steps=8,  # Effective batch = 32
       auto_find_batch_size=True
   )
   ```

3. **Consider gradient checkpointing**: Reduces memory at cost of compute
   ```python
   TrainingArguments(
       gradient_checkpointing=True,
       per_device_train_batch_size=8  # Can use larger batch
   )
   ```

== Related Pages ==

* [[uses_heuristic::Implementation:huggingface_transformers_Trainer_train]]
* [[uses_heuristic::Implementation:huggingface_transformers_TrainingArguments]]
* [[uses_heuristic::Workflow:huggingface_transformers_Training]]
