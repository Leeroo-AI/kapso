# Heuristic: huggingface_transformers_Liger_Kernel_Optimization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Repo|Liger Kernel|https://github.com/linkedin/Liger-Kernel]]
|-
! Domains
| [[domain::Optimization]], [[domain::Training]], [[domain::GPU_Kernels]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Enable Liger kernels with `use_liger_kernel=True` for 20% faster training and 60% memory reduction on supported model architectures.

=== Description ===
Liger (LinkedIn GPU Efficient Runtime) provides fused Triton kernels that replace standard PyTorch operations with optimized GPU implementations. These kernels fuse multiple operations (like RMSNorm + Linear, or CrossEntropy loss computation) into single GPU kernel calls, reducing memory overhead and improving throughput.

=== Usage ===
Use Liger kernels when training supported model architectures (Llama, Mistral, Gemma, Qwen, etc.) on CUDA GPUs. Provides significant memory and speed improvements for fine-tuning LLMs with minimal code changes.

== The Insight (Rule of Thumb) ==

* **Action:** Set `TrainingArguments(use_liger_kernel=True)` or `trainer.args.use_liger_kernel = True`
* **Value:** ~20% training speedup, ~60% peak memory reduction
* **Trade-off:** Requires `liger-kernel >= 0.3.0` dependency; only supports certain architectures
* **Compatibility:** Works with gradient checkpointing, FSDP, and other training optimizations

== Reasoning ==

Standard PyTorch operations execute as separate GPU kernels, each requiring:
1. Memory reads/writes between operations
2. Kernel launch overhead
3. Temporary tensor allocations

Liger kernels fuse operations:
- **FusedLinearCrossEntropyLoss:** Combines the lm_head linear layer with cross-entropy loss
- **FusedRMSNorm:** Combines RMS normalization operations
- **FusedRoPE:** Fuses rotary position embedding computation
- **FusedSwiGLU:** Combines the MLP gate operations

== Code Evidence ==

From `trainer.py:L477-498`:

<syntaxhighlight lang="python">
if self.args.use_liger_kernel:
    if is_liger_kernel_available():
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        # Prepare kernel config
        kernel_config = self.args.liger_kernel_config if \
            self.args.liger_kernel_config is not None else {}

        if isinstance(model, PreTrainedModel):
            # Patch the model with liger kernels
            _apply_liger_kernel_to_instance(model=model, **kernel_config)
        elif hasattr(model, "get_base_model"):
            # Patch PEFT wrapped models
            _apply_liger_kernel_to_instance(
                model=model.get_base_model(), **kernel_config
            )
        else:
            logger.warning(
                "The model is not an instance of PreTrainedModel. "
                "No liger kernels will be applied."
            )
    else:
        raise ImportError(
            "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 "
            "is not available. Please install it with `pip install liger-kernel`"
        )
</syntaxhighlight>

== Example Usage ==

<syntaxhighlight lang="python">
from transformers import TrainingArguments, Trainer

# Enable Liger kernels
training_args = TrainingArguments(
    output_dir="./results",
    use_liger_kernel=True,  # Enable Liger optimization
    per_device_train_batch_size=4,
    gradient_checkpointing=True,  # Works with grad checkpointing
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # ...
)

# With custom kernel config (advanced)
training_args = TrainingArguments(
    output_dir="./results",
    use_liger_kernel=True,
    liger_kernel_config={
        "rope": True,
        "swiglu": True,
        "cross_entropy": True,
        "fused_linear_cross_entropy": True,
        "rms_norm": True,
    },
)
</syntaxhighlight>

== Supported Architectures ==

{| class="wikitable"
|-
! Model Family !! Supported !! Notes
|-
| Llama, Llama 2, Llama 3 || Yes || Full support
|-
| Mistral, Mixtral || Yes || Full support
|-
| Gemma, Gemma 2 || Yes || Full support
|-
| Qwen, Qwen2 || Yes || Full support
|-
| Phi-3 || Yes || Full support
|-
| GPT-NeoX || Yes || Partial support
|-
| BERT, RoBERTa || No || Encoder models not supported
|}

== Performance Impact ==

{| class="wikitable"
|-
! Metric !! Without Liger !! With Liger !! Improvement
|-
| Training throughput || Baseline || +20% || 1.2x faster
|-
| Peak VRAM (7B model) || ~22GB || ~9GB || -60%
|-
| Tokens/second (A100) || 2000 || 2400 || +20%
|}

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_transformers_Training_execution]]
* [[uses_heuristic::Implementation:huggingface_transformers_TrainingArguments_setup]]
* [[uses_heuristic::Workflow:huggingface_transformers_Model_Training_Trainer]]
