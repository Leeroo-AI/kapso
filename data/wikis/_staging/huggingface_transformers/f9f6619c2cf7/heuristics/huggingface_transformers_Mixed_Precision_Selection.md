# Heuristic: huggingface_transformers_Mixed_Precision_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Mixed Precision Training|https://huggingface.co/docs/transformers/perf_train_gpu_one#mixed-precision-training]]
|-
! Domains
| [[domain::Optimization]], [[domain::Training]], [[domain::Precision]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Choose bf16 over fp16 for modern GPUs (Ampere+) to avoid gradient overflow issues while maintaining 2x memory savings.

=== Description ===
Mixed precision training uses lower-precision formats (fp16 or bf16) for most operations while keeping critical computations in fp32. This halves memory usage and speeds up training on Tensor Core GPUs. BFloat16 (bf16) is preferred on Ampere and newer architectures because it has the same exponent range as fp32, avoiding the gradient underflow/overflow issues common with fp16.

=== Usage ===
Use mixed precision for all GPU training to reduce memory and increase speed. Choose `bf16=True` on Ampere+ GPUs (RTX 30xx, A100, H100) and `fp16=True` only on older architectures (V100, RTX 20xx).

== The Insight (Rule of Thumb) ==

* **Action:** Set `bf16=True` for Ampere+ GPUs, `fp16=True` for Volta/Turing
* **Value:** bf16 is default recommendation; fp16 requires loss scaling
* **Trade-off:** ~2x memory reduction, ~1.5-2x speed improvement on Tensor Cores
* **Warning:** Never enable both `fp16=True` and `bf16=True` simultaneously

== Reasoning ==

- **fp16:** 5 exponent bits, 10 mantissa bits. Prone to overflow (values > 65504 become inf) and underflow (small gradients become 0).
- **bf16:** 8 exponent bits (same as fp32), 7 mantissa bits. Same dynamic range as fp32, just lower precision.

BF16 is safer for training because:
1. No gradient overflow in typical training scenarios
2. No need for loss scaling
3. More numerically stable for large models

== Code Evidence ==

From `training_args.py:L377-387`:

<syntaxhighlight lang="python">
bf16 (`bool`, *optional*, defaults to `False`):
    Whether to use bf16 16-bit (mixed) precision training instead of 32-bit
    training. Requires Ampere or higher NVIDIA architecture or Intel XPU
    or using CPU (use_cpu) or Ascend NPU.
fp16 (`bool`, *optional*, defaults to `False`):
    Whether to use fp16 16-bit (mixed) precision training instead of 32-bit
    training.
</syntaxhighlight>

Hardware detection from `training_args.py`:

<syntaxhighlight lang="python">
if is_torch_bf16_gpu_available():
    # BF16 is available on Ampere+ GPUs
    ...
if is_torch_cuda_available():
    # FP16 is available on all CUDA GPUs
    ...
</syntaxhighlight>

TF32 mode from `training_args.py:L388-392`:

<syntaxhighlight lang="python">
tf32 (`bool`, *optional*):
    Whether to enable the TF32 mode, available in Ampere and newer GPU
    architectures. The default value depends on PyTorch's version default.
</syntaxhighlight>

== Example Usage ==

<syntaxhighlight lang="python">
from transformers import TrainingArguments

# For Ampere+ GPUs (RTX 30xx, A100, H100)
training_args = TrainingArguments(
    output_dir="./results",
    bf16=True,  # Recommended for modern GPUs
    # tf32=True,  # Also enable TF32 for additional speedup
    per_device_train_batch_size=4,
)

# For older GPUs (V100, RTX 20xx)
training_args = TrainingArguments(
    output_dir="./results",
    fp16=True,  # Use fp16 with loss scaling
    per_device_train_batch_size=4,
)
</syntaxhighlight>

== GPU Compatibility ==

{| class="wikitable"
|-
! GPU Architecture !! fp16 !! bf16 !! tf32 !! Recommended
|-
| Volta (V100) || Yes || No || No || fp16
|-
| Turing (RTX 20xx, T4) || Yes || No || No || fp16
|-
| Ampere (RTX 30xx, A100) || Yes || Yes || Yes || bf16 + tf32
|-
| Hopper (H100) || Yes || Yes || Yes || bf16 + tf32
|-
| Intel XPU || Yes || Yes || N/A || bf16
|}

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_transformers_TrainingArguments_setup]]
* [[uses_heuristic::Implementation:huggingface_transformers_Training_execution]]
* [[uses_heuristic::Workflow:huggingface_transformers_Model_Training_Trainer]]
