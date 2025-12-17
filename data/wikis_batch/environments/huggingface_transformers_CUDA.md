# Environment: huggingface_transformers_CUDA

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|GPU Training|https://huggingface.co/docs/transformers/perf_train_gpu_one]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::GPU_Computing]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

GPU-accelerated environment with CUDA support for training and inference on NVIDIA GPUs.

=== Description ===

This environment extends the base PyTorch environment with CUDA GPU acceleration. It enables high-performance training and inference using NVIDIA GPUs, with support for mixed-precision training (fp16/bf16), TensorFloat-32 (TF32) on Ampere+ architectures, and multi-GPU configurations. The environment automatically detects available GPUs and configures optimal device placement.

=== Usage ===

Use this environment for:
- **GPU-accelerated training**: Fine-tuning models with `Trainer`
- **Distributed training**: Multi-GPU and multi-node setups via Accelerate
- **Fast inference**: GPU-accelerated pipeline inference
- **Quantization workflows**: BitsAndBytes, GPTQ, AWQ (requires this as base)

This is required when `device_map="auto"` or `device_map="cuda"` is specified, or when training with GPU acceleration.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+) || Windows with CUDA also supported
|-
| Hardware || NVIDIA GPU || Minimum: Pascal (GTX 10xx), Recommended: Ampere (RTX 30xx/A100)
|-
| VRAM || 8GB+ || 16GB+ recommended for 7B models, 40GB+ for larger
|-
| CUDA || 11.0+ || 11.8+ recommended for best compatibility
|-
| Driver || 450.0+ || Match CUDA version requirements
|}

== Dependencies ==

=== Inherits from PyTorch Environment ===

All dependencies from `huggingface_transformers_PyTorch`, plus:

=== Additional Requirements ===

* `torch` >= 2.2 (with CUDA support)
* CUDA Toolkit 11.0+
* cuDNN 8.0+
* `accelerate` >= 1.1.0 (for device management)

=== Optional GPU-Specific ===

* `deepspeed` >= 0.9.3 (for distributed training)
* `nvidia-ml-py3` (for GPU monitoring)
* `torch.compile` compatible (requires PyTorch 2.0+)

== Credentials ==

Same as PyTorch environment, plus:

* `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible to the process
* `LOCAL_RANK`: Set by distributed launchers for multi-GPU
* `WORLD_SIZE`: Total number of processes in distributed training

== Quick Install ==

<syntaxhighlight lang="bash">
# Install PyTorch with CUDA support
pip install torch>=2.2 --index-url https://download.pytorch.org/whl/cu118

# Install transformers and accelerate
pip install transformers accelerate

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
</syntaxhighlight>

== Code Evidence ==

CUDA detection in quantizer from `quantizer_bnb_8bit.py:86-97`:

<syntaxhighlight lang="python">
def update_device_map(self, device_map):
    if device_map is None:
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        elif is_torch_npu_available():
            device_map = {"": f"npu:{torch.npu.current_device()}"}
        elif is_torch_hpu_available():
            device_map = {"": f"hpu:{torch.hpu.current_device()}"}
        elif is_torch_xpu_available():
            device_map = {"": torch.xpu.current_device()}
        else:
            device_map = {"": "cpu"}
</syntaxhighlight>

Device placement logic in `trainer.py:540-549`:

<syntaxhighlight lang="python">
# one place to sort out whether to place the model on device or not
# postpone switching model to cuda when:
# 1. MP - since we are trying to fit a much bigger than 1 gpu model
# 2. fp16-enabled DeepSpeed loads the model in half the size
# 3. full bf16 or fp16 eval - since the model needs to be cast first
# 4. FSDP - same as MP
self.place_model_on_device = args.place_model_on_device
</syntaxhighlight>

Multi-device detection in `trainer.py:470-475`:

<syntaxhighlight lang="python">
if getattr(model, "hf_device_map", None) is not None:
    devices = [device for device in set(model.hf_device_map.values())
               if device not in ["cpu", "disk"]]
    if len(devices) > 1:
        self.is_model_parallel = True
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `CUDA out of memory` || Model too large for GPU VRAM || Reduce batch size, use gradient checkpointing, or quantization
|-
|| `CUDA device not found` || No GPU or driver issue || Check `nvidia-smi`, install CUDA drivers
|-
|| `RuntimeError: Expected all tensors on same device` || Mixed CPU/GPU tensors || Use `model.to(device)` consistently
|-
|| `torch.cuda.is_available() returns False` || PyTorch without CUDA support || Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
|}

== Compatibility Notes ==

* **Ampere GPUs (RTX 30xx, A100)**: TF32 enabled by default for faster matmul
* **Hopper GPUs (H100)**: FP8 support available
* **Mixed Precision**: bf16 recommended on Ampere+, fp16 on older GPUs
* **Multi-GPU**: Requires `accelerate` for proper device mapping
* **WSL2**: CUDA works on Windows Subsystem for Linux 2

== Related Pages ==

* [[requires_env::Implementation:huggingface_transformers_Trainer_train]]
* [[requires_env::Implementation:huggingface_transformers_BitsAndBytesConfig]]
* [[requires_env::Implementation:huggingface_transformers_get_hf_quantizer]]
* [[requires_env::Implementation:huggingface_transformers_quantizer_preprocess_model]]
