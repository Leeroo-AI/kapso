# Environment: huggingface_transformers_Training_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Trainer Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Training]], [[domain::Deep_Learning]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
GPU-accelerated training environment with PyTorch 2.2+, Accelerate 1.1.0+, and optional distributed training support.

=== Description ===
This environment provides the full training stack for fine-tuning and training Transformer models using the Trainer class. It includes PyTorch with GPU acceleration (CUDA, XPU, or other backends), the Accelerate library for device management and distributed training, and optional integrations with experiment tracking tools (Weights & Biases, MLflow, TensorBoard).

=== Usage ===
Use this environment for any **training** workflow using the `Trainer` class. Required for fine-tuning pretrained models, training from scratch, and running hyperparameter searches with Optuna or Ray Tune.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (recommended), Windows, macOS || Linux for distributed training
|-
| Python || Python >= 3.10 || Required by transformers
|-
| Hardware || NVIDIA GPU || 16GB+ VRAM recommended for LLMs
|-
| Memory || 16GB+ RAM || Higher for large models
|-
| Disk || 50GB+ SSD || For checkpoints and datasets
|}

== Dependencies ==

=== System Packages ===
* `cuda-toolkit` >= 11.8 - For NVIDIA GPUs
* `cudnn` >= 8.6 - For optimized CUDA operations

=== Python Packages ===
* `transformers` (this package)
* `torch` >= 2.2
* `accelerate` >= 1.1.0 - **Required** for Trainer
* `datasets` >= 2.15.0 - For data loading
* `safetensors` >= 0.4.3 - For checkpoint saving
* `tokenizers` >= 0.22.0
* `huggingface-hub` >= 1.2.1
* `numpy` >= 1.17
* `tqdm` >= 4.27

=== Optional Dependencies ===
* `evaluate` >= 0.4.6 - For metric computation
* `peft` >= 0.18.0 - For parameter-efficient fine-tuning
* `deepspeed` >= 0.9.3 - For ZeRO optimization
* `wandb` - For experiment tracking
* `tensorboard` - For logging
* `optuna` - For hyperparameter search
* `ray[tune]` >= 2.7.0 - For distributed HPO
* `bitsandbytes` - For 8-bit optimizers

== Credentials ==
The following environment variables are optional:
* `HF_TOKEN`: HuggingFace API token for private models
* `WANDB_API_KEY`: Weights & Biases API key
* `HF_HOME`: Custom cache directory

== Quick Install ==

<syntaxhighlight lang="bash">
# Basic training environment
pip install transformers torch accelerate datasets safetensors

# With evaluation metrics
pip install transformers[torch] accelerate datasets evaluate

# With experiment tracking
pip install transformers torch accelerate datasets wandb tensorboard

# With PEFT for efficient fine-tuning
pip install transformers torch accelerate datasets peft
</syntaxhighlight>

== Code Evidence ==

Trainer requires accelerate from `trainer.py:L279-284`:

<syntaxhighlight lang="python">
@requires(
    backends=(
        "torch",
        "accelerate",
    )
)
class Trainer:
</syntaxhighlight>

Minimum versions from `dependency_versions_table.py`:

<syntaxhighlight lang="python">
deps = {
    "accelerate": "accelerate>=1.1.0",
    "datasets": "datasets>=2.15.0",
    "peft": "peft>=0.18.0",
    "deepspeed": "deepspeed>=0.9.3",
}
</syntaxhighlight>

Device setup from `training_args.py`:

<syntaxhighlight lang="python">
if is_torch_cuda_available():
    device = torch.device("cuda")
elif is_torch_xpu_available():
    device = torch.device("xpu")
elif is_torch_npu_available():
    device = torch.device("npu")
elif is_torch_mps_available():
    device = torch.device("mps")
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: accelerate>=1.1.0 is required` || Accelerate not installed || `pip install accelerate>=1.1.0`
|-
|| `RuntimeError: CUDA out of memory` || Model too large for GPU || Enable gradient checkpointing, reduce batch size, or use DeepSpeed
|-
|| `ValueError: metric_for_best_model must be provided` || Using `load_best_model_at_end` without metric || Set `metric_for_best_model` in TrainingArguments
|-
|| `AssertionError: Cannot run Trainer.evaluate if model is on meta device` || Model not materialized || Load model properly before evaluation
|-
|| `ImportError: peft>=0.18.0 is required` || PEFT version too old || `pip install -U peft`
|}

== Compatibility Notes ==

* **Single GPU:** Works out of the box with `device="cuda"`
* **Multi-GPU:** Use `accelerate launch` or `torchrun` for distributed training
* **FSDP:** Requires PyTorch >= 2.2 and accelerate config
* **DeepSpeed:** Requires `deepspeed` config file
* **Apple Silicon:** Limited support, bf16 not available on MPS
* **Mixed Precision:** fp16 requires GPU with Tensor Cores, bf16 requires Ampere+

== Related Pages ==
* [[requires_env::Implementation:huggingface_transformers_TrainingArguments_setup]]
* [[requires_env::Implementation:huggingface_transformers_DataCollator_usage]]
* [[requires_env::Implementation:huggingface_transformers_Trainer_init]]
* [[requires_env::Implementation:huggingface_transformers_Optimizer_creation]]
* [[requires_env::Implementation:huggingface_transformers_Training_execution]]
* [[requires_env::Implementation:huggingface_transformers_Evaluate]]
* [[requires_env::Implementation:huggingface_transformers_Model_saving]]
