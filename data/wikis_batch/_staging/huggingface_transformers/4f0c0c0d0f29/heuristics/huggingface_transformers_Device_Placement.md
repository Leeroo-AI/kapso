# Heuristic: huggingface_transformers_Device_Placement

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Experience|Trainer implementation patterns]]
|-
! Domains
| [[domain::Optimization]], [[domain::Distributed_Training]], [[domain::Memory_Management]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Heuristic for when to postpone automatic model-to-device placement during training initialization.

=== Description ===

The Trainer class implements intelligent logic to determine whether to automatically move the model to the target device (GPU/TPU) during initialization. In certain scenarios—model parallelism, DeepSpeed, FSDP, or full-precision evaluation—automatic device placement should be postponed or skipped entirely to avoid memory issues or interference with specialized device management systems.

=== Usage ===

Apply this heuristic when:
- Configuring `Trainer` with **large models that span multiple GPUs**
- Using **DeepSpeed** or **FSDP** for distributed training
- Running **fp16/bf16 full evaluation** without training
- Implementing **custom device management** strategies

This heuristic helps prevent OOM errors and ensures proper integration with distributed training frameworks.

== The Insight (Rule of Thumb) ==

* **Action**: Set `place_model_on_device=False` or let Trainer auto-detect
* **Conditions to postpone device placement**:
  1. Model Parallelism (MP): Model spans multiple GPUs via `hf_device_map`
  2. DeepSpeed: Handles its own device placement with half-precision loading
  3. Full fp16/bf16 evaluation: Model needs dtype casting before device move
  4. FSDP (XLA or PyTorch): Same reasoning as MP
* **Trade-off**: Manual device management required in some cases; automatic placement handles simple single-GPU scenarios

== Reasoning ==

**Why postpone device placement:**

1. **Model Parallelism**: When a model is split across multiple GPUs via `device_map`, moving it to a single device would break the distribution. The model already has its tensors on the correct devices.

2. **DeepSpeed fp16**: DeepSpeed loads the model in half the size (fp16) and manages device placement internally. Moving the model before DeepSpeed initialization wastes memory and may cause conflicts.

3. **Full fp16/bf16 Evaluation**: During evaluation-only runs, the model needs to be cast to the correct dtype first. Moving to GPU before casting wastes memory (full fp32 on GPU) and may cause OOM.

4. **FSDP**: Fully Sharded Data Parallel handles its own weight sharding and device placement. Pre-placing the model interferes with FSDP's initialization.

== Code Evidence ==

Device placement decision logic from `trainer.py:540-559`:

<syntaxhighlight lang="python">
# one place to sort out whether to place the model on device or not
# postpone switching model to cuda when:
# 1. MP - since we are trying to fit a much bigger than 1 gpu model
# 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
#    and we only use deepspeed for training at the moment
# 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
# 4. FSDP - same as MP
self.place_model_on_device = args.place_model_on_device
if (
    self.is_model_parallel
    or self.is_deepspeed_enabled
    or ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
    or self.is_fsdp_xla_enabled
    or self.is_fsdp_enabled
):
    self.place_model_on_device = False
</syntaxhighlight>

Model parallelism detection from `trainer.py:469-475`:

<syntaxhighlight lang="python">
self.is_model_parallel = False
if getattr(model, "hf_device_map", None) is not None:
    devices = [device for device in set(model.hf_device_map.values())
               if device not in ["cpu", "disk"]]
    if len(devices) > 1:
        self.is_model_parallel = True
    elif len(devices) == 1:
        self.is_model_parallel = self.args.device != torch.device(devices[0])
</syntaxhighlight>

Warning when model already distributed from `trainer.py:844-849`:

<syntaxhighlight lang="python">
def _move_model_to_device(self, model, device):
    if getattr(model, "hf_device_map", None) is not None:
        logger.warning(
            "The model is already on multiple devices. Skipping the move to device specified in `args`."
        )
        return
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:huggingface_transformers_Trainer_init]]
* [[uses_heuristic::Implementation:huggingface_transformers_Trainer_train]]
* [[uses_heuristic::Workflow:huggingface_transformers_Training]]
