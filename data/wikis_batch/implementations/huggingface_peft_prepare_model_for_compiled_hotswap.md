# Implementation: huggingface_peft_prepare_model_for_compiled_hotswap

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Hotswap|https://huggingface.co/docs/peft/package_reference/helpers]]
|-
! Domains
| [[domain::Hotswap]], [[domain::Deployment]], [[domain::torch_compile]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/utils/hotswap.py:L268-367` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_prepare_model_for_compiled_hotswap]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`prepare_model_for_compiled_hotswap()` prepares a PEFT model for efficient adapter hot-swapping with `torch.compile`. It pads LoRA matrices to a target rank so that adapters of different ranks can be swapped without recompilation.

== API Signature ==

```python
from peft.utils.hotswap import prepare_model_for_compiled_hotswap

prepare_model_for_compiled_hotswap(
    model: PeftModel,
    target_rank: int,
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | PeftModel | required | The PEFT model to prepare |
| `target_rank` | int | required | Maximum rank to support for swapping |

== Returns ==

None - Modifies model in place.

== Usage Examples ==

=== Prepare for Hotswap ===
```python
from peft import PeftModel
from peft.utils.hotswap import prepare_model_for_compiled_hotswap
import torch

# Load model with adapter
model = PeftModel.from_pretrained(base_model, "adapter_r8")

# Prepare for swapping (max rank 64)
prepare_model_for_compiled_hotswap(model, target_rank=64)

# Compile the model
model = torch.compile(model)

# Now can swap to any adapter with rank <= 64 without recompile
```

== Key Behavior ==

1. **Pads LoRA matrices** - Extends A/B matrices to target_rank
2. **Enables torch.compile** - Compiled model won't need recompile on swap
3. **Zero padding** - Extra dimensions are zeros, no functional change

== Why This Matters ==

Without preparation, swapping adapters with different ranks forces `torch.compile` to recompile (expensive). With padding, all adapters have the same tensor shapes, preserving the compiled graph.

== Related Functions ==

* [[huggingface_peft_hotswap_adapter]] - Perform the actual swap
* [[huggingface_peft_PeftModel_from_pretrained]] - Load initial adapter

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Hotswap_Preparation]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:Hotswap]]
