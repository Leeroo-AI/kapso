# Implementation: huggingface_peft_hotswap_adapter

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Hotswap|https://huggingface.co/docs/peft/package_reference/helpers]]
|-
! Domains
| [[domain::Hotswap]], [[domain::Deployment]], [[domain::Production]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/utils/hotswap.py:L545-631` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_hotswap_adapter]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`hotswap_adapter()` performs in-place replacement of adapter weights without disrupting inference. This enables zero-downtime adapter switching in production deployments, especially when combined with `torch.compile`.

== API Signature ==

```python
from peft.utils.hotswap import hotswap_adapter

hotswap_adapter(
    model: PeftModel,
    model_id: str,
    adapter_name: str = "default",
    torch_device: Optional[str] = None,
    **kwargs
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | PeftModel | required | Model with adapter to replace |
| `model_id` | str | required | Path or Hub ID of new adapter |
| `adapter_name` | str | "default" | Name of adapter slot to replace |
| `torch_device` | str | None | Device for new weights |

== Returns ==

None - Modifies model weights in place.

== Usage Examples ==

=== Basic Hotswap ===
```python
from peft.utils.hotswap import hotswap_adapter

# Swap to new adapter without reloading model
hotswap_adapter(model, "path/to/new_adapter")
```

=== Production Hotswap with Compile ===
```python
from peft import PeftModel
from peft.utils.hotswap import prepare_model_for_compiled_hotswap, hotswap_adapter
import torch

# Initial setup
model = PeftModel.from_pretrained(base, "adapter_v1")
prepare_model_for_compiled_hotswap(model, target_rank=64)
model = torch.compile(model)

# Serve requests...

# Hot swap to new adapter (no downtime, no recompile)
hotswap_adapter(model, "adapter_v2")

# Continue serving with new adapter
```

== Key Behavior ==

1. **Loads new weights** - Downloads/reads new adapter weights
2. **In-place copy** - Uses `tensor.copy_()` to update weights
3. **Preserves compilation** - torch.compile graph remains valid
4. **Handles rank padding** - Works with prepared models

== Why This Matters ==

Traditional adapter switching requires:
- Deleting old adapter
- Loading new adapter
- Recompiling if using torch.compile

Hotswap enables continuous serving during adapter updates.

== Related Functions ==

* [[huggingface_peft_prepare_model_for_compiled_hotswap]] - Prepare for hotswap
* [[huggingface_peft_load_adapter]] - Alternative: load then switch

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Hotswap_Execution]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:Hotswap]]
