# RoadModel (RoAd Model Manager)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/road/model.py`
**Lines of Code:** 163
**Language:** Python

RoadModel manages RoAd adapter creation and integration, handling layer replacement and adapter lifecycle for rotation-based parameter-efficient fine-tuning.

## Core Implementation

```python
class RoadModel(BaseTuner):
    """RoAd Model Manager

    Creates and manages rotation adaptation layers
    """

    prefix: str = "road_"
    tuner_layer_cls = RoadLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_ROAD_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        road_config: RoadConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        variant = road_config.variant
        group_size = road_config.group_size

        kwargs = {
            "variant": variant,
            "group_size": group_size,
            "init_weights": road_config.init_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        if isinstance(target, RoadLayer):
            target.update_layer(adapter_name, variant, group_size, init_weights=road_config.init_weights)
        else:
            device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
            new_module = self._create_new_module(road_config, adapter_name, target, device_map=device_map, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(road_config: RoadConfig, adapter_name, target, **kwargs):
        dispatchers = []

        # Handle quantized layers
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit
            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit
            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.append(dispatch_default)

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, road_config=road_config, **kwargs)
            if new_module is not None:
                break

        if new_module is None:
            raise ValueError(f"Target module {target} not supported. Only torch.nn.Linear supported.")

        return new_module
```

## Mixed Batch Inference

**Context Manager:** `_enable_peft_forward_hooks()`

Enables different adapters for different samples in a batch:

```python
@contextmanager
def _enable_peft_forward_hooks(self, *args, **kwargs):
    adapter_names = kwargs.pop("adapter_names", None)
    if adapter_names is None:
        yield
        return

    if self.training:
        raise ValueError("Cannot pass `adapter_names` in training mode")

    # Validate adapter names
    expected_adapters = set()
    for layer in self.modules():
        if isinstance(layer, RoadLayer):
            expected_adapters |= layer.road_theta.keys()

    unique_adapters = {name for name in adapter_names if name != "__base__"}
    unexpected_adapters = unique_adapters - expected_adapters
    if unexpected_adapters:
        raise ValueError(f"Non-existing adapters: {', '.join(sorted(unexpected_adapters))}")

    # Inject hooks
    hook_handles = []
    for module in self.modules():
        if isinstance(module, RoadLayer):
            pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
            handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
            hook_handles.append(handle)

    yield

    # Remove hooks
    for handle in hook_handles:
        handle.remove()
```

**Usage:**

```python
# Inference with different adapters per sample
with model._enable_peft_forward_hooks(adapter_names=["task1", "task2", "task1"]):
    output = model(batch)  # Each sample uses specified adapter
```

## Supported Layers

- `torch.nn.Linear`: Standard dense layers
- `bnb.nn.Linear8bitLt`: 8-bit quantized layers
- `bnb.nn.Linear4bit`: 4-bit quantized layers

## Usage Example

```python
from peft import RoadConfig, get_peft_model

config = RoadConfig(variant="road_1", group_size=64)
model = get_peft_model(base_model, config)

# Training
model.train()
for batch in dataloader:
    output = model(batch)
    loss.backward()
    optimizer.step()

# Inference with mixed adapters
model.eval()
with model._enable_peft_forward_hooks(adapter_names=["task1", "task2"]):
    output = model(batch)
```

## References

- **Type**: `PeftType.ROAD`
- **Prefix**: "road_"
- **Key Feature**: Mixed-batch adapter inference
