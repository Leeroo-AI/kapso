# File: `src/peft/tuners/poly/model.py`

**Category:** model

| Property | Value |
|----------|-------|
| Lines | 105 |
| Classes | `PolyModel` |
| Imports | config, contextlib, layer, peft, torch, typing |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Main model class that manages Poly adapters, handling adapter injection, task-based routing, and forward pass coordination for multi-task learning.

**Mechanism:**
- **PolyModel class** extends `BaseTuner` with Poly-specific functionality:
  - `prefix = "poly_"`: Naming prefix for Poly parameters
  - `tuner_layer_cls = PolyLayer`: Layer class to use
  - `target_module_mapping`: Maps model architectures to default target modules

- **Key Methods:**
  - `_create_and_replace()`: Creates Poly layers and replaces target modules
  - `_create_new_module()`: Factory method that creates `Linear` Poly layers (only torch.nn.Linear supported)
  - `_register_pre_hooks()`: Registers forward hooks to inject task_ids into the forward pass
  - `_manage_pre_hooks()`: Context manager that safely manages hook lifecycle
  - `forward()`: Wraps model forward pass with pre-hooks for task ID injection
  - `generate()`: Wraps model generation with pre-hooks for task ID injection

- **Hook Management:**
  - Uses pre-forward hooks to inject `task_ids` into kwargs before each forward call
  - Context manager pattern ensures hooks are properly removed after use
  - Hooks are registered on all `Linear` Poly layers

**Significance:** Core orchestrator for Poly multi-task learning. The key innovation is the hook-based task ID injection mechanism, which allows different tasks to use different routing weights during inference. This enables a single model to handle multiple tasks efficiently by routing inputs through different combinations of the shared LoRA modules. The model automatically handles adapter activation/deactivation and ensures task IDs reach the routing layers.
