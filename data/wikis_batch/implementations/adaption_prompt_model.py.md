# Implementation: adaption_prompt/model.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/adaption_prompt/model.py`
- **Size**: 169 lines
- **Description**: Multi-adapter management for adaption prompts

## Overview

This module manages multiple adaption prompt adapters, handling adapter creation, switching, and enabling/disabling. It uses a caching mechanism to store inactive adapters.

## Core Class: AdaptionPromptModel

### Architecture

**State Management**:
```python
self.peft_config: dict[str, AdaptionPromptConfig] = {}
self._parents: dict[str, list[nn.Module]] = {}        # Module parents
self._cached_adapters: dict[str, list] = {}           # Inactive adapters
self._active_adapter: str = None                       # Current adapter
self._enabled: bool = True                             # Adapter state
```

### Key Methods

**add_adapter(adapter_name, config)**:
1. Find target attention modules (e.g., "self_attn")
2. Select top L layers
3. Create AdaptedAttention wrappers
4. Store parent references

**set_adapter(adapter_name)**:
1. Remove current adapter (store in cache)
2. Retrieve new adapter from cache
3. Swap modules in parent

**enable/disable_adapter_layers()**:
- Remove/restore adapters from model
- Maintain cache for quick re-enable

### Multi-Adapter Pattern

**Cache-Based Switching**:
- Only one adapter active in model at a time
- Others stored in `_cached_adapters`
- Fast switching without re-initialization

**State Consistency**:
- Active adapter never in cache
- Removing adapter moves it to cache
- Setting adapter removes it from cache

## Cross-References

- **Config**: `adaption_prompt/config.py`
- **Layer**: `adaption_prompt/layer.py`
