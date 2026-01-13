# Environment: TRL (Transformer Reinforcement Learning) Integration

## Category
Software/Training

## Summary
Unsloth deeply integrates with Hugging Face TRL library, providing backwards compatibility patches, automatic sample packing, and gradient accumulation fixes across TRL versions.

## Requirements

### Software Requirements
| Package | Version Constraint | Evidence |
|---------|-------------------|----------|
| TRL | >= 0.11.0 | Backwards compatibility layer provided |
| transformers | >= 4.45.2 | For gradient accumulation fix |

### Version Compatibility Matrix

| TRL Version | Unsloth Behavior | Notes |
|-------------|------------------|-------|
| < 0.11.0 | No patching | Older API |
| 0.11.0 - 0.12.x | Backwards compat patches | Parameter migration |
| >= 0.13.0 | Full patching | Config class changes |

## TRL Patching System

From `unsloth/trainer.py:410-438`:

```python
def _patch_trl_trainer():
    import trl

    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"):
        return
    if Version(trl) <= Version("0.11.0"):
        return

    import trl.trainer

    trl_classes = dir(trl.trainer)
    trl_trainers = set(
        x[: -len("Trainer")] for x in trl_classes if x.endswith("Trainer")
    )
    trl_configs = set(x[: -len("Config")] for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        exec(
            f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)",
            globals(),
        )
```

## Backwards Compatibility

From `trainer.py:203-272`:

The `_backwards_compatible_trainer` function handles:
1. `tokenizer` -> `processing_class` rename
2. Config parameter migration from `args` to dedicated Config classes
3. Mutual exclusivity checks bypass

```python
if "processing_class" in trainer_params and "tokenizer" in kwargs:
    kwargs["processing_class"] = kwargs.pop("tokenizer")
```

## Sample Packing Auto-Configuration

From `trainer.py:275-408`:

```python
PADDING_FREE_BLOCKLIST = {
    "gemma2",   # Uses slow_attention_softcapping
    "gpt_oss",  # Uses Flex Attention
}
```

### Auto-Packing Logic
1. Check if packing requested and not blocked
2. Configure sample packing or padding-free training
3. Handle errors gracefully with fallback

```python
if _should_pack(config_arg) and not blocked:
    configure_sample_packing(config_arg)
    packing_active = True
```

## Gradient Accumulation Fix

From `trainer.py:105-124`:

```python
if Version(transformers_version) > Version("4.45.2"):
    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)
else:
    def unsloth_train(trainer, *args, **kwargs):
        # Custom gradient accumulation fix
        return _unsloth_train(trainer)
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `UNSLOTH_DISABLE_AUTO_PADDING_FREE` | Disable auto padding-free | `0` |
| `UNSLOTH_RETURN_LOGITS` | Force logit computation | `0` |

## UnslothTrainer Class

From `trainer.py:182-198`:

```python
class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()
        # Custom optimizer with separate embedding learning rate
```

## Source Evidence

- TRL Patching: `unsloth/trainer.py:410-438`
- Backwards Compat: `unsloth/trainer.py:203-272`
- Sample Packing: `unsloth/trainer.py:275-408`
- Gradient Fix: `unsloth/trainer.py:105-124`

## Backlinks

[[required_by::Implementation:Unslothai_Unsloth_SFTTrainer_train]]
[[required_by::Implementation:Unslothai_Unsloth_SFTConfig]]
[[required_by::Implementation:Unslothai_Unsloth_GRPOTrainer_train]]

## Related

- [[Environment:Unslothai_Unsloth_PEFT]]
- [[Heuristic:Unslothai_Unsloth_Gradient_Checkpointing]]
