{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Model Loading|https://huggingface.co/docs/transformers/main_classes/model]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Post Loading Hooks are finalization procedures executed after weights are loaded to ensure model consistency, establish parameter relationships, and complete initialization.

=== Description ===
After model weights have been loaded into the architecture, additional processing is often required to make the model fully functional. Post Loading Hooks encompass operations like tying weights (sharing parameters between multiple layers, such as embedding and output layers), registering tied weight keys for proper tracking, setting up parallel execution plans for distributed inference, and executing any model-specific initialization logic. These hooks ensure that models are in a valid, ready-to-use state by establishing parameter relationships, validating model structure, and configuring runtime behavior.

The principle handles two critical operations: weight tying (where certain layers share the same underlying parameters to reduce memory and enforce consistency) and post-initialization setup (where parallel execution plans, attention configurations, and generation parameters are finalized). This is particularly important for language models where input embeddings and output projection layers often share weights.

=== Usage ===
Post Loading Hooks should be applied as the final step in model loading, after all weights have been loaded and devices have been assigned. They are essential when loading pretrained models, ensuring proper weight sharing, validating model integrity, and preparing models for inference or fine-tuning.

== Theoretical Basis ==

Post-loading finalization involves two main operations:

1. '''Weight Tying Mechanism''':

Weight tying establishes parameter sharing relationships:
```
For a language model with:
- Input embeddings E_in ∈ R^(V×D)  (V=vocab_size, D=hidden_dim)
- Output projection W_out ∈ R^(D×V)

Weight tying creates: W_out = E_in^T

This reduces parameters from (V×D + D×V) to (V×D) by sharing memory.
```

2. '''Tie Weights Algorithm''':
```
function tie_weights(model, missing_keys=None, recompute_mapping=False):
    # Get tied weight mapping {target: source}
    if recompute_mapping:
        tied_keys = model.get_expanded_tied_weights_keys(all_submodels=True)
    else:
        tied_keys = model.all_tied_weights_keys

    for target_name, source_name in tied_keys.items():
        # Handle checkpoint loading symmetrically
        if missing_keys is not None:
            source_exists = source_name not in missing_keys
            target_exists = target_name not in missing_keys

            # Both exist → config error, don't tie
            if source_exists and target_exists:
                warn("Both parameters exist, skipping tie")
                continue

            # Neither exists → will be caught later
            if not source_exists and not target_exists:
                continue

            # Swap if only target exists (checkpoint has target but not source)
            if target_exists and not source_exists:
                source_name, target_name = target_name, source_name

        # Get source and target modules
        source_module, source_param_name = get_module_and_param(model, source_name)
        target_module, target_param_name = get_module_and_param(model, target_name)

        # Perform tying (make target point to source)
        source_param = getattr(source_module, source_param_name)
        setattr(target_module, target_param_name, source_param)

        # Mark as tied
        mark_as_tied(target_name)

        # Remove from missing keys if present
        if missing_keys and target_name in missing_keys:
            missing_keys.remove(target_name)
```

3. '''Post Initialization Algorithm''':
```
function post_init(model):
    # Initialize parallel execution plans
    model._tp_plan = {}  # Tensor parallelism
    model._ep_plan = {}  # Expert parallelism
    model._pp_plan = {}  # Pipeline parallelism

    # Register current model's tied weights
    model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(
        all_submodels=False
    )

    # If base model, attach config-level parallel plans
    if model.base_model is model:
        model._tp_plan = copy(config.base_model_tp_plan)
        model._ep_plan = copy(config.base_model_ep_plan)
        model._pp_plan = copy(config.base_model_pp_plan)

    # Recursively collect plans from submodules
    for name, child_module in model.named_children():
        if hasattr(child_module, '_tp_plan'):
            model._tp_plan.update(prefix_keys(child_module._tp_plan, name))
        if hasattr(child_module, '_ep_plan'):
            model._ep_plan.update(prefix_keys(child_module._ep_plan, name))
        if hasattr(child_module, '_pp_plan'):
            model._pp_plan.update(prefix_keys(child_module._pp_plan, name))
        if hasattr(child_module, 'all_tied_weights_keys'):
            model.all_tied_weights_keys.update(
                prefix_keys(child_module.all_tied_weights_keys, name)
            )

    return model
```

'''Key Properties''':
* '''Idempotence''': Multiple calls to post_init/tie_weights should be safe
* '''Symmetry''': Weight tying works regardless of which parameter is in checkpoint
* '''Memory Efficiency''': Tied weights use single memory allocation
* '''Gradient Consistency''': Updates to tied parameters affect all references

'''Example Weight Tying''':
```
Before tying:
  embeddings.weight: Tensor(30522, 768) @ 0x1000
  lm_head.weight:    Tensor(768, 30522) @ 0x2000
  Total memory: 2 × 30522 × 768 × 4 bytes = 178 MB

After tying:
  embeddings.weight: Tensor(30522, 768) @ 0x1000
  lm_head.weight:    → points to embeddings.weight.T
  Total memory: 1 × 30522 × 768 × 4 bytes = 89 MB
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Post_init_processing]]
