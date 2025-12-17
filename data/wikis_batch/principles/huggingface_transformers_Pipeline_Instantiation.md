{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Pipeline instantiation is the orchestration of task resolution, component loading, and configuration to create fully functional inference pipelines.

=== Description ===

Pipeline instantiation represents the factory pattern applied to machine learning inference workflows. It coordinates multiple subsystems—task validation, model loading, tokenizer initialization, device placement, and parameter configuration—to produce a unified, callable object that encapsulates an end-to-end ML inference workflow. The principle addresses the complexity of creating ML systems where dozens of configuration parameters interact in non-obvious ways, and sensible defaults must be provided while allowing expert overrides.

The instantiation process must handle several architectural challenges: component compatibility (ensuring the tokenizer matches the model), device orchestration (placing components on appropriate hardware), lazy loading (deferring expensive operations until necessary), and configuration inheritance (merging user parameters with task defaults and model-specific settings). It provides a progressive disclosure interface where simple use cases require minimal specification ("text-classification") while advanced scenarios can control every detail (custom models, multi-GPU placement, generation parameters).

The pattern separates the concerns of what components to use (task resolution), how to load them (component loading), and how to configure their interaction (pipeline instantiation). This separation enables testing each concern independently and allows the framework to evolve individual subsystems without breaking the overall interface.

=== Usage ===

Use pipeline instantiation when you need to:
* Provide a single entry point for creating diverse ML workflows
* Coordinate multiple interdependent components (model, tokenizer, processor)
* Apply sensible defaults while allowing granular control
* Handle device placement across CPU, GPU, and multi-device configurations
* Support both beginner-friendly and expert-level APIs
* Enable testing and validation before expensive resource loading

== Theoretical Basis ==

Pipeline instantiation follows a multi-stage factory pattern with progressive refinement:

```
Input: task, model, tokenizer, device, dtype, batch_size, **kwargs

Stage 1: Task Resolution and Validation
  if task is None and model is None:
    raise ValueError("Must specify task or model")

  if task is not None:
    normalized_task, task_defaults, task_options = resolve_task(task)
  else:
    # Infer task from model config
    normalized_task = infer_task_from_model(model)

Stage 2: Component Specification Resolution
  # Merge user specifications with task defaults
  model_name = model or task_defaults.get("default", {}).get("model")
  pipeline_class = pipeline_class or task_defaults["impl"]
  model_classes = task_defaults.get("pt", ())

  if model_name is None:
    raise ValueError(f"No model specified and no default for task {normalized_task}")

Stage 3: Configuration Loading
  # Load or validate config
  if isinstance(config, str):
    config = AutoConfig.from_pretrained(config, **config_kwargs)
  elif config is None and isinstance(model_name, str):
    config = AutoConfig.from_pretrained(model_name, **config_kwargs)

Stage 4: Model Loading
  if isinstance(model_name, str):
    model = load_model(
      model=model_name,
      config=config,
      model_classes=model_classes,
      task=normalized_task,
      device_map=device_map,
      dtype=dtype,
      **model_kwargs
    )
  else:
    model = model_name  # Already loaded

Stage 5: Preprocessing Component Loading
  # Load tokenizer (for text tasks)
  if tokenizer is None and task requires tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      use_fast=use_fast,
      **tokenizer_kwargs
    )

  # Load image processor (for vision tasks)
  if image_processor is None and task requires image_processor:
    image_processor = AutoImageProcessor.from_pretrained(
      model_name,
      **processor_kwargs
    )

  # Load feature extractor (for audio tasks)
  if feature_extractor is None and task requires feature_extractor:
    feature_extractor = AutoFeatureExtractor.from_pretrained(
      model_name,
      **processor_kwargs
    )

  # Load processor (for multimodal tasks)
  if processor is not None:
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

Stage 6: Device Placement
  # Handle device specification
  if device_map is not None:
    # Model already placed via device_map in loading
    pipeline_device = model.device
  elif device is not None:
    # Move model to specified device
    if isinstance(device, str):
      device = torch.device(device)
    elif isinstance(device, int):
      device = torch.device(f"cuda:{device}" if device >= 0 else "cpu")
    model.to(device)
    pipeline_device = device
  else:
    # Use model's current device
    pipeline_device = model.device

Stage 7: Pipeline Instantiation
  pipeline_instance = pipeline_class(
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    image_processor=image_processor,
    processor=processor,
    device=pipeline_device,
    task=normalized_task,
    batch_size=batch_size,
    **kwargs
  )

Stage 8: Parameter Sanitization
  # Pipeline extracts task-specific parameters
  preprocess_params, forward_params, postprocess_params = \
    pipeline_instance._sanitize_parameters(**kwargs)

  # Store for use during inference
  pipeline_instance._preprocess_params = preprocess_params
  pipeline_instance._forward_params = forward_params
  pipeline_instance._postprocess_params = postprocess_params

Output: Configured Pipeline instance
```

Key architectural patterns:

1. **Progressive Defaults**: Each stage provides defaults for the next
2. **Lazy Evaluation**: Expensive operations (loading) deferred until parameters finalized
3. **Polymorphic Inputs**: Accept strings or objects uniformly
4. **Explicit Over Implicit**: User specifications override all defaults
5. **Fail-Early Validation**: Check requirements before loading resources
6. **Component Compatibility**: Ensure tokenizer/processor matches model

The instantiation process creates a self-contained object that encapsulates:
* Preprocessing logic (text tokenization, image normalization)
* Model inference (forward pass)
* Postprocessing (formatting outputs)
* Batching and device management

This enables the simple calling interface: `pipeline("text-classification")("I love this!")`

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_pipeline_factory]]

=== Uses Principles ===
* [[uses::Principle:huggingface_transformers_Task_Resolution]]
* [[uses::Principle:huggingface_transformers_Pipeline_Component_Loading]]

=== Produces ===
* [[produces::Principle:huggingface_transformers_Pipeline_Preprocessing]]
* [[produces::Principle:huggingface_transformers_Pipeline_Model_Forward]]
* [[produces::Principle:huggingface_transformers_Pipeline_Postprocessing]]
