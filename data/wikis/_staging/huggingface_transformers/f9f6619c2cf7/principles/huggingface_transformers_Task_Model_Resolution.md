{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Documentation|https://huggingface.co/docs/transformers]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Task-based pipeline resolution automatically selects the appropriate model architecture and processing components for a given machine learning task.

=== Description ===
Task-model resolution is the process of determining which model, tokenizer, feature extractor, and other processing components are required for a specific ML task (e.g., "text-classification", "image-to-text", "question-answering"). Rather than requiring users to manually specify each component, this principle allows systems to automatically infer and load compatible components based on either an explicit task string or model identifier. This simplifies the user experience by reducing the configuration burden while ensuring that all components work together correctly.

The resolution process handles multiple scenarios: using task defaults when no model is specified, inferring tasks from model configurations, loading task-specific preprocessing components, and validating component compatibility. This pattern is particularly useful in high-level APIs where users want quick access to inference capabilities without needing deep knowledge of model architectures.

=== Usage ===
Apply this principle when building factory functions or constructors that need to:
* Instantiate complete inference pipelines from minimal user input
* Map abstract task descriptions to concrete model implementations
* Automatically load compatible preprocessing components for a given model
* Provide sensible defaults while allowing expert users to override components
* Support both hub-based model identifiers and local model instances

== Theoretical Basis ==

=== Resolution Algorithm ===

The task-model resolution follows a priority-based decision tree:

'''Step 1: Validate Input'''
<pre>
IF task is None AND model is None:
    RAISE error (insufficient information)
IF task is None:
    task = infer_task_from_model(model)
</pre>

'''Step 2: Resolve Configuration'''
<pre>
config = NONE
IF config_param is string:
    config = load_config(config_param)
ELSE IF model is string:
    config = load_config(model)
</pre>

'''Step 3: Determine Pipeline Class'''
<pre>
IF task in custom_tasks:
    pipeline_class = load_custom_pipeline(config, task)
ELSE:
    pipeline_class = TASK_MAPPING[normalize(task)]
</pre>

'''Step 4: Resolve Model'''
<pre>
IF model is None:
    model = get_default_model_for_task(task)
IF model is string:
    model = load_model_from_identifier(model, config)
</pre>

'''Step 5: Resolve Processors'''
<pre>
FOR processor_type IN [tokenizer, feature_extractor, image_processor]:
    IF processor_type is None:
        processor_type = auto_load_processor(model, processor_type, config)
</pre>

=== Component Compatibility Matrix ===

The resolution ensures components satisfy task-specific requirements:

<pre>
Task Requirements:
- NLP tasks → require tokenizer
- Vision tasks → require image_processor OR feature_extractor
- Audio tasks → require feature_extractor
- Multimodal tasks → require multiple processors

Validation Rules:
1. IF tokenizer specified WITHOUT model → ERROR
2. IF feature_extractor specified WITHOUT model → ERROR
3. IF task requires processor AND processor unavailable → ERROR
</pre>

=== Priority Hierarchy ===

When multiple sources provide configuration, resolution follows this precedence:

<pre>
1. Explicit user-provided instances (highest priority)
2. User-provided identifiers (model names/paths)
3. Config-based inference
4. Task-based defaults
5. Framework fallbacks (lowest priority)
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Pipeline_factory_function]]
