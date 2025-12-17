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

Task resolution is the process of mapping user-specified task identifiers to their canonical forms and associated implementation components.

=== Description ===

Task resolution is a fundamental pattern in machine learning frameworks that provides a unified interface for diverse ML capabilities. Rather than requiring users to know specific class names and configurations, task resolution allows them to specify high-level intent (like "text-classification" or "translation") and automatically resolves this to the appropriate pipeline class, model architecture, and default pretrained weights. This abstraction layer handles several critical concerns: alias normalization (mapping "sentiment-analysis" to "text-classification"), parametrized task expansion (parsing "translation_en_to_fr" into task="translation" with language parameters), and default model selection (providing sensible starting points for each task).

The principle separates concerns between what the user wants to accomplish (the task) and how the system accomplishes it (the implementation details). This enables framework evolution without breaking user code, as internal class names and architectures can change while maintaining stable task identifiers. It also provides extensibility, allowing new tasks to be registered without modifying core framework code.

=== Usage ===

Use task resolution when you need to:
* Provide user-friendly task names that map to complex implementations
* Support task aliases for common use cases (e.g., "ner" -> "token-classification")
* Handle parametrized tasks with embedded parameters (e.g., "translation_xx_to_yy")
* Enable task discovery and validation before resource loading
* Maintain backward compatibility while evolving implementations

== Theoretical Basis ==

Task resolution follows a registry pattern combined with name normalization:

```
Input: task_string (e.g., "sentiment-analysis", "translation_en_to_fr")

Step 1: Alias Resolution
  - Lookup task_string in alias_map
  - If found, replace with canonical name
  - Example: "sentiment-analysis" -> "text-classification"

Step 2: Parameter Extraction
  - Parse task_string for embedded parameters
  - Extract structured parameters from string
  - Example: "translation_en_to_fr" -> ("translation", {src: "en", tgt: "fr"})

Step 3: Registry Lookup
  - Query task registry with normalized name
  - Retrieve task configuration bundle
  - Bundle contains:
    * Pipeline implementation class
    * Compatible model architecture classes
    * Default pretrained model identifier
    * Task-specific configuration defaults

Step 4: Validation
  - Verify task exists in registry
  - Check framework capabilities (e.g., PyTorch available)
  - Raise informative errors for unsupported tasks

Output: (normalized_task, task_config, task_parameters)
```

The task registry is typically implemented as a centralized mapping data structure:

```
registry = {
  "text-classification": {
    "impl": TextClassificationPipeline,
    "pt": (AutoModelForSequenceClassification,),
    "default": {"model": "distilbert-base-uncased-finetuned-sst-2-english"},
    "type": "text"
  },
  "translation": {
    "impl": TranslationPipeline,
    "pt": (AutoModelForSeq2SeqLM,),
    "default": {"model": "t5-base"},
    "type": "text"
  },
  ...
}

alias_map = {
  "sentiment-analysis": "text-classification",
  "ner": "token-classification",
  "text-to-speech": "text-to-audio"
}
```

This design enables O(1) lookup time and clear separation between user-facing identifiers and internal implementation details.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_check_task]]

=== Related Principles ===
* [[related::Principle:huggingface_transformers_Pipeline_Instantiation]]
