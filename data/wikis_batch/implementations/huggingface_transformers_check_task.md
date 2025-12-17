{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for validating and normalizing pipeline task names provided by the HuggingFace Transformers library.

=== Description ===

The `check_task` function is the entry point for task resolution in the Transformers pipeline system. It validates a user-provided task string (e.g., "text-classification", "sentiment-analysis") and returns the normalized task name, the corresponding pipeline class, and default model information. This function handles task aliases (like "sentiment-analysis" -> "text-classification") and parametrized tasks (like "translation_en_to_fr"). It delegates to the PIPELINE_REGISTRY which maintains a mapping of all supported tasks to their implementation classes.

=== Usage ===

Use this function when you need to:
* Validate a task string before creating a pipeline
* Resolve task aliases to their canonical names
* Determine which pipeline class and model to use for a given task
* Get default model recommendations for a specific task

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/__init__.py (lines 371-417)

=== Signature ===
<syntaxhighlight lang="python">
def check_task(task: str) -> tuple[str, dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default
    Pipeline and Model classes, and default models if they exist.

    Args:
        task (str): The task defining which pipeline will be returned.

    Returns:
        tuple[str, dict, Any]: The normalized task name (removed alias and options),
        the actual dictionary required to initialize the pipeline, and some extra
        task options for parametrized tasks like "translation_xx_to_yy"
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.pipelines import check_task
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| task || str || Yes || Task identifier string (e.g., "text-classification", "sentiment-analysis", "translation_en_to_fr")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| normalized_task || str || Canonical task name with aliases resolved and options removed
|-
| task_defaults || dict || Dictionary containing pipeline class, model classes, default model info
|-
| task_options || Any || Extra options for parametrized tasks (e.g., source/target languages for translation), or None
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers.pipelines import check_task

# Check a standard task
normalized_task, task_defaults, task_options = check_task("text-classification")
print(f"Task: {normalized_task}")
print(f"Pipeline class: {task_defaults['impl']}")
print(f"Default model: {task_defaults.get('default', {}).get('model')}")

# Check a task alias
normalized_task, task_defaults, task_options = check_task("sentiment-analysis")
print(f"Normalized: {normalized_task}")  # Output: "text-classification"

# Check a parametrized task
normalized_task, task_defaults, task_options = check_task("translation_en_to_fr")
print(f"Task: {normalized_task}")  # Output: "translation"
print(f"Options: {task_options}")  # Output: ("en", "fr")
</syntaxhighlight>

=== Integration Example ===
<syntaxhighlight lang="python">
from transformers.pipelines import check_task
from transformers import pipeline

# Validate task before creating pipeline
task_name = "sentiment-analysis"
try:
    normalized_task, task_defaults, task_options = check_task(task_name)
    print(f"Creating pipeline for task: {normalized_task}")

    # Now create the pipeline with validated task
    pipe = pipeline(normalized_task)
    result = pipe("I love this!")
    print(result)
except KeyError:
    print(f"Invalid task: {task_name}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Task_Resolution]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]

=== Used By ===
* [[used_by::Implementation:huggingface_transformers_pipeline_factory]]
