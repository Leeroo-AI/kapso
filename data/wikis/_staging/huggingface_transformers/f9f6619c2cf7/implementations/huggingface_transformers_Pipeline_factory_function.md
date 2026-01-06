{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Documentation|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for task-based pipeline resolution provided by HuggingFace Transformers.

=== Description ===
The `pipeline()` factory function is the primary entry point for creating inference pipelines in HuggingFace Transformers. It implements task-model resolution by accepting a task identifier (or inferring it from a model) and automatically instantiating the appropriate Pipeline subclass with compatible preprocessing components. The function handles model loading from the Hugging Face Hub or local paths, manages device placement, and configures task-specific parameters.

This implementation resolves over 25 different task types (text classification, image captioning, question answering, etc.) to their corresponding pipeline classes, loads models using AutoModel classes, and automatically selects tokenizers, feature extractors, or image processors based on the task requirements. It supports advanced features like custom pipelines defined in model repos, adapter/PEFT models, device mapping for distributed inference, and dtype configuration.

=== Usage ===
Import and use this function when you need to:
* Quickly set up an inference pipeline without manual component configuration
* Load a pre-trained model for a specific task from the Hugging Face Hub
* Create a pipeline with custom model instances and preprocessing components
* Configure device placement and memory optimization for model inference
* Access task-specific inference capabilities with minimal boilerplate code

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/__init__.py:L516-850

=== Signature ===
<syntaxhighlight lang="python">
def pipeline(
    task: Optional[str] = None,
    model: Optional[Union[str, "PreTrainedModel"]] = None,
    config: Optional[Union[str, PreTrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    processor: Optional[Union[str, ProcessorMixin]] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None,
    dtype: Optional[Union[str, "torch.dtype"]] = "auto",
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs: Any,
) -> Pipeline:
    """
    Utility factory method to build a [`Pipeline`].

    A pipeline consists of:
        - One or more components for pre-processing model inputs
        - A model that generates predictions from the inputs
        - Optional post-processing steps to refine the model's output

    Returns:
        [`Pipeline`]: A suitable pipeline for the task.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import pipeline
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| task || str || No || Task identifier (e.g., "text-classification", "image-to-text"). Required if model is None.
|-
| model || str or PreTrainedModel || No || Model identifier (Hub name/local path) or instance. If None, uses task default.
|-
| config || str or PreTrainedConfig || No || Config identifier or instance. Auto-loaded from model if not provided.
|-
| tokenizer || str or PreTrainedTokenizer || No || Tokenizer identifier or instance. Auto-loaded for NLP tasks.
|-
| feature_extractor || str or PreTrainedFeatureExtractor || No || Feature extractor identifier or instance. Auto-loaded for audio/vision tasks.
|-
| image_processor || str or BaseImageProcessor || No || Image processor identifier or instance. Auto-loaded for vision tasks.
|-
| processor || str or ProcessorMixin || No || Unified processor identifier or instance. Auto-loaded for multimodal tasks.
|-
| device || int or str or torch.device || No || Target device (e.g., "cpu", "cuda:0", 0). Defaults to available accelerator or CPU.
|-
| device_map || str or dict || No || Device mapping strategy for distributed models (e.g., "auto"). Conflicts with device.
|-
| dtype || str or torch.dtype || No || Model precision ("auto", torch.float16, torch.bfloat16). Defaults to "auto".
|-
| trust_remote_code || bool || No || Allow execution of custom code from Hub repos. Defaults to False.
|-
| use_fast || bool || No || Use fast tokenizer implementation if available. Defaults to True.
|-
| revision || str || No || Model version (branch/tag/commit). Defaults to "main".
|-
| token || str or bool || No || Hugging Face Hub authentication token. True uses cached token.
|-
| model_kwargs || dict || No || Additional arguments passed to model's from_pretrained method.
|-
| pipeline_class || type || No || Override automatic pipeline class selection with custom class.
|-
| **kwargs || Any || No || Task-specific pipeline initialization parameters.
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| pipeline || Pipeline || Configured pipeline instance (subclass determined by task) ready for inference with __call__ method.
|}

== Usage Examples ==

=== Example 1: Simple Task-Based Pipeline ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create sentiment analysis pipeline with default model
classifier = pipeline("sentiment-analysis")

# Run inference
result = classifier("I love using transformers!")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
</syntaxhighlight>

=== Example 2: Pipeline with Specific Model ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create question answering pipeline with specific model
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad",
    tokenizer="google-bert/bert-base-cased"
)

# Run inference
result = qa_pipeline(
    question="What is the capital of France?",
    context="Paris is the capital and largest city of France."
)
# Output: {'answer': 'Paris', 'score': 0.98, 'start': 0, 'end': 5}
</syntaxhighlight>

=== Example 3: Pipeline with Custom Model Instances ===
<syntaxhighlight lang="python">
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Load model and tokenizer separately
model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Create NER pipeline with custom instances
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Run inference
result = ner_pipeline("Hugging Face is based in New York City.")
# Output: [{'entity': 'B-ORG', 'word': 'Hugging', ...}, ...]
</syntaxhighlight>

=== Example 4: Pipeline with Device and Dtype Configuration ===
<syntaxhighlight lang="python">
from transformers import pipeline
import torch

# Create image classification pipeline with GPU and mixed precision
classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    device="cuda:0",
    dtype=torch.float16
)

# Run inference on image
result = classifier("path/to/image.jpg")
# Output: [{'label': 'Egyptian cat', 'score': 0.89}, ...]
</syntaxhighlight>

=== Example 5: Pipeline with Device Map for Large Models ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create text generation pipeline with automatic device mapping
generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    device_map="auto",  # Automatically splits model across available GPUs
    dtype="auto"
)

# Run inference
result = generator(
    "Once upon a time",
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)
# Output: [{'generated_text': 'Once upon a time, in a faraway land...'}]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Task_Model_Resolution]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Pipeline_Environment]]
