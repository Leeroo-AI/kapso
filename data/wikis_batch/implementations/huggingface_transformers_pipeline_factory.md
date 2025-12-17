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

Concrete tool for creating configured pipeline instances provided by the HuggingFace Transformers library.

=== Description ===

The `pipeline` function is the main factory method for creating inference pipelines in Transformers. It orchestrates the entire pipeline instantiation process: validating the task, loading the model and tokenizer/processor, configuring device placement, and returning a fully initialized pipeline object. This function provides a unified interface for creating pipelines across 30+ different tasks (text classification, translation, image generation, etc.). It handles both simple use cases (just specify a task) and advanced scenarios (custom models, specific devices, dtype configuration, device_map for large models).

=== Usage ===

Use this function when you need to:
* Create an end-to-end inference pipeline for any supported task
* Load a pipeline with automatic model and tokenizer detection
* Configure device placement (CPU, GPU, multi-GPU) for inference
* Set up pipelines with custom models or task-specific parameters
* Batch process inputs with automatic batching support

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/__init__.py (lines 516-900)

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
    Utility factory method to build a Pipeline.

    A pipeline consists of:
    - One or more components for pre-processing model inputs (tokenizer, image_processor, etc.)
    - A model that generates predictions from the inputs
    - Optional post-processing steps to refine the model's output

    Args:
        task (str): The task defining which pipeline will be returned
        model (str or PreTrainedModel, optional): Model identifier or instance
        tokenizer, feature_extractor, image_processor, processor: Preprocessing components
        device (int, str, or torch.device, optional): Device for inference
        device_map (str or dict, optional): Device mapping for large models
        dtype (str or torch.dtype): Data type for model weights
        batch_size (int): Batch size for inference
        **kwargs: Additional task-specific parameters

    Returns:
        Pipeline: Fully configured pipeline instance for the specified task
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
| task || str || No || Task identifier (e.g., "text-classification"). Can be inferred from model if not provided
|-
| model || str or PreTrainedModel || No || Model identifier or instance. Uses task default if not provided
|-
| tokenizer || str or PreTrainedTokenizer || No || Tokenizer identifier or instance. Auto-loaded if not provided
|-
| device || int, str, or torch.device || No || Device for inference (-1 for CPU, 0+ for GPU, "cuda", etc.)
|-
| device_map || str or dict || No || Device mapping strategy ("auto", "balanced", or custom dict)
|-
| dtype || str or torch.dtype || No || Model dtype ("auto", torch.float16, torch.bfloat16, etc.)
|-
| batch_size || int || No || Batch size for processing multiple inputs
|-
| **kwargs || Any || No || Additional task-specific parameters
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| pipeline || Pipeline || Configured pipeline instance ready for inference with __call__ method
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create a sentiment analysis pipeline with default model
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Create a translation pipeline
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
print(result)  # [{'translation_text': 'Bonjour, comment allez-vous?'}]
</syntaxhighlight>

=== Advanced Usage with Custom Model and Device ===
<syntaxhighlight lang="python">
from transformers import pipeline
import torch

# Create pipeline with specific model and GPU
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0,  # Use GPU 0
    batch_size=8
)

# Batch inference
texts = ["Great product!", "Terrible service.", "It's okay."]
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text}: {result['label']} ({result['score']:.4f})")
</syntaxhighlight>

=== Large Model with Device Map ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Load large model with automatic device mapping
generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    device_map="auto",
    dtype="float16"
)

prompt = "Once upon a time"
output = generator(prompt, max_new_tokens=50)
print(output[0]['generated_text'])
</syntaxhighlight>

=== Multi-Modal Pipeline (Image to Text) ===
<syntaxhighlight lang="python">
from transformers import pipeline
from PIL import Image

# Create image captioning pipeline
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Load image and generate caption
image = Image.open("photo.jpg")
result = captioner(image)
print(result)  # [{'generated_text': 'a dog playing in the park'}]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pipeline_Instantiation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]

=== Uses ===
* [[uses::Implementation:huggingface_transformers_check_task]]
* [[uses::Implementation:huggingface_transformers_pipeline_load_model]]

=== Creates ===
* [[creates::Implementation:huggingface_transformers_Pipeline_preprocess]]
* [[creates::Implementation:huggingface_transformers_Pipeline_forward]]
* [[creates::Implementation:huggingface_transformers_Pipeline_postprocess]]
