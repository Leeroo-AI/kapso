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
Concrete tool for pipeline model loading provided by HuggingFace Transformers.

=== Description ===
The `Pipeline.__init__` method is the constructor for all pipeline classes in the Transformers library. It implements the model loading principle by accepting a model instance along with optional preprocessing components, determining the appropriate compute device (CPU, CUDA, MPS, etc.), transferring the model to that device, and configuring task-specific settings like generation parameters. This initialization process creates a unified interface for inference across different model architectures and modalities.

The implementation handles complex device management scenarios including distributed model placement using the Accelerate library's device mapping, automatic detection of available hardware accelerators (CUDA, Apple MPS, Intel XPU/HPU, AMD, etc.), validation that device-mapped models aren't moved after loading, and proper device synchronization in distributed training contexts. For generative models, it creates isolated generation configurations to allow per-pipeline customization without affecting the underlying model, loads assistant models for speculative decoding, and ensures tokenizer padding tokens align with generation settings.

=== Usage ===
This constructor is typically called internally by the `pipeline()` factory function, but can be invoked directly when:
* Creating custom pipeline subclasses with specific initialization logic
* Building inference systems that need fine-grained control over model and device setup
* Implementing pipeline caching or pooling mechanisms
* Testing pipeline behavior with mock models or processors
* Extending transformers with new task types or model architectures

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/base.py:L778-940

=== Signature ===
<syntaxhighlight lang="python">
def __init__(
    self,
    model: "PreTrainedModel",
    tokenizer: PreTrainedTokenizer | None = None,
    feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
    image_processor: BaseImageProcessor | None = None,
    processor: ProcessorMixin | None = None,
    modelcard: ModelCard | None = None,
    task: str = "",
    device: Union[int, "torch.device"] | None = None,
    binary_output: bool = False,
    **kwargs,
):
    """
    Initialize a Pipeline with a model and optional preprocessing components.

    Args:
        model: Pretrained model for inference
        tokenizer: Tokenizer for text preprocessing
        feature_extractor: Feature extractor for audio/legacy vision preprocessing
        image_processor: Image processor for vision preprocessing
        processor: Unified processor for multimodal preprocessing
        modelcard: Model card with metadata
        task: Task identifier (e.g., "text-classification")
        device: Target compute device (int, str, or torch.device)
        binary_output: Whether to return binary data (for certain tasks)
        **kwargs: Task-specific parameters and generation config overrides
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Not typically imported directly; used via pipeline() factory
from transformers import pipeline
from transformers.pipelines.base import Pipeline  # For subclassing
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Instantiated model for inference. Must inherit from PreTrainedModel.
|-
| tokenizer || PreTrainedTokenizer or None || No || Tokenizer for text preprocessing. Required for NLP tasks.
|-
| feature_extractor || PreTrainedFeatureExtractor or None || No || Feature extractor for audio or legacy vision preprocessing.
|-
| image_processor || BaseImageProcessor or None || No || Image processor for vision tasks. Preferred over feature_extractor for images.
|-
| processor || ProcessorMixin or None || No || Unified processor for multimodal tasks. Contains tokenizer and image_processor.
|-
| modelcard || ModelCard or None || No || Model card with metadata about the model's training and capabilities.
|-
| task || str || No || Task identifier (e.g., "text-generation"). Used for task-specific configuration. Defaults to empty string.
|-
| device || int or str or torch.device or None || No || Target device. Can be device index (0, 1), string ("cpu", "cuda:0"), or torch.device object. None auto-detects.
|-
| binary_output || bool || No || Whether pipeline should return binary data. Used by specific tasks like audio generation. Defaults to False.
|-
| batch_size || int || No || Default batch size for batch inference. Can be overridden in __call__.
|-
| num_workers || int || No || Number of workers for data loading in batch inference.
|-
| **kwargs || Any || No || Task-specific parameters (e.g., max_new_tokens, temperature for generation) and sanitized by _sanitize_parameters.
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| pipeline || Pipeline || Initialized pipeline instance with model on target device, processors attached, and task-specific configurations applied. Ready to call with inputs.
|}

== Usage Examples ==

=== Example 1: Basic Pipeline Initialization via Factory ===
<syntaxhighlight lang="python">
from transformers import pipeline, AutoModel, AutoTokenizer

# Standard usage: pipeline() factory calls Pipeline.__init__ internally
nlp = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0  # Use first GPU
)

# The pipeline is now initialized and ready
result = nlp("This movie was amazing!")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
</syntaxhighlight>

=== Example 2: Direct Pipeline Subclass Initialization ===
<syntaxhighlight lang="python">
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines import TextClassificationPipeline
import torch

# Load components separately
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Directly instantiate pipeline with explicit device
pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=torch.device("cuda:0"),
    task="sentiment-analysis",
    top_k=2  # Task-specific parameter
)

result = pipeline("This is great!")
# Output: [{'label': 'POSITIVE', 'score': 0.999}, {'label': 'NEGATIVE', 'score': 0.001}]
</syntaxhighlight>

=== Example 3: Multi-GPU Model with Device Map ===
<syntaxhighlight lang="python">
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load large model with automatic device mapping across GPUs
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",  # Automatically distribute across available GPUs
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Pipeline detects device_map and doesn't move the model
# device parameter would raise an error here
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0 would fail - conflicts with device_map
)

# Model is already distributed; device resolved from hf_device_map
print(f"Pipeline device: {generator.device}")
# Output: Pipeline device: cuda:0 (first device in map)
</syntaxhighlight>

=== Example 4: Vision Pipeline with Image Processor ===
<syntaxhighlight lang="python">
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    ImageClassificationPipeline
)
import torch

# Load vision model and processor
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

# Initialize pipeline with CPU device (for systems without GPU)
classifier = ImageClassificationPipeline(
    model=model,
    image_processor=image_processor,
    device=torch.device("cpu"),
    task="image-classification"
)

from PIL import Image
result = classifier(Image.open("cat.jpg"))
# Output: [{'label': 'Egyptian cat', 'score': 0.89}, ...]
</syntaxhighlight>

=== Example 5: Custom Pipeline with Generation Config ===
<syntaxhighlight lang="python">
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Create text generation pipeline with custom generation parameters
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0,
    max_new_tokens=100,  # Passed to generation_config via **kwargs
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=1.2
)

# Pipeline.__init__ configured generation_config with these parameters
result = generator(
    "In a surprising turn of events,",
    num_return_sequences=2
)
# Output: [
#   {'generated_text': 'In a surprising turn of events, the team won...'},
#   {'generated_text': 'In a surprising turn of events, scientists discovered...'}
# ]

# Each pipeline has isolated generation config
print(f"Max new tokens: {generator.generation_config.max_new_tokens}")
# Output: Max new tokens: 100
</syntaxhighlight>

=== Example 6: Multimodal Pipeline with Processor ===
<syntaxhighlight lang="python">
from transformers import AutoProcessor, AutoModel
from transformers.pipelines import Pipeline
from PIL import Image

# Load multimodal model (CLIP)
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Processor contains both tokenizer and image_processor
# Pipeline.__init__ extracts them automatically
class CLIPPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        return self.processor(**inputs, return_tensors="pt")

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, outputs):
        return outputs

clip_pipeline = CLIPPipeline(
    model=model,
    processor=processor,  # Contains tokenizer + image_processor
    device="cuda:0",
    task="zero-shot-image-classification"
)

# Pipeline.__init__ extracted: tokenizer, image_processor from processor
print(f"Has tokenizer: {clip_pipeline.tokenizer is not None}")
print(f"Has image processor: {clip_pipeline.image_processor is not None}")
# Output: True, True
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pipeline_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Pipeline_Environment]]
