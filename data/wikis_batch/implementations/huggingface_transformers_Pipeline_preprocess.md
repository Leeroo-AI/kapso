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

Concrete tool for converting raw inputs into model-ready tensors in HuggingFace Transformers pipelines.

=== Description ===

The `preprocess` method is an abstract method defined in the base `Pipeline` class that must be implemented by each task-specific pipeline. It transforms raw user inputs (text strings, images, audio files, etc.) into the tensor format required by the model. This includes tokenization for text, image processing for vision tasks, audio feature extraction, and any task-specific transformations. The method returns a dictionary containing tensors that will be passed to the model's forward pass. Each pipeline implementation (TextClassificationPipeline, ImageClassificationPipeline, etc.) provides its own preprocessing logic tailored to its input modality and model requirements.

=== Usage ===

Use this method when you need to:
* Transform raw user inputs into model-compatible tensors
* Apply tokenization, normalization, or feature extraction
* Prepare inputs for batched or single inference
* Implement custom preprocessing logic for new pipeline types

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/base.py (lines 1140-1145)

=== Signature ===
<syntaxhighlight lang="python">
@abstractmethod
def preprocess(self, input_: Any, **preprocess_parameters: dict) -> dict[str, GenericTensor]:
    """
    Preprocess will take the input_ of a specific pipeline and return a dictionary of everything
    necessary for _forward to run properly. It should contain at least one tensor, but might have
    arbitrary other items.

    Args:
        input_ (Any): Raw input data (text, image, audio, etc.)
        **preprocess_parameters: Additional preprocessing parameters

    Returns:
        dict[str, GenericTensor]: Dictionary containing model-ready tensors (input_ids, attention_mask, etc.)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Access preprocess through pipeline instance
pipe = pipeline("text-classification")
# pipe.preprocess() is called automatically, but can be accessed directly
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_ || Any || Yes || Raw input data specific to the pipeline (str for text, PIL.Image for images, etc.)
|-
| **preprocess_parameters || dict || No || Additional parameters like padding, truncation, max_length for text tasks
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| preprocessed_data || dict[str, GenericTensor] || Dictionary with model-ready tensors (input_ids, attention_mask, pixel_values, etc.)
|}

== Usage Examples ==

=== Text Classification Preprocessing ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create pipeline
classifier = pipeline("text-classification", model="bert-base-uncased")

# Access preprocess method directly (normally called internally)
text = "This is a great product!"
preprocessed = classifier.preprocess(text)

print(preprocessed.keys())  # dict_keys(['input_ids', 'attention_mask'])
print(preprocessed['input_ids'].shape)  # torch.Size([1, sequence_length])
print(preprocessed['attention_mask'].shape)  # torch.Size([1, sequence_length])
</syntaxhighlight>

=== Image Classification Preprocessing ===
<syntaxhighlight lang="python">
from transformers import pipeline
from PIL import Image

# Create image classification pipeline
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Load image
image = Image.open("cat.jpg")

# Preprocess image
preprocessed = classifier.preprocess(image)

print(preprocessed.keys())  # dict_keys(['pixel_values'])
print(preprocessed['pixel_values'].shape)  # torch.Size([1, 3, 224, 224])
</syntaxhighlight>

=== Custom Preprocessing Parameters ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Preprocess with custom parameters
text = "Long article text here..." * 100
preprocessed = summarizer.preprocess(
    text,
    truncation=True,
    max_length=1024,
    padding="max_length"
)

print(preprocessed['input_ids'].shape)  # torch.Size([1, 1024])
</syntaxhighlight>

=== Understanding the Pipeline Flow ===
<syntaxhighlight lang="python">
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# When you call the pipeline, it internally does:
# 1. preprocess: text -> tensors
# 2. _forward: tensors -> model outputs
# 3. postprocess: model outputs -> human-readable results

# Full pipeline call (automatic)
result = classifier("I love this!")

# Manual step-by-step (for debugging/understanding)
preprocessed = classifier.preprocess("I love this!")
print(f"Preprocessed: {preprocessed}")

model_outputs = classifier._forward(preprocessed)
print(f"Model outputs: {model_outputs}")

final_result = classifier.postprocess(model_outputs)
print(f"Final result: {final_result}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pipeline_Preprocessing]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]

=== Part Of ===
* [[part_of::Implementation:huggingface_transformers_pipeline_factory]]

=== Related Methods ===
* [[related::Implementation:huggingface_transformers_Pipeline_forward]]
* [[related::Implementation:huggingface_transformers_Pipeline_postprocess]]
