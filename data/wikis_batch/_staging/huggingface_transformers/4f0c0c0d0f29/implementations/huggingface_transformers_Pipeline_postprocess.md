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

Concrete tool for transforming raw model outputs into human-readable results in HuggingFace Transformers pipelines.

=== Description ===

The `postprocess` method is an abstract method defined in the base `Pipeline` class that must be implemented by each task-specific pipeline. It receives raw model outputs (logits, embeddings, generated token IDs) from the `_forward` method and transforms them into user-friendly formats. This includes applying softmax to logits, mapping indices to labels, decoding token IDs to text, formatting bounding boxes, and any other task-specific output formatting. The method returns data structures (lists, dicts) containing human-readable results like predicted labels, confidence scores, generated text, or structured predictions. Each pipeline implementation provides its own postprocessing logic tailored to its output format.

=== Usage ===

Use this method when you need to:
* Convert raw model outputs to human-readable results
* Apply softmax, decode tokens, or format predictions
* Map class indices to meaningful labels
* Implement custom output formatting for new pipeline types

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/base.py (lines 1161-1167)

=== Signature ===
<syntaxhighlight lang="python">
@abstractmethod
def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: dict) -> Any:
    """
    Postprocess will receive the raw outputs of the _forward method, generally tensors, and reformat
    them into something more friendly. Generally it will output a list or a dict of results
    (containing just strings and numbers).

    Args:
        model_outputs (ModelOutput): Raw outputs from the model's forward pass
        **postprocess_parameters: Additional postprocessing parameters

    Returns:
        Any: Human-readable results (typically list[dict] with labels, scores, text, etc.)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Access postprocess through pipeline instance
pipe = pipeline("text-classification")
# pipe.postprocess() is called automatically, but can be accessed directly
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_outputs || ModelOutput || Yes || Raw model outputs containing logits, hidden_states, generated_ids, etc.
|-
| **postprocess_parameters || dict || No || Additional parameters like top_k, return_all_scores for classification tasks
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| results || Any || Human-readable results, typically list[dict] with labels, scores, text, bounding boxes, etc.
|}

== Usage Examples ==

=== Text Classification Postprocessing ===
<syntaxhighlight lang="python">
from transformers import pipeline
import torch

# Create pipeline
classifier = pipeline("text-classification", model="bert-base-uncased")

# Manual pipeline steps
text = "This is a great product!"
preprocessed = classifier.preprocess(text)
model_outputs = classifier._forward(preprocessed)

# Postprocess raw logits to labels and scores
results = classifier.postprocess(model_outputs)

print(results)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# With top_k parameter
results = classifier.postprocess(model_outputs, top_k=2)
print(results)
# [[{'label': 'POSITIVE', 'score': 0.9998}, {'label': 'NEGATIVE', 'score': 0.0002}]]
</syntaxhighlight>

=== Text Generation Postprocessing ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Process input
prompt = "Once upon a time"
preprocessed = generator.preprocess(prompt)
model_outputs = generator._forward(preprocessed, max_new_tokens=30)

# Postprocess token IDs to readable text
results = generator.postprocess(model_outputs)

print(results)
# [{'generated_text': 'Once upon a time, there was a little girl who lived in a small village.'}]
</syntaxhighlight>

=== Named Entity Recognition Postprocessing ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create NER pipeline
ner = pipeline("token-classification", model="dslim/bert-base-NER")

# Process text
text = "My name is John and I live in New York."
preprocessed = ner.preprocess(text)
model_outputs = ner._forward(preprocessed)

# Postprocess to aggregate tokens and create entity spans
results = ner.postprocess(model_outputs, aggregation_strategy="simple")

print(results)
# [
#   {'entity_group': 'PER', 'score': 0.9995, 'word': 'John', 'start': 11, 'end': 15},
#   {'entity_group': 'LOC', 'score': 0.9992, 'word': 'New York', 'start': 29, 'end': 37}
# ]
</syntaxhighlight>

=== Object Detection Postprocessing ===
<syntaxhighlight lang="python">
from transformers import pipeline
from PIL import Image

# Create object detection pipeline
detector = pipeline("object-detection", model="facebook/detr-resnet-50")

# Process image
image = Image.open("street.jpg")
preprocessed = detector.preprocess(image)
model_outputs = detector._forward(preprocessed)

# Postprocess to filter by threshold and format bounding boxes
results = detector.postprocess(model_outputs, threshold=0.9)

print(results)
# [
#   {'score': 0.9982, 'label': 'car', 'box': {'xmin': 123, 'ymin': 45, 'xmax': 456, 'ymax': 234}},
#   {'score': 0.9521, 'label': 'person', 'box': {'xmin': 67, 'ymin': 89, 'xmax': 234, 'ymax': 567}}
# ]
</syntaxhighlight>

=== Understanding the Complete Pipeline Flow ===
<syntaxhighlight lang="python">
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# Full automatic pipeline
result = classifier("I love this!")
print(f"Automatic result: {result}")

# Manual step-by-step to understand each stage
text = "I love this!"

# 1. Preprocess: text -> tensors
preprocessed = classifier.preprocess(text)
print(f"Preprocessed keys: {preprocessed.keys()}")
# {'input_ids', 'attention_mask'}

# 2. Forward: tensors -> raw model outputs
model_outputs = classifier._forward(preprocessed)
print(f"Raw logits: {model_outputs.logits}")
# tensor([[-4.2341,  4.5678]])

# 3. Postprocess: raw outputs -> human-readable
final_result = classifier.postprocess(model_outputs)
print(f"Final result: {final_result}")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# All three steps produce the same result as the automatic call
assert result == final_result
</syntaxhighlight>

=== Custom Postprocessing Parameters ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "This is a movie review about an action film."
candidate_labels = ["politics", "entertainment", "sports"]

preprocessed = classifier.preprocess(text, candidate_labels=candidate_labels)
model_outputs = classifier._forward(preprocessed)

# Postprocess with custom parameters
results = classifier.postprocess(
    model_outputs,
    multi_label=False  # Single label classification
)

print(results)
# {
#   'sequence': 'This is a movie review about an action film.',
#   'labels': ['entertainment', 'sports', 'politics'],
#   'scores': [0.9234, 0.0512, 0.0254]
# }
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pipeline_Postprocessing]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]

=== Part Of ===
* [[part_of::Implementation:huggingface_transformers_pipeline_factory]]

=== Related Methods ===
* [[related::Implementation:huggingface_transformers_Pipeline_preprocess]]
* [[related::Implementation:huggingface_transformers_Pipeline_forward]]
