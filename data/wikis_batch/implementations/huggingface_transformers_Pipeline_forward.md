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

Concrete tool for executing model forward passes in HuggingFace Transformers pipelines.

=== Description ===

The `_forward` method is an abstract method defined in the base `Pipeline` class that must be implemented by each task-specific pipeline. It receives preprocessed tensors from the `preprocess` method and executes the model's forward pass to generate predictions. This method is device-agnostic and focuses on the core inference computation, running the neural network to produce raw outputs (logits, hidden states, embeddings, etc.). The method is wrapped by a public `forward` method that handles device placement, gradient disabling, and other infrastructure concerns. Each pipeline implementation provides task-specific forward logic, including any model.generate() calls for generative tasks.

=== Usage ===

Use this method when you need to:
* Execute model inference on preprocessed tensors
* Generate raw model outputs (logits, embeddings, etc.)
* Implement custom forward logic for new pipeline types
* Access intermediate model outputs for analysis

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/base.py (lines 1148-1158)

=== Signature ===
<syntaxhighlight lang="python">
@abstractmethod
def _forward(self, input_tensors: dict[str, GenericTensor], **forward_parameters: dict) -> ModelOutput:
    """
    _forward will receive the prepared dictionary from preprocess and run it on the model.
    This method might involve the GPU or the CPU and should be agnostic to it. Isolating this
    function is the reason for preprocess and postprocess to exist, so that the hot path,
    this method, generally can run as fast as possible.

    It is not meant to be called directly, forward is preferred. It is basically the same but
    contains additional code surrounding _forward making sure tensors and models are on the
    same device, disabling the training part of the code (leading to faster inference).

    Args:
        input_tensors (dict[str, GenericTensor]): Preprocessed tensors from preprocess()
        **forward_parameters: Additional forward pass parameters

    Returns:
        ModelOutput: Model output containing logits, hidden_states, attentions, etc.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Access _forward through pipeline instance
pipe = pipeline("text-classification")
# pipe._forward() is called automatically, but can be accessed directly
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_tensors || dict[str, GenericTensor] || Yes || Dictionary of preprocessed tensors (input_ids, attention_mask, pixel_values, etc.)
|-
| **forward_parameters || dict || No || Additional forward parameters like num_beams, temperature for generation tasks
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model_output || ModelOutput || Model output object containing logits, hidden_states, attentions, and task-specific outputs
|}

== Usage Examples ==

=== Text Classification Forward Pass ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create pipeline
classifier = pipeline("text-classification", model="bert-base-uncased")

# Manual pipeline steps
text = "This is a great product!"
preprocessed = classifier.preprocess(text)

# Execute forward pass
model_outputs = classifier._forward(preprocessed)

print(type(model_outputs))  # transformers.modeling_outputs.SequenceClassifierOutput
print(model_outputs.logits.shape)  # torch.Size([1, num_labels])
print(model_outputs.logits)  # Raw logits before softmax
</syntaxhighlight>

=== Text Generation Forward Pass ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Preprocess input
prompt = "Once upon a time"
preprocessed = generator.preprocess(prompt)

# Forward pass with generation parameters
model_outputs = generator._forward(
    preprocessed,
    max_new_tokens=50,
    temperature=0.8,
    do_sample=True
)

print(type(model_outputs))  # Contains generated token IDs
print(model_outputs)  # Model-specific output format
</syntaxhighlight>

=== Image Classification Forward Pass ===
<syntaxhighlight lang="python">
from transformers import pipeline
from PIL import Image

# Create image classification pipeline
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Load and preprocess image
image = Image.open("cat.jpg")
preprocessed = classifier.preprocess(image)

# Execute forward pass
model_outputs = classifier._forward(preprocessed)

print(model_outputs.logits.shape)  # torch.Size([1, num_classes])
print(model_outputs.logits.argmax(dim=-1))  # Predicted class index
</syntaxhighlight>

=== Understanding Device Management ===
<syntaxhighlight lang="python">
from transformers import pipeline
import torch

# Create pipeline on GPU
classifier = pipeline("text-classification", device=0)

# Preprocess (returns CPU tensors)
text = "Great product!"
preprocessed = classifier.preprocess(text)
print(preprocessed['input_ids'].device)  # cpu

# _forward internally moves tensors to correct device
model_outputs = classifier._forward(preprocessed)
print(model_outputs.logits.device)  # cuda:0

# The public forward() method handles this automatically:
# - Moves input tensors to model device
# - Disables gradients with torch.no_grad()
# - Executes _forward
# - Returns outputs on model device
</syntaxhighlight>

=== Accessing Forward with Custom Parameters ===
<syntaxhighlight lang="python">
from transformers import pipeline

# Create summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Preprocess long text
text = "Very long article text..." * 100
preprocessed = summarizer.preprocess(text, truncation=True, max_length=1024)

# Forward with custom generation parameters
model_outputs = summarizer._forward(
    preprocessed,
    max_length=150,
    min_length=50,
    num_beams=4,
    early_stopping=True
)

print(model_outputs)  # Generated summary token IDs
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pipeline_Model_Forward]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]

=== Part Of ===
* [[part_of::Implementation:huggingface_transformers_pipeline_factory]]

=== Related Methods ===
* [[related::Implementation:huggingface_transformers_Pipeline_preprocess]]
* [[related::Implementation:huggingface_transformers_Pipeline_postprocess]]
