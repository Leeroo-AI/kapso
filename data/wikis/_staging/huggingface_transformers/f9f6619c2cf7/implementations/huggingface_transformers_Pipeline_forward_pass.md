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
Concrete tool for pipeline forward pass provided by HuggingFace Transformers.

=== Description ===
The `_forward` method is an abstract method in the `Pipeline` base class that subclasses must implement to execute model inference. It works in conjunction with the concrete `forward` method which handles cross-cutting concerns like device management and gradient context. The base `forward` method sets up the inference environment (torch.no_grad context, device placement), calls the subclass's `_forward` implementation, and manages tensor transfers. This separation allows task-specific pipelines to focus on model invocation logic while the framework handles optimization and safety concerns.

The `_forward` implementation receives preprocessed tensors from the preprocess step and must call the model with the appropriate signature for its task type. For standard models, this means calling `model(**model_inputs)`, but for generative tasks it may involve `model.generate()`, for retrieval tasks it may extract embeddings, and for complex multi-stage models it may involve multiple forward passes. The method returns raw model outputs (logits, hidden states, generated tokens) which are then passed to postprocessing.

=== Usage ===
Implement this method when:
* Creating custom pipeline subclasses for new task types
* Adapting pipelines to use non-standard model invocation patterns
* Implementing multi-stage inference with multiple model calls
* Adding support for generation with custom decoding strategies
* Building pipelines that extract specific model internals (embeddings, attention weights)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/base.py:L1147-1158, L1172-1179

=== Signature ===
<syntaxhighlight lang="python">
# Abstract method that subclasses implement
@abstractmethod
def _forward(self, input_tensors: dict[str, GenericTensor], **forward_parameters: dict) -> ModelOutput:
    """
    _forward will receive the prepared dictionary from `preprocess` and run it on the model.
    This method might involve the GPU or the CPU and should be agnostic to it. Isolating this
    function is the reason for `preprocess` and `postprocess` to exist, so that the hot path,
    this method generally can run as fast as possible.

    It is not meant to be called directly, `forward` is preferred. It is basically the same
    but contains additional code surrounding `_forward` making sure tensors and models are on
    the same device, disabling the training part of the code (leading to faster inference).

    Args:
        input_tensors: Dictionary of tensors from preprocess step
        **forward_parameters: Task-specific forward pass parameters

    Returns:
        ModelOutput: Raw model outputs (logits, hidden states, generated tokens, etc.)

    Raises:
        NotImplementedError: This is an abstract method that must be overridden.
    """
    raise NotImplementedError("_forward not implemented")

# Concrete wrapper method in base Pipeline class
def forward(self, model_inputs, **forward_params):
    """
    Wrapper that manages device placement and inference context.
    Calls _forward with properly prepared inputs.
    """
    with self.device_placement():
        inference_context = self.get_inference_context()
        with inference_context():
            model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
            model_outputs = self._forward(model_inputs, **forward_params)
            model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
    return model_outputs
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# For implementing custom pipelines
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import ModelOutput

class CustomPipeline(Pipeline):
    def _forward(self, input_tensors: dict, **forward_parameters: dict) -> ModelOutput:
        # Custom implementation here
        pass
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_tensors || dict[str, Tensor] || Yes || Preprocessed tensors from preprocess method. Keys vary by task (input_ids, pixel_values, etc.). Tensors already on model device.
|-
| output_hidden_states || bool || No || Return intermediate layer outputs. Defaults to False for efficiency.
|-
| output_attentions || bool || No || Return attention weights. Defaults to False for efficiency.
|-
| return_dict || bool || No || Return ModelOutput object vs tuple. Defaults to True in modern transformers.
|-
| **forward_parameters || dict || No || Additional model-specific parameters (e.g., generation parameters for generative models, decoder inputs for encoder-decoder).
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model_outputs || ModelOutput or Tensor || Raw model outputs. For classification: ModelOutput with logits. For generation: Tensor of token IDs. For encoder-decoder: Seq2SeqModelOutput. Structure varies by task.
|}

== Usage Examples ==

=== Example 1: Text Classification Forward Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import SequenceClassifierOutput

class TextClassificationPipeline(Pipeline):
    def _forward(self, input_tensors: dict, **forward_parameters: dict) -> SequenceClassifierOutput:
        """
        Execute model forward pass for classification.

        Args:
            input_tensors: {"input_ids": Tensor, "attention_mask": Tensor}
            forward_parameters: output_hidden_states, output_attentions, etc.
        """
        # Simple forward pass
        outputs = self.model(**input_tensors, **forward_parameters)

        # outputs.logits: [batch, num_labels]
        # outputs.hidden_states: tuple of [batch, seq_len, hidden_dim] (if requested)
        # outputs.attentions: tuple of [batch, num_heads, seq_len, seq_len] (if requested)

        return outputs

# Usage (forward called internally by pipeline)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)
result = pipeline("I love this!")
# Internally: preprocess → forward (calls _forward) → postprocess
</syntaxhighlight>

=== Example 2: Text Generation Forward Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.generation.utils import GenerateOutput

class TextGenerationPipeline(Pipeline):
    def _forward(self, input_tensors: dict, **forward_parameters: dict):
        """
        Execute generation instead of standard forward.

        Args:
            input_tensors: {"input_ids": Tensor, "attention_mask": Tensor}
            forward_parameters: max_new_tokens, temperature, top_p, etc.
        """
        # Extract prompt information for postprocessing
        prompt_ids = input_tensors["input_ids"]

        # Use generate instead of forward
        generated_ids = self.model.generate(
            **input_tensors,
            **forward_parameters
        )

        # Return dict with both prompt and generated tokens
        return {
            "generated_sequence": generated_ids,
            "input_ids": prompt_ids
        }

# Usage
pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device="cuda:0"
)

result = pipeline(
    "Once upon a time",
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8
)
# _forward calls model.generate with these parameters
</syntaxhighlight>

=== Example 3: Image Classification Forward Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import ImageClassifierOutput

class ImageClassificationPipeline(Pipeline):
    def _forward(self, input_tensors: dict, **forward_parameters: dict) -> ImageClassifierOutput:
        """
        Execute model forward pass for image classification.

        Args:
            input_tensors: {"pixel_values": Tensor}  # [batch, channels, height, width]
            forward_parameters: output_hidden_states, etc.
        """
        outputs = self.model(**input_tensors, **forward_parameters)

        # outputs.logits: [batch, num_classes]
        # outputs.hidden_states: tuple of feature maps (if requested)

        return outputs

# Usage
pipeline = ImageClassificationPipeline(
    model=model,
    image_processor=image_processor,
    device=0
)

from PIL import Image
result = pipeline(Image.open("cat.jpg"))
# _forward receives {"pixel_values": tensor([[[[...]]]])} from preprocess
</syntaxhighlight>

=== Example 4: Question Answering Forward with Custom Parameters ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import QuestionAnsweringModelOutput

class QuestionAnsweringPipeline(Pipeline):
    def _forward(self, input_tensors: dict, **forward_parameters: dict) -> QuestionAnsweringModelOutput:
        """
        Execute QA model forward pass.

        Args:
            input_tensors: {"input_ids": Tensor, "attention_mask": Tensor, "token_type_ids": Tensor}
            forward_parameters: output_attentions, etc.
        """
        # Standard forward pass
        outputs = self.model(**input_tensors, **forward_parameters)

        # outputs.start_logits: [batch, seq_len] - probabilities for answer start
        # outputs.end_logits: [batch, seq_len] - probabilities for answer end

        return outputs

# Usage
pipeline = QuestionAnsweringPipeline(
    model=model,
    tokenizer=tokenizer,
    device="cuda:0"
)

result = pipeline(
    question="What is the capital?",
    context="Paris is the capital of France.",
    output_attentions=True  # Forward parameter
)
# _forward receives preprocessed tensors and output_attentions param
</syntaxhighlight>

=== Example 5: Feature Extraction Forward (No Task Head) ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import BaseModelOutput

class FeatureExtractionPipeline(Pipeline):
    def _forward(self, input_tensors: dict, **forward_parameters: dict) -> BaseModelOutput:
        """
        Extract embeddings/features without task-specific head.

        Args:
            input_tensors: {"input_ids": Tensor, "attention_mask": Tensor}
            forward_parameters: output_hidden_states, etc.
        """
        # Get base model outputs (no classification head)
        outputs = self.model(**input_tensors, **forward_parameters)

        # outputs.last_hidden_state: [batch, seq_len, hidden_dim]
        # For sentence embeddings, typically use [CLS] token or mean pooling

        return outputs

# Usage
pipeline = FeatureExtractionPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0
)

result = pipeline("This is a sentence to encode.")
# _forward returns hidden states, postprocess extracts embeddings
</syntaxhighlight>

=== Example 6: Encoder-Decoder Forward (Translation) ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline

class TranslationPipeline(Pipeline):
    def _forward(self, input_tensors: dict, **forward_parameters: dict):
        """
        Execute translation using encoder-decoder architecture.

        Args:
            input_tensors: {
                "input_ids": Tensor,  # Source language
                "attention_mask": Tensor
            }
            forward_parameters: Generation parameters
        """
        # Use generate for translation (autoregressive decoding)
        generated_tokens = self.model.generate(
            **input_tensors,
            **forward_parameters
        )

        return {"generated_tokens": generated_tokens}

# Usage
pipeline = TranslationPipeline(
    model=model,  # e.g., MarianMT
    tokenizer=tokenizer,
    device="cuda:0"
)

result = pipeline(
    "Hello, how are you?",
    max_length=50,
    num_beams=4  # Beam search for better translations
)
# _forward calls model.generate with beam search
</syntaxhighlight>

=== Example 7: Custom Multi-Stage Forward ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
import torch

class CustomMultiStagePipeline(Pipeline):
    def _forward(self, input_tensors: dict, **forward_parameters: dict) -> dict:
        """
        Custom pipeline with multiple model calls.

        Args:
            input_tensors: {"input_ids": Tensor, "attention_mask": Tensor}
            forward_parameters: stage_specific parameters
        """
        # Stage 1: Get encoder outputs
        encoder_outputs = self.model.encoder(
            input_ids=input_tensors["input_ids"],
            attention_mask=input_tensors["attention_mask"]
        )

        # Stage 2: Custom processing on hidden states
        hidden_states = encoder_outputs.last_hidden_state
        # Apply custom transformation
        processed_hidden = torch.mean(hidden_states, dim=1)  # [batch, hidden_dim]

        # Stage 3: Pass through classification head
        logits = self.model.classifier(processed_hidden)

        return {
            "logits": logits,
            "hidden_states": encoder_outputs.hidden_states,
            "pooled_output": processed_hidden
        }

# Usage
pipeline = CustomMultiStagePipeline(
    model=model,
    tokenizer=tokenizer,
    device=0
)

result = pipeline("Custom multi-stage processing")
# _forward executes multiple model operations in sequence
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pipeline_Forward]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Pipeline_Environment]]
