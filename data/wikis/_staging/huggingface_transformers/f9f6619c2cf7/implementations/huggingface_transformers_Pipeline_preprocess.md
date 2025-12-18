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
Concrete tool for pipeline preprocessing provided by HuggingFace Transformers.

=== Description ===
The `preprocess` method is an abstract method defined in the `Pipeline` base class that must be implemented by all pipeline subclasses. It serves as the interface contract for converting raw task-specific inputs into model-ready tensors. Each pipeline subclass (TextClassificationPipeline, ImageClassificationPipeline, QuestionAnsweringPipeline, etc.) provides its own implementation tailored to its task's input format and model requirements.

This pattern-based implementation allows the Pipeline framework to handle common concerns (batching, device management, error handling) while delegating task-specific logic to subclasses. The preprocessing method receives raw inputs and parameters, applies the appropriate processor (tokenizer, image processor, feature extractor), handles special cases like padding and truncation, and returns a standardized dictionary of tensors. The base Pipeline class calls this method during the inference workflow, ensuring consistent preprocessing across all task types.

=== Usage ===
Implement this method when:
* Creating custom pipeline subclasses for new task types
* Adapting existing pipelines to handle non-standard input formats
* Optimizing preprocessing for specific model architectures or use cases
* Building specialized preprocessing logic for domain-specific applications
* Extending transformers with custom multimodal or structured data pipelines

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/base.py:L1139-1145

=== Signature ===
<syntaxhighlight lang="python">
@abstractmethod
def preprocess(self, input_: Any, **preprocess_parameters: dict) -> dict[str, GenericTensor]:
    """
    Preprocess will take the `input_` of a specific pipeline and return a dictionary
    of everything necessary for `_forward` to run properly. It should contain at least
    one tensor, but might have arbitrary other items.

    Args:
        input_: Raw input in task-appropriate format (str, PIL.Image, dict, etc.)
        **preprocess_parameters: Task-specific preprocessing configuration parameters

    Returns:
        Dictionary mapping string keys to tensors, ready for model forward pass.
        Must include batch dimension even for single inputs.

    Raises:
        NotImplementedError: This is an abstract method that must be overridden.

    Notes:
        - Subclasses must implement this method with task-specific logic
        - Output tensors should remain on CPU (device transfer handled by forward)
        - Dictionary keys must match model's forward signature parameter names
        - Batch dimension must be present even for single inputs
    """
    raise NotImplementedError("preprocess not implemented")
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# For implementing custom pipelines
from transformers.pipelines.base import Pipeline
from typing import Any

class CustomPipeline(Pipeline):
    def preprocess(self, input_: Any, **preprocess_parameters: dict) -> dict[str, Any]:
        # Custom implementation here
        pass
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_ || Any || Yes || Raw input in task-appropriate format. Type varies by task: str for text, PIL.Image for images, dict for complex inputs, list for batches.
|-
| padding || bool or str || No || Padding strategy: True (pad to longest in batch), "max_length" (pad to model max), False (no padding).
|-
| truncation || bool or str || No || Truncation strategy: True (truncate to max_length), "longest_first" (truncate longest sequence first), False (no truncation).
|-
| max_length || int || No || Maximum sequence length for padding/truncation. Defaults to model's max_position_embeddings.
|-
| return_tensors || str || No || Tensor framework: "pt" (PyTorch), "np" (NumPy). Usually "pt" for PyTorch models.
|-
| **preprocess_parameters || dict || No || Additional task-specific preprocessing parameters (e.g., do_resize, image_size, sampling_rate).
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| preprocessed || dict[str, Tensor] || Dictionary with tensor inputs for model. Keys vary by task but commonly include input_ids, attention_mask, pixel_values, etc. All tensors include batch dimension.
|}

== Usage Examples ==

=== Example 1: Text Classification Preprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from typing import Any

class TextClassificationPipeline(Pipeline):
    def preprocess(self, input_: Any, **preprocess_parameters: dict) -> dict[str, Any]:
        """
        Preprocess text input for classification.

        Args:
            input_: str or list of str
            preprocess_parameters: padding, truncation, max_length, etc.
        """
        # Extract parameters with defaults
        padding = preprocess_parameters.get("padding", True)
        truncation = preprocess_parameters.get("truncation", True)
        max_length = preprocess_parameters.get("max_length", None)

        # Tokenize input
        model_inputs = self.tokenizer(
            input_,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt"
        )

        return model_inputs

# Usage
pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0
)

# Preprocessing happens internally during __call__
result = pipeline("This is a test sentence")
# preprocess converts to: {"input_ids": tensor([[...]]), "attention_mask": tensor([[...]])}
</syntaxhighlight>

=== Example 2: Image Classification Preprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from PIL import Image
from typing import Any, Union

class ImageClassificationPipeline(Pipeline):
    def preprocess(self, input_: Union[str, Image.Image], **preprocess_parameters: dict) -> dict[str, Any]:
        """
        Preprocess image input for classification.

        Args:
            input_: PIL.Image, image path, or numpy array
            preprocess_parameters: image processing configuration
        """
        # Load image if path provided
        if isinstance(input_, str):
            image = Image.open(input_)
        elif isinstance(input_, Image.Image):
            image = input_
        else:
            # Handle numpy arrays, tensors, etc.
            image = Image.fromarray(input_)

        # Process image using image processor
        model_inputs = self.image_processor(
            images=image,
            return_tensors="pt"
        )

        return model_inputs

# Usage
pipeline = ImageClassificationPipeline(
    model=model,
    image_processor=image_processor,
    device="cuda:0"
)

result = pipeline("cat.jpg")
# preprocess converts to: {"pixel_values": tensor([[[[...]]]])}
</syntaxhighlight>

=== Example 3: Question Answering Preprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from typing import Any

class QuestionAnsweringPipeline(Pipeline):
    def preprocess(self, input_: dict, **preprocess_parameters: dict) -> dict[str, Any]:
        """
        Preprocess question-context pairs for QA.

        Args:
            input_: dict with "question" and "context" keys
            preprocess_parameters: tokenization parameters
        """
        # Extract question and context
        question = input_["question"]
        context = input_["context"]

        # Tokenize as question-context pair
        # token_type_ids distinguish question from context
        model_inputs = self.tokenizer(
            question,
            context,
            padding=preprocess_parameters.get("padding", True),
            truncation=preprocess_parameters.get("truncation", "only_second"),
            max_length=preprocess_parameters.get("max_length", 384),
            return_tensors="pt",
            return_token_type_ids=True
        )

        return model_inputs

# Usage
pipeline = QuestionAnsweringPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0
)

result = pipeline({
    "question": "What is the capital?",
    "context": "Paris is the capital of France."
})
# preprocess converts to: {
#     "input_ids": tensor([[...]]),
#     "attention_mask": tensor([[...]]),
#     "token_type_ids": tensor([[0,0,0,1,1,1,1,1,1]])  # 0=question, 1=context
# }
</syntaxhighlight>

=== Example 4: Token Classification (NER) with Offset Mapping ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from typing import Any

class TokenClassificationPipeline(Pipeline):
    def preprocess(self, input_: str, **preprocess_parameters: dict) -> dict[str, Any]:
        """
        Preprocess text for token classification (NER, POS tagging).

        Args:
            input_: str (single sentence)
            preprocess_parameters: tokenization parameters
        """
        # Tokenize with offset mapping for token→character alignment
        model_inputs = self.tokenizer(
            input_,
            padding=preprocess_parameters.get("padding", True),
            truncation=preprocess_parameters.get("truncation", True),
            max_length=preprocess_parameters.get("max_length", None),
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_offsets_mapping=True  # Critical for NER
        )

        # Store offset mapping for postprocessing (not passed to model)
        model_inputs["offset_mapping"] = model_inputs.pop("offset_mapping")
        model_inputs["special_tokens_mask"] = model_inputs.pop("special_tokens_mask")

        # Store original sentence for postprocessing
        model_inputs["sentence"] = input_

        return model_inputs

# Usage
pipeline = TokenClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0
)

result = pipeline("John Smith works at Microsoft in Seattle.")
# preprocess converts to: {
#     "input_ids": tensor([[...]]),
#     "attention_mask": tensor([[...]]),
#     "offset_mapping": [[(0,0), (0,4), (5,10), ...]],  # Character spans
#     "special_tokens_mask": [[1,0,0,0,...]],
#     "sentence": "John Smith works at Microsoft in Seattle."
# }
</syntaxhighlight>

=== Example 5: Multimodal Visual Question Answering ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from PIL import Image
from typing import Any

class VisualQuestionAnsweringPipeline(Pipeline):
    def preprocess(self, input_: dict, **preprocess_parameters: dict) -> dict[str, Any]:
        """
        Preprocess image-question pairs for VQA.

        Args:
            input_: dict with "image" and "question" keys
            preprocess_parameters: processing parameters
        """
        # Extract inputs
        image = input_["image"]
        question = input_["question"]

        # Process image
        if isinstance(image, str):
            image = Image.open(image)

        # Use processor (combines tokenizer + image processor)
        model_inputs = self.processor(
            images=image,
            text=question,
            padding=preprocess_parameters.get("padding", True),
            truncation=preprocess_parameters.get("truncation", True),
            return_tensors="pt"
        )

        return model_inputs

# Usage
pipeline = VisualQuestionAnsweringPipeline(
    model=model,
    processor=processor,
    device="cuda:0"
)

result = pipeline({
    "image": "photo.jpg",
    "question": "What color is the car?"
})
# preprocess converts to: {
#     "pixel_values": tensor([[[[...]]]]),
#     "input_ids": tensor([[...]]),
#     "attention_mask": tensor([[...]])
# }
</syntaxhighlight>

=== Example 6: Custom Preprocessing with Data Augmentation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from PIL import Image, ImageEnhance
from typing import Any
import random

class AugmentedImageClassificationPipeline(Pipeline):
    def preprocess(self, input_: Any, **preprocess_parameters: dict) -> dict[str, Any]:
        """
        Custom preprocessing with optional data augmentation.

        Args:
            input_: image input
            preprocess_parameters: augment (bool), brightness_range, etc.
        """
        # Load image
        if isinstance(input_, str):
            image = Image.open(input_)
        else:
            image = input_

        # Apply augmentation if requested
        if preprocess_parameters.get("augment", False):
            # Random brightness adjustment
            brightness_factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)

            # Random rotation
            angle = random.uniform(-10, 10)
            image = image.rotate(angle)

        # Standard preprocessing
        model_inputs = self.image_processor(
            images=image,
            return_tensors="pt"
        )

        return model_inputs

# Usage with augmentation
pipeline = AugmentedImageClassificationPipeline(
    model=model,
    image_processor=image_processor,
    device=0
)

result = pipeline("image.jpg", augment=True)
# preprocess applies augmentation, then converts to tensors
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pipeline_Preprocessing]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Pipeline_Environment]]
