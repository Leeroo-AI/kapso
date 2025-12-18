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
Concrete tool for automatic processor loading provided by HuggingFace Transformers.

=== Description ===
The AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor, and AutoProcessor classes provide automatic instantiation of preprocessing components based on model configuration. These factory classes use the `from_pretrained` class method to determine the appropriate processor subclass by examining the model's configuration file, then instantiate and return the correct processor type. This implementation eliminates the need for users to know which specific processor class (e.g., BertTokenizer, ViTImageProcessor) is required for a given model.

The system maintains internal mappings from model types to processor classes, handles both Hugging Face Hub identifiers and local paths, manages fast vs. slow tokenizer selection, and supports custom processors defined in model repositories. The processors handle modality-specific transformations: tokenizers convert text to token IDs with special tokens and padding, image processors perform resizing/normalization/format conversion, feature extractors process audio into spectrograms, and unified processors coordinate multimodal inputs.

=== Usage ===
Import and use these classes when you need to:
* Load a preprocessor for a model without knowing the specific processor class
* Ensure compatibility between model and preprocessing pipeline
* Support multiple model architectures with a single code path
* Load processors from Hugging Face Hub or local directories
* Select fast tokenizer implementations when available for performance optimization

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/models/auto/tokenization_auto.py:L551-620, src/transformers/models/auto/image_processing_auto.py:L394-430, src/transformers/processing_utils.py:L100-300

=== Signature ===
<syntaxhighlight lang="python">
# AutoTokenizer
class AutoTokenizer:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *inputs,
        **kwargs
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """
        Instantiate one of the tokenizer classes of the library from a pretrained
        model vocabulary.

        The tokenizer class is selected based on the model_type property of the
        config object or by pattern matching on pretrained_model_name_or_path.
        """

# AutoImageProcessor
class AutoImageProcessor:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *inputs,
        **kwargs
    ) -> BaseImageProcessor:
        """
        Instantiate one of the image processor classes of the library from a
        pretrained model.

        The image processor class is selected based on the model_type property
        of the config object or by pattern matching.
        """

# AutoFeatureExtractor (similar pattern for audio/legacy vision)
# AutoProcessor (similar pattern for multimodal)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor, AutoProcessor
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pretrained_model_name_or_path || str or os.PathLike || Yes || Model identifier (Hub name like "bert-base-uncased") or local directory path containing processor files.
|-
| config || PreTrainedConfig || No || Configuration object to determine processor class. Auto-loaded if not provided.
|-
| cache_dir || str or os.PathLike || No || Directory to cache downloaded processor files. Uses default cache if not specified.
|-
| force_download || bool || No || Re-download files even if cached. Defaults to False.
|-
| resume_download || bool || No || Resume incomplete downloads. Defaults to False.
|-
| proxies || dict || No || Proxy configuration for downloads.
|-
| revision || str || No || Specific model version (branch/tag/commit). Defaults to "main".
|-
| token || str or bool || No || Hugging Face Hub authentication token. True uses cached token from huggingface-cli login.
|-
| trust_remote_code || bool || No || Allow loading custom processor code from Hub. Defaults to False for security.
|-
| use_fast || bool || No || (Tokenizer only) Prefer fast tokenizer implementation. Defaults to True.
|-
| local_files_only || bool || No || Only use locally cached files without downloading. Defaults to False.
|-
| **kwargs || Any || No || Additional processor-specific configuration parameters passed to processor __init__.
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| processor || PreTrainedTokenizer or BaseImageProcessor or PreTrainedFeatureExtractor or ProcessorMixin || Instantiated processor of the appropriate subclass for the model, configured and ready to preprocess inputs.
|}

== Usage Examples ==

=== Example 1: Load Tokenizer for Text Model ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load tokenizer for BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Use tokenizer
inputs = tokenizer(
    "Hello, world!",
    padding=True,
    truncation=True,
    return_tensors="pt"
)
# Output: {'input_ids': tensor([[...]), 'attention_mask': tensor([[...]])}

print(f"Tokenizer type: {type(tokenizer).__name__}")
# Output: BertTokenizerFast
</syntaxhighlight>

=== Example 2: Load Image Processor for Vision Model ===
<syntaxhighlight lang="python">
from transformers import AutoImageProcessor
from PIL import Image

# Load image processor for Vision Transformer
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Process image
image = Image.open("cat.jpg")
inputs = image_processor(images=image, return_tensors="pt")
# Output: {'pixel_values': tensor([[[[...]]]])}

print(f"Image processor type: {type(image_processor).__name__}")
# Output: ViTImageProcessor
</syntaxhighlight>

=== Example 3: Load Feature Extractor for Audio Model ===
<syntaxhighlight lang="python">
from transformers import AutoFeatureExtractor
import numpy as np

# Load feature extractor for Wav2Vec2
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

# Process audio (16kHz waveform)
audio = np.random.randn(16000)  # 1 second of audio
inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)
# Output: {'input_values': tensor([[...]])}
</syntaxhighlight>

=== Example 4: Load Processor for Multimodal Model ===
<syntaxhighlight lang="python">
from transformers import AutoProcessor
from PIL import Image

# Load processor for vision-language model (CLIP)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process both text and image
image = Image.open("dog.jpg")
inputs = processor(
    text=["a photo of a dog", "a photo of a cat"],
    images=image,
    return_tensors="pt",
    padding=True
)
# Output: {'input_ids': tensor([[...]]), 'pixel_values': tensor([[[[...]]]]), 'attention_mask': tensor([[...]])}

# Processor combines tokenizer and image processor
print(f"Has tokenizer: {processor.tokenizer is not None}")
print(f"Has image processor: {processor.image_processor is not None}")
# Output: True, True
</syntaxhighlight>

=== Example 5: Load Tokenizer with Custom Configuration ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load tokenizer with custom settings
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    use_fast=True,
    padding_side="left",  # Custom padding direction
    model_max_length=512,  # Override default max length
    cache_dir="./custom_cache"
)

# Configure pad token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize with custom settings
inputs = tokenizer(
    ["Short text", "This is a much longer text that demonstrates padding"],
    padding=True,
    truncation=True,
    max_length=20,
    return_tensors="pt"
)
print(inputs['input_ids'].shape)
# Output: torch.Size([2, 20])
</syntaxhighlight>

=== Example 6: Load from Local Directory ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Save tokenizer to local directory
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained("./my_tokenizer")

# Load from local directory
local_tokenizer = AutoTokenizer.from_pretrained(
    "./my_tokenizer",
    local_files_only=True  # Don't attempt to download
)

print(f"Loaded locally: {local_tokenizer is not None}")
# Output: True
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Processor_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Pipeline_Environment]]
