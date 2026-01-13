# Implementation: FastVisionModel_from_pretrained

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Vision Models|https://huggingface.co/docs/transformers/model_doc/auto]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::NLP]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for loading vision-language models with quantization provided by the Unsloth library.

=== Description ===

`FastVisionModel.from_pretrained` loads vision-language models (Qwen2-VL, Llama 3.2 Vision, Pixtral) with Unsloth optimizations. It returns both a model and a processor (not tokenizer) that handles image preprocessing.

Key features:
* 4-bit quantization for both vision and language components
* Automatic processor configuration for image handling
* Support for variable resolution images
* Integration with AutoModelForVision2Seq or AutoModelForImageTextToText

=== Usage ===

Import this function when working with multimodal datasets containing images. Note that vision models return a processor instead of a tokenizer, which handles both image preprocessing and text tokenization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/vision.py
* '''Lines:''' 321-919 (FastBaseModel.from_pretrained)
* '''Alias:''' unsloth/models/loader.py:1369-1370 (FastVisionModel = FastModel)

=== Signature ===
<syntaxhighlight lang="python">
class FastVisionModel:
    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        load_in_16bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
        device_map: str = "sequential",
        trust_remote_code: bool = False,
        use_gradient_checkpointing: Union[str, bool] = "unsloth",
        # Vision-specific
        auto_model: type = AutoModelForVision2Seq,
        **kwargs,
    ) -> Tuple[PreTrainedModel, AutoProcessor]:
        """
        Load a vision-language model with optional quantization.

        Args:
            model_name: HuggingFace model ID for a VLM
            max_seq_length: Maximum sequence length
            dtype: Model precision
            load_in_4bit: Enable 4-bit quantization
            token: HuggingFace API token
            device_map: Device placement strategy
            auto_model: Model class (AutoModelForVision2Seq or AutoModelForImageTextToText)

        Returns:
            Tuple of (model, processor) where processor handles both images and text
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || Yes || HuggingFace VLM model ID (e.g., "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
|-
| max_seq_length || int || No || Maximum sequence length (default: 2048)
|-
| dtype || torch.dtype || No || Model precision (auto-detected if None)
|-
| load_in_4bit || bool || No || Enable 4-bit quantization (default: True)
|-
| token || str || No || HuggingFace API token for private models
|-
| auto_model || type || No || Model class (default: AutoModelForVision2Seq)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Vision-language model with Unsloth optimizations
|-
| processor || AutoProcessor || Handles both image preprocessing and text tokenization
|}

== Usage Examples ==

=== Basic Vision Model Loading ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# Load a Qwen2-VL model
model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Note: processor, not tokenizer!
print(f"Model loaded: {model.config.model_type}")
</syntaxhighlight>

=== Loading Llama 3.2 Vision ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# Load Llama 3.2 Vision
model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
</syntaxhighlight>

=== Using for Inference ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel
from PIL import Image

model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load an image
image = Image.open("example.jpg")

# Create conversation with image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Process inputs
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

# Generate
FastVisionModel.for_inference(model)
outputs = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(outputs[0], skip_special_tokens=True)
</syntaxhighlight>

== Supported Models ==

{| class="wikitable"
|-
! Model Family !! Example Model !! Notes
|-
| Qwen2-VL || unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit || Dynamic resolution, M-RoPE
|-
| Llama 3.2 Vision || unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit || Cross-attention vision
|-
| Pixtral || mistralai/Pixtral-12B-2409 || Variable resolution
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Vision_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_11]]
* [[requires_env::Environment:Unslothai_Unsloth_Vision]]
