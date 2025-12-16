{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Vision Fine-tuning Guide|https://docs.unsloth.ai/basics/vision-fine-tuning]]
* [[source::Blog|Vision Models Blog|https://unsloth.ai/blog/vision]]
|-
! Domains
| [[domain::Vision_Language_Models]], [[domain::Fine_Tuning]], [[domain::Multimodal]], [[domain::QLoRA]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Concrete tool for loading and optimizing Vision-Language Models (VLMs) with quantization and memory optimizations, provided by the Unsloth library.

=== Description ===

`FastVisionModel` is an alias for `FastModel` that provides a unified API for loading multimodal models that process both images and text. It supports:

* **Multi-architecture VLMs**: Qwen2.5-VL, Llama 3.2 Vision, Pixtral, LLaVA-style models, Gemma 3 Vision
* **Component-aware loading**: Loads both vision encoder and language model with appropriate optimizations
* **Memory optimization**: 4-bit quantization, gradient checkpointing, and efficient image processing
* **Automatic processor setup**: Handles image tokenization, attention masks, and multi-image inputs

The class handles architecture-specific requirements including:
- Vision encoder configuration detection
- Cross-attention vs unified attention architectures
- Image token placeholder handling

=== Usage ===

Import `FastVisionModel` when you need to fine-tune a vision-language model for tasks like:
- Visual Question Answering (VQA)
- Image captioning
- OCR and document understanding
- Multi-image reasoning

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py unsloth/models/loader.py]
* '''Lines:''' 1257-1258 (FastVisionModel class definition)
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/vision.py unsloth/models/vision.py]
* '''Lines:''' 1-1263 (FastBaseModel with VLM support)

=== Signature ===
<syntaxhighlight lang="python">
class FastVisionModel(FastModel):
    """Alias for FastModel with VLM-specific documentation."""

    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        max_seq_length: int = 2048,
        dtype = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        load_in_16bit: bool = False,
        full_finetuning: bool = False,
        token: str = None,
        device_map: str = "sequential",
        trust_remote_code: bool = False,
        use_gradient_checkpointing: str = "unsloth",
        *args,
        **kwargs,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a Vision-Language Model with Unsloth optimizations.

        Automatically detects VLM architectures and applies appropriate
        optimizations to both vision encoder and language model components.
        """

    @staticmethod
    def get_peft_model(
        model,
        r: int = 16,
        target_modules: str = "all-linear",
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        **kwargs,
    ) -> PeftModel:
        """
        Apply LoRA to VLM with granular control over vision/language components.
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
| model_name || str || Yes || HuggingFace model ID (e.g., "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
|-
| max_seq_length || int || No || Maximum sequence length including image tokens (default: 2048)
|-
| load_in_4bit || bool || No || Enable 4-bit quantization (default: True)
|-
| trust_remote_code || bool || No || Allow custom model code (required for some VLMs)
|-
| use_gradient_checkpointing || str || No || "unsloth" for optimized checkpointing
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || VLM with vision encoder + language model, Unsloth optimizations applied
|-
| tokenizer || PreTrainedTokenizer || Processor/tokenizer with image handling capabilities
|}

== Usage Examples ==

=== Basic VLM Loading ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# Load a vision-language model with 4-bit quantization
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

# Check model architecture
print(f"Vision config: {model.config.vision_config}")
print(f"Text config: {model.config.text_config}")
</syntaxhighlight>

=== Apply LoRA to VLM ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA with fine-grained control
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    target_modules="all-linear",      # Apply to all linear layers
    finetune_vision_layers=True,      # Train vision encoder
    finetune_language_layers=True,    # Train language model
    finetune_attention_modules=True,  # Include attention
    finetune_mlp_modules=True,        # Include MLP
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
</syntaxhighlight>

=== Vision-Only Fine-tuning ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Only train vision encoder, freeze language model
model = FastVisionModel.get_peft_model(
    model,
    r=32,
    finetune_vision_layers=True,      # Train vision
    finetune_language_layers=False,   # Freeze language
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)
</syntaxhighlight>

=== Prepare Image-Text Data ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastVisionModel.get_peft_model(model, r=16)

# Create data collator for VLM training
data_collator = UnslothVisionDataCollator(model, tokenizer)

# Example data format
sample = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/path/to/image.jpg"},
                {"type": "text", "text": "What is in this image?"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This image shows..."}]
        }
    ]
}
</syntaxhighlight>

== Supported Models ==

{| class="wikitable"
|-
! Model Family !! Example Model !! Notes
|-
| Qwen2.5-VL || unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit || Best overall performance
|-
| Llama 3.2 Vision || unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit || Good for general VQA
|-
| Pixtral || mistralai/Pixtral-12B-2409 || Requires trust_remote_code
|-
| Gemma 3 Vision || google/gemma-3-12b-vision || Requires transformers>=4.50
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Vision_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
