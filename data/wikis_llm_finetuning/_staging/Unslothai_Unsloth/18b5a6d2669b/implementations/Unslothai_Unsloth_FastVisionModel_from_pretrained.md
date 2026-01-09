# Implementation: FastVisionModel_from_pretrained

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Transformers Vision Models|https://huggingface.co/docs/transformers/model_doc/llava]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::NLP]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for loading Vision-Language Models (VLMs) with 4-bit quantization for memory-efficient multimodal fine-tuning, provided by Unsloth.

=== Description ===

`FastVisionModel.from_pretrained` loads pre-trained VLMs (Llama-3.2-Vision, Qwen2-VL, Pixtral, etc.) with Unsloth optimizations. Unlike `FastLanguageModel`, it returns an `AutoProcessor` instead of a tokenizer, handling both text and image preprocessing.

Key capabilities:
* **4-bit quantization** - Reduces VLM memory footprint significantly
* **AutoProcessor** - Combined tokenizer + image processor
* **Multi-architecture support** - Llama 3.2 Vision, Qwen2-VL, Pixtral, Gemma3
* **Gradient checkpointing** - Memory-efficient training

=== Usage ===

Use when fine-tuning vision-language models for tasks like:
* Image captioning
* Visual question answering
* Document/OCR understanding
* Chart/diagram comprehension

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/loader.py (FastVisionModel), unsloth/models/vision.py (FastBaseModel)
* '''Lines:''' L702-900 (loader.py), L321-918 (vision.py)

=== Signature ===
<syntaxhighlight lang="python">
class FastVisionModel:
    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        load_in_16bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
        device_map: str = "sequential",
        trust_remote_code: bool = False,
        use_gradient_checkpointing: str = "unsloth",
        fast_inference: bool = False,
        gpu_memory_utilization: float = 0.5,
        **kwargs,
    ) -> Tuple[PreTrainedModel, AutoProcessor]:
        """
        Load a Vision-Language Model with Unsloth optimizations.

        Args:
            model_name: HuggingFace VLM ID or local path
            max_seq_length: Maximum sequence length
            load_in_4bit: Enable 4-bit quantization
            use_gradient_checkpointing: "unsloth" for optimized checkpointing

        Returns:
            Tuple of (model, processor) where processor is AutoProcessor
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
| model_name || str || Yes || HuggingFace VLM ID (e.g., "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit")
|-
| max_seq_length || int || No (default: 2048) || Maximum sequence length for text
|-
| load_in_4bit || bool || No (default: True) || Enable 4-bit quantization
|-
| use_gradient_checkpointing || str || No (default: "unsloth") || Memory-efficient checkpointing
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Vision-language model with Unsloth patches
|-
| processor || AutoProcessor || Combined tokenizer and image processor
|}

== Usage Examples ==

=== Basic VLM Loading ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# Load Llama 3.2 Vision model
model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# processor handles both text and images
print(f"Processor: {type(processor)}")
</syntaxhighlight>

=== Loading Qwen2-VL ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# Load Qwen2-VL for document understanding
model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length = 4096,  # Longer for documents
    load_in_4bit = True,
)
</syntaxhighlight>

=== Loading with Custom Memory Settings ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# Load for limited GPU memory
model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    max_seq_length = 1024,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",  # Memory-efficient
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Vision_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Gradient_Checkpointing_Tip]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_BFloat16_vs_Float16_Tip]]
