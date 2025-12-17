# PromptType

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Inference]], [[domain::Input_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Concrete tool for structuring LLM inputs using TextPrompt, TokensPrompt, or other supported formats for batch inference.

=== Description ===

`PromptType` is a type alias representing all valid input formats for vLLM's `LLM.generate()` method. The primary concrete types are:

* **TextPrompt**: Dictionary with `prompt` key for text that needs tokenization
* **TokensPrompt**: Dictionary with `prompt_token_ids` for pre-tokenized input
* **EmbedsPrompt**: Dictionary with `prompt_embeds` for direct embedding input

These TypedDict classes provide type safety and IDE support while allowing flexible input handling.

=== Usage ===

Use these prompt types when:
* **TextPrompt**: Standard text inference with optional multimodal data
* **TokensPrompt**: Pre-tokenized inputs, token type IDs, or bypassing tokenizer
* **EmbedsPrompt**: Direct embedding input for specialized models

For simple cases, plain strings work and are converted to TextPrompt internally.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm]
* '''File:''' vllm/inputs/data.py

=== Signature ===
<syntaxhighlight lang="python">
class TextPrompt(TypedDict):
    """Schema for a text prompt."""
    prompt: str
    """The input text to be tokenized."""

    multi_modal_data: NotRequired[MultiModalDataDict | None]
    """Optional multi-modal data (images, audio)."""

    mm_processor_kwargs: NotRequired[dict[str, Any] | None]
    """Optional kwargs for multimodal processor."""

    cache_salt: NotRequired[str]
    """Optional cache key for prefix caching."""


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""
    prompt_token_ids: list[int]
    """Pre-tokenized token IDs."""

    prompt: NotRequired[str]
    """Original text (optional, for logging)."""

    token_type_ids: NotRequired[list[int]]
    """Token type IDs for cross-encoder models."""

    multi_modal_data: NotRequired[MultiModalDataDict | None]
    """Optional multi-modal data."""


# Type alias for all prompt types
SingletonPrompt: TypeAlias = str | TextPrompt | TokensPrompt | EmbedsPrompt
PromptType: TypeAlias = SingletonPrompt | ExplicitEncoderDecoderPrompt | DataPrompt
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.inputs import TextPrompt, TokensPrompt, PromptType
# Or simply use from vllm import LLM (prompts are dicts)
</syntaxhighlight>

== I/O Contract ==

=== Inputs (TextPrompt) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| prompt || str || Yes || Text to tokenize
|-
| multi_modal_data || dict || No || Images/audio keyed by modality
|-
| mm_processor_kwargs || dict || No || Multimodal processor options
|-
| cache_salt || str || No || Custom prefix cache key
|}

=== Inputs (TokensPrompt) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| prompt_token_ids || list[int] || Yes || Pre-tokenized token IDs
|-
| prompt || str || No || Original text for logging
|-
| token_type_ids || list[int] || No || Token types for cross-encoders
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (dict) || TypedDict || Structured prompt ready for LLM.generate()
|}

== Usage Examples ==

=== Simple String Input ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Raw strings are the simplest format
prompts = [
    "What is the capital of France?",
    "Explain quantum computing briefly.",
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
</syntaxhighlight>

=== TextPrompt with Multimodal Data ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# TextPrompt with image data
prompt = {
    "prompt": "<image>\nDescribe this image in detail.",
    "multi_modal_data": {
        "image": Image.open("photo.jpg")
    }
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=256))
</syntaxhighlight>

=== TokensPrompt for Pre-tokenized Input ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Pre-tokenize for maximum control
text = "What is machine learning?"
token_ids = tokenizer.encode(text, add_special_tokens=True)

prompt = {
    "prompt_token_ids": token_ids,
    "prompt": text,  # Optional, for logging
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Batch with Mixed Types ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Mix of raw strings and structured prompts
prompts = [
    "Simple string prompt",                          # str
    {"prompt": "TextPrompt format"},                 # TextPrompt
    {"prompt_token_ids": [128000, 9906, 1917]},     # TokensPrompt
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=50))
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Input_Formatting]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
