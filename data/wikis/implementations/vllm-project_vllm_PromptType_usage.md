{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Input_Processing]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Pattern documentation for formatting prompts as text strings, pre-tokenized IDs, or multimodal inputs for vLLM's generate methods.

=== Description ===

`PromptType` is a union type that defines the acceptable input formats for vLLM's generation methods. It supports multiple input modalities:

- **TextPrompt:** Raw text strings (most common)
- **TokensPrompt:** Pre-tokenized token IDs (for advanced control)
- **Dict format:** Dictionary with "prompt" or "prompt_token_ids" keys
- **Multimodal:** Dictionaries with "multi_modal_data" for images/video

This flexibility allows users to choose the most convenient format for their use case while vLLM handles the normalization internally.

=== Usage ===

Use the appropriate prompt format based on your needs:
- **Text strings:** Most common case, vLLM tokenizes automatically
- **Token IDs:** When you've pre-tokenized or need exact token control
- **Dictionaries:** When combining text with multimodal data (images)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/inputs/__init__.py
* '''Lines:''' L1-50 (re-exports from vllm/inputs/data.py)

=== Interface Specification ===
<syntaxhighlight lang="python">
# Type definitions for prompts
PromptType = Union[
    str,                           # Simple text prompt
    TextPrompt,                    # TypedDict with "prompt" key
    TokensPrompt,                  # TypedDict with "prompt_token_ids" key
    ExplicitEncoderDecoderPrompt,  # For encoder-decoder models
]

# TextPrompt structure
class TextPrompt(TypedDict):
    prompt: str
    multi_modal_data: NotRequired[MultiModalDataDict]

# TokensPrompt structure
class TokensPrompt(TypedDict):
    prompt_token_ids: list[int]
    multi_modal_data: NotRequired[MultiModalDataDict]

# MultiModalDataDict for images/video
MultiModalDataDict = dict[str, Any]  # e.g., {"image": PIL.Image}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.inputs import PromptType, TextPrompt, TokensPrompt
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| prompt || str || Conditional || Raw text string (if using TextPrompt)
|-
| prompt_token_ids || list[int] || Conditional || Pre-tokenized token IDs (if using TokensPrompt)
|-
| multi_modal_data || dict || No || Multimodal data (images, video) keyed by modality
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| PromptType || Union type || Normalized prompt accepted by LLM.generate()
|}

== Usage Examples ==

=== Simple Text Prompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Simplest form: just a string
prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Pre-tokenized Input ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
tokenizer = llm.get_tokenizer()

# Pre-tokenize for exact control
text = "Hello, world!"
token_ids = tokenizer.encode(text)

# Use TokensPrompt format
prompt = {"prompt_token_ids": token_ids}
outputs = llm.generate([prompt], SamplingParams(max_tokens=50))
</syntaxhighlight>

=== Batch with Mixed Formats ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Mix of string and dict formats (all valid)
prompts = [
    "Simple string prompt",
    {"prompt": "Dict with prompt key"},
    {"prompt_token_ids": [1, 2, 3, 4, 5]},
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Chat Template Formatting ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
tokenizer = llm.get_tokenizer()

# Apply chat template for instruction-tuned models
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2?"},
]

# Tokenizer applies chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Prompt_Formatting]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
