# Implementation: get_chat_template

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Chat Templates|https://huggingface.co/docs/transformers/chat_templating]]
|-
! Domains
| [[domain::NLP]], [[domain::Data_Preprocessing]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for applying chat templates to tokenizers, enabling consistent conversation formatting for instruction-tuned model fine-tuning, provided by Unsloth.

=== Description ===

`get_chat_template` configures a tokenizer with a specific chat template format (ChatML, Llama-3, Alpaca, etc.). It handles special token mapping, EOS token configuration, and dataset field remapping for consistent training data formatting.

Key capabilities:
* **20+ built-in templates** - ChatML, Llama-3, Alpaca, Gemma, Phi, Mistral, etc.
* **EOS token mapping** - Maps template stop tokens to existing tokenizer vocabulary
* **Dataset field mapping** - Remaps ShareGPT format fields (role→from, content→value)
* **Custom template support** - Pass (template_string, stop_token) tuple

=== Usage ===

Call after loading model and tokenizer, before dataset preparation. This ensures all training data uses consistent formatting that matches the model's instruction-tuning format.

Template selection guidelines:
* Use model's native format when available (e.g., "llama-3" for Llama-3 models)
* Use "chatml" as a universal fallback for most models
* Match template to your inference deployment (Ollama uses chatml by default)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/chat_templates.py
* '''Lines:''' L2123-2400

=== Signature ===
<syntaxhighlight lang="python">
def get_chat_template(
    tokenizer: PreTrainedTokenizer,
    chat_template: str = "chatml",
    mapping: dict = {
        "role": "role",
        "content": "content",
        "user": "user",
        "assistant": "assistant"
    },
    map_eos_token: bool = True,
    system_message: Optional[str] = None,
) -> PreTrainedTokenizer:
    """
    Apply a chat template to a tokenizer.

    Args:
        tokenizer: Tokenizer from FastLanguageModel.from_pretrained
        chat_template: Template name or (template_string, stop_token) tuple
        mapping: Field name remapping for dataset columns
        map_eos_token: Map template stop token to tokenizer's EOS
        system_message: Optional system message to prepend

    Returns:
        Tokenizer with chat_template attribute configured
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.chat_templates import get_chat_template
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer from model loading step
|-
| chat_template || str or tuple || No (default: "chatml") || Template name or (template, stop_token) tuple
|-
| mapping || dict || No || Field remapping: {"role": "from", "content": "value"} for ShareGPT
|-
| map_eos_token || bool || No (default: True) || Map template's stop token to tokenizer EOS
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tokenizer || PreTrainedTokenizer || Tokenizer with chat_template configured for apply_chat_template()
|}

== Usage Examples ==

=== Standard ChatML Template ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Apply ChatML template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",  # <|im_start|>role\ncontent<|im_end|>
)

# Test the template
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)
# <|im_start|>user
# What is 2+2?<|im_end|>
# <|im_start|>assistant
# 4<|im_end|>
</syntaxhighlight>

=== Llama-3 Native Template ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Use Llama-3's native template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
)

# Produces: <|begin_of_text|><|start_header_id|>user<|end_header_id|>...
</syntaxhighlight>

=== ShareGPT Dataset Mapping ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Map ShareGPT format: {"from": "human", "value": "..."}
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {
        "role": "from",
        "content": "value",
        "user": "human",
        "assistant": "gpt",
    },
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Data_Formatting]]
* [[implements::Principle:Unslothai_Unsloth_Chat_Template_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
