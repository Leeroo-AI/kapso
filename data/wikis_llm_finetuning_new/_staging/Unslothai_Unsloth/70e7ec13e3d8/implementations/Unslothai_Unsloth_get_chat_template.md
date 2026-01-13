# Implementation: get_chat_template

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Chat Templating|https://huggingface.co/docs/transformers/chat_templating]]
|-
! Domains
| [[domain::NLP]], [[domain::Data_Preprocessing]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for applying chat templates to tokenizers provided by the Unsloth library.

=== Description ===

`get_chat_template` configures a tokenizer with a specific chat template format. It handles:

* Applying predefined templates (chatml, llama3, mistral, gemma, qwen25, etc.)
* Mapping stop tokens to EOS for proper generation termination
* Token remapping for models with non-standard special tokens
* Custom Jinja2 template support
* ShareGPT-style column mapping (from/value → role/content)

The function modifies the tokenizer's `chat_template` attribute and optionally remaps special tokens in the vocabulary.

=== Usage ===

Import this function after loading your model and tokenizer. Apply it to ensure your tokenizer correctly formats conversational data. This is essential when your dataset format differs from the model's default or when you need specific stop token behavior.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/chat_templates.py
* '''Lines:''' 2123-2349

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
    Apply a chat template to the tokenizer.

    Args:
        tokenizer: The tokenizer to modify
        chat_template: Template name or custom Jinja2 template string
        mapping: Column name mapping for ShareGPT-style datasets
        map_eos_token: Whether to map the stop token to EOS
        system_message: Optional default system message

    Returns:
        Modified tokenizer with chat_template attribute set
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
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer from FastLanguageModel.from_pretrained
|-
| chat_template || str || No || Template name ("chatml", "llama3", "mistral", "gemma", "qwen25") or custom Jinja2 string (default: "chatml")
|-
| mapping || dict || No || Column mapping for ShareGPT format (default: standard role/content)
|-
| map_eos_token || bool || No || Map template stop token to EOS (default: True)
|-
| system_message || str || No || Default system message to include
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tokenizer || PreTrainedTokenizer || Tokenizer with chat_template attribute configured
|}

== Usage Examples ==

=== Using ChatML Template ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply ChatML template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    map_eos_token=True,
)

# Test the template
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
]
print(tokenizer.apply_chat_template(messages, tokenize=False))
</syntaxhighlight>

=== Using Llama-3 Template ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply Llama-3 native template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
    map_eos_token=True,
)
</syntaxhighlight>

=== ShareGPT Dataset Mapping ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Map ShareGPT format (from/value) to standard format (role/content)
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={
        "role": "from",
        "content": "value",
        "user": "human",
        "assistant": "gpt",
    },
)
</syntaxhighlight>

=== Custom System Message ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Set a default system message
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    system_message="You are a helpful coding assistant.",
)
</syntaxhighlight>

== Supported Templates ==

{| class="wikitable"
|-
! Template Name !! Stop Token !! Models
|-
| chatml || <nowiki><|im_end|></nowiki> || General purpose, many open models
|-
| llama-3.1 || <nowiki><|eot_id|></nowiki> || Llama 3, Llama 3.1, Llama 3.2
|-
| mistral || </s> || Mistral, Mixtral
|-
| gemma || <end_of_turn> || Gemma, Gemma 2
|-
| qwen25 || <nowiki><|im_end|></nowiki> || Qwen 2.5
|-
| phi-3 || <nowiki><|end|></nowiki> || Phi-3
|-
| alpaca || (none) || Alpaca-style single-turn
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Data_Formatting]]

