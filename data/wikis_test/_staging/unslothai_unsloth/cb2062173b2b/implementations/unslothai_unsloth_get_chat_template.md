{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|Chat Templates|https://docs.unsloth.ai/basics/chat-templates]]
|-
! Domains
| [[domain::LLMs]], [[domain::Data_Formatting]], [[domain::Chat_Templates]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:00 GMT]]
|}

== Overview ==
Concrete tool for applying chat templates to tokenizers for instruction-tuning provided by the Unsloth library.

=== Description ===
`get_chat_template()` is a function that configures tokenizers with chat formatting templates (Jinja2-based). It:

1. **Applies template formatting** using predefined templates (Llama-3, ChatML, Alpaca, etc.)
2. **Maps EOS tokens** correctly for each template format
3. **Handles system messages** with template-specific defaults
4. **Generates Ollama Modelfiles** with matching prompt formats
5. **Supports 30+ templates** including custom user-defined formats

The function modifies the tokenizer's `chat_template` attribute, enabling proper use of `tokenizer.apply_chat_template()` for conversation formatting. This is essential for instruction-tuning as it ensures the model learns the correct prompt structure.

=== Usage ===
Use this function when you need to:
- Format training data for instruction-tuning
- Apply a specific chat format (Llama-3, ChatML, Alpaca)
- Ensure tokenizer matches model's expected format
- Generate properly formatted prompts for inference

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L2123-L2400 unsloth/chat_templates.py]
* '''Lines:''' 2123-2400

Source Files: unsloth/chat_templates.py:L2123-L2400

=== Signature ===
<syntaxhighlight lang="python">
def get_chat_template(
    tokenizer: PreTrainedTokenizer,
    chat_template: str = "chatml",
    mapping: Dict[str, str] = {
        "role": "role",
        "content": "content",
        "user": "user",
        "assistant": "assistant"
    },
    map_eos_token: bool = True,
    system_message: Optional[str] = None,
) -> PreTrainedTokenizer:
    """
    Apply a chat template to a tokenizer for instruction formatting.

    Args:
        tokenizer: The tokenizer to configure
        chat_template: Template name or custom (template, stop_word) tuple
        mapping: Field name mapping for dataset compatibility
        map_eos_token: Whether to update tokenizer's EOS token
        system_message: Custom system prompt (uses template default if None)

    Returns:
        Configured tokenizer with chat_template set
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
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer to configure
|-
| chat_template || str || No (default: "chatml") || Template name (see available templates)
|-
| mapping || Dict || No || Field name mapping for datasets
|-
| map_eos_token || bool || No (default: True) || Update tokenizer EOS token
|-
| system_message || str || No || Custom system prompt
|}

=== Available Templates ===
{| class="wikitable"
|-
! Template !! Model Family !! Format
|-
| llama-3 / llama-3.1 || Llama 3.x || <nowiki><|start_header_id|>...<|end_header_id|></nowiki>
|-
| chatml || Qwen, Yi, many others || <nowiki><|im_start|>...<|im_end|></nowiki>
|-
| alpaca || Alpaca-style || ### Instruction: ... ### Response:
|-
| mistral || Mistral Instruct || [INST] ... [/INST]
|-
| gemma || Gemma || <nowiki><start_of_turn>...<end_of_turn></nowiki>
|-
| vicuna || Vicuna || USER: ... ASSISTANT:
|-
| zephyr || Zephyr || <nowiki><|user|>...<|assistant|></nowiki>
|-
| phi-3 || Phi-3 || <nowiki><|user|>...<|assistant|></nowiki>
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tokenizer || PreTrainedTokenizer || Tokenizer with chat_template configured
|}

== Usage Examples ==

=== Apply Llama-3 Template ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
)

# Apply Llama-3 chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# Now use apply_chat_template for formatting
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(text)
# Output: <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant.<|eot_id|>...
</syntaxhighlight>

=== Apply ChatML Template (Qwen, etc.) ===
<syntaxhighlight lang="python">
from unsloth.chat_templates import get_chat_template

# ChatML is default and works with many models
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
)

messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
# Output: <|im_start|>user
# Hello!<|im_end|>
# <|im_start|>assistant
# Hi there!<|im_end|>
</syntaxhighlight>

=== Alpaca Format for Instruction Tuning ===
<syntaxhighlight lang="python">
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="alpaca",
    system_message="Below are instructions. Write appropriate responses.",
)

# Format for Alpaca-style datasets
messages = [
    {"role": "user", "content": "Summarize the following text..."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
# Output: Below are instructions...
#
# ### Instruction:
# Summarize the following text...
#
# ### Response:
</syntaxhighlight>

=== Custom System Message ===
<syntaxhighlight lang="python">
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
    system_message="You are a coding assistant specialized in Python.",
)

# System message will be used when none is provided in messages
messages = [{"role": "user", "content": "Write a function to add two numbers"}]
text = tokenizer.apply_chat_template(messages, tokenize=False)
</syntaxhighlight>

=== Format Training Dataset ===
<syntaxhighlight lang="python">
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

def formatting_func(examples):
    """Format dataset rows as chat conversations."""
    conversations = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        conv = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        conversations.append(conv)

    texts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False
        )
        for conv in conversations
    ]
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True)
</syntaxhighlight>

== Related Pages ==
''No additional environment or heuristic requirements.''
