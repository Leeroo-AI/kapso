# Implementation: unslothai_unsloth_get_chat_template

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
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for configuring chat templates on tokenizers to enable proper formatting of conversational data for fine-tuning.

=== Description ===

`get_chat_template` configures a tokenizer with the appropriate chat template for a given model family. Chat templates define how multi-turn conversations are formatted into token sequences, including:

1. **System message handling**: Where and how system prompts are inserted
2. **Turn markers**: Tokens that delineate user vs assistant messages
3. **Special tokens**: BOS, EOS, and template-specific tokens

Unsloth supports 50+ chat template formats including Llama-3, ChatML, Alpaca, Mistral, Phi-3, and more.

=== Usage ===

Use this after loading a model and tokenizer, before applying chat formatting to your dataset. This is essential for instruction-following fine-tuning to ensure the model learns the correct conversation structure.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/chat_templates.py
* '''Lines:''' L50-500

=== Signature ===
<syntaxhighlight lang="python">
def get_chat_template(
    tokenizer: PreTrainedTokenizer,
    chat_template: str = "chatml",
    mapping: Optional[Dict[str, str]] = None,
    map_eos_token: bool = True,
    system_message: Optional[str] = None,
) -> PreTrainedTokenizer:
    """
    Configure a tokenizer with a chat template.

    Args:
        tokenizer: The tokenizer to configure
        chat_template: Template name (e.g., "llama-3", "chatml", "alpaca")
        mapping: Optional token remapping dictionary
        map_eos_token: Whether to remap EOS token
        system_message: Default system message to use

    Returns:
        Configured tokenizer with chat_template attribute set
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.chat_templates import get_chat_template
# or
from unsloth import get_chat_template
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer from model loading
|-
| chat_template || str || No (default: "chatml") || Template name: "llama-3", "chatml", "alpaca", "mistral", "phi-3", etc.
|-
| mapping || Dict[str, str] || No || Custom token remapping
|-
| map_eos_token || bool || No (default: True) || Remap EOS token for template
|-
| system_message || str || No || Default system message to prepend
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tokenizer || PreTrainedTokenizer || Tokenizer with `chat_template` attribute configured
|}

== Usage Examples ==

=== Llama-3 Template ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel, get_chat_template

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Configure Llama-3 chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",  # Llama 3 format
)

# Now tokenizer.apply_chat_template works correctly
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
)
print(formatted)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
</syntaxhighlight>

=== ChatML Template (Qwen, Yi) ===
<syntaxhighlight lang="python">
# ChatML format used by Qwen, Yi, and many others
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
)

# ChatML uses <|im_start|> and <|im_end|> markers
messages = [
    {"role": "user", "content": "What is Python?"},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
# <|im_start|>user\nWhat is Python?<|im_end|>\n<|im_start|>assistant\n
</syntaxhighlight>

=== Alpaca Template (Instruction Format) ===
<syntaxhighlight lang="python">
# Alpaca format for instruction-following
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "alpaca",
)

# Alpaca uses ### Instruction:, ### Input:, ### Response:
messages = [
    {"role": "user", "content": "Explain quantum computing."},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
# ### Instruction:\nExplain quantum computing.\n\n### Response:\n
</syntaxhighlight>

=== Custom System Message ===
<syntaxhighlight lang="python">
# Set a default system message for all conversations
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
    system_message = "You are a coding assistant specializing in Python.",
)
</syntaxhighlight>

=== Dataset Formatting with Chat Template ===
<syntaxhighlight lang="python">
from datasets import load_dataset

# Load and configure
model, tokenizer = FastLanguageModel.from_pretrained(...)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

# Define formatting function
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize = False,
            add_generation_prompt = False,
        )
        texts.append(text)
    return {"text": texts}

# Apply to dataset
dataset = load_dataset("your_dataset", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
</syntaxhighlight>

== Supported Templates ==

| Template | Models | Format |
|----------|--------|--------|
| `llama-3` / `llama-3.1` | Llama 3.x | `<\|start_header_id\|>role<\|end_header_id\|>\n\ncontent<\|eot_id\|>` |
| `chatml` | Qwen, Yi, OpenHermes | `<\|im_start\|>role\ncontent<\|im_end\|>` |
| `alpaca` | Alpaca, Vicuna | `### Instruction:\n...\n### Response:\n` |
| `mistral` | Mistral Instruct | `[INST] content [/INST]` |
| `phi-3` | Phi-3 | `<\|user\|>\ncontent<\|end\|>\n<\|assistant\|>` |
| `gemma` | Gemma | `<start_of_turn>role\ncontent<end_of_turn>` |
| `zephyr` | Zephyr | `<\|user\|>\ncontent<\|assistant\|>` |

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Data_Formatting]]
* [[implements::Principle:unslothai_unsloth_Chat_Template_Setup]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
