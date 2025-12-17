{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Chat]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for formatting chat conversations into model-specific input sequences provided by the HuggingFace Transformers library.

=== Description ===

This method converts structured chat conversations (messages with roles like 'user', 'assistant', 'system') into properly formatted text using Jinja2 templates. It handles model-specific formatting conventions, adds generation prompts, supports tool/function calling schemas, and optionally tokenizes the output. Essential for chat and instruction-tuned models.

=== Usage ===

Use this when working with chat models to format multi-turn conversations correctly. Ensures proper formatting with special tokens, role indicators, and generation prompts that match the model's training format.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py (lines 3027-3197)
* '''Helper methods:''' get_chat_template (lines 3259-3311), save_chat_templates (lines 3313-3366)

=== Signature ===
<syntaxhighlight lang="python">
def apply_chat_template(
    self,
    conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
    tools: Optional[list[Union[dict, Callable]]] = None,
    documents: Optional[list[dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = False,
    continue_final_message: bool = False,
    tokenize: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_dict: bool = True,
    return_assistant_tokens_mask: bool = False,
    tokenizer_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Union[str, list[int], list[str], list[list[int]], BatchEncoding]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| conversation || list[dict] or list[list[dict]] || Yes || Messages with 'role' and 'content' keys, or batch thereof
|-
| tools || list[dict or Callable] || No || Tool/function definitions for function calling (JSON Schema format)
|-
| documents || list[dict] || No || Documents for RAG (retrieval-augmented generation)
|-
| chat_template || str || No || Custom Jinja template (uses model default if not specified)
|-
| add_generation_prompt || bool || No || Add prompt for assistant response (default: False)
|-
| continue_final_message || bool || No || Format to continue final message instead of starting new (default: False)
|-
| tokenize || bool || No || Whether to tokenize output (default: True)
|-
| padding || bool or str || No || Padding strategy when tokenizing
|-
| truncation || bool || No || Whether to truncate when tokenizing
|-
| max_length || int || No || Maximum length when tokenizing
|-
| return_tensors || str || No || Return format: 'pt' or 'np' when tokenizing
|-
| return_dict || bool || No || Return BatchEncoding dict (default: True)
|-
| return_assistant_tokens_mask || bool || No || Return mask for assistant-generated tokens
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || str, list[int], or BatchEncoding || Formatted chat string (tokenize=False) or tokenized IDs/BatchEncoding (tokenize=True)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# Basic chat formatting
conversation = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is its population?"}
]

# Get tokenized input with generation prompt
inputs = tokenizer.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt"
)

# Generate response
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
outputs = model.generate(inputs, max_new_tokens=100)

# Get formatted string without tokenizing
formatted_chat = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)
print(formatted_chat)

# Batch processing
conversations = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "user", "content": "How are you?"}]
]
batch_inputs = tokenizer.apply_chat_template(
    conversations,
    padding=True,
    return_tensors="pt"
)

# System message example
conversation_with_system = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."}
]
inputs = tokenizer.apply_chat_template(
    conversation_with_system,
    add_generation_prompt=True,
    return_tensors="pt"
)

# Function calling with tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }
]

conversation = [
    {"role": "user", "content": "What's the weather in Paris?"}
]

inputs = tokenizer.apply_chat_template(
    conversation,
    tools=tools,
    add_generation_prompt=True,
    return_tensors="pt"
)

# Continue generation (prefilling)
conversation = [
    {"role": "user", "content": "Write a poem"},
    {"role": "assistant", "content": "Roses are red,"}
]

inputs = tokenizer.apply_chat_template(
    conversation,
    continue_final_message=True,
    return_tensors="pt"
)

# Get assistant token masks (for training)
inputs = tokenizer.apply_chat_template(
    conversation,
    return_dict=True,
    return_assistant_tokens_mask=True,
    return_tensors="pt"
)
# inputs["assistant_masks"] indicates assistant tokens
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Chat_Templates]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
