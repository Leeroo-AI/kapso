{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Dataset Formatting|https://docs.unsloth.ai/basics/data-prep]]
|-
! Domains
| [[domain::Data_Preparation]], [[domain::NLP]], [[domain::Chat_Templates]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of transforming raw instruction-response data into model-specific chat template format for supervised fine-tuning.

=== Description ===

Data formatting for LLM fine-tuning involves:

**Chat Template Application:**
- Each model family has specific chat templates (Llama 3, Mistral, Qwen, etc.)
- Templates define special tokens, roles, and formatting
- Unsloth provides 50+ built-in templates via `get_chat_template()`

**Key Considerations:**
- Proper EOS token placement for training
- Response-only loss masking (optional)
- Handling of system prompts
- Multi-turn conversation formatting

**Common Formats:**
- Alpaca: instruction/input/output fields
- ShareGPT: conversations list with role/content
- ChatML: XML-like tags for roles

=== Usage ===

Format data when:
- Preparing custom datasets for fine-tuning
- Converting between dataset formats
- Applying model-specific chat templates

== Practical Guide ==

=== Using HuggingFace Tokenizer Templates ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Format a conversation using the model's chat template
def format_conversation(example):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]

    # Apply model's native template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_conversation)
</syntaxhighlight>

=== Using Unsloth Chat Templates ===
<syntaxhighlight lang="python">
from unsloth.chat_templates import get_chat_template

# Get template for specific model
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # or "mistral", "chatml", "qwen", etc.
)

# Now tokenizer.apply_chat_template uses the correct format
</syntaxhighlight>

=== Alpaca Format ===
<syntaxhighlight lang="python">
# Raw Alpaca format
{
    "instruction": "Summarize the following text.",
    "input": "The quick brown fox...",
    "output": "A fox jumps over a dog."
}

# Formatted output
"""
### Instruction:
Summarize the following text.

### Input:
The quick brown fox...

### Response:
A fox jumps over a dog.
"""
</syntaxhighlight>

=== Response-Only Training ===
<syntaxhighlight lang="python">
from unsloth import train_on_responses_only

# Train only on assistant responses, not user prompts
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>",
    response_part="<|start_header_id|>assistant<|end_header_id|>",
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
