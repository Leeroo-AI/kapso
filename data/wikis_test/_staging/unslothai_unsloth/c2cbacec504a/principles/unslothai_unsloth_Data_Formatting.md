# Principle: unslothai_unsloth_Data_Formatting

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Chat Templates|https://huggingface.co/docs/transformers/chat_templating]]
* [[source::Blog|ShareGPT Format|https://github.com/lm-sys/FastChat/blob/main/docs/commands/data.md]]
* [[source::Paper|FLAN-T5: Scaling Instruction-Finetuned Models|https://arxiv.org/abs/2210.11416]]
|-
! Domains
| [[domain::NLP]], [[domain::Data_Preprocessing]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for formatting conversational and instruction-following data into sequences that language models can learn from during fine-tuning.

=== Description ===

Data Formatting is the process of converting structured conversation data (user/assistant turns) into properly tokenized sequences. This involves:

1. **Chat Template Application**: Using model-specific formatting with special tokens
2. **Role Delineation**: Marking boundaries between different speakers
3. **Response Masking**: Optionally training only on assistant responses
4. **Sequence Construction**: Building complete training examples from conversations

Proper formatting is critical because models learn the statistical patterns in training data—incorrect formatting leads to malformed outputs during inference.

=== Usage ===

Use this principle when:
- Preparing any conversation or instruction dataset for fine-tuning
- Converting between dataset formats (ShareGPT, Alpaca, OpenAI format)
- Ensuring training data matches the target model's expected format

This step comes after model loading and before trainer configuration.

== Theoretical Basis ==

=== Chat Template Structure ===

A chat template is a Jinja2 template that converts conversation objects to token sequences:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Abstract chat template structure
template = """
{bos_token}
{% for message in messages %}
    {% if message.role == 'system' %}
        {system_marker}{message.content}{end_marker}
    {% elif message.role == 'user' %}
        {user_marker}{message.content}{end_marker}
    {% elif message.role == 'assistant' %}
        {assistant_marker}{message.content}{eos_token}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {assistant_marker}
{% endif %}
"""
</syntaxhighlight>

=== Role Markers by Model Family ===

Different models use different special tokens:

{| class="wikitable"
|-
! Model Family !! User Marker !! Assistant Marker !! End Marker
|-
| Llama-3 || `<\|start_header_id\|>user<\|end_header_id\|>` || `<\|start_header_id\|>assistant<\|end_header_id\|>` || `<\|eot_id\|>`
|-
| ChatML || `<\|im_start\|>user` || `<\|im_start\|>assistant` || `<\|im_end\|>`
|-
| Alpaca || `### Instruction:` || `### Response:` || (none)
|-
| Mistral || `[INST]` || (none) || `[/INST]`
|}

=== Training Signal Masking ===

For instruction-following, we typically only want to train on assistant responses:

<syntaxhighlight lang="python">
# Example sequence after tokenization
# [User tokens...] [Assistant tokens...]
# Labels:
# [-100, -100, ...] [actual_tokens...]
#  ^^^ignored^^^    ^^^trained on^^^

def create_masked_labels(input_ids, assistant_start_idx):
    labels = input_ids.clone()
    # Mask everything before assistant response
    labels[:assistant_start_idx] = -100  # Ignored in loss
    return labels
</syntaxhighlight>

=== Dataset Format Standards ===

Common dataset formats and their structures:

'''ShareGPT Format:'''
<syntaxhighlight lang="python">
{
    "conversations": [
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "Hi there!"},
    ]
}
</syntaxhighlight>

'''OpenAI Format:'''
<syntaxhighlight lang="python">
{
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
}
</syntaxhighlight>

'''Alpaca Format:'''
<syntaxhighlight lang="python">
{
    "instruction": "Explain quantum computing",
    "input": "",  # Optional context
    "output": "Quantum computing is..."
}
</syntaxhighlight>

=== Sequence Length Considerations ===

Formatting affects sequence length:

<syntaxhighlight lang="python">
# Template overhead varies by format
llama3_overhead = len("<|begin_of_text|><|start_header_id|>system<|end_header_id|>...")  # ~50 tokens per turn
chatml_overhead = len("<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...")  # ~12 tokens per turn

# Effective content length
effective_content = max_seq_length - template_overhead * num_turns
</syntaxhighlight>

== Practical Guide ==

=== Format Conversion Workflow ===

<syntaxhighlight lang="python">
# 1. Load raw dataset
dataset = load_dataset("your_dataset")

# 2. Standardize to messages format
def convert_to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }
dataset = dataset.map(convert_to_messages)

# 3. Apply chat template
def apply_template(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}
dataset = dataset.map(apply_template, batched=True)
</syntaxhighlight>

=== Best Practices ===

1. **Match template to model**: Use the template the model was trained with
2. **Include EOS tokens**: Ensure responses end with the model's EOS token
3. **Handle truncation carefully**: Preserve conversation structure when truncating
4. **Validate formatting**: Check a few examples manually before training

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_get_chat_template]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
