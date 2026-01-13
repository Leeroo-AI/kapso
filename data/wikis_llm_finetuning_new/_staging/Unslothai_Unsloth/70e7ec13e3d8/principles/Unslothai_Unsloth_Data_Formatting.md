# Principle: Data_Formatting

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Chat Templating|https://huggingface.co/docs/transformers/chat_templating]]
* [[source::Blog|ChatML Format|https://github.com/openai/openai-python/blob/main/chatml.md]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Data_Preprocessing]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for applying chat templates to tokenizers to properly format conversational data for instruction fine-tuning.

=== Description ===

Data Formatting in LLM fine-tuning involves applying a chat template to the tokenizer that defines how multi-turn conversations are structured. This ensures the model learns the correct boundaries between user messages, assistant responses, and system prompts.

Different model families use different chat formats:
* **ChatML**: `<|im_start|>role\ncontent<|im_end|>` (used by many open models)
* **Llama-3**: `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>`
* **Mistral**: `[INST] user message [/INST] assistant response`
* **Gemma**: `<start_of_turn>role\ncontent<end_of_turn>`

The chat template is a Jinja2 template that the tokenizer uses to convert structured conversation data into a single tokenized sequence with appropriate special tokens.

=== Usage ===

Use this principle when:
* Preparing conversational datasets for instruction fine-tuning
* Ensuring the tokenizer correctly formats multi-turn dialogues
* The model's default chat template doesn't match your dataset format
* You need to map EOS tokens to specific stop tokens for generation

This step comes after model and adapter setup, before training configuration.

== Theoretical Basis ==

Chat templates transform structured messages into a linear token sequence:

'''Input Format (List of Messages):'''
<syntaxhighlight lang="python">
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
]
</syntaxhighlight>

'''Output (Tokenized String via Template):'''
<syntaxhighlight lang="text">
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>
</syntaxhighlight>

'''Key Considerations:'''
1. **Stop Tokens**: The template's stop token (e.g., `<|im_end|>`) should be mapped to EOS for proper generation stopping
2. **Role Mapping**: Dataset columns ("from", "value") must map to template expectations ("role", "content")
3. **BOS Handling**: Some models require BOS token at sequence start (e.g., Gemma)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_get_chat_template]]

