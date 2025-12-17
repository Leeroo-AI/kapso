# Chat Formatting

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI Chat API|https://platform.openai.com/docs/api-reference/chat]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Serving]], [[domain::Input_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Principle for structuring chat messages in the OpenAI format with roles, content, and conversation history for multi-turn interactions.

=== Description ===

Chat Formatting defines how conversation messages are structured for the chat completions API. The format follows OpenAI's specification with:

* **system**: Initial instructions/persona for the model
* **user**: Human messages in the conversation
* **assistant**: Model responses (for context in multi-turn)
* **tool**: Tool/function call results

Proper formatting ensures the model understands conversation context and follows instructions correctly.

=== Usage ===

Format chat messages when:
* Building conversational applications
* Implementing multi-turn chat interfaces
* Providing system prompts for model behavior
* Maintaining conversation history

== Theoretical Basis ==

'''Message Structure:'''
<syntaxhighlight lang="python">
# Chat message format
message = {
    "role": "system" | "user" | "assistant" | "tool",
    "content": str | list[ContentPart],  # Text or multimodal
    "name": str,        # Optional: identifier for the speaker
    "tool_calls": [...] # Optional: for function calling
}
</syntaxhighlight>

'''Chat Template Application:'''

vLLM applies the model's chat template to convert messages to the expected prompt format:
<syntaxhighlight lang="python">
# Abstract template application
def apply_chat_template(messages, tokenizer):
    # Each model has its own template format
    # Llama: <|begin_of_text|><|start_header_id|>system...
    # ChatML: <|im_start|>system\n...
    return tokenizer.apply_chat_template(messages)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_chat_message_format]]
