{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Chat]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Structured formatting systems that convert conversational messages into model-specific input sequences.

=== Description ===

Chat templates are Jinja2-based formatting systems that transform structured conversations (with roles like user, assistant, system) into the exact text format expected by instruction-tuned and chat-optimized language models. Each model has a specific template that defines how to arrange messages, insert special tokens, add role indicators, and structure multi-turn conversations. Templates handle diverse features including system prompts, tool/function calling, retrieval-augmented generation contexts, and generation prompt insertion. They ensure conversations are formatted consistently with how models were trained.

=== Usage ===

Use chat templates when working with instruction-following or conversational models to ensure proper message formatting. Essential for multi-turn conversations, chat applications, instruction-tuning, and any scenario where conversation structure matters. Enables proper model behavior by matching training-time formatting.

== Theoretical Basis ==

=== Core Concepts ===

'''Message Structure:'''
* '''role''': Message sender (user, assistant, system, tool)
* '''content''': Message text
* '''name''': Optional identifier for multi-agent scenarios
* '''tool_calls''': Function calling metadata

'''Template Components:'''
* '''Role Indicators''': Tokens or text marking speaker (e.g., "User:", "<|im_start|>user")
* '''Message Separators''': Delimiters between turns (e.g., newlines, special tokens)
* '''System Prompt''': Optional instruction defining assistant behavior
* '''Generation Prompt''': Starter text for assistant response
* '''Special Tokens''': Model-specific control tokens (BOS, EOS, etc.)

'''Template Types:'''
* '''Basic Chat''': Simple role + content formatting
* '''Instruction''': Task description + input/output structure
* '''Tool Use''': Function definitions and call/response formatting
* '''RAG''': Document context integration with queries

=== Algorithm ===

<syntaxhighlight lang="text">
function APPLY_CHAT_TEMPLATE(conversation, template, options):
    // conversation: [{"role": "user", "content": "..."}, ...]
    // template: Jinja2 template string

    // Step 1: Prepare template context
    context = {
        "messages": conversation,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "add_generation_prompt": options.add_generation_prompt,
        "tools": options.tools,
        "documents": options.documents
    }

    // Step 2: Render template
    formatted_text = jinja2.render(template, context)

    // Step 3: Handle generation prompt
    if options.add_generation_prompt:
        formatted_text += get_assistant_prompt_starter()

    // Step 4: Handle continuation
    if options.continue_final_message:
        // Remove EOS from last message
        formatted_text = formatted_text.rstrip(tokenizer.eos_token)

    // Step 5: Tokenize if requested
    if options.tokenize:
        token_ids = tokenizer(
            formatted_text,
            add_special_tokens=False,  // Template handles specials
            **options.tokenizer_kwargs
        )
        return token_ids
    else:
        return formatted_text

function GET_CHAT_TEMPLATE(tokenizer, template_name, tools):
    // Handle multiple templates
    if tokenizer.chat_template is dict:
        if template_name:
            return tokenizer.chat_template[template_name]
        else if tools and "tool_use" in tokenizer.chat_template:
            return tokenizer.chat_template["tool_use"]
        else if "default" in tokenizer.chat_template:
            return tokenizer.chat_template["default"]
        else:
            raise Error("No default template")
    else:
        return tokenizer.chat_template

function ENCODE_MESSAGE_INCREMENTAL(message, history, template):
    // For streaming/incremental encoding
    full_conversation = history + [message]

    // Encode full conversation
    full_tokens = apply_chat_template(full_conversation, template)

    // Encode history only
    history_tokens = apply_chat_template(history, template)

    // Return only new tokens (delta)
    return full_tokens[len(history_tokens):]
</syntaxhighlight>

=== Template Examples ===

'''Llama 3 Template:'''
<syntaxhighlight lang="jinja">
{{bos_token}}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{- '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {%- elif message['role'] == 'user' %}
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
</syntaxhighlight>

'''ChatML Template:'''
<syntaxhighlight lang="jinja">
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
</syntaxhighlight>

'''Mistral Instruct Template:'''
<syntaxhighlight lang="jinja">
{{bos_token}}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '[INST] ' + message['content'] + ' [/INST]' }}
    {%- elif message['role'] == 'assistant' %}
        {{- message['content'] + eos_token }}
    {%- endif %}
{%- endfor %}
</syntaxhighlight>

=== Mathematical Properties ===

'''Bijectivity (Ideal):'''

Template should define bijection <math>T: C \rightarrow S</math> where:
* <math>C</math> = Structured conversation (messages with roles)
* <math>S</math> = Formatted string

In practice, formatting is surjective but not injective (multiple conversations can produce same string).

'''Token Position Mapping:'''

For token at position <math>i</math> in formatted sequence, define:
* <math>role(i)</math> = Role of message containing token <math>i</math>
* <math>msg(i)</math> = Message index containing token <math>i</math>
* <math>special(i)</math> = Boolean, whether token is template special token

=== Key Properties ===

* '''Consistency''': Same conversation structure produces same formatted output
* '''Composability''': Can append new messages and re-format incrementally
* '''Reversibility''': Some models support parsing responses back to structured format
* '''Extensibility''': Templates support custom roles and metadata
* '''Model-Specificity''': Each model family has unique template conventions

=== Design Patterns ===

'''System Message Handling:'''
* '''Prepend''': System message at start (Llama, ChatML)
* '''Inline''': System as first user message (some models)
* '''Ignore''': Drop system messages (base models)

'''Multi-Turn Structure:'''
* '''Alternating''': Strict user-assistant alternation enforced
* '''Flexible''': Multiple consecutive messages from same role allowed
* '''Grouped''': Adjacent same-role messages merged

'''Tool/Function Calling:'''
* '''In-Content''': Tool calls embedded in assistant content
* '''Structured''': Separate tool_calls field with JSON schema
* '''Tag-Based''': XML/special tokens for function calls

'''Context Integration (RAG):'''
* '''Prefix''': Documents before user query
* '''Inline''': Documents in system message
* '''Interleaved''': Documents between user messages

=== Design Considerations ===

* '''Backward Compatibility''': Template changes can break existing conversations
* '''Token Efficiency''': Template overhead consumes context window
* '''Parse Complexity''': Complex templates harder to reverse-engineer
* '''Standardization''': Lack of standard across models complicates multi-model support
* '''Special Token Conflict''': Template tokens must not appear in user content

=== Common Issues ===

* '''Missing Templates''': Base models lack chat templates (need manual formatting)
* '''Template Mismatch''': Wrong template produces poor model behavior
* '''Generation Prompt Timing''': Adding at wrong time causes issues
* '''Role Validation''': Unknown roles cause template rendering failures
* '''Token Budget''': Template overhead reduces available context length

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_apply_chat_template]]
