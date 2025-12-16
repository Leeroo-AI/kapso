{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Training Language Models to Follow Instructions|https://arxiv.org/abs/2203.02155]]
* [[source::Blog|Chat Templates in Transformers|https://huggingface.co/docs/transformers/chat_templating]]
* [[source::Paper|Self-Instruct: Aligning LMs with Self-Generated Instructions|https://arxiv.org/abs/2212.10560]]
|-
! Domains
| [[domain::NLP]], [[domain::Data_Processing]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Technique for structuring training data into conversation-style formats using chat templates, enabling models to learn appropriate instruction-following behavior through consistent prompt/response patterns.

=== Description ===
Chat template formatting is the process of converting raw instruction-response pairs into the specific token format expected by a language model. Each model family (Llama, Mistral, Qwen, etc.) uses distinct special tokens and formatting conventions that the model was pre-trained to recognize.

Proper data formatting is critical because:
1. **Template Mismatch**: Using the wrong format leads to degraded performance as the model sees unfamiliar token patterns
2. **EOS Token Handling**: Incorrect end-of-sequence tokens cause generation issues (endless or cut-off responses)
3. **Role Delineation**: Clear role markers (system, user, assistant) help the model understand conversational structure
4. **Multi-turn Conversations**: Proper formatting enables learning from dialogue context

Common template formats include:
- **Llama-3**: `<|start_header_id|>role<|end_header_id|>content<|eot_id|>`
- **ChatML**: `<|im_start|>role\ncontent<|im_end|>`
- **Alpaca**: `### Instruction:\n...\n### Response:\n`
- **Mistral**: `[INST] ... [/INST]`

The template system uses Jinja2 templates to dynamically construct formatted text from message lists.

=== Usage ===
Use chat template formatting when:
- Preparing instruction-tuning datasets for supervised fine-tuning
- Converting between dataset formats (ShareGPT, Alpaca, OpenAI)
- Ensuring tokenizer configuration matches model expectations
- Generating properly formatted inference prompts

Template selection criteria:
- Use the template matching your base model's pre-training format
- For instruction-tuned models, match their fine-tuning template
- For base models being instruction-tuned, choose based on desired deployment format

== Theoretical Basis ==
Chat templates transform structured message objects into linearized text suitable for language model training.

'''Template Application Process:'''
<syntaxhighlight lang="python">
# Pseudo-code for chat template formatting
def apply_chat_template(messages, template, add_generation_prompt=False):
    """
    messages: List of {"role": str, "content": str}
    template: Jinja2 template string
    """
    formatted = ""

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            formatted += format_system(content, template)
        elif role == "user":
            formatted += format_user(content, template)
        elif role == "assistant":
            formatted += format_assistant(content, template)

    if add_generation_prompt:
        # Add assistant header to prompt generation
        formatted += get_assistant_prefix(template)

    return formatted
</syntaxhighlight>

'''Llama-3 Template Example:'''
<syntaxhighlight lang="text">
# Input messages:
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
]

# Formatted output:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

4<|eot_id|>
</syntaxhighlight>

'''EOS Token Mapping:'''
Different templates use different end tokens:
<syntaxhighlight lang="python">
# Pseudo-code for EOS token configuration
EOS_TOKENS = {
    "llama-3": "<|eot_id|>",
    "chatml": "<|im_end|>",
    "alpaca": "</s>",
    "mistral": "</s>",
}

def configure_tokenizer(tokenizer, template_name):
    eos_token = EOS_TOKENS[template_name]
    tokenizer.eos_token = eos_token
    tokenizer.pad_token = eos_token  # Often same as EOS
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_get_chat_template]]

=== Tips and Tricks ===
