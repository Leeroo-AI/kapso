{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Chat Templates|https://huggingface.co/docs/transformers/chat_templating]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Input_Processing]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of structuring input data into formats that LLM inference engines can process, including text strings, token sequences, and multimodal data.

=== Description ===

Prompt Formatting is the process of converting user inputs into a format consumable by language models. This involves:

1. **Text-to-Token Conversion:** Tokenizing raw text into model vocabulary IDs
2. **Chat Template Application:** Formatting conversations with special tokens for instruction-tuned models
3. **Special Token Handling:** Adding BOS, EOS, and role-specific tokens
4. **Multimodal Integration:** Combining text with image/video placeholders
5. **Batching Preparation:** Organizing multiple prompts for efficient batch processing

Proper prompt formatting is essential for:
- Model correctness (instruction models require specific formats)
- Efficiency (pre-tokenized inputs skip tokenization overhead)
- Multimodal support (images must be properly aligned with text tokens)

=== Usage ===

Apply prompt formatting when:
- Preparing inputs for chat/instruction-tuned models
- Processing batches of diverse input types
- Integrating multimodal data (images, video)
- Optimizing throughput with pre-tokenized inputs
- Ensuring reproducibility across different tokenizer versions

== Theoretical Basis ==

'''Tokenization Process:'''

Text is converted to tokens using a vocabulary-based encoding:

<syntaxhighlight lang="python">
# Conceptual tokenization
def tokenize(text, vocabulary):
    tokens = []
    while text:
        # Find longest matching vocabulary item
        match = find_longest_prefix_match(text, vocabulary)
        tokens.append(vocabulary[match])
        text = text[len(match):]
    return tokens
</syntaxhighlight>

'''Chat Template Structure:'''

Instruction-tuned models expect specific formatting:

<syntaxhighlight lang="python">
# Llama-style chat template (conceptual)
def apply_chat_template(messages):
    formatted = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted
</syntaxhighlight>

'''Token-Text Alignment:'''

For multimodal inputs, special tokens mark image positions:

<syntaxhighlight lang="python">
# Multimodal prompt structure
prompt = {
    "prompt": "Describe this image: <image>",
    "multi_modal_data": {
        "image": pil_image_object
    }
}
# The <image> token is replaced with image embeddings internally
</syntaxhighlight>

'''Input Type Normalization:'''

vLLM normalizes different input formats to a common internal representation:

<syntaxhighlight lang="python">
# Internal normalization (conceptual)
def normalize_prompt(prompt, tokenizer):
    if isinstance(prompt, str):
        return {"prompt_token_ids": tokenizer.encode(prompt)}
    elif "prompt" in prompt:
        return {"prompt_token_ids": tokenizer.encode(prompt["prompt"]),
                "multi_modal_data": prompt.get("multi_modal_data")}
    elif "prompt_token_ids" in prompt:
        return prompt  # Already normalized
    else:
        raise ValueError("Invalid prompt format")
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_PromptType_usage]]
