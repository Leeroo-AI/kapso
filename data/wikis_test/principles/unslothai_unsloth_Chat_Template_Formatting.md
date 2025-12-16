# Principle: Chat Template Formatting

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Hugging Face Chat Templates|https://huggingface.co/docs/transformers/chat_templating]]
* [[source::Doc|Ollama Modelfile Templates|https://github.com/ollama/ollama/blob/main/docs/modelfile.md]]
* [[source::Blog|OpenAI Chat Format|https://platform.openai.com/docs/guides/chat]]
|-
! Domains
| [[domain::NLP]], [[domain::LLMs]], [[domain::Prompt_Engineering]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Standardized formatting system that structures conversations into model-specific token sequences, enabling consistent instruction-following behavior across different inference frameworks and model families.

=== Description ===
Chat templates solve the critical problem of formatting consistency between training and inference. Each model family uses distinct special tokens and formatting conventions:

'''Major Template Families:'''
1. **Llama 3.x** - Header-based format with role markers
2. **ChatML** - XML-style tags (im_start/im_end), used by Qwen, many others
3. **Alpaca** - Instruction/Input/Response structure
4. **Mistral** - [INST] [/INST] bracketed format
5. **Gemma** - Turn-based with <start_of_turn> markers

'''Why Templates Matter:'''
- **Training-Inference Mismatch:** Wrong template degrades quality significantly
- **Multi-Turn Support:** Proper handling of conversation history
- **System Prompts:** Consistent placement of system instructions
- **Stop Sequences:** Correct termination to prevent generation loops

'''Template Components:'''
- BOS/EOS tokens: Sequence boundaries
- Role markers: Identify speaker (system/user/assistant)
- Turn delimiters: Separate conversation turns
- Stop tokens: Signal generation completion

=== Usage ===
Use proper chat template formatting when:
- Fine-tuning models for instruction following
- Deploying models with Ollama or other inference servers
- Converting between model formats (HF → GGUF)
- Ensuring consistent behavior across deployments

'''Selection Guidelines:'''
- Use the template the model was trained with
- Check tokenizer.chat_template for HuggingFace models
- Match Ollama Modelfile template to training format

== Theoretical Basis ==
'''Template Structure:'''

<syntaxhighlight lang="python">
# Generic chat template structure
def apply_chat_template(messages, template_type):
    """Format messages according to template type."""

    if template_type == "llama-3":
        return format_llama3(messages)
    elif template_type == "chatml":
        return format_chatml(messages)
    elif template_type == "alpaca":
        return format_alpaca(messages)
    # ... etc

def format_llama3(messages):
    """Llama 3.x instruction format."""
    output = "<|begin_of_text|>"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        output += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    # Add generation prompt for assistant
    output += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return output

def format_chatml(messages):
    """ChatML format (Qwen, DeepSeek, etc.)."""
    output = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        output += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    output += "<|im_start|>assistant\n"
    return output
</syntaxhighlight>

'''Ollama Template Syntax (Go Templates):'''
<syntaxhighlight lang="go">
// Llama 3.x Ollama Template
TEMPLATE """
<|begin_of_text|>
{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}
<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>
"""

// ChatML Ollama Template
TEMPLATE """
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
</syntaxhighlight>

'''Token-Level Analysis:'''
<syntaxhighlight lang="python">
# Understanding template tokens
def analyze_template_tokens(text, tokenizer):
    """Show how templates tokenize."""

    # Example Llama 3 conversation
    text = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>"""

    tokens = tokenizer.encode(text)
    # Special tokens are single IDs:
    # <|begin_of_text|> = 128000
    # <|start_header_id|> = 128006
    # <|end_header_id|> = 128007
    # <|eot_id|> = 128009

    return tokens

# Stop token configuration
STOP_TOKENS = {
    "llama-3": ["<|eot_id|>"],
    "chatml": ["<|im_end|>", "<|im_start|>"],
    "mistral": ["</s>", "[INST]"],
}
</syntaxhighlight>

'''Training Mask Generation:'''
<syntaxhighlight lang="python">
def create_response_mask(input_ids, template_type):
    """Create mask for response-only training."""

    if template_type == "llama-3":
        # Find assistant header positions
        assistant_start = encode("<|start_header_id|>assistant<|end_header_id|>")
        turn_end = encode("<|eot_id|>")
    elif template_type == "chatml":
        assistant_start = encode("<|im_start|>assistant\n")
        turn_end = encode("<|im_end|>")

    # Mask = 1 only for assistant response tokens
    mask = create_mask_between(input_ids, assistant_start, turn_end)
    return mask
</syntaxhighlight>

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_OLLAMA_TEMPLATES]]
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]

=== Tips and Tricks ===
