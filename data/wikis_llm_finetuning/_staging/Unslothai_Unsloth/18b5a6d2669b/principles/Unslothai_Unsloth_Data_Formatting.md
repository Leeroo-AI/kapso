# Principle: Data_Formatting

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Blog|HuggingFace Chat Templates|https://huggingface.co/docs/transformers/chat_templating]]
* [[source::Blog|ChatML Specification|https://github.com/openai/openai-python/blob/main/chatml.md]]
|-
! Domains
| [[domain::NLP]], [[domain::Data_Engineering]], [[domain::Instruction_Tuning]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Technique for formatting conversational data into structured templates that instruction-tuned language models expect during training and inference.

=== Description ===

Data Formatting for LLM fine-tuning involves converting raw conversation data into a consistent template format that distinguishes between system instructions, user inputs, and model responses. This structured formatting is critical because instruction-tuned models learn to recognize these boundaries during pre-training.

Common formats include:
* **ChatML**: `<|im_start|>role\ncontent<|im_end|>`
* **Llama-3**: `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>`
* **Alpaca**: `### Instruction:\n{instruction}\n### Response:\n{response}`

The principle ensures that fine-tuning data matches the model's expected input structure, enabling effective transfer of instruction-following capabilities to new tasks.

=== Usage ===

Apply data formatting when:
* Fine-tuning on conversational/instruction datasets
* Training chat or assistant models
* Ensuring consistency between training and inference formats
* Converting between dataset formats (ShareGPT → ChatML)

Format selection guidelines:
* Match the base model's pre-training format when possible
* Use ChatML for deployment to Ollama (default format)
* Ensure EOS token placement for proper generation stopping

== Theoretical Basis ==

=== Template Structure ===

A chat template defines:
1. **Role markers**: Tokens identifying speaker (user, assistant, system)
2. **Content boundaries**: Where the actual text begins and ends
3. **Turn separators**: How to distinguish between turns
4. **Stop tokens**: Signals for generation termination

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Chat template formatting (abstract)
def format_conversation(messages, template):
    output = ""
    for msg in messages:
        role = msg["role"]  # "user", "assistant", "system"
        content = msg["content"]

        # Apply template structure
        output += template.start_token(role)
        output += content
        output += template.end_token(role)

    # Add generation prompt for inference
    if not messages[-1]["role"] == "assistant":
        output += template.start_token("assistant")

    return output
</syntaxhighlight>

=== Loss Masking ===

During training, loss is typically computed only on assistant responses:

<math>
\mathcal{L} = -\sum_{t \in \text{assistant\_tokens}} \log P(x_t | x_{<t})
</math>

User and system tokens are masked from the loss to prevent the model from learning to generate user messages.

=== Token Efficiency ===

Different templates have different token overheads:
* ChatML: ~6 tokens per turn
* Llama-3: ~8 tokens per turn
* Alpaca: ~12 tokens per turn

For long conversations, this overhead compounds and affects effective context utilization.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_get_chat_template]]
