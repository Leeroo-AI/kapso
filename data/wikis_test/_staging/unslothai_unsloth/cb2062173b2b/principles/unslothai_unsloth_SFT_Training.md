{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Training Language Models to Follow Instructions|https://arxiv.org/abs/2203.02155]]
* [[source::Paper|LIMA: Less Is More for Alignment|https://arxiv.org/abs/2305.11206]]
* [[source::Blog|TRL: Transformer Reinforcement Learning|https://huggingface.co/docs/trl/sft_trainer]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Fine_Tuning]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Training technique that fine-tunes language models on demonstration data by computing loss only on assistant response tokens, teaching the model to generate appropriate outputs without learning to repeat prompts.

=== Description ===
Supervised Fine-Tuning (SFT) with response-only loss masking is the standard approach for instruction-tuning language models. The key insight is that during fine-tuning, we want the model to learn how to generate responses, not how to regenerate the instructions it was given.

By masking the loss on instruction tokens (setting their labels to -100, which PyTorch's CrossEntropyLoss ignores), we:
1. **Focus learning signal**: Gradients only flow through response tokens
2. **Reduce noise**: Instruction formatting doesn't confuse the learning objective
3. **Match inference task**: Training aligns with what the model does at inference time (generating responses)
4. **Improve efficiency**: Shorter effective sequence lengths for gradient computation

The masking is applied at the data collator level, identifying response boundaries using template-specific markers (e.g., `<|start_header_id|>assistant` for Llama-3).

=== Usage ===
Use SFT with response-only loss when:
- Instruction-tuning a base model to follow commands
- Fine-tuning on conversational data where prompts are provided
- Training chatbots or assistants on demonstration data
- Any supervised fine-tuning where input/output roles are distinct

Response-only loss is particularly beneficial when:
- Prompts are long relative to responses
- Instruction formatting is complex (multi-turn, system prompts)
- You want faster convergence on response quality

== Theoretical Basis ==
Standard language model training computes cross-entropy loss over all tokens:

<math>
\mathcal{L}_{full} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})
</math>

Response-only training modifies this to only include response tokens:

<math>
\mathcal{L}_{response} = -\sum_{t \in \mathcal{R}} \log P(x_t | x_{<t})
</math>

Where <math>\mathcal{R}</math> is the set of token positions belonging to assistant responses.

'''Loss Masking Implementation:'''
<syntaxhighlight lang="python">
# Pseudo-code for response-only loss masking
def mask_instruction_tokens(input_ids, labels, instruction_end_marker):
    """
    Set labels to -100 for all tokens before assistant response.
    -100 is ignored by CrossEntropyLoss.
    """
    # Find where instruction ends and response begins
    response_start = find_marker_position(input_ids, instruction_end_marker)

    # Mask all tokens before response
    masked_labels = labels.clone()
    masked_labels[:response_start] = -100

    return masked_labels
</syntaxhighlight>

'''Data Collator Modification:'''
<syntaxhighlight lang="python">
# Pseudo-code for train_on_responses_only collator
class ResponseOnlyCollator:
    def __init__(self, tokenizer, instruction_part, response_part):
        self.instruction_part = instruction_part
        self.response_part = response_part

    def __call__(self, examples):
        batch = default_collate(examples)

        for i, (ids, labels) in enumerate(zip(batch["input_ids"], batch["labels"])):
            # Find instruction/response boundary
            boundary = find_template_boundary(
                ids, self.instruction_part, self.response_part
            )

            # Mask instruction tokens
            batch["labels"][i, :boundary] = -100

        return batch
</syntaxhighlight>

'''Gradient Flow:'''
With response-only masking, gradients only propagate through response tokens:

<syntaxhighlight lang="text">
Token sequence: [INST] What is 2+2? [/INST] The answer is 4.
Labels:         [-100] [-100] [-100] [-100] [answer] [is] [4] [.]
Gradient:       [0]    [0]    [0]    [0]    [grad]   [grad] [grad] [grad]
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_train_on_responses_only]]

=== Tips and Tricks ===
