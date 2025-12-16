# Principle: Supervised Fine-Tuning (SFT)

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Training language models to follow instructions|https://arxiv.org/abs/2203.02155]]
* [[source::Paper|LIMA: Less Is More for Alignment|https://arxiv.org/abs/2305.11206]]
* [[source::Doc|TRL SFTTrainer Documentation|https://huggingface.co/docs/trl/sft_trainer]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::LLMs]], [[domain::Instruction_Tuning]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Training methodology that adapts pre-trained language models to follow instructions by optimizing on curated input-output pairs, enabling task-specific behavior without changing model architecture.

=== Description ===
Supervised Fine-Tuning (SFT) is the primary technique for adapting base language models into instruction-following assistants. The process involves:

1. **Dataset Curation** - Collecting high-quality instruction-response pairs
2. **Format Application** - Structuring data with chat templates (system/user/assistant)
3. **Causal Language Modeling** - Training to predict next tokens given context
4. **Response Masking** - Optionally training only on assistant responses

'''Key Concepts:'''
- **Instruction Tuning:** Training on diverse tasks to improve zero-shot generalization
- **Chat Templates:** Formatting conversation structure with special tokens
- **Sample Packing:** Combining multiple short examples into single sequences
- **Response-Only Training:** Computing loss only on assistant turns

'''SFT vs Other Methods:'''
| Method | Data Needed | Complexity | Use Case |
|--------|-------------|------------|----------|
| SFT | Demonstrations | Low | Initial instruction tuning |
| RLHF | Preferences | High | Alignment refinement |
| DPO | Preferences | Medium | Simpler preference learning |
| GRPO | Rewards | Medium | Reward-based optimization |

=== Usage ===
Use Supervised Fine-Tuning when:
- Adapting a base model to follow instructions
- Training on domain-specific tasks (code, math, legal, medical)
- You have high-quality demonstration data
- You need deterministic, reproducible training

'''Best Practices:'''
- Quality > Quantity: 1,000 high-quality examples often beats 100,000 noisy ones
- Diverse Tasks: Include variety for better generalization
- Proper Formatting: Use consistent chat templates throughout
- Response-Only Loss: Train only on assistant responses for chat models

== Theoretical Basis ==
'''Training Objective:'''

SFT optimizes the standard causal language modeling objective on formatted conversations:

<math>
\mathcal{L}_{SFT} = -\sum_{t=1}^{T} m_t \cdot \log P(x_t | x_{<t}; \theta)
</math>

Where:
- x_t is the token at position t
- m_t is the mask (1 for assistant tokens, 0 for user/system)
- θ are the model parameters (or LoRA parameters)

'''Chat Template Application:'''
<syntaxhighlight lang="python">
def format_conversation(messages, tokenizer):
    """Apply chat template to conversation."""
    # Example for Llama 3 format:
    formatted = "<|begin_of_text|>"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    return formatted

# Training mask creation
def create_training_mask(input_ids, tokenizer):
    """Create mask to train only on assistant responses."""
    mask = torch.zeros_like(input_ids)

    # Find assistant response regions
    assistant_start = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>")

    in_assistant_turn = False
    for i, token in enumerate(input_ids):
        if is_sequence_at(input_ids, i, assistant_start):
            in_assistant_turn = True
        elif is_sequence_at(input_ids, i, tokenizer.encode("<|eot_id|>")):
            in_assistant_turn = False

        if in_assistant_turn:
            mask[i] = 1

    return mask
</syntaxhighlight>

'''Sample Packing:'''
<syntaxhighlight lang="python">
def pack_sequences(examples, max_length):
    """Pack multiple short sequences into one for efficiency."""
    packed = []
    current_pack = []
    current_length = 0

    for ex in examples:
        ex_len = len(ex["input_ids"])

        if current_length + ex_len <= max_length:
            current_pack.append(ex)
            current_length += ex_len
        else:
            if current_pack:
                packed.append(concatenate(current_pack))
            current_pack = [ex]
            current_length = ex_len

    return packed

# Benefits:
# - Reduces padding waste
# - Improves GPU utilization
# - Faster training per sample
</syntaxhighlight>

'''Gradient Checkpointing:'''
Memory-efficient training by recomputing activations during backward pass:

<math>
Memory_{checkpointed} = O(\sqrt{L}) \text{ vs } O(L) \text{ standard}
</math>

Where L is the number of transformer layers.

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_UnslothTrainer]]
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]

=== Tips and Tricks ===
