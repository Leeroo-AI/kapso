# Principle: Supervised_Finetuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Instruction Tuning|https://arxiv.org/abs/2109.01652]]
* [[source::Paper|FLAN|https://arxiv.org/abs/2210.11416]]
* [[source::Blog|TRL Documentation|https://huggingface.co/docs/trl/sft_trainer]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Instruction_Tuning]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Training methodology where a pre-trained language model learns to follow instructions by minimizing cross-entropy loss on human-labeled input-output pairs.

=== Description ===

Supervised Fine-Tuning (SFT) adapts a pre-trained language model to follow specific instructions or complete particular tasks. The model learns from demonstrations—pairs of inputs (instructions/prompts) and desired outputs (completions/responses).

SFT differs from pre-training in several key aspects:
* **Data**: Curated instruction-response pairs vs raw text
* **Objective**: Task completion vs next-token prediction on arbitrary text
* **Scale**: Thousands to millions of examples vs trillions of tokens
* **Format**: Structured templates vs unstructured documents

In the QLoRA context, SFT updates only the low-rank adapter weights while keeping quantized base weights frozen.

=== Usage ===

Apply supervised fine-tuning when:
* Creating instruction-following assistants
* Adapting models to specific domains (medical, legal, code)
* Teaching new output formats (JSON, markdown, specific structures)
* Improving task performance with labeled examples

SFT is typically the first alignment step, followed by preference optimization (RLHF/DPO) for further refinement.

== Theoretical Basis ==

=== Cross-Entropy Loss ===

SFT minimizes the negative log-likelihood of target tokens:

<math>
\mathcal{L}_{SFT} = -\sum_{t=1}^{T} \log P_\theta(y_t | x, y_{<t})
</math>

Where:
* x: Input prompt/instruction
* y: Target response
* θ: Model parameters (LoRA adapters in QLoRA)

=== Loss Masking ===

Only response tokens contribute to the loss:

<math>
\mathcal{L} = -\sum_{t \in \text{response}} \log P_\theta(y_t | \text{prompt}, y_{<t})
</math>

Prompt tokens are attended to but not predicted, preventing the model from learning to generate instructions.

=== Sample Packing ===

For efficiency, multiple short sequences can be packed into single training examples:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Sample packing (abstract)
def pack_sequences(sequences, max_length):
    packed = []
    current = []
    current_length = 0

    for seq in sequences:
        if current_length + len(seq) <= max_length:
            current.append(seq)
            current_length += len(seq)
        else:
            packed.append(concat_with_separator(current))
            current = [seq]
            current_length = len(seq)

    return packed
</syntaxhighlight>

Benefits:
* Better GPU utilization (fewer padding tokens)
* Faster training (more examples per batch)
* Consistent sequence lengths

=== Gradient Checkpointing ===

Trade compute for memory by recomputing activations during backward pass:

<math>
\text{Memory}_{checkpoint} = O(\sqrt{L}) \text{ vs } O(L) \text{ standard}
</math>

Where L is the number of layers. Unsloth's implementation further optimizes this with selective checkpointing.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_SFTTrainer_train]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Sample_Packing_Tip]]
