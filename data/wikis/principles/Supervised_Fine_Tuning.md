{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|InstructGPT|https://arxiv.org/abs/2203.02155]]
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Blog|Fine-tuning LLMs|https://huggingface.co/blog/fine-tune-llama-2]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::LLMs]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Training technique that adapts pre-trained language models to follow instructions by training on input-output demonstration pairs.

=== Description ===
Supervised Fine-Tuning (SFT) is the standard method for teaching language models to follow instructions. A pre-trained base model is trained on curated datasets of (instruction, response) pairs, learning to generate helpful, task-specific outputs. SFT is typically the first step in the RLHF pipeline, creating a model that can be further aligned using preference learning (DPO, PPO, GRPO).

=== Usage ===
Use this principle when adapting a base language model to a specific task or behavior. Apply SFT when you have demonstration data showing desired inputs and outputs. Common use cases include instruction following, chat capability, domain-specific knowledge injection, and format adherence (JSON, code, etc.).

== Theoretical Basis ==
'''Objective Function:'''
Standard cross-entropy loss over target tokens:

\[
\mathcal{L}_{SFT} = -\sum_{t=1}^{T} \log P_\theta(y_t | x, y_{<t})
\]

Where:
* x = input/instruction
* y = target response
* T = response length

'''Training Data Format:'''
<syntaxhighlight lang="python">
# Typical instruction format
example = {
    "instruction": "Summarize the following article:",
    "input": "Article text here...",
    "output": "Summary of the article..."
}

# Chat format
example = {
    "messages": [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is..."}
    ]
}
</syntaxhighlight>

'''Key Considerations:'''
1. **Data Quality**: Model learns from demonstrations; garbage in = garbage out
2. **Format Consistency**: Use consistent prompt templates
3. **Loss Masking**: Only compute loss on response tokens, not input

'''Loss Masking:'''
<syntaxhighlight lang="python">
def compute_sft_loss(logits, labels, input_mask):
    """
    Only compute loss on response tokens
    """
    # Mask out input tokens (set to -100 for ignore)
    labels[input_mask] = -100
    
    loss = cross_entropy(logits, labels, ignore_index=-100)
    return loss
</syntaxhighlight>

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:TRL_SFTTrainer]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]
* [[uses_heuristic::Heuristic:Sequence_Packing_Optimization]]

