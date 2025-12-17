# Principle: unslothai_unsloth_SFT_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Training Language Models to Follow Instructions|https://arxiv.org/abs/2203.02155]]
* [[source::Paper|Self-Instruct|https://arxiv.org/abs/2212.10560]]
* [[source::Doc|TRL Documentation|https://huggingface.co/docs/trl]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Supervised_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for fine-tuning language models on demonstration data where the model learns to generate desired outputs given inputs.

=== Description ===

Supervised Fine-Tuning (SFT) is the process of training a language model on input-output pairs where:

1. **Input**: A prompt, instruction, or conversation context
2. **Output**: The desired response the model should generate

The model learns to maximize the probability of the target tokens given the input context. This is the standard approach for:

- Instruction following (ChatGPT-style models)
- Task-specific fine-tuning
- Domain adaptation
- Initial training before reinforcement learning

=== Usage ===

Use SFT when:
- You have paired input-output examples
- Teaching the model new response formats
- Adapting a base model to follow instructions
- As a warmup phase before RLHF/GRPO

SFT is often the first training stage after pre-training.

== Theoretical Basis ==

=== Cross-Entropy Loss ===

SFT minimizes the cross-entropy loss between model predictions and target tokens:

<math>
\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, x)
</math>

Where:
- <math>x</math> is the input context
- <math>y_t</math> is the target token at position t
- <math>T</math> is the sequence length

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
def compute_sft_loss(model, input_ids, labels):
    # Forward pass
    logits = model(input_ids).logits

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Cross-entropy loss (ignore -100 labels)
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1)
    )
    return loss
</syntaxhighlight>

=== Response-Only Training ===

Often we only want to train on the assistant's response, not the user prompt:

<syntaxhighlight lang="python">
# Original sequence
# [User prompt tokens...] [Assistant response tokens...]

# Labels for response-only training
# [-100, -100, ..., -100] [actual_labels...]
#   ^^^^ ignored ^^^^      ^^^^ trained on ^^^^

def mask_prompt_tokens(input_ids, response_start_idx):
    labels = input_ids.clone()
    labels[:response_start_idx] = -100  # Ignore in loss
    return labels
</syntaxhighlight>

=== Training Dynamics ===

SFT training typically shows:

1. **Rapid initial loss decrease**: Model quickly learns surface patterns
2. **Slow refinement phase**: Model learns nuanced responses
3. **Potential overfitting**: Loss keeps decreasing but quality plateaus

<syntaxhighlight lang="python">
# Typical loss curve
# Step 0:    Loss ~3.0 (random initialization contribution)
# Step 100:  Loss ~1.5 (learned basic patterns)
# Step 500:  Loss ~1.0 (refined responses)
# Step 1000: Loss ~0.7 (risk of overfitting)

# Early stopping heuristic
def should_stop(current_loss, eval_loss, patience=3):
    if eval_loss has not improved for patience evaluations:
        return True
    return False
</syntaxhighlight>

=== Sample Efficiency ===

With LoRA, relatively few examples are needed:

{| class="wikitable"
|-
! Dataset Size !! Training Epochs !! Expected Result
|-
| 100-1K || 3-5 || Format learning, style adaptation
|-
| 1K-10K || 1-3 || Good instruction following
|-
| 10K-100K || 1 || Strong domain expertise
|-
| 100K+ || 0.5-1 || State-of-the-art quality
|}

=== Curriculum Learning ===

Training order can matter:

<syntaxhighlight lang="python">
# Curriculum strategy
# 1. Start with simple, clear examples
simple_examples = filter(dataset, max_length=256)

# 2. Progress to complex multi-turn
complex_examples = filter(dataset, min_turns=3)

# 3. Optionally mix in hard negative examples
curriculum_dataset = concat([
    simple_examples,  # First epoch
    full_dataset,     # Remaining epochs
])
</syntaxhighlight>

== Practical Guide ==

=== Data Quality > Quantity ===

High-quality data matters more than volume:

1. **Clean formatting**: Consistent templates, no broken conversations
2. **Accurate responses**: Factually correct, well-reasoned
3. **Diverse coverage**: Multiple task types, domains
4. **Appropriate length**: Match expected inference usage

=== Common Pitfalls ===

| Issue | Symptom | Solution |
|-------|---------|----------|
| Repetition | Model repeats phrases | Add diversity to data, lower temperature sampling |
| Short responses | Truncated outputs | Train on longer examples, check EOS handling |
| Format errors | Wrong structure | Validate template consistency |
| Overfitting | Low train loss, poor eval | Early stopping, more data, regularization |

=== Evaluation During Training ===

<syntaxhighlight lang="python">
# Monitor these metrics
metrics_to_track = {
    "train_loss": "Should decrease steadily",
    "eval_loss": "Should track train_loss",
    "learning_rate": "Follows schedule",
    "grad_norm": "Should be stable (not exploding)",
}

# Periodic manual evaluation
# Generate on held-out prompts every N steps
# Human evaluation of output quality
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_trainer_train]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
