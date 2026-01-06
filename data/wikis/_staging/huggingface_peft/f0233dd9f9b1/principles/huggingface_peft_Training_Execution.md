{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Doc|Transformers Trainer|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Training]], [[domain::Optimization]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for executing the training loop to optimize adapter parameters for task-specific performance.

=== Description ===

Training Execution runs the optimization loop that updates adapter weights while keeping base model weights frozen. The training process involves:
1. Forward pass through adapted model
2. Loss computation (typically cross-entropy for language modeling)
3. Backward pass computing gradients only for adapter parameters
4. Optimizer step updating adapter weights

Key considerations include learning rate selection (typically higher than full fine-tuning), gradient accumulation for effective batch sizes, and mixed precision for efficiency.

=== Usage ===

Apply this principle using HuggingFace Trainer or custom training loops:
* **Learning rate:** Use 1e-4 to 2e-4 (higher than full fine-tuning's 1e-5)
* **Batch size:** Use gradient accumulation to achieve effective batch of 16-64
* **Precision:** Use fp16/bf16 mixed precision for memory efficiency
* **Scheduling:** Cosine or linear decay with warmup (3-10% of steps)

== Theoretical Basis ==

'''Optimization Objective:'''

The training minimizes loss over adapter parameters only:
<math>\min_{A, B} \mathcal{L}(\theta_0 + BA; \mathcal{D})</math>

Where:
* <math>\theta_0</math> are frozen base model parameters
* <math>A, B</math> are trainable LoRA matrices
* <math>\mathcal{D}</math> is the training dataset

'''Gradient Flow:'''

During backpropagation:
<syntaxhighlight lang="python">
# Pseudo-code for gradient computation
def backward(loss, model):
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:  # Only adapter params
            param.grad  # Computed and stored
        else:  # Base model params
            param.grad  # None (not computed)
</syntaxhighlight>

'''Learning Rate Considerations:'''

LoRA typically uses higher learning rates because:
1. Fewer parameters to optimize → can use larger steps
2. Low-rank structure constrains the optimization space
3. Base model provides strong initialization

Empirical guidance:
* Full fine-tuning: 1e-5 to 5e-5
* LoRA fine-tuning: 1e-4 to 3e-4

'''Gradient Accumulation:'''

For memory-constrained settings:
<math>\text{Effective batch size} = \text{micro batch} \times \text{accumulation steps}</math>

<syntaxhighlight lang="python">
# Pseudo-code for gradient accumulation
for step, batch in enumerate(dataloader):
    loss = model(**batch).loss / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_Trainer_train]]

