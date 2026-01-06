{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Training Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Systematic configuration of gradient-based optimization algorithms and learning rate schedules for iterative model parameter updates.

=== Description ===
Optimizer and scheduler setup is a fundamental principle in neural network training that governs how model parameters are updated during the learning process. This principle addresses the critical challenge of efficiently navigating high-dimensional parameter spaces to minimize loss functions while avoiding common pitfalls like slow convergence, instability, and poor generalization.

The principle encompasses two interconnected components: the optimizer, which defines the parameter update rule based on computed gradients, and the learning rate scheduler, which dynamically adjusts the learning rate throughout training. Together, they determine the trajectory of the optimization process and significantly impact both training efficiency and final model quality.

Key aspects include: separating parameters into groups with different update strategies (e.g., applying weight decay selectively), selecting appropriate optimization algorithms (Adam, SGD, AdamW, etc.) based on task characteristics, configuring hyperparameters like learning rate and momentum, designing learning rate schedules that balance exploration and convergence (linear warmup, cosine annealing, etc.), and handling special cases like layer-wise learning rates or gradient clipping.

The principle recognizes that different model components may benefit from different optimization strategies, such as excluding bias terms and layer normalization parameters from weight decay, or using lower learning rates for pre-trained layers during fine-tuning.

=== Usage ===
Apply this principle when initializing any gradient-based training process. Setup should occur after model initialization but before the training loop begins. Create the optimizer first by organizing parameters into groups, then create the scheduler based on the planned training duration. Use it when you need to configure how gradients update model weights, when implementing custom optimization strategies, or when you need dynamic learning rate adjustment during training.

== Theoretical Basis ==

The optimizer and scheduler setup principle involves several key operations:

'''1. Parameter Grouping with Selective Weight Decay:'''
<pre>
# Separate parameters that should and shouldn't have weight decay
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    # Typically exclude bias and normalization layers from decay
    if "bias" in name or "LayerNorm" in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

param_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0}
]
</pre>

'''2. Optimizer Instantiation:'''
<pre>
# General optimizer pattern
optimizer = OptimizerClass(
    params=param_groups,
    lr=learning_rate,
    betas=(beta1, beta2),
    eps=epsilon,
    weight_decay=weight_decay  # Applied per group
)

# Common optimizers:
# AdamW: Decoupled weight decay + adaptive learning rates
# SGD: Simple but effective with momentum
# Adafactor: Memory-efficient for large models
</pre>

'''3. Learning Rate Schedule Computation:'''
<pre>
# Calculate warmup and total steps
if warmup_ratio > 0:
    warmup_steps = total_training_steps * warmup_ratio
else:
    warmup_steps = specified_warmup_steps

# Common schedule types:
# Linear: lr = lr_initial * (1 - progress)
# Cosine: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * progress))
# Constant with warmup: lr increases linearly then stays constant
</pre>

'''4. Scheduler Instantiation:'''
<pre>
scheduler = get_scheduler(
    name=scheduler_type,  # "linear", "cosine", "constant", etc.
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)

# Scheduler is called after each optimizer step:
# optimizer.step()
# scheduler.step()
</pre>

'''5. Warmup Strategy:'''
<pre>
# Linear warmup prevents instability in early training
if current_step < warmup_steps:
    lr_multiplier = current_step / warmup_steps
    current_lr = initial_lr * lr_multiplier
else:
    # Apply main schedule after warmup
    current_lr = schedule_function(current_step - warmup_steps)
</pre>

'''6. Parameter Update Formula (AdamW example):'''
<pre>
# First-order moment (momentum)
m_t = beta1 * m_(t-1) + (1 - beta1) * gradient_t

# Second-order moment (adaptive learning rate)
v_t = beta2 * v_(t-1) + (1 - beta2) * gradient_t^2

# Bias correction
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

# Weight decay (decoupled from gradient)
parameter = parameter - lr * weight_decay * parameter

# Gradient-based update
parameter = parameter - lr * m_hat / (sqrt(v_hat) + eps)
</pre>

'''7. Advanced: Layer-wise Learning Rates:'''
<pre>
# Different learning rates for different layer groups
param_groups = [
    {"params": embedding_params, "lr": lr * 0.1},      # Lower for embeddings
    {"params": encoder_params, "lr": lr},              # Standard for encoder
    {"params": classifier_params, "lr": lr * 10}       # Higher for classifier
]
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Optimizer_creation]]
