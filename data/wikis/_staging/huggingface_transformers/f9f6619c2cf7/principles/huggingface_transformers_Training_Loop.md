{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Training Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Iterative process of forward propagation, loss calculation, backpropagation, and parameter updates over multiple passes through training data.

=== Description ===
The training loop is the core principle of supervised machine learning that defines the iterative optimization process for improving model parameters. This principle orchestrates the complete cycle of presenting training examples to the model, computing prediction errors, propagating gradients backward through the network, and updating parameters to reduce future errors.

The training loop encompasses the entire temporal structure of the learning process: organizing data into epochs (complete passes through the dataset), batching data for efficient computation, executing forward and backward passes, applying gradient updates through the optimizer, evaluating performance at intervals, saving checkpoints, and managing early stopping conditions. It represents the fundamental algorithm by which neural networks learn from data.

Key aspects include: epoch and step management (tracking progress through training), batch iteration (loading and processing data mini-batches), forward pass execution (computing model predictions and loss), backward pass execution (computing gradients via automatic differentiation), gradient accumulation (combining gradients across multiple batches before updating), optimizer stepping (applying computed gradients to parameters), learning rate scheduling (adjusting learning rate during training), periodic evaluation (assessing model on validation data), checkpoint saving (preserving model state), and logging (tracking metrics and progress).

The principle must handle various execution modes including distributed training across multiple devices, mixed precision training for memory efficiency, gradient checkpointing for memory-compute tradeoffs, and resumption from interrupted training sessions.

=== Usage ===
Apply this principle when executing the actual training process after all components (model, optimizer, scheduler, data) have been initialized. The training loop is invoked once per training session and runs until completion or interruption. Use it when you need to train a model from scratch, continue training from a checkpoint, or fine-tune a pre-trained model. The loop should integrate evaluation, checkpointing, and early stopping logic based on your training strategy.

== Theoretical Basis ==

The training loop principle follows this algorithmic structure:

'''1. Epoch-Level Loop:'''
<pre>
for epoch in range(num_train_epochs):
    model.train()  # Set to training mode

    for batch in training_dataloader:
        # Process batch (see below)

    if eval_strategy == "epoch":
        evaluate_model()

    if save_strategy == "epoch":
        save_checkpoint()
</pre>

'''2. Batch-Level Processing:'''
<pre>
accumulated_loss = 0
optimizer.zero_grad()  # Reset gradients

for step, batch in enumerate(dataloader):
    # Forward pass
    inputs = prepare_inputs(batch)
    outputs = model(**inputs)
    loss = outputs.loss

    # Scale loss for gradient accumulation
    loss = loss / gradient_accumulation_steps

    # Backward pass
    loss.backward()  # Compute gradients

    accumulated_loss += loss.item()

    # Update parameters after accumulating gradients
    if (step + 1) % gradient_accumulation_steps == 0:
        # Gradient clipping (optional)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()  # Update learning rate
        optimizer.zero_grad()  # Reset for next accumulation

        global_step += 1
</pre>

'''3. Mixed Precision Training:'''
<pre>
# Using automatic mixed precision for memory efficiency
scaler = GradScaler()

for batch in dataloader:
    with autocast():  # Forward pass in fp16/bf16
        outputs = model(**inputs)
        loss = outputs.loss / gradient_accumulation_steps

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()

    if should_update:
        scaler.unscale_(optimizer)  # Unscale for clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
</pre>

'''4. Periodic Evaluation:'''
<pre>
if global_step % eval_steps == 0:
    model.eval()  # Set to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        eval_metrics = evaluate(eval_dataloader)

    model.train()  # Return to training mode

    # Update best model tracking
    if eval_metrics[metric_for_best] > best_metric:
        best_metric = eval_metrics[metric_for_best]
        save_best_checkpoint()
</pre>

'''5. Checkpoint Saving:'''
<pre>
if global_step % save_steps == 0:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "rng_state": get_rng_state(),
    }
    save_checkpoint(checkpoint, output_dir)
</pre>

'''6. Resumption from Checkpoint:'''
<pre>
if resume_from_checkpoint:
    checkpoint = load_checkpoint(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    start_step = checkpoint["global_step"]
    set_rng_state(checkpoint["rng_state"])
</pre>

'''7. Progress Tracking:'''
<pre>
training_state = {
    "epoch": current_epoch,
    "global_step": global_step,
    "training_loss": accumulated_loss,
    "learning_rate": get_last_lr(),
    "samples_processed": global_step * batch_size,
}

log_metrics(training_state)
</pre>

'''8. Early Stopping:'''
<pre>
if eval_metric has_not_improved for patience_epochs:
    print("Early stopping triggered")
    if load_best_model_at_end:
        load_checkpoint(best_checkpoint_path)
    break
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Training_execution]]
