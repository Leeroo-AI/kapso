{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Training Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]], [[domain::Model_Persistence]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Systematic preservation of model state and training progress to enable recovery, deployment, and reproducibility.

=== Description ===
Checkpoint saving is a critical principle in machine learning that defines how model states, optimizer states, and training progress are persistently stored to disk. This principle addresses several essential needs: protecting against training interruptions and hardware failures, enabling training resumption from intermediate states, facilitating model deployment and inference, supporting model versioning and experimentation tracking, and enabling best model selection based on validation performance.

The principle encompasses multiple layers of state preservation: model parameters (weights and biases), optimizer state (momentum buffers, adaptive learning rate statistics), learning rate scheduler state, random number generator states for reproducibility, training metadata (epoch number, global step, best metric), configuration objects, and tokenizers/processors needed for inference.

Key aspects include: determining what components to save (full model vs. only weights, optimizer state vs. stateless), deciding when to save (at intervals, after each epoch, when metrics improve), managing storage (limiting total checkpoints, keeping only best N, implementing rotation), organizing checkpoint structure (single file vs. directory, naming conventions), handling distributed training (saving from only one process, gathering distributed states), and supporting various serialization formats (PyTorch state dicts, SafeTensors, full model serialization).

The principle recognizes that checkpoints serve multiple purposes: crash recovery (frequent, may be temporary), best model preservation (based on metrics, for deployment), and final model saving (for distribution and inference). Different purposes may require different saving strategies and retention policies.

=== Usage ===
Apply this principle throughout the training lifecycle. Save checkpoints periodically during training to protect against interruptions, save when validation metrics improve to track the best model, and save the final model when training completes. Use checkpoint saving when training takes hours or days, when you need to compare model versions, when deploying models to production, or when sharing models with others. Implement appropriate retention policies to balance storage costs with recovery needs.

== Theoretical Basis ==

The checkpoint saving principle involves several key operations:

'''1. State Collection:'''
<pre>
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epoch": current_epoch,
    "global_step": global_step,
    "best_metric": best_metric_value,
    "rng_state": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    },
    "training_args": training_args.to_dict(),
}
</pre>

'''2. Model-Only Saving (for Inference):'''
<pre>
# Save just the model for deployment
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
config.save_pretrained(output_dir)

# Saves:
# - pytorch_model.bin or model.safetensors (weights)
# - config.json (model configuration)
# - tokenizer files (vocab, special tokens, etc.)
</pre>

'''3. Full Checkpoint Saving (for Resumption):'''
<pre>
checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
os.makedirs(checkpoint_dir, exist_ok=True)

# Save model
model.save_pretrained(checkpoint_dir)

# Save optimizer and scheduler
torch.save(optimizer.state_dict(), f"{checkpoint_dir}/optimizer.pt")
torch.save(scheduler.state_dict(), f"{checkpoint_dir}/scheduler.pt")

# Save training state
torch.save({
    "epoch": epoch,
    "global_step": global_step,
    "best_metric": best_metric,
}, f"{checkpoint_dir}/trainer_state.json")

# Save RNG states
torch.save(rng_state, f"{checkpoint_dir}/rng_state.pth")
</pre>

'''4. Checkpoint Rotation (Limiting Storage):'''
<pre>
# Keep only the N most recent checkpoints
saved_checkpoints = sorted(list_checkpoints(output_dir))

if len(saved_checkpoints) > max_checkpoints:
    checkpoints_to_delete = saved_checkpoints[:-max_checkpoints]
    for checkpoint_path in checkpoints_to_delete:
        shutil.rmtree(checkpoint_path)
</pre>

'''5. Best Model Tracking:'''
<pre>
if eval_metric > best_metric:
    best_metric = eval_metric

    # Save as best checkpoint
    best_checkpoint_dir = f"{output_dir}/checkpoint-best"
    save_checkpoint(model, optimizer, scheduler, best_checkpoint_dir)

    # Or copy current checkpoint to best
    shutil.copytree(current_checkpoint, best_checkpoint_dir, dirs_exist_ok=True)
</pre>

'''6. Distributed Training Checkpoint:'''
<pre>
# Only save from main process to avoid conflicts
if is_main_process():
    save_checkpoint(checkpoint, output_dir)

# Synchronize all processes
barrier()  # Wait for save to complete

# For FSDP/DeepSpeed, need special handling
if is_fsdp_enabled:
    # Gather full state dict from sharded model
    full_state_dict = get_full_state_dict(model)
    if is_main_process():
        torch.save(full_state_dict, checkpoint_path)
</pre>

'''7. Checkpoint Loading:'''
<pre>
def load_checkpoint(checkpoint_path):
    # Load model
    model = AutoModel.from_pretrained(checkpoint_path)

    # Load optimizer
    optimizer_state = torch.load(f"{checkpoint_path}/optimizer.pt")
    optimizer.load_state_dict(optimizer_state)

    # Load scheduler
    scheduler_state = torch.load(f"{checkpoint_path}/scheduler.pt")
    scheduler.load_state_dict(scheduler_state)

    # Load training state
    training_state = json.load(open(f"{checkpoint_path}/trainer_state.json"))
    epoch = training_state["epoch"]
    global_step = training_state["global_step"]

    # Restore RNG state
    rng_state = torch.load(f"{checkpoint_path}/rng_state.pth")
    set_rng_state(rng_state)

    return model, optimizer, scheduler, epoch, global_step
</pre>

'''8. Conditional Saving Strategy:'''
<pre>
# Save based on strategy
if save_strategy == "steps" and global_step % save_steps == 0:
    save_checkpoint()
elif save_strategy == "epoch" and batch_is_last_in_epoch:
    save_checkpoint()
elif save_strategy == "best" and is_best_metric:
    save_checkpoint()
elif save_strategy == "no":
    pass  # Only save at end
</pre>

'''9. Efficient Storage Formats:'''
<pre>
# SafeTensors format (faster loading, more secure)
from safetensors.torch import save_file
save_file(model.state_dict(), "model.safetensors")

# PyTorch format (traditional)
torch.save(model.state_dict(), "pytorch_model.bin")

# Sharded checkpoints for large models
model.save_pretrained(output_dir, max_shard_size="5GB")
# Creates: pytorch_model-00001-of-00003.bin, etc.
</pre>

'''10. Metadata Preservation:'''
<pre>
# Save training configuration and metadata
metadata = {
    "training_args": training_args,
    "model_config": model.config,
    "dataset_info": dataset.info,
    "training_start_time": start_time,
    "training_end_time": end_time,
    "final_metrics": final_metrics,
    "git_commit": get_git_commit(),
}

with open(f"{output_dir}/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Model_saving]]
