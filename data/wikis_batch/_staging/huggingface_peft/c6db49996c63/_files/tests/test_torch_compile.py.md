# tests/test_torch_compile.py

## Understanding

### Purpose
Documents which PEFT features work with torch.compile by testing all adapter types and scenarios with both Trainer and PyTorch training loops, requiring explicit PEFT_DEBUG_WITH_TORCH_COMPILE=1 to run due to slowness.

### Mechanism
- SETTINGS defines 15+ configurations: adalora, boft, dora, ia3, ln_tuning, loha, lokr, lora, lora-target-embeddings, lora-with-modules-to-save, oft, vblora, vera, hra, bone, bone-bat, miss, miss-bat, miss-mini
- TestTorchCompileCausalLM uses fake_compile flag to disable compilation during test development (must be disabled before PR)
- test_causal_lm_training_trainer_compile tests with TrainingArguments(torch_compile=True) and Trainer, validates outputs before/after training differ, loss < max_train_loss (15.0), save/load works
- test_causal_lm_training_pytorch_compile tests with torch.compile(model) and manual training loop, same validations
- Tests BNB quantization (4bit) with torch.compile via test_causal_lm_training_lora_bnb_compile
- Tests advanced scenarios: multiple adapters, disable_adapter, merging (single/multiple adapters), merge_and_unload, mixed_batch, add_weighted_adapter

### Significance
Critical documentation of torch.compile compatibility with PEFT, ensuring users know which features work and preventing regressions, especially important as torch.compile doesn't work with all PEFT features (some tests marked xfail(strict=True)).
