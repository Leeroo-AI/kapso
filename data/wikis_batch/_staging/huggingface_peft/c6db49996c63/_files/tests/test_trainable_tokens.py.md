# tests/test_trainable_tokens.py

## Understanding

### Purpose
Tests TrainableTokens feature which allows fine-tuning specific token embeddings (e.g., [0, 1, 3]) without modifying the full embedding matrix, supporting both standalone usage and combination with LoRA.

### Mechanism
- TestTrainableTokens uses fixtures for various model types: model (Llama), model_multi_embedding, model_emb, model_embed_in, model_embed_multiple, model_weight_tied, model_weight_untied
- test_stand_alone_usage validates TrainableTokensConfig(target_modules=["embed_tokens"], token_indices=[0,1,3]) modifies only specified tokens, verify W_load[:, idcs_to_modify] != W_orig[:, idcs_to_modify] but W_load[:, idcs_to_keep] == W_orig[:, idcs_to_keep]
- test_combined_with_peft_method_usage validates LoraConfig(trainable_token_indices={"embed_tokens": [0,1,3]}) combines TrainableTokens with LoRA
- Tests basic_training verifies trainable_tokens_delta updates but trainable_tokens_original stays constant
- Tests weight tying: model_weight_tied uses patch to simulate transformers <5 (list) vs >=5 (mapping), validates emb_out(1/emb_in(x)) has embed_dim on diagonal, ensures tied weights excluded from state dict, merge_and_unload restores weight tying
- Tests multiple adapters with different/overlapping token indices, mixed_forward with adapter_names per batch item
- Tests embed_scale for Gemma3 models: orig_embedding.embed_scale.fill_(10000.0) produces max_embedding_output > 100, fill_(0) produces all zeros

### Significance
Enables efficient fine-tuning of specific tokens (e.g., new vocabulary, special tokens) without modifying full embedding matrix, critical for low-resource scenarios and models with tied input/output embeddings where weight tying must be preserved across save/load/merge operations.
