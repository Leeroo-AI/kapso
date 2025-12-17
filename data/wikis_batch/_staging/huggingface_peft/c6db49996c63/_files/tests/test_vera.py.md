# tests/test_vera.py

## Understanding

### Purpose
Tests VeRA (Vector-based Random Matrix Adaptation) adapter which uses shared frozen random projection matrices (vera_A, vera_B) across layers with trainable scaling vectors (vera_lambda_b, vera_lambda_d).

### Mechanism
- TestVera uses mlp_same_prng fixture: two adapters with same projection_prng_key=0 share vera_A/vera_B via data_ptr() checks
- test_multiple_adapters_same_prng_weights validates mlp_same_prng.base_model.model.lin1.vera_A["default"] is mlp_same_prng.base_model.model.lin1.vera_A["other"] (shared memory)
- test_multiple_adapters_different_prng_raises validates ValueError when projection_prng_key differs: "Vera PRNG initialisation key must be the same for all adapters. Got config.projection_prng_key=123 but previous config had 0"
- test_multiple_adapters_save_load_save_projection_true validates save_projection=True saves vera_A/vera_B in adapter_model.safetensors with shape (256, 20) and (20, 256)
- test_multiple_adapters_save_load_save_projection_false validates save_projection=False does NOT save vera_A/vera_B in adapter_model.safetensors
- test_vera_A_vera_B_share_memory validates vera_A.data_ptr() == mlp_same_prng.base_model.model.lin1.vera_A["default"].data_ptr() across layers
- test_vera_lambda_dont_share_memory validates vera_lambda_b/vera_lambda_d are NOT shared across adapters or layers
- test_vera_different_shapes validates vera_A shape (rank, largest_in), vera_B shape (largest_out, rank) for layers with different shapes

### Significance
VeRA reduces parameters by sharing frozen random projection matrices across layers (only trainable scaling vectors differ per layer), with projection_prng_key ensuring reproducibility and save_projection controlling whether to save large vera_A/vera_B matrices.
