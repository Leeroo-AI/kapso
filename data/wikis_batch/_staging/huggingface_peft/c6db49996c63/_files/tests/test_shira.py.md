# tests/test_shira.py

## Understanding

### Purpose
Tests SHiRA (Sparse High-Rank Adaptation) adapter which uses sparse masks to select specific weight indices for adaptation, supporting both random and custom mask functions with reproducible seeding.

### Mechanism
- TestShira class tests MLP model with SHiRA adapters targeting lin1, lin2, lin3 layers
- Validates shira_weight and shira_indices shapes match expected r*(m+n) parameters for linear layers
- Tests save/load with multiple adapters having different ranks (r=2 vs r=3) and random_seeds (56, 67)
- Verifies custom mask functions work via custom_random_mask_function_with_custom_kwargs that uses custom_arg as seed
- Tests default random mask with fixed random_seed ensures reproducible mask generation across save/load cycles
- Validates dtype support (float32, float16, bfloat16)

### Significance
SHiRA enables parameter-efficient fine-tuning by adapting only sparse subsets of weights rather than dense matrices, providing flexibility through custom mask functions while maintaining reproducibility through seeded random masks, critical for research requiring exact mask replication.
