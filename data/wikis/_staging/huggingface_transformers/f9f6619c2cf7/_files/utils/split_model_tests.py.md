# split_model_tests.py

**Path:** `/tmp/praxium_repo_d5p6fp4d/utils/split_model_tests.py`

## Understanding

### Purpose
Splits model test directories into chunks.

### Mechanism
This script lists all directories under `tests/models` and other top-level test directories, then splits them into N slices for parallel execution in CI workflows. It can optionally filter to a pre-computed list of subdirectories (provided via `--subdirs` as a Python literal string). The script handles both prefixed paths (`models/bert`) and non-prefixed paths (`bert`), automatically detecting which form is provided and converting appropriately. The splitting algorithm distributes directories evenly across the requested number of splits, with any remainder distributed across the first few splits.

### Significance
Enables the transformers repository to bypass GitHub Actions' 256 job matrix limit by creating multiple matrix slices, each with up to 256 jobs. This is critical for large-scale model testing where the number of model directories exceeds matrix constraints, allowing comprehensive parallel testing of all models while maintaining reasonable CI execution times.
