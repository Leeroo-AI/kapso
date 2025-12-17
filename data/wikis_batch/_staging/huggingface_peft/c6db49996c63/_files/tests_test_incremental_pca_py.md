# File: `tests/test_incremental_pca.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 188 |
| Functions | `iris`, `test_incremental_pca`, `test_incremental_pca_check_projection`, `test_incremental_pca_validation`, `test_n_components_none`, `test_incremental_pca_num_features_change`, `test_incremental_pca_batch_signs`, `test_incremental_pca_batch_values`, `... +2 more` |
| Imports | datasets, peft, pytest, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for incremental PCA implementation used in PEFT utilities.

**Mechanism:** Adapted from scikit-learn's test suite, validates the IncrementalPCA class with tests covering basic functionality (fit/transform), projection accuracy, input validation (n_components constraints), batch processing consistency, partial fitting equivalence, lowrank mode, and automatic component inference. Uses the Iris dataset for realistic testing and compares results against standard PCA via SVD decomposition.

**Significance:** Validates the incremental PCA utility used internally by PEFT for dimensionality reduction and initialization strategies (e.g., LoftQ, PiSSA). Ensures batch processing produces stable, accurate results compared to standard PCA while supporting memory-efficient streaming computation for large datasets.
