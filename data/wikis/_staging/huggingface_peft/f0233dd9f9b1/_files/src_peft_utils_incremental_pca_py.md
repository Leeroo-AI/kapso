# File: `src/peft/utils/incremental_pca.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 338 |
| Classes | `IncrementalPCA` |
| Imports | torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** GPU-accelerated implementation of Incremental PCA for dimensionality reduction that can process data in batches, avoiding memory overflow on large datasets.

**Mechanism:** Adapted from scikit-learn's IncrementalPCA to use PyTorch tensors. partial_fit() incrementally updates mean, variance, and SVD components as new batches arrive. Supports both full SVD (torch.linalg.svd) and low-rank SVD (torch.svd_lowrank) with configurable parameters. _incremental_mean_and_var() updates statistics without storing all data. _svd_flip() ensures deterministic output.

**Significance:** Utility component for advanced PEFT methods that may require dimensionality reduction or feature analysis. Enables memory-efficient PCA on GPU for large activation matrices or weight analysis. The low-rank mode with seeding provides reproducible approximate decomposition for very large matrices.
