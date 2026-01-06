# File: `src/peft/tuners/lora/corda.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 360 |
| Classes | `CordaEigens` |
| Functions | `target_modules`, `get_model_device`, `preprocess_corda`, `calib_cov_distribution`, `collect_eigens`, `collect_eigens_for_layer`, `crop_corda_eigens` |
| Imports | attr, collections, os, peft, torch, tqdm, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** CorDA initialization with covariance

**Mechanism:** Implements CorDA (Covariance-aware Rank-Decomposed Adaptation) initialization. preprocess_corda() computes input activation covariance matrices by running inference on calibration data using forward hooks. calib_cov_distribution() builds covariance from activation statistics. collect_eigens_for_layer() performs SVD on covariance-adjusted weight matrices (W @ Cov) with inverse covariance compensation to find optimal low-rank subspaces. Supports both IPM (Important Parameter Matching) using top singular vectors and KPM (Keep Parameter Matching) using bottom vectors. Caches results to avoid recomputation.

**Significance:** Most sophisticated LoRA initialization method that accounts for actual input distributions. By incorporating activation covariance, CorDA finds low-rank decompositions that better preserve model behavior on target data. Particularly valuable for domain adaptation where input distributions differ significantly from pretraining.
