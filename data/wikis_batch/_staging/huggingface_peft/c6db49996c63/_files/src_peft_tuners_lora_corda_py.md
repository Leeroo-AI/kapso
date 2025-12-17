# File: `src/peft/tuners/lora/corda.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 360 |
| Classes | `CordaEigens` |
| Functions | `target_modules`, `get_model_device`, `preprocess_corda`, `calib_cov_distribution`, `collect_eigens`, `collect_eigens_for_layer`, `crop_corda_eigens` |
| Imports | attr, collections, os, peft, torch, tqdm, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** CorDA (Correlation-aware low-rank Decomposition for Adaptation) initialization method

**Mechanism:** Preprocesses calibration data to compute activation covariance distributions, performs eigendecomposition of weight matrices weighted by input correlations, and initializes LoRA matrices using principal eigenvectors. The CordaEigens class stores eigenvalue information for optimal rank selection.

**Significance:** Advanced initialization technique that leverages activation statistics to better capture the most important directions in weight space. Provides faster convergence and better performance than random initialization by aligning LoRA updates with input correlation patterns observed during calibration.
