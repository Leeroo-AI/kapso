# File: `src/transformers/time_series_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 225 |
| Classes | `AffineTransformed`, `ParameterProjection`, `LambdaLayer`, `DistributionOutput`, `StudentTOutput`, `NormalOutput`, `NegativeBinomialOutput` |
| Imports | collections, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides probability distribution outputs for time series forecasting models. Defines classes for various distribution types (Student-T, Normal, NegativeBinomial) that can be used as output layers in time series models to produce probabilistic predictions.

**Mechanism:** Uses PyTorch's distribution classes (torch.distributions) with custom wrappers. `DistributionOutput` is the base class that provides `domain_map` to constrain parameters to valid ranges (e.g., scale must be positive). `ParameterProjection` maps model outputs to distribution parameters. `AffineTransformed` applies location-scale transformations to distributions. Each distribution class implements `squareplus` activation for positive constraints.

**Significance:** Enables probabilistic forecasting in time series models rather than point predictions. Particularly relevant for uncertainty quantification in time series tasks. Used by time series models like TimeSeriesTransformer, Autoformer, and Informer to output not just predictions but entire probability distributions over future values.
