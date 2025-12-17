# File: `src/transformers/time_series_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 225 |
| Classes | `AffineTransformed`, `ParameterProjection`, `LambdaLayer`, `DistributionOutput`, `StudentTOutput`, `NormalOutput`, `NegativeBinomialOutput` |
| Imports | collections, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides probabilistic distribution output layers for time series forecasting models. Implements infrastructure for models to predict distribution parameters rather than point estimates, enabling uncertainty quantification in forecasts.

**Mechanism:** DistributionOutput is an abstract base class that defines how model outputs map to distribution parameters via ParameterProjection (projects features to parameter space) and domain_map (ensures parameters are in valid domains). Implements specific distributions: StudentTOutput (heavy-tailed, robust to outliers), NormalOutput (Gaussian), and NegativeBinomialOutput (count data). AffineTransformed applies location-scale transformations to distributions. Uses PyTorch's distribution classes internally and provides properties like mean, variance, and sampling methods. The squareplus activation (x + sqrt(x^2 + 4))/2 ensures positive parameters.

**Significance:** Enables probabilistic forecasting in transformer-based time series models like Informer, Autoformer, and TimeSeriesTransformer. Critical for applications requiring prediction intervals and uncertainty estimates (financial forecasting, demand prediction, anomaly detection). Provides the foundation for distributional loss functions that properly handle forecast uncertainty rather than just point predictions.
