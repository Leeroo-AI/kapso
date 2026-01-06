# Time Series Utils - HuggingFace Transformers

## Metadata

| Property | Value |
|----------|-------|
| Source | `src/transformers/time_series_utils.py` |
| Repository | huggingface/transformers |
| Commit Hash | f9f6619c2cf7 |
| Domain | Machine Learning / Time Series Forecasting |
| Primary Language | Python |
| License | Apache License 2.0 |
| Last Updated | 2025-12-18 |

## Overview

The `time_series_utils.py` module provides probability distribution classes and utilities specifically designed for time series forecasting with transformer models. It implements distributional output heads that allow models to predict full probability distributions rather than point estimates, enabling uncertainty quantification in forecasts.

## Description

This module extends PyTorch's distribution framework to provide time series-specific functionality for probabilistic forecasting. It is used primarily with time series transformer models like Time Series Transformer, Autoformer, and Informer.

Key components include:

1. **AffineTransformed**: A distribution wrapper that applies location and scale transformations to base distributions, useful for normalizing data and transforming predictions back to the original scale.

2. **ParameterProjection**: A neural network module that projects model hidden states to distribution parameters through learned linear layers and domain mapping functions.

3. **DistributionOutput**: An abstract base class defining the interface for creating probability distributions from model outputs, with support for univariate and multivariate predictions.

4. **Concrete Distribution Implementations**:
   - **StudentTOutput**: Student's t-distribution for heavy-tailed data
   - **NormalOutput**: Gaussian distribution for standard forecasting
   - **NegativeBinomialOutput**: For count data and discrete predictions

These classes enable transformer models to output full probability distributions, allowing for probabilistic forecasting with confidence intervals and risk assessment.

### Key Features

- Affine transformations for distribution scaling and shifting
- Automatic parameter projection from hidden states
- Support for univariate and multivariate time series
- Multiple distribution families (Normal, Student-T, Negative Binomial)
- Domain mapping with squareplus activation for positive parameters
- Independent multivariate distributions through event dimension handling

## Usage

### Basic Usage

```python
from transformers.time_series_utils import StudentTOutput, NormalOutput
import torch

# Create a distribution output handler
distr_output = StudentTOutput(dim=1)  # Univariate prediction

# Get parameter projection layer
hidden_size = 768
param_proj = distr_output.get_parameter_projection(hidden_size)

# Project hidden states to distribution parameters
hidden_states = torch.randn(32, 48, 768)  # (batch, time, hidden)
distr_params = param_proj(hidden_states)

# Create distribution from parameters
distribution = distr_output.distribution(distr_params)

# Sample from the distribution
samples = distribution.sample((100,))  # 100 sample paths

# Compute statistics
mean_forecast = distribution.mean
variance_forecast = distribution.variance
```

### Multivariate Forecasting

```python
from transformers.time_series_utils import NormalOutput

# Multivariate distribution (e.g., 7 variables)
distr_output = NormalOutput(dim=7)

# Parameters are automatically scaled for multivariate case
param_proj = distr_output.get_parameter_projection(hidden_size=768)

# Forward pass
distr_params = param_proj(hidden_states)
distribution = distr_output.distribution(distr_params)

# Sample multivariate forecasts
samples = distribution.sample((1000,))  # Shape: (1000, batch, time, 7)

print(f"Mean forecast shape: {distribution.mean.shape}")  # (batch, time, 7)
```

### Scaling Transformations

```python
from transformers.time_series_utils import NormalOutput

distr_output = NormalOutput(dim=1)

# Get distribution parameters
distr_params = param_proj(hidden_states)

# Apply inverse transformation to get predictions in original scale
loc = batch["loc"]  # Location parameter from normalization
scale = batch["scale"]  # Scale parameter from normalization

distribution = distr_output.distribution(
    distr_params,
    loc=loc,
    scale=scale
)

# Predictions are now in original scale
forecast = distribution.mean
```

## Code Reference

### Main Classes

#### AffineTransformed

```python
class AffineTransformed(TransformedDistribution):
    """
    Distribution with affine transformation applied.

    Applies the transformation: Y = scale * X + loc

    Args:
        base_distribution: The base distribution to transform
        loc: Location parameter (shift)
        scale: Scale parameter (multiplicative)
        event_dim: Number of event dimensions
    """

    def __init__(
        self,
        base_distribution: Distribution,
        loc=None,
        scale=None,
        event_dim=0
    ):
        """Initialize with base distribution and transformation parameters."""

    @property
    def mean(self):
        """Returns the mean of the transformed distribution."""

    @property
    def variance(self):
        """Returns the variance of the transformed distribution."""

    @property
    def stddev(self):
        """Returns the standard deviation of the transformed distribution."""
```

#### ParameterProjection

```python
class ParameterProjection(nn.Module):
    """
    Projects input features to distribution parameters.

    Args:
        in_features: Input feature dimension
        args_dim: Dictionary mapping parameter names to dimensions
        domain_map: Function that maps unbounded values to valid parameter domains
    """

    def __init__(
        self,
        in_features: int,
        args_dim: dict[str, int],
        domain_map: Callable[..., tuple[torch.Tensor]],
        **kwargs
    ) -> None:
        """Initialize parameter projection layers."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Project input to distribution parameters.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Tuple of parameter tensors in valid domains
        """
```

#### DistributionOutput

```python
class DistributionOutput:
    """
    Abstract base class for distribution outputs.

    Attributes:
        distribution_class: PyTorch distribution class
        in_features: Input feature dimension
        args_dim: Dictionary of parameter dimensions
        dim: Number of output dimensions
    """

    def __init__(self, dim: int = 1) -> None:
        """Initialize with output dimension."""

    def distribution(
        self,
        distr_args,
        loc: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> Distribution:
        """
        Create distribution from parameters with optional affine transform.

        Args:
            distr_args: Tuple of distribution parameters
            loc: Optional location parameter
            scale: Optional scale parameter

        Returns:
            PyTorch Distribution object
        """

    def get_parameter_projection(self, in_features: int) -> nn.Module:
        """
        Return parameter projection layer.

        Args:
            in_features: Input feature dimension

        Returns:
            ParameterProjection module
        """

    @property
    def event_shape(self) -> tuple:
        """Shape of each individual event."""

    @property
    def event_dim(self) -> int:
        """Number of event dimensions."""

    @property
    def value_in_support(self) -> float:
        """Valid numeric value for the distribution support."""

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        """
        Maps inputs to positive orthant using square-plus operation.

        Args:
            x: Input tensor

        Returns:
            Positive tensor
        """

    def domain_map(self, *args: torch.Tensor):
        """
        Convert arguments to the right shape and domain.

        Must be implemented by subclasses.
        """
```

#### StudentTOutput

```python
class StudentTOutput(DistributionOutput):
    """
    Student's t-distribution output for heavy-tailed data.

    Parameters:
        df: Degrees of freedom (controls tail heaviness)
        loc: Location parameter (mean)
        scale: Scale parameter (spread)

    Attributes:
        args_dim: {"df": 1, "loc": 1, "scale": 1}
        distribution_class: StudentT
    """

    @classmethod
    def domain_map(cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        """
        Map unbounded parameters to valid domains.

        Args:
            df: Unbounded degrees of freedom
            loc: Unbounded location
            scale: Unbounded scale

        Returns:
            Tuple of (df, loc, scale) in valid domains
        """
```

#### NormalOutput

```python
class NormalOutput(DistributionOutput):
    """
    Normal (Gaussian) distribution output.

    Parameters:
        loc: Mean of the distribution
        scale: Standard deviation

    Attributes:
        args_dim: {"loc": 1, "scale": 1}
        distribution_class: Normal
    """

    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):
        """
        Map unbounded parameters to valid domains.

        Args:
            loc: Unbounded location (mean)
            scale: Unbounded scale (std dev)

        Returns:
            Tuple of (loc, scale) in valid domains
        """
```

#### NegativeBinomialOutput

```python
class NegativeBinomialOutput(DistributionOutput):
    """
    Negative Binomial distribution for count data.

    Parameters:
        total_count: Number of failures until experiment stops
        logits: Logits of success probability

    Attributes:
        args_dim: {"total_count": 1, "logits": 1}
        distribution_class: NegativeBinomial
    """

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):
        """
        Map unbounded parameters to valid domains.

        Args:
            total_count: Unbounded total count
            logits: Unbounded logits

        Returns:
            Tuple of (total_count, logits) in valid domains
        """

    def distribution(
        self,
        distr_args,
        loc: torch.Tensor | None = None,
        scale: torch.Tensor | None = None
    ) -> Distribution:
        """
        Create negative binomial distribution with parameter scaling.

        Note: Scale is applied to logits since affine transform
        doesn't preserve the integer nature of the distribution.
        """
```

#### LambdaLayer

```python
class LambdaLayer(nn.Module):
    """
    Wraps a function as a PyTorch module.

    Args:
        function: The function to wrap
    """

    def __init__(self, function):
        """Initialize with function."""

    def forward(self, x, *args):
        """Apply the wrapped function."""
```

### Imports

```python
from collections.abc import Callable

import torch
from torch import nn
from torch.distributions import (
    AffineTransform,
    Distribution,
    Independent,
    NegativeBinomial,
    Normal,
    StudentT,
    TransformedDistribution,
)
```

## I/O Contracts

### AffineTransformed

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| base_distribution | Distribution | Base distribution to transform | Required |
| loc | float/Tensor | Location shift | None |
| scale | float/Tensor | Scale multiplier | None |
| event_dim | int | Number of event dimensions | 0 |

#### Properties
| Property | Type | Description |
|----------|------|-------------|
| mean | Tensor | Mean of transformed distribution |
| variance | Tensor | Variance of transformed distribution |
| stddev | Tensor | Standard deviation of transformed distribution |

### ParameterProjection

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| x | torch.Tensor | Input features (..., in_features) | Required |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| params | tuple[Tensor] | Distribution parameters in valid domains |

### DistributionOutput.distribution

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| distr_args | tuple | Distribution parameters | Required |
| loc | Tensor | Optional location shift | None |
| scale | Tensor | Optional scale multiplier | None |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| distribution | Distribution | PyTorch distribution object |

### StudentTOutput.domain_map

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| df | Tensor | Unbounded degrees of freedom | Required |
| loc | Tensor | Unbounded location | Required |
| scale | Tensor | Unbounded scale | Required |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| df | Tensor | Degrees of freedom (>= 2) |
| loc | Tensor | Location parameter |
| scale | Tensor | Positive scale parameter |

### NormalOutput.domain_map

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| loc | Tensor | Unbounded location | Required |
| scale | Tensor | Unbounded scale | Required |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| loc | Tensor | Mean parameter |
| scale | Tensor | Positive standard deviation |

### NegativeBinomialOutput.domain_map

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| total_count | Tensor | Unbounded count | Required |
| logits | Tensor | Unbounded logits | Required |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| total_count | Tensor | Positive count parameter |
| logits | Tensor | Logits parameter |

## Usage Examples

### Example 1: Complete Time Series Forecasting Pipeline

```python
import torch
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
from transformers.time_series_utils import StudentTOutput

# Configure model with probabilistic output
config = TimeSeriesTransformerConfig(
    prediction_length=24,
    context_length=48,
    distribution_output="student_t",
    num_parallel_samples=100
)

# Create distributional output head
distr_output = StudentTOutput(dim=1)

class ForecastingModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TimeSeriesTransformerModel(config)
        self.distr_output = StudentTOutput(dim=1)
        self.param_proj = self.distr_output.get_parameter_projection(
            config.d_model
        )

    def forward(self, past_values, past_time_features, future_time_features):
        # Encode past values
        outputs = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features
        )

        # Project to distribution parameters
        params = self.param_proj(outputs.last_hidden_state)

        # Create distribution
        distr = self.distr_output.distribution(params)

        return distr

# Use the model
model = ForecastingModel(config)
model.eval()

with torch.no_grad():
    distribution = model(
        past_values=past_values,
        past_time_features=past_time_features,
        future_time_features=future_time_features
    )

    # Get point forecast
    forecast = distribution.mean

    # Get uncertainty estimates
    forecast_std = distribution.stddev

    # Generate prediction samples
    samples = distribution.sample((100,))

    # Compute quantiles for confidence intervals
    quantiles = torch.quantile(samples, torch.tensor([0.1, 0.5, 0.9]), dim=0)
    lower_bound = quantiles[0]
    median = quantiles[1]
    upper_bound = quantiles[2]
```

### Example 2: Normalized Data Handling

```python
from transformers.time_series_utils import NormalOutput
import torch

class NormalizedForecaster(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TimeSeriesEncoder(config)
        self.distr_output = NormalOutput(dim=1)
        self.param_proj = self.distr_output.get_parameter_projection(config.d_model)

    def forward(self, past_values, loc, scale):
        # Normalize input data
        normalized_values = (past_values - loc) / scale

        # Encode and project
        hidden = self.encoder(normalized_values)
        params = self.param_proj(hidden)

        # Create distribution in normalized space
        # Then transform back to original scale
        distribution = self.distr_output.distribution(
            params,
            loc=loc,
            scale=scale
        )

        return distribution

# Usage with data normalization
forecaster = NormalizedForecaster(config)

# Compute normalization statistics from training data
train_mean = train_data.mean(dim=1, keepdim=True)
train_std = train_data.std(dim=1, keepdim=True)

# Forecast
distribution = forecaster(
    past_values=test_data,
    loc=train_mean,
    scale=train_std
)

# Predictions are automatically in original scale
forecast = distribution.mean
```

### Example 3: Multivariate Forecasting

```python
from transformers.time_series_utils import NormalOutput

class MultivariateForecast(torch.nn.Module):
    def __init__(self, config, num_variables=7):
        super().__init__()
        self.encoder = TimeSeriesEncoder(config)

        # Multivariate output
        self.distr_output = NormalOutput(dim=num_variables)
        self.param_proj = self.distr_output.get_parameter_projection(config.d_model)

    def forward(self, past_values):
        # past_values shape: (batch, time, num_variables)
        hidden = self.encoder(past_values)

        # Project to multivariate distribution parameters
        params = self.param_proj(hidden)

        # Create independent multivariate distribution
        distribution = self.distr_output.distribution(params)

        return distribution

# Usage
model = MultivariateForecast(config, num_variables=7)
past_values = torch.randn(32, 48, 7)  # 32 series, 48 timesteps, 7 variables

distribution = model(past_values)

# Forecast all variables
mean_forecast = distribution.mean  # Shape: (32, prediction_length, 7)

# Sample joint forecasts
samples = distribution.sample((1000,))  # Shape: (1000, 32, prediction_length, 7)

# Compute covariance (per timestamp)
samples_reshaped = samples.transpose(0, 2)  # (32, prediction_length, 1000, 7)
for t in range(prediction_length):
    cov_t = torch.cov(samples_reshaped[:, t, :, :].squeeze())
    print(f"Covariance at t={t}:\n{cov_t}")
```

### Example 4: Count Data Forecasting

```python
from transformers.time_series_utils import NegativeBinomialOutput

class CountForecaster(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TimeSeriesEncoder(config)

        # Use Negative Binomial for count data
        self.distr_output = NegativeBinomialOutput(dim=1)
        self.param_proj = self.distr_output.get_parameter_projection(config.d_model)

    def forward(self, past_counts):
        hidden = self.encoder(past_counts.float())
        params = self.param_proj(hidden)

        # Create count distribution
        distribution = self.distr_output.distribution(params)

        return distribution

# Usage for forecasting discrete counts (e.g., sales, visits)
model = CountForecaster(config)
past_sales = torch.tensor([[10, 15, 12, 20, 18, 25]])  # Historical sales counts

distribution = model(past_sales)

# Generate count samples
samples = distribution.sample((1000,))
mean_sales = samples.float().mean(dim=0)  # Expected sales
print(f"Expected future sales: {mean_sales}")

# Probability of specific outcomes
prob_zero = torch.exp(distribution.log_prob(torch.zeros_like(mean_sales)))
print(f"Probability of zero sales: {prob_zero}")
```

### Example 5: Custom Distribution Output

```python
from transformers.time_series_utils import DistributionOutput
from torch.distributions import Laplace

class LaplaceOutput(DistributionOutput):
    """Custom Laplace distribution for robust forecasting."""

    args_dim: dict[str, int] = {"loc": 1, "scale": 1}
    distribution_class = Laplace

    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):
        """Map to valid parameter domains."""
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        return loc.squeeze(-1), scale.squeeze(-1)

# Usage
distr_output = LaplaceOutput(dim=1)
param_proj = distr_output.get_parameter_projection(hidden_size=768)

# In your model
hidden_states = encoder(input_data)
params = param_proj(hidden_states)
distribution = distr_output.distribution(params)

# Laplace distribution is more robust to outliers than Normal
forecast = distribution.mean
```

### Example 6: Ensemble Forecasting

```python
from transformers.time_series_utils import StudentTOutput, NormalOutput

class EnsembleForecast(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TimeSeriesEncoder(config)

        # Multiple distribution heads
        self.student_output = StudentTOutput(dim=1)
        self.normal_output = NormalOutput(dim=1)

        self.student_proj = self.student_output.get_parameter_projection(config.d_model)
        self.normal_proj = self.normal_output.get_parameter_projection(config.d_model)

        # Mixture weight
        self.weight_proj = torch.nn.Linear(config.d_model, 1)

    def forward(self, past_values):
        hidden = self.encoder(past_values)

        # Get both distributions
        student_params = self.student_proj(hidden)
        normal_params = self.normal_proj(hidden)

        student_dist = self.student_output.distribution(student_params)
        normal_dist = self.normal_output.distribution(normal_params)

        # Mixture weight
        weight = torch.sigmoid(self.weight_proj(hidden))

        # Sample from mixture
        samples_student = student_dist.sample((500,))
        samples_normal = normal_dist.sample((500,))

        # Combine samples based on weight
        mask = torch.rand_like(samples_student) < weight
        mixed_samples = torch.where(mask, samples_student, samples_normal)

        return mixed_samples

# Usage
model = EnsembleForecast(config)
samples = model(past_values)
forecast = samples.mean(dim=0)
uncertainty = samples.std(dim=0)
```

## Related Pages

- (To be populated)
