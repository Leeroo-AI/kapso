# Implementation: huggingface_transformers_time_series_utils

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Time Series Models|https://huggingface.co/docs/transformers/model_doc/time_series_transformer]]
|-
! Domains
| [[domain::Time_Series]], [[domain::Probabilistic_Modeling]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Probabilistic forecasting distribution output classes for time series models, providing PyTorch distributions for Normal, Student-T, and Negative Binomial outputs.

=== Description ===

The `time_series_utils` module provides distribution output classes that map neural network outputs to probability distributions. These classes are essential for probabilistic time series forecasting models (like TimeSeriesTransformer, Informer, Autoformer) that output distribution parameters rather than point predictions. The module includes:

- `AffineTransformed`: A transformed distribution applying location-scale transformations
- `DistributionOutput`: Abstract base class for distribution outputs
- `StudentTOutput`: Student-T distribution for heavy-tailed forecasts
- `NormalOutput`: Gaussian distribution for symmetric forecasts
- `NegativeBinomialOutput`: For count data forecasting

=== Usage ===

Use this module when building or customizing time series forecasting models that require probabilistic outputs. These classes bridge the model's final hidden states to interpretable probability distributions for uncertainty quantification in forecasts.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/src/transformers/time_series_utils.py src/transformers/time_series_utils.py]
* '''Lines:''' 1-226

=== Signature ===
<syntaxhighlight lang="python">
class DistributionOutput:
    """Base class for distribution outputs."""
    distribution_class: type
    in_features: int
    args_dim: dict[str, int]

    def __init__(self, dim: int = 1) -> None:
        """Initialize with output dimension."""

    def distribution(
        self,
        distr_args,
        loc: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> Distribution:
        """Create distribution from parameters."""

    def get_parameter_projection(self, in_features: int) -> nn.Module:
        """Return projection layer for distribution parameters."""


class StudentTOutput(DistributionOutput):
    """Student-T distribution output for heavy-tailed forecasts."""
    args_dim: dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distribution_class: type = StudentT


class NormalOutput(DistributionOutput):
    """Normal distribution output for Gaussian forecasts."""
    args_dim: dict[str, int] = {"loc": 1, "scale": 1}
    distribution_class: type = Normal


class NegativeBinomialOutput(DistributionOutput):
    """Negative Binomial distribution for count data."""
    args_dim: dict[str, int] = {"total_count": 1, "logits": 1}
    distribution_class: type = NegativeBinomial
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.time_series_utils import (
    DistributionOutput,
    StudentTOutput,
    NormalOutput,
    NegativeBinomialOutput,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| in_features || int || Yes || Hidden dimension from model output
|-
| dim || int || No || Output dimension (default: 1)
|-
| distr_args || tuple[torch.Tensor] || Yes || Distribution parameters from projection
|-
| loc || torch.Tensor || No || Location parameter for affine transform
|-
| scale || torch.Tensor || No || Scale parameter for affine transform
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| distribution || torch.distributions.Distribution || Configured probability distribution
|-
| parameter_projection || nn.Module || Linear layers projecting to distribution params
|-
| mean || torch.Tensor || Expected value of distribution
|-
| variance || torch.Tensor || Variance of distribution
|}

== Usage Examples ==

=== Creating Distribution Output for Model ===
<syntaxhighlight lang="python">
import torch
from transformers.time_series_utils import NormalOutput, StudentTOutput

# Create distribution output for Gaussian predictions
normal_output = NormalOutput(dim=1)

# Get projection layer for hidden dimension 256
projection = normal_output.get_parameter_projection(in_features=256)

# Simulate model hidden states (batch=32, seq=10, hidden=256)
hidden_states = torch.randn(32, 10, 256)

# Project to distribution parameters
distr_args = projection(hidden_states)

# Create distribution
distribution = normal_output.distribution(distr_args)

# Sample predictions
samples = distribution.sample((100,))  # 100 samples
print(f"Mean forecast: {distribution.mean}")
print(f"Std forecast: {distribution.stddev}")
</syntaxhighlight>

=== Using with Time Series Transformer ===
<syntaxhighlight lang="python">
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from transformers.time_series_utils import StudentTOutput

# Configure model with Student-T distribution for robust forecasts
config = TimeSeriesTransformerConfig(
    prediction_length=24,
    context_length=48,
    distribution_output="student_t",  # Uses StudentTOutput internally
    input_size=1,
    d_model=64,
)

model = TimeSeriesTransformerForPrediction(config)

# The model will output Student-T distribution parameters
# allowing for uncertainty quantification with heavy tails
</syntaxhighlight>

== Related Pages ==
