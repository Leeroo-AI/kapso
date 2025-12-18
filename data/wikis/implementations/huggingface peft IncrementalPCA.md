{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Dimensionality_Reduction]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

GPU-accelerated Incremental Principal Component Analysis for processing large datasets in batches, used internally by PEFT methods like CorDA for covariance matrix decomposition.

=== Description ===

IncrementalPCA implements batch-wise PCA using PyTorch with GPU acceleration. It processes data incrementally, maintaining running mean/variance statistics and updating principal components via SVD after each batch. Supports both full SVD (torch.linalg.svd) and low-rank approximation (torch.svd_lowrank) for speed. The implementation adapts scikit-learn's IncrementalPCA to PyTorch tensors with CUDA support.

=== Usage ===

Use IncrementalPCA when computing principal components on large datasets that don't fit in memory, or when GPU acceleration is needed. It's used internally by CorDA for weight initialization. The partial_fit method allows streaming data through the model.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/utils/incremental_pca.py src/peft/utils/incremental_pca.py]
* '''Lines:''' 1-339

=== Signature ===
<syntaxhighlight lang="python">
class IncrementalPCA:
    """
    GPU-accelerated Incremental Principal Components Analysis.

    Args:
        n_components: Number of components to keep
        copy: If False, overwrite input data
        batch_size: Samples per batch for fit()
        svd_driver: cuSOLVER method (gesvd, gesvdj, gesvda)
        lowrank: Use torch.svd_lowrank for speed
        lowrank_q: Approximation parameter (default: n_components * 2)
        lowrank_niter: Subspace iterations for lowrank
        lowrank_seed: Seed for reproducible lowrank results

    Attributes:
        components_: Principal axes [n_components, n_features]
        singular_values_: Singular values
        mean_: Per-feature mean
        var_: Per-feature variance
        explained_variance_: Variance explained by each component
        explained_variance_ratio_: Percentage of variance explained
        n_samples_seen_: Total samples processed
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: Optional[bool] = True,
        batch_size: Optional[int] = None,
        svd_driver: Optional[str] = None,
        lowrank: bool = False,
        lowrank_q: Optional[int] = None,
        lowrank_niter: int = 4,
        lowrank_seed: Optional[int] = None,
    ): ...

    def fit(self, X, check_input=True):
        """Fit model using batched processing."""

    def partial_fit(self, X, check_input=True):
        """Incrementally fit model with a single batch."""

    def transform(self, X) -> torch.Tensor:
        """Project data onto principal components."""

    @staticmethod
    def gen_batches(n: int, batch_size: int, min_batch_size: int = 0):
        """Generate batch slices for processing."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.utils.incremental_pca import IncrementalPCA
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| n_components || int || No || Number of principal components to keep
|-
| X || torch.Tensor || Yes || Input data [n_samples, n_features]
|-
| batch_size || int || No || Samples per batch (default: 5 * n_features)
|-
| lowrank || bool || No || Use low-rank SVD approximation
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| components_ || torch.Tensor || Principal axes [n_components, n_features]
|-
| transform() || torch.Tensor || Projected data [n_samples, n_components]
|-
| explained_variance_ratio_ || torch.Tensor || Variance explained per component
|}

== Usage Examples ==

=== Basic Incremental PCA ===
<syntaxhighlight lang="python">
from peft.utils.incremental_pca import IncrementalPCA
import torch

# Create PCA with 64 components
ipca = IncrementalPCA(n_components=64)

# Fit on large dataset in batches
data = torch.randn(10000, 4096).cuda()
ipca.fit(data, batch_size=1000)

# Transform new data
new_data = torch.randn(100, 4096).cuda()
projected = ipca.transform(new_data)
# Shape: [100, 64]
</syntaxhighlight>

=== Streaming Partial Fit ===
<syntaxhighlight lang="python">
from peft.utils.incremental_pca import IncrementalPCA
import torch

ipca = IncrementalPCA(n_components=32)

# Process data stream incrementally
for batch in data_loader:
    ipca.partial_fit(batch.cuda())

# Access learned components
print(f"Components shape: {ipca.components_.shape}")
print(f"Variance explained: {ipca.explained_variance_ratio_.sum():.2%}")
</syntaxhighlight>

=== Low-Rank SVD for Speed ===
<syntaxhighlight lang="python">
from peft.utils.incremental_pca import IncrementalPCA

# Use low-rank SVD for faster computation
ipca = IncrementalPCA(
    n_components=64,
    lowrank=True,           # Use torch.svd_lowrank
    lowrank_q=128,          # Approximation rank
    lowrank_niter=4,        # Subspace iterations
    lowrank_seed=42,        # Reproducible results
)

ipca.fit(large_data.cuda())
</syntaxhighlight>

=== Used by CorDA Initialization ===
<syntaxhighlight lang="python">
# IncrementalPCA is used internally by CorDA
# to compute covariance-weighted SVD initialization
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    init_lora_weights="corda",
    corda_config={
        "covariance_path": "./covariances/",
        # Uses IncrementalPCA for PCA decomposition
    },
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
