{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|CorDA|https://arxiv.org/abs/2406.05223]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::SVD_Initialization]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Context-oriented Decomposition Adaptation that initializes LoRA weights using covariance-weighted SVD decomposition of the original weights for improved fine-tuning.

=== Description ===

CorDA (Context-oriented Decomposition Adaptation) improves LoRA initialization by considering the input covariance distribution. The preprocessing computes a covariance matrix from model activations on sample data, then performs SVD on the covariance-weighted weight matrix. This produces eigenvectors aligned with the data distribution, enabling better knowledge preservation. CorDA supports two modes: IPM (Important Principal Modes) keeps top singular vectors, while KPM (Knowledge Principal Modes) keeps bottom vectors for knowledge retention.

=== Usage ===

Use CorDA when you want better LoRA initialization that preserves important model knowledge. CorDA requires a preprocessing step where you run the model on representative data to collect covariance statistics. The resulting eigenvectors are cached and used to initialize LoRA weights. CorDA is particularly effective for domain adaptation tasks.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/corda.py src/peft/tuners/lora/corda.py]
* '''Lines:''' 1-361

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class CordaEigens:
    """Eigendecomposition results for CorDA."""
    S_WC: torch.Tensor  # Singular values
    U_WC: torch.Tensor  # Left singular vectors
    V_WC: torch.Tensor  # Right singular vectors (covariance-weighted)

def preprocess_corda(
    model: nn.Module,
    lora_config: LoraConfig,
    run_model: Optional[Callable[[], None]] = None,
    hooked_model: Optional[nn.Module] = None,
):
    """
    Build CorDA eigenvectors from model and data.

    Args:
        model: Model to preprocess
        lora_config: Config with corda_config settings
        run_model: Callback to run model on sample data
        hooked_model: Optional separate model for hooks
    """

def calib_cov_distribution(model, config, run_model, hooked_model, covariance_file):
    """Collect covariance matrices via forward hooks."""

def collect_eigens(model, config, verbose):
    """Compute SVD eigenvectors for each target layer."""

def collect_eigens_for_layer(linear, config) -> CordaEigens:
    """Compute covariance-weighted SVD for a single layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lora.corda import preprocess_corda, CordaEigens
from peft import LoraConfig, CordaConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || nn.Module || Yes || Model to preprocess for CorDA
|-
| lora_config || LoraConfig || Yes || Config with corda_config settings
|-
| run_model || Callable || Yes* || Callback to collect covariance (if not cached)
|-
| cache_file || str || No || Path to save/load CorDA eigenvectors
|-
| covariance_file || str || No || Path to save/load covariance matrices
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| module.eigens || CordaEigens || Eigendecomposition stored on each target module
|-
| U_WC || torch.Tensor || Left singular vectors [out_features, rank]
|-
| V_WC || torch.Tensor || Covariance-weighted right vectors [in_features, rank]
|}

== Usage Examples ==

=== Basic CorDA Preprocessing ===
<syntaxhighlight lang="python">
from peft import LoraConfig, get_peft_model, CordaConfig
from peft.tuners.lora.corda import preprocess_corda

# Configure CorDA
corda_config = CordaConfig(
    corda_method="ipm",         # Important Principal Modes
    cache_file="./corda_cache.pt",
    covariance_file="./corda_cov.pt",
)

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights="corda",
    corda_config=corda_config,
)

# Define data collection callback
def run_model():
    for batch in dataloader:
        with torch.no_grad():
            model(**batch)

# Preprocess model with CorDA
preprocess_corda(model, lora_config, run_model=run_model)

# Create PEFT model with CorDA-initialized weights
peft_model = get_peft_model(model, lora_config)
</syntaxhighlight>

=== CorDA with KPM Mode ===
<syntaxhighlight lang="python">
from peft import CordaConfig, LoraConfig

# Use Knowledge Principal Modes for knowledge preservation
corda_config = CordaConfig(
    corda_method="kpm",         # Keep bottom singular vectors
    cache_file="./kpm_cache.pt",
)

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    init_lora_weights="corda",
    corda_config=corda_config,
)
</syntaxhighlight>

=== Using Cached CorDA ===
<syntaxhighlight lang="python">
from peft.tuners.lora.corda import preprocess_corda

# If cache exists, preprocess_corda loads from file
# No need to run model again
corda_config = CordaConfig(
    cache_file="./existing_corda_cache.pt",
)

preprocess_corda(model, lora_config)  # Loads from cache
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
