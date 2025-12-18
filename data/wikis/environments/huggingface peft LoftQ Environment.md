# Environment: huggingface_peft_LoftQ_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|LoftQ|https://huggingface.co/papers/2310.08659]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Quantization]], [[domain::Initialization]]
|-
! Last Updated
| [[last_updated::2024-12-18 00:00 GMT]]
|}

== Overview ==
PEFT environment with scipy for LoftQ (LoRA-Fine-Tuning-Aware Quantization) initialization.

=== Description ===
This environment extends the Quantization Environment with scipy support for LoftQ initialization. LoftQ is an initialization method that quantizes backbone weights while simultaneously initializing LoRA matrices to minimize the quantization error. This results in better starting points for QLoRA training.

=== Usage ===
Use this environment when you need to:
- Initialize LoRA weights using LoftQ method (`init_lora_weights='loftq'`)
- Reduce quantization error compared to standard QLoRA initialization
- Train with `LoftQConfig` for custom quantization parameters

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, Windows || Linux recommended
|-
| Python || >= 3.10.0 || Required by PEFT
|-
| Hardware || NVIDIA GPU with CUDA || Required for quantization
|}

== Dependencies ==
=== Python Packages ===
* All packages from `huggingface_peft_Quantization_Environment`
* `scipy` (required for SVD operations in LoftQ)

== Quick Install ==
<syntaxhighlight lang="bash">
# Install PEFT with LoftQ dependencies
pip install peft bitsandbytes scipy

# Verify scipy
python -c "import scipy; print(scipy.__version__)"
</syntaxhighlight>

== Code Evidence ==

Scipy requirement check from `config.py:799-803`:
<syntaxhighlight lang="python">
# handle init_lora_weights and loftq_config
if self.init_lora_weights == "loftq":
    import importlib

    if not importlib.util.find_spec("scipy"):
        raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")
</syntaxhighlight>

Scipy import in LoftQ utils from `loftq_utils.py:69`:
<syntaxhighlight lang="python">
if not importlib.util.find_spec("scipy"):
    raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")
</syntaxhighlight>

LoftQ config validation from `config.py:804-810`:
<syntaxhighlight lang="python">
if not self.loftq_config:
    raise ValueError("`loftq_config` must be specified when `init_lora_weights` is 'loftq'.")
if not isinstance(self.loftq_config, dict):
    # convert loftq_config to dict
    self.loftq_config = vars(self.loftq_config)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: The required package 'scipy' is not installed` || scipy missing || `pip install scipy`
|-
|| `ValueError: loftq_config must be specified when init_lora_weights is 'loftq'` || Missing config || Pass `LoftQConfig()` when using LoftQ
|-
|| `ValueError: Only support 2, 4, 8 bits quantization` || Invalid bit setting || Use 2, 4, or 8 bits in LoftQConfig
|-
|| `ValueError: bitsandbytes is not available` || bitsandbytes not installed || `pip install bitsandbytes`
|}

== Compatibility Notes ==

* **Quantization bits**: LoftQ supports 2, 4, and 8-bit quantization
* **SVD operations**: Uses scipy for efficient SVD computation
* **First-time initialization**: Set `fake=True` in LoftQConfig for first run to save weights, then load with `fake=False`

== Related Pages ==
* [[requires_env::Implementation:huggingface_peft_LoraConfig_init]]
