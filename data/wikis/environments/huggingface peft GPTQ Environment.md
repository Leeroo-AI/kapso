# Environment: huggingface_peft_GPTQ_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Repo|AutoGPTQ|https://github.com/AutoGPTQ/AutoGPTQ]]
* [[source::Repo|GPTQModel|https://github.com/ModelCloud/GPTQModel]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Quantization]], [[domain::GPTQ]]
|-
! Last Updated
| [[last_updated::2024-12-18 00:00 GMT]]
|}

== Overview ==
PEFT environment with GPTQ quantization support via AutoGPTQ or GPTQModel backends.

=== Description ===
This environment provides support for training LoRA adapters on GPTQ-quantized models. GPTQ is a one-shot weight quantization method that achieves high compression ratios while maintaining model quality. This environment supports both the AutoGPTQ and GPTQModel backends.

=== Usage ===
Use this environment when you need to:
- Train LoRA adapters on GPTQ-quantized models
- Use pre-quantized models from HuggingFace Hub (e.g., TheBloke's GPTQ models)
- Apply QALoRA (Quantization-Aware LoRA) for GPTQ models

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux || Windows has limited support
|-
| Python || >= 3.10.0 || Required by PEFT
|-
| Hardware || NVIDIA GPU with CUDA || Required for GPTQ inference/training
|-
| CUDA || >= 11.0 || Check AutoGPTQ/GPTQModel requirements
|}

== Dependencies ==
=== System Packages ===
* CUDA toolkit
* cuDNN

=== Python Packages ===
* All packages from `huggingface_peft_Core_Environment`
* Option A: `auto_gptq` >= 0.5.0
* Option B: `gptqmodel` >= 2.0.0 AND `optimum` >= 1.24.0

== Credentials ==
* Same as `huggingface_peft_Core_Environment`

== Quick Install ==
<syntaxhighlight lang="bash">
# Option A: Install with AutoGPTQ
pip install peft auto-gptq

# Option B: Install with GPTQModel (newer)
pip install peft gptqmodel optimum>=1.24.0

# Verify installation
python -c "from peft.import_utils import is_auto_gptq_available; print(is_auto_gptq_available())"
</syntaxhighlight>

== Code Evidence ==

AutoGPTQ version check from `import_utils.py:39-49`:
<syntaxhighlight lang="python">
@lru_cache
def is_auto_gptq_available():
    if importlib.util.find_spec("auto_gptq") is not None:
        AUTOGPTQ_MINIMUM_VERSION = packaging.version.parse("0.5.0")
        version_autogptq = packaging.version.parse(importlib_metadata.version("auto_gptq"))
        if AUTOGPTQ_MINIMUM_VERSION <= version_autogptq:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-gptq. Found version {version_autogptq}, "
                f"but only versions above {AUTOGPTQ_MINIMUM_VERSION} are supported"
            )
</syntaxhighlight>

GPTQModel version check from `import_utils.py:53-76`:
<syntaxhighlight lang="python">
@lru_cache
def is_gptqmodel_available():
    if importlib.util.find_spec("gptqmodel") is not None:
        GPTQMODEL_MINIMUM_VERSION = packaging.version.parse("2.0.0")
        OPTIMUM_MINIMUM_VERSION = packaging.version.parse("1.24.0")
        version_gptqmodel = packaging.version.parse(importlib_metadata.version("gptqmodel"))
        if GPTQMODEL_MINIMUM_VERSION <= version_gptqmodel:
            if is_optimum_available():
                version_optimum = packaging.version.parse(importlib_metadata.version("optimum"))
                if OPTIMUM_MINIMUM_VERSION <= version_optimum:
                    return True
                else:
                    raise ImportError(
                        f"gptqmodel requires optimum version `{OPTIMUM_MINIMUM_VERSION}` or higher."
                    )
</syntaxhighlight>

QALoRA support from `config.py:663-673`:
<syntaxhighlight lang="python">
use_qalora: bool = field(
    default=False,
    metadata={
        "help": (
            "It is only implemented in GPTQ for now. Enable Quantization-Aware Low-Rank Adaptation (QALoRA)."
            "This technique combines quantization-aware training "
            "with LoRA to improve performance for quantized models."
        )
    },
)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: Found an incompatible version of auto-gptq` || AutoGPTQ too old || `pip install -U auto-gptq>=0.5.0`
|-
|| `ImportError: gptqmodel requires optimum version 1.24.0 or higher` || Optimum missing or too old || `pip install optimum>=1.24.0`
|-
|| `RuntimeError: ExLlama kernel not available` || ExLlama not compiled || Reinstall auto-gptq with CUDA
|}

== Compatibility Notes ==

* **AutoGPTQ vs GPTQModel**: GPTQModel is newer and may have better performance
* **ExLlama kernels**: AutoGPTQ supports ExLlama v1/v2 for faster inference
* **Mixed precision**: GPTQ models typically use fp16 for non-quantized components
* **QALoRA**: Only supported with GPTQ quantization

== Related Pages ==
* [[requires_env::Implementation:huggingface_peft_get_peft_model]]
