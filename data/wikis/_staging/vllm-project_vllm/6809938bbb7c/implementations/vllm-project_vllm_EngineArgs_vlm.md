{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Vision]], [[domain::NLP]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for configuring vLLM to load and serve vision-language models (VLMs) with multimodal input support.

=== Description ===

VLM configuration in vLLM involves specifying multimodal constraints and processor settings:
- **limit_mm_per_prompt:** Maximum images/videos per input
- **mm_processor_kwargs:** Model-specific image processor configuration
- **max_model_len:** Adjusted for image token overhead

vLLM supports 60+ vision-language models including LLaVA, Qwen-VL, Pixtral, and Phi-3-Vision.

=== Usage ===

Configure VLM settings when:
- Deploying vision-language models for inference
- Limiting multimodal input size for memory management
- Customizing image preprocessing behavior
- Optimizing for specific VLM architectures

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/engine/arg_utils.py, vllm/entrypoints/llm.py
* '''Lines:''' L1-300 (arg_utils.py), L190-337 (llm.py)

=== Signature ===
<syntaxhighlight lang="python">
# VLM-specific parameters via LLM class
llm = LLM(
    model: str,
    # VLM-specific
    limit_mm_per_prompt: dict[str, int] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
    # Standard params
    trust_remote_code: bool = False,
    max_model_len: int | None = None,
    # ... other params
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || VLM model name (e.g., "llava-hf/llava-1.5-7b-hf")
|-
| limit_mm_per_prompt || dict || No || Max images/videos per prompt (e.g., {"image": 2})
|-
| mm_processor_kwargs || dict || No || Image processor config (model-specific)
|-
| trust_remote_code || bool || No || Required for some VLMs
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| LLM || LLM || VLM instance ready for multimodal inference
|}

== Usage Examples ==

=== Basic VLM Setup ===
<syntaxhighlight lang="python">
from vllm import LLM

# Load LLaVA model
llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    trust_remote_code=True,
)
</syntaxhighlight>

=== Multi-Image Configuration ===
<syntaxhighlight lang="python">
from vllm import LLM

# Allow up to 4 images per prompt
llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    limit_mm_per_prompt={"image": 4},
    trust_remote_code=True,
)
</syntaxhighlight>

=== Custom Image Processor ===
<syntaxhighlight lang="python">
from vllm import LLM

# Customize image preprocessing
llm = LLM(
    model="microsoft/Phi-3-vision-128k-instruct",
    mm_processor_kwargs={
        "max_dynamic_patch": 4,
        "dynamic_image_size": True,
    },
    trust_remote_code=True,
)
</syntaxhighlight>

=== CLI Configuration ===
<syntaxhighlight lang="bash">
vllm serve llava-hf/llava-1.5-7b-hf \
    --trust-remote-code \
    --limit-mm-per-prompt image=2 \
    --max-model-len 4096
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_VLM_Model_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
